"""The optional bytecode cache system. This is useful if you have very
complex template situations and the compilation of all those templates
slows down your application too much.

Situations where this is useful are often forking web applications that
are initialized on the first request.
"""

import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from dataclasses import dataclass
from hashlib import sha1
from io import BytesIO
from types import CodeType

from .exceptions import TemplateBatchError

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

    class _MemcachedClient(te.Protocol):
        def get(self, key: str) -> bytes: ...

        def set(self, key: str, value: bytes, timeout: int | None = None) -> None: ...


bc_version = 5
# Magic bytes to identify Jinja bytecode cache files. Contains the
# Python major and minor version to avoid loading incompatible bytecode
# if a project upgrades its Python version.
bc_magic = (
    b"j2"
    + pickle.dumps(bc_version, 2)
    + pickle.dumps((sys.version_info[0] << 24) | sys.version_info[1], 2)
)


class Bucket:
    """Buckets are used to store the bytecode for one template.  It's created
    and initialized by the bytecode cache and passed to the loading functions.

    The buckets get an internal checksum from the cache assigned and use this
    to automatically reject outdated cache material.  Individual bytecode
    cache subclasses don't have to care about cache invalidation.
    """

    def __init__(self, environment: "Environment", key: str, checksum: str) -> None:
        self.environment = environment
        self.key = key
        self.checksum = checksum
        self.reset()

    def reset(self) -> None:
        """Resets the bucket (unloads the bytecode)."""
        self.code: CodeType | None = None

    def load_bytecode(self, f: t.BinaryIO) -> None:
        """Loads bytecode from a file or file like object."""
        # make sure the magic header is correct
        magic = f.read(len(bc_magic))
        if magic != bc_magic:
            self.reset()
            return
        # the source code of the file changed, we need to reload
        checksum = pickle.load(f)
        if self.checksum != checksum:
            self.reset()
            return
        # if marshal_load fails then we need to reload
        try:
            self.code = marshal.load(f)
        except (EOFError, ValueError, TypeError):
            self.reset()
            return

    def write_bytecode(self, f: t.IO[bytes]) -> None:
        """Dump the bytecode into the file or file like object passed."""
        if self.code is None:
            raise TypeError("can't write empty bucket")
        f.write(bc_magic)
        pickle.dump(self.checksum, f, 2)
        marshal.dump(self.code, f)

    def bytecode_from_string(self, string: bytes) -> None:
        """Load bytecode from bytes."""
        self.load_bytecode(BytesIO(string))

    def bytecode_to_string(self) -> bytes:
        """Return the bytecode as bytes."""
        out = BytesIO()
        self.write_bytecode(out)
        return out.getvalue()


def _compile_fingerprint(environment: "Environment") -> tuple[t.Any, ...]:
    """Return a hashable fingerprint of the compile-relevant configuration
    of the environment.  If any of these values change between prepare and
    commit, the compiled bytecode might be incompatible and the batch must
    be rejected.
    """
    try:
        from importlib.metadata import version

        jinja_version = version("jinja2")
    except Exception:
        jinja_version = "unknown"

    return (
        jinja_version,
        environment.is_async,
        environment.optimized,
        getattr(environment, "sandboxed", False),
        environment.finalize,
        environment.block_start_string,
        environment.block_end_string,
        environment.variable_start_string,
        environment.variable_end_string,
        environment.comment_start_string,
        environment.comment_end_string,
        environment.line_statement_prefix,
        environment.line_comment_prefix,
        environment.trim_blocks,
        environment.lstrip_blocks,
        environment.newline_sequence,
        environment.keep_trailing_newline,
        tuple(sorted(environment.extensions)),
        tuple(
            (type(ext).__module__, type(ext).__qualname__)
            for ext in environment.extensions.values()
        ),
    )


@dataclass
class _BatchItem:
    name: str
    filename: str | None
    checksum: str
    bucket: Bucket


class PreparedTemplateBatch:
    """A staging area for a batch of templates that were compiled ahead of
    time without touching the real bytecode cache.

    Create a batch with
    :meth:`BytecodeCache.prepare_template_batch` (or
    :meth:`Environment.prepare_template_batch`), then call :meth:`commit`
    once all templates are prepared::

        batch = bytecode_cache.prepare_template_batch(env, ["a.html", "b.html"])
        batch.commit()

    The prepare phase only compiles the explicitly named templates into
    memory; dependencies such as imported or included templates are not
    followed and the cache is not modified.  If any template fails to
    compile, the exception propagates and nothing is staged.

    :meth:`commit` fetches every template's source again and compares it to
    the checksum recorded at prepare time, and also compares the compile
    configuration.  If anything changed, the whole batch is rejected with
    :class:`~jinja2.TemplateBatchError`: nothing is written to the cache and
    existing cache entries stay untouched.

    Templates are only compiled, never executed, so global functions or
    render expressions in the template source are not run by prepare or
    commit.
    """

    def __init__(self, cache: "BytecodeCache", environment: "Environment") -> None:
        self.cache = cache
        self.environment = environment
        self._items: dict[str, _BatchItem] = {}
        self._fingerprint = _compile_fingerprint(environment)
        self.committed = False

    def add(self, name: str) -> None:
        """Compile the template `name` into the staging area.

        Only the named template is compiled; dependencies such as imported
        or included templates are not followed.  The template is only
        compiled, never executed, so global functions or render expressions
        in the source are not run.
        """
        if self.committed:
            raise TemplateBatchError("batch was already committed")
        if self.environment.loader is None:
            raise TypeError("no loader for this environment specified")
        source, filename, _ = self.environment.loader.get_source(self.environment, name)
        checksum = self.cache.get_source_checksum(source)
        bucket = Bucket(
            self.environment, self.cache.get_cache_key(name, filename), checksum
        )
        # Compile the source to bytecode.  This parses and generates code
        # but never executes the template, so no global functions or render
        # expressions are invoked.
        bucket.code = self.environment.compile(source, name, filename)
        self._items[name] = _BatchItem(name, filename, checksum, bucket)

    def commit(self) -> None:
        """Publish all compiled templates to the bytecode cache.

        Before publishing, every template's source is fetched again and
        compared to the checksum recorded at prepare time, and the compile
        configuration is compared as well.  If anything changed, the whole
        batch is rejected: nothing is written to the cache and existing
        cache entries stay untouched.
        """
        if self.committed:
            raise TemplateBatchError("batch was already committed")
        if not self._items:
            raise TemplateBatchError("cannot commit an empty template batch")
        if self.environment.loader is None:
            raise TypeError("no loader for this environment specified")
        if _compile_fingerprint(self.environment) != self._fingerprint:
            raise TemplateBatchError(
                "compile configuration changed after prepare; the batch was"
                " rejected, nothing was written to the cache"
            )
        for item in self._items.values():
            source, filename, _ = self.environment.loader.get_source(
                self.environment, item.name
            )
            if filename != item.filename or (
                self.cache.get_source_checksum(source) != item.checksum
            ):
                raise TemplateBatchError(
                    f"source of template {item.name!r} changed after prepare;"
                    " the batch was rejected, nothing was written to the cache"
                )
        for item in self._items.values():
            self.cache.set_bucket(item.bucket)
        self.committed = True


class BytecodeCache:
    """To implement your own bytecode cache you have to subclass this class
    and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
    these methods are passed a :class:`~jinja2.bccache.Bucket`.

    A very basic bytecode cache that saves the bytecode on the file system::

        from os import path

        class MyCache(BytecodeCache):

            def __init__(self, directory):
                self.directory = directory

            def load_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                if path.exists(filename):
                    with open(filename, 'rb') as f:
                        bucket.load_bytecode(f)

            def dump_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                with open(filename, 'wb') as f:
                    bucket.write_bytecode(f)

    A more advanced version of a filesystem based bytecode cache is part of
    Jinja.
    """

    def load_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to load bytecode into a
        bucket.  If they are not able to find code in the cache for the
        bucket, it must not do anything.
        """
        raise NotImplementedError()

    def dump_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to write the bytecode
        from a bucket back to the cache.  If it unable to do so it must not
        fail silently but raise an exception.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """Clears the cache.  This method is not used by Jinja but should be
        implemented to allow applications to clear the bytecode cache used
        by a particular environment.
        """

    def get_cache_key(self, name: str, filename: str | None = None) -> str:
        """Returns the unique hash key for this template name."""
        hash = sha1(name.encode("utf-8"))

        if filename is not None:
            hash.update(f"|{filename}".encode())

        return hash.hexdigest()

    def get_source_checksum(self, source: str) -> str:
        """Returns a checksum for the source."""
        return sha1(source.encode("utf-8")).hexdigest()

    def get_bucket(
        self,
        environment: "Environment",
        name: str,
        filename: str | None,
        source: str,
    ) -> Bucket:
        """Return a cache bucket for the given template.  All arguments are
        mandatory but filename may be `None`.
        """
        key = self.get_cache_key(name, filename)
        checksum = self.get_source_checksum(source)
        bucket = Bucket(environment, key, checksum)
        self.load_bytecode(bucket)
        return bucket

    def set_bucket(self, bucket: Bucket) -> None:
        """Put the bucket into the cache."""
        self.dump_bytecode(bucket)

    def prepare_template_batch(
        self,
        environment: "Environment",
        names: t.Iterable[str],
    ) -> PreparedTemplateBatch:
        """Compile a batch of templates into a staging area without writing
        to the cache.

        `names` is an iterable of explicit template names.  Only the named
        templates are compiled; dependencies such as imported or included
        templates are not followed.  If any template fails to compile, the
        exception propagates and nothing is staged.

        Call :meth:`~PreparedTemplateBatch.commit` on the returned batch to
        publish the compiled templates to the cache.  If a template's source
        or the compile configuration changed in the meantime, the commit is
        rejected with :class:`~jinja2.TemplateBatchError` and the cache is
        left untouched.

        .. versionadded:: 3.2
        """
        if isinstance(names, str):
            raise TypeError(
                "names must be an iterable of template names, not a single string"
            )
        batch = PreparedTemplateBatch(self, environment)
        for name in names:
            batch.add(name)
        return batch


class FileSystemBytecodeCache(BytecodeCache):
    """A bytecode cache that stores bytecode on the filesystem.  It accepts
    two arguments: The directory where the cache items are stored and a
    pattern string that is used to build the filename.

    If no directory is specified a default cache directory is selected.  On
    Windows the user's temp directory is used, on UNIX systems a directory
    is created for the user in the system temp directory.

    The pattern can be used to have multiple separate caches operate on the
    same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
    is replaced with the cache key.

    >>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

    This bytecode cache supports clearing of the cache using the clear method.
    """

    def __init__(
        self, directory: str | None = None, pattern: str = "__jinja2_%s.cache"
    ) -> None:
        if directory is None:
            directory = self._get_default_cache_dir()
        self.directory = directory
        self.pattern = pattern

    def _get_default_cache_dir(self) -> str:
        def _unsafe_dir() -> "te.NoReturn":
            raise RuntimeError(
                "Cannot determine safe temp directory.  You "
                "need to explicitly provide one."
            )

        tmpdir = tempfile.gettempdir()

        # On windows the temporary directory is used specific unless
        # explicitly forced otherwise.  We can just use that.
        if os.name == "nt":
            return tmpdir
        if not hasattr(os, "getuid"):
            _unsafe_dir()

        dirname = f"_jinja2-cache-{os.getuid()}"
        actual_dir = os.path.join(tmpdir, dirname)

        try:
            os.mkdir(actual_dir, stat.S_IRWXU)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.chmod(actual_dir, stat.S_IRWXU)
            actual_dir_stat = os.lstat(actual_dir)
            if (
                actual_dir_stat.st_uid != os.getuid()
                or not stat.S_ISDIR(actual_dir_stat.st_mode)
                or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
            ):
                _unsafe_dir()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        actual_dir_stat = os.lstat(actual_dir)
        if (
            actual_dir_stat.st_uid != os.getuid()
            or not stat.S_ISDIR(actual_dir_stat.st_mode)
            or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
        ):
            _unsafe_dir()

        return actual_dir

    def _get_cache_filename(self, bucket: Bucket) -> str:
        return os.path.join(self.directory, self.pattern % (bucket.key,))

    def load_bytecode(self, bucket: Bucket) -> None:
        filename = self._get_cache_filename(bucket)

        # Don't test for existence before opening the file, since the
        # file could disappear after the test before the open.
        try:
            f = open(filename, "rb")
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            # PermissionError can occur on Windows when an operation is
            # in progress, such as calling clear().
            return

        with f:
            bucket.load_bytecode(f)

    def dump_bytecode(self, bucket: Bucket) -> None:
        # Write to a temporary file, then rename to the real name after
        # writing. This avoids another process reading the file before
        # it is fully written.
        name = self._get_cache_filename(bucket)
        f = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(name),
            prefix=os.path.basename(name),
            suffix=".tmp",
            delete=False,
        )

        def remove_silent() -> None:
            try:
                os.remove(f.name)
            except OSError:
                # Another process may have called clear(). On Windows,
                # another program may be holding the file open.
                pass

        try:
            with f:
                bucket.write_bytecode(f)
        except BaseException:
            remove_silent()
            raise

        try:
            os.replace(f.name, name)
        except OSError:
            # Another process may have called clear(). On Windows,
            # another program may be holding the file open.
            remove_silent()
        except BaseException:
            remove_silent()
            raise

    def clear(self) -> None:
        # imported lazily here because google app-engine doesn't support
        # write access on the file system and the function does not exist
        # normally.
        from os import remove

        files = fnmatch.filter(os.listdir(self.directory), self.pattern % ("*",))
        for filename in files:
            try:
                remove(os.path.join(self.directory, filename))
            except OSError:
                pass


class MemcachedBytecodeCache(BytecodeCache):
    """This class implements a bytecode cache that uses a memcache cache for
    storing the information.  It does not enforce a specific memcache library
    (tummy's memcache or cmemcache) but will accept any class that provides
    the minimal interface required.

    Libraries compatible with this class:

    -   `cachelib <https://github.com/pallets/cachelib>`_
    -   `python-memcached <https://pypi.org/project/python-memcached/>`_

    (Unfortunately the django cache interface is not compatible because it
    does not support storing binary data, only text. You can however pass
    the underlying cache client to the bytecode cache which is available
    as `django.core.cache.cache._client`.)

    The minimal interface for the client passed to the constructor is this:

    .. class:: MinimalClientInterface

        .. method:: set(key, value[, timeout])

            Stores the bytecode in the cache.  `value` is a string and
            `timeout` the timeout of the key.  If timeout is not provided
            a default timeout or no timeout should be assumed, if it's
            provided it's an integer with the number of seconds the cache
            item should exist.

        .. method:: get(key)

            Returns the value for the cache key.  If the item does not
            exist in the cache the return value must be `None`.

    The other arguments to the constructor are the prefix for all keys that
    is added before the actual cache key and the timeout for the bytecode in
    the cache system.  We recommend a high (or no) timeout.

    This bytecode cache does not support clearing of used items in the cache.
    The clear method is a no-operation function.

    .. versionadded:: 2.7
       Added support for ignoring memcache errors through the
       `ignore_memcache_errors` parameter.
    """

    def __init__(
        self,
        client: "_MemcachedClient",
        prefix: str = "jinja2/bytecode/",
        timeout: int | None = None,
        ignore_memcache_errors: bool = True,
    ):
        self.client = client
        self.prefix = prefix
        self.timeout = timeout
        self.ignore_memcache_errors = ignore_memcache_errors

    def load_bytecode(self, bucket: Bucket) -> None:
        try:
            code = self.client.get(self.prefix + bucket.key)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
        else:
            bucket.bytecode_from_string(code)

    def dump_bytecode(self, bucket: Bucket) -> None:
        key = self.prefix + bucket.key
        value = bucket.bytecode_to_string()

        try:
            if self.timeout is not None:
                self.client.set(key, value, self.timeout)
            else:
                self.client.set(key, value)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
