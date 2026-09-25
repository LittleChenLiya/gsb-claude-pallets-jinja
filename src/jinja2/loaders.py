"""API and implementations for loading templates from different data
sources.
"""

import importlib.util
import os
import posixpath
import sys
import threading
import typing as t
import weakref
import zipimport
from collections import abc
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha1
from importlib import import_module
from types import ModuleType

from .exceptions import TemplateNotFound
from .utils import internalcode

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template

#: Versions pinned for the currently rendering templates, mapping a
#: :class:`SnapshotLoader` to the snapshot version that template lookups
#: in the current context must use.  This is set while a template loaded
#: from a :class:`SnapshotLoader` renders, so that `extends`, `include`
#: and `import` resolve to the same snapshot version.
_pinned_versions: ContextVar["dict[SnapshotLoader, int] | None"] = ContextVar(
    "jinja2_pinned_snapshot_versions", default=None
)

#: The snapshot version most recently resolved by
#: :meth:`SnapshotLoader.get_cache_key`.  The environment resolves the
#: cache key right before calling :meth:`SnapshotLoader.load`, which
#: consumes this value so that a concurrent :meth:`SnapshotLoader.publish`
#: cannot make the cached template and its cache key disagree.
_key_resolution: ContextVar["tuple[SnapshotLoader, int] | None"] = ContextVar(
    "jinja2_snapshot_key_resolution", default=None
)


def split_template_path(template: str) -> list[str]:
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split("/"):
        if (
            os.sep in piece
            or (os.path.altsep and os.path.altsep in piece)
            or piece == os.path.pardir
        ):
            raise TemplateNotFound(template)
        elif piece and piece != ".":
            pieces.append(piece)
    return pieces


class BaseLoader:
    """Baseclass for all loaders.  Subclass this and override `get_source` to
    implement a custom loading mechanism.  The environment provides a
    `get_template` method that calls the loader's `load` method to get the
    :class:`Template` object.

    A very basic example for a loader that looks up templates on the file
    system could look like this::

        from jinja2 import BaseLoader, TemplateNotFound
        from os.path import join, exists, getmtime

        class MyLoader(BaseLoader):

            def __init__(self, path):
                self.path = path

            def get_source(self, environment, template):
                path = join(self.path, template)
                if not exists(path):
                    raise TemplateNotFound(template)
                mtime = getmtime(path)
                with open(path) as f:
                    source = f.read()
                return source, path, lambda: mtime == getmtime(path)
    """

    #: if set to `False` it indicates that the loader cannot provide access
    #: to the source of templates.
    #:
    #: .. versionadded:: 2.4
    has_source_access = True

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        """Get the template source, filename and reload helper for a template.
        It's passed the environment and template name and has to return a
        tuple in the form ``(source, filename, uptodate)`` or raise a
        `TemplateNotFound` error if it can't locate the template.

        The source part of the returned tuple must be the source of the
        template as a string. The filename should be the name of the
        file on the filesystem if it was loaded from there, otherwise
        ``None``. The filename is used by Python for the tracebacks
        if no loader extension is used.

        The last item in the tuple is the `uptodate` function.  If auto
        reloading is enabled it's always called to check if the template
        changed.  No arguments are passed so the function must store the
        old state somewhere (for example in a closure).  If it returns `False`
        the template will be reloaded.
        """
        if not self.has_source_access:
            raise RuntimeError(
                f"{type(self).__name__} cannot provide access to the source"
            )
        raise TemplateNotFound(template)

    def list_templates(self) -> list[str]:
        """Iterates over all templates.  If the loader does not support that
        it should raise a :exc:`TypeError` which is the default behavior.
        """
        raise TypeError("this loader cannot iterate over all templates")

    def get_cache_key(self, name: str) -> t.Hashable:
        """Return the key the environment uses to cache the compiled
        template for `name`.  The default is the template name itself,
        which is scoped to the environment (and therefore to its
        configuration) because every environment keeps its own cache.
        Loaders that can serve different source for the same name, such
        as :class:`SnapshotLoader`, should override this to include
        the extra state in the key.

        .. versionadded:: 3.2
        """
        return name

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        """Loads a template.  This method looks up the template in the cache
        or loads one by calling :meth:`get_source`.  Subclasses should not
        override this method as loaders working on collections of other
        loaders (such as :class:`PrefixLoader` or :class:`ChoiceLoader`)
        will not call this method but `get_source` directly.
        """
        code = None
        if globals is None:
            globals = {}

        # first we try to get the source for this template together
        # with the filename and the uptodate function.
        source, filename, uptodate = self.get_source(environment, name)

        # try to load the code from the bytecode cache if there is a
        # bytecode cache configured.
        bcc = environment.bytecode_cache
        if bcc is not None:
            bucket = bcc.get_bucket(environment, name, filename, source)
            code = bucket.code

        # if we don't have code so far (not cached, no longer up to
        # date) etc. we compile the template
        if code is None:
            code = environment.compile(source, name, filename)

        # if the bytecode cache is available and the bucket doesn't
        # have a code so far, we give the bucket the new code and put
        # it back to the bytecode cache.
        if bcc is not None and bucket.code is None:
            bucket.code = code
            bcc.set_bucket(bucket)

        return environment.template_class.from_code(
            environment, code, globals, uptodate
        )


class FileSystemLoader(BaseLoader):
    """Load templates from a directory in the file system.

    The path can be relative or absolute. Relative paths are relative to
    the current working directory.

    .. code-block:: python

        loader = FileSystemLoader("templates")

    A list of paths can be given. The directories will be searched in
    order, stopping at the first matching template.

    .. code-block:: python

        loader = FileSystemLoader(["/override/templates", "/default/templates"])

    :param searchpath: A path, or list of paths, to the directory that
        contains the templates.
    :param encoding: Use this encoding to read the text from template
        files.
    :param followlinks: Follow symbolic links in the path.

    .. versionchanged:: 2.8
        Added the ``followlinks`` parameter.
    """

    def __init__(
        self,
        searchpath: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
        encoding: str = "utf-8",
        followlinks: bool = False,
    ) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]

        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str, t.Callable[[], bool]]:
        pieces = split_template_path(template)

        for searchpath in self.searchpath:
            # Use posixpath even on Windows to avoid "drive:" or UNC
            # segments breaking out of the search directory.
            filename = posixpath.join(searchpath, *pieces)

            if os.path.isfile(filename):
                break
        else:
            plural = "path" if len(self.searchpath) == 1 else "paths"
            paths_str = ", ".join(repr(p) for p in self.searchpath)
            raise TemplateNotFound(
                template,
                f"{template!r} not found in search {plural}: {paths_str}",
            )

        with open(filename, encoding=self.encoding) as f:
            contents = f.read()

        mtime = os.path.getmtime(filename)

        def uptodate() -> bool:
            try:
                return os.path.getmtime(filename) == mtime
            except OSError:
                return False

        # Use normpath to convert Windows altsep to sep.
        return contents, os.path.normpath(filename), uptodate

    def list_templates(self) -> list[str]:
        found = set()
        for searchpath in self.searchpath:
            walk_dir = os.walk(searchpath, followlinks=self.followlinks)
            for dirpath, _, filenames in walk_dir:
                for filename in filenames:
                    template = (
                        os.path.join(dirpath, filename)[len(searchpath) :]
                        .strip(os.sep)
                        .replace(os.sep, "/")
                    )
                    if template[:2] == "./":
                        template = template[2:]
                    if template not in found:
                        found.add(template)
        return sorted(found)


if sys.version_info >= (3, 13):

    def _get_zipimporter_files(z: t.Any) -> dict[str, object]:
        try:
            get_files = z._get_files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required metadata to list templates."
            ) from e
        return get_files()

else:

    def _get_zipimporter_files(z: t.Any) -> dict[str, object]:
        try:
            files = z._files
        except AttributeError as e:
            raise TypeError(
                "This zip import does not have the required metadata to list templates."
            ) from e
        return files  # type: ignore[no-any-return]


class PackageLoader(BaseLoader):
    """Load templates from a directory in a Python package.

    :param package_name: Import name of the package that contains the
        template directory.
    :param package_path: Directory within the imported package that
        contains the templates.
    :param encoding: Encoding of template files.

    The following example looks up templates in the ``pages`` directory
    within the ``project.ui`` package.

    .. code-block:: python

        loader = PackageLoader("project.ui", "pages")

    Only packages installed as directories (standard pip behavior) or
    zip/egg files (less common) are supported. The Python API for
    introspecting data in packages is too limited to support other
    installation methods the way this loader requires.

    There is limited support for :pep:`420` namespace packages. The
    template directory is assumed to only be in one namespace
    contributor. Zip files contributing to a namespace are not
    supported.

    .. versionchanged:: 3.0
        No longer uses ``setuptools`` as a dependency.

    .. versionchanged:: 3.0
        Limited PEP 420 namespace package support.
    """

    def __init__(
        self,
        package_name: str,
        package_path: "str" = "templates",
        encoding: str = "utf-8",
    ) -> None:
        package_path = os.path.normpath(package_path).rstrip(os.sep)

        # normpath preserves ".", which isn't valid in zip paths.
        if package_path == os.path.curdir:
            package_path = ""
        elif package_path[:2] == os.path.curdir + os.sep:
            package_path = package_path[2:]

        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding

        # Make sure the package exists. This also makes namespace
        # packages work, otherwise get_loader returns None.
        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, "An import spec was not found for the package."
        loader = spec.loader
        assert loader is not None, "A loader was not found for the package."
        self._loader = loader
        self._archive = None

        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))  # type: ignore
            template_root = os.path.join(pkgdir, package_path).rstrip(os.sep)
        else:
            roots: list[str] = []

            # One element for regular packages, multiple for namespace
            # packages, or None for single module file.
            if spec.submodule_search_locations:
                roots.extend(spec.submodule_search_locations)
            # A single module file, use the parent directory instead.
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))

            if not roots:
                raise ValueError(
                    f"The {package_name!r} package was not installed in a"
                    " way that PackageLoader understands."
                )

            for root in roots:
                root = os.path.join(root, package_path)

                if os.path.isdir(root):
                    template_root = root
                    break
            else:
                raise ValueError(
                    f"PackageLoader could not find a {package_path!r} directory"
                    f" in the {package_name!r} package."
                )

        self._template_root = template_root

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str, t.Callable[[], bool] | None]:
        # Use posixpath even on Windows to avoid "drive:" or UNC
        # segments breaking out of the search directory. Use normpath to
        # convert Windows altsep to sep.
        p = os.path.normpath(
            posixpath.join(self._template_root, *split_template_path(template))
        )
        up_to_date: t.Callable[[], bool] | None

        if self._archive is None:
            # Package is a directory.
            if not os.path.isfile(p):
                raise TemplateNotFound(template)

            with open(p, "rb") as f:
                source = f.read()

            mtime = os.path.getmtime(p)

            def up_to_date() -> bool:
                return os.path.isfile(p) and os.path.getmtime(p) == mtime

        else:
            # Package is a zip file.
            try:
                source = self._loader.get_data(p)  # type: ignore
            except OSError as e:
                raise TemplateNotFound(template) from e

            # Could use the zip's mtime for all template mtimes, but
            # would need to safely reload the module if it's out of
            # date, so just report it as always current.
            up_to_date = None

        return source.decode(self.encoding), p, up_to_date

    def list_templates(self) -> list[str]:
        results: list[str] = []

        if self._archive is None:
            # Package is a directory.
            offset = len(self._template_root)

            for dirpath, _, filenames in os.walk(self._template_root):
                dirpath = dirpath[offset:].lstrip(os.sep)
                results.extend(
                    os.path.join(dirpath, name).replace(os.sep, "/")
                    for name in filenames
                )
        else:
            files = _get_zipimporter_files(self._loader)

            # Package is a zip file.
            prefix = self._template_root[len(self._archive) :].lstrip(os.sep) + os.sep
            offset = len(prefix)

            for name in files:
                # Find names under the templates directory that aren't directories.
                if name.startswith(prefix) and name[-1] != os.sep:
                    results.append(name[offset:].replace(os.sep, "/"))

        results.sort()
        return results


class DictLoader(BaseLoader):
    """Loads a template from a Python dict mapping template names to
    template source.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled by default.
    """

    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, None, t.Callable[[], bool]]:
        if template in self.mapping:
            source = self.mapping[template]
            return source, None, lambda: source == self.mapping.get(template)
        raise TemplateNotFound(template)

    def list_templates(self) -> list[str]:
        return sorted(self.mapping)


class SnapshotLoader(BaseLoader):
    """Load templates from an immutable, versioned snapshot of a
    name-to-source mapping.  A new snapshot is published atomically with
    :meth:`publish`, which validates that every template in the mapping
    compiles before it replaces the current version.  If validation fails,
    the previous version stays active and no version number is consumed.

    Each top-level :meth:`Template.render`, :meth:`Template.render_async`,
    :meth:`Template.generate` and :meth:`Template.generate_async` pins the
    snapshot version of the template it renders.  Templates pulled in with
    ``extends``, ``include`` and ``import`` during that render are loaded
    from the same version, so a concurrent :meth:`publish` never mixes
    parent and child templates from different versions.  Streaming renders
    that have already started keep their version until they are closed.

    The environment cache key includes the snapshot version (and is scoped
    to the environment, and therefore to its configuration), so templates
    compiled from different versions or different environment
    configurations never share a cache entry.

    >>> loader = SnapshotLoader({"index.html": "Hello {{ name }}!"})
    >>> env = Environment(loader=loader)
    >>> loader.publish({"index.html": "Hi {{ name }}!"})
    2

    :param mapping: Initial mapping of template names to source, published
    as version 1.  If omitted, the loader starts empty at version 0
    and the first :meth:`publish` returns version 1.
    :param environment: The environment used to validate templates during
    :meth:`publish`.  If omitted, the environment the loader is first
    bound to is used, falling back to a default environment.

    .. versionadded:: 3.2
    """

    def __init__(
        self,
        mapping: t.Mapping[str, str] | None = None,
        environment: "Environment | None" = None,
    ) -> None:
        self._environment = environment
        self._lock = threading.Lock()
        self._versions: dict[int, dict[str, str]] = {}
        self._version = 0

        if mapping is not None:
            self.publish(mapping)

    @property
    def version(self) -> int:
        """The current snapshot version.  Starts at 0 and increases by 1
        for every successful :meth:`publish`.
        """
        return self._version

    @property
    def versions(self) -> tuple[int, ...]:
        """The successfully published snapshot versions, in ascending
        order.  Failed publishes never appear here.
        """
        return tuple(sorted(self._versions))

    def publish(
        self,
        mapping: t.Mapping[str, str],
        environment: "Environment | None" = None,
    ) -> int:
        """Validate and atomically publish a complete mapping of template
        names to source, replacing the current snapshot.

        Every template in `mapping` is compiled first.  If any template
        fails to compile, the exception is raised, the current snapshot is
        left untouched and no version number is consumed.  Otherwise the
        new snapshot becomes current and its version number, which
        increases monotonically, is returned.

        :param mapping: Complete mapping of template names to source for
        the new snapshot.
        :param environment: Environment used to validate the templates.
        Defaults to the environment given to the constructor, then to
        the environment the loader is bound to.
        """
        env = environment or self._environment

        if env is None:
            from .environment import Environment

            env = Environment()

        snapshot = dict(mapping)

        for name, source in snapshot.items():
            env.compile(source, name=name)

        with self._lock:
            version = self._version + 1
            self._versions[version] = snapshot
            self._version = version

        return version

    def _resolve_version(self) -> int:
        """The version to load from: the pinned version for the current
        render context if there is one, otherwise the current version.
        """
        pins = _pinned_versions.get()

        if pins is not None:
            pinned = pins.get(self)

            if pinned is not None:
                return pinned

        return self._version

    @contextmanager
    def _pin_version(self, version: int) -> t.Iterator[None]:
        """Pin `version` for template lookups in the current context."""
        pins = _pinned_versions.get()
        new_pins = dict(pins) if pins is not None else {}
        new_pins[self] = version
        token = _pinned_versions.set(new_pins)

        try:
            yield
        finally:
            _pinned_versions.reset(token)

    def _snapshot_pin_factory(
        self, template: "Template"
    ) -> "t.Callable[[], t.ContextManager[None]] | None":
        """Return a factory of context managers pinning the snapshot
        version `template` was loaded from, or `None` if the template is
        not versioned.  Used by the environment while rendering.
        """
        version = getattr(template, "_snapshot_version", None)

        if version is None:
            return None

        def factory() -> t.ContextManager[None]:
            return self._pin_version(version)

        return factory

    def get_cache_key(self, name: str) -> t.Hashable:
        version = self._resolve_version()
        _key_resolution.set((self, version))
        return (version, name)

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, None, t.Callable[[], bool]]:
        version = self._resolve_version()
        mapping = self._versions.get(version)

        if mapping is None or template not in mapping:
            raise TemplateNotFound(template)

        # Snapshots are immutable, a cached template never becomes stale.
        return mapping[template], None, lambda: True

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        if self._environment is None:
            self._environment = environment

        # The environment resolves the cache key right before calling this
        # method.  Consume that resolution so the loaded template always
        # matches the key it will be cached under, even if a publish
        # happened in between.
        resolution = _key_resolution.get()

        if resolution is not None and resolution[0] is self:
            _key_resolution.set(None)
            version = resolution[1]
        else:
            version = self._resolve_version()

        with self._pin_version(version):
            template = super().load(environment, name, globals)

        template._snapshot_version = version  # type: ignore[attr-defined]
        return template

    def list_templates(self) -> list[str]:
        return sorted(self._versions.get(self._version, {}))


class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    a string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(
        self,
        load_func: t.Callable[
            [str],
            str | tuple[str, str | None, t.Callable[[], bool] | None] | None,
        ],
    ) -> None:
        self.load_func = load_func

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        rv = self.load_func(template)

        if rv is None:
            raise TemplateNotFound(template)

        if isinstance(rv, str):
            return rv, None, None

        return rv


class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(
        self, mapping: t.Mapping[str, BaseLoader], delimiter: str = "/"
    ) -> None:
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template: str) -> tuple[BaseLoader, str]:
        try:
            prefix, name = template.split(self.delimiter, 1)
            loader = self.mapping[prefix]
        except (ValueError, KeyError) as e:
            raise TemplateNotFound(template) from e
        return loader, name

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(template) from e

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound as e:
            # re-raise the exception with the correct filename here.
            # (the one that includes the prefix)
            raise TemplateNotFound(name) from e

    def list_templates(self) -> list[str]:
        result = []
        for prefix, loader in self.mapping.items():
            for template in loader.list_templates():
                result.append(prefix + self.delimiter + template)
        return result


class ChoiceLoader(BaseLoader):
    """This loader works like the `PrefixLoader` just that no prefix is
    specified.  If a template could not be found by one loader the next one
    is tried.

    >>> loader = ChoiceLoader([
    ...     FileSystemLoader('/path/to/user/templates'),
    ...     FileSystemLoader('/path/to/system/templates')
    ... ])

    This is useful if you want to allow users to override builtin templates
    from a different location.
    """

    def __init__(self, loaders: t.Sequence[BaseLoader]) -> None:
        self.loaders = loaders

    def get_source(
        self, environment: "Environment", template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        for loader in self.loaders:
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(template)

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        for loader in self.loaders:
            try:
                return loader.load(environment, name, globals)
            except TemplateNotFound:
                pass
        raise TemplateNotFound(name)

    def list_templates(self) -> list[str]:
        found = set()
        for loader in self.loaders:
            found.update(loader.list_templates())
        return sorted(found)


class _TemplateModule(ModuleType):
    """Like a normal module but with support for weak references"""


class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ModuleLoader('/path/to/compiled/templates')

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """

    has_source_access = False

    def __init__(
        self,
        path: t.Union[
            str, "os.PathLike[str]", t.Sequence[t.Union[str, "os.PathLike[str]"]]
        ],
    ) -> None:
        package_name = f"_jinja2_module_templates_{id(self):x}"

        # create a fake module that looks for the templates in the
        # path given.
        mod = _TemplateModule(package_name)

        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]

        mod.__path__ = [os.fspath(p) for p in path]

        sys.modules[package_name] = weakref.proxy(
            mod, lambda x: sys.modules.pop(package_name, None)
        )

        # the only strong reference, the sys.modules entry is weak
        # so that the garbage collector can remove it once the
        # loader that created it goes out of business.
        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name: str) -> str:
        return "tmpl_" + sha1(name.encode("utf-8")).hexdigest()

    @staticmethod
    def get_module_filename(name: str) -> str:
        return ModuleLoader.get_template_key(name) + ".py"

    @internalcode
    def load(
        self,
        environment: "Environment",
        name: str,
        globals: t.MutableMapping[str, t.Any] | None = None,
    ) -> "Template":
        key = self.get_template_key(name)
        module = f"{self.package_name}.{key}"
        mod = getattr(self.module, module, None)

        if mod is None:
            try:
                mod = __import__(module, None, None, ["root"])
            except ImportError as e:
                raise TemplateNotFound(name) from e

            # remove the entry from sys.modules, we only want the attribute
            # on the module object we have stored on the loader.
            sys.modules.pop(module, None)

        if globals is None:
            globals = {}

        return environment.template_class.from_module_dict(
            environment, mod.__dict__, globals
        )
