"""API and implementations for loading templates from different data
sources.
"""

import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType

from .exceptions import TemplateNotFound
from .utils import internalcode

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template


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

    def explain(
        self,
        environment: "Environment",
        template: str,
        adapters: t.Mapping[type, "LoaderExplainAdapter"] | None = None,
    ) -> "LoaderTry":
        """Explain which sources this loader would try for *template*,
        without rendering the template.

        This is a shortcut for :func:`explain_loader`.  It returns a
        :class:`LoaderTry` tree that preserves the order in which sources
        are tried, marks sources that are shadowed by an earlier hit, and
        records the final hit.

        Custom loaders are not inspected by calling their ``get_source``
        method.  Provide a :class:`LoaderExplainAdapter` for them via
        *adapters*; loaders without an adapter are recorded as
        :data:`EXPLAIN_UNSUPPORTED`.
        """
        return explain_loader(self, environment, template, adapters)

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


# ---------------------------------------------------------------------------
# Loader explain
# ---------------------------------------------------------------------------

#: The loader would find the template.
EXPLAIN_HIT = "hit"
#: The loader would not find the template.
EXPLAIN_MISS = "miss"
#: The loader was not tried because an earlier loader already hit.
EXPLAIN_SHADOWED = "shadowed"
#: The loader cannot be inspected because no explain adapter is registered.
EXPLAIN_UNSUPPORTED = "unsupported"
#: The explain adapter for the loader raised an exception.
EXPLAIN_ERROR = "error"


class LoaderExplainAdapter:
    """Adapters teach :func:`explain_loader` how to inspect a loader without
    calling its ``get_source`` method.

    This is intended for custom loaders.  Instead of guessing what the
    loader would do by calling :meth:`BaseLoader.get_source` (which may have
    side effects such as counting calls or doing network I/O), an adapter
    is called with the same arguments and returns a :class:`LoaderTry`
    describing the result.

    >>> class MyAdapter(LoaderExplainAdapter):
    ...     def explain(self, loader, environment, template):
    ...         if template in loader.templates:
    ...             return LoaderTry(loader, template, EXPLAIN_HIT)
    ...         return LoaderTry(loader, template, EXPLAIN_MISS)
    """

    def explain(
        self,
        loader: "BaseLoader",
        environment: "Environment",
        template: str,
    ) -> "LoaderTry":
        raise NotImplementedError()


class LoaderTry:
    """A single attempt in an explanation tree.

    Returned by :func:`explain_loader`.  Each node describes one loader being
    asked for one name.  Composite loaders (such as :class:`ChoiceLoader` and
    :class:`PrefixLoader`) have :attr:`children` describing the loaders they
    delegate to.

    Attributes:
        loader: the loader that made the attempt.
        name: the name the loader was asked for.  This may differ from the
            name passed to :func:`explain_loader` because a
            :class:`PrefixLoader` strips the prefix before delegating.
        status: one of :data:`EXPLAIN_HIT`, :data:`EXPLAIN_MISS`,
            :data:`EXPLAIN_SHADOWED`, :data:`EXPLAIN_UNSUPPORTED` or
            :data:`EXPLAIN_ERROR`.
        children: attempts made by sub-loaders.
        message: an optional human readable explanation of the result.
    """

    __slots__ = ("loader", "name", "status", "children", "message")

    def __init__(
        self,
        loader: "BaseLoader",
        name: str,
        status: str,
        children: t.Sequence["LoaderTry"] = (),
        message: str | None = None,
    ) -> None:
        self.loader = loader
        self.name = name
        self.status = status
        self.children = tuple(children)
        self.message = message

    @property
    def hit(self) -> bool:
        """`True` if the loader would find the template."""
        return self.status == EXPLAIN_HIT

    def iter_tries(self) -> t.Iterator["LoaderTry"]:
        """Iterate over all attempts in the tree, depth first."""
        yield self
        for child in self.children:
            yield from child.iter_tries()

    def render(self) -> str:
        """Render the tree as a human readable string, showing the order in
        which loaders are tried, name changes and shadowing.
        """
        lines: list[str] = []
        self._render(lines, 0)
        return "\n".join(lines)

    def _render(self, lines: list[str], depth: int) -> None:
        indent = "  " * depth
        label = f"{indent}{type(self.loader).__name__}({self.name!r}) -> {self.status}"
        if self.message:
            label += f"  [{self.message}]"
        lines.append(label)
        for child in self.children:
            child._render(lines, depth + 1)

    def __repr__(self) -> str:
        return (
            f"LoaderTry({type(self.loader).__name__}, {self.name!r}, {self.status!r})"
        )


class _DictLoaderExplainAdapter(LoaderExplainAdapter):
    def explain(
        self,
        loader: "DictLoader",
        environment: "Environment",
        template: str,
    ) -> LoaderTry:
        if template in loader.mapping:
            return LoaderTry(loader, template, EXPLAIN_HIT, message="found in mapping")
        return LoaderTry(loader, template, EXPLAIN_MISS, message="not in mapping")


class _FileSystemLoaderExplainAdapter(LoaderExplainAdapter):
    def explain(
        self,
        loader: "FileSystemLoader",
        environment: "Environment",
        template: str,
    ) -> LoaderTry:
        pieces = split_template_path(template)
        children: list[LoaderTry] = []
        status = EXPLAIN_MISS

        for searchpath in loader.searchpath:
            filename = posixpath.join(searchpath, *pieces)

            if os.path.isfile(filename):
                children.append(
                    LoaderTry(loader, filename, EXPLAIN_HIT, message="file exists")
                )
                status = EXPLAIN_HIT
                break

            children.append(
                LoaderTry(loader, filename, EXPLAIN_MISS, message="no such file")
            )

        if status == EXPLAIN_HIT:
            message = (
                f"found in search path {len(children)} of {len(loader.searchpath)}"
            )
        else:
            message = f"not found in any of {len(loader.searchpath)} search paths"

        return LoaderTry(loader, template, status, children=children, message=message)


class _PackageLoaderExplainAdapter(LoaderExplainAdapter):
    def explain(
        self,
        loader: "PackageLoader",
        environment: "Environment",
        template: str,
    ) -> LoaderTry:
        p = os.path.normpath(
            posixpath.join(loader._template_root, *split_template_path(template))
        )

        if loader._archive is None:
            exists = os.path.isfile(p)
        else:
            names = {
                name.replace("/", os.sep)
                for name in _get_zipimporter_files(loader._loader)
            }
            exists = p in names

        if exists:
            return LoaderTry(loader, template, EXPLAIN_HIT, message=f"found at {p}")
        return LoaderTry(loader, template, EXPLAIN_MISS, message=f"not found at {p}")


class _ModuleLoaderExplainAdapter(LoaderExplainAdapter):
    def explain(
        self,
        loader: "ModuleLoader",
        environment: "Environment",
        template: str,
    ) -> LoaderTry:
        filename = ModuleLoader.get_module_filename(template)
        paths = [os.path.join(p, filename) for p in loader.module.__path__]

        for path in paths:
            if os.path.isfile(path):
                return LoaderTry(
                    loader,
                    template,
                    EXPLAIN_HIT,
                    message=f"compiled template exists at {path}",
                )

        return LoaderTry(
            loader, template, EXPLAIN_MISS, message="no compiled template module"
        )


#: Built-in adapters used when no user adapter is registered for a loader.
_DEFAULT_EXPLAIN_ADAPTERS: dict[type, LoaderExplainAdapter] = {}


def explain_loader(
    loader: "BaseLoader",
    environment: "Environment",
    template: str,
    adapters: t.Mapping[type, LoaderExplainAdapter] | None = None,
) -> LoaderTry:
    """Explain which sources *loader* would try for *template*, without
    rendering the template.

    Returns a :class:`LoaderTry` tree.  The tree preserves the order in which
    composite loaders try their children, marks children that are shadowed by
    an earlier hit, and records the final hit.  The name stored on each node
    is the name that loader was actually asked for, so prefix stripping is
    visible in the tree.

    Custom loaders are not inspected by calling their ``get_source`` method.
    Provide a :class:`LoaderExplainAdapter` for them via *adapters*; loaders
    without an adapter are recorded as :data:`EXPLAIN_UNSUPPORTED`.

    :param adapters: an optional mapping of loader classes to adapters.  User
        adapters take precedence over the built-in adapters and are matched
        with :func:`isinstance` semantics.
    """
    return _explain_loader(loader, environment, template, dict(adapters or {}))


def _explain_loader(
    loader: "BaseLoader",
    environment: "Environment",
    template: str,
    adapters: dict[type, LoaderExplainAdapter],
) -> LoaderTry:
    if isinstance(loader, ChoiceLoader):
        return _explain_choice_loader(loader, environment, template, adapters)

    if isinstance(loader, PrefixLoader):
        return _explain_prefix_loader(loader, environment, template, adapters)

    return _explain_leaf_loader(loader, environment, template, adapters)


def _explain_choice_loader(
    loader: "ChoiceLoader",
    environment: "Environment",
    template: str,
    adapters: dict[type, LoaderExplainAdapter],
) -> LoaderTry:
    children: list[LoaderTry] = []
    shadowed = False

    for sub in loader.loaders:
        if shadowed:
            # A loader already hit, the rest would never be called.
            children.append(
                LoaderTry(
                    sub,
                    template,
                    EXPLAIN_SHADOWED,
                    message="shadowed by an earlier hit",
                )
            )
            continue

        child = _explain_loader(sub, environment, template, adapters)
        children.append(child)

        if child.hit:
            shadowed = True

    status = EXPLAIN_HIT if shadowed else EXPLAIN_MISS
    return LoaderTry(loader, template, status, children=children)


def _explain_prefix_loader(
    loader: "PrefixLoader",
    environment: "Environment",
    template: str,
    adapters: dict[type, LoaderExplainAdapter],
) -> LoaderTry:
    try:
        prefix, name = template.split(loader.delimiter, 1)
        sub = loader.mapping[prefix]
    except ValueError:
        return LoaderTry(
            loader,
            template,
            EXPLAIN_MISS,
            message=f"name does not contain delimiter {loader.delimiter!r}",
        )
    except KeyError:
        return LoaderTry(
            loader,
            template,
            EXPLAIN_MISS,
            message=f"no loader registered for prefix {prefix!r}",
        )

    # Only the matching branch is entered, with the prefix stripped.  The
    # stripped name must not leak into any other branch.
    child = _explain_loader(sub, environment, name, adapters)
    return LoaderTry(
        loader,
        template,
        child.status,
        children=[child],
        message=f"prefix {prefix!r} delegates {template!r} to {name!r}",
    )


def _explain_leaf_loader(
    loader: "BaseLoader",
    environment: "Environment",
    template: str,
    adapters: dict[type, LoaderExplainAdapter],
) -> LoaderTry:
    adapter = _find_explain_adapter(loader, adapters)

    if adapter is None:
        return LoaderTry(
            loader,
            template,
            EXPLAIN_UNSUPPORTED,
            message=f"no explain adapter for {type(loader).__name__}",
        )

    try:
        node = adapter.explain(loader, environment, template)
    except Exception as e:
        return LoaderTry(
            loader,
            template,
            EXPLAIN_ERROR,
            message=f"adapter raised {type(e).__name__}: {e}",
        )

    if not isinstance(node, LoaderTry):
        return LoaderTry(
            loader,
            template,
            EXPLAIN_ERROR,
            message="adapter did not return a LoaderTry",
        )

    return node


def _find_explain_adapter(
    loader: "BaseLoader", adapters: dict[type, LoaderExplainAdapter]
) -> LoaderExplainAdapter | None:
    for cls in type(loader).__mro__:
        if cls in adapters:
            return adapters[cls]

        if cls in _DEFAULT_EXPLAIN_ADAPTERS:
            return _DEFAULT_EXPLAIN_ADAPTERS[cls]

    return None


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


_DEFAULT_EXPLAIN_ADAPTERS.update(
    {
        DictLoader: _DictLoaderExplainAdapter(),
        FileSystemLoader: _FileSystemLoaderExplainAdapter(),
        PackageLoader: _PackageLoaderExplainAdapter(),
        ModuleLoader: _ModuleLoaderExplainAdapter(),
    }
)
