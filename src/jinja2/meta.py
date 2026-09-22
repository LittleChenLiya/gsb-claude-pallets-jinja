"""Functions that expose information about templates that might be
interesting for introspection.
"""

import typing as t
from dataclasses import dataclass
from dataclasses import field

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .exceptions import TemplateNotFound
from .exceptions import TemplateSyntaxError
from .loaders import BaseLoader
from .loaders import ChoiceLoader
from .loaders import PrefixLoader

if t.TYPE_CHECKING:
    from .environment import Environment


class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment: "Environment") -> None:
        super().__init__(environment, "<introspection>", "<introspection>")
        self.undeclared_identifiers: set[str] = set()

    def write(self, x: str) -> None:
        """Don't write."""

    def enter_frame(self, frame: Frame) -> None:
        """Remember all undeclared identifiers."""
        super().enter_frame(frame)

        for _, (action, param) in frame.symbols.loads.items():
            if action == "resolve" and param not in self.environment.globals:
                self.undeclared_identifiers.add(param)


def find_undeclared_variables(ast: nodes.Template) -> set[str]:
    """Returns a set of all variables in the AST that will be looked up from
    the context at runtime.  Because at compile time it's not known which
    variables will be used depending on the path the execution takes at
    runtime, all variables are returned.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
    >>> meta.find_undeclared_variables(ast) == {'bar'}
    True

    .. admonition:: Implementation

       Internally the code generator is used for finding undeclared variables.
       This is good to know because the code generator might raise a
       :exc:`TemplateAssertionError` during compilation and as a matter of
       fact this function can currently raise that exception as well.
    """
    codegen = TrackingCodeGenerator(ast.environment)  # type: ignore
    codegen.visit(ast)
    return codegen.undeclared_identifiers


_ref_types = (nodes.Extends, nodes.FromImport, nodes.Import, nodes.Include)
_RefType = nodes.Extends | nodes.FromImport | nodes.Import | nodes.Include


def find_referenced_templates(ast: nodes.Template) -> t.Iterator[str | None]:
    """Finds all the referenced templates from the AST.  This will return an
    iterator over all the hardcoded template extensions, inclusions and
    imports.  If dynamic inheritance or inclusion is used, `None` will be
    yielded.

    >>> from jinja2 import Environment, meta
    >>> env = Environment()
    >>> ast = env.parse('{% extends "layout.html" %}{% include helper %}')
    >>> list(meta.find_referenced_templates(ast))
    ['layout.html', None]

    This function is useful for dependency tracking.  For example if you want
    to rebuild parts of the website after a layout template has changed.
    """
    template_name: t.Any

    for node in ast.find_all(_ref_types):
        template: nodes.Expr = node.template  # type: ignore

        if not isinstance(template, nodes.Const):
            # a tuple with some non consts in there
            if isinstance(template, (nodes.Tuple, nodes.List)):
                for template_name in template.items:
                    # something const, only yield the strings and ignore
                    # non-string consts that really just make no sense
                    if isinstance(template_name, nodes.Const):
                        if isinstance(template_name.value, str):
                            yield template_name.value
                    # something dynamic in there
                    else:
                        yield None
            # something dynamic we don't know about here
            else:
                yield None
            continue
        # constant is a basestring, direct template name
        if isinstance(template.value, str):
            yield template.value
        # a tuple or list (latter *should* not happen) made of consts,
        # yield the consts that are strings.  We could warn here for
        # non string values
        elif isinstance(node, nodes.Include) and isinstance(
            template.value, (tuple, list)
        ):
            for template_name in template.value:
                if isinstance(template_name, str):
                    yield template_name
        # something else we don't care about, we could warn here
        else:
            yield None


_ref_kinds: dict[type[nodes.Node], str] = {
    nodes.Extends: "extends",
    nodes.Include: "include",
    nodes.Import: "import",
    nodes.FromImport: "from_import",
}


@dataclass(frozen=True)
class TemplateReference:
    """A single edge in the template dependency graph.  Describes one
    ``extends``, ``include``, ``import`` or ``from import``
    statement found while statically analyzing a template.

    .. versionadded:: 3.2
    """

    #: Name of the template that contains the statement.
    source: str
    #: One of ``"extends"``, ``"include"``, ``"import"`` or
    #: ``"from_import"``.
    kind: str
    #: Line number of the statement in the source template, 1-based.
    lineno: int
    #: The resolved template name, or ``None`` if the reference is
    #: dynamic or the target could not be found.
    target: str | None
    #: Whether the reference could be resolved to an existing template.
    resolved: bool
    #: Whether the reference uses a dynamic expression.  Dynamic references
    #: are never resolved to a guessed name.
    dynamic: bool
    #: Description of the loader chain that provided the source template,
    #: for example ``"ChoiceLoader[0] -> DictLoader"``.
    source_origin: str | None
    #: Description of the loader chain that would provide the target
    #: template, or ``None`` if the reference is unresolved.
    target_origin: str | None
    #: A short summary of the source line that contains the statement.
    snippet: str
    #: Whether this is an ``include`` with ``ignore missing``.
    ignore_missing: bool = False


@dataclass(frozen=True)
class DependencyIssue:
    """A problem found while analyzing the dependency graph, such as a
    missing template, a loader read error, or a syntax error.

    .. versionadded:: 3.2
    """

    #: Name of the template the problem is about.
    template: str
    #: Name of the template that referenced it, or ``None`` for the
    #: entry template itself.
    referenced_by: str | None
    #: Line number related to the problem, or ``None`` if not
    #: applicable.
    lineno: int | None
    #: One of ``"missing"``, ``"read"`` or ``"syntax"``.
    kind: str
    #: ``"error"`` or ``"warning"``.  Missing templates referenced
    #: by an ``include`` with ``ignore missing`` are reported as
    #: warnings.
    severity: str
    #: The original exception type and message.
    reason: str


@dataclass
class TemplateDependencyGraph:
    """The result of :func:`analyze_template_dependencies`.  All lists
    are in a deterministic discovery order so repeated runs over unchanged
    templates produce identical output.

    .. versionadded:: 3.2
    """

    #: The template name the analysis started from.
    entry: str
    #: Names of all templates that were loaded and parsed, in visit order.
    #: Each template appears exactly once.
    templates: list[str] = field(default_factory=list)
    #: Maps each visited template name to its loader origin description.
    origins: dict[str, str] = field(default_factory=dict)
    #: All reference edges, in discovery order.
    edges: list[TemplateReference] = field(default_factory=list)
    #: All diagnostics, in discovery order.
    issues: list[DependencyIssue] = field(default_factory=list)
    #: Dependency cycles, each a list of template names where the last
    #: element repeats the first.  Cycles are reported but do not stop the
    #: analysis.
    cycles: list[list[str]] = field(default_factory=list)

    @property
    def dynamic_edges(self) -> list[TemplateReference]:
        """Edges whose target is a dynamic expression and were left
        unresolved."""
        return [edge for edge in self.edges if edge.dynamic]

    @property
    def missing(self) -> list[DependencyIssue]:
        """Issues about templates that could not be found."""
        return [issue for issue in self.issues if issue.kind == "missing"]

    @property
    def errors(self) -> list[DependencyIssue]:
        """Issues with ``"error"`` severity."""
        return [issue for issue in self.issues if issue.severity == "error"]


def _find_loader_source(
    loader: BaseLoader, environment: "Environment", name: str, trail: list[str]
) -> tuple[str, str]:
    """Return ``(source, origin)`` for *name*, mirroring the lookup
    order of :class:`ChoiceLoader` and :class:`PrefixLoader` so the
    reported origin matches what an actual load would use.
    """
    desc = type(loader).__name__

    if isinstance(loader, ChoiceLoader):
        for index, sub_loader in enumerate(loader.loaders):
            try:
                return _find_loader_source(
                    sub_loader, environment, name, [*trail, f"{desc}[{index}]"]
                )
            except TemplateNotFound:
                continue

        raise TemplateNotFound(name)

    if isinstance(loader, PrefixLoader):
        sub_loader, local_name = loader.get_loader(name)
        prefix = name.split(loader.delimiter, 1)[0]

        try:
            return _find_loader_source(
                sub_loader, environment, local_name, [*trail, f"{desc}[{prefix!r}]"]
            )
        except TemplateNotFound:
            raise TemplateNotFound(name) from None

    source, _, _ = loader.get_source(environment, name)
    return source, " -> ".join([*trail, desc])


def _reference_candidates(node: _RefType) -> list[str] | None:
    """Return the constant candidate names for a reference node, or
    ``None`` if the reference uses a dynamic expression.
    """
    template: nodes.Expr = node.template

    if isinstance(template, nodes.Const):
        if isinstance(template.value, str):
            return [template.value]

        if isinstance(node, nodes.Include) and isinstance(
            template.value, (tuple, list)
        ):
            if all(isinstance(item, str) for item in template.value):
                return list(template.value)

        return None

    if isinstance(node, nodes.Include) and isinstance(
        template, (nodes.Tuple, nodes.List)
    ):
        candidates = []

        for item in template.items:
            if isinstance(item, nodes.Const) and isinstance(item.value, str):
                candidates.append(item.value)
            else:
                return None

        return candidates

    return None


def analyze_template_dependencies(
    environment: "Environment", template_name: str
) -> TemplateDependencyGraph:
    """Statically analyze the dependency graph of the template
    *template_name* in the given environment, following ``extends``,
    ``include``, ``import`` and ``from import`` statements.

    The analysis only reads and parses template source code.  Templates
    are never rendered, and no filters, global functions, tests, or user
    expressions are executed.  Each template is loaded and parsed at most
    once; cyclic references are reported in
    :attr:`TemplateDependencyGraph.cycles` and do not stop the analysis.

    References that use dynamic expressions are reported as unresolved
    edges (:attr:`TemplateDependencyGraph.dynamic_edges`) without
    guessing a name.  Missing templates and read or syntax errors are
    reported in :attr:`TemplateDependencyGraph.issues`; an ``include``
    with ``ignore missing`` only lowers the severity of its missing
    diagnostics to a warning.

    When an ``include`` uses a constant list of names, the first name
    that exists according to the loader's actual lookup order is selected,
    just like at render time.

    >>> from jinja2 import DictLoader, Environment, meta
    >>> env = Environment(loader=DictLoader({
    ...     "page": '{% extends "layout" %}{% include "partial" %}',
    ...     "layout": "",
    ...     "partial": "",
    ... }))
    >>> graph = meta.analyze_template_dependencies(env, "page")
    >>> [(edge.kind, edge.target) for edge in graph.edges]
    [('extends', 'layout'), ('include', 'partial')]

    .. versionadded:: 3.2
    """
    graph = TemplateDependencyGraph(entry=template_name)
    resolve_cache: dict[str, tuple[str | None, str | None, BaseException | None]] = {}

    def resolve(
        name: str,
    ) -> tuple[str | None, str | None, BaseException | None]:
        if name not in resolve_cache:
            if environment.loader is None:
                resolve_cache[name] = (None, None, TemplateNotFound(name))
            else:
                try:
                    source, origin = _find_loader_source(
                        environment.loader, environment, name, []
                    )
                    resolve_cache[name] = (source, origin, None)
                except BaseException as e:
                    resolve_cache[name] = (None, None, e)

        return resolve_cache[name]

    def add_issue(
        name: str,
        referenced_by: str | None,
        lineno: int | None,
        error: BaseException,
        severity: str = "error",
    ) -> None:
        if isinstance(error, TemplateSyntaxError):
            kind = "syntax"
            lineno = error.lineno
        elif isinstance(error, TemplateNotFound):
            kind = "missing"
        else:
            kind = "read"

        graph.issues.append(
            DependencyIssue(
                template=name,
                referenced_by=referenced_by,
                lineno=lineno,
                kind=kind,
                severity=severity,
                reason=f"{type(error).__name__}: {error}",
            )
        )

    visited: set[str] = set()

    def visit(name: str, stack: list[str]) -> None:
        visited.add(name)
        source, origin, error = resolve(name)

        if error is not None:
            add_issue(name, None, None, error)
            return

        assert source is not None and origin is not None
        graph.templates.append(name)
        graph.origins[name] = origin

        try:
            ast = environment.parse(source)
        except TemplateSyntaxError as e:
            add_issue(name, None, e.lineno, e)
            return

        lines = source.splitlines()

        for node in t.cast("t.Iterator[_RefType]", ast.find_all(_ref_types)):
            kind = _ref_kinds[type(node)]
            lineno = node.lineno
            snippet = ""

            if 0 < lineno <= len(lines):
                snippet = lines[lineno - 1].strip()[:120]

            ignore_missing = isinstance(node, nodes.Include) and node.ignore_missing
            candidates = _reference_candidates(node)

            if candidates is None:
                graph.edges.append(
                    TemplateReference(
                        source=name,
                        kind=kind,
                        lineno=lineno,
                        target=None,
                        resolved=False,
                        dynamic=True,
                        source_origin=origin,
                        target_origin=None,
                        snippet=snippet,
                        ignore_missing=ignore_missing,
                    )
                )
                continue

            if not candidates:
                continue

            chosen: tuple[str, str] | None = None
            failures: list[tuple[str, BaseException]] = []

            for candidate in candidates:
                _, candidate_origin, candidate_error = resolve(candidate)

                if candidate_error is None:
                    assert candidate_origin is not None
                    chosen = (candidate, candidate_origin)
                    break

                failures.append((candidate, candidate_error))

            if chosen is None:
                severity = "warning" if ignore_missing else "error"

                for candidate, candidate_error in failures:
                    add_issue(candidate, name, lineno, candidate_error, severity)

                graph.edges.append(
                    TemplateReference(
                        source=name,
                        kind=kind,
                        lineno=lineno,
                        target=None,
                        resolved=False,
                        dynamic=False,
                        source_origin=origin,
                        target_origin=None,
                        snippet=snippet,
                        ignore_missing=ignore_missing,
                    )
                )
                continue

            target, target_origin = chosen
            graph.edges.append(
                TemplateReference(
                    source=name,
                    kind=kind,
                    lineno=lineno,
                    target=target,
                    resolved=True,
                    dynamic=False,
                    source_origin=origin,
                    target_origin=target_origin,
                    snippet=snippet,
                    ignore_missing=ignore_missing,
                )
            )

            if target in stack:
                graph.cycles.append([*stack[stack.index(target) :], target])
            elif target not in visited:
                visit(target, [*stack, target])

    visit(template_name, [template_name])
    return graph
