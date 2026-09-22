"""Functions that expose information about templates that might be
interesting for introspection.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .exceptions import TemplateNotFound

if t.TYPE_CHECKING:
    from .environment import Environment
    from .loaders import BaseLoader


__all__ = [
    "DependencyEdge",
    "DependencyGraph",
    "DependencyType",
    "DynamicReference",
    "MissingTemplate",
    "TemplateAnalysisError",
    "TemplateDependency",
    "TemplateInfo",
    "analyze_dependencies",
    "find_referenced_templates",
    "find_undeclared_variables",
]


class TrackingCodeGenerator(CodeGenerator):
    """We abuse the code generator for introspection."""

    def __init__(self, environment: Environment) -> None:
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


class DependencyType(str, Enum):
    """The kind of reference represented by a dependency graph edge.

    .. versionadded:: 3.2.0
    """

    EXTENDS = "extends"
    INCLUDE = "include"
    IMPORT = "import"
    FROM_IMPORT = "from_import"


@dataclass(frozen=True)
class TemplateInfo:
    """Information about a template visited by analyze_dependencies.

    :param name: The name used to look the template up with the loader.
    :param source: The loader-provided origin of the template, usually a
        filename. None for loaders such as DictLoader that do not provide
        one.
    :param loader: A stable description of the loader that supplied the
        template, including the position inside composite loaders such as
        ChoiceLoader and PrefixLoader.
    :param summary: The first non-empty line of the template source, with
        whitespace normalized and long lines truncated.

    .. versionadded:: 3.2.0
    """

    name: str
    source: str | None
    loader: str
    summary: str


@dataclass(frozen=True)
class DependencyEdge:
    """A directed edge from ``source`` to one of its references.

    A reference whose name is a runtime expression has ``target`` set to
    None and is also reported in ``DependencyGraph.dynamic``. A static
    reference to a missing template likewise has ``target`` set to None and
    is reported in ``DependencyGraph.missing``; its ``expression`` is None.

    :param source: Name of the template containing the reference.
    :param target: Resolved template name, or None when unresolved.
    :param type: The kind of reference.
    :param lineno: Line of the reference tag in ``source``.
    :param source_origin: The TemplateInfo of the target, or None when the
        target is missing or unresolved.
    :param ignore_missing: Whether the reference was marked
        ``ignore missing``. Only include references can set this.
    :param candidates: For an include with a constant list of names, the
        names in written order; for a resolved edge these are the names
        tried before ``target``.
    :param expression: Structural description of an unresolved dynamic
        expression. The expression is never evaluated.

    .. versionadded:: 3.2.0
    """

    source: str
    target: str | None
    type: DependencyType
    lineno: int
    source_origin: TemplateInfo | None = None
    ignore_missing: bool = False
    candidates: tuple[str, ...] = ()
    expression: str | None = None

    @property
    def resolved(self) -> bool:
        """Whether the edge points at an existing template."""
        return self.target is not None and self.source_origin is not None


@dataclass(frozen=True)
class TemplateDependency:
    """A resolved static dependency of one template.

    :param target: The resolved name of the referenced template.
    :param type: The kind of reference.
    :param lineno: The line of the reference tag in the referencing template.
    :param ignore_missing: Whether the reference was marked
        ``ignore missing``.
    :param candidates: Constant include-list names tried before ``target``.

    .. versionadded:: 3.2.0
    """

    target: str
    type: DependencyType
    lineno: int
    ignore_missing: bool = False
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class DynamicReference:
    """A reference whose target name is only known while rendering.

    :param source: Name of the template containing the reference.
    :param type: The kind of reference.
    :param lineno: Line of the reference tag.
    :param expression: Structural description of the unresolved expression.
    :param candidates: Known constant names in a partially dynamic include
        list, in written order. Empty for a fully dynamic reference.
    :param ignore_missing: Whether the reference was marked
        ``ignore missing``.

    .. versionadded:: 3.2.0
    """

    source: str
    type: DependencyType
    lineno: int
    expression: str
    candidates: tuple[str, ...] = ()
    ignore_missing: bool = False


@dataclass(frozen=True)
class MissingTemplate:
    """A static reference that the loader could not resolve.

    :param name: The name that could not be loaded, or None for an include
        list where every candidate was missing.
    :param referenced_by: Name of the template containing the reference, or
        None when the analyzed entry template itself is missing.
    :param type: Kind of the reference, or None for the entry template.
    :param lineno: Line of the reference, or None for the entry template.
    :param ignore_missing: When True the diagnostic is a warning rather than
        an error.
    :param candidates: All names tried for an include list, in order.

    .. versionadded:: 3.2.0
    """

    name: str | None
    referenced_by: str | None
    type: DependencyType | None
    lineno: int | None
    ignore_missing: bool = False
    candidates: tuple[str, ...] = ()

    @property
    def severity(self) -> str:
        """Return "warning" with ignore missing, otherwise "error"."""
        return "warning" if self.ignore_missing else "error"


@dataclass(frozen=True)
class TemplateAnalysisError:
    """A loading or syntax error encountered while reading a template.

    :param name: Name of the template being read or parsed.
    :param stage: "load" for a loader error or "parse" for a syntax error.
    :param error_type: Name of the original exception class.
    :param message: Message of the original exception.
    :param lineno: Line reported by a syntax error, otherwise None.
    :param filename: Filename reported by the original exception, if any.
    :param reason: The original exception object, retained for inspection.

    .. versionadded:: 3.2.0
    """

    name: str
    stage: str
    error_type: str
    message: str
    lineno: int | None
    filename: str | None
    reason: BaseException


@dataclass(frozen=True)
class DependencyGraph:
    """Result of analyze_dependencies.

    Ordering is deterministic: ``templates`` follows template discovery
    order during the depth-first walk, ``edges`` follows that order and
    then reference line order, and ``missing``, ``dynamic`` and ``errors``
    follow encounter order. ``cycles`` is sorted.

    :param entry: Name of the analyzed entry template.
    :param templates: Every successfully read template and its TemplateInfo,
        in discovery order.
    :param edges: Every reference found while traversing the templates.
    :param missing: Static references the loader could not resolve.
    :param dynamic: References whose targets depend on runtime expressions.
    :param cycles: Each value is a closed loop of template names with the
        first name repeated at the end, for example ("a", "b", "a").
    :param errors: Loading and syntax errors, with their original causes.

    .. versionadded:: 3.2.0
    """

    entry: str
    templates: t.MutableMapping[str, TemplateInfo] = field(default_factory=dict)
    edges: tuple[DependencyEdge, ...] = ()
    missing: tuple[MissingTemplate, ...] = ()
    dynamic: tuple[DynamicReference, ...] = ()
    cycles: tuple[tuple[str, ...], ...] = ()
    errors: tuple[TemplateAnalysisError, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether there are no errors or non-ignorable missing templates.

        Dynamic references and ignore-missing warnings do not make the
        result unsuccessful.
        """
        return not self.errors and all(
            item.severity != "error" for item in self.missing
        )

    def entries(self, name: str) -> tuple[DependencyEdge, ...]:
        """Return all edges originating from ``name`` in order."""
        return tuple(edge for edge in self.edges if edge.source == name)

    def dependencies(self, name: str) -> tuple[TemplateDependency, ...]:
        """Return the resolved static dependencies of ``name`` in order."""
        return tuple(
            TemplateDependency(
                target=t.cast(str, edge.target),
                type=edge.type,
                lineno=edge.lineno,
                ignore_missing=edge.ignore_missing,
                candidates=edge.candidates,
            )
            for edge in self.entries(name)
            if edge.resolved
        )


def _summarize(source: str, limit: int = 80) -> str:
    for line in source.splitlines():
        stripped = " ".join(line.split())
        if stripped:
            if len(stripped) > limit:
                return stripped[: limit - 1].rstrip() + "..."
            return stripped
    return ""


def _describe_expr(node: nodes.Node) -> str:
    """Describe a name expression structurally without evaluating it."""
    if isinstance(node, nodes.Name):
        return f"variable {node.name!r}"
    if isinstance(node, nodes.Const):
        return f"constant {node.value!r}"
    if isinstance(node, nodes.Getattr):
        return f"{_describe_expr(node.node)}.{node.attr}"
    if isinstance(node, nodes.Getitem):
        return f"{_describe_expr(node.node)}[...]"
    if isinstance(node, nodes.CondExpr):
        else_expr = (
            _describe_expr(node.expr2) if node.expr2 is not None else "undefined"
        )
        return (
            f"conditional ({_describe_expr(node.test)} ? "
            f"{_describe_expr(node.expr1)} : {else_expr})"
        )
    if isinstance(node, nodes.Call):
        return f"call {_describe_expr(node.node)}(...)"
    if isinstance(node, (nodes.Tuple, nodes.List)):
        parts = []
        for item in node.items:
            if isinstance(item, nodes.Const) and isinstance(item.value, str):
                parts.append(repr(item.value))
            else:
                parts.append(_describe_expr(item))
        return "[" + ", ".join(parts) + "]"
    return type(node).__name__


_REF_TYPE_MAP: dict[type[nodes.Node], DependencyType] = {
    nodes.Extends: DependencyType.EXTENDS,
    nodes.Include: DependencyType.INCLUDE,
    nodes.Import: DependencyType.IMPORT,
    nodes.FromImport: DependencyType.FROM_IMPORT,
}

_RefNode = nodes.Extends | nodes.Include | nodes.Import | nodes.FromImport
_RefRecord = tuple[
    DependencyType, int, bool, str | list[str] | None, tuple[str, ...], str
]
_FetchResult = tuple[TemplateInfo, nodes.Template]


def _read_source(
    environment: Environment,
    loader: BaseLoader,
    template: str,
    path: tuple[str, ...] = (),
) -> tuple[str, str | None, str]:
    """Read a template through the loader without compiling it.

    The walk through ChoiceLoader and PrefixLoader mirrors their actual
    lookup order, so the selected child loader matches a real render.
    """
    from .loaders import ChoiceLoader
    from .loaders import PrefixLoader

    if isinstance(loader, ChoiceLoader):
        last_error: TemplateNotFound | None = None
        for index, child in enumerate(loader.loaders):
            label = f"{type(child).__name__}[{index}]"
            try:
                return _read_source(environment, child, template, (*path, label))
            except TemplateNotFound as error:
                last_error = error
        raise TemplateNotFound(template) from last_error

    if isinstance(loader, PrefixLoader):
        prefix_child: BaseLoader | None = None
        local_name = template
        prefix = ""
        try:
            prefix, local_name = template.split(loader.delimiter, 1)
            prefix_child = loader.mapping.get(prefix)
        except ValueError:
            prefix_child = None
        if prefix_child is not None:
            label = f"{type(prefix_child).__name__} prefix={prefix!r}"
            return _read_source(environment, prefix_child, local_name, (*path, label))
        raise TemplateNotFound(template)

    source, filename, _uptodate = loader.get_source(environment, template)
    origin = " > ".join(path) if path else type(loader).__name__
    return source, filename, origin


def _split_reference(
    node: _RefNode,
) -> tuple[str | list[str] | None, tuple[str, ...]]:
    """Return (target, candidates) for one reference node.

    ``target`` is a single name, a list of constant include candidates, or
    None for dynamic references. ``candidates`` collects the constant names
    of a partially dynamic include list.
    """
    template_expr = node.template

    if isinstance(template_expr, nodes.Const):
        if isinstance(template_expr.value, str):
            return template_expr.value, ()
        if isinstance(template_expr.value, (tuple, list)) and all(
            isinstance(item, str) for item in template_expr.value
        ):
            return list(template_expr.value), ()
        return None, ()

    if isinstance(template_expr, (nodes.Tuple, nodes.List)):
        if not template_expr.items:
            return [], ()
        names = [
            item.value
            for item in template_expr.items
            if isinstance(item, nodes.Const) and isinstance(item.value, str)
        ]
        if names and len(names) == len(template_expr.items):
            return names, ()
        return None, tuple(names)

    return None, ()


def analyze_dependencies(environment: Environment, entry_name: str) -> DependencyGraph:
    """Analyze the static template dependency graph rooted at ``entry_name``.

    Templates are only read through the environment loader and parsed with
    :meth:`Environment.parse`; nothing is rendered and no filters, globals
    or user expressions are evaluated.

    ``extends``, ``include``, ``import`` and ``from ... import`` references
    are followed transitively. Each template is parsed at most once and
    dependency cycles complete normally. An ``include`` with a constant list
    resolves to the first name the loader can find, using the same lookup
    order as :meth:`Environment.select_template`, including ChoiceLoader and
    PrefixLoader. ``ignore missing`` only downgrades a missing template from
    an error to a warning. Dynamic name expressions are reported in
    :attr:`DependencyGraph.dynamic` as unresolved edges; names are never
    guessed.

    .. versionadded:: 3.2.0
    """
    if environment.loader is None:
        raise TypeError("no loader for this environment specified")
    root_loader: BaseLoader = environment.loader

    templates: t.MutableMapping[str, TemplateInfo] = {}
    parsed: dict[str, nodes.Template] = {}
    errors: list[TemplateAnalysisError] = []

    def record_error(
        name: str, stage: str, error: BaseException, source: str | None
    ) -> None:
        errors.append(
            TemplateAnalysisError(
                name=name,
                stage=stage,
                error_type=type(error).__name__,
                message=str(error),
                lineno=getattr(error, "lineno", None),
                filename=getattr(error, "filename", source),
                reason=error,
            )
        )

    failed: set[str] = set()

    def fetch(name: str) -> _FetchResult | t.Literal["missing", "error"]:
        """Load and parse a template, returning a status sentinel on failure.

        "missing" means the loader could not find the template, "error"
        means a loader or syntax error was recorded in ``errors``.
        """
        if name in parsed:
            return templates[name], parsed[name]
        if name in failed:
            return "error"
        try:
            source, filename, origin = _read_source(
                environment,
                root_loader,
                name,
                (type(root_loader).__name__,),
            )
        except TemplateNotFound:
            return "missing"
        except Exception as error:
            failed.add(name)
            record_error(name, "load", error, None)
            return "error"

        info = TemplateInfo(
            name=name,
            source=filename,
            loader=origin,
            summary=_summarize(source),
        )
        try:
            ast = environment.parse(source, name=name, filename=filename)
        except Exception as error:
            failed.add(name)
            record_error(name, "parse", error, filename)
            return "error"
        templates[name] = info
        parsed[name] = ast
        return info, ast

    edges_by_name: dict[str, list[DependencyEdge]] = {}
    dynamic_by_name: dict[str, list[DynamicReference]] = {}
    missing_by_name: dict[str, list[MissingTemplate]] = {}

    def extract_refs(
        ast: nodes.Template,
    ) -> list[_RefRecord]:
        refs = []
        for node in ast.find_all(tuple(_REF_TYPE_MAP)):
            assert isinstance(
                node, (nodes.Extends, nodes.Include, nodes.Import, nodes.FromImport)
            )
            ref_type = _REF_TYPE_MAP[type(node)]
            ignore_missing = bool(getattr(node, "ignore_missing", False))
            target, candidates = _split_reference(node)
            refs.append(
                (
                    ref_type,
                    node.lineno,
                    ignore_missing,
                    target,
                    candidates,
                    _describe_expr(node.template),
                )
            )
        return refs

    entry_result = fetch(entry_name)
    if entry_result == "missing":
        missing_by_name[""] = [
            MissingTemplate(
                name=entry_name,
                referenced_by=None,
                type=None,
                lineno=None,
            )
        ]
    elif isinstance(entry_result, tuple):
        _, entry_ast = entry_result
        refs_by_name: dict[str, list[_RefRecord]] = {
            entry_name: extract_refs(entry_ast)
        }
        stack_names: list[str] = [entry_name]
        stack_indexes: list[int] = [0]
        cycle_keys: set[tuple[str, ...]] = set()
        cycles_raw: list[tuple[str, ...]] = []

        def register_successor(
            current: str,
            resolved_name: str,
            info: TemplateInfo,
            ast: nodes.Template,
        ) -> None:
            if resolved_name in stack_names:
                start = stack_names.index(resolved_name)
                cycle = tuple(stack_names[start:] + [resolved_name])
                loop = cycle[:-1]
                key = min(tuple(loop[i:] + loop[:i]) for i in range(len(loop)))
                if key not in cycle_keys:
                    cycle_keys.add(key)
                    cycles_raw.append(cycle)
            elif resolved_name not in refs_by_name:
                refs_by_name[resolved_name] = extract_refs(ast)
                stack_names.append(resolved_name)
                stack_indexes.append(0)

        while stack_names:
            current = stack_names[-1]
            index = stack_indexes[-1]
            refs = refs_by_name[current]
            if index >= len(refs):
                stack_names.pop()
                stack_indexes.pop()
                continue
            stack_indexes[-1] = index + 1
            (
                ref_type,
                lineno,
                ignore_missing,
                target,
                candidates,
                expression,
            ) = refs[index]
            edges_by_name.setdefault(current, [])

            if target is None:
                edges_by_name[current].append(
                    DependencyEdge(
                        source=current,
                        target=None,
                        type=ref_type,
                        lineno=lineno,
                        ignore_missing=ignore_missing,
                        candidates=candidates,
                        expression=expression,
                    )
                )
                dynamic_by_name.setdefault(current, []).append(
                    DynamicReference(
                        source=current,
                        type=ref_type,
                        lineno=lineno,
                        expression=expression,
                        candidates=candidates,
                        ignore_missing=ignore_missing,
                    )
                )
                continue

            if isinstance(target, list):
                if not target:
                    edges_by_name[current].append(
                        DependencyEdge(
                            source=current,
                            target=None,
                            type=ref_type,
                            lineno=lineno,
                            ignore_missing=ignore_missing,
                        )
                    )
                    missing_by_name.setdefault(current, []).append(
                        MissingTemplate(
                            name=None,
                            referenced_by=current,
                            type=ref_type,
                            lineno=lineno,
                            ignore_missing=ignore_missing,
                        )
                    )
                    continue
                chosen: _FetchResult | None = None
                chosen_name = ""
                tried: list[str] = []
                selection_error = False
                for candidate in target:
                    resolved_name = environment.join_path(candidate, current)
                    tried.append(resolved_name)
                    result = fetch(resolved_name)
                    if result == "missing":
                        continue
                    if result == "error":
                        selection_error = True
                        chosen_name = resolved_name
                        break
                    assert isinstance(result, tuple)
                    chosen = result
                    chosen_name = resolved_name
                    break
                tried_before = tuple(tried[:-1])
                if chosen is None:
                    edges_by_name[current].append(
                        DependencyEdge(
                            source=current,
                            target=None,
                            type=ref_type,
                            lineno=lineno,
                            ignore_missing=ignore_missing,
                            candidates=tuple(tried + [chosen_name])
                            if selection_error
                            else tuple(tried),
                        )
                    )
                    if not selection_error:
                        missing_by_name.setdefault(current, []).append(
                            MissingTemplate(
                                name=None,
                                referenced_by=current,
                                type=ref_type,
                                lineno=lineno,
                                ignore_missing=ignore_missing,
                                candidates=tuple(tried),
                            )
                        )
                    continue
                chosen_info, chosen_ast = chosen
                edges_by_name[current].append(
                    DependencyEdge(
                        source=current,
                        target=chosen_name,
                        type=ref_type,
                        lineno=lineno,
                        source_origin=chosen_info,
                        ignore_missing=ignore_missing,
                        candidates=tried_before,
                    )
                )
                register_successor(
                    current,
                    chosen_name,
                    chosen_info,
                    chosen_ast,
                )
                continue

            resolved_name = environment.join_path(target, current)
            result = fetch(resolved_name)
            if result == "missing":
                edges_by_name[current].append(
                    DependencyEdge(
                        source=current,
                        target=None,
                        type=ref_type,
                        lineno=lineno,
                        ignore_missing=ignore_missing,
                    )
                )
                missing_by_name.setdefault(current, []).append(
                    MissingTemplate(
                        name=resolved_name,
                        referenced_by=current,
                        type=ref_type,
                        lineno=lineno,
                        ignore_missing=ignore_missing,
                    )
                )
                continue
            if result == "error":
                edges_by_name[current].append(
                    DependencyEdge(
                        source=current,
                        target=resolved_name,
                        type=ref_type,
                        lineno=lineno,
                        ignore_missing=ignore_missing,
                    )
                )
                continue
            assert isinstance(result, tuple)
            info, ast = result
            edges_by_name[current].append(
                DependencyEdge(
                    source=current,
                    target=resolved_name,
                    type=ref_type,
                    lineno=lineno,
                    source_origin=info,
                    ignore_missing=ignore_missing,
                )
            )
            register_successor(current, resolved_name, info, ast)

    ordered_names = list(templates)
    ordered_edges: list[DependencyEdge] = []
    ordered_dynamic: list[DynamicReference] = []
    ordered_missing: list[MissingTemplate] = list(missing_by_name.get("", []))
    for name in ordered_names:
        ordered_edges.extend(edges_by_name.get(name, ()))
        ordered_dynamic.extend(dynamic_by_name.get(name, ()))
        ordered_missing.extend(missing_by_name.get(name, ()))

    return DependencyGraph(
        entry=entry_name,
        templates=templates,
        edges=tuple(ordered_edges),
        missing=tuple(ordered_missing),
        dynamic=tuple(ordered_dynamic),
        cycles=tuple(sorted(cycles_raw))
        if entry_result not in ("missing", "error")
        else (),
        errors=tuple(errors),
    )
