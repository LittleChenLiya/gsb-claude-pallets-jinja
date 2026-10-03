"""Read-only macro contract exporter.

This module inspects the abstract syntax tree of a template without
executing any of it.  It reports the *contract* of every macro: its name,
the ordered list of parameters, the source position of every default
expression, whether the macro receives a ``caller`` and whether it reads
the special ``kwargs`` / ``varargs`` catch-all parameters.

The rules mirror the ones the compiler applies in
:meth:`jinja2.compiler.CodeGenerator.macro_body`:

* defaults are aligned to the *end* of the parameter list,
* an explicit ``caller`` parameter must be omitted or given a default,
* a body that reads ``caller`` / ``kwargs`` / ``varargs`` without
  declaring them gets the corresponding implicit special parameter.

Nothing is loaded or executed.  Dynamic includes are reported as
unknown instead of being resolved.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass
from dataclasses import field

from . import nodes
from .exceptions import TemplateSyntaxError
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from .environment import Environment

__all__ = (
    "ContractError",
    "ContractReport",
    "DynamicInclude",
    "MacroContract",
    "MacroParam",
    "export_macro_contracts",
    "macro_contracts",
)


@dataclass
class MacroParam:
    """A single macro parameter.

    `lineno` is the position of the parameter name.  `default_lineno` is
    the position of the *default expression itself* (``None`` when the
    parameter has no default).  It deliberately is not the position of
    the whole macro definition.
    """

    name: str
    lineno: int
    default_lineno: int | None = None

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "default_lineno": self.default_lineno,
        }


@dataclass
class MacroContract:
    """The exported contract of one macro definition."""

    name: str
    lineno: int
    scope: tuple[str, ...]
    params: list[MacroParam]
    receives_caller: bool
    accesses_kwargs: bool
    accesses_varargs: bool

    @property
    def accepts_extra_keywords(self) -> bool:
        """`True` when the macro can be given extra keyword arguments.

        This is the case whenever the body reads the implicit ``kwargs``
        catch-all or declares an explicit ``kwargs`` parameter.
        """
        return self.accesses_kwargs or any(p.name == "kwargs" for p in self.params)

    @property
    def accepts_extra_positionals(self) -> bool:
        """`True` when the macro can be given extra positional arguments."""
        return self.accesses_varargs or any(
            p.name == "varargs" for p in self.params
        )

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "scope": list(self.scope),
            "params": [p.to_dict() for p in self.params],
            "receives_caller": self.receives_caller,
            "accesses_kwargs": self.accesses_kwargs,
            "accesses_varargs": self.accesses_varargs,
            "accepts_extra_keywords": self.accepts_extra_keywords,
            "accepts_extra_positionals": self.accepts_extra_positionals,
        }


@dataclass
class DynamicInclude:
    """An ``{% include %}`` whose template name is not a literal string.

    The exporter cannot know which macros the included template defines
    without loading it, so it is reported as unknown instead.
    """

    lineno: int
    template: str

    def to_dict(self) -> dict[str, t.Any]:
        return {"lineno": self.lineno, "template": self.template}


@dataclass
class ContractError:
    """A syntax or macro-rule error found while exporting contracts."""

    lineno: int | None
    message: str

    def to_dict(self) -> dict[str, t.Any]:
        return {"lineno": self.lineno, "message": self.message}


@dataclass
class ContractReport:
    """The result of exporting macro contracts from one template."""

    macros: list[MacroContract] = field(default_factory=list)
    dynamic_includes: list[DynamicInclude] = field(default_factory=list)
    errors: list[ContractError] = field(default_factory=list)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "macros": [m.to_dict() for m in self.macros],
            "dynamic_includes": [d.to_dict() for d in self.dynamic_includes],
            "errors": [e.to_dict() for e in self.errors],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return format_report(self)


def _special_names(node: nodes.Macro | nodes.CallBlock) -> set[str]:
    """Find the special names (`caller`, `kwargs`, `varargs`) the body of
    a macro reads without declaring them itself.

    This mirrors :func:`jinja2.compiler.find_undeclared` but stops at
    nested macro / call-block boundaries so that a nested macro's use of
    a special name is not attributed to the enclosing macro.
    """

    class _Visitor(NodeVisitor):
        def __init__(self) -> None:
            self.names = {"caller", "kwargs", "varargs"}
            self.undeclared: set[str] = set()

        def visit_Name(self, n: nodes.Name) -> None:
            if n.ctx == "load" and n.name in self.names:
                self.undeclared.add(n.name)
            else:
                self.names.discard(n.name)

        def visit_Macro(self, n: nodes.Macro) -> None:
            # Nested macros keep their own scope.
            return None

        def visit_CallBlock(self, n: nodes.CallBlock) -> None:
            return None

    visitor = _Visitor()
    for child in node.body:
        visitor.visit(child)
    return visitor.undeclared


def _dynamic_template(node: nodes.Expr) -> str | None:
    """Return a human readable description of a non-literal include
    target, or `None` when the target is a literal string (or a tuple /
    list of literal strings)."""
    if isinstance(node, nodes.Const):
        if isinstance(node.value, str):
            return None
        # A tuple/list of consts is handled below; anything else is
        # dynamic.
        if not isinstance(node.value, (tuple, list)):
            return repr(node.value)
        items = node.value
    elif isinstance(node, (nodes.Tuple, nodes.List)):
        items = node.items
    else:
        return _describe(node)

    for item in items:
        if isinstance(item, nodes.Const) and isinstance(item.value, str):
            continue
        return _describe(item)
    return None


def _describe(node: nodes.Node) -> str:
    if isinstance(node, nodes.Name):
        return node.name
    if isinstance(node, nodes.Const):
        return repr(node.value)
    return type(node).__name__


class _Collector(NodeVisitor):
    """Walks the template AST and collects macro contracts in declaration
    order, tracking the enclosing macro scope."""

    def __init__(self) -> None:
        self.macros: list[MacroContract] = []
        self.dynamic_includes: list[DynamicInclude] = []
        self.errors: list[ContractError] = []
        self._scope: tuple[str, ...] = ()

    def visit_Macro(self, node: nodes.Macro) -> None:
        contract, error = build_macro_contract(node, self._scope)
        self.macros.append(contract)
        if error is not None:
            self.errors.append(error)

        previous = self._scope
        self._scope = previous + (node.name,)
        for child in node.body:
            self.visit(child)
        self._scope = previous

    def visit_Include(self, node: nodes.Include) -> None:
        dynamic = _dynamic_template(node.template)
        if dynamic is not None:
            self.dynamic_includes.append(DynamicInclude(node.lineno, dynamic))
        self.generic_visit(node)


def build_macro_contract(
    node: nodes.Macro, scope: tuple[str, ...] = ()
) -> tuple[MacroContract, ContractError | None]:
    """Build the :class:`MacroContract` for a single macro node.

    Returns the contract and an optional :class:`ContractError` when the
    macro violates a compiler rule (e.g. an explicit ``caller`` argument
    without a default).
    """
    explicit = {arg.name for arg in node.args}
    offset = len(node.args) - len(node.defaults)
    defaults: dict[str, nodes.Expr] = {}
    for index, default in enumerate(node.defaults):
        defaults[node.args[offset + index].name] = default

    undeclared = _special_names(node)

    accesses_caller = "caller" in undeclared
    accesses_kwargs = "kwargs" in undeclared and "kwargs" not in explicit
    accesses_varargs = "varargs" in undeclared and "varargs" not in explicit

    error: ContractError | None = None
    if "caller" in explicit and "caller" not in defaults:
        error = ContractError(
            node.lineno,
            'the special "caller" argument must be omitted or be given a'
            " default",
        )

    params: list[MacroParam] = []
    for arg in node.args:
        default = defaults.get(arg.name)
        params.append(
            MacroParam(
                name=arg.name,
                lineno=arg.lineno,
                default_lineno=default.lineno if default is not None else None,
            )
        )

    contract = MacroContract(
        name=node.name,
        lineno=node.lineno,
        scope=scope,
        params=params,
        receives_caller=accesses_caller or "caller" in explicit,
        accesses_kwargs=accesses_kwargs,
        accesses_varargs=accesses_varargs,
    )
    return contract, error


def macro_contracts(ast: nodes.Template) -> ContractReport:
    """Export macro contracts from a parsed template AST.

    The AST is only inspected, never executed.  Dynamic includes are
    reported as unknown and are not loaded.
    """
    collector = _Collector()
    collector.visit(ast)
    return ContractReport(
        macros=collector.macros,
        dynamic_includes=collector.dynamic_includes,
        errors=collector.errors,
    )


def export_macro_contracts(
    source: str,
    *,
    name: str | None = None,
    filename: str | None = None,
    environment: Environment | None = None,
) -> ContractReport:
    """Parse `source` read-only and export its macro contracts.

    :param source: the template source text.
    :param name: optional template name used in error messages.
    :param filename: optional filename used in error messages.
    :param environment: an optional :class:`~jinja2.Environment` to parse
        with.  A plain environment is used when omitted.

    Nothing is loaded or executed: default expressions are never
    evaluated and dynamic includes are only marked as unknown.
    """
    if environment is None:
        from .environment import Environment

        environment = Environment()

    try:
        ast = environment.parse(source, name=name, filename=filename)
    except TemplateSyntaxError as exc:  # pragma: no cover - exercised by tests
        return ContractReport(errors=[ContractError(exc.lineno, exc.message)])

    return macro_contracts(ast)


def format_report(report: ContractReport) -> str:
    """Render `report` as a human readable text report."""
    lines: list[str] = []
    if report.macros:
        lines.append(f"{len(report.macros)} macro(s):")
        for contract in report.macros:
            scope = ".".join(contract.scope) if contract.scope else "<top-level>"
            lines.append(
                f"  macro {contract.name}  [scope: {scope}]"
                f"  (line {contract.lineno})"
            )
            if contract.params:
                for param in contract.params:
                    if param.default_lineno is not None:
                        lines.append(
                            f"    - {param.name} (line {param.lineno}):"
                            f" default at line {param.default_lineno}"
                        )
                    else:
                        lines.append(f"    - {param.name} (line {param.lineno})")
            else:
                lines.append("    (no parameters)")
            flags: list[str] = []
            if contract.receives_caller:
                flags.append("receives caller")
            if contract.accepts_extra_keywords:
                flags.append("accepts extra keywords (kwargs)")
            if contract.accepts_extra_positionals:
                flags.append("accepts extra positionals (varargs)")
            if flags:
                lines.append(f"    flags: {', '.join(flags)}")
    else:
        lines.append("no macros found")

    if report.dynamic_includes:
        lines.append("")
        lines.append(
            f"{len(report.dynamic_includes)} dynamic include(s) (unknown):"
        )
        for dynamic in report.dynamic_includes:
            lines.append(
                f"  - include of {dynamic.template!r} at line {dynamic.lineno}"
            )

    if report.errors:
        lines.append("")
        lines.append(f"{len(report.errors)} error(s):")
        for error in report.errors:
            where = (
                f"line {error.lineno}"
                if error.lineno is not None
                else "unknown location"
            )
            lines.append(f"  - {where}: {error.message}")

    return "\n".join(lines)
