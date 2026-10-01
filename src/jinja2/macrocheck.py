"""Static macro contract checker.

This module analyzes templates *without executing them*.  It collects
macro definitions, static `{% import %}` / `{% from ... import %}`
aliases, and macro call sites from the parsed AST, then checks every
statically resolvable call against the callee's signature:

*   missing required arguments (parameters without a default)
*   duplicate arguments (positionally *and* by keyword, or a repeated
    keyword)
*   unexpected keyword arguments and too many positional arguments
    (unless the macro accesses `kwargs` / `varargs`)
*   `{% call %}` blocks passed to macros that do not accept a
    `caller`

Defaults, `caller`, `varargs` and `kwargs` are handled exactly
the way the compiler and :class:`~jinja2.runtime.Macro` handle them
at runtime.  Default expressions are only checked for existence, they
are never evaluated.  No filter, test, or user global function is ever
called; every place where the checker deliberately refuses to evaluate
something is counted as a *safety sentinel*.

Anything that cannot be decided statically -- dynamic template names,
dynamic calls, argument unpacking (`*args` / `**kwargs`), or names
shadowed by assignments -- is reported as an *unknown* instead of a
diagnostic.  Templates that import each other (and macros that call
each other) are analyzed exactly once, so analysis always terminates.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from dataclasses import field

from . import nodes
from .compiler import find_undeclared
from .exceptions import TemplateNotFound
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from .environment import Environment

#: Names that receive special treatment inside macro bodies, mirroring
#: the compiler and runtime behavior.
_SPECIAL_NAMES = ("caller", "kwargs", "varargs")


@dataclass
class Diagnostic:
    """A definite macro contract violation."""

    template: str | None
    lineno: int
    code: str
    message: str

    def __str__(self) -> str:
        name = self.template or "<template>"
        return f"{name}:{self.lineno}: {self.code}: {self.message}"


@dataclass
class Unknown:
    """A call or import that cannot be decided statically."""

    template: str | None
    lineno: int
    reason: str
    message: str

    def __str__(self) -> str:
        name = self.template or "<template>"
        return f"{name}:{self.lineno}: {self.reason}: {self.message}"


@dataclass
class MacroSignature:
    """Statically derived signature of a macro definition."""

    name: str
    template: str | None
    lineno: int
    params: list[str]
    has_default: set[str]
    accesses_caller: bool
    accesses_kwargs: bool
    accesses_varargs: bool

    @property
    def required(self) -> list[str]:
        """Parameters without a default, in declaration order."""
        return [p for p in self.params if p not in self.has_default]

    @property
    def accepts_caller(self) -> bool:
        """Whether an implicit `caller` keyword is consumed."""
        return self.accesses_caller or "caller" in self.params or self.accesses_kwargs


@dataclass
class CheckReport:
    """Result of a static macro contract check."""

    templates: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    unknowns: list[Unknown] = field(default_factory=list)
    sentinel_count: int = 0

    @property
    def ok(self) -> bool:
        """True when no definite contract violation was found."""
        return not self.diagnostics

    def format(self) -> str:
        """Render a human readable static report."""
        lines = ["== Macro contract report =="]
        lines.append(f"templates checked: {len(self.templates)}")
        for name in self.templates:
            lines.append(f"  - {name}")
        lines.append(f"diagnostics: {len(self.diagnostics)}")
        for diag in self.diagnostics:
            lines.append(f"  {diag}")
        lines.append(f"unknowns: {len(self.unknowns)}")
        for unknown in self.unknowns:
            lines.append(f"  {unknown}")
        lines.append(f"safety sentinels: {self.sentinel_count}")
        return "\n".join(lines)


class _TemplateInfo:
    """Static facts collected from one parsed template."""

    def __init__(self, name: str | None) -> None:
        self.name = name
        self.macros: dict[str, MacroSignature] = {}
        self.module_aliases: dict[str, str] = {}
        self.macro_aliases: dict[str, tuple[str, str]] = {}
        self.assigned: set[str] = set()
        self.shadowed: set[str] = set()
        self.calls: list[tuple[nodes.Call, bool]] = []
        self.imported_templates: set[str] = set()
        self.unknown_imports: list[tuple[int, str | None]] = []
        self.sentinels = 0


class _Collector(NodeVisitor):
    """Walk one template AST and collect static facts."""

    def __init__(self, info: _TemplateInfo) -> None:
        self.info = info
        self._callblock_calls: set[int] = set()

    def visit_Macro(self, node: nodes.Macro) -> None:
        params = [arg.name for arg in node.args]
        defaults = node.defaults
        if defaults:
            has_default = set(params[len(params) - len(defaults) :])
        else:
            has_default = set()
        undeclared = find_undeclared(node.body, _SPECIAL_NAMES)
        self.info.macros[node.name] = MacroSignature(
            name=node.name,
            template=self.info.name,
            lineno=node.lineno,
            params=params,
            has_default=has_default,
            accesses_caller="caller" in undeclared,
            accesses_kwargs="kwargs" in undeclared and "kwargs" not in params,
            accesses_varargs="varargs" in undeclared and "varargs" not in params,
        )
        # Each default expression is checked for existence only; the
        # checker never evaluates it.  Count one sentinel per default.
        self.info.sentinels += len(defaults)
        self.generic_visit(node)

    def visit_Import(self, node: nodes.Import) -> None:
        if isinstance(node.template, nodes.Const) and isinstance(
            node.template.value, str
        ):
            self.info.module_aliases[node.target] = node.template.value
            self.info.imported_templates.add(node.template.value)
        else:
            self.info.unknown_imports.append((node.lineno, node.target))
        self.generic_visit(node)

    def visit_FromImport(self, node: nodes.FromImport) -> None:
        if isinstance(node.template, nodes.Const) and isinstance(
            node.template.value, str
        ):
            template = node.template.value
            self.info.imported_templates.add(template)
            for item in node.names:
                if isinstance(item, tuple):
                    original, alias = item
                else:
                    original = alias = item
                self.info.macro_aliases[alias] = (template, original)
        else:
            self.info.unknown_imports.append((node.lineno, None))
        self.generic_visit(node)

    def visit_Name(self, node: nodes.Name) -> None:
        if node.ctx in ("store", "param"):
            self.info.assigned.add(node.name)
        self.generic_visit(node)

    def visit_CallBlock(self, node: nodes.CallBlock) -> None:
        self._callblock_calls.add(id(node.call))
        self.generic_visit(node)

    def visit_Call(self, node: nodes.Call) -> None:
        self.info.calls.append((node, id(node) in self._callblock_calls))
        self.generic_visit(node)

    def visit_Filter(self, node: nodes.Filter) -> None:
        # A filter the checker refuses to call.
        self.info.sentinels += 1
        self.generic_visit(node)

    def visit_Test(self, node: nodes.Test) -> None:
        # A test the checker refuses to call.
        self.info.sentinels += 1
        self.generic_visit(node)


class MacroContractChecker:
    """Check macro call contracts without executing templates.

    :param environment: the environment used to load and parse
        templates.  Only parsing is performed; nothing is rendered and
        no filter, test, or global function is invoked.
    """

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self._infos: dict[str, _TemplateInfo | None] = {}
        self._checked: set[str] = set()
        self._report = CheckReport()

    def check(self, names: str | t.Iterable[str]) -> CheckReport:
        """Statically check *names* and every template they statically
        import, transitively.  Returns a :class:`CheckReport`.
        """
        self._infos = {}
        self._checked = set()
        self._report = CheckReport()
        if isinstance(names, str):
            names = [names]
        for name in names:
            self._check_template(name)
        return self._report

    def _load_info(self, name: str) -> _TemplateInfo | None:
        if name in self._infos:
            return self._infos[name]
        self._infos[name] = None  # guard against import cycles
        if self.environment.loader is None:
            self._report.unknowns.append(
                Unknown(name, 0, "unresolvable-template", "environment has no loader")
            )
            return None
        try:
            source, _, _ = self.environment.loader.get_source(self.environment, name)
        except TemplateNotFound:
            self._report.unknowns.append(
                Unknown(
                    name, 0, "unresolvable-template", f"template {name!r} not found"
                )
            )
            return None
        ast = self.environment.parse(source)
        info = _TemplateInfo(name)
        _Collector(info).visit(ast)
        info.shadowed = info.assigned & (
            set(info.macros) | set(info.module_aliases) | set(info.macro_aliases)
        )
        self._infos[name] = info
        self._report.sentinel_count += info.sentinels
        for lineno, target in info.unknown_imports:
            detail = "import of a non-constant template name"
            if target:
                detail += f" as {target!r}"
            self._report.unknowns.append(
                Unknown(name, lineno, "dynamic-import", detail)
            )
        return info

    def _check_template(self, name: str) -> None:
        if name in self._checked:
            return
        self._checked.add(name)
        info = self._load_info(name)
        if info is None:
            return
        self._report.templates.append(name)
        for call, is_callblock in info.calls:
            self._check_call(info, call, is_callblock)
        for imported in sorted(info.imported_templates):
            self._check_template(imported)

    def _resolve(self, info: _TemplateInfo, call: nodes.Call) -> MacroSignature | None:
        """Resolve the callee of *call* to a macro signature.

        Records an unknown and returns `None` when the callee cannot
        be decided statically.  Returns `None` silently for calls
        that are clearly not macro calls.
        """
        callee = call.node
        if isinstance(callee, nodes.Name):
            name = callee.name
            if name in info.shadowed:
                self._unknown(
                    info, call, "shadowed-name", f"{name!r} is also assigned locally"
                )
                return None
            if name in info.macros:
                return info.macros[name]
            if name in info.macro_aliases:
                template, original = info.macro_aliases[name]
                return self._resolve_imported(info, call, template, original)
            # A call to something that is not a known macro: a context
            # variable or a user global function.  Never called; count
            # a safety sentinel and move on.
            self._report.sentinel_count += 1
            return None
        if isinstance(callee, nodes.Getattr):
            base = callee.node
            if (
                isinstance(base, nodes.Name)
                and base.name in info.module_aliases
                and base.name not in info.shadowed
            ):
                return self._resolve_imported(
                    info, call, info.module_aliases[base.name], callee.attr
                )
            self._unknown(info, call, "dynamic-call", "call of a dynamic attribute")
            return None
        self._unknown(
            info, call, "dynamic-call", "callee cannot be resolved statically"
        )
        return None

    def _resolve_imported(
        self, info: _TemplateInfo, call: nodes.Call, template: str, macro_name: str
    ) -> MacroSignature | None:
        other = self._load_info(template)
        if other is None:
            self._unknown(
                info, call, "unresolvable-template", f"template {template!r} not found"
            )
            return None
        if macro_name in other.macros:
            return other.macros[macro_name]
        if macro_name in other.assigned:
            self._unknown(
                info,
                call,
                "dynamic-call",
                f"{macro_name!r} is an assigned variable in {template!r}",
            )
            return None
        self._report.diagnostics.append(
            Diagnostic(
                info.name,
                call.lineno,
                "unknown-macro",
                f"template {template!r} has no macro named {macro_name!r}",
            )
        )
        return None

    def _unknown(
        self, info: _TemplateInfo, call: nodes.Call, reason: str, detail: str
    ) -> None:
        self._report.unknowns.append(Unknown(info.name, call.lineno, reason, detail))

    def _check_call(
        self, info: _TemplateInfo, call: nodes.Call, is_callblock: bool
    ) -> None:
        if call.dyn_args is not None or call.dyn_kwargs is not None:
            self._unknown(
                info, call, "argument-unpacking", "call uses *args or **kwargs"
            )
            return
        sig = self._resolve(info, call)
        if sig is None:
            return
        # Diagnostics that correspond to a runtime TypeError.  When one
        # is emitted the call fails before the body runs, so a cascading
        # missing-argument report for the same call would be noise.
        had_error = False
        params = sig.params
        positional = len(call.args)
        if positional > len(params) and not sig.accesses_varargs:
            had_error = True
            self._report.diagnostics.append(
                Diagnostic(
                    info.name,
                    call.lineno,
                    "too-many-positional",
                    f"macro {sig.name!r} takes not more than {len(params)}"
                    f" argument(s) but {positional} were given",
                )
            )
        filled = set(params[:positional])
        kw_provided: set[str] = set()
        seen: set[str] = set()
        for kw in call.kwargs:
            if kw.key in seen:
                had_error = True
                self._report.diagnostics.append(
                    Diagnostic(
                        info.name,
                        call.lineno,
                        "duplicate-keyword",
                        f"keyword argument {kw.key!r} repeated in call to"
                        f" macro {sig.name!r}",
                    )
                )
                continue
            seen.add(kw.key)
            if kw.key in filled:
                had_error = True
                self._report.diagnostics.append(
                    Diagnostic(
                        info.name,
                        call.lineno,
                        "duplicate-argument",
                        f"macro {sig.name!r} got multiple values for"
                        f" argument {kw.key!r}",
                    )
                )
            elif kw.key in params:
                kw_provided.add(kw.key)
            elif kw.key == "caller" and sig.accesses_caller:
                pass
            elif sig.accesses_kwargs:
                pass  # collected into the special `kwargs` variable
            else:
                had_error = True
                self._report.diagnostics.append(
                    Diagnostic(
                        info.name,
                        call.lineno,
                        "unexpected-keyword",
                        f"macro {sig.name!r} takes no keyword argument {kw.key!r}",
                    )
                )
        if is_callblock and not sig.accepts_caller:
            had_error = True
            self._report.diagnostics.append(
                Diagnostic(
                    info.name,
                    call.lineno,
                    "caller-not-accepted",
                    f"macro {sig.name!r} is called with a caller block but"
                    " does not use 'caller'",
                )
            )
        if had_error:
            return
        missing = [p for p in sig.required if p not in filled and p not in kw_provided]
        if missing:
            names = ", ".join(repr(p) for p in missing)
            self._report.diagnostics.append(
                Diagnostic(
                    info.name,
                    call.lineno,
                    "missing-argument",
                    f"macro {sig.name!r} is missing required argument(s): {names}",
                )
            )


def check_templates(
    environment: Environment, names: str | t.Iterable[str]
) -> CheckReport:
    """Convenience wrapper around :class:`MacroContractChecker`."""
    return MacroContractChecker(environment).check(names)
