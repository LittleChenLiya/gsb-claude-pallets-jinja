"""Static macro contract checker.

This module analyzes templates *without executing them* and reports
macro call sites that violate the macro's declared signature, following
the same rules the compiler and runtime apply:

-   Local macro definitions and static `{% import %}` /
    `{% from ... import ... %}` aliases are collected per template.
-   Every statically resolvable call to a macro is checked for missing
    required arguments, too many positional arguments, duplicate
    keyword arguments, arguments passed both positionally and by
    keyword, and keyword arguments that do not exist in the signature.
-   `caller`, `varargs`, and `kwargs` are handled exactly like
    the compiler handles them: a macro whose body accesses `varargs`
    accepts extra positional arguments, one that accesses `kwargs`
    accepts extra keyword arguments, and one that accesses `caller`
    may be invoked through a `{% call %}` block.
-   Default values are only checked for *existence*. They are never
    evaluated during static analysis.
-   Anything that cannot be decided statically -- dynamic template
    names in imports, dynamic attribute access, argument unpacking
    (`*args` / `**kwargs`), `{% call %}` blocks, and names
    shadowed by local assignments -- is reported as *unknown* instead
    of as an error.

The checker never renders a template and never calls a filter, test,
or user global. Macros that call each other (directly, mutually, or
across templates that import each other) do not cause recursion: each
template's AST is walked exactly once and calls are validated against
collected signatures rather than expanded.

Every issue carries the template name and the line number of the call
or import it was reported for.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from dataclasses import field

from . import nodes
from .compiler import find_undeclared
from .exceptions import TemplateSyntaxError
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from .environment import Environment


@dataclass
class MacroIssue:
    """A single finding of the macro contract checker."""

    template: str
    """Name of the template the issue was found in."""

    line: int
    """Line number of the call or import the issue refers to."""

    message: str
    """Human readable description of the issue."""

    def __str__(self) -> str:
        return f"{self.template}:{self.line}: {self.message}"


@dataclass
class MacroReport:
    """Result of :func:`check_templates`.

    `errors` are definite contract violations. `unknowns` are call
    or import sites that cannot be decided statically and are listed
    separately so they can be reviewed by hand.
    """

    errors: list[MacroIssue] = field(default_factory=list)
    unknowns: list[MacroIssue] = field(default_factory=list)
    templates_checked: int = 0
    macros_found: int = 0
    calls_checked: int = 0

    @property
    def ok(self) -> bool:
        """True if no definite errors were found."""
        return not self.errors

    @property
    def sentinels(self) -> dict[str, int]:
        """Safety sentinel counts proving what the analysis covered."""
        return {
            "templates": self.templates_checked,
            "macros": self.macros_found,
            "calls": self.calls_checked,
            "errors": len(self.errors),
            "unknowns": len(self.unknowns),
        }

    def render_text(self) -> str:
        """Render the report as a plain text static analysis report."""
        lines = ["macro contract report", "====================="]

        lines.append("errors:")
        if self.errors:
            lines.extend(f"  {issue}" for issue in self.errors)
        else:
            lines.append("  (none)")

        lines.append("unknowns:")
        if self.unknowns:
            lines.extend(f"  {issue}" for issue in self.unknowns)
        else:
            lines.append("  (none)")

        sentinels = " ".join(f"{key}={value}" for key, value in self.sentinels.items())
        lines.append(f"sentinels: {sentinels}")
        return "\n".join(lines)


class _MacroSignature:
    """Static view of a macro definition, mirroring the compiler rules."""

    def __init__(self, node: nodes.Macro) -> None:
        self.name = node.name
        self.params = [arg.name for arg in node.args]
        # Defaults align with the tail of the parameter list. Only their
        # existence matters here, the default expressions are never
        # evaluated.
        self.num_defaults = len(node.defaults)
        undeclared = find_undeclared(node.body, ("caller", "kwargs", "varargs"))
        self.uses_caller = "caller" in undeclared
        self.catch_kwargs = "kwargs" in undeclared and "kwargs" not in self.params
        self.catch_varargs = "varargs" in undeclared and "varargs" not in self.params

    @property
    def required(self) -> list[str]:
        return self.params[: len(self.params) - self.num_defaults]


class _TemplateInfo:
    """Collected static facts about one template."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.macros: dict[str, _MacroSignature] = {}
        # alias -> (template name, macro name or None for module aliases)
        self.aliases: dict[str, tuple[str, str | None]] = {}
        # Names assigned anywhere in the template (set / for / with).
        self.assigned: set[str] = set()
        self.calls: list[nodes.Call] = []
        self.call_blocks: list[nodes.CallBlock] = []
        self.dynamic_imports: list[nodes.Import | nodes.FromImport] = []
        self.imports: list[nodes.Import | nodes.FromImport] = []


class _Collector(NodeVisitor):
    """Single AST walk collecting macros, imports, assignments, calls."""

    def __init__(self, info: _TemplateInfo) -> None:
        self.info = info

    def visit_Macro(self, node: nodes.Macro) -> None:
        self.info.macros[node.name] = _MacroSignature(node)
        self.generic_visit(node)

    def visit_Import(self, node: nodes.Import) -> None:
        self.info.imports.append(node)

        if isinstance(node.template, nodes.Const) and isinstance(
            node.template.value, str
        ):
            self.info.aliases[node.target] = (node.template.value, None)
        else:
            self.info.dynamic_imports.append(node)

        self.generic_visit(node)

    def visit_FromImport(self, node: nodes.FromImport) -> None:
        self.info.imports.append(node)

        if isinstance(node.template, nodes.Const) and isinstance(
            node.template.value, str
        ):
            for item in node.names:
                if isinstance(item, tuple):
                    imported, alias = item
                else:
                    imported = alias = item

                self.info.aliases[alias] = (node.template.value, imported)
        else:
            self.info.dynamic_imports.append(node)

        self.generic_visit(node)

    def visit_Assign(self, node: nodes.Assign) -> None:
        self._collect_targets(node.target)
        self.generic_visit(node)

    def visit_AssignBlock(self, node: nodes.AssignBlock) -> None:
        self._collect_targets(node.target)
        self.generic_visit(node)

    def visit_For(self, node: nodes.For) -> None:
        self._collect_targets(node.target)
        self.generic_visit(node)

    def visit_With(self, node: nodes.With) -> None:
        for target in node.targets:
            self._collect_targets(target)

        self.generic_visit(node)

    def _collect_targets(self, node: nodes.Node) -> None:
        if isinstance(node, nodes.Name):
            self.info.assigned.add(node.name)
        elif isinstance(node, nodes.Tuple):
            for item in node.items:
                self._collect_targets(item)

    def visit_CallBlock(self, node: nodes.CallBlock) -> None:
        self.info.call_blocks.append(node)
        # Visit everything except the inner call node itself. The call
        # receives the implicit `caller` argument and is reported as
        # unknown instead of being checked like a regular call.
        self.visit(node.call.node)

        for arg in node.call.args:
            self.visit(arg)

        for kwarg in node.call.kwargs:
            self.visit(kwarg.value)

        for default in node.defaults:
            self.visit(default)

        for child in node.body:
            self.visit(child)

    def visit_Call(self, node: nodes.Call) -> None:
        self.info.calls.append(node)
        self.generic_visit(node)


class _Checker:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.report = MacroReport()
        self.templates: dict[str, _TemplateInfo] = {}
        self.pending: list[str] = []

    def analyze(self, name: str) -> _TemplateInfo | None:
        """Collect static facts for one template. Each template is
        analyzed at most once, so templates that import each other
        terminate.
        """
        if name in self.templates:
            return self.templates[name]

        info = _TemplateInfo(name)
        # Register before recursing into imports to guard against
        # import cycles.
        self.templates[name] = info

        if self.environment.loader is None:
            self.report.unknowns.append(
                MacroIssue(name, 0, "template cannot be loaded: no loader configured")
            )
            return info

        try:
            source = self.environment.loader.get_source(self.environment, name)[0]
        except Exception as e:
            self.report.unknowns.append(
                MacroIssue(name, 0, f"template cannot be loaded: {e}")
            )
            return info

        try:
            ast = self.environment.parse(source)
        except TemplateSyntaxError as e:
            self.report.errors.append(
                MacroIssue(name, e.lineno, f"syntax error: {e.message}")
            )
            return info

        self.report.templates_checked += 1
        _Collector(info).visit(ast)
        self.report.macros_found += len(info.macros)

        for node in info.dynamic_imports:
            self.report.unknowns.append(
                MacroIssue(
                    name,
                    node.lineno,
                    "import with a dynamic template name cannot be resolved statically",
                )
            )

        # Follow static imports so signatures of imported macros are
        # available. Cycles terminate because of the registry above.
        for template, _imported in info.aliases.values():
            self.analyze(template)

        self.pending.append(name)
        return info

    def finish(self) -> MacroReport:
        # Check calls only after every reachable template was analyzed,
        # so that mutual imports see each other's signatures.
        for name in self.pending:
            info = self.templates[name]

            for call_block in info.call_blocks:
                self.report.unknowns.append(
                    MacroIssue(
                        name,
                        call_block.lineno,
                        "call block passes an implicit 'caller' argument"
                        " and cannot be checked statically",
                    )
                )

            for call in info.calls:
                self._check_call(info, call)

        return self.report

    def _resolve_alias(
        self, info: _TemplateInfo, alias: str, line: int
    ) -> _MacroSignature | None:
        template, imported = info.aliases[alias]
        target = self.templates.get(template)

        if target is None:
            self.report.unknowns.append(
                MacroIssue(
                    info.name,
                    line,
                    f"template {template!r} imported as {alias!r} was not analyzed",
                )
            )
            return None

        if imported is None:
            return None

        signature = target.macros.get(imported)

        if signature is None:
            self.report.errors.append(
                MacroIssue(
                    info.name,
                    line,
                    f"template {template!r} has no macro {imported!r}",
                )
            )

        return signature

    def _check_call(self, info: _TemplateInfo, call: nodes.Call) -> None:
        callee = call.node
        signature: _MacroSignature | None = None
        description: str | None = None

        if isinstance(callee, nodes.Name):
            name = callee.name

            if name in info.assigned and (name in info.macros or name in info.aliases):
                self.report.unknowns.append(
                    MacroIssue(
                        info.name,
                        call.lineno,
                        f"{name!r} is shadowed by a local assignment and"
                        " cannot be resolved statically",
                    )
                )
                return

            if name in info.macros:
                signature = info.macros[name]
                description = f"macro {name!r}"
            elif name in info.aliases:
                signature = self._resolve_alias(info, name, call.lineno)
                description = f"macro {name!r}"
            else:
                # A filter, test, user global, or unrelated value. Not a
                # macro call, and never invoked by this checker.
                return
        elif isinstance(callee, nodes.Getattr) and isinstance(callee.node, nodes.Name):
            base = callee.node.name

            if base not in info.aliases:
                return

            if base in info.assigned:
                self.report.unknowns.append(
                    MacroIssue(
                        info.name,
                        call.lineno,
                        f"{base!r} is shadowed by a local assignment and"
                        " cannot be resolved statically",
                    )
                )
                return

            template, imported = info.aliases[base]

            if imported is not None:
                # A from-import alias used as a namespace object.
                return

            target = self.templates.get(template)

            if target is None:
                self.report.unknowns.append(
                    MacroIssue(
                        info.name,
                        call.lineno,
                        f"template {template!r} imported as {base!r} was not analyzed",
                    )
                )
                return

            signature = target.macros.get(callee.attr)
            description = f"macro {callee.attr!r} from {template!r}"

            if signature is None:
                self.report.errors.append(
                    MacroIssue(
                        info.name,
                        call.lineno,
                        f"template {template!r} has no macro {callee.attr!r}",
                    )
                )
                return
        elif (
            isinstance(callee, nodes.Getitem)
            and isinstance(callee.node, nodes.Name)
            and callee.node.name in info.aliases
        ):
            self.report.unknowns.append(
                MacroIssue(
                    info.name,
                    call.lineno,
                    "dynamic attribute access on an imported template"
                    " cannot be resolved statically",
                )
            )
            return
        else:
            return

        if signature is None:
            return

        if call.dyn_args is not None or call.dyn_kwargs is not None:
            self.report.unknowns.append(
                MacroIssue(
                    info.name,
                    call.lineno,
                    f"call to {description} uses argument unpacking and"
                    " cannot be checked statically",
                )
            )
            return

        self.report.calls_checked += 1
        self._check_arguments(info, call, signature, description)

    def _check_arguments(
        self,
        info: _TemplateInfo,
        call: nodes.Call,
        signature: _MacroSignature,
        description: str,
    ) -> None:
        params = signature.params
        num_positional = len(call.args)

        def error(message: str) -> None:
            self.report.errors.append(MacroIssue(info.name, call.lineno, message))

        seen_keywords: set[str] = set()

        for kwarg in call.kwargs:
            key = kwarg.key

            if key in seen_keywords:
                error(
                    f"call to {description} passes keyword argument"
                    f" {key!r} more than once"
                )
                continue

            seen_keywords.add(key)

            if key in params:
                if params.index(key) < num_positional:
                    error(
                        f"call to {description} passes argument {key!r}"
                        " both positionally and by keyword"
                    )
            elif key == "caller" and signature.uses_caller:
                pass
            elif not signature.catch_kwargs:
                error(f"call to {description} passes unknown keyword argument {key!r}")

        provided = set(params[:num_positional]) | (seen_keywords & set(params))

        for name in signature.required:
            if name not in provided:
                error(f"call to {description} is missing required argument {name!r}")

        if num_positional > len(params) and not signature.catch_varargs:
            error(
                f"call to {description} passes {num_positional} positional"
                f" arguments but the macro takes at most {len(params)}"
            )


def check_templates(
    environment: Environment, names: t.Iterable[str] | None = None
) -> MacroReport:
    """Statically check macro call contracts in the given templates.

    :param environment: The environment used to parse the templates.
        Its loader is used to fetch template sources. Nothing is
        rendered and no filter, test, or global is called.
    :param names: Template names to check. If not given, all templates
        the loader lists are checked. Statically imported templates are
        analyzed as well, even if not listed.

    Returns a :class:`MacroReport` with definite `errors` and
    separately listed `unknowns`.
    """
    checker = _Checker(environment)

    if names is None:
        names = environment.list_templates()

    for name in names:
        checker.analyze(name)

    return checker.finish()
