"""Read-only extraction of macro contracts from template source.

This module helps documentation and analysis tools inspect a template
*without* compiling or rendering it.  It parses the source into the
regular Jinja AST and reports, for every macro declared in the template:

-   the macro name (and its nesting scope),
-   the declared parameters in order,
-   the source position of each default expression (pointing at the
    expression itself, not at the whole ``{% macro %}`` tag),
-   whether the macro receives the special ``caller``, ``**kwargs`` or
    ``*varargs`` parameters.

Nothing is ever executed: default expressions are only located and
copied from the source, never evaluated.  Dynamic ``{% include %}``
tags are flagged as *unknown*; the referenced templates are neither
loaded nor inferred.

The rules for the special ``caller`` / ``kwargs`` / ``varargs``
parameters intentionally mirror the ones used by the compiler in
:meth:`jinja2.compiler.CodeGenerator.macro_body`.

The module can also be run as a script to print a report::

    python -m jinja2.macro_contract path/to/template.html
"""

from __future__ import annotations

import argparse
import sys
import typing as t
from dataclasses import dataclass
from pathlib import Path

from . import nodes
from .compiler import find_undeclared
from .environment import Environment
from .visitor import NodeVisitor

__all__ = [
    "IncludeInfo",
    "MacroContract",
    "MacroParameter",
    "TemplateContract",
    "extract_macro_contracts",
    "format_report",
    "main",
]

#: Names that receive special treatment in macro signatures, mirroring
#: the compiler.
_SPECIAL_PARAMS = ("caller", "kwargs", "varargs")


@dataclass(frozen=True)
class MacroParameter:
    """One declared macro parameter, in declaration order."""

    #: The parameter name.
    name: str
    #: Whether a default expression was declared for this parameter.
    has_default: bool
    #: Line number (1-based) of the default expression itself, not of
    #: the whole ``{% macro %}`` tag.  `None` if there is no default.
    default_lineno: int | None = None
    #: Column offset (0-based) of the default expression within its
    #: line.  `None` if there is no default.
    default_col_offset: int | None = None
    #: The exact source text of the default expression, copied from the
    #: template source.  The expression is never evaluated.  `None` if
    #: there is no default.
    default_source: str | None = None


@dataclass(frozen=True)
class MacroContract:
    """The contract of one macro (or anonymous ``{% call %}`` block)."""

    #: The macro name, or `None` for anonymous ``{% call %}`` blocks.
    name: str | None
    #: Line number of the ``{% macro %}`` / ``{% call %}`` tag.
    lineno: int
    #: Names of the enclosing macros, outermost first.  Nested macros
    #: and macros that share a name keep their own scope and are
    #: reported as separate entries.
    scope: tuple[str, ...]
    #: The declared parameters, in declaration order.
    parameters: tuple[MacroParameter, ...]
    #: Whether the macro receives the special ``caller`` parameter.
    accepts_caller: bool
    #: Whether the macro reads ``kwargs`` and therefore accepts extra
    #: keyword arguments beyond the declared parameters.
    accepts_kwargs: bool
    #: Whether the macro reads ``varargs`` and therefore accepts extra
    #: positional arguments beyond the declared parameters.
    accepts_varargs: bool

    @property
    def qualified_name(self) -> str:
        """The macro name prefixed by its enclosing scope, e.g.
        ``"outer.inner"``.  Anonymous call blocks show as ``<call>``.
        """
        label = self.name if self.name is not None else "<call>"
        return ".".join((*self.scope, label))


@dataclass(frozen=True)
class IncludeInfo:
    """One ``{% include %}`` reference found in the template."""

    #: Line number of the ``{% include %}`` tag.
    lineno: int
    #: Whether the included template is dynamic (computed at runtime).
    #: Dynamic includes are only flagged as unknown; they are never
    #: loaded or inferred.
    dynamic: bool
    #: The static template name, or `None` for dynamic includes.
    source: str | None


@dataclass(frozen=True)
class TemplateContract:
    """The extracted contract of one template."""

    #: All macros declared in the template, in declaration order.
    #: Nested and same-name macros appear as separate entries.
    macros: tuple[MacroContract, ...] = ()
    #: All ``{% include %}`` references, in declaration order.
    includes: tuple[IncludeInfo, ...] = ()

    @property
    def has_dynamic_includes(self) -> bool:
        """Whether the template contains at least one dynamic
        ``{% include %}`` whose target is unknown.
        """
        return any(include.dynamic for include in self.includes)


class _PositionedToken(t.NamedTuple):
    """A raw lexer token located in the (newline-normalized) source."""

    lineno: int
    type: str
    value: str
    start: int
    end: int


def _positioned_tokens(
    environment: Environment, source: str
) -> tuple[str, list[_PositionedToken]]:
    """Lex *source* with the environment's lexer and return the
    newline-normalized source together with the raw tokens, each
    carrying its exact offset in that source.

    The raw token values are exact slices of the source, so every
    token can be located by scanning forward.  Data tokens are the
    only exception: whitespace stripping (``lstrip_blocks`` or
    ``{%-``) may truncate their end, so they are only used to advance
    the scan position, never matched against.
    """
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[_PositionedToken] = []
    pos = 0

    for lineno, token_type, value in environment.lexer.tokeniter(
        normalized, None, None
    ):
        if not value:
            continue

        if token_type == "data":
            if normalized.startswith(value, pos):
                pos += len(value)

            continue

        start = normalized.find(value, pos)

        if start == -1:  # pragma: no cover - lexer and source stay in sync
            raise RuntimeError(f"could not locate token {value!r} in template source")

        pos = start + len(value)
        tokens.append(_PositionedToken(lineno, token_type, value, start, pos))

    return normalized, tokens


def _extract_signatures(
    environment: Environment, source: str
) -> list[dict[str, tuple[int, int, str]]]:
    """Lex *source* and collect, for every ``{% macro %}`` and
    ``{% call %}`` signature in document order, a mapping of parameter
    name to ``(lineno, col_offset, source_text)`` of its default
    expression.

    The lexer (like the parser) skips comments and ``{% raw %}``
    blocks, so the returned list lines up with the macro nodes found
    in the AST.
    """
    normalized, tokens = _positioned_tokens(environment, source)
    # Whitespace between tokens is irrelevant for the signature
    # structure; positions come from the remaining tokens.
    tokens = [token for token in tokens if token.type != "whitespace"]
    signatures: list[dict[str, tuple[int, int, str]]] = []
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]

        if (
            token.type == "block_begin"
            and i + 1 < n
            and tokens[i + 1].type == "name"
            and tokens[i + 1].value in ("macro", "call")
        ):
            kind = tokens[i + 1].value
            i += 2

            if kind == "macro":
                i += 1  # skip the macro name

            signature: dict[str, tuple[int, int, str]] = {}

            if i < n and tokens[i].value == "(":
                i += 1

                while i < n and tokens[i].value != ")":
                    if tokens[i].type == "name":
                        param = tokens[i].value
                        i += 1

                        if i < n and tokens[i].value == "=":
                            i += 1
                            start = i
                            depth = 0

                            while i < n:
                                current = tokens[i]

                                if depth == 0 and current.value in (",", ")"):
                                    break

                                if current.value in ("(", "[", "{"):
                                    depth += 1
                                elif current.value in (")", "]", "}"):
                                    depth -= 1

                                i += 1

                            first_token = tokens[start]
                            last_token = tokens[i - 1]
                            line_start = (
                                normalized.rfind("\n", 0, first_token.start) + 1
                            )
                            signature[param] = (
                                first_token.lineno,
                                first_token.start - line_start,
                                normalized[first_token.start : last_token.end],
                            )
                            continue

                        continue

                    i += 1

            signatures.append(signature)

        i += 1

    return signatures


class _ContractVisitor(NodeVisitor):
    """Collects macro contracts and include references from the AST in
    declaration order, tracking the macro nesting scope.
    """

    def __init__(self, signatures: list[dict[str, tuple[int, int, str]]]) -> None:
        self.macros: list[MacroContract] = []
        self.includes: list[IncludeInfo] = []
        self._scope: list[str] = []
        self._signatures = iter(signatures)

    def visit_Macro(self, node: nodes.Macro) -> None:
        self._record(node, node.name)
        self._scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    def visit_CallBlock(self, node: nodes.CallBlock) -> None:
        self._record(node, None)
        self._scope.append("<call>")
        for child in node.body:
            self.visit(child)
        self._scope.pop()

    def visit_Include(self, node: nodes.Include) -> None:
        template = node.template

        if isinstance(template, nodes.Const) and isinstance(template.value, str):
            self.includes.append(
                IncludeInfo(node.lineno, dynamic=False, source=template.value)
            )
        elif isinstance(template, (nodes.Tuple, nodes.List)):
            for item in template.items:
                if isinstance(item, nodes.Const) and isinstance(item.value, str):
                    self.includes.append(
                        IncludeInfo(node.lineno, dynamic=False, source=item.value)
                    )
                else:
                    # A dynamic entry in the list: flag as unknown, do
                    # not load or infer it.
                    self.includes.append(
                        IncludeInfo(node.lineno, dynamic=True, source=None)
                    )
        else:
            # Dynamic include: flag as unknown, do not load or infer it.
            self.includes.append(IncludeInfo(node.lineno, dynamic=True, source=None))

    def _record(self, node: nodes.Macro | nodes.CallBlock, name: str | None) -> None:
        try:
            signature = next(self._signatures)
        except StopIteration:  # pragma: no cover - lexer/parser stay in sync
            signature = {}

        parameters: list[MacroParameter] = []
        first_default = len(node.args) - len(node.defaults)

        for idx, arg in enumerate(node.args):
            if idx >= first_default:
                default_node = node.defaults[idx - first_default]
                token_info = signature.get(arg.name)
                parameters.append(
                    MacroParameter(
                        name=arg.name,
                        has_default=True,
                        # The position points at the default expression
                        # itself, not at the whole macro tag.
                        default_lineno=default_node.lineno,
                        default_col_offset=(
                            token_info[1] if token_info is not None else None
                        ),
                        default_source=(
                            token_info[2] if token_info is not None else None
                        ),
                    )
                )
            else:
                parameters.append(MacroParameter(name=arg.name, has_default=False))

        # Mirror the compiler's macro parameter rules: a macro receives
        # the special caller/kwargs/varargs parameters when the name is
        # used undeclared in the body, unless it is an explicitly
        # declared parameter.
        skip_special = {
            arg.name for arg in node.args if arg.name in ("kwargs", "varargs")
        }
        undeclared = find_undeclared(node.body, _SPECIAL_PARAMS)

        self.macros.append(
            MacroContract(
                name=name,
                lineno=node.lineno,
                scope=tuple(self._scope),
                parameters=tuple(parameters),
                accepts_caller="caller" in undeclared,
                accepts_kwargs="kwargs" in undeclared and "kwargs" not in skip_special,
                accepts_varargs="varargs" in undeclared
                and "varargs" not in skip_special,
            )
        )


def extract_macro_contracts(environment: Environment, source: str) -> TemplateContract:
    """Parse *source* with *environment* and return the template's
    :class:`TemplateContract` without compiling or rendering anything.

    Default expressions are never evaluated; only their source position
    and source text are reported.  Macros are returned in declaration
    order, including nested and same-name macros, which keep their own
    scope.  Dynamic ``{% include %}`` tags are flagged as unknown
    (:attr:`IncludeInfo.dynamic`) and are never loaded.

    Raises :exc:`jinja2.TemplateSyntaxError` if the template cannot be
    parsed.

    :param environment: The environment used for parsing, so that
        custom tags and extensions are honored.
    :param source: The template source to inspect.
    """
    ast = environment.parse(source)
    signatures = _extract_signatures(environment, source)
    visitor = _ContractVisitor(signatures)
    visitor.visit(ast)
    return TemplateContract(
        macros=tuple(visitor.macros), includes=tuple(visitor.includes)
    )


def format_report(contract: TemplateContract, name: str | None = None) -> str:
    """Format a :class:`TemplateContract` as a human-readable report."""
    lines = ["macro contract report" + (f" for {name}" if name else "")]

    if not contract.macros:
        lines.append("  no macros declared")

    for macro in contract.macros:
        lines.append(f"  macro {macro.qualified_name} (line {macro.lineno})")

        if not macro.parameters:
            lines.append("    no parameters")

        for param in macro.parameters:
            if param.has_default:
                source = (
                    param.default_source
                    if param.default_source is not None
                    else "<expression>"
                )
                lines.append(
                    f"    param {param.name} = {source}"
                    f" (default at line {param.default_lineno},"
                    f" col {param.default_col_offset})"
                )
            else:
                lines.append(f"    param {param.name}")

        lines.append(f"    accepts caller: {'yes' if macro.accepts_caller else 'no'}")
        lines.append(
            "    accepts extra keywords (**kwargs):"
            f" {'yes' if macro.accepts_kwargs else 'no'}"
        )
        lines.append(
            "    accepts extra positional (*varargs):"
            f" {'yes' if macro.accepts_varargs else 'no'}"
        )

    if contract.includes:
        lines.append("  includes:")

        for include in contract.includes:
            if include.dynamic:
                lines.append(f"    line {include.lineno}: dynamic include (unknown)")
            else:
                lines.append(f"    line {include.lineno}: {include.source!r}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: print the macro contract report for a
    template file.

    Usage: ``python -m jinja2.macro_contract path/to/template.html``
    """
    parser = argparse.ArgumentParser(
        prog="python -m jinja2.macro_contract",
        description=(
            "Parse a Jinja template read-only and print the contract of"
            " every macro it declares. Nothing is compiled or executed."
        ),
    )
    parser.add_argument("template", help="path to the template file")
    parser.add_argument("--encoding", default="utf-8", help="template file encoding")
    args = parser.parse_args(argv)

    source = Path(args.template).read_text(encoding=args.encoding)

    try:
        contract = extract_macro_contracts(Environment(), source)
    except Exception as e:
        # Includes TemplateSyntaxError for broken templates.
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(format_report(contract, name=args.template))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
