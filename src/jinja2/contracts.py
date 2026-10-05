"""Static contracts for filters.

This module provides an opt-in, static (non-executing) check that verifies
the arguments passed to filters in a template match explicitly declared
input/output categories.  It never calls the filters it inspects and never
modifies the global ``filters`` mapping of an environment.

The check is entirely separate from normal template compilation and
rendering: :class:`StaticContractChecker` must be instantiated and run
explicitly.  Compiling or rendering a template does not enable it.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from . import nodes
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from .environment import Environment

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class Category:
    """Named categories used to describe filter inputs and outputs.

    A category is a plain string.  The constants on this class are provided
    for convenience and readability; any string is a valid category.
    """

    UNKNOWN = "unknown"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    BYTES = "bytes"
    LIST = "list"
    DICT = "dict"
    TUPLE = "tuple"
    NONE = "none"
    MARKUP = "markup"


#: Categories that describe numeric values.
_NUMERIC_CATEGORIES = frozenset({Category.INT, Category.FLOAT, Category.BOOL})

#: Categories that describe textual values.
_TEXT_CATEGORIES = frozenset({Category.STR, Category.MARKUP})


# ---------------------------------------------------------------------------
# Contracts and diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterContract:
    """An explicit static contract for a single filter.

    :param name: the name of the filter, as used in a template.
    :param input: a category or an iterable of categories that the filter's
        piped-in value is allowed to have.
    :param output: the category of the value the filter produces.  This is
        used to propagate categories through a chain of filters.
    """

    name: str
    input: frozenset[str]
    output: str

    def __init__(
        self,
        name: str,
        input: str | t.Iterable[str] = Category.UNKNOWN,
        output: str = Category.UNKNOWN,
    ) -> None:
        if isinstance(input, str):
            input_categories = frozenset({input})
        else:
            input_categories = frozenset(input)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "input", input_categories)
        object.__setattr__(self, "output", output)


@dataclass(frozen=True)
class ContractDiagnostic:
    """A single finding produced by :class:`StaticContractChecker`.

    :ivar kind: ``"type_error"`` if a deterministic value does not match the
        contract, or ``"pending"`` if the value cannot be determined
        statically and needs to be judged at runtime.
    :ivar message: a human readable description.
    :ivar lineno: the line in the template where the offending filter call
        appears.
    :ivar filter_name: the name of the filter.
    :ivar template_name: the name of the template, if known.
    """

    kind: str
    message: str
    lineno: int
    filter_name: str
    template_name: str | None = None

    @property
    def is_error(self) -> bool:
        """``True`` if this diagnostic is a definite type error."""
        return self.kind == "type_error"

    @property
    def is_pending(self) -> bool:
        """``True`` if this diagnostic only marks a value as undetermined."""
        return self.kind == "pending"


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------


def _const_category(value: t.Any) -> str:
    """Return the category of a constant Python value."""
    if value is None:
        return Category.NONE
    if isinstance(value, bool):
        return Category.BOOL
    if isinstance(value, int):
        return Category.INT
    if isinstance(value, float):
        return Category.FLOAT
    if isinstance(value, str):
        return Category.STR
    if isinstance(value, bytes):
        return Category.BYTES
    if isinstance(value, list):
        return Category.LIST
    if isinstance(value, tuple):
        return Category.TUPLE
    if isinstance(value, dict):
        return Category.DICT
    return Category.UNKNOWN


def _merge_numeric(left: str, right: str, operator: str) -> str:
    """Infer the category of a binary arithmetic expression."""
    if left not in _NUMERIC_CATEGORIES or right not in _NUMERIC_CATEGORIES:
        return Category.UNKNOWN
    # In Python 3, true division always produces a float.
    if operator == "/":
        return Category.FLOAT
    if Category.FLOAT in (left, right):
        return Category.FLOAT
    return Category.INT


def _merge_text(left: str, right: str, operator: str) -> str:
    """Infer the category of a binary text expression."""
    if left in _TEXT_CATEGORIES and right in _TEXT_CATEGORIES:
        return Category.STR
    return Category.UNKNOWN


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


class StaticContractChecker(NodeVisitor):
    """Check filter calls against explicit static contracts.

    The checker walks a parsed template AST and, for every filter call whose
    filter has a registered :class:`FilterContract`, compares the inferred
    category of the piped-in value with the contract's allowed input
    categories.  It never executes the filter and never modifies
    ``env.filters``.

    Deterministic values (literals, lists, dicts, tuples) are checked
    directly.  Values that cannot be determined statically (variables,
    calls, attribute lookups, ...) are marked as ``pending`` instead of being
    treated as valid or invalid.  Chained filters propagate the output
    category of the previous filter as the input category of the next one,
    but only when the previous output is actually known.

    Macro bodies and included templates are traversed as well so that an
    unknown value inside a macro or include is still marked ``pending``
    rather than being silently accepted.
    """

    def __init__(
        self,
        contracts: t.Iterable[FilterContract] | t.Mapping[str, FilterContract] = (),
        environment: "Environment | None" = None,
    ) -> None:
        self._contracts: dict[str, FilterContract] = {}
        self._environment = environment
        self._visited_includes: set[str] = set()
        self.diagnostics: list[ContractDiagnostic] = []

        if isinstance(contracts, t.Mapping):
            for contract in contracts.values():
                self.register(contract)
        else:
            for contract in contracts:
                self.register(contract)

    def register(self, contract: FilterContract) -> None:
        """Register a :class:`FilterContract`.

        Registering a contract does not add or modify anything in the
        environment's ``filters`` mapping; contracts are kept private to
        this checker.
        """
        self._contracts[contract.name] = contract

    @property
    def contracts(self) -> t.Mapping[str, FilterContract]:
        """The currently registered contracts."""
        return self._contracts

    def check(
        self,
        node: nodes.Node,
        template_name: str | None = None,
    ) -> list[ContractDiagnostic]:
        """Check a parsed template (or any AST node) and return diagnostics.

        :param node: a :class:`~jinja2.nodes.Template` or any other node.
        :param template_name: an optional name used in the diagnostics.
        """
        self.diagnostics = []
        self._visited_includes = set()
        self._template_name = template_name
        self.visit(node)
        return list(self.diagnostics)

    # -- expression inference -------------------------------------------------

    def infer_category(self, node: nodes.Expr) -> str:
        """Infer the category of an expression.

        As a side effect, every filter call encountered while inferring the
        category is checked against its contract.
        """
        method = getattr(self, f"_infer_{type(node).__name__}", None)
        if method is not None:
            return method(node)
        return Category.UNKNOWN

    def _infer_Const(self, node: nodes.Const) -> str:
        return _const_category(node.value)

    def _infer_List(self, node: nodes.List) -> str:
        return Category.LIST

    def _infer_Tuple(self, node: nodes.Tuple) -> str:
        return Category.TUPLE

    def _infer_Dict(self, node: nodes.Dict) -> str:
        return Category.DICT

    def _infer_Name(self, node: nodes.Name) -> str:
        return Category.UNKNOWN

    def _infer_Call(self, node: nodes.Call) -> str:
        return Category.UNKNOWN

    def _infer_Getitem(self, node: nodes.Getitem) -> str:
        return Category.UNKNOWN

    def _infer_Getattr(self, node: nodes.Getattr) -> str:
        return Category.UNKNOWN

    def _infer_ImportedName(self, node: nodes.ImportedName) -> str:
        return Category.UNKNOWN

    def _infer_InternalName(self, node: nodes.InternalName) -> str:
        return Category.UNKNOWN

    def _infer_NSRef(self, node: nodes.NSRef) -> str:
        return Category.UNKNOWN

    def _infer_EnvironmentAttribute(self, node: nodes.EnvironmentAttribute) -> str:
        return Category.UNKNOWN

    def _infer_ExtensionAttribute(self, node: nodes.ExtensionAttribute) -> str:
        return Category.UNKNOWN

    def _infer_CondExpr(self, node: nodes.CondExpr) -> str:
        cat1 = self.infer_category(node.expr1)
        if node.expr2 is None:
            return Category.UNKNOWN
        cat2 = self.infer_category(node.expr2)
        if cat1 == cat2 and cat1 != Category.UNKNOWN:
            return cat1
        return Category.UNKNOWN

    def _infer_Concat(self, node: nodes.Concat) -> str:
        # The ``~`` operator always concatenates to a string.
        for item in node.items:
            self.infer_category(item)
        return Category.STR

    def _infer_Compare(self, node: nodes.Compare) -> str:
        self.infer_category(node.expr)
        for op in node.ops:
            self.infer_category(op.expr)
        return Category.BOOL

    def _infer_And(self, node: nodes.And) -> str:
        self.infer_category(node.left)
        self.infer_category(node.right)
        return Category.BOOL

    def _infer_Or(self, node: nodes.Or) -> str:
        self.infer_category(node.left)
        self.infer_category(node.right)
        return Category.BOOL

    def _infer_Not(self, node: nodes.Not) -> str:
        self.infer_category(node.node)
        return Category.BOOL

    def _infer_Neg(self, node: nodes.Neg) -> str:
        cat = self.infer_category(node.node)
        if cat in _NUMERIC_CATEGORIES:
            return cat
        return Category.UNKNOWN

    def _infer_Pos(self, node: nodes.Pos) -> str:
        cat = self.infer_category(node.node)
        if cat in _NUMERIC_CATEGORIES:
            return cat
        return Category.UNKNOWN

    def _infer_Add(self, node: nodes.Add) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        numeric = _merge_numeric(left, right, "+")
        if numeric != Category.UNKNOWN:
            return numeric
        return _merge_text(left, right, "+")

    def _infer_Sub(self, node: nodes.Sub) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "-")

    def _infer_Mul(self, node: nodes.Mul) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "*")

    def _infer_Div(self, node: nodes.Div) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "/")

    def _infer_FloorDiv(self, node: nodes.FloorDiv) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "//")

    def _infer_Mod(self, node: nodes.Mod) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "%")

    def _infer_Pow(self, node: nodes.Pow) -> str:
        left = self.infer_category(node.left)
        right = self.infer_category(node.right)
        return _merge_numeric(left, right, "**")

    def _infer_MarkSafe(self, node: nodes.MarkSafe) -> str:
        self.infer_category(node.expr)
        return Category.MARKUP

    def _infer_MarkSafeIfAutoescape(self, node: nodes.MarkSafeIfAutoescape) -> str:
        self.infer_category(node.expr)
        return Category.UNKNOWN

    def _infer_Filter(self, node: nodes.Filter) -> str:
        return self._check_filter(node)

    def _infer_Test(self, node: nodes.Test) -> str:
        # Tests are not covered by filter contracts; just traverse.
        self.infer_category(node.node)
        for arg in node.args:
            self.infer_category(arg)
        for kwarg in node.kwargs:
            self.infer_category(kwarg.value)
        return Category.UNKNOWN

    # -- filter checking ------------------------------------------------------

    def _check_filter(
        self, node: nodes.Filter, input_category: str | None = None
    ) -> str:
        """Check a filter node against its contract.

        Returns the known output category, or :data:`Category.UNKNOWN` if the
        output cannot be propagated.
        """
        if input_category is None:
            if node.node is None:
                # A filter block: the rendered body is passed as a string.
                input_category = Category.STR
            else:
                input_category = self.infer_category(node.node)

        # Also check the filter's own arguments for pending values.
        for arg in node.args:
            self.infer_category(arg)
        for kwarg in node.kwargs:
            self.infer_category(kwarg.value)

        contract = self._contracts.get(node.name)
        if contract is None:
            return Category.UNKNOWN

        if input_category == Category.UNKNOWN:
            self._pending(node, contract)
            return Category.UNKNOWN

        if input_category in contract.input:
            return contract.output

        self._type_error(node, contract, input_category)
        return Category.UNKNOWN

    def _pending(self, node: nodes.Filter, contract: FilterContract) -> None:
        self.diagnostics.append(
            ContractDiagnostic(
                kind="pending",
                message=(
                    f"cannot statically determine the input category of filter"
                    f" {contract.name!r}; the value needs to be judged at runtime"
                ),
                lineno=node.lineno,
                filter_name=contract.name,
                template_name=self._template_name,
            )
        )

    def _type_error(
        self, node: nodes.Filter, contract: FilterContract, actual: str
    ) -> None:
        allowed = ", ".join(sorted(contract.input))
        self.diagnostics.append(
            ContractDiagnostic(
                kind="type_error",
                message=(
                    f"filter {contract.name!r} expects input of category"
                    f" {allowed!r}, got {actual!r}"
                ),
                lineno=node.lineno,
                filter_name=contract.name,
                template_name=self._template_name,
            )
        )

    # -- statement traversal --------------------------------------------------

    def visit(self, node: nodes.Node) -> t.Any:
        if isinstance(node, nodes.Expr):
            return self.infer_category(node)
        return super().visit(node)

    def visit_FilterBlock(self, node: nodes.FilterBlock) -> None:
        # Traverse the body so filters inside it are checked.  The filter
        # child itself is visited by generic_visit and, because its node is
        # None, is checked against the rendered body (a string).
        self.generic_visit(node)

    def visit_Include(self, node: nodes.Include) -> None:
        self.generic_visit(node)
        self._check_include(node)

    def _check_include(self, node: nodes.Include) -> None:
        if self._environment is None or self._environment.loader is None:
            return
        template_node = node.template
        if not isinstance(template_node, nodes.Const):
            return
        name = template_node.value
        if not isinstance(name, str):
            return
        if name in self._visited_includes:
            return
        self._visited_includes.add(name)
        try:
            source, _filename, _uptodate = self._environment.loader.get_source(
                self._environment, name
            )
        except Exception:
            # The included template cannot be resolved; leave its unknowns
            # as pending rather than guessing.
            return
        try:
            included = self._environment.parse(source, name=name)
        except Exception:
            return
        previous_name = self._template_name
        self._template_name = name
        self.visit(included)
        self._template_name = previous_name


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def check_filter_contracts(
    node: nodes.Node,
    contracts: t.Iterable[FilterContract] | t.Mapping[str, FilterContract],
    environment: "Environment | None" = None,
    template_name: str | None = None,
) -> list[ContractDiagnostic]:
    """Check a parsed template against filter contracts.

    This is a convenience wrapper around :class:`StaticContractChecker`.

    :param node: a :class:`~jinja2.nodes.Template` or any other node.
    :param contracts: the contracts to check against.
    :param environment: an optional environment used to resolve includes.
    :param template_name: an optional name used in the diagnostics.
    :return: a list of :class:`ContractDiagnostic` objects.
    """
    checker = StaticContractChecker(contracts, environment=environment)
    return checker.check(node, template_name=template_name)
