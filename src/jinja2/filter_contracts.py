"""Explicit static contracts for filters.

This module implements an **opt-in** static checker that validates how
filters are called in a template *without running any filter function*.

A :class:`FilterContract` declares, for one filter name:

* which input *categories* are allowed (``accepts``), and
* which category the filter produces (``returns``), so that filter
  chains such as ``{{ value|int_to_text|shout }}`` can be checked by
  feeding the known output category of one step into the next.

The checker walks the parsed template AST and classifies every filter
input expression:

* **known** -- literals and literal collections have a statically known
  category (``int``, ``str``, ``sequence``, ...).  A known category that
  is not accepted by the contract produces an ``error`` diagnostic.
* **unknown** -- variables, macro parameters, attribute access, calls,
  and anything else that cannot be proven statically stay ``unknown``.
  An unknown input is *never* silently accepted: it produces a
  ``pending`` diagnostic (undecided) so that macro bodies and templates
  pulled in through ``{% include %}`` cannot misjudge unknown values as
  legal.

Every diagnostic carries the template name and line number.  Checking is
never enabled implicitly: normal compilation (``Environment.compile`` /
``from_string``) and rendering do not consult contracts at all, and
registering a contract never touches ``Environment.filters`` or
``jinja2.filters.FILTERS`` -- contracts live in a separate
:class:`ContractRegistry` and cannot modify the global filter functions.

Example
-------

.. code-block:: python

    from jinja2 import Environment
    from jinja2.filter_contracts import (
        FilterContract, ContractRegistry, check_source,
    )

    contracts = ContractRegistry([
        FilterContract("int_to_text", accepts={"int"}, returns="str"),
        FilterContract("shout", accepts={"str"}, returns="str"),
    ])

    env = Environment()
    diags = check_source(env, "{{ 42|int_to_text|shout }}", contracts)
    assert diags == []

Run the targeted tests for this feature with::

    python -m pytest tests/test_filter_contracts.py -v
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from . import nodes
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from .environment import Environment

#: Category of expressions whose type cannot be determined statically.
UNKNOWN: t.Final[str] = "unknown"

#: Severity of a diagnostic for a definite contract violation.
ERROR: t.Final[str] = "error"

#: Severity of a diagnostic for a call that cannot be decided statically.
PENDING: t.Final[str] = "pending"

#: Well-known input categories.
KNOWN_CATEGORIES: t.Final[frozenset] = frozenset(
    {"none", "bool", "int", "float", "str", "sequence", "mapping"}
)

#: Category groups that may be used in ``FilterContract.accepts`` in
#: addition to concrete categories and the wildcard ``"any"``.
CATEGORY_GROUPS: t.Final[t.Mapping] = {
    "number": frozenset({"int", "float"}),
}


def category_of_value(value):
    """Return the static category of a literal Python value."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "sequence"
    if isinstance(value, dict):
        return "mapping"
    return UNKNOWN


@dataclass(frozen=True)
class FilterContract:
    """A static contract for one filter.

    :param name: filter name as used in templates.
    :param accepts: iterable of accepted input categories.  Entries may
        be concrete categories (``"int"``, ``"str"``, ...), a group from
        :data:`CATEGORY_GROUPS` (``"number"``), or the wildcard
        ``"any"``.
    :param returns: the output category, used to check filter chains.
        Defaults to ``"unknown"``.

    A contract is pure data.  Registering it in a
    :class:`ContractRegistry` does not install, wrap, or modify any
    filter function.
    """

    name: str
    accepts: frozenset
    returns: str = UNKNOWN

    def __init__(self, name, accepts, returns=UNKNOWN):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "accepts", frozenset(accepts))
        object.__setattr__(self, "returns", returns)

    def accepts_category(self, category):
        """Check *category* against this contract.

        Returns ``True`` if the category is accepted, ``False`` if it is
        definitely rejected, and ``None`` if the category is
        :data:`UNKNOWN` and therefore cannot be decided statically.
        """
        if category == UNKNOWN:
            return None
        if "any" in self.accepts or category in self.accepts:
            return True
        for entry in self.accepts:
            group = CATEGORY_GROUPS.get(entry)
            if group is not None and category in group:
                return True
        return False

    def describe_accepts(self):
        """Human readable description of the accepted categories."""
        return " or ".join(sorted(self.accepts)) or "nothing"


@dataclass
class FilterDiagnostic:
    """One static finding about a filter call.

    Always carries the template position (``filename`` and ``lineno``)
    of the filter expression it refers to.
    """

    severity: str  # ERROR or PENDING
    message: str
    filter_name: str
    lineno: int
    filename: t.Optional[str] = None

    def __str__(self):
        location = f"{self.filename or '<template>'}:{self.lineno}"
        return f"{location}: {self.severity}: {self.message}"


class ContractRegistry:
    """An isolated collection of :class:`FilterContract` objects.

    The registry is deliberately separate from
    ``jinja2.filters.FILTERS`` and ``Environment.filters``:
    registering a contract here cannot modify the global (or any
    environment's) filter functions.
    """

    def __init__(self, contracts=()):
        self._contracts = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract):
        """Register *contract* and return it (usable as a decorator)."""
        if not isinstance(contract, FilterContract):
            raise TypeError(f"expected FilterContract, got {type(contract)!r}")
        self._contracts[contract.name] = contract
        return contract

    def get(self, name):
        """Return the contract for *name*, or ``None`` if unregistered."""
        return self._contracts.get(name)

    def __contains__(self, name):
        return name in self._contracts

    def __len__(self):
        return len(self._contracts)

    def names(self):
        """Names of all registered contracts, sorted."""
        return tuple(sorted(self._contracts))


class _StaticFilterChecker(NodeVisitor):
    """Walk a parsed template and check filter calls against contracts.

    No filter function is ever called; only the AST is inspected.
    """

    def __init__(self, contracts, filename=None, env=None,
                 follow_includes=False, _seen=None):
        self.contracts = contracts
        self.filename = filename
        self.env = env
        self.follow_includes = follow_includes
        self._seen = _seen if _seen is not None else {filename}
        self.diagnostics = []

    # -- category inference -------------------------------------------------

    def infer(self, node):
        """Statically infer the category of an expression.

        Anything that cannot be proven (names, macro parameters,
        attribute/item access, calls, ...) stays :data:`UNKNOWN`.
        For a filter node, the *declared* output category of its
        contract is used, which is what makes chain checking possible.
        """
        if isinstance(node, nodes.Const):
            return category_of_value(node.value)
        if isinstance(node, (nodes.List, nodes.Tuple)):
            return "sequence"
        if isinstance(node, nodes.Dict):
            return "mapping"
        if isinstance(node, nodes.Filter):
            contract = self.contracts.get(node.name)
            return contract.returns if contract is not None else UNKNOWN
        return UNKNOWN

    # -- diagnostics ----------------------------------------------------------

    def _check_filter(self, filt, input_category=None):
        contract = self.contracts.get(filt.name)
        if contract is None:
            return  # no contract registered: nothing to check
        if input_category is None:
            input_category = self.infer(filt.node)
        verdict = contract.accepts_category(input_category)
        if verdict is False:
            self.diagnostics.append(
                FilterDiagnostic(
                    severity=ERROR,
                    message=(
                        f"filter {filt.name!r} accepts"
                        f" {contract.describe_accepts()}, but the input is"
                        f" statically known to be {input_category!r}"
                    ),
                    filter_name=filt.name,
                    lineno=filt.lineno,
                    filename=self.filename,
                )
            )
        elif verdict is None:
            self.diagnostics.append(
                FilterDiagnostic(
                    severity=PENDING,
                    message=(
                        f"cannot statically determine the input category of"
                        f" filter {filt.name!r} (accepts"
                        f" {contract.describe_accepts()}); left undecided"
                    ),
                    filter_name=filt.name,
                    lineno=filt.lineno,
                    filename=self.filename,
                )
            )

    # -- node visitors --------------------------------------------------------

    def visit_Filter(self, node):
        # Check inner chain links first so diagnostics follow the chain
        # order; each link uses the previous link's declared output.
        if node.node is not None:
            self.visit(node.node)
        self._check_filter(node)
        for arg in node.args:
            self.visit(arg)
        for kwarg in node.kwargs:
            self.visit(kwarg)

    def visit_FilterBlock(self, node):
        # A filter block feeds rendered text (always ``str``) to the filter.
        self._check_filter(node.filter, input_category="str")
        for child in node.body:
            self.visit(child)
        for arg in node.filter.args:
            self.visit(arg)

    def visit_Include(self, node):
        if not (self.follow_includes and self.env is not None):
            return
        if not isinstance(node.template, nodes.Const):
            return  # dynamic include target: stays unknown, nothing checked
        name = node.template.value
        if not isinstance(name, str) or name in self._seen:
            return
        self._seen.add(name)
        if self.env.loader is None:
            return
        try:
            source, _, _ = self.env.loader.get_source(self.env, name)
        except Exception:
            return  # unresolvable include: nothing can be decided
        # The included template is checked with an all-unknown context:
        # its variables produce ``pending`` diagnostics, never silent
        # acceptance.
        child = _StaticFilterChecker(
            self.contracts,
            filename=name,
            env=self.env,
            follow_includes=True,
            _seen=self._seen,
        )
        child.visit(self.env.parse(source))
        self.diagnostics.extend(child.diagnostics)


def check_source(env, source, contracts, name=None, follow_includes=False):
    """Statically check filter usage in *source* against *contracts*.

    The template is parsed but never compiled or rendered, and no filter
    function is called.  Returns a list of :class:`FilterDiagnostic`;
    each entry has severity ``error`` (definite contract violation) or
    ``pending`` (statically undecidable input).

    This is strictly opt-in: normal compilation and rendering never call
    this function.
    """
    if not isinstance(contracts, ContractRegistry):
        contracts = ContractRegistry(contracts)
    checker = _StaticFilterChecker(
        contracts, filename=name, env=env, follow_includes=follow_includes
    )
    checker.visit(env.parse(source))
    return checker.diagnostics


def check_template(env, template_name, contracts, follow_includes=False):
    """Like :func:`check_source`, but load *template_name* from the
    environment's loader first.  Diagnostics carry the template name as
    their filename."""
    if env.loader is None:
        raise TypeError("the environment has no loader configured")
    source, _, _ = env.loader.get_source(env, template_name)
    return check_source(
        env, source, contracts, name=template_name,
        follow_includes=follow_includes,
    )
