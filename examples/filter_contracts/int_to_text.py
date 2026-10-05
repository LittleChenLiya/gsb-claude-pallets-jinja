"""Example: explicit static contracts for filters.

Declares a contract for an ``int_to_text`` filter (accepts integers,
produces text) and statically checks templates *without running any
filter*.  Run from the repository root with::

    python examples/filter_contracts/int_to_text.py

The targeted tests for this feature live in
``tests/test_filter_contracts.py``::

    python -m pytest tests/test_filter_contracts.py -v
"""

from jinja2 import Environment
from jinja2.filter_contracts import ContractRegistry
from jinja2.filter_contracts import FilterContract
from jinja2.filter_contracts import check_source

# Contract examples: pure data, kept in a registry that is separate
# from the environment's (and the global) filter functions.
contracts = ContractRegistry(
    [
        # int_to_text only accepts integers and produces text.
        FilterContract("int_to_text", accepts={"int"}, returns="str"),
        # shout only accepts text and produces text.
        FilterContract("shout", accepts={"str"}, returns="str"),
    ]
)

env = Environment()

templates = {
    "ok literal": "{{ 42|int_to_text }}",
    "ok chain": "{{ 42|int_to_text|shout }}",
    "bad literal": "{{ 'hello'|int_to_text }}",
    "bad chain": "{{ 42|int_to_text|int_to_text }}",
    "unknown variable": "{{ value|int_to_text }}",
    "macro parameter": "{% macro m(x) %}{{ x|int_to_text }}{% endmacro %}",
}

for title, source in templates.items():
    print(f"--- {title}: {source}")
    diagnostics = check_source(env, source, contracts)
    if not diagnostics:
        print("    (no diagnostics)")
    for diag in diagnostics:
        print(f"    {diag}")
