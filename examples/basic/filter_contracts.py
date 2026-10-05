"""Static filter contracts: check filter arguments without running filters.

This example registers an ``int_to_text`` filter that only accepts integers
and produces a string, then uses :class:`~jinja2.contracts.StaticContractChecker`
to verify templates *statically* (the filters are never executed).

Run the targeted tests for this feature with::

    pytest tests/test_contracts.py -v

The checker is entirely opt-in: normal compilation and rendering do not
enable it, and registering a contract never modifies ``env.filters``.
"""

from jinja2 import Environment
from jinja2.contracts import Category
from jinja2.contracts import FilterContract
from jinja2.contracts import StaticContractChecker


def int_to_text(value: int) -> str:
    """A filter that only accepts integers."""
    return str(value)


def main() -> None:
    env = Environment()
    # The actual filter is registered as usual.
    env.filters["int_to_text"] = int_to_text

    # The contract declares the allowed input category and the known output
    # category.  This is separate from env.filters and never modifies it.
    contracts = [
        FilterContract("int_to_text", input=Category.INT, output=Category.STR),
    ]
    checker = StaticContractChecker(contracts)

    templates = {
        "good": "{{ 5|int_to_text }}",
        "bad": '{{ "abc"|int_to_text }}',
        "unknown": "{{ x|int_to_text }}",
        "chain": "{{ 5|int_to_text|upper }}",
    }

    for name, source in templates.items():
        diagnostics = checker.check(env.parse(source))
        print(f"--- {name} ---")
        print(source)
        if not diagnostics:
            print("  OK")
        for diag in diagnostics:
            print(f"  [{diag.kind}] line {diag.lineno}: {diag.message}")

    # Rendering still works normally; the checker is not involved.
    tmpl = env.from_string("{{ 5|int_to_text }}")
    print("rendered:", tmpl.render())


if __name__ == "__main__":
    main()
