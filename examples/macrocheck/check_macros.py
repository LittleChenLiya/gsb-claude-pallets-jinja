"""Static macro contract check example.

Runs :class:`jinja2.macrocheck.MacroContractChecker` over a small set
of templates -- two library templates that import macros from each
other, a valid page, a page with a broken call, and a page full of
constructs that cannot be decided statically -- and prints a static
report including the safety sentinel count.  Nothing is rendered and
no filter, test, or global function is called.

Run with:  python examples/macrocheck/check_macros.py
Exits with status 1 if any definite contract violation is found.
"""

import sys

from jinja2 import Environment
from jinja2.loaders import DictLoader
from jinja2.macrocheck import MacroContractChecker

TEMPLATES = {
    "lib_a.html": "{% macro greet(name, punct='!') %}"
    "Hello, {{ name }}{{ punct }}{% endmacro %}\n"
    "{% macro full_name(first, last) %}{{ first + ' ' + last }}{% endmacro %}\n"
    "{% from 'lib_b.html' import farewell %}\n"
    "{% macro greet_and_farewell(name) %}"
    "{{ greet(name) }} / {{ farewell(name) }}{% endmacro %}\n",
    "lib_b.html": "{% import 'lib_a.html' as lib_a %}\n"
    "{% macro farewell(name) %}Bye, {{ name }}{% endmacro %}\n"
    "{% macro wave(target) %}{{ lib_a.greet(target, '?') }}{% endmacro %}\n",
    "good.html": "{% from 'lib_a.html' import greet, full_name %}\n"
    "{% import 'lib_b.html' as lib_b %}\n"
    "{{ greet('Ada') }}\n"
    "{{ greet('Ada', punct='?') }}\n"
    "{{ full_name('Ada', 'Lovelace') }}\n"
    "{{ lib_b.wave('Ada') }}\n",
    "bad_missing.html": "{% from 'lib_a.html' import full_name %}\n"
    "{{ full_name('Ada') }}\n",
    "unknowns.html": "{% import tpl_name as dyn %}\n"
    "{% macro local(a) %}{{ a }}{% endmacro %}\n"
    "{% set local = 'shadowed' %}\n"
    "{{ local(1) }}\n"
    "{% macro packed(a, b) %}{{ a }}{{ b }}{% endmacro %}\n"
    "{{ packed(*args) }}\n"
    "{% call box.wrap() %}body{% endcall %}\n",
}

PAGES = ["good.html", "bad_missing.html", "unknowns.html"]


def main() -> int:
    env = Environment(loader=DictLoader(TEMPLATES))
    checker = MacroContractChecker(env)
    failed = False
    for page in PAGES:
        report = checker.check(page)
        print(report.format())
        print()
        if not report.ok:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
