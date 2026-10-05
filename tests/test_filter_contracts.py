"""Targeted tests for the opt-in static filter contract checker.

Run only these tests with::

    python -m pytest tests/test_filter_contracts.py -v
"""

import pytest

import jinja2.filters
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.filter_contracts import ERROR
from jinja2.filter_contracts import PENDING
from jinja2.filter_contracts import ContractRegistry
from jinja2.filter_contracts import FilterContract
from jinja2.filter_contracts import check_source
from jinja2.filter_contracts import check_template

# --- contract examples -------------------------------------------------------
# int_to_text: only accepts integers, produces text.
INT_TO_TEXT = FilterContract("int_to_text", accepts={"int"}, returns="str")
# shout: only accepts text, produces text (used for chain checks).
SHOUT = FilterContract("shout", accepts={"str"}, returns="str")


@pytest.fixture()
def contracts():
    return ContractRegistry([INT_TO_TEXT, SHOUT])


@pytest.fixture()
def env():
    return Environment()


def errors(diags):
    return [d for d in diags if d.severity == ERROR]


def pendings(diags):
    return [d for d in diags if d.severity == PENDING]


def test_literal_int_accepted(env, contracts):
    assert check_source(env, "{{ 42|int_to_text }}", contracts) == []


def test_literal_string_rejected(env, contracts):
    diags = check_source(env, "{{ 'hello'|int_to_text }}", contracts)
    assert len(diags) == 1
    (diag,) = diags
    assert diag.severity == ERROR
    assert diag.filter_name == "int_to_text"
    assert diag.lineno == 1
    assert "'str'" in diag.message
    assert "int" in diag.message
    # position is rendered into the string form
    assert "<template>:1" in str(diag)


def test_unknown_variable_is_pending_not_error(env, contracts):
    diags = check_source(env, "{{ value|int_to_text }}", contracts)
    assert errors(diags) == []
    assert len(pendings(diags)) == 1
    diag = pendings(diags)[0]
    assert diag.filter_name == "int_to_text"
    assert "undecided" in diag.message


def test_filters_are_never_executed(env, contracts):
    calls = []

    def int_to_text(value):
        calls.append(value)
        return str(value)

    env.filters["int_to_text"] = int_to_text
    diags = check_source(
        env,
        "{{ 42|int_to_text }} {{ 'x'|int_to_text }} {{ v|int_to_text }}",
        contracts,
    )
    assert calls == []  # filter count stays 0: checking runs no filters
    assert len(errors(diags)) == 1
    assert len(pendings(diags)) == 1


def test_chain_output_inference(env, contracts):
    # int -> int_to_text -> str -> shout -> str: whole chain is legal.
    assert check_source(env, "{{ 42|int_to_text|shout }}", contracts) == []
    # int_to_text yields str, so a second int_to_text must fail on the
    # *known* output category of the previous link.
    diags = check_source(env, "{{ 42|int_to_text|int_to_text }}", contracts)
    assert len(diags) == 1
    assert diags[0].severity == ERROR
    assert "'str'" in diags[0].message
    # shout yields str, so feeding it into int_to_text fails too.
    diags = check_source(env, "{{ 'a'|shout|int_to_text }}", contracts)
    assert [d.severity for d in diags] == [ERROR]
    # unknown head of a chain: the first link is pending, the second link
    # still sees the declared output category of the first.
    diags = check_source(env, "{{ value|int_to_text|shout }}", contracts)
    assert [d.severity for d in diags] == [PENDING]


def test_macro_parameter_is_pending_not_legal(env, contracts):
    source = "{% macro show(x) %}{{ x|int_to_text }}{% endmacro %}"
    diags = check_source(env, source, contracts)
    assert errors(diags) == []
    assert len(pendings(diags)) == 1
    assert pendings(diags)[0].filter_name == "int_to_text"


def test_macro_literal_call_argument_stays_unknown(env, contracts):
    # Calling a macro with a literal does not make the parameter known
    # inside the macro body: the scope boundary keeps it unknown, and
    # unknown is reported as pending, never silently accepted.
    source = (
        "{% macro show(x) %}{{ x|int_to_text }}{% endmacro %}"
        "{{ show(42) }}"
    )
    diags = check_source(env, source, contracts)
    assert errors(diags) == []
    assert len(pendings(diags)) == 1


def test_include_scope_unknown_is_pending(contracts):
    loader = DictLoader(
        {
            "parent": "{% include 'child' %}",
            "child": "{{ child_var|int_to_text }}",
        }
    )
    env = Environment(loader=loader)
    diags = check_template(env, "parent", contracts, follow_includes=True)
    assert errors(diags) == []
    (diag,) = pendings(diags)
    assert diag.filename == "child"
    assert diag.lineno == 1
    assert "child:1" in str(diag)


def test_include_literal_violation_reported_with_position(contracts):
    loader = DictLoader(
        {
            "parent": "line\n{% include 'child' %}",
            "child": "one\ntwo\n{{ 'text'|int_to_text }}",
        }
    )
    env = Environment(loader=loader)
    diags = check_template(env, "parent", contracts, follow_includes=True)
    (diag,) = errors(diags)
    assert diag.filename == "child"
    assert diag.lineno == 3


def test_include_not_followed_by_default(contracts):
    loader = DictLoader(
        {"parent": "{% include 'child' %}", "child": "{{ 'x'|int_to_text }}"}
    )
    env = Environment(loader=loader)
    assert check_template(env, "parent", contracts) == []


def test_filter_block_input_is_str(env, contracts):
    src = "{% filter int_to_text %}hi{% endfilter %}"
    diags = check_source(env, src, contracts)
    assert [d.severity for d in diags] == [ERROR]


def test_unregistered_filter_is_ignored(env, contracts):
    assert check_source(env, "{{ 'x'|no_such_contract }}", contracts) == []


def test_contracts_cannot_modify_global_filters(env, contracts):
    defaults_before = dict(jinja2.filters.FILTERS)
    env_filters_before = dict(env.filters)
    registry = ContractRegistry()
    registry.register(FilterContract("brand_new_filter", accepts={"int"}))
    registry.register(INT_TO_TEXT)
    check_source(env, "{{ 1|brand_new_filter }}", registry)
    assert jinja2.filters.FILTERS == defaults_before
    assert env.filters == env_filters_before
    assert "brand_new_filter" not in jinja2.filters.FILTERS
    assert "brand_new_filter" not in env.filters


def test_normal_compile_and_render_do_not_check(env, contracts):
    # A template that violates the contract renders without any
    # diagnostics: the checker is strictly opt-in.
    env.filters["int_to_text"] = lambda v: str(v)
    tmpl = env.from_string("{{ 'not an int'|int_to_text }}")
    assert tmpl.render() == "not an int"
    # compiling directly is equally unaffected
    env.compile("{{ 'not an int'|int_to_text }}")


def test_number_group_and_any_wildcard():
    registry = ContractRegistry(
        [
            FilterContract("num", accepts={"number"}),
            FilterContract("anything", accepts={"any"}),
        ]
    )
    env = Environment()
    assert check_source(env, "{{ 1.5|num }} {{ 2|num }}", registry) == []
    diags = check_source(env, "{{ 's'|num }}", registry)
    assert [d.severity for d in diags] == [ERROR]
    diags = check_source(env, "{{ 's'|anything }} {{ 1|anything }}", registry)
    assert diags == []


def test_diagnostic_carries_template_name(env, contracts):
    diags = check_source(
        env, "{{ 'x'|int_to_text }}", contracts, name="page.html"
    )
    assert diags[0].filename == "page.html"
    assert str(diags[0]).startswith("page.html:1: error:")


def test_registry_rejects_non_contract():
    registry = ContractRegistry()
    with pytest.raises(TypeError):
        registry.register("not a contract")
