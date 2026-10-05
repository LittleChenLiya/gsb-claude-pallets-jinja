"""Tests for the static filter contract checker.

Run just these tests with::

    pytest tests/test_contracts.py -v

The checker is entirely opt-in: normal compilation and rendering never
enable it, and registering a contract never modifies ``env.filters``.
"""

import pytest

from jinja2 import Environment
from jinja2.contracts import Category
from jinja2.contracts import FilterContract
from jinja2.contracts import StaticContractChecker
from jinja2.contracts import check_filter_contracts
from jinja2.loaders import DictLoader


@pytest.fixture
def env():
    return Environment()


@pytest.fixture
def int_to_text_contract():
    # int_to_text only accepts integers and produces a string.
    return FilterContract("int_to_text", input=Category.INT, output=Category.STR)


@pytest.fixture
def upper_contract():
    return FilterContract("upper", input=Category.STR, output=Category.STR)


def check(env, contracts, source, template_name=None):
    checker = StaticContractChecker(contracts, environment=env)
    return checker.check(env.parse(source), template_name=template_name)


class TestFilterContract:
    def test_contract_input_normalized_to_set(self):
        c = FilterContract("f", input=Category.INT)
        assert c.input == frozenset({Category.INT})

    def test_contract_accepts_multiple_inputs(self):
        c = FilterContract("f", input=[Category.INT, Category.FLOAT])
        assert c.input == frozenset({Category.INT, Category.FLOAT})

    def test_contract_defaults_to_unknown(self):
        c = FilterContract("f")
        assert c.input == frozenset({Category.UNKNOWN})
        assert c.output == Category.UNKNOWN


class TestStaticContractChecker:
    def test_literal_string_call_is_type_error(self, env, int_to_text_contract):
        diags = check(env, [int_to_text_contract], '{{ "abc"|int_to_text }}')
        assert len(diags) == 1
        assert diags[0].is_error
        assert not diags[0].is_pending
        assert diags[0].filter_name == "int_to_text"
        assert diags[0].lineno == 1

    def test_literal_int_call_is_ok(self, env, int_to_text_contract):
        diags = check(env, [int_to_text_contract], "{{ 5|int_to_text }}")
        assert diags == []

    def test_unknown_variable_is_pending(self, env, int_to_text_contract):
        diags = check(env, [int_to_text_contract], "{{ x|int_to_text }}")
        assert len(diags) == 1
        assert diags[0].is_pending
        assert not diags[0].is_error
        assert diags[0].filter_name == "int_to_text"

    def test_multiple_calls_same_line(self, env, int_to_text_contract):
        diags = check(
            env,
            [int_to_text_contract],
            '{{ 5|int_to_text }}{{ x|int_to_text }}{{ "s"|int_to_text }}',
        )
        kinds = [d.kind for d in diags]
        assert kinds == ["pending", "type_error"]

    def test_diagnostics_have_template_position(self, env, int_to_text_contract):
        source = "line one\nline two\n{{ x|int_to_text }}"
        diags = check(env, [int_to_text_contract], source)
        assert len(diags) == 1
        assert diags[0].lineno == 3

    def test_no_contract_means_no_diagnostics(self, env):
        diags = check(env, [], "{{ x|int_to_text }}")
        assert diags == []

    def test_unknown_filter_is_ignored(self, env, int_to_text_contract):
        diags = check(env, [int_to_text_contract], "{{ x|unknown_filter }}")
        assert diags == []


class TestChainOutputPropagation:
    def test_chain_known_output_is_propagated(
        self, env, int_to_text_contract, upper_contract
    ):
        # int -> str (int_to_text) -> str (upper) is fine.
        diags = check(
            env,
            [int_to_text_contract, upper_contract],
            "{{ 5|int_to_text|upper }}",
        )
        assert diags == []

    def test_chain_unknown_makes_next_filter_pending(
        self, env, int_to_text_contract, upper_contract
    ):
        # x is unknown, so int_to_text is pending and its output is unknown;
        # upper therefore cannot be checked and is also pending.
        diags = check(
            env,
            [int_to_text_contract, upper_contract],
            "{{ x|int_to_text|upper }}",
        )
        kinds = [d.kind for d in diags]
        assert kinds == ["pending", "pending"]

    def test_chain_type_error_stops_propagation(
        self, env, int_to_text_contract, upper_contract
    ):
        # "x" is a string, int_to_text reports a type error and its output is
        # unknown, so upper is pending rather than silently accepted.
        diags = check(
            env,
            [int_to_text_contract, upper_contract],
            '{{ "x"|int_to_text|upper }}',
        )
        kinds = [d.kind for d in diags]
        assert kinds == ["type_error", "pending"]

    def test_chain_without_contract_cannot_propagate(self, env, upper_contract):
        # middle filter has no contract, so its output is unknown.
        diags = check(env, [upper_contract], "{{ x|middle|upper }}")
        assert len(diags) == 1
        assert diags[0].is_pending


class TestMacrosAndIncludes:
    def test_macro_body_unknown_is_pending(self, env, int_to_text_contract):
        diags = check(
            env,
            [int_to_text_contract],
            "{% macro m() %}{{ x|int_to_text }}{% endmacro %}",
        )
        assert len(diags) == 1
        assert diags[0].is_pending

    def test_macro_body_literal_is_type_error(self, env, int_to_text_contract):
        diags = check(
            env,
            [int_to_text_contract],
            "{% macro m() %}{{ 's'|int_to_text }}{% endmacro %}",
        )
        assert len(diags) == 1
        assert diags[0].is_error

    def test_include_unknown_is_pending(self, int_to_text_contract):
        env = Environment(loader=DictLoader({"inc.html": "{{ y|int_to_text }}"}))
        diags = check(
            env,
            [int_to_text_contract],
            '{% include "inc.html" %}',
            template_name="main.html",
        )
        assert len(diags) == 1
        assert diags[0].is_pending
        assert diags[0].template_name == "inc.html"

    def test_include_literal_is_type_error(self, int_to_text_contract):
        env = Environment(
            loader=DictLoader({"inc.html": '{{ "bad"|int_to_text }}'})
        )
        diags = check(
            env,
            [int_to_text_contract],
            '{% include "inc.html" %}',
        )
        assert len(diags) == 1
        assert diags[0].is_error
        assert diags[0].template_name == "inc.html"

    def test_include_without_loader_is_skipped(self, env, int_to_text_contract):
        # No loader: the include cannot be resolved, but the unknown value is
        # not misjudged as legal.
        diags = check(env, [int_to_text_contract], '{% include "inc.html" %}')
        assert diags == []


class TestNoExecutionNoGlobalModification:
    def test_checker_does_not_execute_filters(self, env, int_to_text_contract):
        calls = []

        def counting_filter(value):
            calls.append(value)
            return str(value)

        env.filters["int_to_text"] = counting_filter
        checker = StaticContractChecker([int_to_text_contract])
        checker.check(env.parse('{{ "abc"|int_to_text }}{{ x|int_to_text }}'))
        assert calls == []

    def test_checker_does_not_modify_env_filters(self, env, int_to_text_contract):
        before = len(env.filters)
        checker = StaticContractChecker([int_to_text_contract])
        checker.check(env.parse('{{ "abc"|int_to_text }}'))
        assert len(env.filters) == before
        assert "int_to_text" not in env.filters

    def test_register_does_not_modify_env_filters(self, env, int_to_text_contract):
        before = len(env.filters)
        checker = StaticContractChecker()
        checker.register(int_to_text_contract)
        assert len(env.filters) == before
        assert "int_to_text" not in env.filters

    def test_normal_compilation_does_not_enable_checker(
        self, env, int_to_text_contract
    ):
        # Compiling and rendering a template must not require or run the
        # checker; the filter does not even need to exist.
        checker = StaticContractChecker([int_to_text_contract])
        assert "int_to_text" not in env.filters
        tmpl = env.from_string("{{ 5 }}")
        assert tmpl.render() == "5"
        # The checker is still opt-in and unaffected.
        diags = checker.check(env.parse('{{ "abc"|int_to_text }}'))
        assert len(diags) == 1


class TestConvenienceFunction:
    def test_check_filter_contracts(self, env, int_to_text_contract):
        diags = check_filter_contracts(
            env.parse('{{ "abc"|int_to_text }}'),
            [int_to_text_contract],
        )
        assert len(diags) == 1
        assert diags[0].is_error
