"""Tests for the read-only macro contract exporter."""

import pytest

from jinja2 import Environment
from jinja2 import nodes
from jinja2.contracts import build_macro_contract
from jinja2.contracts import ContractReport
from jinja2.contracts import export_macro_contracts
from jinja2.contracts import macro_contracts


@pytest.fixture
def env():
    return Environment()


class TestGreetContract:
    """The acceptance macro: greet(name, suffix=make_suffix())."""

    def test_default_expression_position(self, env):
        src = (
            "line1\n"
            "line2\n"
            "{% macro greet(name,\n"
            "          suffix=make_suffix()) %}\n"
            "hi {{ name }}\n"
            "{% endmacro %}\n"
        )
        report = export_macro_contracts(src)
        assert report.errors == []
        assert len(report.macros) == 1
        macro = report.macros[0]
        assert macro.name == "greet"
        assert macro.lineno == 3
        assert [p.name for p in macro.params] == ["name", "suffix"]
        assert macro.params[0].default_lineno is None
        # The default expression starts on line 4, not on the macro line.
        assert macro.params[1].default_lineno == 4

    def test_default_expression_is_not_evaluated(self, env):
        calls = []

        def make_suffix():
            calls.append(1)
            return "!"

        src = "{% macro greet(name, suffix=make_suffix()) %}{{ name }}{% endmacro %}"
        report = export_macro_contracts(src, environment=env)
        assert report.errors == []
        assert calls == []
        assert report.macros[0].params[1].default_lineno == 1

    def test_no_defaults(self, env):
        src = "{% macro greet(name) %}{{ name }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert [p.name for p in macro.params] == ["name"]
        assert macro.params[0].default_lineno is None


class TestSpecialNames:
    def test_kwargs_detected(self, env):
        src = "{% macro m() %}{{ kwargs }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.accesses_kwargs is True
        assert macro.accepts_extra_keywords is True
        assert macro.accesses_varargs is False

    def test_varargs_detected(self, env):
        src = "{% macro m() %}{{ varargs }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.accesses_varargs is True
        assert macro.accepts_extra_positionals is True
        assert macro.accesses_kwargs is False

    def test_caller_detected(self, env):
        src = "{% macro m() %}{{ caller() }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.receives_caller is True

    def test_explicit_kwargs_param(self, env):
        src = "{% macro m(kwargs) %}{{ kwargs }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.accepts_extra_keywords is True
        assert macro.accesses_kwargs is False

    def test_explicit_varargs_param(self, env):
        src = "{% macro m(varargs) %}{{ varargs }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.accepts_extra_positionals is True
        assert macro.accesses_varargs is False

    def test_explicit_caller_with_default(self, env):
        src = "{% macro m(caller=none) %}{{ caller() }}{% endmacro %}"
        report = export_macro_contracts(src)
        macro = report.macros[0]
        assert macro.receives_caller is True
        assert report.errors == []

    def test_explicit_caller_without_default_is_error(self, env):
        src = "{% macro m(caller) %}{{ caller() }}{% endmacro %}"
        report = export_macro_contracts(src)
        assert len(report.errors) == 1
        assert "caller" in report.errors[0].message


class TestScopes:
    def test_nested_macro_keeps_scope(self, env):
        src = (
            "{% macro outer() %}"
            "{% macro inner() %}{{ kwargs }}{% endmacro %}"
            "{{ caller() }}"
            "{% endmacro %}"
        )
        report = export_macro_contracts(src)
        by_name = {m.name: m for m in report.macros}
        assert by_name["outer"].scope == ()
        assert by_name["outer"].receives_caller is True
        # inner's kwargs must not leak onto outer.
        assert by_name["outer"].accesses_kwargs is False
        assert by_name["inner"].scope == ("outer",)
        assert by_name["inner"].accesses_kwargs is True

    def test_same_name_macros_kept_separate(self, env):
        src = (
            "{% macro same() %}a{% endmacro %}"
            "{% macro same() %}b{% endmacro %}"
        )
        report = export_macro_contracts(src)
        assert len(report.macros) == 2
        assert all(m.name == "same" for m in report.macros)

    def test_nested_declaration_order(self, env):
        src = (
            "{% macro a() %}"
            "{% macro b() %}"
            "{% macro c() %}x{% endmacro %}"
            "{% endmacro %}"
            "{% endmacro %}"
        )
        report = export_macro_contracts(src)
        assert [(m.name, m.scope) for m in report.macros] == [
            ("a", ()),
            ("b", ("a",)),
            ("c", ("a", "b")),
        ]


class TestDynamicInclude:
    def test_dynamic_include_marked_unknown(self, env):
        src = "{% include partial %}{% macro m() %}x{% endmacro %}"
        report = export_macro_contracts(src)
        assert len(report.dynamic_includes) == 1
        assert report.dynamic_includes[0].template == "partial"
        assert report.dynamic_includes[0].lineno == 1
        # The macro is still exported; the include is not loaded.
        assert [m.name for m in report.macros] == ["m"]

    def test_static_include_not_dynamic(self, env):
        src = '{% include "header.html" %}{% macro m() %}x{% endmacro %}'
        report = export_macro_contracts(src)
        assert report.dynamic_includes == []

    def test_dynamic_include_not_loaded(self, env):
        # A dynamic include must not trigger the loader.
        class ExplodingLoader:
            def get_source(self, *args, **kwargs):
                raise AssertionError("include must not be loaded")

        env.loader = ExplodingLoader()
        src = "{% include partial %}"
        report = export_macro_contracts(src, environment=env)
        assert len(report.dynamic_includes) == 1


class TestSyntaxErrors:
    def test_syntax_error_reported(self, env):
        src = "{% macro broken( %}x{% endmacro %}"
        report = export_macro_contracts(src)
        assert len(report.errors) == 1
        assert report.errors[0].lineno == 1
        assert report.macros == []

    def test_syntax_error_does_not_raise(self, env):
        report = export_macro_contracts("{% macro broken( %}")
        assert isinstance(report, ContractReport)
        assert len(report.errors) == 1


class TestApi:
    def test_macro_contracts_from_ast(self, env):
        ast = env.parse("{% macro m(a, b=2) %}{{ a }}{% endmacro %}")
        report = macro_contracts(ast)
        assert len(report.macros) == 1
        assert report.macros[0].params[1].default_lineno == 1

    def test_build_macro_contract_directly(self, env):
        ast = env.parse("{% macro m(a, b=foo()) %}{{ a }}{% endmacro %}")
        node = ast.find(nodes.Macro)
        contract, error = build_macro_contract(node, ("outer",))
        assert contract.name == "m"
        assert contract.scope == ("outer",)
        assert [p.name for p in contract.params] == ["a", "b"]
        assert contract.params[1].default_lineno == 1
        assert error is None

    def test_report_to_dict_and_json(self, env):
        src = "{% macro m(a=1) %}{{ a }}{% endmacro %}"
        report = export_macro_contracts(src)
        data = report.to_dict()
        assert data["macros"][0]["name"] == "m"
        assert data["macros"][0]["params"][0]["default_lineno"] == 1
        assert "macros" in report.to_json()

    def test_report_str(self, env):
        src = "{% macro m(a=1) %}{{ a }}{% endmacro %}"
        report = export_macro_contracts(src)
        text = str(report)
        assert "macro m" in text
        assert "default at line 1" in text

    def test_empty_template(self, env):
        report = export_macro_contracts("just some text")
        assert report.macros == []
        assert report.errors == []
        assert "no macros found" in str(report)
