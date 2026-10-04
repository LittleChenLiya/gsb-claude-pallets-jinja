import pytest

from jinja2 import Environment
from jinja2.exceptions import TemplateSyntaxError
from jinja2.loaders import BaseLoader
from jinja2.macro_contract import extract_macro_contracts
from jinja2.macro_contract import format_report
from jinja2.macro_contract import main


@pytest.fixture
def env():
    return Environment()


def test_greet_parameters_and_default_position(env):
    """greet(name, suffix=make_suffix()) reports both parameters in
    order and the position of the suffix default expression, without
    ever calling make_suffix.
    """
    calls = []
    env.globals["make_suffix"] = lambda: calls.append(1)
    source = (
        "{% macro greet(name, suffix=make_suffix()) %}"
        "{{ name }}{{ suffix }}"
        "{% endmacro %}"
    )
    contract = extract_macro_contracts(env, source)

    assert len(contract.macros) == 1
    (greet,) = contract.macros
    assert greet.name == "greet"
    assert greet.qualified_name == "greet"
    assert greet.scope == ()
    assert [p.name for p in greet.parameters] == ["name", "suffix"]

    name, suffix = greet.parameters
    assert not name.has_default
    assert name.default_lineno is None
    assert suffix.has_default
    # The position points at the default expression itself.
    assert suffix.default_lineno == 1
    assert suffix.default_col_offset == source.index("make_suffix()")
    assert suffix.default_source == "make_suffix()"

    assert not greet.accepts_caller
    assert not greet.accepts_kwargs
    assert not greet.accepts_varargs

    # Extracting the contract must not evaluate the default expression.
    assert calls == []


def test_default_position_points_at_expression_not_macro_tag(env):
    """The default position is the position of the expression itself,
    not of the whole {% macro %} tag.
    """
    source = "prefix\n{% macro greet(name, suffix=make_suffix()) %}{% endmacro %}"
    (greet,) = extract_macro_contracts(env, source).macros
    suffix = greet.parameters[1]

    assert greet.lineno == 2
    assert suffix.default_lineno == 2
    # The column points at the expression, not at the tag start.
    tag_col = source.splitlines()[1].index("{%")
    assert suffix.default_col_offset != tag_col
    assert suffix.default_col_offset == source.splitlines()[1].index("make_suffix()")
    # The position identifies the expression source text exactly.
    line = source.splitlines()[1]
    assert line[suffix.default_col_offset :].startswith("make_suffix()")


def test_declaration_order_is_stable(env):
    source = (
        "{% macro zebra() %}{% endmacro %}"
        "{% macro alpha(x) %}{% endmacro %}"
        "{% macro mango(a, b=1) %}{% endmacro %}"
    )
    contract = extract_macro_contracts(env, source)
    assert [m.name for m in contract.macros] == ["zebra", "alpha", "mango"]


def test_kwargs_macro_accepts_extra_keywords(env):
    source = "{% macro render(title) %}{{ kwargs }}{% endmacro %}"
    (macro,) = extract_macro_contracts(env, source).macros
    assert macro.accepts_kwargs
    assert not macro.accepts_varargs
    assert not macro.accepts_caller


def test_varargs_macro_accepts_extra_positional(env):
    source = "{% macro render() %}{{ varargs }}{% endmacro %}"
    (macro,) = extract_macro_contracts(env, source).macros
    assert macro.accepts_varargs
    assert not macro.accepts_kwargs


def test_caller_macro(env):
    source = "{% macro wrapper() %}{{ caller() }}{% endmacro %}"
    (macro,) = extract_macro_contracts(env, source).macros
    assert macro.accepts_caller


def test_explicit_special_params_are_not_flagged(env):
    """Explicitly declared kwargs/varargs parameters follow the
    compiler's rules and are not reported as implicit access.
    """
    source = (
        "{% macro a(kwargs) %}{{ kwargs }}{% endmacro %}"
        "{% macro b(varargs) %}{{ varargs }}{% endmacro %}"
    )
    a, b = extract_macro_contracts(env, source).macros
    assert not a.accepts_kwargs
    assert not b.accepts_varargs


def test_nested_and_same_name_macros_keep_scopes(env):
    """Nested macros and macros sharing a name are reported as separate
    entries with their own scope, in declaration order.
    """
    source = (
        "{% macro item() %}"
        "{% macro item(inner_param=1) %}{{ inner_param }}{% endmacro %}"
        "{% endmacro %}"
        "{% macro item() %}{% endmacro %}"
    )
    contract = extract_macro_contracts(env, source)
    assert len(contract.macros) == 3

    outer, inner, sibling = contract.macros

    assert outer.qualified_name == "item"
    assert outer.scope == ()
    assert [p.name for p in outer.parameters] == []

    # The nested macro keeps its own scope and its own parameters;
    # they are not mixed into the outer or sibling entries.
    assert inner.qualified_name == "item.item"
    assert inner.scope == ("item",)
    assert [p.name for p in inner.parameters] == ["inner_param"]

    assert sibling.qualified_name == "item"
    assert sibling.scope == ()
    assert [p.name for p in sibling.parameters] == []


def test_nested_special_param_matches_compiler_rules(env):
    """Mirroring the compiler: a special parameter used by a nested
    macro is received through the enclosing macro as well (closure),
    while an unrelated macro with the same name is unaffected.
    """
    source = (
        "{% macro item() %}"
        "{% macro inner() %}{{ kwargs }}{% endmacro %}"
        "{% endmacro %}"
        "{% macro item() %}{% endmacro %}"
    )
    outer, inner, sibling = extract_macro_contracts(env, source).macros
    assert inner.accepts_kwargs
    # Same closure behavior as the compiler's macro_body.
    assert outer.accepts_kwargs
    # The sibling scope is not mixed in.
    assert not sibling.accepts_kwargs


def test_call_block_is_reported_as_anonymous_macro(env):
    source = (
        "{% macro render_items(items) %}"
        "{% call(item) render(items) %}{{ item }}{% endcall %}"
        "{% endmacro %}"
    )
    contract = extract_macro_contracts(env, source)
    outer, call_block = contract.macros
    assert outer.name == "render_items"
    assert call_block.name is None
    assert call_block.qualified_name == "render_items.<call>"
    assert [p.name for p in call_block.parameters] == ["item"]


def test_dynamic_include_is_flagged_unknown_and_not_loaded():
    """A dynamic include is only flagged as unknown; the loader is
    never invoked to infer anything.
    """

    class ExplodingLoader(BaseLoader):
        def get_source(self, environment, template):
            raise AssertionError("loader must not be called")

    env = Environment(loader=ExplodingLoader())
    source = (
        "{% include template_name %}\n"
        "{% include 'static.html' %}\n"
        "{% include ['fallback.html', template_name] %}"
    )
    contract = extract_macro_contracts(env, source)

    assert contract.has_dynamic_includes
    assert [(i.dynamic, i.source) for i in contract.includes] == [
        (True, None),
        (False, "static.html"),
        (False, "fallback.html"),
        (True, None),
    ]
    assert contract.includes[0].lineno == 1


def test_syntax_error_is_raised(env):
    with pytest.raises(TemplateSyntaxError):
        extract_macro_contracts(env, "{% macro broken( %}")


def test_default_expressions_with_complex_source(env):
    source = "{% macro m(a=1 + 2, b={'k': 'v'}, c=(1, 2)) %}{% endmacro %}"
    (macro,) = extract_macro_contracts(env, source).macros
    a, b, c = macro.parameters
    assert a.default_source == "1 + 2"
    assert b.default_source == "{'k': 'v'}"
    assert c.default_source == "(1, 2)"
    line = source.splitlines()[0]
    for param in (a, b, c):
        assert line[param.default_col_offset :].startswith(param.default_source)


def test_format_report(env):
    source = (
        "{% macro greet(name, suffix=make_suffix()) %}{% endmacro %}\n"
        "{% include dynamic_name %}"
    )
    contract = extract_macro_contracts(env, source)
    report = format_report(contract, name="example.html")
    assert "macro greet (line 1)" in report
    assert "param name" in report
    assert "param suffix = make_suffix() (default at line 1, col 28)" in report
    assert "accepts caller: no" in report
    assert "dynamic include (unknown)" in report


def test_main_prints_report(env, tmp_path, capsys):
    template = tmp_path / "page.html"
    template.write_text(
        "{% macro greet(name, suffix=make_suffix()) %}{% endmacro %}",
        encoding="utf-8",
    )
    assert main([str(template)]) == 0
    out = capsys.readouterr().out
    assert "macro greet (line 1)" in out
    assert "param suffix = make_suffix() (default at line 1, col 28)" in out


def test_main_reports_syntax_error(tmp_path, capsys):
    template = tmp_path / "broken.html"
    template.write_text("{% macro broken( %}", encoding="utf-8")
    assert main([str(template)]) == 1
    assert "error" in capsys.readouterr().err
