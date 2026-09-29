"""Tests for the explicitly enabled diagnostic rendering entry point."""

import pytest

import jinja2
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import MissingVariable

MAIN_TEMPLATE = (
    "Hello {{ user.name }}!\n"  # line 1: user.name missing
    "{% include 'part' %}\n"  # line 2
)

PART_TEMPLATE = (
    "header\n"  # line 1
    "price={{ items[0].price }}\n"  # line 2: items[0].price missing
)


@pytest.fixture
def loader_env():
    return Environment(
        loader=DictLoader({"main": MAIN_TEMPLATE, "part": PART_TEMPLATE})
    )


def test_missing_values_in_main_and_included_template(loader_env):
    template = loader_env.get_template("main")
    result = template.render_with_diagnostics(items=[])

    assert result.output == "Hello !\nheader\nprice="
    assert result.missing == (
        MissingVariable(name="main", line=1, path="user.name"),
        MissingVariable(name="part", line=2, path="items[0].price"),
    )


def test_missing_records_have_template_name_line_and_path():
    env = Environment()
    template = env.from_string("a={{ a.b }} x={{ x['k'].y }}")
    result = template.render_with_diagnostics(a={})

    assert result.output == "a= x="
    assert result.missing == (
        MissingVariable(name=None, line=1, path="a.b"),
        MissingVariable(name=None, line=1, path="x['k'].y"),
    )
    assert str(result.missing[0]) == "a.b is undefined (line 1)"


def test_default_filter_is_not_reported(env):
    template = env.from_string("{{ missing_var|default('fallback') }}")
    result = template.render_with_diagnostics()

    assert result.output == "fallback"
    assert result.missing == ()


def test_is_defined_is_not_reported(env):
    template = env.from_string(
        "{% if other_var is defined %}yes{% else %}no{% endif %}"
    )
    result = template.render_with_diagnostics()

    assert result.output == "no"
    assert result.missing == ()


def test_is_undefined_is_not_reported(env):
    template = env.from_string("{% if other_var is undefined %}missing{% endif %}")
    result = template.render_with_diagnostics()

    assert result.output == "missing"
    assert result.missing == ()


def test_loop_local_variables_are_not_reported(env):
    template = env.from_string(
        "{% for item in items %}"
        "{{ item.name }}-{{ item.price }}-{{ loop.index }}"
        "{% endfor %}"
    )
    result = template.render_with_diagnostics(
        items=[{"name": "a"}, {"name": "b", "price": 5}]
    )

    assert result.output == "a--1b-5-2"
    assert result.missing == (MissingVariable(name=None, line=1, path="item.price"),)


def test_real_error_is_not_swallowed(env):
    template = env.from_string("{{ 1 / 0 }}")
    with pytest.raises(ZeroDivisionError):
        template.render_with_diagnostics()


def test_undefined_used_in_arithmetic_still_raises(env):
    template = env.from_string("{{ missing_var + 1 }}")
    with pytest.raises(jinja2.UndefinedError):
        template.render_with_diagnostics()


def test_independent_renders_do_not_share_records(loader_env):
    template = loader_env.get_template("main")

    first = template.render_with_diagnostics(items=[])
    second = template.render_with_diagnostics(
        user={"name": "Bob"}, items=[{"price": 9}]
    )
    third = template.render_with_diagnostics(items=[])

    assert len(first.missing) == 2
    assert second.output == "Hello Bob!\nheader\nprice=9"
    assert second.missing == ()
    assert len(third.missing) == 2


def test_normal_render_does_not_collect_or_leak(loader_env):
    template = loader_env.get_template("main")
    output = template.render(user={"name": "Ann"}, items=[{"price": 3}])
    assert output == "Hello Ann!\nheader\nprice=3"

    result = template.render_with_diagnostics(items=[])
    assert len(result.missing) == 2


def test_result_is_dataclass_api(env):
    from jinja2 import DiagnosticResult

    template = env.from_string("{{ foo }}")
    result = template.render_with_diagnostics()
    assert isinstance(result, DiagnosticResult)
    assert isinstance(result.output, str)
    assert all(isinstance(item, MissingVariable) for item in result.missing)


def test_duplicate_access_is_recorded_once(env):
    template = env.from_string("{{ foo }}{{ foo }}")
    result = template.render_with_diagnostics()
    assert result.missing == (MissingVariable(name=None, line=1, path="foo"),)


def test_deep_chain_with_missing_intermediate(env):
    template = env.from_string("{{ x['k'].y }}")
    result = template.render_with_diagnostics()

    assert result.output == ""
    assert result.missing == (MissingVariable(name=None, line=1, path="x['k'].y"),)


def test_dynamic_index_missing_records_outer_chain(env):
    template = env.from_string("{{ items[idx] }}")
    result = template.render_with_diagnostics(items=[])

    assert result.missing == (MissingVariable(name=None, line=1, path="items[?]"),)


def test_defined_values_are_not_reported(env):
    template = env.from_string("{{ user.name }} {{ items[0].price }}")
    result = template.render_with_diagnostics(
        user={"name": "Ann"}, items=[{"price": 4}]
    )

    assert result.output == "Ann 4"
    assert result.missing == ()
