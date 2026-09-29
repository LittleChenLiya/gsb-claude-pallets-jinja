import pytest

from jinja2 import DiagnosticResult
from jinja2 import DiagnosticUndefined
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import MissingValue


@pytest.fixture
def diagnostic_env():
    return Environment(
        loader=DictLoader(
            {
                "main.html": "Hello {{ user.name }}!\n"
                "{% include 'part.html' %}\n"
                "Done\n",
                "part.html": "first line\nPrice: {{ items[0].price }}\n",
            }
        ),
        diagnostic=True,
    )


def test_render_diagnostic_reports_path_and_line(diagnostic_env):
    template = diagnostic_env.get_template("main.html")
    result = template.render_diagnostic()

    assert isinstance(result, DiagnosticResult)
    assert result.text == "Hello !\nfirst line\nPrice: \nDone"
    assert result.missing == [
        MissingValue(template="main.html", line=1, path="user.name"),
        MissingValue(template="part.html", line=2, path="items[0].price"),
    ]


def test_render_diagnostic_result_unpacking(diagnostic_env):
    template = diagnostic_env.get_template("main.html")
    text, missing = template.render_diagnostic()

    assert text == "Hello !\nfirst line\nPrice: \nDone"
    assert [record.path for record in missing] == ["user.name", "items[0].price"]


def test_render_diagnostic_requires_diagnostic_environment():
    template = Environment().from_string("{{ user.name }}")

    with pytest.raises(RuntimeError, match="diagnostic=True"):
        template.render_diagnostic()


def test_diagnostic_environment_uses_diagnostic_undefined():
    env = Environment(diagnostic=True)
    assert env.undefined is DiagnosticUndefined


def test_default_filter_is_not_reported():
    env = Environment(diagnostic=True)
    template = env.from_string("{{ user.name|default('anonymous') }}")
    result = template.render_diagnostic()

    assert result.text == "anonymous"
    assert result.missing == []


def test_is_defined_test_is_not_reported():
    env = Environment(diagnostic=True)
    template = env.from_string(
        "{% if user is defined %}{{ user.name }}{% else %}no user{% endif %}"
    )
    result = template.render_diagnostic()

    assert result.text == "no user"
    assert result.missing == []


def test_loop_local_variable_is_not_reported():
    env = Environment(diagnostic=True)
    template = env.from_string(
        "{% for item in items %}{{ loop.index }}: {{ item }};{% endfor %}"
    )
    result = template.render_diagnostic(items=["a", "b"])

    assert result.text == "1: a;2: b;"
    assert result.missing == []


def test_real_errors_are_not_swallowed():
    env = Environment(diagnostic=True)
    template = env.from_string("{{ 1 / 0 }}")

    with pytest.raises(ZeroDivisionError):
        template.render_diagnostic()


def test_records_do_not_leak_between_renders():
    env = Environment(diagnostic=True)
    first = env.from_string("{{ missing_value }}").render_diagnostic()
    second = env.from_string("everything is {{ 'fine' }}").render_diagnostic()

    assert [record.path for record in first.missing] == ["missing_value"]
    assert second.text == "everything is fine"
    assert second.missing == []


def test_diagnostic_render_does_not_affect_normal_render():
    env = Environment(diagnostic=True)
    template = env.from_string("{{ user.name }}")
    result = template.render_diagnostic()

    assert [record.path for record in result.missing] == ["user.name"]
    assert template.render() == ""
