import subprocess
import sys
from pathlib import Path

import pytest

from jinja2 import Environment
from jinja2 import UndefinedError
from jinja2.loaders import DictLoader
from jinja2.loaders import FileSystemLoader
from jinja2.macrocheck import check_templates
from jinja2.macrocheck import MacroContractChecker


@pytest.fixture
def macro_env():
    here = Path(__file__).parent.resolve()
    return Environment(loader=FileSystemLoader(here / "res" / "templates"))


def test_mutual_import_legal_defaults_pass(macro_env):
    """Two templates import macros from each other; legal calls with
    default arguments produce no diagnostics."""
    report = MacroContractChecker(macro_env).check("macrocheck/good.html")
    assert report.ok, [str(d) for d in report.diagnostics]
    # Imported templates are analyzed transitively, exactly once each.
    assert report.templates == [
        "macrocheck/good.html",
        "macrocheck/lib_a.html",
        "macrocheck/lib_b.html",
    ]


def test_legal_calls_render(macro_env):
    """The same legal calls also render cleanly.  (The mutually
    importing pair above is only checked statically: Jinja's runtime
    module system does not support circular imports, which is exactly
    the situation the static checker covers.)"""
    report = check_templates(macro_env, "macrocheck/good_render.html")
    assert report.ok, [str(d) for d in report.diagnostics]
    out = macro_env.get_template("macrocheck/good_render.html").render()
    assert "Hello, Ada!" in out
    assert "Ada Lovelace" in out


def test_missing_required_argument_located(macro_env):
    report = check_templates(macro_env, "macrocheck/bad_missing.html")
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "missing-argument"
    assert diag.template == "macrocheck/bad_missing.html"
    assert diag.lineno == 2
    assert "'last'" in diag.message
    # Rendering agrees: the missing parameter is undefined when used.
    with pytest.raises(UndefinedError):
        macro_env.get_template("macrocheck/bad_missing.html").render()


def test_duplicate_argument_located(macro_env):
    report = check_templates(macro_env, "macrocheck/bad_duplicate.html")
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "duplicate-argument"
    assert diag.template == "macrocheck/bad_duplicate.html"
    assert diag.lineno == 2
    assert "'first'" in diag.message
    # Rendering agrees: the runtime raises TypeError for the duplicate.
    with pytest.raises(TypeError):
        macro_env.get_template("macrocheck/bad_duplicate.html").render()


def test_unknowns_listed_separately(macro_env):
    report = check_templates(macro_env, "macrocheck/unknowns.html")
    assert report.ok
    reasons = {u.reason for u in report.unknowns}
    assert reasons == {
        "dynamic-import",
        "shadowed-name",
        "argument-unpacking",
        "dynamic-call",
    }
    for unknown in report.unknowns:
        assert unknown.template == "macrocheck/unknowns.html"
        assert unknown.lineno > 0


def test_unknown_shadowed_variable(macro_env):
    report = check_templates(macro_env, "macrocheck/unknowns.html")
    (unknown,) = [u for u in report.unknowns if u.reason == "shadowed-name"]
    assert unknown.lineno == 4
    assert "'local'" in unknown.message


def test_unknown_dynamic_import(macro_env):
    report = check_templates(macro_env, "macrocheck/unknowns.html")
    (unknown,) = [u for u in report.unknowns if u.reason == "dynamic-import"]
    assert unknown.lineno == 1


def test_unknown_argument_unpacking(macro_env):
    report = check_templates(macro_env, "macrocheck/unknowns.html")
    (unknown,) = [u for u in report.unknowns if u.reason == "argument-unpacking"]
    assert unknown.lineno == 6


def test_unknown_dynamic_call_block(macro_env):
    report = check_templates(macro_env, "macrocheck/unknowns.html")
    (unknown,) = [u for u in report.unknowns if u.reason == "dynamic-call"]
    assert unknown.lineno == 7


def test_call_block_with_caller_passes():
    env = Environment(
        loader=DictLoader(
            {
                "t": "{% macro wrap() %}[{{ caller() }}]{% endmacro %}\n"
                "{% call wrap() %}body{% endcall %}"
            }
        )
    )
    report = check_templates(env, "t")
    assert report.ok, [str(d) for d in report.diagnostics]
    assert env.get_template("t").render().strip() == "[body]"


def test_call_block_without_caller_fails():
    env = Environment(
        loader=DictLoader(
            {
                "t": "{% macro plain() %}x{% endmacro %}\n"
                "{% call plain() %}body{% endcall %}"
            }
        )
    )
    report = check_templates(env, "t")
    assert len(report.diagnostics) == 1
    diag = report.diagnostics[0]
    assert diag.code == "caller-not-accepted"
    assert diag.lineno == 2
    # Rendering agrees: the runtime rejects the extra caller argument.
    with pytest.raises(TypeError):
        env.get_template("t").render()


def test_kwargs_and_varargs_follow_compiler_rules():
    env = Environment(
        loader=DictLoader(
            {
                "kw": "{% macro m() %}{{ kwargs }}{% endmacro %}{{ m(extra=1) }}",
                "va": "{% macro m(a) %}{{ varargs }}{% endmacro %}{{ m(1, 2, 3) }}",
                "bad_kw": "{% macro m(a) %}{{ a }}{% endmacro %}\n{{ m(1, bad=2) }}",
                "bad_va": "{% macro m(a) %}{{ a }}{% endmacro %}\n{{ m(1, 2) }}",
            }
        )
    )
    assert check_templates(env, "kw").ok
    assert check_templates(env, "va").ok
    assert env.get_template("kw").render() == "{'extra': 1}"
    assert env.get_template("va").render() == "(2, 3)"

    report = check_templates(env, "bad_kw")
    assert [d.code for d in report.diagnostics] == ["unexpected-keyword"]
    assert report.diagnostics[0].lineno == 2
    with pytest.raises(TypeError):
        env.get_template("bad_kw").render()

    report = check_templates(env, "bad_va")
    assert [d.code for d in report.diagnostics] == ["too-many-positional"]
    assert report.diagnostics[0].lineno == 2
    with pytest.raises(TypeError):
        env.get_template("bad_va").render()


def test_defaults_checked_for_existence_not_evaluated():
    def boom():
        raise AssertionError("default expression was evaluated")

    env = Environment(
        loader=DictLoader(
            {
                "t": "{% macro m(a, b=boom()) %}{{ a }}{% endmacro %}\n"
                "{{ m('x') }}\n"
                "{{ m('x', 'y') }}\n"
                "{{ m() }}"
            }
        )
    )
    env.globals["boom"] = boom
    report = check_templates(env, "t")
    # Only the genuinely missing argument is reported; the default
    # expression itself is never evaluated.
    assert [d.code for d in report.diagnostics] == ["missing-argument"]
    assert report.diagnostics[0].lineno == 4
    assert report.sentinel_count >= 1


def test_filters_tests_and_globals_are_never_called():
    def explode(*args, **kwargs):
        raise AssertionError("checker executed template code")

    env = Environment(
        loader=DictLoader(
            {
                "t": "{% macro m(a=boom()) %}{{ a | explode_filter }}{% endmacro %}\n"
                "{{ 1 is explode_test }}\n"
                "{{ boom() }}\n"
                "{{ m('x') }}"
            }
        )
    )
    env.filters["explode_filter"] = explode
    env.tests["explode_test"] = explode
    env.globals["boom"] = explode
    report = check_templates(env, "t")
    assert report.ok, [str(d) for d in report.diagnostics]
    # Filter, test, default expression and foreign call were all
    # skipped safely and counted as sentinels.
    assert report.sentinel_count >= 4


@pytest.mark.timeout(10)
def test_mutual_macro_calls_terminate():
    env = Environment(
        loader=DictLoader(
            {
                "a": "{% import 'b' as b %}\n"
                "{% macro ma(n) %}{{ b.mb(n) }}{% endmacro %}\n"
                "{{ ma(1) }}",
                "b": "{% import 'a' as a %}\n"
                "{% macro mb(n) %}{{ a.ma(n) }}{% endmacro %}\n"
                "{{ mb(1) }}",
            }
        )
    )
    report = check_templates(env, ["a", "b"])
    assert report.ok, [str(d) for d in report.diagnostics]
    assert sorted(report.templates) == ["a", "b"]


def test_report_format_contains_sections(macro_env):
    report = check_templates(macro_env, "macrocheck/bad_missing.html")
    text = report.format()
    assert "== Macro contract report ==" in text
    assert "diagnostics: 1" in text
    assert "macrocheck/bad_missing.html:2: missing-argument" in text
    assert "unknowns: 0" in text
    assert "safety sentinels:" in text


def test_example_script_outputs_report_within_90_seconds():
    script = (
        Path(__file__).parent.parent / "examples" / "macrocheck" / "check_macros.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    out = proc.stdout + proc.stderr
    assert "== Macro contract report ==" in out
    assert "safety sentinels:" in out
    assert "missing-argument" in out
    assert "unknowns:" in out
