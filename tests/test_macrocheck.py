import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2 import UndefinedError
from jinja2.macrocheck import check_templates


def make_env(templates, **kwargs):
    return Environment(loader=DictLoader(templates), **kwargs)


MUTUAL = {
    "a.html": (
        "{% import 'b.html' as b %}\n"
        "{% macro greet(name, punct='!') %}Hello {{ name }}{{ punct }}{% endmacro %}\n"
        "{{ b.wrap(greet('World')) }}\n"
    ),
    "b.html": (
        "{% import 'a.html' as a %}\n"
        "{% macro wrap(body) %}[{{ body }}]{% endmacro %}\n"
        "{{ a.greet('World', punct='?') }}\n"
    ),
}


class TestMutualImports:
    def test_legal_default_calls_pass(self):
        env = make_env(MUTUAL)
        report = check_templates(env, ["a.html", "b.html"])
        assert report.ok
        assert report.errors == []
        assert report.unknowns == []
        assert report.sentinels == {
            "templates": 2,
            "macros": 2,
            "calls": 3,
            "errors": 0,
            "unknowns": 0,
        }

    def test_analysis_terminates(self):
        # Mutual imports and macros calling each other must not recurse.
        env = make_env(MUTUAL)
        report = check_templates(env)
        assert report.ok
        assert report.templates_checked == 2

    def test_legal_default_call_confirmed_by_rendering(self):
        env = make_env(
            {
                "lib.html": (
                    "{% macro greet(name, punct='!') %}"
                    "Hello {{ name }}{{ punct }}{% endmacro %}"
                ),
                "page.html": ("{% import 'lib.html' as lib %}{{ lib.greet('World') }}"),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.ok
        assert env.get_template("page.html").render() == "Hello World!"


MISSING = {
    "page.html": (
        "{% macro show(title, body) %}{{ title }}: {{ body }}{% endmacro %}\n"
        "{{ show('only-title') }}\n"
    ),
}


class TestMissingRequiredArgument:
    def test_located_at_call_line(self):
        env = make_env(MISSING)
        report = check_templates(env, ["page.html"])
        assert not report.ok
        assert len(report.errors) == 1
        issue = report.errors[0]
        assert issue.template == "page.html"
        assert issue.line == 2
        assert "'body'" in issue.message
        assert "missing required argument" in issue.message

    def test_confirmed_by_rendering(self):
        env = make_env(MISSING, undefined=StrictUndefined)
        with pytest.raises(UndefinedError):
            env.get_template("page.html").render()


DUPLICATE = {
    "page.html": (
        "{% macro show(title, body) %}{{ title }}: {{ body }}{% endmacro %}\n"
        "{{ show('t', title='x') }}\n"
    ),
}


class TestDuplicateArgument:
    def test_located_at_call_line(self):
        env = make_env(DUPLICATE)
        report = check_templates(env, ["page.html"])
        assert not report.ok
        assert len(report.errors) == 2
        issue = next(
            issue
            for issue in report.errors
            if "both positionally and by keyword" in issue.message
        )
        assert issue.template == "page.html"
        assert issue.line == 2
        assert "'title'" in issue.message
        # The unfilled 'body' parameter is reported at the same line.
        assert all(issue.line == 2 for issue in report.errors)

    def test_confirmed_by_rendering(self):
        env = make_env(DUPLICATE)
        with pytest.raises(TypeError):
            env.get_template("page.html").render()

    def test_duplicate_keyword(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro show(title, body) %}x{% endmacro %}\n"
                    "{{ show(title='a', title='b', body='c') }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert len(report.errors) == 1
        assert report.errors[0].line == 2
        assert "more than once" in report.errors[0].message


class TestUnknownKeywordArgument:
    def test_located_at_call_line(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro show(title) %}{{ title }}{% endmacro %}\n"
                    "{{ show(title='t', extra=1) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert len(report.errors) == 1
        assert report.errors[0].line == 2
        assert "unknown keyword argument 'extra'" in report.errors[0].message

    def test_confirmed_by_rendering(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro show(title) %}{{ title }}{% endmacro %}\n"
                    "{{ show(title='t', extra=1) }}\n"
                ),
            }
        )
        with pytest.raises(TypeError):
            env.get_template("page.html").render()


class TestTooManyPositionalArguments:
    def test_located_at_call_line(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro show(title, body) %}x{% endmacro %}\n"
                    "{{ show('a', 'b', 'c') }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert len(report.errors) == 1
        assert report.errors[0].line == 2
        assert "at most 2" in report.errors[0].message

    def test_confirmed_by_rendering(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro show(title, body) %}x{% endmacro %}\n"
                    "{{ show('a', 'b', 'c') }}\n"
                ),
            }
        )
        with pytest.raises(TypeError):
            env.get_template("page.html").render()


class TestVarargsKwargsCompilerRules:
    def test_varargs_accepts_extra_positional(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro m(a) %}{{ a }}:{{ varargs|join(',') }}{% endmacro %}\n"
                    "{{ m(1, 2, 3) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.ok
        assert env.get_template("page.html").render() == "\n1:2,3"

    def test_kwargs_accepts_extra_keywords(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro m(a) %}{{ a }}:{{ kwargs['extra'] }}{% endmacro %}\n"
                    "{{ m(1, extra=2) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.ok
        assert env.get_template("page.html").render() == "\n1:2"

    def test_caller_keyword_allowed_when_macro_uses_caller(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro m() %}[{{ caller() }}]{% endmacro %}\n"
                    "{% macro other() %}called{% endmacro %}\n"
                    "{{ m(caller=other) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.ok
        assert env.get_template("page.html").render() == "\n\n[called]"


UNKNOWNS = {
    "u.html": (
        "{% import 'lib.html' as lib %}\n"
        "{% macro m(a) %}{{ a }}{% endmacro %}\n"
        "{% set m = 42 %}\n"
        "{{ m(1) }}\n"
        "{% macro w() %}{{ caller() }}{% endmacro %}\n"
        "{% call w() %}x{% endcall %}\n"
        "{% import some_var as dyn %}\n"
        "{{ lib.m(*args) }}\n"
        "{{ lib.m(**kw) }}\n"
    ),
    "lib.html": "{% macro m(a) %}{{ a }}{% endmacro %}",
}


class TestUnknownsListedSeparately:
    def test_all_unknown_kinds(self):
        env = make_env(UNKNOWNS)
        report = check_templates(env, ["u.html"])
        assert report.errors == []
        by_line = {issue.line: issue for issue in report.unknowns}
        assert set(by_line) == {4, 6, 7, 8, 9}
        assert "shadowed" in by_line[4].message
        assert "call block" in by_line[6].message
        assert "dynamic template name" in by_line[7].message
        assert "unpacking" in by_line[8].message
        assert "unpacking" in by_line[9].message
        for issue in report.unknowns:
            assert issue.template == "u.html"

    def test_shadowed_import_alias(self):
        env = make_env(
            {
                "lib.html": "{% macro m(a) %}{% endmacro %}",
                "page.html": (
                    "{% import 'lib.html' as lib %}\n"
                    "{% set lib = 1 %}\n"
                    "{{ lib.m(1) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.errors == []
        assert len(report.unknowns) == 1
        assert report.unknowns[0].line == 3
        assert "shadowed" in report.unknowns[0].message

    def test_dynamic_attribute_access(self):
        env = make_env(
            {
                "lib.html": "{% macro m(a) %}{% endmacro %}",
                "page.html": ("{% import 'lib.html' as lib %}\n{{ lib['m'](1) }}\n"),
            }
        )
        report = check_templates(env, ["page.html"])
        assert report.errors == []
        assert len(report.unknowns) == 1
        assert report.unknowns[0].line == 2


class TestDefaultsNotEvaluated:
    def test_default_expression_never_evaluated(self):
        env = make_env(
            {
                "page.html": (
                    "{% macro m(a=boom()) %}{{ a }}{% endmacro %}\n{{ m() }}\n"
                ),
            }
        )
        # Only the existence of the default is checked. Evaluating the
        # default would fail because 'boom' is undefined.
        report = check_templates(env, ["page.html"])
        assert report.ok


class TestNoFilterTestOrGlobalCalls:
    def test_never_invoked(self):
        called = []

        def boom(*args, **kwargs):
            called.append(True)
            raise AssertionError("must not be called")

        env = make_env(
            {
                "page.html": (
                    "{% macro m(a) %}{{ a }}{% endmacro %}\n"
                    "{{ m(1) }} {{ 'x'|boomfilter }} {{ boomglobal() }}"
                    " {{ 1 is boomtest }}\n"
                ),
            }
        )
        env.filters["boomfilter"] = boom
        env.tests["boomtest"] = boom
        env.globals["boomglobal"] = boom
        report = check_templates(env, ["page.html"])
        assert report.ok
        assert called == []


class TestFromImport:
    def test_from_import_alias_checked(self):
        env = make_env(
            {
                "lib.html": "{% macro m(a, b=2) %}{{ a }}{{ b }}{% endmacro %}",
                "page.html": (
                    "{% from 'lib.html' import m as mm %}\n"
                    "{{ mm(1) }}\n"
                    "{{ mm(1, 2, 3) }}\n"
                ),
            }
        )
        report = check_templates(env, ["page.html"])
        assert len(report.errors) == 1
        assert report.errors[0].line == 3
        assert "at most 2" in report.errors[0].message

    def test_missing_imported_macro(self):
        env = make_env(
            {
                "lib.html": "{% macro m() %}{% endmacro %}",
                "page.html": ("{% import 'lib.html' as lib %}\n{{ lib.nope() }}\n"),
            }
        )
        report = check_templates(env, ["page.html"])
        assert len(report.errors) == 1
        assert report.errors[0].line == 2
        assert "no macro 'nope'" in report.errors[0].message


class TestReportOutput:
    def test_render_text_contains_sections_and_sentinels(self):
        env = make_env(MUTUAL)
        report = check_templates(env, ["a.html", "b.html"])
        text = report.render_text()
        assert "errors:" in text
        assert "unknowns:" in text
        assert "sentinels: templates=2 macros=2 calls=3 errors=0 unknowns=0" in text

    def test_issue_str_includes_template_and_line(self):
        env = make_env(MISSING)
        report = check_templates(env, ["page.html"])
        assert str(report.errors[0]).startswith("page.html:2: ")


class TestFixtureTemplates:
    def test_fixture_templates_pass_and_render(self, filesystem_loader):
        env = Environment(loader=filesystem_loader)
        report = check_templates(env, ["macrocheck_page.html"])
        assert report.ok
        assert report.templates_checked == 2
        rendered = env.get_template("macrocheck_page.html").render()
        assert rendered == "1x apple 3x pear"
