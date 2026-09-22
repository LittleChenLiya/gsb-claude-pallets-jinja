from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import PrefixLoader
from jinja2.meta import analyze_template_dependencies


def make_env(mapping, **kwargs):
    return Environment(loader=DictLoader(mapping), **kwargs)


class TestAcceptance:
    def test_choice_loader_cycle_and_include_list(self):
        env = Environment(
            loader=ChoiceLoader(
                [
                    DictLoader({}),
                    DictLoader(
                        {
                            "page": (
                                '{% extends "layout" %}\n'
                                '{% include ["missing", "partial"] %}'
                            ),
                            "layout": "layout",
                            "partial": '{% include "page" %}',
                        }
                    ),
                ]
            )
        )
        graph = analyze_template_dependencies(env, "page")

        assert graph.templates == ["page", "layout", "partial"]
        assert graph.issues == []

        extends, include, back = graph.edges
        assert (extends.kind, extends.target, extends.lineno) == (
            "extends",
            "layout",
            1,
        )
        # the constant list picks the first existing template
        assert (include.kind, include.target, include.lineno) == (
            "include",
            "partial",
            2,
        )
        assert (back.kind, back.target, back.lineno) == ("include", "page", 1)

        # the closed loop is reported and analysis terminates normally
        assert graph.cycles == [["page", "partial", "page"]]

        # loader origins match what a real load would use
        assert graph.origins["page"] == "ChoiceLoader[1] -> DictLoader"
        assert extends.target_origin == "ChoiceLoader[1] -> DictLoader"
        assert "extends" in extends.snippet

    def test_dynamic_and_ignore_missing(self):
        env = make_env(
            {
                "page": (
                    "{% include name %}\n"
                    '{% include "gone" ignore missing %}\n'
                    '{% include "also_gone" %}'
                )
            }
        )
        graph = analyze_template_dependencies(env, "page")

        dynamic = graph.dynamic_edges
        assert len(dynamic) == 1
        assert dynamic[0].target is None
        assert dynamic[0].lineno == 1
        assert not dynamic[0].resolved

        missing = {issue.template: issue for issue in graph.missing}
        assert missing["gone"].severity == "warning"
        assert missing["gone"].lineno == 2
        assert missing["also_gone"].severity == "error"
        assert missing["also_gone"].referenced_by == "page"
        assert "TemplateNotFound" in missing["gone"].reason


class TestNoExecution:
    def test_globals_filters_and_expressions_not_evaluated(self):
        def boom():
            raise AssertionError("global function was called")

        def boom_filter(value):
            raise AssertionError("filter was called")

        env = make_env(
            {
                "page": (
                    '{{ boom() }}\n{{ value | boom_filter }}\n{% include "partial" %}'
                ),
                "partial": "{{ missing_undeclared_name }}",
            }
        )
        env.globals["boom"] = boom
        env.filters["boom_filter"] = boom_filter

        graph = analyze_template_dependencies(env, "page")

        assert graph.templates == ["page", "partial"]
        assert graph.issues == []
        assert [edge.target for edge in graph.edges] == ["partial"]


class TestLoaderSemantics:
    def test_shadowing_by_choice_loader(self):
        env = Environment(
            loader=ChoiceLoader(
                [
                    DictLoader({"shared": '{% extends "base" %}', "base": "new"}),
                    DictLoader({"shared": "old", "base": "old"}),
                ]
            )
        )
        graph = analyze_template_dependencies(env, "shared")

        assert graph.origins["shared"] == "ChoiceLoader[0] -> DictLoader"
        assert graph.edges[0].target_origin == "ChoiceLoader[0] -> DictLoader"
        assert graph.templates == ["shared", "base"]

    def test_prefix_loader_origin(self):
        env = Environment(
            loader=PrefixLoader(
                {
                    "app": DictLoader(
                        {"index": '{% import "app/macros" as m %}', "macros": ""}
                    )
                }
            )
        )
        graph = analyze_template_dependencies(env, "app/index")

        assert graph.origins["app/index"] == "PrefixLoader['app'] -> DictLoader"
        edge = graph.edges[0]
        assert edge.kind == "import"
        assert edge.target == "app/macros"
        assert edge.target_origin == "PrefixLoader['app'] -> DictLoader"

    def test_from_import_tracked(self):
        env = make_env({"page": '{% from "macros" import thing %}', "macros": ""})
        graph = analyze_template_dependencies(env, "page")

        assert [(e.kind, e.target) for e in graph.edges] == [("from_import", "macros")]


class TestErrors:
    def test_syntax_error_reported_with_name_and_lineno(self):
        env = make_env({"page": '{% extends "layout" %}', "layout": "{% if %}"})
        graph = analyze_template_dependencies(env, "page")

        assert graph.templates == ["page", "layout"]
        (issue,) = graph.issues
        assert issue.kind == "syntax"
        assert issue.template == "layout"
        assert issue.lineno == 1
        assert "TemplateSyntaxError" in issue.reason

    def test_missing_entry_template(self):
        env = make_env({})
        graph = analyze_template_dependencies(env, "nope")

        assert graph.templates == []
        (issue,) = graph.issues
        assert issue.kind == "missing"
        assert issue.template == "nope"
        assert issue.severity == "error"

    def test_missing_include_list_all_candidates(self):
        env = make_env({"page": '{% include ["a", "b"] %}'})
        graph = analyze_template_dependencies(env, "page")

        assert {issue.template for issue in graph.missing} == {"a", "b"}
        (edge,) = graph.edges
        assert not edge.resolved
        assert not edge.dynamic


class TestParseOnce:
    def test_diamond_dependency_loaded_and_parsed_once(self):
        reads = []

        class CountingLoader(DictLoader):
            def get_source(self, environment, template):
                reads.append(template)
                return super().get_source(environment, template)

        env = Environment(
            loader=CountingLoader(
                {
                    "a": '{% include "b" %}\n{% include "c" %}',
                    "b": '{% include "d" %}',
                    "c": '{% include "d" %}',
                    "d": "leaf",
                }
            )
        )
        graph = analyze_template_dependencies(env, "a")

        assert graph.templates == ["a", "b", "d", "c"]
        assert sorted(reads) == ["a", "b", "c", "d"]
        assert reads.count("d") == 1
        # both paths to the diamond tip still produce edges
        assert [(e.source, e.target) for e in graph.edges] == [
            ("a", "b"),
            ("b", "d"),
            ("a", "c"),
            ("c", "d"),
        ]

    def test_self_cycle_terminates(self):
        env = make_env({"page": '{% include "page" %}'})
        graph = analyze_template_dependencies(env, "page")

        assert graph.templates == ["page"]
        assert graph.cycles == [["page", "page"]]

    def test_deterministic_output(self):
        mapping = {
            "page": '{% extends "layout" %}{% include "partial" %}',
            "layout": '{% include "partial" %}',
            "partial": "",
        }
        first = analyze_template_dependencies(make_env(mapping), "page")
        second = analyze_template_dependencies(make_env(mapping), "page")

        assert first == second
