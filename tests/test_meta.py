from __future__ import annotations

import pytest

from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import meta
from jinja2 import PrefixLoader
from jinja2 import TemplateSyntaxError


def test_analyze_extends_include_cycle_and_line_numbers():
    templates = {
        "page": ('{% extends "layout" %}\n{% include ["missing", "partial"] %}\n'),
        "layout": "layout source\nsecond line",
        "partial": '{% extends "page" %}',
    }
    env = Environment(loader=DictLoader(templates))
    graph = meta.analyze_dependencies(env, "page")

    assert list(graph.templates) == ["page", "layout", "partial"]
    assert graph.templates["layout"].summary == "layout source"
    assert graph.templates["page"].loader == "DictLoader"

    edges = graph.entries("page")
    assert [(edge.type, edge.lineno, edge.target) for edge in edges] == [
        (meta.DependencyType.EXTENDS, 1, "layout"),
        (meta.DependencyType.INCLUDE, 2, "partial"),
    ]
    assert edges[1].candidates == ("missing",)
    assert graph.entries("partial")[0].lineno == 1
    assert graph.cycles == (("page", "partial", "page"),)
    assert graph.missing == ()
    assert graph.ok is True


def test_analyze_dynamic_and_ignore_missing_are_distinct():
    templates = {
        "page": (
            "{% include dynamic_name %}\n"
            '{% include "gone" ignore missing %}\n'
            '{% include ["also_gone"] ignore missing %}\n'
            '{% include ["dyn_const", other_var] %}\n'
        ),
        "dyn_const": "dyn const",
    }
    env = Environment(loader=DictLoader(templates))
    graph = meta.analyze_dependencies(env, "page")

    assert len(graph.dynamic) == 2
    assert graph.dynamic[0].expression == "variable 'dynamic_name'"
    assert graph.dynamic[0].candidates == ()
    assert graph.dynamic[1].candidates == ("dyn_const",)
    assert graph.dynamic[1].expression.startswith("[")
    assert [edge.target for edge in graph.entries("page")] == [
        None,
        None,
        None,
        None,
    ]
    assert [(item.name, item.severity) for item in graph.missing] == [
        ("gone", "warning"),
        (None, "warning"),
    ]
    assert graph.ok is True


def test_analyze_does_not_evaluate_globals_filters_or_expressions():
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("analysis must not call globals or filters")

    env = Environment(
        loader=DictLoader(
            {
                "page": (
                    "{{ boom() }}\n"
                    '{{ "x" | boom_filter }}\n'
                    "{% include boom_target() %}\n"
                ),
            }
        )
    )
    env.globals["boom"] = boom
    env.filters["boom_filter"] = boom
    env.globals["boom_target"] = boom

    graph = meta.analyze_dependencies(env, "page")
    assert calls == []
    assert graph.dynamic[0].expression == "call variable 'boom_target'(...)"
    assert graph.entries("page")[0].target is None


def test_analyze_import_and_from_import():
    env = Environment(
        loader=DictLoader(
            {
                "page": (
                    '{% import "lib" as lib %}\n{% from "other" import a, b as c %}\n'
                ),
                "lib": "{% macro x() %}x{% endmacro %}",
                "other": "{% set a = 1 %}{% set b = 2 %}",
            }
        )
    )
    graph = meta.analyze_dependencies(env, "page")
    assert [
        (edge.type, edge.lineno, edge.target) for edge in graph.entries("page")
    ] == [
        (meta.DependencyType.IMPORT, 1, "lib"),
        (meta.DependencyType.FROM_IMPORT, 2, "other"),
    ]


def test_analyze_empty_constant_include_list_is_missing_not_dynamic():
    env = Environment(loader=DictLoader({"page": "{% include [] %}"}))
    graph = meta.analyze_dependencies(env, "page")
    assert graph.dynamic == ()
    assert graph.edges[0].expression is None
    assert graph.missing[0].name is None
    assert graph.missing[0].candidates == ()
    assert graph.missing[0].severity == "error"
    assert graph.ok is False


def test_analyze_choice_loader_shadowing_and_prefix_loader():
    first = {"shared": "first", "a/p": '{% extends "shared" %}'}
    second = {"shared": "second", "a/p": "ignored"}
    env = Environment(loader=ChoiceLoader([DictLoader(first), DictLoader(second)]))
    graph = meta.analyze_dependencies(env, "a/p")
    assert graph.templates["a/p"].loader == "ChoiceLoader > DictLoader[0]"
    assert graph.templates["shared"].loader == "ChoiceLoader > DictLoader[0]"

    prefixed = PrefixLoader({"a": DictLoader({"p": '{% extends "shared" %}'})})
    env2 = Environment(loader=ChoiceLoader([DictLoader({"shared": "root"}), prefixed]))
    graph2 = meta.analyze_dependencies(env2, "a/p")
    assert graph2.templates["a/p"].loader == (
        "ChoiceLoader > PrefixLoader[1] > DictLoader prefix='a'"
    )
    assert graph2.templates["shared"].loader == "ChoiceLoader > DictLoader[0]"


def test_analyze_diamond_reads_each_template_once():
    reads = []

    class CountingDictLoader(DictLoader):
        def get_source(self, environment, template):
            reads.append(template)
            return super().get_source(environment, template)

    loader = CountingDictLoader(
        {
            "top": '{% include "a" %}{% include "b" %}',
            "a": '{% include "leaf" %}',
            "b": '{% include "leaf" %}',
            "leaf": "leaf",
        }
    )
    env = Environment(loader=loader)
    graph = meta.analyze_dependencies(env, "top")
    assert list(graph.templates) == ["top", "a", "leaf", "b"]
    assert reads == ["top", "a", "leaf", "b"]


def test_analyze_missing_entry_and_ordering():
    env = Environment(
        loader=DictLoader(
            {
                "page": ('{% include "z" %}\n{% include "a" %}\n'),
                "a": "a",
            }
        )
    )
    graph = meta.analyze_dependencies(env, "page")
    assert [edge.target for edge in graph.edges] == [None, "a"]
    assert [(item.name, item.lineno) for item in graph.missing] == [
        ("z", 1),
    ]
    assert graph.ok is False

    graph = meta.analyze_dependencies(Environment(loader=DictLoader({})), "gone")
    assert graph.templates == {}
    assert graph.missing[0].name == "gone"
    assert graph.missing[0].referenced_by is None
    assert graph.ok is False


def test_analyze_syntax_error_keeps_name_line_and_cause():
    env = Environment(loader=DictLoader({"bad": "{# ok #}\n{% for %}\n"}))
    graph = meta.analyze_dependencies(env, "bad")
    assert graph.templates == {}
    error = graph.errors[0]
    assert error.name == "bad"
    assert error.stage == "parse"
    assert error.lineno == 2
    assert error.error_type == "TemplateSyntaxError"
    assert isinstance(error.reason, TemplateSyntaxError)
    assert "Expected an expression" in error.message
    assert graph.ok is False


def test_analyze_syntax_error_in_dependency_continues_walk():
    env = Environment(
        loader=DictLoader(
            {
                "page": ('{% include "broken" %}\n{% include "ok" %}\n'),
                "broken": "{% for %}",
                "ok": "ok",
            }
        )
    )
    graph = meta.analyze_dependencies(env, "page")
    assert [edge.target for edge in graph.edges] == ["broken", "ok"]
    assert graph.errors[0].name == "broken"
    assert list(graph.templates) == ["page", "ok"]


def test_analyze_requires_loader():
    with pytest.raises(TypeError):
        meta.analyze_dependencies(Environment(), "page")
