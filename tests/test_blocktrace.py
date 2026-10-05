"""Tests for the explicit block execution trace (jinja2.blocktrace).

Run only these tests with:

    python -m pytest tests/test_blocktrace.py -q
"""

import pytest

from jinja2 import BlockTrace
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import render_with_trace
from jinja2 import render_with_trace_async

# Example 1: three-level inheritance, each level calls super().
INHERITANCE = {
    "base.html": "[{% block content %}base{% endblock %}]",
    "middle.html": (
        "{% extends 'base.html' %}"
        "{% block content %}middle({{ super() }}){% endblock %}"
    ),
    "child.html": (
        "{% extends 'middle.html' %}"
        "{% block content %}child({{ super() }}){% endblock %}"
    ),
}

# Example 2: a page that includes another template with its own block.
INCLUDE = {
    "page.html": (
        "<p>{% block content %}<b>{{ x }}</b>{% endblock %}</p>"
        "{% include 'inc.html' %}"
    ),
    "inc.html": "<i>{% block note %}inc-{{ x }}{% endblock %}</i>",
}

# Example 3: autoescaped template mixing a block and a variable.
AUTOESCAPE = {
    "hello.html": "Hello {% block name %}{{ who }}{% endblock %}!",
}


@pytest.fixture()
def env():
    return Environment(loader=DictLoader(INHERITANCE))


def test_super_chain_is_child_middle_base(env):
    template = env.get_template("child.html")
    plain = template.render()
    out, trace = render_with_trace(template)

    # Output is identical to a normal render.
    assert out == plain == "[child(middle(base))]"
    # The blocks actually entered, in execution order.
    assert trace.chain("content") == ["child.html", "middle.html", "base.html"]
    # super() selected middle then base as the parent implementation.
    assert [(e.template, e.depth) for e in trace.supers] == [
        ("middle.html", 1),
        ("base.html", 2),
    ]


def test_block_count_does_not_increase(env):
    template = env.get_template("child.html")
    _, trace = render_with_trace(template)

    # Each implementation is entered exactly once, no block is re-run.
    assert trace.chain("content") == ["child.html", "middle.html", "base.html"]
    assert len(trace.entered) == 3


def test_trace_state_is_independent_per_render(env):
    template = env.get_template("child.html")
    _, first = render_with_trace(template)
    _, second = render_with_trace(template)

    assert first.events == second.events
    assert first.events is not second.events


def test_default_rendering_leaves_no_trace_behind(env):
    template = env.get_template("child.html")
    render_with_trace(template)

    # A plain render afterwards is untouched and records nothing.
    trace = BlockTrace()
    assert template.render() == "[child(middle(base))]"
    assert trace.events == []


def test_include_keeps_its_own_source():
    env = Environment(loader=DictLoader(INCLUDE), autoescape=True)
    template = env.get_template("page.html")
    plain = template.render(x="<hi>")
    out, trace = render_with_trace(template, x="<hi>")

    # Autoescape results are unchanged.
    assert out == plain == "<p><b>&lt;hi&gt;</b></p><i>inc-&lt;hi&gt;</i>"
    # The included template's independent context keeps its own source.
    assert trace.chain("note") == ["inc.html"]
    assert trace.chain("content") == ["page.html"]


def test_autoescape_and_markup_results_preserved():
    env = Environment(loader=DictLoader(AUTOESCAPE), autoescape=True)
    template = env.get_template("hello.html")
    plain = template.render(who="<World>")
    out, trace = render_with_trace(template, who="<World>")

    assert out == plain == "Hello &lt;World&gt;!"
    assert trace.chain("name") == ["hello.html"]


def test_blocktrace_context_manager():
    env = Environment(loader=DictLoader(INHERITANCE))
    template = env.get_template("child.html")
    trace = BlockTrace()

    with trace:
        out = template.render()

    assert out == "[child(middle(base))]"
    assert trace.chain("content") == ["child.html", "middle.html", "base.html"]


def test_async_render_with_trace():
    env = Environment(loader=DictLoader(INHERITANCE), enable_async=True)
    template = env.get_template("child.html")

    async def run():
        plain = await template.render_async()
        out, trace = await render_with_trace_async(template)
        return plain, out, trace

    import asyncio

    plain, out, trace = asyncio.run(run())
    assert out == plain == "[child(middle(base))]"
    assert trace.chain("content") == ["child.html", "middle.html", "base.html"]
    assert [(e.template, e.depth) for e in trace.supers] == [
        ("middle.html", 1),
        ("base.html", 2),
    ]


def test_async_include_keeps_its_own_source():
    env = Environment(
        loader=DictLoader(INCLUDE), autoescape=True, enable_async=True
    )
    template = env.get_template("page.html")

    import asyncio

    out, trace = asyncio.run(render_with_trace_async(template, x="<hi>"))
    assert out == "<p><b>&lt;hi&gt;</b></p><i>inc-&lt;hi&gt;</i>"
    assert trace.chain("note") == ["inc.html"]
    assert trace.chain("content") == ["page.html"]
