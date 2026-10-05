"""Tests for the block execution trace feature.

The block trace records which template's block was actually entered during
a render and which parent implementation super selected.  It is collected
during the actual render, so no loader or block is re-run to obtain it.

Three short template examples are used throughout:

* ``base.html`` -- the base template with a ``content`` block.
* ``middle.html`` -- extends ``base.html`` and calls ``super()``.
* ``child.html`` -- extends ``middle.html`` and calls ``super()``.

The expected super chain is ``child -> middle -> base``.

Run locally::

    pytest tests/test_block_trace.py -v
"""

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.runtime import BlockTrace

# -- Three short template examples -----------------------------------------

BASE_TEMPLATE = """\
{% block content %}base{% endblock %}"""

MIDDLE_TEMPLATE = """\
{% extends "base.html" %}\
{% block content %}middle {{ super() }}{% endblock %}"""

CHILD_TEMPLATE = """\
{% extends "middle.html" %}\
{% block content %}child {{ super() }}{% endblock %}"""

THREE_LEVEL_TEMPLATES = {
    "base.html": BASE_TEMPLATE,
    "middle.html": MIDDLE_TEMPLATE,
    "child.html": CHILD_TEMPLATE,
}


@pytest.fixture
def env():
    """Return an environment with block tracing enabled."""
    return Environment(
        loader=DictLoader(THREE_LEVEL_TEMPLATES),
        enable_block_trace=True,
    )


@pytest.fixture
def template(env):
    """Return the child template."""
    return env.get_template("child.html")


def test_three_level_super_chain(template):
    """Three-level inheritance calls super in the order child, middle, base.

    The output must be identical to a normal render, and the trace must
    record the full super chain.
    """
    trace = BlockTrace()
    output = template.render(block_trace=trace)

    # Output is the same as a normal render.
    assert output == "child middle base"

    # The trace records the enter and super events in order.
    assert trace.events == [
        ("enter", "child.html", "content"),
        ("super", "child.html", "content", "middle.html"),
        ("enter", "middle.html", "content"),
        ("super", "middle.html", "content", "base.html"),
        ("enter", "base.html", "content"),
    ]


def test_output_matches_normal_render(template):
    """The traced render produces the same output as a normal render."""
    normal_output = template.render()
    trace = BlockTrace()
    traced_output = template.render(block_trace=trace)

    assert traced_output == normal_output


def test_block_count_not_increased(template):
    """Tracing does not add extra block calls.

    The number of enter events must equal the number of blocks actually
    rendered (one per level in the inheritance chain).
    """
    trace = BlockTrace()
    template.render(block_trace=trace)

    enter_events = [e for e in trace.events if e[0] == "enter"]
    assert len(enter_events) == 3  # child, middle, base


def test_trace_is_per_render(template):
    """Each render gets an independent trace."""
    trace1 = BlockTrace()
    trace2 = BlockTrace()

    template.render(block_trace=trace1)
    template.render(block_trace=trace2)

    # Both traces have the same events but are independent objects.
    assert trace1 is not trace2
    assert trace1.events == trace2.events


def test_no_trace_when_disabled():
    """When tracing is disabled (default), no trace hooks are emitted.

    The generated code is identical to the original, so default rendering
    is completely unchanged.
    """
    env = Environment(loader=DictLoader(THREE_LEVEL_TEMPLATES))
    template = env.get_template("child.html")

    # The generated source must not contain trace hooks.
    source = env.loader.get_source(env, "child.html")[0]
    compiled = env.compile(source, raw=True)
    assert "_jinja_block_enter" not in compiled
    assert "_jinja_template" not in compiled

    # The output is still correct.
    assert template.render() == "child middle base"


def test_no_trace_without_block_trace(template):
    """When no BlockTrace is passed, the trace hooks are no-ops.

    The output is unchanged even though tracing is enabled in the
    environment.
    """
    output = template.render()
    assert output == "child middle base"


def test_include_keeps_own_source():
    """An included template keeps its own source in the trace.

    The include's blocks are recorded with the included template's name,
    and the include's context (vars) does not leak to the outer template.
    """
    templates = {
        "outer.html": (
            "{% block outer_block %}OUTER{% endblock %} {% include 'inner.html' %}"
        ),
        "inner.html": "{% block inner_block %}INNER{% endblock %}",
    }
    env = Environment(loader=DictLoader(templates), enable_block_trace=True)
    template = env.get_template("outer.html")

    trace = BlockTrace()
    output = template.render(block_trace=trace)

    assert output == "OUTER INNER"

    # The trace records both blocks with their own source templates.
    assert trace.events == [
        ("enter", "outer.html", "outer_block"),
        ("enter", "inner.html", "inner_block"),
    ]


def test_include_context_does_not_leak():
    """The include's independent Context source does not leak to the outer.

    Variables set inside the include must not appear in the outer context.
    """
    templates = {
        "outer.html": (
            "{% set outer_var = 'outer' %}{{ outer_var }} "
            "{% include 'inner.html' %} after:{{ outer_var }}"
        ),
        "inner.html": "{% set inner_var = 'inner' %}{{ inner_var }}",
    }
    env = Environment(loader=DictLoader(templates), enable_block_trace=True)
    template = env.get_template("outer.html")

    output = template.render()
    assert output == "outer inner after:outer"


def test_async_trace():
    """The trace works in async mode as well."""
    env = Environment(
        loader=DictLoader(THREE_LEVEL_TEMPLATES),
        enable_block_trace=True,
        enable_async=True,
    )
    template = env.get_template("child.html")

    import asyncio

    async def run():
        trace = BlockTrace()
        output = await template.render_async(block_trace=trace)
        return output, trace

    output, trace = asyncio.run(run())

    assert output == "child middle base"
    assert trace.events == [
        ("enter", "child.html", "content"),
        ("super", "child.html", "content", "middle.html"),
        ("enter", "middle.html", "content"),
        ("super", "middle.html", "content", "base.html"),
        ("enter", "base.html", "content"),
    ]


def test_trace_does_not_rerun_loader(template):
    """The trace is collected during the actual render.

    No loader or block is re-run to obtain the report.  This is verified
    by checking that the loader's get_source is called only once per
    template (during the initial load), not during the trace collection.
    """
    load_counts = {"base.html": 0, "middle.html": 0, "child.html": 0}

    class CountingLoader(DictLoader):
        def get_source(self, environment, template):
            load_counts[template] += 1
            return super().get_source(environment, template)

    env = Environment(
        loader=CountingLoader(THREE_LEVEL_TEMPLATES),
        enable_block_trace=True,
    )
    template = env.get_template("child.html")

    trace = BlockTrace()
    template.render(block_trace=trace)

    # Each template is loaded only once (during get_template).
    # The trace does not trigger any additional loads.
    assert load_counts == {"base.html": 1, "middle.html": 1, "child.html": 1}


def test_block_trace_events_format():
    """The trace events have a consistent format.

    Enter events are ``("enter", template_name, block_name)``.
    Super events are ``("super", current_template, block_name, parent_template)``.
    """
    trace = BlockTrace()
    assert trace.events == []

    trace.record_enter("tpl", "block")
    assert trace.events == [("enter", "tpl", "block")]

    trace.record_super("child", "block", "parent")
    assert trace.events == [
        ("enter", "tpl", "block"),
        ("super", "child", "block", "parent"),
    ]


def test_block_within_loop_is_traced():
    """Blocks called within loops are also traced.

    The derived context used for loop variables shares the per-render
    trace, so block enter events inside loops are recorded.
    """
    templates = {
        "base.html": "{% block content %}base{% endblock %}",
        "child.html": (
            "{% extends 'base.html' %}"
            "{% block content %}"
            "{% for i in range(3) %}{{ super() }}{% endfor %}"
            "{% endblock %}"
        ),
    }
    env = Environment(loader=DictLoader(templates), enable_block_trace=True)
    template = env.get_template("child.html")

    trace = BlockTrace()
    output = template.render(block_trace=trace)

    # super() is called 3 times in the loop, so 3 super events.
    super_events = [e for e in trace.events if e[0] == "super"]
    assert len(super_events) == 3

    # The output contains "base" 3 times.
    assert output.count("base") == 3
