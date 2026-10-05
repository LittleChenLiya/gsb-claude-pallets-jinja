"""Example: block execution trace with three-level inheritance.

This example demonstrates how to use the block trace feature to record
which template's block was actually entered and which parent implementation
super selected.

Three short templates are used:

* ``base.html`` -- the base template with a ``content`` block.
* ``middle.html`` -- extends ``base.html`` and calls ``super()``.
* ``child.html`` -- extends ``middle.html`` and calls ``super()``.

The expected super chain is ``child -> middle -> base``.

Run locally::

    python examples/block_trace_example.py
"""

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.runtime import BlockTrace

# Three short template examples
TEMPLATES = {
    "base.html": "{% block content %}base{% endblock %}",
    "middle.html": (
        '{% extends "base.html" %}{% block content %}middle {{ super() }}{% endblock %}'
    ),
    "child.html": (
        '{% extends "middle.html" %}'
        "{% block content %}child {{ super() }}{% endblock %}"
    ),
}


def main() -> None:
    # Create an environment with block tracing enabled
    env = Environment(loader=DictLoader(TEMPLATES), enable_block_trace=True)
    template = env.get_template("child.html")

    # Create a trace and render
    trace = BlockTrace()
    output = template.render(block_trace=trace)

    print("Output:", repr(output))
    print()
    print("Block execution trace:")
    for event in trace.events:
        print("  ", event)

    # Verify the super chain
    super_events = [e for e in trace.events if e[0] == "super"]
    chain = [e[1] for e in super_events]
    print()
    print("Super chain:", " -> ".join(chain))
    assert chain == ["child.html", "middle.html"]


if __name__ == "__main__":
    main()
