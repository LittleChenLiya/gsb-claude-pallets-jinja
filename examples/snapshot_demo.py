"""Demonstrate atomic, versioned template publishing with SnapshotLoader.

Run from the repository root:

    python examples/snapshot_demo.py

The demo finishes in a few seconds. It prints every publish version and
the rendered results, showing that an in-flight render keeps its snapshot
while new renders use the freshly published one, and that a failed publish
leaves the previous version untouched.
"""

import os
import sys
import threading

# Make the demo work from a source checkout without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jinja2 import Environment  # noqa: E402
from jinja2 import SnapshotLoader  # noqa: E402
from jinja2.exceptions import TemplateSyntaxError  # noqa: E402


def page_source():
    return (
        '{% extends "layout.html" %}'
        '{% block content %}{{ wait() }}{% include "item.html" %}{% endblock %}'
    )


def mapping(layout, item):
    return {
        "layout.html": layout + "/{% block content %}{% endblock %}",
        "page.html": page_source(),
        "item.html": item,
    }


def main():
    loader = SnapshotLoader(mapping("old", "old"))
    env = Environment(loader=loader)

    armed = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    def wait():
        if armed.is_set():
            entered.set()
            release.wait(10)
        return ""

    env.globals["wait"] = wait

    print(f"current version: {loader.current_version}")
    print(f"render v{loader.current_version}: {env.get_template('page.html').render()}")

    # Block an in-flight render inside the page, publish a new version
    # while it is stuck, then let it finish.
    armed.set()
    result = []

    def in_flight_render():
        result.append(env.get_template("page.html").render())

    thread = threading.Thread(target=in_flight_render)
    thread.start()
    entered.wait(10)

    version = loader.publish(mapping("new", "new"), env)
    print(f"published version: {version}")

    release.set()
    thread.join(10)
    print(f"in-flight render (started on v1): {result[0]}")
    print(f"new render after publish: {env.get_template('page.html').render()}")

    # A failed publish validates first, raises, and consumes no version.
    broken = mapping("broken", "broken")
    broken["item.html"] = "{% if %}"

    try:
        loader.publish(broken, env)
    except TemplateSyntaxError as e:
        print(f"failed publish rejected: {e.message!r}")

    print(f"current version after failed publish: {loader.current_version}")
    print(f"queryable versions: {loader.versions()}")
    print(f"render after failed publish: {env.get_template('page.html').render()}")

    version = loader.publish(mapping("final", "final"), env)
    print(f"published version: {version}")
    print(f"render v{version}: {env.get_template('page.html').render()}")
    print("DEMO OK")


if __name__ == "__main__":
    main()
