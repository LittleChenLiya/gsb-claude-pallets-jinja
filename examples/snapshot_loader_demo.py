"""Demonstrates atomic, versioned template publishing with SnapshotLoader.

An old render is paused mid-render with a barrier, a new snapshot version
is published, and the old render still finishes with old/old while new
renders produce new/new.  A failing publish is rejected without consuming
a version.  Runs in well under 90 seconds:

    python examples/snapshot_loader_demo.py
"""

import threading

from jinja2 import Environment
from jinja2 import SnapshotLoader
from jinja2.exceptions import TemplateSyntaxError


def main() -> None:
    loader = SnapshotLoader()
    env = Environment(loader=loader)

    entered = threading.Event()
    release = threading.Event()

    def wait() -> str:
        # Pause the render here so the publish happens mid-render.
        entered.set()
        release.wait(timeout=10)
        return ""

    env.globals["wait"] = wait

    v1 = loader.publish(
        {
            "page": "old/{{ wait() }}{% include 'item' %}",
            "item": "old",
        }
    )
    print(f"[publish] version {v1} active")

    # Start rendering the old version in the background and pause it
    # inside the template, before its include is resolved.
    template = env.get_template("page")
    outcome: list[str] = []
    thread = threading.Thread(target=lambda: outcome.append(template.render()))
    thread.start()
    entered.wait(timeout=10)

    v2 = loader.publish(
        {
            "page": "new/{% include 'item' %}",
            "item": "new",
        }
    )
    print(f"[publish] version {v2} active while a version {v1} render is in flight")
    release.set()
    thread.join(timeout=10)

    print(f"[render] in-flight render, pinned to version {v1}: {outcome[0]!r}")
    new_result = env.get_template("page").render()
    print(f"[render] new request, using version {v2}: {new_result!r}")

    try:
        loader.publish({"broken": "{% if %}"})
    except TemplateSyntaxError as e:
        print(f"[publish] rejected invalid snapshot: {e.message!r}")

    print(
        f"[status] current version is still {loader.version},"
        f" published versions: {loader.versions}"
    )
    print(f"[render] after failed publish: {env.get_template('page').render()!r}")


if __name__ == "__main__":
    main()
