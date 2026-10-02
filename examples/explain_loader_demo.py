"""Reproducible demo of the read-only loader explanation API.

Run from the repository root with the local source on the path:

    PYTHONPATH=src python examples/explain_loader_demo.py

It prints, for a few template names, which sources would be tried, in
which order, which are shadowed, and which one finally wins -- all
without rendering anything and without calling ``get_source`` on
loaders that cannot be inspected read-only.
"""

from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import explain_load
from jinja2 import PrefixLoader
from jinja2.loaders import BaseLoader


class CountingLoader(BaseLoader):
    """A custom loader that cannot be inspected read-only.

    It counts ``get_source`` calls and raises if called, proving that
    explaining leaves it untouched unless an adapter is provided.
    """

    def __init__(self):
        self.calls = 0

    def get_source(self, environment, template):
        self.calls += 1
        raise RuntimeError("must not be called by explain")


def show(title, explanation):
    print(f"== {title} ==")
    print(explanation.format())
    print()


def main():
    counting = CountingLoader()

    loader = ChoiceLoader(
        [
            PrefixLoader(
                {
                    "a": ChoiceLoader(
                        [
                            DictLoader({}),
                            DictLoader({"x.html": "nested a"}),
                        ]
                    ),
                    "b": DictLoader({"x.html": "branch b"}),
                }
            ),
            DictLoader({"x.html": "first dict", "a/x.html": "top-level"}),
            DictLoader({"x.html": "second dict"}),
            counting,
        ]
    )

    # Two DictLoaders contain x.html: only the first is a hit, the
    # second is shadowed. The custom loader has no adapter, so it is
    # reported as unsupported and never called.
    show("shadowing in a ChoiceLoader", explain_load(loader, "x.html"))

    # a/x.html only enters the 'a' branch; the stripped name x.html is
    # never applied to branch 'b'.
    show("prefix routing", explain_load(loader, "a/x.html"))

    # A miss keeps every attempted path.
    show("miss keeps attempted paths", explain_load(loader, "a/nope.html"))

    # With an explicit explain adapter, the custom loader participates
    # -- still without get_source being called.
    adapters = {counting: lambda loader, name: name == "x.html"}
    show(
        "custom loader with an explain adapter",
        explain_load(loader, "x.html", adapters=adapters),
    )

    assert counting.calls == 0, "get_source must never be called"
    print(f"custom loader get_source calls: {counting.calls}")


if __name__ == "__main__":
    main()
