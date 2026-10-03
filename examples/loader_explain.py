"""Demonstrate the loader explain API.

Run with::

    python examples/loader_explain.py

The explain API shows which sources a loader combination would try for a
template name, without rendering the template.  It preserves the order in
which loaders are tried, marks loaders that are shadowed by an earlier
hit, and records the final hit.  Composite loaders are recursed into.

Custom loaders are never inspected by calling their ``get_source`` method
(which may have side effects).  They are reported as ``unsupported``
unless a :class:`~jinja2.LoaderExplainAdapter` is registered for them.
"""

from jinja2 import BaseLoader
from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import EXPLAIN_HIT
from jinja2 import explain_loader
from jinja2 import EXPLAIN_MISS
from jinja2 import LoaderExplainAdapter
from jinja2 import LoaderTry
from jinja2 import PrefixLoader

env = Environment()


def show(title, loader, name, **kwargs):
    print(f"== {title} ==")
    print(f"loader: {type(loader).__name__}, name: {name!r}")
    print(explain_loader(loader, env, name, **kwargs).render())
    print()


# 1. Two DictLoaders both contain "x.html".  The first one wins, the
#    second is shadowed and never consulted.
show(
    "shadowing in a ChoiceLoader",
    ChoiceLoader([DictLoader({"x.html": "A"}), DictLoader({"x.html": "B"})]),
    "x.html",
)

# 2. A PrefixLoader only enters the matching branch and strips the
#    prefix before delegating.  The stripped name is not tried in any
#    other branch.
show(
    "prefix stripping in a PrefixLoader",
    PrefixLoader({"a": DictLoader({"x.html": "A"}), "b": DictLoader({"y.html": "B"})}),
    "a/x.html",
)

# 3. When the name is missing, the tried paths are preserved.
show(
    "a miss keeps the tried paths",
    ChoiceLoader([DictLoader({}), DictLoader({})]),
    "missing.html",
)

# 4. Nested composites are recursed into.
show(
    "nested composite loaders",
    ChoiceLoader(
        [
            PrefixLoader({"a": DictLoader({"x.html": "A"})}),
            DictLoader({"x.html": "B"}),
        ]
    ),
    "a/x.html",
)


# 5. A custom loader is not called without an adapter.  The counter
#    stays at 0 even though get_source would raise.
class CountingLoader(BaseLoader):
    def __init__(self):
        self.calls = 0

    def get_source(self, environment, template):
        self.calls += 1
        raise RuntimeError("boom")


custom = CountingLoader()
show(
    "custom loader without an adapter",
    ChoiceLoader([custom, DictLoader({"x.html": "A"})]),
    "x.html",
)
print(f"get_source calls after explain: {custom.calls}")
print()


# 6. An adapter lets the explain API inspect a custom loader without
#    calling get_source.
class CountingAdapter(LoaderExplainAdapter):
    def explain(self, loader, environment, template):
        if template in loader.templates:
            return LoaderTry(loader, template, EXPLAIN_HIT)
        return LoaderTry(loader, template, EXPLAIN_MISS)


class KnownLoader(BaseLoader):
    templates = {"x.html": "A"}

    def get_source(self, environment, template):
        raise RuntimeError("must not be called")


show(
    "custom loader with an adapter",
    ChoiceLoader([KnownLoader(), DictLoader({"y.html": "B"})]),
    "x.html",
    adapters={KnownLoader: CountingAdapter()},
)
