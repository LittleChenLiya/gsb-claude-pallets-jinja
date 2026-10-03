"""Warm up the template caches for an explicit batch, atomically.

``Environment.prepare_template_batch`` compiles named templates into a
private staging area. The batch only reaches the environment caches when
``PreparedTemplateBatch.commit`` succeeds; if one template's source or
the environment configuration changed in the meantime, the whole batch
is rejected and existing cache entries stay untouched. Templates are
never executed while preparing or committing, so global functions and
render expressions are not called.

Run it directly to see the cache state after each step::

    python prepare_template_batch.py
"""

import weakref

from jinja2 import BytecodeCache
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import TemplateBatchError
from jinja2 import TemplateSyntaxError


class MemoryBytecodeCache(BytecodeCache):
    """Minimal bytecode cache used here to print its state."""

    def __init__(self):
        self.storage = {}

    def load_bytecode(self, bucket):
        data = self.storage.get(bucket.key)

        if data is not None:
            bucket.bytecode_from_string(data)

    def dump_bytecode(self, bucket):
        self.storage[bucket.key] = bucket.bytecode_to_string()


class CountingLoader(DictLoader):
    """Like DictLoader, but counts how often sources are requested."""

    def __init__(self, mapping):
        super().__init__(mapping)
        self.loads = 0

    def get_source(self, environment, template):
        self.loads += 1
        return super().get_source(environment, template)


def cached_template(env, name):
    if env.cache is None:
        return None

    return env.cache.get((weakref.ref(env.loader), name))


def print_cache_state(env, bcc, label):
    names = sorted(name for _ref, name in env.cache)
    print(f"{label}: memory={names} bytecode={sorted(bcc.storage)}")


def main():
    calls = []

    def counted_global():
        # Global functions must not run during prepare/commit.
        calls.append(1)
        return str(len(calls))

    sources = {
        "a": "a calls global: {{ counted_global() }}",
        "b": "broken: {% for %}",  # syntax error on purpose
    }
    loader = CountingLoader(sources)
    bcc = MemoryBytecodeCache()
    env = Environment(loader=loader, bytecode_cache=bcc)
    env.globals["counted_global"] = counted_global

    print("1) Cache the old version of 'a'.")
    print(env.get_template("a").render())
    print_cache_state(env, bcc, "   state")

    print("\n2) Prepare new 'a' together with syntactically broken 'b'.")
    sources["a"] = "new a calls global: {{ counted_global() }}"

    try:
        env.prepare_template_batch(["a", "b"])
    except TemplateSyntaxError as e:
        print(f"   prepare failed as expected: {e.message}")

    print_cache_state(env, bcc, "   state")
    untouched = cached_template(env, "a")
    print("   old 'a' still in cache and renders:", untouched.render())
    print("   'b' in memory cache:", cached_template(env, "b") is not None)
    print("   global call count:", len(calls))

    print("\n3) Fix 'b' and prepare + commit both together.")
    sources["b"] = "b calls global: {{ counted_global() }}"
    batch = env.prepare_template_batch(["a", "b"])
    print_cache_state(env, bcc, "   after prepare")
    batch.commit()
    print_cache_state(env, bcc, "   after commit ")
    print("   global call count:", len(calls))
    print("   render a:", env.get_template("a").render())
    print("   render b:", env.get_template("b").render())
    print("   global call count after rendering:", len(calls))

    print("\n4) Change 'a' between prepare and commit; whole batch rejected.")
    before = dict(bcc.storage)
    batch = env.prepare_template_batch(["a", "b"])
    sources["a"] = "changed again: {{ counted_global() }}"

    try:
        batch.commit()
    except TemplateBatchError as e:
        print("   commit rejected:", e.message)
        print("   stale names:", e.stale)

    print_cache_state(env, bcc, "   state")
    print("   bytecode cache unchanged:", bcc.storage == before)


if __name__ == "__main__":
    main()
