from collections import Counter

import pytest

from jinja2 import Environment
from jinja2 import TemplateBatchError
from jinja2.bccache import Bucket
from jinja2.bccache import BytecodeCache
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound
from jinja2.exceptions import TemplateSyntaxError
from jinja2.loaders import DictLoader


class CountingBytecodeCache(BytecodeCache):
    """In-memory bytecode cache that counts cache hits and dumps per key."""

    def __init__(self):
        self.data = {}
        self.hits = Counter()
        self.dumps = Counter()

    def load_bytecode(self, bucket):
        if bucket.key in self.data:
            bucket.bytecode_from_string(self.data[bucket.key])
            if bucket.code is not None:
                self.hits[bucket.key] += 1

    def dump_bytecode(self, bucket):
        self.data[bucket.key] = bucket.bytecode_to_string()
        self.dumps[bucket.key] += 1

    def clear(self):
        self.data.clear()
        self.hits.clear()
        self.dumps.clear()


def make_env(sources):
    cache = CountingBytecodeCache()
    env = Environment(loader=DictLoader(sources), bytecode_cache=cache)
    return env, cache


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")


class TestPrepareTemplateBatch:
    def test_prepare_commit_batch(self):
        sources = {"a": "old a", "b": "b"}
        env, cache = make_env(sources)

        # cache the old a first
        assert env.get_template("a").render() == "old a"
        a_key = cache.get_cache_key("a")
        assert cache.hits[a_key] == 0
        assert cache.dumps[a_key] == 1

        # prepare new a together with a broken b
        sources["a"] = "new a"
        sources["b"] = "{% broken"
        with pytest.raises(TemplateSyntaxError):
            env.prepare_template_batch(["a", "b"])

        # the old a is still in the cache, b is not
        assert cache.get_bucket(env, "a", None, "old a").code is not None
        assert cache.get_bucket(env, "a", None, "new a").code is None
        assert cache.get_bucket(env, "b", None, "{% broken").code is None
        assert cache.dumps[a_key] == 1

        # fix b, prepare again and commit the whole batch
        sources["b"] = "fixed b"
        batch = env.prepare_template_batch(["a", "b"])
        batch.commit()

        # both templates are now cached with the new sources
        assert cache.get_bucket(env, "a", None, "new a").code is not None
        assert cache.get_bucket(env, "b", None, "fixed b").code is not None
        assert cache.dumps[a_key] == 2
        b_key = cache.get_cache_key("b")
        assert cache.dumps[b_key] == 1

        # rendering hits the cache, nothing is recompiled
        cache.hits.clear()
        assert env.get_template("a").render() == "new a"
        assert env.get_template("b").render() == "fixed b"
        assert cache.hits[a_key] == 1
        assert cache.hits[b_key] == 1

    def test_commit_rejects_changed_source_atomically(self):
        sources = {"a": "a", "b": "b"}
        env, cache = make_env(sources)
        env.get_template("a").render()  # populate the cache with old a

        batch = env.prepare_template_batch(["a", "b"])
        sources["a"] = "a changed after prepare"
        dumps_before = sum(cache.dumps.values())

        with pytest.raises(TemplateBatchError):
            batch.commit()

        # nothing was written, existing cache entries are untouched
        assert sum(cache.dumps.values()) == dumps_before
        assert cache.get_bucket(env, "a", None, "a").code is not None
        assert cache.get_bucket(env, "b", None, "b").code is None
        # rendering still works
        assert env.get_template("a").render() == "a changed after prepare"

    def test_commit_rejects_changed_config(self):
        sources = {"a": "a"}
        env, cache = make_env(sources)
        dumps_before = sum(cache.dumps.values())

        batch = env.prepare_template_batch(["a"])
        env.optimized = False

        with pytest.raises(TemplateBatchError):
            batch.commit()

        assert sum(cache.dumps.values()) == dumps_before
        assert cache.get_bucket(env, "a", None, "a").code is None

    def test_prepare_does_not_execute_templates(self):
        calls = []
        sources = {"a": "{{ counter() }}"}
        env, cache = make_env(sources)
        env.globals["counter"] = lambda: calls.append(1) or "x"

        batch = env.prepare_template_batch(["a"])
        batch.commit()
        assert calls == []

        assert env.get_template("a").render() == "x"
        assert len(calls) == 1

    def test_prepare_follows_no_dependencies(self):
        sources = {"a": "{% include 'c' %}A", "c": "C"}
        env, cache = make_env(sources)

        batch = env.prepare_template_batch(["a"])
        batch.commit()

        assert cache.get_bucket(env, "a", None, sources["a"]).code is not None
        assert cache.get_bucket(env, "c", None, sources["c"]).code is None

    def test_commit_empty_batch(self):
        env, _ = make_env({"a": "a"})
        batch = env.prepare_template_batch([])
        with pytest.raises(TemplateBatchError):
            batch.commit()

    def test_commit_twice(self):
        env, _ = make_env({"a": "a"})
        batch = env.prepare_template_batch(["a"])
        batch.commit()
        with pytest.raises(TemplateBatchError):
            batch.commit()

    def test_prepare_requires_explicit_names(self):
        env, _ = make_env({"a": "a"})
        with pytest.raises(TypeError):
            env.prepare_template_batch("a")

    def test_environment_requires_bytecode_cache(self):
        env = Environment(loader=DictLoader({"a": "a"}))
        with pytest.raises(RuntimeError):
            env.prepare_template_batch(["a"])


class MockMemcached:
    class Error(Exception):
        pass

    key = None
    value = None
    timeout = None

    def get(self, key):
        return self.value

    def set(self, key, value, timeout=None):
        self.key = key
        self.value = value
        self.timeout = timeout

    def get_side_effect(self, key):
        raise self.Error()

    def set_side_effect(self, *args):
        raise self.Error()


class TestMemcachedBytecodeCache:
    def test_dump_load(self):
        memcached = MockMemcached()
        m = MemcachedBytecodeCache(memcached)

        b = Bucket(None, "key", "")
        b.code = "code"
        m.dump_bytecode(b)
        assert memcached.key == "jinja2/bytecode/key"

        b = Bucket(None, "key", "")
        m.load_bytecode(b)
        assert b.code == "code"

    def test_exception(self):
        memcached = MockMemcached()
        memcached.get = memcached.get_side_effect
        memcached.set = memcached.set_side_effect
        m = MemcachedBytecodeCache(memcached)
        b = Bucket(None, "key", "")
        b.code = "code"

        m.dump_bytecode(b)
        m.load_bytecode(b)

        m.ignore_memcache_errors = False

        with pytest.raises(MockMemcached.Error):
            m.dump_bytecode(b)

        with pytest.raises(MockMemcached.Error):
            m.load_bytecode(b)
