import pytest

from jinja2 import Environment
from jinja2.bccache import Bucket
from jinja2.bccache import BytecodeCache
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.bccache import MemcachedBytecodeCache
from jinja2.exceptions import TemplateNotFound
from jinja2.ext import Extension
from jinja2.loaders import DictLoader


@pytest.fixture
def env(package_loader, tmp_path):
    bytecode_cache = FileSystemBytecodeCache(str(tmp_path))
    return Environment(loader=package_loader, bytecode_cache=bytecode_cache)


class TestByteCodeCache:
    def test_simple(self, env):
        tmpl = env.get_template("test.html")
        assert tmpl.render().strip() == "BAR"
        pytest.raises(TemplateNotFound, env.get_template, "missing.html")


class CountingBytecodeCache(BytecodeCache):
    """In memory bytecode cache recording load/dump calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = {}
        self.loads = 0
        self.dumps = 0

    def load_bytecode(self, bucket):
        self.loads += 1
        data = self.storage.get(bucket.key)
        if data is not None:
            bucket.bytecode_from_string(data)

    def dump_bytecode(self, bucket):
        self.dumps += 1
        self.storage[bucket.key] = bucket.bytecode_to_string()


class CountingEnvironment(Environment):
    """Environment counting the number of actual compilations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_count = 0

    def compile(self, source, name=None, filename=None, raw=False, defer_init=False):
        if not raw:
            self.compile_count += 1
        return super().compile(source, name, filename, raw, defer_init)


TEMPLATE_NAME = "index.html"
TEMPLATE_SOURCE = "{{ value }}"


def make_env(cache, **kwargs):
    kwargs.setdefault("cache_size", 0)
    return CountingEnvironment(
        loader=DictLoader({TEMPLATE_NAME: TEMPLATE_SOURCE}),
        bytecode_cache=cache,
        **kwargs,
    )


def fingerprint_cache(**kwargs):
    return CountingBytecodeCache(use_environment_fingerprint=True, **kwargs)


class FingerprintExtA(Extension):
    pass


class FingerprintExtB(Extension):
    pass


class TestEnvironmentFingerprint:
    def test_sync_and_async_are_isolated(self):
        cache = fingerprint_cache()
        sync_env = make_env(cache)
        async_env = make_env(cache, enable_async=True)

        sync_tmpl = sync_env.get_template(TEMPLATE_NAME)
        async_tmpl = async_env.get_template(TEMPLATE_NAME)

        assert sync_env.compile_count == 1
        assert async_env.compile_count == 1
        assert cache.dumps == 2
        assert cache.loads == 2
        assert sync_tmpl.render(value="x") == "x"
        assert async_tmpl.render(value="x") == "x"

        sync_env2 = make_env(cache)
        async_env2 = make_env(cache, enable_async=True)
        assert sync_env2.get_template(TEMPLATE_NAME).render(value="y") == "y"
        assert async_env2.get_template(TEMPLATE_NAME).render(value="y") == "y"
        assert sync_env2.compile_count == 0
        assert async_env2.compile_count == 0

    def test_delimiters_are_isolated(self):
        cache = fingerprint_cache()
        source = "value = [[ value ]]"
        loader = DictLoader({TEMPLATE_NAME: source})

        default_env = CountingEnvironment(
            loader=loader,
            bytecode_cache=cache,
            cache_size=0,
        )
        changed_env = CountingEnvironment(
            loader=loader,
            bytecode_cache=cache,
            cache_size=0,
            variable_start_string="[[",
            variable_end_string="]]",
        )

        assert default_env.get_template(TEMPLATE_NAME).render(value=1) == (
            "value = [[ value ]]"
        )
        assert changed_env.get_template(TEMPLATE_NAME).render(value=1) == ("value = 1")
        assert default_env.compile_count == 1
        assert changed_env.compile_count == 1

        default_env2 = CountingEnvironment(
            loader=loader, bytecode_cache=cache, cache_size=0
        )
        changed_env2 = CountingEnvironment(
            loader=loader,
            bytecode_cache=cache,
            cache_size=0,
            variable_start_string="[[",
            variable_end_string="]]",
        )
        assert default_env2.get_template(TEMPLATE_NAME).render(value=2) == (
            "value = [[ value ]]"
        )
        assert changed_env2.get_template(TEMPLATE_NAME).render(value=2) == ("value = 2")
        assert default_env2.compile_count == 0
        assert changed_env2.compile_count == 0

    def test_same_configuration_hits_cache(self):
        cache = fingerprint_cache()
        first = make_env(cache, trim_blocks=True)
        assert first.get_template(TEMPLATE_NAME).render(value="a") == "a"
        assert first.compile_count == 1

        second = make_env(cache, trim_blocks=True)
        assert second.get_template(TEMPLATE_NAME).render(value="b") == "b"
        assert second.compile_count == 0
        assert cache.dumps == 1
        assert cache.loads == 2

    def test_overlay_hits_and_invalidates(self):
        cache = fingerprint_cache()
        base = make_env(cache)
        assert base.get_template(TEMPLATE_NAME).render(value="a") == "a"
        assert base.compile_count == 1

        same_overlay = base.overlay()
        assert same_overlay.get_template(TEMPLATE_NAME).render(value="b") == "b"
        assert cache.dumps == 1

        changed_overlay = base.overlay(
            variable_start_string="[[", variable_end_string="]]"
        )
        source = "value = [[ value ]]"
        changed_overlay.loader = DictLoader({TEMPLATE_NAME: source})
        assert changed_overlay.get_template(TEMPLATE_NAME).render(value=2) == (
            "value = 2"
        )
        assert cache.dumps == 2

    def test_extension_order_is_significant(self):
        cache = fingerprint_cache()
        first = make_env(cache, extensions=(FingerprintExtA, FingerprintExtB))
        assert first.get_template(TEMPLATE_NAME).render(value="a") == "a"
        assert first.compile_count == 1

        same_order = make_env(cache, extensions=(FingerprintExtA, FingerprintExtB))
        assert same_order.get_template(TEMPLATE_NAME).render(value="b") == "b"
        assert same_order.compile_count == 0

        reversed_order = make_env(cache, extensions=(FingerprintExtB, FingerprintExtA))
        assert reversed_order.get_template(TEMPLATE_NAME).render(value="c") == "c"
        assert reversed_order.compile_count == 1
        assert cache.dumps == 2

    def test_extension_classes_change_fingerprint(self):
        cache = fingerprint_cache()
        first = make_env(cache, extensions=(FingerprintExtA,))
        first.get_template(TEMPLATE_NAME)
        second = make_env(cache, extensions=(FingerprintExtB,))
        second.get_template(TEMPLATE_NAME)
        assert second.compile_count == 1
        assert cache.dumps == 2

    def test_compile_version_change_invalidates(self):
        cache = fingerprint_cache(compile_version="v1")
        first = make_env(cache)
        assert first.get_template(TEMPLATE_NAME).render(value="a") == "a"
        assert first.compile_count == 1

        same = make_env(cache)
        assert same.get_template(TEMPLATE_NAME).render(value="b") == "b"
        assert same.compile_count == 0

        cache.compile_version = "v2"
        upgraded = make_env(cache)
        assert upgraded.get_template(TEMPLATE_NAME).render(value="c") == "c"
        assert upgraded.compile_count == 1
        assert cache.dumps == 2

    def test_compile_version_types(self, tmp_path):
        versions = (1, "v1", b"v1")
        fingerprints = []
        for version in versions:
            cache = FileSystemBytecodeCache(
                str(tmp_path),
                "__jinja2_%s.cache",
                use_environment_fingerprint=True,
                compile_version=version,
            )
            fingerprints.append(cache.get_environment_fingerprint(make_env(cache)))
        assert fingerprints[0] != fingerprints[1]
        assert fingerprints[1] != fingerprints[2]

        invalid_cache = fingerprint_cache(compile_version=["invalid"])
        with pytest.raises(TypeError):
            invalid_cache.get_environment_fingerprint(make_env(invalid_cache))

    def test_fingerprint_is_stable_and_canonical(self):
        cache = fingerprint_cache()
        first = cache.get_environment_fingerprint(make_env(cache))
        second = cache.get_environment_fingerprint(make_env(cache))
        assert first == second
        assert isinstance(first, str)
        assert all(char in "0123456789abcdef" for char in first)

        other = CountingBytecodeCache()
        assert not other.use_environment_fingerprint

    def test_legacy_entries_are_ignored(self, tmp_path):
        source_loader = DictLoader({TEMPLATE_NAME: TEMPLATE_SOURCE})
        legacy = FileSystemBytecodeCache(str(tmp_path))
        legacy_env = CountingEnvironment(
            loader=source_loader,
            bytecode_cache=legacy,
            cache_size=0,
        )
        assert legacy_env.get_template(TEMPLATE_NAME).render(value=1) == "1"
        assert legacy_env.compile_count == 1

        fingerprinted = FileSystemBytecodeCache(
            str(tmp_path), use_environment_fingerprint=True
        )
        fingerprint_env = CountingEnvironment(
            loader=source_loader,
            bytecode_cache=fingerprinted,
            cache_size=0,
        )
        assert fingerprint_env.get_template(TEMPLATE_NAME).render(value=2) == "2"
        assert fingerprint_env.compile_count == 1

        again = CountingEnvironment(
            loader=source_loader,
            bytecode_cache=fingerprinted,
            cache_size=0,
        )
        assert again.get_template(TEMPLATE_NAME).render(value=3) == "3"
        assert again.compile_count == 0

    def test_corrupt_entry_is_ignored(self):
        cache = fingerprint_cache()
        env = make_env(cache)
        cache.storage[cache.get_cache_key(TEMPLATE_NAME, None, env)] = b""
        assert env.get_template(TEMPLATE_NAME).render(value=1) == "1"
        assert env.compile_count == 1
        assert cache.dumps == 1

        cache.storage[cache.get_cache_key(TEMPLATE_NAME, None, env)] = (
            b"not a cache entry"
        )
        other = make_env(cache)
        assert other.get_template(TEMPLATE_NAME).render(value=2) == "2"
        assert other.compile_count == 1

    def test_corrupt_filesystem_entry_is_ignored(self, tmp_path):
        fs_cache = FileSystemBytecodeCache(
            str(tmp_path), use_environment_fingerprint=True
        )
        env = make_env(fs_cache)
        key = fs_cache.get_cache_key(TEMPLATE_NAME, None, env)
        (tmp_path / f"__jinja2_{key}.cache").write_bytes(b"j2" + b"\x00garbage")
        assert env.get_template(TEMPLATE_NAME).render(value=1) == "1"
        assert env.compile_count == 1


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
