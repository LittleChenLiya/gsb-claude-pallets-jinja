from __future__ import annotations

import asyncio
import hashlib
import inspect
import pickle
import typing as t

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.bccache import bc_magic
from jinja2.bccache import BytecodeCache
from jinja2.bccache import FingerprintedBytecodeCache
from jinja2.ext import Extension


class MemoryBytecodeCache(BytecodeCache):
    """Minimal shared backing store that records the keys it uses."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.loaded_keys: list[str] = []
        self.dumped_keys: list[str] = []

    def load_bytecode(self, bucket) -> None:
        self.loaded_keys.append(bucket.key)
        data = self.store.get(bucket.key)
        if data is not None:
            bucket.bytecode_from_string(data)

    def dump_bytecode(self, bucket) -> None:
        self.dumped_keys.append(bucket.key)
        self.store[bucket.key] = bucket.bytecode_to_string()

    def clear(self) -> None:
        self.store.clear()


class CountingEnvironment(Environment):
    """Environment that counts real calls to the Python ``compile`` builtin."""

    compile_count = 0

    def _compile(self, source: str, filename: str):
        type(self).compile_count += 1
        return super()._compile(source, filename)


class DummyExtension(Extension):
    pass


class OtherExtension(Extension):
    pass


@pytest.fixture
def loader():
    return DictLoader(
        {
            "template": "{% for item in items %}{{ item }}|{% endfor %}",
            "square": "[= item =]",
        }
    )


@pytest.fixture
def backing():
    return MemoryBytecodeCache()


@pytest.fixture(autouse=True)
def reset_compile_count():
    CountingEnvironment.compile_count = 0
    yield
    CountingEnvironment.compile_count = 0


def make_environment(
    backing: MemoryBytecodeCache,
    loader: DictLoader,
    *,
    enable_async: bool = False,
    compile_version: str | int | None = None,
    **kwargs: t.Any,
) -> CountingEnvironment:
    return CountingEnvironment(
        loader=loader,
        # cache_size=0 disables the in-memory template cache, so every
        # get_template() goes through the bytecode cache again.
        cache_size=0,
        bytecode_cache=FingerprintedBytecodeCache(backing),
        enable_async=enable_async,
        compile_version=compile_version,
        **kwargs,
    )


class TestEnvironmentFingerprint:
    def test_stable_for_same_configuration(self, backing):
        # Two independently constructed environments with identical
        # configuration must produce identical fingerprints, regardless of
        # object identity.
        kwargs = {
            "trim_blocks": True,
            "enable_async": True,
            "compile_version": "v1",
            "extensions": [DummyExtension, OtherExtension],
        }
        fp_a = backing.get_environment_fingerprint(Environment(**kwargs))
        fp_b = backing.get_environment_fingerprint(Environment(**kwargs))

        assert fp_a == fp_b
        assert len(fp_a) == hashlib.sha1().digest_size * 2

    def test_extension_set_and_order_matters(self, backing):
        """The extension tuple is serialized in registration order."""
        from jinja2.ext import ExprStmtExtension
        from jinja2.ext import LoopControlExtension

        env_a = Environment(extensions=[ExprStmtExtension, LoopControlExtension])
        env_b = Environment(extensions=[LoopControlExtension, ExprStmtExtension])
        env_c = Environment(extensions=[ExprStmtExtension])

        assert backing.get_environment_fingerprint(env_a) != (
            backing.get_environment_fingerprint(env_b)
        )
        assert backing.get_environment_fingerprint(env_a) != (
            backing.get_environment_fingerprint(env_c)
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enable_async": True},
            {"optimized": False},
            {"block_start_string": "[%", "block_end_string": "%]"},
            {"variable_start_string": "[=", "variable_end_string": "=]"},
            {"comment_start_string": "[#", "comment_end_string": "#]"},
            {"trim_blocks": True},
            {"lstrip_blocks": True},
            {"line_statement_prefix": "#"},
            {"line_comment_prefix": "##"},
            {"newline_sequence": "\r\n"},
            {"keep_trailing_newline": True},
            {"compile_version": "v2"},
        ],
    )
    def test_config_changes_fingerprint(self, backing, kwargs):
        base = backing.get_environment_fingerprint(Environment())
        changed = backing.get_environment_fingerprint(Environment(**kwargs))
        assert base != changed

    def test_compile_version_normalized(self, backing):
        env_int = Environment(compile_version=7)
        env_str = Environment(compile_version="7")
        assert env_int.compile_version == env_str.compile_version == "7"
        assert backing.get_environment_fingerprint(env_int) == (
            backing.get_environment_fingerprint(env_str)
        )

    def test_fingerprint_has_no_object_address(self, backing):
        env = Environment(extensions=[DummyExtension])
        fp = backing.get_environment_fingerprint(env)
        # sha1 hexdigest: only hex chars, no repr or id() leakage
        assert all(char in "0123456789abcdef" for char in fp)


class TestFingerprintedBytecodeCache:
    def test_sync_and_async_are_isolated(self, backing, loader):
        sync_env = make_environment(backing, loader)
        async_env = make_environment(backing, loader, enable_async=True)

        sync_template = sync_env.get_template("template")
        async_template = async_env.get_template("template")
        assert CountingEnvironment.compile_count == 2
        assert len(backing.store) == 2

        # Repeat loads are cache hits for both environments.
        assert sync_env.get_template("template").render(items=(1, 2)) == ("1|2|")
        assert (
            asyncio.run(async_env.get_template("template").render_async(items=(1, 2)))
            == "1|2|"
        )
        assert CountingEnvironment.compile_count == 2

        # The code objects really are sync vs async.
        assert not inspect.isasyncgenfunction(sync_template.root_render_func)
        assert inspect.isasyncgenfunction(async_template.root_render_func)

    def test_same_configuration_hits_cache_with_new_environment(self, backing, loader):
        first = make_environment(backing, loader)
        second = make_environment(backing, loader)

        first.get_template("template")
        assert CountingEnvironment.compile_count == 1
        second.get_template("template")
        assert CountingEnvironment.compile_count == 1

    def test_delimiters_are_isolated(self, backing, loader):
        default_env = make_environment(backing, loader)
        square_env = make_environment(
            backing,
            loader,
            variable_start_string="[=",
            variable_end_string="=]",
        )

        assert default_env.get_template("square").render(item="x") == ("[= item =]")
        assert square_env.get_template("square").render(item="x") == "x"
        assert CountingEnvironment.compile_count == 2

        # Switching back reuses each environment's own cached bytecode.
        assert square_env.get_template("square").render(item="y") == "y"
        assert default_env.get_template("square").render(item="y") == ("[= item =]")
        assert CountingEnvironment.compile_count == 2

    def test_overlay_shares_and_overrides_fingerprint(self, backing, loader):
        base = make_environment(backing, loader)
        base.get_template("template")
        assert CountingEnvironment.compile_count == 1

        # An overlay with the same configuration is a cache hit.
        same_overlay = base.overlay()
        assert same_overlay.get_template("template").render(items=(1,)) == ("1|")
        assert CountingEnvironment.compile_count == 1

        # An overlay that changes a compile-relevant setting gets its own
        # bytecode, and the change does not leak back into the base.
        async_overlay = base.overlay(enable_async=True)
        assert (
            asyncio.run(async_overlay.get_template("template").render_async(items=(1,)))
            == "1|"
        )
        assert CountingEnvironment.compile_count == 2
        assert base.get_template("template").render(items=(2,)) == "2|"
        assert CountingEnvironment.compile_count == 2

        # compile_version can be overridden through overlay() as well.
        versioned_overlay = base.overlay(compile_version="v9")
        versioned_overlay.get_template("template")
        assert CountingEnvironment.compile_count == 3

    def test_extension_set_and_order_isolation(self, backing, loader):
        from jinja2.ext import ExprStmtExtension
        from jinja2.ext import LoopControlExtension

        do_env = make_environment(backing, loader, extensions=[ExprStmtExtension])
        loop_env = make_environment(backing, loader, extensions=[LoopControlExtension])

        do_env.get_template("template")
        loop_env.get_template("template")
        assert CountingEnvironment.compile_count == 2
        assert len(backing.store) == 2

        # A new environment with the same extension set is a hit.
        do_again = make_environment(backing, loader, extensions=[ExprStmtExtension])
        do_again.get_template("template")
        assert CountingEnvironment.compile_count == 2

        # A different registration order is a different fingerprint.
        ordered = make_environment(
            backing,
            loader,
            extensions=[
                LoopControlExtension,
                ExprStmtExtension,
            ],
        )
        reverse = make_environment(
            backing,
            loader,
            extensions=[
                ExprStmtExtension,
                LoopControlExtension,
            ],
        )
        ordered.get_template("template")
        reverse.get_template("template")
        assert CountingEnvironment.compile_count == 4

    def test_compile_version_change_invalidates(self, backing, loader):
        env_v1 = make_environment(backing, loader, compile_version="v1")
        env_v1.get_template("template")
        assert CountingEnvironment.compile_count == 1
        env_v1.get_template("template")
        assert CountingEnvironment.compile_count == 1

        env_v2 = make_environment(backing, loader, compile_version="v2")
        env_v2.get_template("template")
        assert CountingEnvironment.compile_count == 2

        # Going back to v1 reuses the originally cached bytecode.
        env_v1_again = make_environment(backing, loader, compile_version="v1")
        env_v1_again.get_template("template")
        assert CountingEnvironment.compile_count == 2

        # Environments without a version tag use a third key.
        env_no_version = make_environment(backing, loader)
        env_no_version.get_template("template")
        assert CountingEnvironment.compile_count == 3

    def test_old_cache_entry_is_ignored(self, backing, loader):
        """An entry from a non-fingerprinted cache has a different key."""
        env = make_environment(backing, loader)
        old_key = backing.get_cache_key("template", None)
        assert old_key not in backing.dumped_keys

        # Seed the old, un-fingerprinted key with valid bytecode.
        old_env = Environment(loader=loader, cache_size=0, bytecode_cache=backing)
        old_env.get_template("template")
        assert old_key in backing.dumped_keys
        assert CountingEnvironment.compile_count == 0

        # The fingerprinted environment does not read the old key and
        # compiles its own bytecode.
        env.get_template("template")
        assert CountingEnvironment.compile_count == 1
        assert len(backing.store) == 2

    def test_corrupt_cache_entry_is_ignored(self, backing, loader):
        env = make_environment(backing, loader)
        key = backing.get_fingerprinted_cache_key(env, "template", None)
        backing.store[key] = b"not bytecode"

        template = env.get_template("template")
        assert template.render(items=(1,)) == "1|"
        assert CountingEnvironment.compile_count == 1

        # The corrupt entry was replaced with valid cached bytecode.
        template = env.get_template("template")
        assert template.render(items=(2,)) == "2|"
        assert CountingEnvironment.compile_count == 1

    def test_corrupt_marshal_is_ignored(self, backing, loader):
        env = make_environment(backing, loader)
        key = backing.get_fingerprinted_cache_key(env, "template", None)
        checksum = backing.get_source_checksum(loader.get_source(env, "template")[0])
        backing.store[key] = bc_magic + pickle.dumps(checksum, 2) + b"broken"

        template = env.get_template("template")
        assert template.render(items=(1,)) == "1|"
        assert CountingEnvironment.compile_count == 1

    def test_backing_cache_handles_storage(self, backing, loader):
        env = make_environment(backing, loader)
        env.get_template("template")

        assert len(backing.store) == 1
        assert backing.loaded_keys and backing.dumped_keys
        # Keys handed to the backing cache carry the fingerprint suffix.
        key = next(iter(backing.store))
        assert key.count(".") == 1
        assert len(key.split(".", 1)[1]) == 40

    def test_clear_delegates(self, backing, loader):
        env = make_environment(backing, loader)
        env.get_template("template")
        assert backing.store
        FingerprintedBytecodeCache(backing).clear()
        assert not backing.store
