"""Tests for atomic template batch cache warm-up.

Covers :meth:`Environment.prepare_template_batch` and
:meth:`PreparedTemplateBatch.commit`.
"""

from __future__ import annotations

import typing as t

import pytest

from jinja2 import BytecodeCache
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import FunctionLoader
from jinja2 import PreparedTemplateBatch
from jinja2 import TemplateBatchError
from jinja2 import TemplateSyntaxError


def make_env(mapping: dict[str, str], **kwargs: t.Any) -> Environment:
    kwargs.setdefault("auto_reload", True)
    return Environment(loader=DictLoader(mapping), **kwargs)


def cache_keys(env: Environment) -> set[str]:
    if env.cache is None:
        return set()

    return {name for _ref, name in env.cache}


class MemoryBytecodeCache(BytecodeCache):
    """Bytecode cache recording which keys it stored."""

    def __init__(self) -> None:
        self.storage: dict[str, bytes] = {}

    def load_bytecode(self, bucket):
        data = self.storage.get(bucket.key)

        if data is not None:
            bucket.bytecode_from_string(data)

    def dump_bytecode(self, bucket):
        self.storage[bucket.key] = bucket.bytecode_to_string()

    def keys(self) -> set[str]:
        return set(self.storage)


def count_marker(calls: list[int]) -> t.Callable[[], str]:
    def marker() -> str:
        calls.append(1)
        return f"n={len(calls)}"

    return marker


def cached_templates(env: Environment) -> dict[str, object]:
    if env.cache is None:
        return {}

    return {name: template for (_ref, name), template in env.cache.items()}


class TestPrepareBatch:
    def test_syntax_error_leaves_old_cache_intact(self):
        # Old "a" keeps working; broken "b" never reaches a cache.
        sources = {"a": "old a", "b": "{% for %}"}
        env = make_env(sources)

        # Warm the cache with the old "a".
        old_a = env.get_template("a")
        assert old_a.render() == "old a"
        assert cache_keys(env) == {"a"}

        # A new "a" is prepared together with a syntactically broken "b".
        sources["a"] = "new a"

        with pytest.raises(TemplateSyntaxError):
            env.prepare_template_batch(["a", "b"])

        # Nothing staged or published: the cache still holds exactly the
        # old "a" object, and "b" was never compiled into any cache.
        cached = cached_templates(env)
        assert set(cached) == {"a"}
        assert cached["a"] is old_a
        assert cached["a"].render() == "old a"

        # Even with auto reload off the batch could not have replaced it.
        env.auto_reload = False
        assert env.get_template("a") is old_a
        assert env.get_template("a").render() == "old a"

    def test_fixed_batch_commits_both_templates(self):
        # After fixing "b" the whole batch commits and hits cache.
        sources = {"a": "old a", "b": "{% for %}"}
        env = make_env(sources)

        env.get_template("a")

        sources["a"] = "new a"

        with pytest.raises(TemplateSyntaxError):
            env.prepare_template_batch(["a", "b"])

        sources["b"] = "b"
        batch = env.prepare_template_batch(["a", "b"])

        assert isinstance(batch, PreparedTemplateBatch)
        assert batch.names == ["a", "b"]
        committed = batch.commit()
        assert committed == ["a", "b"]

        new_a = env.get_template("a")
        new_b = env.get_template("b")
        assert cache_keys(env) == {"a", "b"}
        assert new_a.render() == "new a"
        assert new_b.render() == "b"

    def test_changed_source_rejects_whole_batch(self):
        # One changed template rejects the entire batch together.
        sources = {"a": "a v1", "b": "b v1"}
        env = make_env(sources)

        old_a = env.get_template("a")

        batch = env.prepare_template_batch(["a", "b"])
        assert not batch.is_stale()

        # "a" changes after prepare, "b" is still valid.
        sources["a"] = "a v2"
        assert batch.stale_names() == ["a"]
        assert batch.is_stale()

        with pytest.raises(TemplateBatchError) as exc_info:
            batch.commit()

        assert exc_info.value.stale == ["a"]

        # Whole batch rejected: the cache still holds exactly the old
        # "a" object, and "b" never entered the staging result cache.
        cached = cached_templates(env)
        assert set(cached) == {"a"}
        assert cached["a"] is old_a
        assert cached["a"].render() == "a v1"

        # Auto reload does not matter: the rejected batch never wrote.
        env.auto_reload = False
        assert env.get_template("a") is old_a

        # Loading "b" normally compiles a fresh template, not the
        # rejected staging area.
        assert "b" not in cached
        assert env.get_template("b").render() == "b v1"

        # A fresh prepare/commit now publishes both current sources.
        batch2 = env.prepare_template_batch(["a", "b"])
        batch2.commit()
        assert cache_keys(env) == {"a", "b"}
        assert env.get_template("a").render() == "a v2"

    def test_global_function_is_never_called(self):
        # Prepare/commit never execute templates or call globals.
        calls: list[int] = []
        sources = {"a": "{{ marker() }}", "b": "{% set x = marker() %}b"}
        env = make_env(sources)
        env.globals["marker"] = count_marker(calls)

        env.get_template("a")
        calls.clear()

        batch = env.prepare_template_batch(["a", "b"])
        assert calls == []

        batch.commit()
        assert calls == []

        assert env.get_template("a").render() == "n=1"
        assert len(calls) == 1
        assert env.get_template("b").render() == "b"
        assert len(calls) == 2

    def test_no_dependency_recursion(self):
        # Only explicitly named templates are compiled.
        sources = {
            "a": '{% include "missing.html" %}',
            "b": '{% extends "base.html" %}',
        }
        env = make_env(sources)

        # Preparation succeeds even though dependencies are absent.
        batch = env.prepare_template_batch(["a", "b"])
        assert batch.names == ["a", "b"]
        assert cache_keys(env) == set()
        batch.commit()
        assert cache_keys(env) == {"a", "b"}

        with pytest.raises(LookupError):
            env.get_template("a").render()

    def test_duplicate_and_empty_names(self):
        sources = {"a": "a", "b": "b"}
        env = make_env(sources)

        batch = env.prepare_template_batch(["a", "a", "b", "a"])
        assert batch.names == ["a", "b"]
        assert batch.commit() == ["a", "b"]

        empty = env.prepare_template_batch(())
        assert empty.names == []
        assert empty.commit() == []

    def test_non_string_name_is_rejected(self):
        env = make_env({"a": "a"})

        with pytest.raises(TypeError):
            env.prepare_template_batch(["a", 42])  # type: ignore[list-item]

    def test_missing_template_aborts_prepare(self):
        env = make_env({"a": "a"})

        with pytest.raises(LookupError):
            env.prepare_template_batch(["a", "nope"])

        assert cache_keys(env) == set()

    def test_no_loader(self):
        env = Environment()

        with pytest.raises(TypeError):
            env.prepare_template_batch(["a"])

    def test_config_change_after_prepare_rejects_batch(self):
        sources = {"a": "a", "b": "b"}
        env = make_env(sources)

        batch = env.prepare_template_batch(["a", "b"])
        env.trim_blocks = not env.trim_blocks

        assert set(batch.stale_names()) == {"a", "b"}

        with pytest.raises(TemplateBatchError):
            batch.commit()

        assert cache_keys(env) == set()

    def test_retry_after_fixing_source(self):
        sources = {"a": "a v1"}
        env = make_env(sources)

        batch = env.prepare_template_batch(["a"])
        sources["a"] = "a v2"

        with pytest.raises(TemplateBatchError):
            batch.commit()

        assert cache_keys(env) == set()

        batch2 = env.prepare_template_batch(["a"])
        assert batch2.commit() == ["a"]
        assert env.get_template("a").render() == "a v2"

    def test_uptodate_callback_rejects_commit(self):
        state = {"current": "v1", "mtime": 1}

        def load(name: str):
            if name != "a":
                return None

            mtime = state["mtime"]
            return state["current"], None, lambda: state["mtime"] == mtime

        env = Environment(loader=FunctionLoader(load))

        batch = env.prepare_template_batch(["a"])
        state["mtime"] += 1  # source string unchanged, mtime changed

        with pytest.raises(TemplateBatchError):
            batch.commit()

    def test_bytecode_cache_published_only_on_commit(self):
        sources = {"a": "a", "b": "b"}
        bcc = MemoryBytecodeCache()
        env = make_env(sources, bytecode_cache=bcc)

        batch = env.prepare_template_batch(["a", "b"])
        assert bcc.keys() == set()

        batch.commit()
        assert bcc.keys()
        assert cache_keys(env) == {"a", "b"}

        # A new environment sharing the bytecode cache gets the code
        # without compiling.
        env2 = make_env(sources, bytecode_cache=bcc)
        assert env2.get_template("a").render() == "a"

    def test_failed_commit_does_not_write_bytecode_cache(self):
        sources = {"a": "a v1", "b": "b"}
        bcc = MemoryBytecodeCache()
        env = make_env(sources, bytecode_cache=bcc)

        batch = env.prepare_template_batch(["a", "b"])
        sources["a"] = "a v2"

        with pytest.raises(TemplateBatchError):
            batch.commit()

        assert bcc.keys() == set()
        assert cache_keys(env) == set()

    def test_missing_source_at_commit_marks_invalid_and_rejects_all(self):
        sources = {"a": "a", "b": "b"}
        env = make_env(sources)

        batch = env.prepare_template_batch(["a", "b"])
        del sources["a"]

        with pytest.raises(TemplateBatchError) as exc_info:
            batch.commit()

        assert exc_info.value.invalid == ["a"]
        assert cache_keys(env) == set()

    def test_autoescape_config_change_is_detected(self):
        sources = {"a": "<b>", "b": "x"}
        env = make_env(sources, autoescape=False)

        batch = env.prepare_template_batch(["a", "b"])
        env.autoescape = True

        with pytest.raises(TemplateBatchError) as exc_info:
            batch.commit()

        assert set(exc_info.value.stale) == {"a", "b"}
        assert cache_keys(env) == set()

    def test_async_environment_batch(self):
        import asyncio

        calls: list[int] = []
        sources = {
            "a": "hi {{ marker() }}",
            "b": "{% macro m() %}x{% endmacro %}[{{ m() }}]",
        }
        env = make_env(sources, enable_async=True)
        env.globals["marker"] = count_marker(calls)

        batch = env.prepare_template_batch(["a", "b"])
        assert calls == []
        batch.commit()
        assert calls == []

        async def render_all() -> tuple[str, str]:
            return (
                await env.get_template("a").render_async(),
                await env.get_template("b").render_async(),
            )

        a, b = asyncio.run(render_all())
        assert a == "hi n=1"
        assert b == "[x]"
        assert len(calls) == 1

    def test_no_cache_commit_is_validation_only(self):
        # cache_size=0 and no bytecode cache: nothing to write, but
        # stale batches must still be rejected.
        sources = {"a": "a"}
        env = make_env(sources, cache_size=0)
        assert env.cache is None

        batch = env.prepare_template_batch(["a"])
        assert batch.commit() == ["a"]

        stale_batch = env.prepare_template_batch(["a"])
        sources["a"] = "b"

        with pytest.raises(TemplateBatchError):
            stale_batch.commit()
