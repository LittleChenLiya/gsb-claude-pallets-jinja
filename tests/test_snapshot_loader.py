import asyncio
import threading

import pytest

from jinja2 import Environment
from jinja2 import SnapshotLoader
from jinja2.exceptions import TemplateNotFound
from jinja2.exceptions import TemplateSyntaxError


def make_env(mapping, **kwargs):
    loader = SnapshotLoader(mapping)
    env = Environment(loader=loader, **kwargs)
    return env, loader


V1 = {
    "base.html": "old-base[{% block body %}{% endblock %}]",
    "page.html": (
        '{% extends "base.html" %}{% block body %}'
        '{% import "macros.html" as m %}{{ m.tag() }}'
        '{% include "item.html" %}{% endblock %}'
    ),
    "item.html": "old-item",
    "macros.html": "{% macro tag() %}old-macro{% endmacro %}",
}

V2 = {
    "base.html": "new-base[{% block body %}{% endblock %}]",
    "page.html": (
        '{% extends "base.html" %}{% block body %}'
        '{% import "macros.html" as m %}{{ m.tag() }}'
        '{% include "item.html" %}{% endblock %}'
    ),
    "item.html": "new-item",
    "macros.html": "{% macro tag() %}new-macro{% endmacro %}",
}


class TestPublish:
    def test_publish_returns_monotonic_versions(self):
        env, loader = make_env({"a.html": "one"})
        assert loader.current_version == 1
        assert loader.publish({"a.html": "two"}, env) == 2
        assert loader.publish({"a.html": "three"}, env) == 3
        assert loader.versions() == [1, 2, 3]
        assert loader.current_version == 3

    def test_failed_publish_keeps_old_version(self):
        env, loader = make_env(dict(V1))
        before = loader.versions()

        with pytest.raises(TemplateSyntaxError):
            loader.publish({"page.html": "{% if %}", "item.html": "x"}, env)

        # No version was consumed by the failed publish.
        assert loader.versions() == before
        assert loader.current_version == 1
        # The old version is still fully usable.
        assert env.get_template("page.html").render() == "old-base[old-macroold-item]"
        # The next successful publish gets the next number, the failed
        # attempt did not occupy a queryable version.
        assert loader.publish(dict(V2), env) == 2
        assert loader.versions() == [1, 2]

    def test_publish_validates_every_template(self):
        env, loader = make_env({"a.html": "ok"})
        mapping = {"a.html": "fine", "b.html": "{{ broken"}

        with pytest.raises(TemplateSyntaxError):
            loader.publish(mapping, env)

        assert env.get_template("a.html").render() == "ok"
        with pytest.raises(TemplateNotFound):
            env.get_template("b.html")

    def test_publish_uses_bound_environment(self):
        env, loader = make_env({"a.html": "one"})
        env.get_template("a.html")
        # No explicit environment: falls back to the one that loaded.
        assert loader.publish({"a.html": "two"}) == 2
        assert env.get_template("a.html").render() == "two"

    def test_publish_without_environment_raises(self):
        loader = SnapshotLoader()
        with pytest.raises(RuntimeError):
            loader.publish({"a.html": "x"})

    def test_list_templates_reflects_current_version(self):
        env, loader = make_env({"a.html": "1"})
        assert loader.list_templates() == ["a.html"]
        loader.publish({"a.html": "1", "b.html": "2"}, env)
        assert loader.list_templates() == ["a.html", "b.html"]


class TestVersionPinning:
    def test_render_uses_one_version_for_all_lookups(self):
        env, loader = make_env(dict(V1))
        old_template = env.get_template("page.html")
        assert old_template.render() == "old-base[old-macroold-item]"

        loader.publish(dict(V2), env)

        # New renders use the new snapshot for extends, include, import.
        assert env.get_template("page.html").render() == "new-base[new-macronew-item]"
        # A template loaded before the publish still renders its own
        # snapshot: parent and children are not mixed.
        assert old_template.render() == "old-base[old-macroold-item]"

    def test_concurrent_publish_during_blocked_include(self):
        mapping = {
            "page.html": '{{ wait() }}{% include "item.html" %}',
            "item.html": "old",
        }
        env, loader = make_env(mapping)
        entered = threading.Event()
        release = threading.Event()

        def wait():
            entered.set()
            release.wait(10)
            return ""

        env.globals["wait"] = wait
        result = []

        def render_old():
            result.append(env.get_template("page.html").render())

        thread = threading.Thread(target=render_old)
        thread.start()
        # Wait until the old render is blocked inside the page.
        assert entered.wait(10)
        loader.publish(
            {
                "page.html": '{{ wait() }}{% include "item.html" %}',
                "item.html": "new",
            },
            env,
        )
        release.set()
        thread.join(10)

        # The in-flight render finished with the old snapshot.
        assert result == ["old"]
        # New renders see the new snapshot.
        assert env.get_template("page.html").render() == "new"

    def test_streaming_render_keeps_old_version_until_close(self):
        env, loader = make_env(
            {"page.html": 'A{% include "item.html" %}B', "item.html": "old"}
        )
        stream = env.get_template("page.html").generate()
        assert next(stream) == "A"

        loader.publish(
            {"page.html": 'A{% include "item.html" %}B', "item.html": "new"}, env
        )

        # The started stream keeps reading the old snapshot.
        assert "".join(stream) == "oldB"
        stream.close()
        # New streams use the new snapshot.
        assert "".join(env.get_template("page.html").generate()) == "AnewB"

    def test_generator_closed_early_releases_version(self):
        env, loader = make_env(
            {"page.html": 'A{% include "item.html" %}B', "item.html": "old"}
        )
        stream = env.get_template("page.html").generate()
        assert next(stream) == "A"
        loader.publish(
            {"page.html": 'A{% include "item.html" %}B', "item.html": "new"}, env
        )
        stream.close()

        # No pin leaks out of the closed generator.
        assert loader._pinned_version() is None
        assert env.get_template("page.html").render() == "AnewB"


class TestCache:
    def test_cache_distinguishes_versions(self):
        env, loader = make_env({"page.html": "old"})
        old_template = env.get_template("page.html")
        assert env.get_template("page.html") is old_template

        loader.publish({"page.html": "new"}, env)

        new_template = env.get_template("page.html")
        assert new_template is not old_template
        assert env.get_template("page.html") is new_template
        # Both cache entries render their own version: no leak.
        assert old_template.render() == "old"
        assert new_template.render() == "new"

    def test_cache_distinguishes_environment_config(self):
        loader = SnapshotLoader({"page.html": "{{ value }}"})
        plain = Environment(loader=loader)
        escaped = Environment(loader=loader, autoescape=True)

        assert plain.get_template("page.html").render(value="<") == "<"
        assert escaped.get_template("page.html").render(value="<") == "&lt;"


class TestAsync:
    def test_render_async_pins_version(self):
        async def main():
            env, loader = make_env(
                {
                    "page.html": '{{ wait() }}{% include "item.html" %}',
                    "item.html": "old",
                },
                enable_async=True,
            )
            entered = asyncio.Event()
            release = asyncio.Event()

            async def wait():
                entered.set()
                await release.wait()
                return ""

            env.globals["wait"] = wait

            async def publish_new():
                await entered.wait()
                loader.publish(
                    {
                        "page.html": '{{ wait() }}{% include "item.html" %}',
                        "item.html": "new",
                    },
                    env,
                )
                release.set()

            old_task = asyncio.create_task(env.get_template("page.html").render_async())
            pub_task = asyncio.create_task(publish_new())
            assert await old_task == "old"
            await pub_task
            assert await env.get_template("page.html").render_async() == "new"

        asyncio.run(main())

    def test_render_async_cancellation_releases_version(self):
        async def main():
            env, loader = make_env({"page.html": "{{ wait() }}old"}, enable_async=True)
            release = asyncio.Event()

            async def wait():
                await release.wait()
                return ""

            env.globals["wait"] = wait
            old_template = env.get_template("page.html")
            task = asyncio.create_task(old_template.render_async())
            await asyncio.sleep(0)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            # The cancelled render did not leak its pin.
            assert loader._pinned_version() is None

            loader.publish({"page.html": "{{ wait() }}new"}, env)
            release.set()
            # Old version still renders, new version is used for new loads.
            assert await old_template.render_async() == "old"
            assert await env.get_template("page.html").render_async() == "new"

        asyncio.run(main())

    def test_generate_async_closed_early_releases_version(self):
        async def main():
            env, loader = make_env(
                {"page.html": 'A{% include "item.html" %}B', "item.html": "old"},
                enable_async=True,
            )
            stream = env.get_template("page.html").generate_async()
            assert await stream.__anext__() == "A"

            loader.publish(
                {"page.html": 'A{% include "item.html" %}B', "item.html": "new"},
                env,
            )

            await stream.aclose()
            assert loader._pinned_version() is None

            stream = env.get_template("page.html").generate_async()
            chunks = [chunk async for chunk in stream]
            assert "".join(chunks) == "AnewB"

        asyncio.run(main())

    def test_generate_async_keeps_old_version_until_close(self):
        async def main():
            env, loader = make_env(
                {"page.html": 'A{% include "item.html" %}B', "item.html": "old"},
                enable_async=True,
            )
            stream = env.get_template("page.html").generate_async()
            assert await stream.__anext__() == "A"

            loader.publish(
                {"page.html": 'A{% include "item.html" %}B', "item.html": "new"},
                env,
            )

            rest = [chunk async for chunk in stream]
            assert "".join(rest) == "oldB"

        asyncio.run(main())
