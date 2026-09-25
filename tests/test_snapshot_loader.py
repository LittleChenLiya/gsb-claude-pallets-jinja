"""Tests for :class:`jinja2.SnapshotLoader`: atomic versioned publishing,
per-render version pinning, and version-aware caching.
"""

import threading

import anyio
import pytest

from jinja2 import Environment
from jinja2 import SnapshotLoader
from jinja2.exceptions import TemplateNotFound
from jinja2.exceptions import TemplateSyntaxError
from jinja2.loaders import _pinned_versions


@pytest.fixture
def loader():
    return SnapshotLoader()


@pytest.fixture
def env(loader):
    return Environment(loader=loader)


class _Gate:
    """Block a render between ``entered`` and ``release`` so a publish can
    happen while the render is in flight.
    """

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self):
        self.entered.set()
        assert self.release.wait(timeout=30)
        return ""


def test_publish_returns_monotonic_versions(loader, env):
    assert loader.version == 0
    assert loader.versions == ()

    assert loader.publish({"page": "one"}) == 1
    assert loader.publish({"page": "two"}) == 2
    assert loader.publish({"page": "three"}) == 3

    assert loader.version == 3
    assert loader.versions == (1, 2, 3)
    assert env.get_template("page").render() == "three"


def test_constructor_mapping_is_first_version():
    loader = SnapshotLoader({"page": "hello"})
    assert loader.version == 1
    env = Environment(loader=loader)
    assert env.get_template("page").render() == "hello"


def test_failed_publish_keeps_old_version(loader, env):
    v1 = loader.publish({"page": "old"})

    with pytest.raises(TemplateSyntaxError):
        loader.publish({"page": "new", "broken": "{% if %}"})

    # The failed publish neither replaced the snapshot nor consumed a
    # version number.
    assert loader.version == v1
    assert loader.versions == (v1,)
    assert env.get_template("page").render() == "old"

    # The next successful publish continues the sequence.
    v2 = loader.publish({"page": "new"})
    assert v2 == v1 + 1
    assert loader.versions == (v1, v2)
    assert env.get_template("page").render() == "new"


def test_failed_publish_on_empty_loader(loader, env):
    with pytest.raises(TemplateSyntaxError):
        loader.publish({"broken": "{% if %}"})

    assert loader.version == 0
    assert loader.versions == ()

    with pytest.raises(TemplateNotFound):
        env.get_template("broken")

    assert loader.publish({"page": "ok"}) == 1


def test_render_pins_version_across_concurrent_publish(loader, env):
    """A render paused before its include, then resumed after a publish,
    still renders entirely from the old version.  New requests use the
    new version.
    """
    gate = _Gate()
    env.globals["wait"] = gate.wait
    loader.publish(
        {
            "page": "old/{{ wait() }}{% include 'item' %}",
            "item": "old",
        }
    )
    template = env.get_template("page")
    results = []

    thread = threading.Thread(target=lambda: results.append(template.render()))
    thread.start()
    # Wait until the old render is in flight, then publish the new version.
    assert gate.entered.wait(timeout=30)
    loader.publish(
        {
            "page": "new/{% include 'item' %}",
            "item": "new",
        }
    )
    gate.release.set()
    thread.join(timeout=30)

    assert results == ["old/old"]
    assert env.get_template("page").render() == "new/new"


def test_extends_include_import_share_pinned_version(loader, env):
    gate = _Gate()
    env.globals["wait"] = gate.wait
    loader.publish(
        {
            "page": (
                "{% extends 'base' %}{% block b %}{{ wait() }}"
                "{% include 'item' %}{% import 'macros' as m %}{{ m.tag() }}"
                "{% endblock %}"
            ),
            "base": "old/{% block b %}{% endblock %}",
            "item": "old/",
            "macros": "{% macro tag() %}old{% endmacro %}",
        }
    )
    template = env.get_template("page")
    results = []

    thread = threading.Thread(target=lambda: results.append(template.render()))
    thread.start()
    assert gate.entered.wait(timeout=30)
    loader.publish(
        {
            "page": (
                "{% extends 'base' %}{% block b %}{{ wait() }}"
                "{% include 'item' %}{% import 'macros' as m %}{{ m.tag() }}"
                "{% endblock %}"
            ),
            "base": "new/{% block b %}{% endblock %}",
            "item": "new/",
            "macros": "{% macro tag() %}new{% endmacro %}",
        }
    )
    gate.release.set()
    thread.join(timeout=30)

    assert results == ["old/old/old"]
    assert env.get_template("page").render() == "new/new/new"


def test_cache_key_distinguishes_versions(loader, env):
    loader.publish({"page": "v1"})
    t1 = env.get_template("page")
    # Same name, same version: cache hit, identical object.
    assert env.get_template("page") is t1

    loader.publish({"page": "v2"})
    t2 = env.get_template("page")
    # Same name, different version: no cache hit, no version leak.
    assert t2 is not t1
    assert t1.render() == "v1"
    assert t2.render() == "v2"
    assert env.get_template("page") is t2


def test_cache_scoped_to_environment_config(loader):
    loader.publish({"page": "{{ value }}"})
    env_escape = Environment(loader=loader, autoescape=True)
    env_plain = Environment(loader=loader, autoescape=False)

    assert env_escape.get_template("page").render(value="<") == "&lt;"
    assert env_plain.get_template("page").render(value="<") == "<"

    # A publish is visible to both environments, still with their own
    # configuration applied.
    loader.publish({"page": "{{ value }}!"})
    assert env_escape.get_template("page").render(value="<") == "&lt;!"
    assert env_plain.get_template("page").render(value="<") == "<!"


def test_generate_keeps_version_until_closed(loader, env):
    gate = _Gate()
    env.globals["wait"] = gate.wait
    loader.publish(
        {
            "page": "old/{{ wait() }}{% include 'item' %}",
            "item": "old",
        }
    )
    stream = env.get_template("page").generate()
    chunks = []

    thread = threading.Thread(target=lambda: chunks.extend(stream))
    thread.start()
    assert gate.entered.wait(timeout=30)
    loader.publish(
        {
            "page": "new/{% include 'item' %}",
            "item": "new",
        }
    )
    gate.release.set()
    thread.join(timeout=30)

    assert "".join(chunks) == "old/old"
    assert "".join(env.get_template("page").generate()) == "new/new"


def test_generate_closed_early_keeps_old_version_until_close(loader, env):
    loader.publish({"page": "old-{% include 'item' %}", "item": "old"})
    stream = env.get_template("page").generate()
    assert next(stream) == "old-"

    loader.publish({"page": "new-{% include 'item' %}", "item": "new"})

    # The started stream keeps reading the old version until it is closed.
    assert next(stream) == "old"
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)

    assert "".join(env.get_template("page").generate()) == "new-new"


def test_missing_template_raises(loader, env):
    loader.publish({"page": "ok"})

    with pytest.raises(TemplateNotFound):
        env.get_template("missing")


def test_list_templates_uses_current_version(loader):
    loader.publish({"b": "1", "a": "1"})
    assert loader.list_templates() == ["a", "b"]
    loader.publish({"c": "2"})
    assert loader.list_templates() == ["c"]


def test_render_async_pins_version_across_concurrent_publish(run_async_fn):
    loader = SnapshotLoader()
    env = Environment(loader=loader, enable_async=True)

    async def main():
        entered = anyio.Event()
        release = anyio.Event()

        async def wait():
            entered.set()
            await release.wait()
            return ""

        env.globals["wait"] = wait
        loader.publish(
            {
                "page": "old/{{ wait() }}{% include 'item' %}",
                "item": "old",
            }
        )
        template = env.get_template("page")
        results = []

        async with anyio.create_task_group() as tg:

            async def render():
                results.append(await template.render_async())

            tg.start_soon(render)
            await entered.wait()
            loader.publish(
                {
                    "page": "new/{% include 'item' %}",
                    "item": "new",
                }
            )
            release.set()

        assert results == ["old/old"]
        assert await env.get_template("page").render_async() == "new/new"

    run_async_fn(main)


def test_render_async_cancellation_leaves_no_version_leak(run_async_fn):
    loader = SnapshotLoader()
    env = Environment(loader=loader, enable_async=True)

    async def main():
        entered = anyio.Event()
        never = anyio.Event()

        async def wait():
            entered.set()
            await never.wait()
            return ""

        env.globals["wait"] = wait
        loader.publish(
            {
                "page": "old/{{ wait() }}",
                "item": "old",
            }
        )
        template = env.get_template("page")

        async with anyio.create_task_group() as tg:

            async def render():
                await template.render_async()

            tg.start_soon(render)
            await entered.wait()
            tg.cancel_scope.cancel()

        # The cancelled render left no pinned version behind, the loader is
        # still usable, and publishing continues the version sequence.
        assert not (_pinned_versions.get() or {})
        assert loader.version == 1
        assert loader.publish({"page": "new"}) == 2
        assert await env.get_template("page").render_async() == "new"

    run_async_fn(main)


def test_generate_async_pins_version_and_survives_early_close(run_async_fn):
    loader = SnapshotLoader()
    env = Environment(loader=loader, enable_async=True)

    async def main():
        entered = anyio.Event()
        release = anyio.Event()

        async def wait():
            entered.set()
            await release.wait()
            return ""

        env.globals["wait"] = wait
        loader.publish(
            {
                "page": "old/{{ wait() }}{% include 'item' %}",
                "item": "old",
            }
        )
        stream = env.get_template("page").generate_async()
        chunks = []

        async with anyio.create_task_group() as tg:

            async def consume():
                async for chunk in stream:
                    chunks.append(chunk)

            tg.start_soon(consume)
            await entered.wait()
            loader.publish(
                {
                    "page": "new/{% include 'item' %}",
                    "item": "new",
                }
            )
            release.set()

        assert "".join(chunks) == "old/old"

        # A stream closed early does not leak its version either.
        stream2 = env.get_template("page").generate_async()
        assert await stream2.__anext__() == "new/"
        await stream2.aclose()
        assert not (_pinned_versions.get() or {})
        assert await env.get_template("page").render_async() == "new/new"

    run_async_fn(main)


def test_failed_async_publish_keeps_old_version(run_async_fn):
    loader = SnapshotLoader()
    env = Environment(loader=loader, enable_async=True)

    async def main():
        loader.publish({"page": "old"})

        with pytest.raises(TemplateSyntaxError):
            loader.publish({"page": "new", "broken": "{% if %}"})

        assert loader.version == 1
        assert await env.get_template("page").render_async() == "old"

    run_async_fn(main)
