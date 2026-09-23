import asyncio
import threading

import pytest

from jinja2 import ByteBudgetExceeded
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.environment import _budgeted_generate

ITEMS = ["甲", "", "&"]
TEMPLATE = "{% for x in items %}{{ x }}{% endfor %}"


class CountingSource:
    """An iteration source that records how often its finally ran."""

    def __init__(self, items):
        self.items = items
        self.closed = 0

    def __iter__(self):
        try:
            yield from self.items
        finally:
            self.closed += 1


class AsyncCountingSource:
    def __init__(self, items):
        self.items = items
        self.closed = 0

    async def __aiter__(self):
        try:
            for item in self.items:
                yield item
        finally:
            self.closed += 1


@pytest.fixture
def auto_env():
    return Environment(autoescape=True)


def test_budget_exact_fit_passes(auto_env):
    t = auto_env.from_string(TEMPLATE)
    chunks = list(t.generate_with_budget(8, items=ITEMS))
    assert chunks == ["甲", "", "&amp;"]
    assert "".join(chunks) == "甲&amp;"


def test_budget_exceeded(auto_env):
    t = auto_env.from_string(TEMPLATE)
    gen = t.generate_with_budget(7, items=ITEMS)
    assert next(gen) == "甲"
    assert next(gen) == ""

    with pytest.raises(ByteBudgetExceeded) as exc_info:
        next(gen)

    exc = exc_info.value
    assert exc.limit == 7
    assert exc.delivered == 3
    assert exc.rejected == 5
    # The exception must not leak template output.
    assert "甲" not in str(exc)
    assert "&amp;" not in str(exc)


def test_exact_budget_does_not_stop_early(auto_env):
    # Reaching the budget exactly must not declare the stream finished:
    # normal exhaustion, empty chunks, and new non-empty output are still
    # distinguished on the following iterations.
    t = auto_env.from_string(TEMPLATE)
    assert list(t.generate_with_budget(3, items=["甲"])) == ["甲"]
    assert list(t.generate_with_budget(3, items=["甲", ""])) == ["甲", ""]

    gen = t.generate_with_budget(3, items=ITEMS)
    assert next(gen) == "甲"
    assert next(gen) == ""

    with pytest.raises(ByteBudgetExceeded) as exc_info:
        next(gen)

    assert exc_info.value.delivered == 3


@pytest.mark.parametrize("bad", [True, False, 1.5, "8", None])
def test_budget_must_be_real_int(auto_env, bad):
    t = auto_env.from_string(TEMPLATE)

    with pytest.raises(TypeError):
        t.generate_with_budget(bad, items=ITEMS)


def test_budget_must_be_non_negative(auto_env):
    t = auto_env.from_string(TEMPLATE)

    with pytest.raises(ValueError):
        t.generate_with_budget(-1, items=ITEMS)


def test_zero_budget(auto_env):
    t = auto_env.from_string(TEMPLATE)
    assert list(t.generate_with_budget(0, items=[])) == []
    assert list(t.generate_with_budget(0, items=[""])) == [""]

    with pytest.raises(ByteBudgetExceeded) as exc_info:
        list(t.generate_with_budget(0, items=["a"]))

    assert exc_info.value.delivered == 0
    assert exc_info.value.rejected == 1


def test_source_finalized_on_normal_end(auto_env):
    source = CountingSource(ITEMS)
    t = auto_env.from_string("{% for x in source %}{{ x }}{% endfor %}")
    assert list(t.generate_with_budget(8, source=source)) == ["甲", "", "&amp;"]
    assert source.closed == 1


def test_source_finalized_on_budget_overflow(auto_env):
    source = CountingSource(ITEMS)
    t = auto_env.from_string("{% for x in source %}{{ x }}{% endfor %}")

    with pytest.raises(ByteBudgetExceeded):
        list(t.generate_with_budget(7, source=source))

    assert source.closed == 1


def test_source_finalized_on_consumer_close(auto_env):
    source = CountingSource(ITEMS)
    t = auto_env.from_string("{% for x in source %}{{ x }}{% endfor %}")
    gen = t.generate_with_budget(8, source=source)
    assert next(gen) == "甲"
    gen.close()
    assert source.closed == 1


def test_source_finalized_on_template_error(auto_env):
    source = CountingSource(ITEMS)
    t = auto_env.from_string("{% for x in source %}{{ x }}{{ 1 / 0 }}{% endfor %}")

    with pytest.raises(ZeroDivisionError):
        list(t.generate_with_budget(100, source=source))

    assert source.closed == 1


def test_cleanup_error_does_not_hide_template_error(auto_env):
    t = auto_env.from_string("{{ 1 / 0 }}")

    class ExplodingClose:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.wrapped)

        def close(self):
            raise RuntimeError("cleanup failed")

    wrapped = _budgeted_generate(ExplodingClose(iter(t.generate())), 100)

    with pytest.raises(ZeroDivisionError):
        list(wrapped)


def test_inheritance_with_budget():
    env = Environment(
        loader=DictLoader(
            {
                "base.html": "[{% block body %}{% endblock %}]",
                "child.html": (
                    "{% extends 'base.html' %}"
                    "{% block body %}{% for x in items %}{{ x }}{% endfor %}"
                    "{% endblock %}"
                ),
            }
        ),
        autoescape=True,
    )
    t = env.get_template("child.html")
    assert "".join(t.generate_with_budget(10, items=ITEMS)) == "[甲&amp;]"

    with pytest.raises(ByteBudgetExceeded) as exc_info:
        list(t.generate_with_budget(8, items=ITEMS))

    exc = exc_info.value
    assert exc.delivered == 4  # "[" + "甲" + ""
    assert exc.rejected == 5  # "&amp;"


def test_include_with_budget():
    env = Environment(
        loader=DictLoader(
            {"inc.html": "{{ x }}", "main.html": "A{% include 'inc.html' %}B"}
        ),
        autoescape=True,
    )
    t = env.get_template("main.html")
    assert "".join(t.generate_with_budget(100, x="&")) == "A&amp;B"

    with pytest.raises(ByteBudgetExceeded):
        list(t.generate_with_budget(1, x="&"))


def test_shared_environment_concurrent(auto_env):
    # Budget state lives per render call, never on the shared environment.
    t = auto_env.from_string(TEMPLATE)
    barrier = threading.Barrier(4)
    results = {}
    errors = {}

    def worker(name, budget):
        barrier.wait(timeout=10)
        try:
            results[name] = list(t.generate_with_budget(budget, items=ITEMS))
        except ByteBudgetExceeded as e:
            errors[name] = e

    threads = [
        threading.Thread(target=worker, args=(f"ok{i}", 8)) for i in range(2)
    ] + [threading.Thread(target=worker, args=(f"fail{i}", 7)) for i in range(2)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert results == {"ok0": ["甲", "", "&amp;"], "ok1": ["甲", "", "&amp;"]}
    assert set(errors) == {"fail0", "fail1"}

    for exc in errors.values():
        assert exc.limit == 7
        assert exc.delivered == 3


async def _collect(agen):
    return [chunk async for chunk in agen]


def test_async_budget_exact_fit_and_overflow(run_async_fn):
    env = Environment(autoescape=True, enable_async=True)
    t = env.from_string(TEMPLATE)

    async def ok():
        return await _collect(t.generate_async_with_budget(8, items=ITEMS))

    assert run_async_fn(ok) == ["甲", "", "&amp;"]

    async def overflow():
        chunks = []

        try:
            async for chunk in t.generate_async_with_budget(7, items=ITEMS):
                chunks.append(chunk)
        except ByteBudgetExceeded as e:
            return chunks, e.limit, e.delivered, e.rejected

        raise AssertionError("expected ByteBudgetExceeded")

    assert run_async_fn(overflow) == (["甲", ""], 7, 3, 5)


# Jinja does not aclose async sources iterated by a template; they are
# released by garbage collection, which trio reports with a ResourceWarning.
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_async_source_finalized(run_async_fn):
    env = Environment(autoescape=True, enable_async=True)
    t = env.from_string("{% for x in source %}{{ x }}{% endfor %}")

    async def normal(source):
        return await _collect(t.generate_async_with_budget(8, source=source))

    source = AsyncCountingSource(ITEMS)
    assert run_async_fn(normal, source) == ["甲", "", "&amp;"]
    assert source.closed == 1

    async def overflow(source):
        await _collect(t.generate_async_with_budget(7, source=source))

    source = AsyncCountingSource(ITEMS)

    with pytest.raises(ByteBudgetExceeded):
        run_async_fn(overflow, source)

    assert source.closed == 1

    async def consumer_aclose(source):
        agen = t.generate_async_with_budget(8, source=source)
        assert await agen.__anext__() == "甲"
        await agen.aclose()

    source = AsyncCountingSource(ITEMS)
    run_async_fn(consumer_aclose, source)
    assert source.closed == 1


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_async_template_error_releases_source(run_async_fn):
    env = Environment(autoescape=True, enable_async=True)
    t = env.from_string("{% for x in source %}{{ x }}{{ 1 / 0 }}{% endfor %}")

    async def run(source):
        await _collect(t.generate_async_with_budget(100, source=source))

    source = AsyncCountingSource(ITEMS)

    with pytest.raises(ZeroDivisionError):
        run_async_fn(run, source)

    assert source.closed == 1


def test_async_cancellation_releases_source():
    env = Environment(autoescape=True, enable_async=True)
    t = env.from_string("{% for x in source %}{{ x }}{% endfor %}")
    source = AsyncCountingSource(["甲", "", "&"])

    async def main():
        async def consume():
            async for _ in t.generate_async_with_budget(8, source=source):
                await asyncio.sleep(0)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert source.closed == 1
