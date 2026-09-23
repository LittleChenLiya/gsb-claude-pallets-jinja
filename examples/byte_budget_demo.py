"""Demonstrates Template.generate_with_budget and generate_async_with_budget.

Renders a fixed output sequence of "甲", "", "&amp;" (3 + 0 + 5 UTF-8
bytes) and shows that a budget of 8 bytes succeeds, a budget of 7 bytes
fails after delivering "甲", and that the underlying iteration source is
released in every outcome.  Runs in well under 90 seconds:

    python examples/byte_budget_demo.py
"""

import asyncio

from jinja2 import ByteBudgetExceeded
from jinja2 import Environment

ITEMS = ["甲", "", "&"]
TEMPLATE = "{% for x in source %}{{ x }}{% endfor %}"


class CountingSource:
    """Iteration source whose finally clause records its cleanup."""

    def __init__(self, items):
        self.items = items
        self.closed = 0

    def __iter__(self):
        try:
            yield from self.items
        finally:
            self.closed += 1

    async def __aiter__(self):
        try:
            for item in self.items:
                yield item
        finally:
            self.closed += 1


def show(label, outcome, source):
    print(f"{label}: {outcome} | source released: {source.closed == 1}")


def sync_demo(env):
    template = env.from_string(TEMPLATE)

    source = CountingSource(ITEMS)
    chunks = list(template.generate_with_budget(8, source=source))
    show("sync budget=8 ", f"OK, delivered {chunks!r}", source)

    source = CountingSource(ITEMS)
    try:
        list(template.generate_with_budget(7, source=source))
    except ByteBudgetExceeded as e:
        show(
            "sync budget=7 ",
            f"ByteBudgetExceeded(limit={e.limit}, delivered={e.delivered},"
            f" rejected={e.rejected})",
            source,
        )


async def async_demo(env):
    template = env.from_string(TEMPLATE)
    rows = []

    source = CountingSource(ITEMS)
    chunks = [
        chunk async for chunk in template.generate_async_with_budget(8, source=source)
    ]
    rows.append(("async budget=8", f"OK, delivered {chunks!r}", source))

    source = CountingSource(ITEMS)
    try:
        async for _ in template.generate_async_with_budget(7, source=source):
            pass
    except ByteBudgetExceeded as e:
        rows.append(
            (
                "async budget=7",
                f"ByteBudgetExceeded(limit={e.limit}, delivered={e.delivered},"
                f" rejected={e.rejected})",
                source,
            )
        )

    return rows


def main():
    sync_demo(Environment(autoescape=True))
    rows = asyncio.run(async_demo(Environment(autoescape=True, enable_async=True)))

    # The event loop finalizes the released async generators before
    # asyncio.run returns, so the sources are cleaned up by now.
    for label, outcome, source in rows:
        show(label, outcome, source)


if __name__ == "__main__":
    main()
