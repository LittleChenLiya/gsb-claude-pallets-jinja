"""Explicit block execution tracing.

This module provides an *opt-in* way to record which block
implementations are actually entered during a single render, and which
parent implementation each ``super()`` call selects.  It does not only
list the registered block candidates: every event is recorded at the
moment the block generator is really executed (or the parent
implementation is really selected), while the render is running.

Default rendering is completely unchanged.  Tracing is only active
while a :class:`BlockTrace` is bound (for example through
:func:`render_with_trace`), and each render gets its own independent
trace state.  The report is produced from the events recorded during
that single render: the loader is not run again and no block is
re-executed to obtain it.
"""

from __future__ import annotations

import inspect
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from .environment import Template
    from .runtime import Context

#: The trace bound to the current render, if any.  A fresh value is set
#: for every traced render, so concurrent or nested renders in other
#: tasks keep their own independent state.
_active_trace: ContextVar["BlockTrace | None"] = ContextVar(
    "jinja2_block_trace", default=None
)


def current_trace() -> "BlockTrace | None":
    """Return the trace bound to the current render, or ``None``."""
    return _active_trace.get()


@dataclass(frozen=True)
class BlockEvent:
    """One recorded block execution event.

    :ivar kind: ``"enter"`` when a block implementation was actually
        entered, ``"super"`` when ``super()`` selected a parent
        implementation.
    :ivar block: name of the block.
    :ivar template: name of the template that provides the block
        implementation (its real source, e.g. the included template for
        blocks rendered through ``{% include %}``).
    :ivar depth: index in the inheritance stack, ``0`` is the most
        derived implementation.
    """

    kind: str
    block: str
    template: t.Optional[str]
    depth: int


def _template_of(func: t.Callable[..., t.Any]) -> t.Optional[str]:
    """Find the template a block implementation really comes from."""
    if isinstance(func, _TracedBlock):
        return func.template

    globals_ = getattr(func, "__globals__", None)

    if globals_:
        return globals_.get("name")

    return None


class _TracedBlock:
    """Wrap one block implementation to record when it is entered.

    The wrapper compares equal to the wrapped function so that
    ``Context.super`` can still locate the current implementation in the
    block stack with ``list.index``.
    """

    __slots__ = ("func", "trace", "block", "template", "depth")

    def __init__(
        self,
        func: t.Callable[["Context"], t.Any],
        trace: "BlockTrace",
        block: str,
        depth: int,
    ) -> None:
        self.func = func
        self.trace = trace
        self.block = block
        self.template = _template_of(func)
        self.depth = depth

    @property
    def __name__(self) -> str:
        return getattr(self.func, "__name__", "block")

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, _TracedBlock):
            return self.func == other.func

        return self.func == other

    def __ne__(self, other: t.Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.func)

    def __repr__(self) -> str:
        return f"<TracedBlock {self.block!r} of {self.template!r} depth={self.depth}>"

    def __call__(self, context: "Context") -> t.Any:
        # The event is recorded inside the generator body, so it only
        # fires when the block is really entered, not when the generator
        # is merely created.  All items are yielded through untouched,
        # keeping the original yield, Markup and autoescape behavior.
        if inspect.isasyncgenfunction(self.func):
            return self._async_gen(context)

        return self._sync_gen(context)

    def _sync_gen(self, context: "Context") -> t.Iterator[str]:
        self.trace.record("enter", self.block, self.template, self.depth)
        yield from self.func(context)

    async def _async_gen(self, context: "Context") -> t.AsyncGenerator[str, None]:
        self.trace.record("enter", self.block, self.template, self.depth)

        async for event in self.func(context):
            yield event


class _TracingBlockList(list):
    """A block stack that wraps every implementation added to it."""

    def __init__(
        self, trace: "BlockTrace", block: str, items: t.Iterable[t.Any] = ()
    ) -> None:
        self._trace = trace
        self._block = block
        super().__init__()

        for item in items:
            self.append(item)

    def _wrap(self, func: t.Callable[..., t.Any], depth: int) -> t.Any:
        if isinstance(func, _TracedBlock) and func.trace is self._trace:
            # Already wrapped for this render, do not count it twice.
            return func

        return _TracedBlock(func, self._trace, self._block, depth)

    def append(self, func: t.Callable[..., t.Any]) -> None:
        super().append(self._wrap(func, len(self)))


class _TracingBlocksDict(dict):
    """Mapping of block name to :class:`_TracingBlockList`.

    ``setdefault`` is overridden because the code generated for
    ``{% extends %}`` registers parent blocks through it.
    """

    def __init__(self, trace: "BlockTrace", mapping: t.Mapping[str, t.Any]) -> None:
        super().__init__()
        self._trace = trace

        for name, stack in mapping.items():
            self[name] = _TracingBlockList(trace, name, stack)

    def setdefault(self, key: str, default: t.Any = None) -> t.Any:
        if not isinstance(default, _TracingBlockList):
            default = _TracingBlockList(self._trace, key, default or ())

        return super().setdefault(key, default)


class BlockTrace:
    """Ordered record of the block execution of a single render.

    Can be used as a context manager around any render call, or
    indirectly through :func:`render_with_trace`::

        trace = BlockTrace()
        with trace:
            result = template.render()
        print(trace.chain("content"))
    """

    def __init__(self) -> None:
        self.events: t.List[BlockEvent] = []
        self._token: t.Any = None

    def record(
        self, kind: str, block: str, template: t.Optional[str], depth: int
    ) -> None:
        self.events.append(BlockEvent(kind, block, template, depth))

    @property
    def entered(self) -> t.List[BlockEvent]:
        """Events for block implementations actually entered."""
        return [e for e in self.events if e.kind == "enter"]

    @property
    def supers(self) -> t.List[BlockEvent]:
        """Events for parent implementations selected by ``super()``."""
        return [e for e in self.events if e.kind == "super"]

    def chain(self, block: t.Optional[str] = None) -> t.List[t.Optional[str]]:
        """Templates whose block implementation was actually entered, in
        execution order.  If ``block`` is given, only that block's chain
        is returned.
        """
        return [
            e.template
            for e in self.events
            if e.kind == "enter" and (block is None or e.block == block)
        ]

    def install(
        self, blocks: t.Dict[str, t.List[t.Callable[..., t.Any]]]
    ) -> t.Dict[str, t.List[t.Callable[..., t.Any]]]:
        """Wrap a render-local block mapping so executions are recorded."""
        return t.cast(
            t.Dict[str, t.List[t.Callable[..., t.Any]]],
            _TracingBlocksDict(self, blocks),
        )

    def __enter__(self) -> "BlockTrace":
        self._token = _active_trace.set(self)
        return self

    def __exit__(self, *exc_info: t.Any) -> None:
        _active_trace.reset(self._token)
        self._token = None

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> t.Iterator[BlockEvent]:
        return iter(self.events)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} events={self.events!r}>"


def record_super(
    context: "Context",
    name: str,
    blocks: t.List[t.Callable[..., t.Any]],
    index: int,
) -> None:
    """Record which parent implementation ``super()`` selected.

    Called from :meth:`jinja2.runtime.Context.super` and
    :attr:`jinja2.runtime.BlockReference.super`.  This is a no-op when
    no trace is bound to the current render.
    """
    trace = _active_trace.get()

    if trace is None:
        return

    trace.record("super", name, _template_of(blocks[index]), index)


def render_with_trace(
    template: "Template", *args: t.Any, **kwargs: t.Any
) -> t.Tuple[str, BlockTrace]:
    """Render ``template`` once and return ``(output, trace)``.

    The output is identical to a plain :meth:`Template.render` call;
    the trace describes which block implementations were actually
    entered and which parents ``super()`` selected during this render.
    """
    trace = BlockTrace()

    with trace:
        rv = template.render(*args, **kwargs)

    return rv, trace


async def render_with_trace_async(
    template: "Template", *args: t.Any, **kwargs: t.Any
) -> t.Tuple[str, BlockTrace]:
    """Async variant of :func:`render_with_trace` built on
    :meth:`Template.render_async`.
    """
    trace = BlockTrace()

    with trace:
        rv = await template.render_async(*args, **kwargs)

    return rv, trace
