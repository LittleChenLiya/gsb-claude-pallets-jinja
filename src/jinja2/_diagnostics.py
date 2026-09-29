"""Support for diagnostic rendering that records missing values.

Diagnostic rendering is an explicitly enabled alternative to a normal
render.  Every lookup chain (``user.name``, ``items[0].price``) is routed
through one of the ``diag_*`` hooks defined here.  When diagnostics are
not active the hooks behave like the normal runtime, so a normal render
keeps its original behavior.  When diagnostics are active and a lookup
fails, the hooks return a permissive placeholder instead of raising,
allowing the rest of the template to render while recording the missing
value.  Errors unrelated to missing values (for example an integer
division by zero) are never caught.

The active :class:`DiagnosticsRecorder` is stored in a :class:`ContextVar`
so that it is shared by every template involved in a single render
(main template, included templates, macros and blocks) without leaking
into independent renders.
"""

from __future__ import annotations

import types
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass

from .exceptions import UndefinedError
from .runtime import Undefined


@dataclass(frozen=True)
class MissingVariable:
    """A missing value accessed while rendering a template.

    :param name: Name of the template in which the value was accessed.
    :param line: 1-based line number in the template source.
    :param path: Access path of the value, for example ``user.name`` or
        ``items[0].price``.
    """

    name: str | None
    line: int
    path: str

    def __str__(self) -> str:
        location = f"{self.name}:{self.line}" if self.name else f"line {self.line}"
        return f"{self.path} is undefined ({location})"


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of a diagnostic render.

    :param output: The rendered text, exactly as a normal render would
        produce when the missing values are printed.
    :param missing: Missing values encountered while rendering, in the
        order they were first accessed.
    """

    output: str
    missing: tuple[MissingVariable, ...]


class DiagnosticsRecorder:
    """Collects missing values for a single diagnostic render."""

    def __init__(self) -> None:
        self._missing: list[MissingVariable] = []
        self._seen: set[tuple[str | None, int, str]] = set()

    def record(self, name: str | None, line: int, path: str) -> None:
        """Record a missing value. Duplicate accesses are ignored."""
        key = (name, line, path)
        if key not in self._seen:
            self._seen.add(key)
            self._missing.append(MissingVariable(name=name, line=line, path=path))

    @property
    def missing(self) -> tuple[MissingVariable, ...]:
        return tuple(self._missing)


_recorder: ContextVar[DiagnosticsRecorder | None] = ContextVar(
    "jinja2_diagnostics_recorder", default=None
)


def get_diagnostics_recorder() -> DiagnosticsRecorder | None:
    """Return the recorder for the active render, if any."""
    return _recorder.get()


class record_diagnostics:
    """Activate a recorder for the duration of a render."""

    def __init__(self, recorder: DiagnosticsRecorder) -> None:
        self._recorder = recorder
        self._token: t.Any = None

    def __enter__(self) -> DiagnosticsRecorder:
        self._token = _recorder.set(self._recorder)
        return self._recorder

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        _recorder.reset(self._token)


class _DiagUndefined(Undefined):
    """Permissive undefined used inside a failing lookup chain.

    It chains without raising so that later steps in the same chain (and
    printing the result) succeed.  Only the step that first fails on a
    real undefined records; the placeholder itself never records.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> _DiagUndefined:
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)
        return self

    def __getitem__(self, _name: t.Any) -> _DiagUndefined:  # type: ignore[override]
        return self


def _placeholder(
    recorder: DiagnosticsRecorder,
    obj: t.Any,
    attribute: t.Any,
    template_name: str | None,
    line: int,
    path: str,
) -> _DiagUndefined:
    recorder.record(template_name, line, path)
    return _DiagUndefined(obj=obj, name=attribute)


def diag_name(value: t.Any, template_name: str | None, line: int, path: str) -> t.Any:
    """Diagnostic hook for a loaded root variable name."""
    recorder = get_diagnostics_recorder()

    if recorder is None:
        return value

    if isinstance(value, Undefined) and not isinstance(value, _DiagUndefined):
        recorder.record(template_name, line, path)
        return _DiagUndefined(
            hint=value._undefined_hint,
            obj=value._undefined_obj,
            name=value._undefined_name,
            exc=value._undefined_exception,
        )

    return value


def _chain_undefined(
    environment: t.Any,
    value: t.Any,
    kind: str,
    argument: t.Any,
) -> t.Any:
    """Perform one lookup step, returning a placeholder instead of
    raising or returning an undefined when diagnostics are active.
    """
    recorder = get_diagnostics_recorder()

    if recorder is None:
        if kind == "attr":
            return environment.getattr(value, argument)
        return environment.getitem(value, argument)

    if isinstance(value, _DiagUndefined):
        # A previous step already failed; keep propagating the
        # placeholder.  Only the outermost step records.
        return value

    try:
        if kind == "attr":
            result = environment.getattr(value, argument)
        else:
            result = environment.getitem(value, argument)
    except UndefinedError:
        return _DiagUndefined(obj=value, name=argument)

    if isinstance(result, Undefined):
        return _DiagUndefined(obj=value, name=argument)

    return result


def diag_chain_getattr(environment: t.Any, value: t.Any, attribute: str) -> t.Any:
    """Silent diagnostic hook for an inner attribute lookup step."""
    return _chain_undefined(environment, value, "attr", attribute)


def diag_chain_getitem(environment: t.Any, value: t.Any, argument: t.Any) -> t.Any:
    """Silent diagnostic hook for an inner item lookup step."""
    return _chain_undefined(environment, value, "item", argument)


def diag_getattr(
    environment: t.Any,
    value: t.Any,
    attribute: str,
    template_name: str | None,
    line: int,
    path: str,
) -> t.Any:
    """Diagnostic hook for the outermost attribute lookup of a chain."""
    recorder = get_diagnostics_recorder()

    if recorder is None:
        return environment.getattr(value, attribute)

    result = diag_chain_getattr(environment, value, attribute)

    if isinstance(result, _DiagUndefined):
        return _placeholder(recorder, value, attribute, template_name, line, path)
    return result


def diag_getitem(
    environment: t.Any,
    value: t.Any,
    argument: t.Any,
    template_name: str | None,
    line: int,
    path: str,
) -> t.Any:
    """Diagnostic hook for the outermost item lookup of a chain."""
    recorder = get_diagnostics_recorder()

    if recorder is None:
        return environment.getitem(value, argument)

    result = diag_chain_getitem(environment, value, argument)

    if isinstance(result, _DiagUndefined):
        return _placeholder(recorder, value, argument, template_name, line, path)
    return result
