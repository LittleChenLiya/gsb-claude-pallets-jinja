"""A sandbox layer that ensures unsafe operations cannot be performed.
Useful when the template itself comes from an untrusted source.
"""

import operator
import types
import typing as t
from _string import formatter_field_name_split  # type: ignore
from collections import abc
from collections import deque
from functools import update_wrapper
from string import Formatter

from markupsafe import EscapeFormatter
from markupsafe import Markup

from .environment import Environment
from .exceptions import SecurityError
from .runtime import Context
from .runtime import Undefined

F = t.TypeVar("F", bound=t.Callable[..., t.Any])

#: maximum number of items a range may produce
MAX_RANGE = 100000

#: Unsafe function attributes.
UNSAFE_FUNCTION_ATTRIBUTES: set[str] = set()

#: Unsafe method attributes. Function attributes are unsafe for methods too.
UNSAFE_METHOD_ATTRIBUTES: set[str] = set()

#: unsafe generator attributes.
UNSAFE_GENERATOR_ATTRIBUTES = {"gi_frame", "gi_code"}

#: unsafe attributes on coroutines
UNSAFE_COROUTINE_ATTRIBUTES = {"cr_frame", "cr_code"}

#: unsafe attributes on async generators
UNSAFE_ASYNC_GENERATOR_ATTRIBUTES = {"ag_code", "ag_frame"}

_mutable_spec: tuple[tuple[type[t.Any], frozenset[str]], ...] = (
    (
        abc.MutableSet,
        frozenset(
            [
                "add",
                "clear",
                "difference_update",
                "discard",
                "pop",
                "remove",
                "symmetric_difference_update",
                "update",
            ]
        ),
    ),
    (
        abc.MutableMapping,
        frozenset(["clear", "pop", "popitem", "setdefault", "update"]),
    ),
    (
        abc.MutableSequence,
        frozenset(
            ["append", "clear", "pop", "reverse", "insert", "sort", "extend", "remove"]
        ),
    ),
    (
        deque,
        frozenset(
            [
                "append",
                "appendleft",
                "clear",
                "extend",
                "extendleft",
                "pop",
                "popleft",
                "remove",
                "rotate",
            ]
        ),
    ),
)


def safe_range(*args: int) -> range:
    """A range that can't generate ranges with a length of more than
    MAX_RANGE items.
    """
    rng = range(*args)

    if len(rng) > MAX_RANGE:
        raise OverflowError(
            "Range too big. The sandbox blocks ranges larger than"
            f" MAX_RANGE ({MAX_RANGE})."
        )

    return rng


def unsafe(f: F) -> F:
    """Marks a function or method as unsafe.

    .. code-block: python

        @unsafe
        def delete(self):
            pass
    """
    f.unsafe_callable = True  # type: ignore
    return f


def is_internal_attribute(obj: t.Any, attr: str) -> bool:
    """Test if the attribute given is an internal python attribute.  For
    example this function returns `True` for the `func_code` attribute of
    python objects.  This is useful if the environment method
    :meth:`~SandboxedEnvironment.is_safe_attribute` is overridden.

    >>> from jinja2.sandbox import is_internal_attribute
    >>> is_internal_attribute(str, "mro")
    True
    >>> is_internal_attribute(str, "upper")
    False
    """
    if isinstance(obj, types.FunctionType):
        if attr in UNSAFE_FUNCTION_ATTRIBUTES:
            return True
    elif isinstance(obj, types.MethodType):
        if attr in UNSAFE_FUNCTION_ATTRIBUTES or attr in UNSAFE_METHOD_ATTRIBUTES:
            return True
    elif isinstance(obj, type):
        if attr == "mro":
            return True
    elif isinstance(obj, (types.CodeType, types.TracebackType, types.FrameType)):
        return True
    elif isinstance(obj, types.GeneratorType):
        if attr in UNSAFE_GENERATOR_ATTRIBUTES:
            return True
    elif hasattr(types, "CoroutineType") and isinstance(obj, types.CoroutineType):
        if attr in UNSAFE_COROUTINE_ATTRIBUTES:
            return True
    elif hasattr(types, "AsyncGeneratorType") and isinstance(
        obj, types.AsyncGeneratorType
    ):
        if attr in UNSAFE_ASYNC_GENERATOR_ATTRIBUTES:
            return True
    return attr.startswith("__")


def modifies_known_mutable(obj: t.Any, attr: str) -> bool:
    """This function checks if an attribute on a builtin mutable object
    (list, dict, set or deque) or the corresponding ABCs would modify it
    if called.

    >>> modifies_known_mutable({}, "clear")
    True
    >>> modifies_known_mutable({}, "keys")
    False
    >>> modifies_known_mutable([], "append")
    True
    >>> modifies_known_mutable([], "index")
    False

    If called with an unsupported object, ``False`` is returned.

    >>> modifies_known_mutable("foo", "upper")
    False
    """
    for typespec, unsafe in _mutable_spec:
        if isinstance(obj, typespec):
            return attr in unsafe
    return False


class _AuthorizedMethod:
    """Proxy for a single method of an :class:`_AuthorizedObject`.

    The proxy is returned when an authorized attribute is accessed on an
    authorized object.  It never exposes the original bound method and
    re-checks the allow set as well as the sandbox's callable safety on
    every call, so revoking the permission takes effect immediately, even
    if the method was extracted from the object earlier.
    """

    def __init__(
        self,
        env: "SandboxedEnvironment",
        obj: t.Any,
        name: str,
        allowed: set[str],
    ) -> None:
        self._env = env
        self._obj = obj
        self._name = name
        self._allowed = allowed

    def _check(self) -> t.Any:
        """Re-check the authorization and sandbox limits, then return
        the original bound method to call.
        """
        if self._name not in self._allowed:
            raise SecurityError(
                f"call to method {self._name!r} of"
                f" {type(self._obj).__name__!r} object is not permitted;"
                f" the authorization was revoked."
            )

        method = getattr(self._obj, self._name)

        if not self._env.is_safe_callable(method):
            raise SecurityError(f"{method!r} is not safely callable")

        return method

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._check()(*args, **kwargs)


class _AuthorizedObject:
    """Proxy wrapping an object so that only the explicitly authorized
    attributes are accessible from sandboxed templates.

    Access to any other attribute, including private and internal
    attributes, raises a :exc:`SecurityError`.  Callable attributes are
    wrapped in an :class:`_AuthorizedMethod` instead of exposing the
    original bound method.
    """

    def __init__(
        self, env: "SandboxedEnvironment", obj: t.Any, allowed: set[str]
    ) -> None:
        self._env = env
        self._obj = obj
        self._allowed = allowed

    def __getattr__(self, name: str) -> t.Any:
        # __getattr__ is only called for names not found on the proxy
        # itself, so the private attributes of the proxy stay hidden.
        # Access the internals via object.__getattribute__ so that this
        # method does not recurse through itself when the instance state
        # has not been restored yet (e.g. during copy or pickle).
        try:
            obj = object.__getattribute__(self, "_obj")
            allowed = object.__getattribute__(self, "_allowed")
            env = object.__getattribute__(self, "_env")
        except AttributeError:
            raise AttributeError(name) from None

        # Dunder attributes are Python internals rather than template
        # data.  Raise AttributeError so that copy/pickle as well as the
        # sandbox's normal attribute handling keep working; the sandbox
        # still blocks access to them just like for any other object.
        if name.startswith("__"):
            raise AttributeError(name)

        if name not in allowed:
            raise SecurityError(
                f"access to attribute {name!r} of"
                f" {type(obj).__name__!r} object is not permitted;"
                f" only {sorted(allowed)!r} are authorized."
            )

        value = getattr(obj, name)

        if callable(value):
            return _AuthorizedMethod(env, obj, name, allowed)

        return value

    def __repr__(self) -> str:
        return f"<authorized {self._obj!r}>"


class SandboxedEnvironment(Environment):
    """The sandboxed environment.  It works like the regular environment but
    tells the compiler to generate sandboxed code.  Additionally subclasses of
    this environment may override the methods that tell the runtime what
    attributes or functions are safe to access.

    If the template tries to access insecure code a :exc:`SecurityError` is
    raised.  However also other exceptions may occur during the rendering so
    the caller has to ensure that all exceptions are caught.
    """

    sandboxed = True

    #: default callback table for the binary operators.  A copy of this is
    #: available on each instance of a sandboxed environment as
    #: :attr:`binop_table`
    default_binop_table: dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "**": operator.pow,
        "%": operator.mod,
    }

    #: default callback table for the unary operators.  A copy of this is
    #: available on each instance of a sandboxed environment as
    #: :attr:`unop_table`
    default_unop_table: dict[str, t.Callable[[t.Any], t.Any]] = {
        "+": operator.pos,
        "-": operator.neg,
    }

    #: a set of binary operators that should be intercepted.  Each operator
    #: that is added to this set (empty by default) is delegated to the
    #: :meth:`call_binop` method that will perform the operator.  The default
    #: operator callback is specified by :attr:`binop_table`.
    #:
    #: The following binary operators are interceptable:
    #: ``//``, ``%``, ``+``, ``*``, ``-``, ``/``, and ``**``
    #:
    #: The default operation form the operator table corresponds to the
    #: builtin function.  Intercepted calls are always slower than the native
    #: operator call, so make sure only to intercept the ones you are
    #: interested in.
    #:
    #: .. versionadded:: 2.6
    intercepted_binops: frozenset[str] = frozenset()

    #: a set of unary operators that should be intercepted.  Each operator
    #: that is added to this set (empty by default) is delegated to the
    #: :meth:`call_unop` method that will perform the operator.  The default
    #: operator callback is specified by :attr:`unop_table`.
    #:
    #: The following unary operators are interceptable: ``+``, ``-``
    #:
    #: The default operation form the operator table corresponds to the
    #: builtin function.  Intercepted calls are always slower than the native
    #: operator call, so make sure only to intercept the ones you are
    #: interested in.
    #:
    #: .. versionadded:: 2.6
    intercepted_unops: frozenset[str] = frozenset()

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.globals["range"] = safe_range
        self.binop_table = self.default_binop_table.copy()
        self.unop_table = self.default_unop_table.copy()

    def authorize(self, obj: t.Any, allowed: set[str] | t.Iterable[str]) -> t.Any:
        """Create a proxy around *obj* that only exposes the attributes
        listed in *allowed* to sandboxed templates.

        This is useful when an object should be passed to a template but
        only a few of its methods are safe to call.  The original object
        is not modified; the proxy wraps it and intercepts all attribute
        access.

        :param obj: the object to wrap.
        :param allowed: a set of attribute names that may be accessed.
            If a :class:`set` is passed, it is used directly, so the
            permissions can be updated (and revoked) after the proxy was
            created.  Any other iterable is copied into a new set.

        .. code-block:: python

            >>> from jinja2.sandbox import SandboxedEnvironment
            >>> env = SandboxedEnvironment()
            >>> class Service:
            ...     def read(self):
            ...         return "read"
            ...     def write(self):
            ...         return "write"
            >>> permissions = {"read"}
            >>> service = env.authorize(Service(), permissions)
            >>> env.from_string("{{ service.read() }}").render(service=service)
            'read'
            >>> env.from_string("{{ service.write() }}").render(service=service)
            Traceback (most recent call last):
              ...
            SecurityError: access to attribute 'write' of 'Service' object is not
                permitted; only ['read'] are authorized.

        Revoking a permission takes effect immediately, even for methods
        that were extracted from the proxy before:

        .. code-block:: python

            >>> permissions.remove("read")
            >>> env.from_string("{{ service.read() }}").render(service=service)
            Traceback (most recent call last):
              ...
            SecurityError: call to method 'read' of 'Service' object is not
                permitted; the authorization was revoked.

        The sandbox's normal safety checks still apply to the wrapped
        object.  Methods marked with :func:`unsafe` (or with
        ``alters_data``) cannot be called even if they are listed in
        *allowed*.  Sync and async rendering behave identically.
        """
        if not isinstance(allowed, set):
            allowed = set(allowed)

        return _AuthorizedObject(self, obj, allowed)

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        """The sandboxed environment will call this method to check if the
        attribute of an object is safe to access.  Per default all attributes
        starting with an underscore are considered private as well as the
        special attributes of internal python objects as returned by the
        :func:`is_internal_attribute` function.
        """
        return not (attr.startswith("_") or is_internal_attribute(obj, attr))

    def is_safe_callable(self, obj: t.Any) -> bool:
        """Check if an object is safely callable. By default callables
        are considered safe unless decorated with :func:`unsafe`.

        This also recognizes the Django convention of setting
        ``func.alters_data = True``.
        """
        return not (
            getattr(obj, "unsafe_callable", False) or getattr(obj, "alters_data", False)
        )

    def call_binop(
        self, context: Context, operator: str, left: t.Any, right: t.Any
    ) -> t.Any:
        """For intercepted binary operator calls (:meth:`intercepted_binops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.binop_table[operator](left, right)

    def call_unop(self, context: Context, operator: str, arg: t.Any) -> t.Any:
        """For intercepted unary operator calls (:meth:`intercepted_unops`)
        this function is executed instead of the builtin operator.  This can
        be used to fine tune the behavior of certain operators.

        .. versionadded:: 2.6
        """
        return self.unop_table[operator](arg)

    def getitem(self, obj: t.Any, argument: str | t.Any) -> t.Any | Undefined:
        """Subscribe an object from sandboxed code."""
        try:
            return obj[argument]
        except (TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        value = getattr(obj, attr)
                    except AttributeError:
                        pass
                    else:
                        fmt = self.wrap_str_format(value)
                        if fmt is not None:
                            return fmt
                        if self.is_safe_attribute(obj, argument, value):
                            return value
                        return self.unsafe_undefined(obj, argument)
        return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Any | Undefined:
        """Subscribe an object from sandboxed code and prefer the
        attribute.  The attribute passed *must* be a bytestring.
        """
        try:
            value = getattr(obj, attribute)
        except AttributeError:
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                pass
        else:
            fmt = self.wrap_str_format(value)
            if fmt is not None:
                return fmt
            if self.is_safe_attribute(obj, attribute, value):
                return value
            return self.unsafe_undefined(obj, attribute)
        return self.undefined(obj=obj, name=attribute)

    def unsafe_undefined(self, obj: t.Any, attribute: str) -> Undefined:
        """Return an undefined object for unsafe attributes."""
        return self.undefined(
            f"access to attribute {attribute!r} of"
            f" {type(obj).__name__!r} object is unsafe.",
            name=attribute,
            obj=obj,
            exc=SecurityError,
        )

    def wrap_str_format(self, value: t.Any) -> t.Callable[..., str] | None:
        """If the given value is a ``str.format`` or ``str.format_map`` method,
        return a new function than handles sandboxing. This is done at access
        rather than in :meth:`call`, so that calls made without ``call`` are
        also sandboxed.
        """
        if not isinstance(
            value, (types.MethodType, types.BuiltinMethodType)
        ) or value.__name__ not in ("format", "format_map"):
            return None

        f_self: t.Any = value.__self__

        if not isinstance(f_self, str):
            return None

        str_type: type[str] = type(f_self)
        is_format_map = value.__name__ == "format_map"
        formatter: SandboxedFormatter

        if isinstance(f_self, Markup):
            formatter = SandboxedEscapeFormatter(self, escape=f_self.escape)
        else:
            formatter = SandboxedFormatter(self)

        vformat = formatter.vformat

        def wrapper(*args: t.Any, **kwargs: t.Any) -> str:
            if is_format_map:
                if kwargs:
                    raise TypeError("format_map() takes no keyword arguments")

                if len(args) != 1:
                    raise TypeError(
                        f"format_map() takes exactly one argument ({len(args)} given)"
                    )

                kwargs = args[0]
                args = ()

            return str_type(vformat(f_self, args, kwargs))

        return update_wrapper(wrapper, value)

    def call(
        __self,  # noqa: B902
        __context: Context,
        __obj: t.Any,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Call an object from sandboxed code."""

        # the double prefixes are to avoid double keyword argument
        # errors when proxying the call.
        if isinstance(__obj, _AuthorizedMethod):
            # Re-check the authorization and sandbox limits on every
            # call, then let the context call the original method so
            # that pass_context and async calls are handled as usual.
            return __context.call(__obj._check(), *args, **kwargs)

        if not __self.is_safe_callable(__obj):
            raise SecurityError(f"{__obj!r} is not safely callable")
        return __context.call(__obj, *args, **kwargs)


class ImmutableSandboxedEnvironment(SandboxedEnvironment):
    """Works exactly like the regular `SandboxedEnvironment` but does not
    permit modifications on the builtin mutable objects `list`, `set`, and
    `dict` by using the :func:`modifies_known_mutable` function.
    """

    def is_safe_attribute(self, obj: t.Any, attr: str, value: t.Any) -> bool:
        if not super().is_safe_attribute(obj, attr, value):
            return False

        return not modifies_known_mutable(obj, attr)


class SandboxedFormatter(Formatter):
    def __init__(self, env: Environment, **kwargs: t.Any) -> None:
        self._env = env
        super().__init__(**kwargs)

    def get_field(
        self, field_name: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]
    ) -> tuple[t.Any, str]:
        first, rest = formatter_field_name_split(field_name)
        obj = self.get_value(first, args, kwargs)
        for is_attr, i in rest:
            if is_attr:
                obj = self._env.getattr(obj, i)
            else:
                obj = self._env.getitem(obj, i)
        return obj, first


class SandboxedEscapeFormatter(SandboxedFormatter, EscapeFormatter):
    pass
