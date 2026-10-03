import typing as t

if t.TYPE_CHECKING:
    from .runtime import Undefined


class TemplateError(Exception):
    """Baseclass for all template errors."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)

    @property
    def message(self) -> str | None:
        return self.args[0] if self.args else None


class TemplateNotFound(IOError, LookupError, TemplateError):
    """Raised if a template does not exist.

    .. versionchanged:: 2.11
        If the given name is :class:`Undefined` and no message was
        provided, an :exc:`UndefinedError` is raised.
    """

    # Silence the Python warning about message being deprecated since
    # it's not valid here.
    message: str | None = None

    def __init__(
        self,
        name: t.Union[str, "Undefined"] | None,
        message: str | None = None,
    ) -> None:
        IOError.__init__(self, name)

        if message is None:
            from .runtime import Undefined

            if isinstance(name, Undefined):
                name._fail_with_undefined_error()

            message = name

        self.message = message
        self.name = name
        self.templates = [name]

    def __str__(self) -> str:
        return str(self.message)


class TemplatesNotFound(TemplateNotFound):
    """Like :class:`TemplateNotFound` but raised if multiple templates
    are selected.  This is a subclass of :class:`TemplateNotFound`
    exception, so just catching the base exception will catch both.

    .. versionchanged:: 2.11
        If a name in the list of names is :class:`Undefined`, a message
        about it being undefined is shown rather than the empty string.

    .. versionadded:: 2.2
    """

    def __init__(
        self,
        names: t.Sequence[t.Union[str, "Undefined"]] = (),
        message: str | None = None,
    ) -> None:
        if message is None:
            from .runtime import Undefined

            parts = []

            for name in names:
                if isinstance(name, Undefined):
                    parts.append(name._undefined_message)
                else:
                    parts.append(name)

            parts_str = ", ".join(map(str, parts))
            message = f"none of the templates given were found: {parts_str}"

        super().__init__(names[-1] if names else None, message)
        self.templates = list(names)


class TemplateSyntaxError(TemplateError):
    """Raised to tell the user that there is a problem with the template."""

    def __init__(
        self,
        message: str,
        lineno: int,
        name: str | None = None,
        filename: str | None = None,
    ) -> None:
        super().__init__(message)
        self.lineno = lineno
        self.name = name
        self.filename = filename
        self.source: str | None = None

        # this is set to True if the debug.translate_syntax_error
        # function translated the syntax error into a new traceback
        self.translated = False

    def __str__(self) -> str:
        # for translated errors we only return the message
        if self.translated:
            return t.cast(str, self.message)

        # otherwise attach some stuff
        location = f"line {self.lineno}"
        name = self.filename or self.name
        if name:
            location = f'File "{name}", {location}'
        lines = [t.cast(str, self.message), "  " + location]

        # if the source is set, add the line to the output
        if self.source is not None:
            try:
                line = self.source.splitlines()[self.lineno - 1]
            except IndexError:
                pass
            else:
                lines.append("    " + line.strip())

        return "\n".join(lines)

    def __reduce__(self):  # type: ignore
        # https://bugs.python.org/issue1692335 Exceptions that take
        # multiple required arguments have problems with pickling.
        # Without this, raises TypeError: __init__() missing 1 required
        # positional argument: 'lineno'
        return self.__class__, (self.message, self.lineno, self.name, self.filename)


class TemplateAssertionError(TemplateSyntaxError):
    """Like a template syntax error, but covers cases where something in the
    template caused an error at compile time that wasn't necessarily caused
    by a syntax error.  However it's a direct subclass of
    :exc:`TemplateSyntaxError` and has the same attributes.
    """


class TemplateRuntimeError(TemplateError):
    """A generic runtime error in the template engine.  Under some situations
    Jinja may raise this exception.
    """


class UndefinedError(TemplateRuntimeError):
    """Raised if a template tries to operate on :class:`Undefined`."""


class SecurityError(TemplateRuntimeError):
    """Raised if a template tries to do something insecure if the
    sandbox is enabled.
    """


class FilterArgumentError(TemplateRuntimeError):
    """This error is raised if a filter was called with inappropriate
    arguments
    """


class TemplateBatchError(TemplateError):
    """Raised when a prepared batch of templates cannot be committed
    because at least one template is no longer valid.

    This happens when the source reported by the loader, or the compile
    configuration of the environment, changed between
    :meth:`Environment.prepare_template_batch` and
    :meth:`~jinja2.environment.PreparedTemplateBatch.commit`. Nothing is
    published when this is raised; caches keep their previous state.

    .. versionadded:: 3.2
    """

    def __init__(
        self,
        message: str | None = None,
        stale: t.Sequence[str] = (),
        invalid: t.Sequence[str] = (),
    ) -> None:
        if message is None:
            parts = []

            if stale:
                parts.append("changed: " + ", ".join(repr(name) for name in stale))

            if invalid:
                parts.append(
                    "no longer available: " + ", ".join(repr(name) for name in invalid)
                )

            if parts:
                message = "template batch was rejected (" + "; ".join(parts) + ")"
            else:
                message = "template batch was rejected"

        super().__init__(message)
        #: Names whose source or compile configuration changed after
        #: the batch was prepared.
        self.stale = list(stale)
        #: Names that fail validation again, for example because the
        #: loader can no longer locate them.
        self.invalid = list(invalid)
