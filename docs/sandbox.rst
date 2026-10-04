Sandbox
=======

The Jinja sandbox can be used to render untrusted templates. Access to
attributes, method calls, operators, mutating data structures, and
string formatting can be intercepted and prohibited.

.. code-block:: pycon

    >>> from jinja2.sandbox import SandboxedEnvironment
    >>> env = SandboxedEnvironment()
    >>> func = lambda: "Hello, Sandbox!"
    >>> env.from_string("{{ func() }}").render(func=func)
    'Hello, Sandbox!'
    >>> env.from_string("{{ func.__code__.co_code }}").render(func=func)
    Traceback (most recent call last):
      ...
    SecurityError: access to attribute '__code__' of 'function' object is unsafe.

A sandboxed environment can be useful, for example, to allow users of an
internal reporting system to create custom emails. You would document
what data is available in the templates, then the user would write a
template using that information. Your code would generate the report
data and pass it to the user's sandboxed template to render.


Security Considerations
-----------------------

The sandbox alone is not a solution for perfect security. Keep these
things in mind when using the sandbox.

Templates can still raise errors when compiled or rendered. Your code
should attempt to catch errors instead of crashing.

It is possible to construct a relatively small template that renders to
a very large amount of output, which could correspond to a high use of
CPU or memory. You should run your application with limits on resources
such as CPU and memory to mitigate this.

Jinja only renders text, it does not understand, for example, JavaScript
code. Depending on how the rendered template will be used, you may need
to do other postprocessing to restrict the output.

Pass only the data that is relevant to the template. Avoid passing
global data, or objects with methods that have side effects. By default
the sandbox prevents private and internal attribute access. You can
override :meth:`~SandboxedEnvironment.is_safe_attribute` to further
restrict attributes access. Decorate methods with :func:`unsafe` to
prevent calling them from templates when passing objects as data. Use
:class:`ImmutableSandboxedEnvironment` to prevent modifying lists and
dictionaries.


Authorizing Specific Methods
----------------------------

When an object has a few methods that are safe to call from a template
but others that are not, :meth:`~SandboxedEnvironment.authorize` wraps
the object in a proxy that only exposes the methods you list. The
original object is not modified, and the sandbox's normal safety checks
still apply.

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

The proxy re-checks the permissions on every call, so updating the
allowed set takes effect immediately, even for methods that were
extracted from the proxy before the update. Sync and async rendering
behave identically.

.. code-block:: python

    >>> permissions.remove("read")
    >>> env.from_string("{{ service.read() }}").render(service=service)
    Traceback (most recent call last):
      ...
    SecurityError: call to method 'read' of 'Service' object is not
        permitted; the authorization was revoked.

See :meth:`~SandboxedEnvironment.authorize` for the full API. A runnable
demo is available in the ``examples/authorized_proxy.py`` file.


API
---

.. module:: jinja2.sandbox

.. autoclass:: SandboxedEnvironment([options])
    :members: authorize, is_safe_attribute, is_safe_callable,
              default_binop_table, default_unop_table,
              intercepted_binops, intercepted_unops,
              call_binop, call_unop

.. autoclass:: ImmutableSandboxedEnvironment([options])

.. autoexception:: SecurityError

.. autofunction:: unsafe

.. autofunction:: is_internal_attribute

.. autofunction:: modifies_known_mutable


Operator Intercepting
---------------------

For performance, Jinja outputs operators directly when compiling. This
means it's not possible to intercept operator behavior by overriding
:meth:`SandboxEnvironment.call <Environment.call>` by default, because
operator special methods are handled by the Python interpreter, and
might not correspond with exactly one method depending on the operator's
use.

The sandbox can instruct the compiler to output a function to intercept
certain operators instead. Override
:attr:`SandboxedEnvironment.intercepted_binops` and
:attr:`SandboxedEnvironment.intercepted_unops` with the operator symbols
you want to intercept. The compiler will replace the symbols with calls
to :meth:`SandboxedEnvironment.call_binop` and
:meth:`SandboxedEnvironment.call_unop` instead. The default
implementation of those methods will use
:attr:`SandboxedEnvironment.binop_table` and
:attr:`SandboxedEnvironment.unop_table` to translate operator symbols
into :mod:`operator` functions.

For example, the power (``**``) operator can be disabled:

.. code-block:: python

    from jinja2.sandbox import SandboxedEnvironment

    class MyEnvironment(SandboxedEnvironment):
        intercepted_binops = frozenset(["**"])

        def call_binop(self, context, operator, left, right):
            if operator == "**":
                return self.undefined("The power (**) operator is unavailable.")

            return super().call_binop(self, context, operator, left, right)
