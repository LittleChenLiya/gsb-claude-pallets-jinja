import types

import pytest
from markupsafe import escape

from jinja2 import Environment
from jinja2.exceptions import SecurityError
from jinja2.exceptions import TemplateRuntimeError
from jinja2.exceptions import TemplateSyntaxError
from jinja2.nodes import EvalContext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.sandbox import unsafe


class PrivateStuff:
    def bar(self):
        return 23

    @unsafe
    def foo(self):
        return 42

    def __repr__(self):
        return "PrivateStuff"


class PublicStuff:
    def bar(self):
        return 23

    def _foo(self):
        return 42

    def __repr__(self):
        return "PublicStuff"


class TestSandbox:
    def test_unsafe(self, env):
        env = SandboxedEnvironment()
        pytest.raises(
            SecurityError, env.from_string("{{ foo.foo() }}").render, foo=PrivateStuff()
        )
        assert env.from_string("{{ foo.bar() }}").render(foo=PrivateStuff()) == "23"

        pytest.raises(
            SecurityError, env.from_string("{{ foo._foo() }}").render, foo=PublicStuff()
        )
        assert env.from_string("{{ foo.bar() }}").render(foo=PublicStuff()) == "23"
        assert env.from_string("{{ foo.__class__ }}").render(foo=42) == ""
        assert env.from_string("{{ foo.func_code }}").render(foo=lambda: None) == ""
        # security error comes from __class__ already.
        pytest.raises(
            SecurityError,
            env.from_string("{{ foo.__class__.__subclasses__() }}").render,
            foo=42,
        )

    def test_immutable_environment(self, env):
        env = ImmutableSandboxedEnvironment()
        pytest.raises(SecurityError, env.from_string("{{ [].append(23) }}").render)
        pytest.raises(SecurityError, env.from_string("{{ [].clear() }}").render)
        pytest.raises(SecurityError, env.from_string("{{ [1].pop() }}").render)
        pytest.raises(SecurityError, env.from_string("{{ {1:2}.clear() }}").render)

    def test_restricted(self, env):
        env = SandboxedEnvironment()
        pytest.raises(
            TemplateSyntaxError,
            env.from_string,
            "{% for item.attribute in seq %}...{% endfor %}",
        )
        pytest.raises(
            TemplateSyntaxError,
            env.from_string,
            "{% for foo, bar.baz in seq %}...{% endfor %}",
        )

    def test_template_data(self, env):
        env = Environment(autoescape=True)
        t = env.from_string(
            "{% macro say_hello(name) %}"
            "<p>Hello {{ name }}!</p>{% endmacro %}"
            '{{ say_hello("<blink>foo</blink>") }}'
        )
        escaped_out = "<p>Hello &lt;blink&gt;foo&lt;/blink&gt;!</p>"
        assert t.render() == escaped_out
        assert str(t.module) == escaped_out
        assert escape(t.module) == escaped_out
        assert t.module.say_hello("<blink>foo</blink>") == escaped_out
        assert (
            escape(t.module.say_hello(EvalContext(env), "<blink>foo</blink>"))
            == escaped_out
        )
        assert escape(t.module.say_hello("<blink>foo</blink>")) == escaped_out

    def test_attr_filter(self, env):
        env = SandboxedEnvironment()
        tmpl = env.from_string('{{ cls|attr("__subclasses__")() }}')
        pytest.raises(SecurityError, tmpl.render, cls=int)

    def test_binary_operator_intercepting(self, env):
        def disable_op(left, right):
            raise TemplateRuntimeError("that operator so does not work")

        for expr, ctx, rv in ("1 + 2", {}, "3"), ("a + 2", {"a": 2}, "4"):
            env = SandboxedEnvironment()
            env.binop_table["+"] = disable_op
            t = env.from_string(f"{{{{ {expr} }}}}")
            assert t.render(ctx) == rv
            env.intercepted_binops = frozenset(["+"])
            t = env.from_string(f"{{{{ {expr} }}}}")
            with pytest.raises(TemplateRuntimeError):
                t.render(ctx)

    def test_unary_operator_intercepting(self, env):
        def disable_op(arg):
            raise TemplateRuntimeError("that operator so does not work")

        for expr, ctx, rv in ("-1", {}, "-1"), ("-a", {"a": 2}, "-2"):
            env = SandboxedEnvironment()
            env.unop_table["-"] = disable_op
            t = env.from_string(f"{{{{ {expr} }}}}")
            assert t.render(ctx) == rv
            env.intercepted_unops = frozenset(["-"])
            t = env.from_string(f"{{{{ {expr} }}}}")
            with pytest.raises(TemplateRuntimeError):
                t.render(ctx)


class TestStringFormat:
    def test_basic_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{0.__class__}b".format(42) }}')
        assert t.render() == "ab"

    def test_basic_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{0.foo}b".format({"foo": 42}) }}')
        assert t.render() == "a42b"

    def test_safe_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ ("a{0.__class__}b{1}"|safe).format(42, "<foo>") }}')
        assert t.render() == "ab&lt;foo&gt;"

    def test_safe_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ ("a{0.foo}b{1}"|safe).format({"foo": 42}, "<foo>") }}')
        assert t.render() == "a42b&lt;foo&gt;"

    def test_empty_braces_format(self):
        env = SandboxedEnvironment()
        t1 = env.from_string('{{ ("a{}b{}").format("foo", "42")}}')
        t2 = env.from_string('{{ ("a{}b{}"|safe).format(42, "<foo>") }}')
        assert t1.render() == "afoob42"
        assert t2.render() == "a42b&lt;foo&gt;"


class TestStringFormatMap:
    def test_basic_format_safety(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{x.__class__}b".format_map({"x":42}) }}')
        assert t.render() == "ab"

    def test_basic_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string('{{ "a{x.foo}b".format_map({"x":{"foo": 42}}) }}')
        assert t.render() == "a42b"

    def test_safe_format_all_okay(self):
        env = SandboxedEnvironment()
        t = env.from_string(
            '{{ ("a{x.foo}b{y}"|safe).format_map({"x":{"foo": 42}, "y":"<foo>"}) }}'
        )
        assert t.render() == "a42b&lt;foo&gt;"

    def test_indirect_call(self):
        def run(value, arg):
            return value.run(arg)

        env = SandboxedEnvironment()
        env.filters["run"] = run
        t = env.from_string(
            """{% set
                ns = namespace(run="{0.__call__.__builtins__[__import__]}".format)
            %}
            {{ ns | run(not_here) }}
            """
        )

        with pytest.raises(SecurityError):
            t.render()

    def test_attr_filter(self) -> None:
        env = SandboxedEnvironment()
        t = env.from_string(
            """{{ "{0.__call__.__builtins__[__import__]}"
                  | attr("format")(not_here) }}"""
        )

        with pytest.raises(SecurityError):
            t.render()


class ProxyService:
    """An object with a mix of read, write, private, and unsafe methods,
    used to exercise AuthorizedMethodProxy.
    """

    def __init__(self):
        self.writes = 0

    def read(self):
        return "data"

    def write(self):
        self.writes += 1
        return "written"

    def _secret(self):
        return "secret"

    @unsafe
    def reset(self):
        return "reset"


class AsyncProxyService:
    async def read(self):
        return "data"

    async def write(self):
        return "written"

    async def _secret(self):
        return "secret"

    @unsafe
    async def reset(self):
        return "reset"


# The same small templates are rendered through the sync and async paths.
PROXY_READ_TEMPLATE = "{{ service.read() }}"
PROXY_WRITE_TEMPLATE = "{{ service.write() }}"
PROXY_PRIVATE_TEMPLATE = "{{ service._secret() }}"
PROXY_UNSAFE_TEMPLATE = "{{ service.reset() }}"


class TestAuthorizedMethodProxy:
    def test_allowed_method_call(self):
        env = SandboxedEnvironment()
        proxy = env.authorized_proxy(ProxyService(), allowed={"read"})
        assert env.from_string(PROXY_READ_TEMPLATE).render(service=proxy) == "data"
        # item access goes through the same authorization
        assert (
            env.from_string("{{ service['read']() }}").render(service=proxy) == "data"
        )

    def test_unauthorized_method_rejected(self):
        env = SandboxedEnvironment()
        service = ProxyService()
        proxy = env.authorized_proxy(service, allowed={"read"})

        with pytest.raises(SecurityError):
            env.from_string(PROXY_WRITE_TEMPLATE).render(service=proxy)

        # the wrapped object is never called or modified
        assert service.writes == 0

    def test_private_attribute_rejected(self):
        env = SandboxedEnvironment()
        # even explicitly allowing a private name cannot bypass the
        # environment's regular is_safe_attribute check
        proxy = env.authorized_proxy(ProxyService(), allowed={"read", "_secret"})

        with pytest.raises(SecurityError):
            env.from_string(PROXY_PRIVATE_TEMPLATE).render(service=proxy)

    def test_unsafe_callable_marker_still_applies(self):
        env = SandboxedEnvironment()
        proxy = env.authorized_proxy(ProxyService(), allowed={"read", "reset"})

        with pytest.raises(SecurityError):
            env.from_string(PROXY_UNSAFE_TEMPLATE).render(service=proxy)

    def test_attribute_read_does_not_expose_raw_method(self):
        env = SandboxedEnvironment()
        service = ProxyService()
        proxy = env.authorized_proxy(service, allowed={"read"})
        value = env.getattr(proxy, "read")
        # an authorized read returns a re-checking proxy, not the raw
        # bound method of the wrapped object
        assert not isinstance(value, types.MethodType)
        assert value() == "data"
        # an unauthorized read is rejected instead of exposing the method
        pytest.raises(SecurityError, env.getattr, proxy, "write")
        pytest.raises(SecurityError, env.getattr, proxy, "_secret")

    def test_revocation_applies_to_retrieved_method(self):
        env = SandboxedEnvironment()
        proxy = env.authorized_proxy(ProxyService(), allowed={"read"})
        # the template takes the read proxy first, then the permission is
        # revoked, and the already retrieved proxy must obey the new rules
        t = env.from_string("{% set r = service.read %}{{ revoke() }}{{ r() }}")

        with pytest.raises(SecurityError):
            t.render(service=proxy, revoke=lambda: proxy._revoke("read"))

    def test_revoke_and_reallow(self):
        env = SandboxedEnvironment()
        proxy = env.authorized_proxy(ProxyService(), allowed={"read"})
        t = env.from_string(PROXY_READ_TEMPLATE)
        assert t.render(service=proxy) == "data"
        proxy._revoke("read")

        with pytest.raises(SecurityError):
            t.render(service=proxy)

        proxy._allow("read")
        assert t.render(service=proxy) == "data"

    def test_caller_provided_set_is_live(self):
        env = SandboxedEnvironment()
        allowed = {"read"}
        proxy = env.authorized_proxy(ProxyService(), allowed=allowed)
        t = env.from_string(PROXY_READ_TEMPLATE)
        assert t.render(service=proxy) == "data"
        allowed.discard("read")

        with pytest.raises(SecurityError):
            t.render(service=proxy)

    def test_plain_object_behavior_unchanged(self):
        env = SandboxedEnvironment()
        service = ProxyService()
        # without a proxy the regular sandbox rules apply, write is a
        # normal callable method
        assert env.from_string(PROXY_WRITE_TEMPLATE).render(service=service) == (
            "written"
        )
        assert service.writes == 1

    def test_async(self, run_async_fn):
        env = SandboxedEnvironment(enable_async=True)
        proxy = env.authorized_proxy(AsyncProxyService(), allowed={"read"})

        async def render(source):
            return await env.from_string(source).render_async(service=proxy)

        assert run_async_fn(render, PROXY_READ_TEMPLATE) == "data"

        for source in (
            PROXY_WRITE_TEMPLATE,
            PROXY_PRIVATE_TEMPLATE,
            PROXY_UNSAFE_TEMPLATE,
        ):
            with pytest.raises(SecurityError):
                run_async_fn(render, source)

    def test_async_revocation(self, run_async_fn):
        env = SandboxedEnvironment(enable_async=True)
        proxy = env.authorized_proxy(AsyncProxyService(), allowed={"read"})
        t = env.from_string("{% set r = service.read %}{{ revoke() }}{{ r() }}")

        async def render():
            return await t.render_async(
                service=proxy, revoke=lambda: proxy._revoke("read")
            )

        with pytest.raises(SecurityError):
            run_async_fn(render)
