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


class AuthorizedService:
    def read(self):
        return "read"

    async def aread(self):
        return "async read"

    def write(self):
        return "write"

    def _secret(self):
        return "secret"

    @unsafe
    def danger(self):
        return "danger"


class TestAuthorizedProxy:
    def test_authorized_method_call(self):
        env = SandboxedEnvironment()
        service = env.authorize(AuthorizedService(), {"read"})
        assert env.from_string("{{ service.read() }}").render(service=service) == "read"

    def test_unauthorized_method_rejected(self):
        env = SandboxedEnvironment()
        service = env.authorize(AuthorizedService(), {"read"})
        with pytest.raises(SecurityError, match="not permitted"):
            env.from_string("{{ service.write() }}").render(service=service)

    def test_revocation_after_extraction(self):
        env = SandboxedEnvironment()
        permissions = {"read"}
        service = env.authorize(AuthorizedService(), permissions)
        template = env.from_string("{% set m = service.read %}{{ m() }}")
        assert template.render(service=service) == "read"
        # Extract the method proxy before revoking.
        method = service.read
        assert method() == "read"
        permissions.remove("read")
        # The extracted proxy obeys the new rules on every call.
        with pytest.raises(SecurityError, match="revoked"):
            method()
        # Re-rendering the template is also rejected.
        with pytest.raises(SecurityError, match="not permitted"):
            template.render(service=service)

    def test_private_attribute_denied(self):
        env = SandboxedEnvironment()
        service = env.authorize(AuthorizedService(), {"read"})
        with pytest.raises(SecurityError, match="not permitted"):
            env.from_string("{{ service._secret() }}").render(service=service)

    def test_unsafe_callable_denied(self):
        env = SandboxedEnvironment()
        service = env.authorize(AuthorizedService(), {"read", "danger"})
        with pytest.raises(SecurityError, match="not safely callable"):
            env.from_string("{{ service.danger() }}").render(service=service)

    def test_original_object_not_modified(self):
        env = SandboxedEnvironment()
        original = AuthorizedService()
        env.authorize(original, {"read"})
        assert not hasattr(original, "_AuthorizedObject__allowed")
        assert original.read() == "read"

    def test_sync_and_async_consistency(self, run_async_fn):
        template = (
            "{% if mode == 'read' %}{{ service.read() }}"
            "{% elif mode == 'aread' %}{{ service.aread() }}"
            "{% elif mode == 'secret' %}{{ service._secret() }}"
            "{% elif mode == 'danger' %}{{ service.danger() }}"
            "{% endif %}"
        )

        # A sync method works when rendered synchronously in both
        # environment types (render() runs the async loop if needed).
        for env in (SandboxedEnvironment(), SandboxedEnvironment(enable_async=True)):
            service = env.authorize(AuthorizedService(), {"read", "aread"})
            t = env.from_string(template)
            assert t.render(service=service, mode="read") == "read"
            with pytest.raises(SecurityError):
                t.render(service=service, mode="secret")
            with pytest.raises(SecurityError):
                t.render(service=service, mode="danger")

        # Async rendering awaits async methods and rejects the rest.
        async def check_async():
            env = SandboxedEnvironment(enable_async=True)
            service = env.authorize(AuthorizedService(), {"read", "aread"})
            t = env.from_string(template)
            rv = []
            rv.append(await t.render_async(service=service, mode="read"))
            rv.append(await t.render_async(service=service, mode="aread"))
            with pytest.raises(SecurityError):
                await t.render_async(service=service, mode="secret")
            with pytest.raises(SecurityError):
                await t.render_async(service=service, mode="danger")
            return rv

        assert run_async_fn(check_async) == ["read", "async read"]


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
