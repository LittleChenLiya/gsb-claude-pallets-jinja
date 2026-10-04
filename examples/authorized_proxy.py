"""Exposing only a few methods of an object to a sandboxed template.

The SandboxedEnvironment normally lets templates access any attribute of
the objects passed to them.  When an object has methods that are safe to
call and others that are not, :meth:`SandboxedEnvironment.authorize` can
wrap the object in a proxy that only exposes the methods you list.

This demo shows both outcomes: an authorized method call succeeds, while
an unauthorized one is rejected with a diagnosable :class:`SecurityError`.
Revoking a permission takes effect immediately, even for methods that
were extracted from the proxy before the revocation.
"""

from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment


class Service:
    def read(self):
        return "the read result"

    def write(self):
        return "the write result"


env = SandboxedEnvironment()

# Only expose ``read`` to templates.  The set is shared with the proxy,
# so mutating it later updates the permissions live.
permissions = {"read"}
service = env.authorize(Service(), permissions)

template = env.from_string("{{ service.read() }}")
print("authorized call:", template.render(service=service))

try:
    env.from_string("{{ service.write() }}").render(service=service)
except SecurityError as exc:
    print("unauthorized call rejected:", exc)

# Extract the method proxy before revoking the permission.
method = service.read
print("extracted method still works:", method())

permissions.remove("read")

try:
    method()
except SecurityError as exc:
    print("revoked call rejected:", exc)
