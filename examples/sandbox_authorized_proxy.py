"""Short demo of SandboxedEnvironment.authorized_proxy: only explicitly
authorized methods of an object are exposed to sandboxed templates, and
revoking a permission also blocks method proxies retrieved earlier.

Run with: python examples/sandbox_authorized_proxy.py
"""

from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment


class ReportService:
    def read(self):
        return "quarterly report data"

    def write(self):
        return "MUTATED"


env = SandboxedEnvironment()
service = ReportService()
proxy = env.authorized_proxy(service, allowed={"read"})

# Outcome 1: the authorized method can be called from the template.
template = env.from_string("{{ service.read() }}")
print("allowed:", template.render(service=proxy))

# Outcome 2: anything not in the allowed set is rejected with SecurityError.
try:
    env.from_string("{{ service.write() }}").render(service=proxy)
except SecurityError as e:
    print("denied:", e)

# Revocation is live: even a method proxy taken earlier stops working.
reader = env.getattr(proxy, "read")
proxy._revoke("read")
try:
    template.render(service=proxy)
except SecurityError as e:
    print("revoked:", e)
try:
    reader()
except SecurityError as e:
    print("revoked (retrieved earlier):", e)
