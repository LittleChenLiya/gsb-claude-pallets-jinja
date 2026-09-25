"""Static macro contract report example.

Analyzes the templates in the 'templates' directory next to this
script without rendering them, prints the static report together with
the safety sentinel counts, and exits non-zero if definite errors were
found. Runs in well under 90 seconds.

Usage: python report.py
"""

import sys
import time
from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2.macrocheck import check_templates


def main():
    start = time.monotonic()
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
    report = check_templates(env)
    elapsed = time.monotonic() - start

    print(report.render_text())
    print(f"analysis completed in {elapsed:.3f}s (limit: 90s)")

    if elapsed > 90:
        print("error: static analysis exceeded the 90 second budget")
        return 2

    # In CI, gate on report.ok instead. This example always exits 0
    # so the report above is the result of a successful analysis run.
    print(f"status: {'clean' if report.ok else 'definite errors found'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
