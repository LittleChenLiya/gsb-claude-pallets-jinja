"""Command line entry point for Jinja2.

Currently exposes the read-only macro contract exporter::

    python -m jinja2 template.html
    python -m jinja2 template.html --json
    python -m jinja2 contracts template.html

The first form prints a human readable report, the second a JSON
document.  Nothing is loaded or executed: default expressions are never
evaluated and dynamic includes are only marked as unknown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts import export_macro_contracts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jinja2",
        description="Read-only Jinja2 template inspection tools.",
    )
    parser.add_argument("template", nargs="?", help="path to the template file")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the report as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Allow an explicit ``contracts`` subcommand while keeping
    # ``python -m jinja2 <file>`` working.
    if argv is not None and argv and argv[0] == "contracts":
        argv = argv[1:]
    elif argv is None and sys.argv[1:2] == ["contracts"]:
        argv = sys.argv[2:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    template = args.template
    if not template:
        parser.print_help()
        return 2

    path = Path(template)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {template!r}: {exc}", file=sys.stderr)
        return 2

    report = export_macro_contracts(source, name=str(path), filename=str(path))

    if args.as_json:
        print(report.to_json())
    else:
        print(str(report))

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
