.. currentmodule:: jinja2.macrocheck

Static Macro Contract Checks
============================

.. module:: jinja2.macrocheck

The :mod:`jinja2.macrocheck` module analyzes templates *without
executing them*.  It collects local macro definitions, static
``{% import %}`` and ``{% from ... import %}`` aliases, and macro call
sites from the parsed AST, then checks every statically resolvable
call against the callee's signature:

*   missing required arguments (parameters without a default)
*   duplicate arguments (passed positionally *and* by keyword, or a
    repeated keyword)
*   unexpected keyword arguments and too many positional arguments
    (unless the macro accesses ``kwargs`` or ``varargs``)
*   ``{% call %}`` blocks passed to macros that do not accept a
    ``caller``

Defaults, ``caller``, ``varargs``, and ``kwargs`` are handled exactly
the way the compiler and :class:`~jinja2.runtime.Macro` handle them at
runtime.  Default expressions are only checked for existence; they are
never evaluated.  No filter, test, or user global function is ever
called.  Every place where the checker deliberately refuses to
evaluate something is counted as a *safety sentinel* and reported
through :attr:`CheckReport.sentinel_count`.

Anything that cannot be decided statically -- dynamic template names,
dynamic calls, argument unpacking (``*args`` / ``**kwargs``), or names
shadowed by local assignments -- is reported as an *unknown* in
:attr:`CheckReport.unknowns` instead of a diagnostic.  Templates that
import each other (and macros that call each other) are analyzed
exactly once, so analysis always terminates.

Each :class:`Diagnostic` carries the template name and line number of
the offending call.

.. code-block:: python

    from jinja2 import Environment
    from jinja2.loaders import FileSystemLoader
    from jinja2.macrocheck import MacroContractChecker

    env = Environment(loader=FileSystemLoader("templates"))
    report = MacroContractChecker(env).check("page.html")

    if not report.ok:
        for diag in report.diagnostics:
            print(diag)  # page.html:12: missing-argument: ...

    for unknown in report.unknowns:
        print(unknown)

    print(f"safety sentinels: {report.sentinel_count}")

A runnable example that prints a static report and the safety
sentinel count lives in ``examples/macrocheck/check_macros.py`` and
finishes in well under 90 seconds.

API
---

.. autoclass:: MacroContractChecker
    :members: check

.. autofunction:: check_templates

.. autoclass:: CheckReport
    :members: ok, format

.. autoclass:: Diagnostic

.. autoclass:: Unknown

.. autoclass:: MacroSignature

