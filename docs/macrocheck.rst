Static Macro Contract Checking
===============================

.. module:: jinja2.macrocheck

Jinja ships a static macro contract checker that analyzes templates
*without executing them*. It collects local macro definitions, static
``import`` and ``from ... import`` aliases, and every call site it can
resolve, then validates the calls against the macro signatures.

The checker never renders a template and never calls a filter, test,
or user global function.

.. code-block:: python

    from jinja2 import Environment, FileSystemLoader
    from jinja2.macrocheck import check_templates

    env = Environment(loader=FileSystemLoader("templates"))
    report = check_templates(env)

    if not report.ok:
        for issue in report.errors:
            print(issue)  # "page.html:12: call to macro ..."

    for issue in report.unknowns:
        print(issue)  # call sites that cannot be decided statically

    print(report.sentinels)  # safety sentinel counts

Every issue carries the template name and the line number of the call
or import it was reported for.

What is checked
---------------

For every call that can be resolved statically, the checker reports an
error for:

-   a missing required argument (a parameter without a default that is
    not filled positionally or by keyword)
-   more positional arguments than the macro declares
-   a keyword argument passed more than once
-   an argument passed both positionally and by keyword
-   a keyword argument that does not exist in the macro signature
-   a macro that does not exist in a statically imported template

Defaults, ``caller``, ``varargs``, and ``kwargs`` follow the compiler's
actual rules: a default only needs to *exist* (its expression is never
evaluated during static analysis), a macro whose body accesses
``varargs`` accepts extra positional arguments, one that accesses
``kwargs`` accepts extra keyword arguments, and one that accesses
``caller`` may receive a ``caller`` keyword argument.

What is unknown
---------------

Call and import sites that cannot be decided statically are listed
separately as *unknowns* instead of errors:

-   imports with a dynamic template name
-   dynamic attribute access on an imported template
-   argument unpacking (``*args`` or ``**kwargs``)
-   ``{% call %}`` blocks, which pass an implicit ``caller``
-   macro or alias names shadowed by a local assignment

Templates that import each other, and macros that call each other, are
safe: every template's AST is walked exactly once and calls are
validated against collected signatures rather than expanded, so the
analysis always terminates.

A runnable example that prints a static report and the safety sentinel
counts lives in ``examples/macrocheck`` in the source distribution.

API
---

.. autofunction:: check_templates

.. autoclass:: MacroReport
    :members:

.. autoclass:: MacroIssue
    :members:

