# Macro contract check example

`check_macros.py` demonstrates the static macro contract checker
(`jinja2.macrocheck`).  It analyzes a small template set without
executing it:

* `lib_a.html` and `lib_b.html` import macros from each other.
* `good.html` only makes legal calls (including default arguments).
* `bad_missing.html` misses one required argument.
* `unknowns.html` exercises constructs reported as *unknown*:
  shadowed names, dynamic template names, argument unpacking, and a
  call block on a dynamic callee.

Run it:

```
$ python examples/macrocheck/check_macros.py
```

It prints one static report per page -- templates checked, definite
diagnostics with template name and line number, unknowns listed
separately, and the safety sentinel count (places where the checker
deliberately refused to evaluate anything: macro defaults, filters,
tests, and calls to non-macro names).  It finishes in well under 90
seconds and exits with status 1 because `bad_missing.html` has a
definite violation.
