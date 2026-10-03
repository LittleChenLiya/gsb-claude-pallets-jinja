"""Warm up the bytecode cache for a batch of templates ahead of time.

The prepare phase compiles the explicitly named templates into a staging
area without touching the real cache.  The commit phase publishes them all
at once, but only if the templates' source and the compile configuration
did not change in the meantime.  Templates are only compiled, never
executed, so global functions in the templates are not called.
"""

import os
import tempfile

from jinja2 import Environment
from jinja2 import FileSystemBytecodeCache
from jinja2.loaders import DictLoader

templates = {
    "hello.html": "Hello {{ name }}!",
    "bye.html": "Bye {{ name }}!",
}

with tempfile.TemporaryDirectory() as cache_dir:
    cache = FileSystemBytecodeCache(cache_dir)
    env = Environment(loader=DictLoader(templates), bytecode_cache=cache)

    def print_cache_state(label):
        files = sorted(os.listdir(cache_dir))
        print(f"{label}: {len(files)} cached template(s)")
        for filename in files:
            print(f"  {filename}")

    print_cache_state("before prepare")

    batch = env.prepare_template_batch(["hello.html", "bye.html"])
    print_cache_state("after prepare")  # nothing written to the cache yet

    batch.commit()
    print_cache_state("after commit")  # both templates are cached

    # rendering now hits the cache, nothing is recompiled
    print(env.get_template("hello.html").render(name="World"))
    print(env.get_template("bye.html").render(name="World"))
