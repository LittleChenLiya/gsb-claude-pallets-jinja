"""Tests for the loop budget extension.

The loop budget limits the total number of elements that all ``for`` loops
may fetch in a single render.  Nested loops, as well as loops inside
includes and macros, share the same budget.  The check happens before the
element is consumed, and the underlying iterable is never materialized
into a list.
"""

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import LoopBudgetExceeded
from jinja2.ext import LoopBudgetExtension


class TestLoopBudget:
    def test_budget_7_fails(self):
        """Budget 7 fails when 8 elements are fetched (2 outer + 3 + 3 inner).

        Outer loop runs 2 times, each inner loop runs 3 times.  The total
        number of elements fetched is 2 + 3 + 3 = 8, so budget 7 is exceeded
        on the 8th fetch.
        """
        env = Environment(extensions=[LoopBudgetExtension])
        env.loop_budget = 7
        tmpl = env.from_string(
            "{% for i in outer %}{% for j in inner %}{{ i }}-{{ j }} "
            "{% endfor %}{% endfor %}"
        )
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render(outer=[1, 2], inner=[1, 2, 3])
        assert exc_info.value.budget == 7

    def test_budget_8_completes(self):
        """Budget 8 completes when exactly 8 elements are fetched."""
        env = Environment(extensions=[LoopBudgetExtension])
        env.loop_budget = 8
        tmpl = env.from_string(
            "{% for i in outer %}{% for j in inner %}{{ i }}-{{ j }} "
            "{% endfor %}{% endfor %}"
        )
        out = tmpl.render(outer=[1, 2], inner=[1, 2, 3])
        assert out == "1-1 1-2 1-3 2-1 2-2 2-3 "

    def test_consecutive_renders_fresh_budget(self):
        """Consecutive renders start with a fresh budget.

        Each render creates a new budget, so a render that completes does
        not consume quota from the next render.
        """
        env = Environment(extensions=[LoopBudgetExtension])
        env.loop_budget = 8
        tmpl = env.from_string("{% for i in items %}{{ i }}{% endfor %}")
        # First render completes with 8 elements
        out1 = tmpl.render(items=[1, 2, 3, 4, 5, 6, 7, 8])
        assert out1 == "12345678"
        # Second render also completes with a fresh budget
        out2 = tmpl.render(items=[1, 2, 3, 4, 5, 6, 7, 8])
        assert out2 == "12345678"

    def test_include_loop_counts_toward_budget(self):
        """Loops in included templates count toward the same render budget."""
        env = Environment(
            loader=DictLoader(
                {
                    "main.html": "{% include 'inc.html' %}",
                    "inc.html": "{% for i in items %}{{ i }}{% endfor %}",
                }
            ),
            extensions=[LoopBudgetExtension],
        )
        env.loop_budget = 3
        tmpl = env.get_template("main.html")
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render(items=[1, 2, 3, 4])
        assert exc_info.value.budget == 3
        assert exc_info.value.name == "inc.html"

    def test_macro_loop_counts_toward_budget(self):
        """Loops in macros count toward the same render budget."""
        env = Environment(extensions=[LoopBudgetExtension])
        env.loop_budget = 3
        tmpl = env.from_string(
            "{% macro m(items) %}{% for i in items %}{{ i }}{% endfor %}"
            "{% endmacro %}{{ m([1, 2, 3, 4]) }}"
        )
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render()
        assert exc_info.value.budget == 3

    def test_exception_contains_template_name_and_line_number(self):
        """The exception carries the template name and line number of the
        ``for`` statement that exceeded the budget.
        """
        env = Environment(
            loader=DictLoader(
                {
                    "tmpl.html": "line1\nline2\n{% for i in items %}{{ i }}"
                    "{% endfor %}",
                }
            ),
            extensions=[LoopBudgetExtension],
        )
        env.loop_budget = 1
        tmpl = env.get_template("tmpl.html")
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render(items=[1, 2])
        assert exc_info.value.name == "tmpl.html"
        assert exc_info.value.lineno == 3

    def test_generator_not_fetched_extra_item(self):
        """A side-effect generator is not fetched a 9th time.

        The generator yields exactly 8 elements.  The for loop fetches
        those 8 elements and then calls ``next`` once more to detect
        exhaustion, which raises ``StopIteration`` without executing the
        generator's side effect.  The side effect therefore runs only 8
        times.
        """
        env = Environment(extensions=[LoopBudgetExtension])
        env.loop_budget = 8

        def gen():
            for i in range(8):
                gen.count += 1
                yield i

        gen.count = 0
        tmpl = env.from_string("{% for i in gen() %}{{ i }}{% endfor %}")
        out = tmpl.render(gen=gen)
        assert out == "01234567"
        assert gen.count == 8

    def test_no_budget_keeps_original_behavior(self):
        """Environments without a budget retain the original behavior."""
        env = Environment()
        tmpl = env.from_string(
            "{% for i in outer %}{% for j in inner %}{{ i }}-{{ j }} "
            "{% endfor %}{% endfor %}"
        )
        out = tmpl.render(outer=[1, 2], inner=[1, 2, 3])
        assert out == "1-1 1-2 1-3 2-1 2-2 2-3 "

    def test_policy_configuration(self):
        """The budget can be configured via the ``ext.loop_budget.max``
        policy instead of the ``loop_budget`` attribute.
        """
        env = Environment(extensions=[LoopBudgetExtension])
        env.policies["ext.loop_budget.max"] = 3
        tmpl = env.from_string("{% for i in items %}{{ i }}{% endfor %}")
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render(items=[1, 2, 3, 4])
        assert exc_info.value.budget == 3

    def test_budget_exceeded_exception_message(self):
        """The exception message contains the budget, template name, and
        line number.
        """
        env = Environment(
            loader=DictLoader(
                {
                    "tmpl.html": "{% for i in items %}{{ i }}{% endfor %}",
                }
            ),
            extensions=[LoopBudgetExtension],
        )
        env.loop_budget = 2
        tmpl = env.get_template("tmpl.html")
        with pytest.raises(LoopBudgetExceeded) as exc_info:
            tmpl.render(items=[1, 2, 3])
        message = str(exc_info.value)
        assert "2" in message
        assert "tmpl.html" in message
        assert "1" in message
