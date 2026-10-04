import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import LoopBudgetExhausted

LOOP_BUDGET_EXT = "jinja2.ext.LoopBudgetExtension"

#: outer loop runs 2 times, the inner loop runs 3 times per outer
#: iteration, so a render consumes exactly 8 items.
NESTED = "{% for a in outer %}{% for b in a %}x{% endfor %}{% endfor %}"


def budget_env(budget, templates=None):
    env = Environment(
        loader=DictLoader(templates or {}), extensions=[LOOP_BUDGET_EXT]
    )
    env.loop_budget = budget
    return env


def test_budget_exactly_consumed_renders():
    """Outer 2 x inner 3 = 8 items, a budget of 8 is enough."""
    env = budget_env(8)
    tpl = env.from_string(NESTED)
    assert tpl.render(outer=[[0, 0, 0], [0, 0, 0]]) == "xxxxxx"


def test_budget_one_short_fails():
    """The same template fails with a budget of 7."""
    env = budget_env(7)
    tpl = env.from_string(NESTED)

    with pytest.raises(LoopBudgetExhausted):
        tpl.render(outer=[[0, 0, 0], [0, 0, 0]])


def test_side_effect_generator_not_pulled_beyond_budget():
    """A generator with side effects must not be asked for a 9th item
    when the budget of 7 is exceeded by the 8th."""
    pulled = []

    def side_effect(n, tag):
        for i in range(n):
            pulled.append((tag, i))
            yield i

    env = budget_env(7)
    tpl = env.from_string(NESTED)
    outer = (side_effect(3, "inner") for _ in side_effect(2, "outer"))

    with pytest.raises(LoopBudgetExhausted):
        tpl.render(outer=outer)

    # 2 outer + 3 + 3 inner items were taken out, the 9th never was.
    assert len(pulled) == 8


def test_nested_loops_share_budget_across_levels():
    env = budget_env(5)
    tpl = env.from_string(NESTED)

    with pytest.raises(LoopBudgetExhausted):
        tpl.render(outer=[[0, 0], [0, 0], [0, 0]])


def test_macro_loops_count_towards_same_render():
    env = budget_env(4)
    tpl = env.from_string(
        "{% macro m(items) %}{% for i in items %}m{% endfor %}{% endmacro %}"
        "{{ m([1, 2, 3]) }}{% for x in [1, 2] %}t{% endfor %}"
    )

    with pytest.raises(LoopBudgetExhausted):
        tpl.render()

    env.loop_budget = 5
    assert tpl.render() == "mmmtt"


def test_include_loops_count_towards_same_render():
    env = budget_env(
        4,
        {
            "main": "{% include 'sub' %}{% for x in [1, 2, 3] %}t{% endfor %}",
            "sub": "{% for y in [1, 2] %}s{% endfor %}",
        },
    )

    with pytest.raises(LoopBudgetExhausted):
        env.get_template("main").render()

    env.loop_budget = 5
    assert env.get_template("main").render() == "ssttt"


def test_error_reports_included_template_name_and_line():
    env = budget_env(
        1,
        {
            "main": "{% include 'sub' %}",
            "sub": "first line\n{% for y in [1, 2] %}s{% endfor %}",
        },
    )

    with pytest.raises(LoopBudgetExhausted) as exc_info:
        env.get_template("main").render()

    err = exc_info.value
    assert err.name == "sub"
    assert err.lineno == 2
    assert "sub" in str(err)
    assert "2" in str(err)


def test_error_contains_template_name_and_line():
    env = budget_env(1, {"page.html": "hello\n{% for x in [1, 2, 3] %}x{% endfor %}"})

    with pytest.raises(LoopBudgetExhausted) as exc_info:
        env.get_template("page.html").render()

    err = exc_info.value
    assert err.name == "page.html"
    assert err.lineno == 2
    assert "page.html" in str(err)
    assert "line 2" in str(err)


def test_budget_is_fresh_for_each_render():
    env = budget_env(4)
    tpl = env.from_string("{% for x in [1, 2, 3, 4] %}x{% endfor %}")

    assert tpl.render() == "xxxx"
    # budget state is released after a render, the next one starts fresh
    assert tpl.render() == "xxxx"

    env.loop_budget = 3
    with pytest.raises(LoopBudgetExhausted):
        tpl.render()

    # a failed render does not keep budget state around either
    env.loop_budget = 4
    assert tpl.render() == "xxxx"


def test_no_extension_keeps_original_behavior():
    env = Environment()
    tpl = env.from_string("{% for x in range(10000) %}x{% endfor %}")
    assert len(tpl.render()) == 10000
    assert not hasattr(env, "loop_budget")


def test_extension_without_budget_is_unlimited():
    env = Environment(extensions=[LOOP_BUDGET_EXT])
    assert env.loop_budget is None
    tpl = env.from_string("{% for x in range(10000) %}x{% endfor %}")
    assert len(tpl.render()) == 10000


def test_loop_variable_still_works():
    env = budget_env(3)
    tpl = env.from_string("{% for x in [1, 2, 3] %}{{ loop.index }}{% endfor %}")
    assert tpl.render() == "123"

    env.loop_budget = 2
    with pytest.raises(LoopBudgetExhausted):
        tpl.render()


def test_async_loop_budget(run_async_fn):
    env = Environment(extensions=[LOOP_BUDGET_EXT], enable_async=True)
    env.loop_budget = 2
    tpl = env.from_string("{% for x in [1, 2, 3] %}x{% endfor %}")

    with pytest.raises(LoopBudgetExhausted):
        run_async_fn(tpl.render_async)

    env.loop_budget = 3
    assert run_async_fn(tpl.render_async) == "xxx"
