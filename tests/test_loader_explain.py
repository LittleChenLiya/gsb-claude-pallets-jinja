import pytest

from jinja2 import ChoiceLoader
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import explain_load
from jinja2 import ExplainNode
from jinja2 import FunctionLoader
from jinja2 import LoadExplanation
from jinja2 import PrefixLoader
from jinja2.exceptions import TemplateNotFound
from jinja2.loaders import BaseLoader


class CountingLoader(BaseLoader):
    """A custom loader that counts get_source calls and then explodes.

    Used to prove that explaining never probes a custom loader through
    get_source: the count must stay at 0.
    """

    def __init__(self):
        self.calls = 0

    def get_source(self, environment, template):
        self.calls += 1
        raise RuntimeError("get_source must not be called by explain")


@pytest.fixture()
def counting_loader():
    return CountingLoader()


class TestChoiceExplain:
    def test_first_hit_shadows_second(self):
        loader = ChoiceLoader(
            [
                DictLoader({"x.html": "one"}),
                DictLoader({"x.html": "two"}),
            ]
        )
        exp = explain_load(loader, "x.html")

        assert isinstance(exp, LoadExplanation)
        assert exp.found
        assert exp.source == "ChoiceLoader > choice[0] DictLoader"

        first, second = exp.root.children
        assert first.status == "hit"
        # Both dicts contain x.html, but only the first may be marked hit.
        assert second.status == "shadowed"
        assert "shadowed" in exp.format()

    def test_second_wins_when_first_misses(self):
        loader = ChoiceLoader(
            [
                DictLoader({"other.html": "one"}),
                DictLoader({"x.html": "two"}),
            ]
        )
        exp = explain_load(loader, "x.html")

        assert exp.found
        assert exp.source == "ChoiceLoader > choice[1] DictLoader"
        first, second = exp.root.children
        assert first.status == "miss"
        assert second.status == "hit"

    def test_miss_keeps_attempted_paths(self):
        loader = ChoiceLoader(
            [
                DictLoader({"a.html": "1"}),
                DictLoader({"b.html": "2"}),
            ]
        )
        exp = explain_load(loader, "missing.html")

        assert not exp.found
        assert exp.source is None
        # Every attempted source is still recorded, in order.
        assert [n.status for n in exp.attempts] == ["miss", "miss"]
        assert [n.name for n in exp.attempts] == ["missing.html"] * 2


class TestPrefixExplain:
    def test_only_matching_branch_entered(self):
        loader = PrefixLoader(
            {
                "a": DictLoader({"x.html": "from a"}),
                "b": DictLoader({"x.html": "from b"}),
            }
        )
        exp = explain_load(loader, "a/x.html")

        assert exp.found
        assert exp.source == "PrefixLoader > prefix 'a' -> DictLoader"

        branch_a, branch_b = exp.root.children
        # The name travels into branch a with the prefix stripped.
        assert branch_a.status == "hit"
        assert branch_a.name == "x.html"
        # Branch b is not entered at all, even though it also has x.html.
        assert branch_b.status == "skipped"
        assert branch_b.children == []

        # The stripped name must not be applied to the other branch.
        stripped_in_b = [
            node for node in branch_b.walk() if node.name == "x.html"
        ]
        assert stripped_in_b == []

        text = exp.format()
        assert "name changed: 'a/x.html' -> 'x.html'" in text
        assert "skipped" in text

    def test_unknown_prefix_is_miss_without_entering_branches(self):
        loader = PrefixLoader({"a": DictLoader({"x.html": "from a"})})
        exp = explain_load(loader, "c/x.html")

        assert not exp.found
        assert all(node.status == "skipped" for node in exp.root.children)
        assert "not in mapping" in exp.root.detail

    def test_prefixed_miss_keeps_attempted_path(self):
        loader = PrefixLoader({"a": DictLoader({"x.html": "from a"})})
        exp = explain_load(loader, "a/missing.html")

        assert not exp.found
        (branch_a,) = exp.root.children
        assert branch_a.status == "miss"
        assert branch_a.name == "missing.html"


class TestCustomLoaderExplain:
    def test_unsupported_without_adapter_and_not_called(self, counting_loader):
        loader = ChoiceLoader([DictLoader({"x.html": "d"}), counting_loader])
        exp = explain_load(loader, "x.html")

        assert exp.found
        custom = exp.root.children[1]
        assert custom.status == "unsupported"
        assert "get_source was not called" in custom.detail
        # The custom loader was never probed.
        assert counting_loader.calls == 0

    def test_adapter_controls_verdict_without_get_source(self, counting_loader):
        adapters = {counting_loader: lambda loader, name: name == "x.html"}
        loader = ChoiceLoader([counting_loader, DictLoader({"x.html": "d"})])

        exp = explain_load(loader, "x.html", adapters=adapters)
        assert exp.found
        assert exp.root.children[0].status == "hit"
        # The dict loader would also hit, but is shadowed by the adapter hit.
        assert exp.root.children[1].status == "shadowed"
        assert counting_loader.calls == 0

        exp = explain_load(loader, "other.html", adapters=adapters)
        assert not exp.found
        assert exp.root.children[0].status == "miss"
        assert counting_loader.calls == 0

    def test_adapter_registered_by_class(self, counting_loader):
        adapters = {CountingLoader: lambda loader, name: True}
        exp = explain_load(counting_loader, "anything", adapters=adapters)

        assert exp.found
        assert counting_loader.calls == 0

    def test_undecided_adapter_is_unsupported(self, counting_loader):
        adapters = {counting_loader: lambda loader, name: None}
        exp = explain_load(counting_loader, "x.html", adapters=adapters)

        assert exp.root.status == "unsupported"
        assert counting_loader.calls == 0

    def test_function_loader_not_called_without_adapter(self):
        calls = []

        def load_func(name):
            calls.append(name)
            return "source"

        exp = explain_load(FunctionLoader(load_func), "x.html")

        assert exp.root.status == "unsupported"
        assert calls == []


class TestNestedExplain:
    def test_nested_combination(self):
        loader = ChoiceLoader(
            [
                PrefixLoader(
                    {
                        "a": ChoiceLoader(
                            [
                                DictLoader({}),
                                DictLoader({"x.html": "nested"}),
                            ]
                        ),
                        "b": DictLoader({"x.html": "wrong branch"}),
                    }
                ),
                DictLoader({"a/x.html": "top-level"}),
            ]
        )
        exp = explain_load(loader, "a/x.html")

        assert exp.found
        # The nested ChoiceLoader inside prefix 'a' wins through its
        # second dict; the top-level dict is shadowed.
        assert exp.source == (
            "ChoiceLoader > choice[0] PrefixLoader > prefix 'a' ->"
            " ChoiceLoader > choice[1] DictLoader"
        )

        prefix_node, top_dict = exp.root.children
        assert prefix_node.status == "hit"
        assert top_dict.status == "shadowed"

        branch_a, branch_b = prefix_node.children
        assert branch_a.status == "hit"
        assert branch_a.name == "x.html"
        assert branch_b.status == "skipped"

        nested_first, nested_second = branch_a.children
        assert nested_first.status == "miss"
        assert nested_second.status == "hit"
        assert nested_second.name == "x.html"

    def test_nested_miss_keeps_full_attempt_order(self):
        loader = PrefixLoader(
            {
                "a": ChoiceLoader([DictLoader({}), DictLoader({})]),
                "b": DictLoader({}),
            }
        )
        exp = explain_load(loader, "a/nope.html")

        assert not exp.found
        statuses = [node.status for node in exp.attempts]
        assert statuses == ["miss", "miss", "skipped"]

    def test_loader_explain_method_matches_function(self):
        loader = DictLoader({"x.html": "d"})
        exp = loader.explain("x.html")

        assert isinstance(exp, LoadExplanation)
        assert exp.found
        assert isinstance(exp.root, ExplainNode)


class TestExplainIsReadOnly:
    def test_explain_does_not_render_or_load(self, counting_loader):
        env = Environment(
            loader=ChoiceLoader([DictLoader({"x.html": "{{ 1 + 1 }}"})])
        )
        exp = explain_load(env.loader, "x.html")

        assert exp.found
        # Nothing was compiled or rendered; the cache stays empty.
        assert len(env.cache) == 0

        with pytest.raises(TemplateNotFound):
            # Sanity check that explaining a miss matches real loading.
            env.get_template("missing.html")


class TestFileSystemExplain:
    def test_searchpath_order_and_shadowing(self, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "x.html").write_text("one", encoding="utf-8")
        (second / "x.html").write_text("two", encoding="utf-8")

        from jinja2 import FileSystemLoader

        loader = FileSystemLoader([str(first), str(second)])
        exp = explain_load(loader, "x.html")

        assert exp.found
        first_hit, second_hit = exp.root.children
        assert first_hit.status == "hit"
        assert second_hit.status == "shadowed"
        # Attempted paths are recorded in search order.
        assert str(first) in first_hit.detail
        assert str(second) in second_hit.detail

    def test_filesystem_miss_keeps_paths(self, tmp_path):
        from jinja2 import FileSystemLoader

        loader = FileSystemLoader([str(tmp_path / "a"), str(tmp_path / "b")])
        exp = explain_load(loader, "missing.html")

        assert not exp.found
        assert [n.status for n in exp.attempts] == ["miss", "miss"]
