"""Read-only explanation of how loaders resolve a template name.

This module answers the question "which sources would be tried to load
this name, and which one would win?" *without* rendering a template and
*without* calling arbitrary loader code.

The entry point is :func:`explain_load`, which walks a loader (including
nested :class:`~jinja2.ChoiceLoader` and :class:`~jinja2.PrefixLoader`
combinations) and returns a :class:`LoadExplanation` describing:

-   the candidate sources, in the order they would be tried,
-   which candidates are shadowed by an earlier source that wins,
-   which source finally hits, or that the name is not found,
-   how the template name changes as it travels through prefix branches.

Custom (unknown) loaders are never probed through ``get_source``: that
could have side effects. They are only inspected through an explicitly
provided *explain adapter*. Without an adapter they are reported as
``"unsupported"`` and left untouched.
"""

from __future__ import annotations

import os
import posixpath
import typing as t
from dataclasses import dataclass
from dataclasses import field

from .exceptions import TemplateNotFound
from .loaders import BaseLoader
from .loaders import ChoiceLoader
from .loaders import DictLoader
from .loaders import FileSystemLoader
from .loaders import PrefixLoader
from .loaders import split_template_path

#: Status of a candidate that provides the requested template and wins.
HIT = "hit"
#: Status of a candidate that was tried and does not provide the template.
MISS = "miss"
#: Status of a candidate that would provide the template, but an earlier
#: candidate in a :class:`ChoiceLoader` already won.
SHADOWED = "shadowed"
#: Status of a branch that was not entered, for example a
#: :class:`PrefixLoader` branch whose prefix did not match.
SKIPPED = "skipped"
#: Status of a loader that cannot be explained read-only because no
#: explain adapter was provided for it.
UNSUPPORTED = "unsupported"

#: An explain adapter receives ``(loader, name)`` and returns ``True`` if
#: the loader would provide that name, ``False`` if it would not, or
#: ``None`` if it cannot tell (reported as ``"unsupported"``). It must be
#: read-only and must not call ``loader.get_source``.
ExplainAdapter = t.Callable[[BaseLoader, str], t.Optional[bool]]

#: Adapters may be registered per loader instance or per loader class.
AdapterMap = t.Mapping[t.Union[BaseLoader, t.Type[BaseLoader]], ExplainAdapter]


@dataclass
class ExplainNode:
    """One candidate source in an explanation tree.

    :param label: Human-readable description of the loader at this node,
        such as ``"DictLoader"`` or ``"PrefixLoader branch 'a'"``.
    :param name: The template name as this loader sees it. This differs
        from the requested name when a :class:`PrefixLoader` stripped a
        prefix.
    :param status: One of ``"hit"``, ``"miss"``, ``"shadowed"``,
        ``"skipped"`` or ``"unsupported"``.
    :param detail: Optional extra information, such as the path checked
        on the file system or why a loader is unsupported.
    :param children: Nested candidates for composite loaders, in the
        order they are tried.
    """

    label: str
    name: str
    status: str
    detail: str = ""
    children: list["ExplainNode"] = field(default_factory=list)

    @property
    def is_hit(self) -> bool:
        return self.status == HIT

    def walk(self) -> t.Iterator["ExplainNode"]:
        """Yield this node and all descendants, in attempt order."""
        yield self
        for child in self.children:
            yield from child.walk()


class LoadExplanation:
    """The result of :func:`explain_load` for one template name.

    This is a read-only report; producing it did not render anything and
    did not call ``get_source`` on any loader.
    """

    def __init__(self, name: str, root: ExplainNode) -> None:
        #: The template name that was explained.
        self.name = name
        #: The root of the explanation tree.
        self.root = root

    @property
    def found(self) -> bool:
        """Whether any candidate would provide the template."""
        return self.root.status == HIT

    @property
    def source(self) -> str | None:
        """Path of labels to the candidate that finally hits, or ``None``."""
        if not self.found:
            return None
        path = self._hit_path(self.root)
        assert path is not None
        return " > ".join(node.label for node in path)

    def _hit_path(self, node: ExplainNode) -> list[ExplainNode] | None:
        if node.status != HIT:
            return None
        for child in node.children:
            child_path = self._hit_path(child)
            if child_path is not None:
                return [node, *child_path]
        return [node]

    @property
    def attempts(self) -> list[ExplainNode]:
        """Leaf candidates in the order they are tried.

        Attempted paths are kept even when the name is not found, so a
        miss still shows every source that was consulted.
        """
        return [node for node in self.root.walk() if not node.children]

    def format(self) -> str:
        """Render the explanation as an indented tree.

        Shows the template name at each node (including name changes
        caused by prefix stripping) and shadowing information.
        """
        lines = [f"explain {self.name!r} -> {'found' if self.found else 'not found'}"]
        self._format_node(self.root, lines, indent=1, parent_name=None)
        if self.found:
            lines.append(f"final source: {self.source}")
        return "\n".join(lines)

    def _format_node(
        self,
        node: ExplainNode,
        lines: list[str],
        indent: int,
        parent_name: str | None,
    ) -> None:
        pad = "  " * indent
        parts = [f"{pad}{node.label}: {node.name!r} -> {node.status}"]

        if parent_name is not None and node.name != parent_name:
            parts.append(f"name changed: {parent_name!r} -> {node.name!r}")

        if node.status == SHADOWED and not node.detail:
            parts.append("shadowed by an earlier source")

        if node.detail:
            parts.append(f"({node.detail})")

        lines.append("  ".join(parts))

        for child in node.children:
            self._format_node(child, lines, indent + 1, node.name)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"<LoadExplanation {self.name!r} found={self.found}>"


def _find_adapter(
    loader: BaseLoader, adapters: AdapterMap | None
) -> ExplainAdapter | None:
    if not adapters:
        return None

    # Instance registrations win over class registrations.
    for key, adapter in adapters.items():
        if not isinstance(key, type) and key is loader:
            return adapter

    for key, adapter in adapters.items():
        if isinstance(key, type) and isinstance(loader, key):
            return adapter

    return None


def _explain(
    loader: BaseLoader, name: str, adapters: AdapterMap | None
) -> ExplainNode:
    label = type(loader).__name__

    if isinstance(loader, ChoiceLoader):
        return _explain_choice(loader, name, adapters)

    if isinstance(loader, PrefixLoader):
        return _explain_prefix(loader, name, adapters)

    if isinstance(loader, DictLoader):
        if name in loader.mapping:
            return ExplainNode(label, name, HIT)
        return ExplainNode(label, name, MISS, "name not in mapping")

    if isinstance(loader, FileSystemLoader):
        return _explain_filesystem(loader, name)

    adapter = _find_adapter(loader, adapters)

    if adapter is None:
        return ExplainNode(
            label,
            name,
            UNSUPPORTED,
            "no explain adapter registered; get_source was not called",
        )

    verdict = adapter(loader, name)

    if verdict is True:
        return ExplainNode(label, name, HIT, "reported by explain adapter")

    if verdict is False:
        return ExplainNode(label, name, MISS, "reported by explain adapter")

    return ExplainNode(
        label, name, UNSUPPORTED, "explain adapter could not decide"
    )


def _explain_choice(
    loader: ChoiceLoader, name: str, adapters: AdapterMap | None
) -> ExplainNode:
    node = ExplainNode("ChoiceLoader", name, MISS)
    won = False

    for index, sub in enumerate(loader.loaders):
        child = _explain(sub, name, adapters)
        child.label = f"choice[{index}] {child.label}"

        if child.status == HIT:
            if won:
                # A later source that would also hit is hidden by the
                # earlier winner; keep that shadowing visible.
                child.status = SHADOWED
                child.detail = (
                    "would hit, but an earlier choice already provides"
                    " this name"
                )
            else:
                won = True

        node.children.append(child)

    if won:
        node.status = HIT

    return node


def _explain_prefix(
    loader: PrefixLoader, name: str, adapters: AdapterMap | None
) -> ExplainNode:
    node = ExplainNode("PrefixLoader", name, MISS)

    try:
        prefix, local_name = name.split(loader.delimiter, 1)
    except ValueError:
        node.detail = (
            f"name has no delimiter {loader.delimiter!r}; no branch entered"
        )
        return node

    matched = prefix in loader.mapping

    for branch_prefix, sub in loader.mapping.items():
        branch_label = f"prefix {branch_prefix!r} -> {type(sub).__name__}"

        if branch_prefix == prefix:
            # Only the matching branch is entered, and it sees the name
            # with the prefix stripped.
            child = _explain(sub, local_name, adapters)
            child.label = branch_label
            node.children.append(child)

            if child.status == HIT:
                node.status = HIT
        else:
            # The stripped name must never leak into other branches;
            # record them as not entered.
            node.children.append(
                ExplainNode(
                    branch_label,
                    name,
                    SKIPPED,
                    f"prefix {branch_prefix!r} did not match; branch not"
                    " entered",
                )
            )

    if not matched:
        known = ", ".join(repr(p) for p in loader.mapping)
        node.detail = f"prefix {prefix!r} not in mapping (known: {known})"

    return node


def _explain_filesystem(loader: FileSystemLoader, name: str) -> ExplainNode:
    node = ExplainNode("FileSystemLoader", name, MISS)

    try:
        pieces = split_template_path(name)
    except TemplateNotFound:
        node.detail = "name is not a valid template path"
        return node

    for searchpath in loader.searchpath:
        filename = posixpath.join(searchpath, *pieces)
        status = HIT if os.path.isfile(filename) else MISS
        node.children.append(ExplainNode("searchpath", name, status, filename))

    # Real loading stops at the first matching search path, so later
    # matching paths are shadowed by it.
    seen_hit = False

    for child in node.children:
        if child.status == HIT:
            if seen_hit:
                child.status = SHADOWED
                child.detail = f"{child.detail} (shadowed by earlier searchpath)"
            else:
                seen_hit = True

    if seen_hit:
        node.status = HIT

    return node


def explain_load(
    loader: BaseLoader,
    name: str,
    adapters: AdapterMap | None = None,
) -> LoadExplanation:
    """Explain how ``loader`` would resolve the template ``name``,
    without rendering anything.

    Composite loaders (:class:`ChoiceLoader`, :class:`PrefixLoader`) are
    processed recursively, preserving the order in which candidates are
    tried, marking candidates shadowed by an earlier winner, and
    tracking how the name changes when a prefix is stripped. The
    stripped name is only applied to the branch it belongs to.

    Loaders that are not understood read-only are **not** probed through
    ``get_source``. They are reported as ``"unsupported"`` unless an
    explain adapter is explicitly provided for them in ``adapters``.
    Adapters may be registered per loader instance or per loader class,
    and must be read-only callables ``(loader, name) -> bool | None``.

    :param loader: The loader to inspect.
    :param name: The template name to explain.
    :param adapters: Optional mapping of loader instances or classes to
        explain adapter callables.

    .. versionadded:: 3.2
    """
    root = _explain(loader, name, adapters)
    return LoadExplanation(name, root)
