# Copyright (c) Microsoft. All rights reserved.

"""The boundary between the strategies and the benchmark that measures them.

``compaction/`` is meant to be lifted out whole into a repository of its own, and the moment
one module in it reaches for a lab helper that stops being possible. Nothing announces that
at the time: the import works, the tests pass, and the cost arrives at extraction, when
somebody has to decide whether to drag the helper along or rewrite the caller.

So the boundary is checked here rather than remembered. The imports are read out of the
source with :mod:`ast` and resolved to module names, which is not the same as searching for
the package name in the text: a relative ``from .. import`` names nothing that a search would
find, and it is the shortest way through the wall.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The benchmark. Anything under this name is the lab unless it is under the subpackage.
LAB = "agent_framework_lab_cachebench"

#: The strategies, which is the part that leaves.
SUBPACKAGE = f"{LAB}.compaction"

#: The subpackage's directory, reached by path rather than by importing it.
#:
#: Importing it would run the lab's own ``__init__`` first, which is the very dependency this
#: file exists to say the strategies do not have. Reading the source is also the only way to
#: see an import that a ``TYPE_CHECKING`` guard keeps from ever executing.
SOURCE_ROOT = Path(__file__).resolve().parents[2] / LAB / "compaction"


def _resolve(module: str, *, is_package: bool, level: int, target: str | None) -> str:
    """Return the absolute name a relative import in ``module`` refers to.

    The same arithmetic ``importlib`` does: a level of 1 means the importing module's own
    package, and each further level strips one more component off it.

    Args:
        module: Dotted name of the module doing the importing.
        is_package: Whether that module is a package's ``__init__``, which is its own package
            rather than a member of one.
        level: Number of leading dots on the import.
        target: What followed them, or None for a bare ``from .. import name``.

    Returns:
        The absolute module name.
    """
    package = module if is_package else module.rpartition(".")[0]
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{target}" if target else base


def _imported_modules(source: str, module: str, *, is_package: bool) -> set[str]:
    """Return every module name ``source`` imports, relative ones resolved to absolute.

    Walks the whole tree rather than the top-level statements, so an import buried inside a
    function or a ``TYPE_CHECKING`` block counts too. A deferred import is still a dependency;
    it just fails later.

    Args:
        source: The module's source text.
        module: Its dotted name.

    Keyword Args:
        is_package: Whether it is a package's ``__init__``.

    Returns:
        The absolute names it depends on.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add(_resolve(module, is_package=is_package, level=node.level, target=node.module))
            elif node.module:
                names.add(node.module)
    return names


def _lab_imports(source: str, module: str, *, is_package: bool) -> set[str]:
    """Return the imports in ``source`` that cross from the strategies back into the lab.

    Args:
        source: The module's source text.
        module: Its dotted name.

    Keyword Args:
        is_package: Whether it is a package's ``__init__``.

    Returns:
        The offending module names, empty when the boundary holds.
    """
    return {
        name
        for name in _imported_modules(source, module, is_package=is_package)
        if (name == LAB or name.startswith(f"{LAB}.")) and name != SUBPACKAGE and not name.startswith(f"{SUBPACKAGE}.")
    }


def _modules_under(root: Path, package: str) -> list[tuple[Path, str, bool]]:
    """Return every Python module below ``root``, with its dotted name.

    Args:
        root: Directory to walk.
        package: Dotted name that ``root`` itself stands for.

    Returns:
        One tuple of path, module name and package flag per file.
    """
    found: list[tuple[Path, str, bool]] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        stem = parts[:-1] if is_package else parts
        found.append((path, ".".join((package, *stem)) if stem else package, is_package))
    return found


def test_the_strategies_never_import_the_lab() -> None:
    """The subpackage has to be liftable, and one import back into the benchmark ends that.

    The failure this prevents is not a broken import -- reaching into the lab works perfectly
    well from here. It is that the reach is invisible until somebody tries to move the
    strategies out, by which time the helper has grown callers on both sides.
    """
    modules = _modules_under(SOURCE_ROOT, SUBPACKAGE)
    assert modules, f"no modules found under {SOURCE_ROOT}, so this would pass whatever they imported"

    crossings = {
        module: found
        for path, module, is_package in modules
        if (found := _lab_imports(path.read_text(encoding="utf-8"), module, is_package=is_package))
    }

    assert not crossings, f"compaction/ must not import the lab: {crossings}"


def test_the_strategies_tests_never_import_the_lab_either() -> None:
    """They travel with the strategies, so they are inside the boundary too.

    A test that reaches for a lab fixture is the same problem one step removed: the strategies
    would move out and arrive untested, which is the state that makes the first change to them
    after the move unverifiable.
    """
    root = Path(__file__).parent
    crossings = {
        path.name: found
        for path, module, is_package in _modules_under(root, f"{SUBPACKAGE}.tests")
        if (found := _lab_imports(path.read_text(encoding="utf-8"), module, is_package=is_package))
    }

    assert not crossings, f"tests/compaction/ must not import the lab: {crossings}"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from agent_framework_lab_cachebench._live import make_lookup_tool", id="absolute-module"),
        pytest.param("import agent_framework_lab_cachebench._live", id="plain-import"),
        pytest.param("from agent_framework_lab_cachebench import LiveOutcome", id="package-root"),
        pytest.param("from .._live import make_lookup_tool", id="relative-sibling"),
        pytest.param("from .. import LiveOutcome", id="relative-bare"),
        pytest.param("def f():\n    from .._live import make_lookup_tool", id="deferred"),
        pytest.param(
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from .._live import LiveOutcome",
            id="type-checking",
        ),
    ],
)
def test_the_check_catches_every_way_across(source: str) -> None:
    """A walker that misses a form is worth exactly as much as no walker at all.

    Each of these is a real way one of these modules could reach the lab, and two of them --
    the bare ``from .. import`` and the deferred one inside a function -- name nothing a search
    of the text would find.
    """
    assert _lab_imports(source, f"{SUBPACKAGE}._toolsummary", is_package=False)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from agent_framework import Message", id="framework"),
        pytest.param("from agent_framework._compaction import group_messages", id="framework-private"),
        pytest.param("from ._anchored import AnchoredCompactionStrategy", id="sibling"),
        pytest.param("from agent_framework_lab_cachebench.compaction import RecallGate", id="own-package"),
    ],
)
def test_the_check_permits_what_the_subpackage_is_allowed(source: str) -> None:
    """Its own modules and the framework are the whole of what it may depend on."""
    assert not _lab_imports(source, f"{SUBPACKAGE}._toolsummary", is_package=False)
