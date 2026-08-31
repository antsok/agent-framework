# Copyright (c) Microsoft. All rights reserved.

"""Compaction strategies designed against what the cachebench benchmark measured.

These strategies are not part of the benchmark. They are the thing it measures, and they are
meant to leave for a repository of their own, so this subpackage is written to be lifted out
whole: nothing in it imports from the surrounding lab, and its tests sit beside it in
``tests/compaction/``. A test walks every module's imports and fails if that stops being
true, because the boundary rots the first time someone reaches for a lab helper and nobody
finds out until extraction day.

What is here:

- :class:`AnchoredCompactionStrategy` keeps a fixed head and tail verbatim and collapses the
  band between them by a rule that reads a group's *position* and nothing else, so the same
  prefix compacts to the same bytes on every later turn and stays cached.
- :class:`MinimumGainAnchoredCompactionStrategy` is the same strategy with a floor: it
  projects the reduction before mutating anything and declines a collapse too small to repay
  the prompt cache it would invalidate.
- :class:`ToolResultAnchoredSummarizationCompactionStrategy` has the agent write the facts
  into a tool result and then drops the tool groups that result replaced. It needs the other
  three names to work: :func:`make_recall_tool` is the tool the model calls,
  :class:`RecallGate` keeps that tool inert until it is asked for, and
  :class:`ToolResultRecallMiddleware` is what asks.

**This depends on ``agent_framework._compaction``, which is private API.** Grouping, token
annotation and the exclusion flags all come from there; nothing public exposes them. That
module is a moving target -- upstream PR #7912 rewrote the compaction path on 2026-08-31,
changing ``ContextWindowCompactionStrategy``'s tool-eviction phase and making compaction
performed inside the client persist back into the caller's message list. Whoever extracts
this owes it a re-read against the framework version they pin, and a pin narrow enough that
a private module changing shape is a dependency-resolution failure rather than a silent
behaviour change.

**Rename before publishing.** ``agent_framework_lab_cachebench.compaction`` sits inside a
namespace Microsoft owns, and a package published from there would claim an association it
does not have. The distribution and its import path both have to move out of
``agent_framework`` first; only the dependency on it stays.
"""

from ._anchored import (
    DEFAULT_KEEP_TOKENS,
    DEFAULT_MIN_GAIN_FRACTION,
    MARKER_ID_PREFIX,
    REMOVAL_MARKER,
    AnchoredCompactionStrategy,
    MinimumGainAnchoredCompactionStrategy,
)
from ._toolsummary import (
    DEFAULT_RECORD_MAX_TOKENS,
    DEFAULT_RECORD_TARGET_TOKENS,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    RecallGate,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    find_record_index,
    make_recall_tool,
)

__all__ = [
    "DEFAULT_KEEP_TOKENS",
    "DEFAULT_MIN_GAIN_FRACTION",
    "DEFAULT_RECORD_MAX_TOKENS",
    "DEFAULT_RECORD_TARGET_TOKENS",
    "MARKER_ID_PREFIX",
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "REMOVAL_MARKER",
    "AnchoredCompactionStrategy",
    "MinimumGainAnchoredCompactionStrategy",
    "RecallGate",
    "ToolResultAnchoredSummarizationCompactionStrategy",
    "ToolResultRecallMiddleware",
    "find_record_index",
    "make_recall_tool",
]
