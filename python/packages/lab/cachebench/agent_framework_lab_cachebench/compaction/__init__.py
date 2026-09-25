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
- :class:`UserTurnAnchoredSummarizationCompactionStrategy` is the mirror of that one on the
  other half of the conversation: it summarises the *user's* turns between a fixed head and
  tail. What it does with its own earlier summary is a mode: recompact it once the band is
  worth a pass again -- the default, deliberately the opposite of ``_anchored``'s refusal to
  re-trim, for the reason its module docstring gives -- or leave it standing as a boundary the
  next pass compacts only behind, or do that and fold the standing summaries into one once they
  are worth the break. All three are bounded by a minimum band share because the threshold
  alone let it fire once per turn. It touches nothing the other three touch, so the rows stay
  comparable.
- :class:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy` runs those last two
  over one conversation, the record phase first, with its recall middleware asking for a new
  record for every new batch of tool work. It owns no selection rule of its own. What it adds
  is an order; one line for both halves, taken from the record phase's trigger, with the user
  half judged against the prompt the record phase left -- so it acts only when tool compaction
  was not enough, since its edits break the cached prefix; and a last-resort chain that runs
  only while the prompt is over the input budget: merge the records, merge the user summaries,
  rewrite the record harder, and then the record phase's fallback, moved there from straight
  behind the record. Every replacement the chain makes is kept only if it is smaller than what
  it replaces, and nothing is checked against content.
- :func:`set_preserved` and :func:`is_preserved` carry one annotation between the two: a
  message no strategy may shorten, drop or shed. It exists because the record is a tool
  result, the anchored strategy trims tool results, and for a while it trimmed the record --
  destroying the only surviving copy of everything the other strategy had just deleted. The
  record strategy puts the same mark, under :data:`PRESERVE_REASON_UNCOVERED`, on a tool group
  its record failed to cover: first while it asks for another record to cover it, then for
  good if asking stops helping, so the fallback can shorten neither and the row overflows
  loudly rather than losing the group's values quietly. Under :data:`PRESERVE_REASON_UNRECORDED`
  it marks every other tool group no record covers -- those after the newest record above all
  -- before its fallback runs, so that fallback may remove narration and nothing else.

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

What each strategy does, why it is shaped that way and where it fails is in
``STRATEGIES.md`` beside this file. It travels with the package.
"""

from ._anchored import (
    DEFAULT_BAND_SHARE,
    DEFAULT_KEEP_TOKENS,
    DEFAULT_MIN_GAIN_FRACTION,
    MARKER_ID_PREFIX,
    REMOVAL_MARKER,
    AnchoredCompactionStrategy,
    MinimumGainAnchoredCompactionStrategy,
)
from ._composed import (
    DEFAULT_CHAIN_GAIN_FRACTION,
    DEFAULT_HARDER_ATTEMPTS,
    ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy,
)
from ._preserve import (
    PRESERVE_REASON_KEY,
    PRESERVED_KEY,
    any_preserved,
    is_preserved,
    set_preserved,
)
from ._toolsummary import (
    DEFAULT_COVERAGE_SHARE,
    DEFAULT_FALLBACK_FRACTION,
    DEFAULT_RECORD_MAX_TOKENS,
    DEFAULT_RECORD_TARGET_TOKENS,
    DEFAULT_TRIGGER_FRACTION,
    PRESERVE_REASON_UNCOVERED,
    PRESERVE_REASON_UNRECORDED,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    RecallGate,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    find_record_index,
    make_recall_tool,
)
from ._usersummary import (
    DEFAULT_KEEP_HEAD_USER_TURNS,
    DEFAULT_KEEP_TAIL_USER_TURNS,
    DEFAULT_MIN_BAND_SHARE,
    DEFAULT_SUMMARY_MODE,
    DEFAULT_USER_FOLD_PROMPT,
    DEFAULT_USER_SUMMARY_PROMPT,
    DEFAULT_USER_TRIGGER_FRACTION,
    FOLD_EXCLUDE_REASON,
    FOLD_ID_PREFIX,
    SUMMARY_MODE_BOUNDARY,
    SUMMARY_MODE_FOLD,
    SUMMARY_MODE_RECOMPACT,
    SUMMARY_MODES,
    USER_SUMMARY_MARKER,
    UserTurnAnchoredSummarizationCompactionStrategy,
)

__all__ = [
    "DEFAULT_BAND_SHARE",
    "DEFAULT_CHAIN_GAIN_FRACTION",
    "DEFAULT_COVERAGE_SHARE",
    "DEFAULT_FALLBACK_FRACTION",
    "DEFAULT_HARDER_ATTEMPTS",
    "DEFAULT_KEEP_HEAD_USER_TURNS",
    "DEFAULT_KEEP_TAIL_USER_TURNS",
    "DEFAULT_KEEP_TOKENS",
    "DEFAULT_MIN_BAND_SHARE",
    "DEFAULT_MIN_GAIN_FRACTION",
    "DEFAULT_RECORD_MAX_TOKENS",
    "DEFAULT_RECORD_TARGET_TOKENS",
    "DEFAULT_SUMMARY_MODE",
    "DEFAULT_TRIGGER_FRACTION",
    "DEFAULT_USER_FOLD_PROMPT",
    "DEFAULT_USER_SUMMARY_PROMPT",
    "DEFAULT_USER_TRIGGER_FRACTION",
    "FOLD_EXCLUDE_REASON",
    "FOLD_ID_PREFIX",
    "MARKER_ID_PREFIX",
    "PRESERVED_KEY",
    "PRESERVE_REASON_KEY",
    "PRESERVE_REASON_UNCOVERED",
    "PRESERVE_REASON_UNRECORDED",
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "REMOVAL_MARKER",
    "SUMMARY_MODES",
    "SUMMARY_MODE_BOUNDARY",
    "SUMMARY_MODE_FOLD",
    "SUMMARY_MODE_RECOMPACT",
    "USER_SUMMARY_MARKER",
    "AnchoredCompactionStrategy",
    "MinimumGainAnchoredCompactionStrategy",
    "RecallGate",
    "ToolResultAnchoredSummarizationCompactionStrategy",
    "ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy",
    "ToolResultRecallMiddleware",
    "UserTurnAnchoredSummarizationCompactionStrategy",
    "any_preserved",
    "find_record_index",
    "is_preserved",
    "make_recall_tool",
    "set_preserved",
]
