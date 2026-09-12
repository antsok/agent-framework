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
  tail, and it recompacts its own earlier summary once the band is worth a pass again --
  deliberately the opposite of ``_anchored``'s refusal to re-trim, for the reason its module
  docstring gives, and bounded by a minimum band share because the threshold alone let it fire
  once per turn. It touches nothing the other three touch, so the rows stay comparable.
- :class:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy` runs those last two
  over one conversation, the record phase first. It is the only entry here that composes
  rather than compacts: it owns no selection rule and removes nothing itself. What it adds is
  an order; one line for both halves, taken from the record phase's trigger; one reading of the
  prompt, taken before either phase acts, which is what makes that line more than a shared
  number -- a half that would have fired on the size the pass began with fires whatever the
  other half has already removed; one re-read of the conversation between the phases, for the
  in-place rewrites the record phase's fallback makes; and the attribution of a silent second
  phase, which on a row whose halves have been set apart again says which passes the first
  phase's removals kept under the second's trigger, as against the passes the second phase
  declined for its own reasons.
- :func:`set_preserved` and :func:`is_preserved` carry one annotation between the two: a
  message no strategy may shorten, drop or shed. It exists because the record is a tool
  result, the anchored strategy trims tool results, and for a while it trimmed the record --
  destroying the only surviving copy of everything the other strategy had just deleted.

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
from ._composed import ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy
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
    DEFAULT_USER_SUMMARY_PROMPT,
    DEFAULT_USER_TRIGGER_FRACTION,
    USER_SUMMARY_MARKER,
    UserTurnAnchoredSummarizationCompactionStrategy,
)

__all__ = [
    "DEFAULT_BAND_SHARE",
    "DEFAULT_COVERAGE_SHARE",
    "DEFAULT_FALLBACK_FRACTION",
    "DEFAULT_KEEP_HEAD_USER_TURNS",
    "DEFAULT_KEEP_TAIL_USER_TURNS",
    "DEFAULT_KEEP_TOKENS",
    "DEFAULT_MIN_BAND_SHARE",
    "DEFAULT_MIN_GAIN_FRACTION",
    "DEFAULT_RECORD_MAX_TOKENS",
    "DEFAULT_RECORD_TARGET_TOKENS",
    "DEFAULT_TRIGGER_FRACTION",
    "DEFAULT_USER_SUMMARY_PROMPT",
    "DEFAULT_USER_TRIGGER_FRACTION",
    "MARKER_ID_PREFIX",
    "PRESERVED_KEY",
    "PRESERVE_REASON_KEY",
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "REMOVAL_MARKER",
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
