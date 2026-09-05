# Copyright (c) Microsoft. All rights reserved.

"""Have the agent record the facts, then drop everything behind the record.

Every truncating strategy measured here fails the same way, and the failure is arithmetic
rather than tuning. Head-and-tail retention keeps the first and last f/2 of a result, so ``n``
values spread evenly through it -- sitting 1/n apart -- survive only when f exceeds 2/n. With
eight values per result that needs more than 25% of it retained, which is not compaction.
Measured: at 30% retention 3 of 8 codes survived, at 0.9% just 1.

Truncation preserves *positions*. What is needed is something that preserves *information*,
after which the bulk it came from is genuinely redundant and can be dropped outright rather
than sampled.

**Four parts, and all four are the design rather than harness around it.** The strategy
cannot work without a tool for the model to call (:func:`make_recall_tool`) or without
something keeping that tool inert when nobody asked for a record (:class:`RecallGate`), so
both live here beside the middleware that arms them.

**Two phases, split across a middleware and this strategy.**

1. :class:`ToolResultRecallMiddleware` forces ``tool_choice`` to the recall tool on the call
   after the one where it saw the conversation grow past the trigger -- and again later, once
   the agent has done tool work no existing record accounts for. It sends no message at all:
   the tool's own description already says what to pass, and the schema travels on every
   request anyway. Nothing is added to the prompt, and nothing extra reaches the caller's
   stored history. :meth:`ToolResultRecallMiddleware._record_due` is where "no existing record
   accounts for it" is defined, and it is defined nowhere else.
2. The model makes the call, the agent executes it, and the result is persisted through the
   ordinary path. On a later pass this strategy finds that real tool result in the loaded
   history and drops the tool groups in front of it whose contents the record demonstrably
   carries -- see the coverage note below, and ``_drop_before`` for how that is decided.

Sending no message matters for more than tokens. A message appended here carries no history
provider's source tag, so the per-service-call persistence would treat it as new input and
store it -- and an instruction of ours would show up in the conversation the application
replays to its user. Forcing the option leaves no such trace. The tool call and its result do
appear, which is correct: they are a real record of what the agent did.

**Why not synthesise the tool call directly.** A ``call_id`` invented by the client is only
safe when the client owns the conversation. Responses-API routes with ``store=True`` track
tool calls server-side, so a fabricated call is unknown to the service or mismatched against
it, and Gemini's thought signatures behave similarly. Letting the provider issue the call
avoids the problem entirely, at the price of one extra agent turn.

**Why a tool result rather than an assistant message.** A tool result is data. Assistant prose
is the first thing a size-pressed strategy sheds -- this package's own anchored strategy sheds
it as a last resort -- so a record written as narration would be eligible for exactly the step
that destroys it.

**Why forcing beats asking.** Asking required the model to choose, which meant the benchmark
could not pin ``tool_choice`` -- and unpinned, the uncompacted control's cost varied by 102%
between identical runs while the strategy's row gathered eight fewer facts than the control.
Forcing the call keeps every other turn pinned, so the comparison stays measurable, and makes
phase 1 deterministic rather than a compliance rate to be estimated.

**How the record is bounded, and why it takes two numbers rather than one.** The instructions
ask for everything, so the record wants to grow. A ``max_tokens`` cap does not answer that: a
model does not plan to fit a cap, it writes until it is cut, and on a *tool call* the cut
lands inside the arguments JSON, so a cap set where the record should end produces no record
instead of a shorter one. The two bounds therefore do different jobs.
:data:`DEFAULT_RECORD_TARGET_TOKENS` is stated in the tool's own description, which is the
only channel that reaches the model before it writes, since the middleware sends no message.
:data:`DEFAULT_RECORD_MAX_TOKENS` is set on the forced call alone and is roughly twice the
target, so it bounds the bill without ever being the thing that stops the writing. When it is
the thing that stops it, ``ToolResultRecallMiddleware.records_truncated`` says so.

**What the record covers is checked, not assumed.** Phase 2 used to drop every tool group in
front of the record because the record was supposed to have replaced them. Measured, that is
model-dependent: gpt-5.4-mini writes records naming every tool group, gpt-5.6-luna writes one
covering two of six, and the other four were deleted with nothing preserving them and nothing
reporting it. Raising the cap, raising the stated target and rewriting the prompt were each
measured and each changed nothing, so the strategy now drops only the groups the record
demonstrably carries, and ``groups_kept_uncovered`` counts the rest. That turns the failure
direction around: a partial record now costs tokens it should not have cost, instead of
losing facts nobody can trace. ``ToolResultRecallMiddleware``'s ``max_groups_before_record``
is the other half, bounding how much any one record is asked to cover so partial coverage
stops being the normal case.

**Several records, and nothing merges them.** Once the size trigger may fire more than once, a
long conversation accumulates records, and every one of them is preserved: unshrinkable,
undroppable, and counted against the ceiling in full. That is a floor under the prompt that
grows a record at a time, and
:attr:`ToolResultAnchoredSummarizationCompactionStrategy.records_in_conversation` reports it,
because a row whose compaction has stopped paying for a good reason and one whose unshrinkable
part has quietly grown are otherwise the same row. Consolidating them is deliberately not done:
an older record is the sole account of the groups behind *it*, so a merge rewrites the evidence
rather than the bulk.

**Coverage is measured in values, not in tool names, because models do not write tool names.**
The first version of the check asked whether the record contained the group's function name,
on the reading that :data:`RECALL_VALUES_DESCRIPTION` asks for the results "grouped by the
tool that produced it". Models do not comply with that clause the way the check assumed.
Luna's record says *"extra0 deployment lookup returned codes: ..."* and never writes
``lookup_extra0`` anywhere; gpt-5.4-mini, whose records carry every value from every group,
scored ``UNCOVERED:4`` on the same rule, and its compaction fell from a 20% reduction to 5-6%
for no benefit whatsoever. A check that penalises the model that complied is not a check. The
rule is now the first thing that description actually asks for -- "Quote verbatim any value
that cannot be reconstructed or guessed" -- so a group is covered when the record quotes
enough of the distinctive values its results contain. :data:`DEFAULT_COVERAGE_SHARE` is how
much of them, and :func:`_distinctive_tokens` states the rule that finds them and what it
cannot see.

**The record is protected from the fallback, and had to be.** When a record does not free
enough, whatever remains goes to ``fallback``, which defaults to
:class:`~._anchored.AnchoredCompactionStrategy` -- and that strategy shortens and sheds tool
results, of which the record is one. Nothing in it recognised a record, so the record was
trimmed like any other bulk: a live seed's record of four lookups reached the answering prompt
carrying two, 16,617 tokens gone with only three messages removed. Every deletion phase 2
performs is licensed by the record, so trimming the record afterwards destroys the sole
surviving copy of what was already deleted. Both halves therefore agree through
:mod:`._preserve`: this strategy marks every record it observes, and the anchored strategy
skips preserved messages in each of its three removal paths.
"""

from __future__ import annotations

import string
from collections.abc import Awaitable, Callable, Sequence
from math import ceil
from typing import TYPE_CHECKING, Any, Final

from agent_framework import ChatContext, ChatMiddleware, ChatResponse, Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    annotate_message_groups,
    annotate_token_counts,
    group_messages,
    included_token_count,
    set_excluded,
)

from ._anchored import AnchoredCompactionStrategy
from ._preserve import any_preserved, set_preserved

if TYPE_CHECKING:
    from agent_framework import CompactionStrategy, TokenizerProtocol

__all__ = [
    "DEFAULT_COVERAGE_SHARE",
    "DEFAULT_FALLBACK_FRACTION",
    "DEFAULT_RECORD_MAX_TOKENS",
    "DEFAULT_RECORD_TARGET_TOKENS",
    "DEFAULT_TRIGGER_FRACTION",
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "RecallGate",
    "ToolResultAnchoredSummarizationCompactionStrategy",
    "ToolResultRecallMiddleware",
    "find_record_index",
    "make_recall_tool",
]

#: Name of the tool the agent must call. The strategy looks for this name in the history, so
#: the tool the caller registers has to match it.
RECALL_TOOL_NAME: Final[str] = "recall_earlier_tool_results"

#: Prefix the recall tool puts on a result that counts as a record.
#:
#: The tool cannot be hidden from the model. Tools passed through per-call options reach the
#: model but not the executor -- ``FunctionInvocationLayer`` wraps ``ChatMiddlewareLayer``, so
#: it has already built its tool map by the time a middleware could add one, and the model's
#: call goes unanswered. A registered tool is therefore advertised on every request, and this
#: one was called unprompted on the unpinned follow-up call in every run.
#:
#: So the tool stays visible and becomes *inert* instead: it records only while the middleware
#: has armed it, and a result without this marker is not a record. The model may still call
#: it; calling it uninvited simply achieves nothing.
RECORD_MARKER: Final[str] = "[recorded by compaction]"

#: Hard ceiling put on the forced call's response, so a runaway record cannot cost more than
#: intended.
#:
#: A cap is not a plan. The model does not shorten to fit one; it writes until it is cut, and
#: on a *tool call* the cut lands inside the arguments JSON, so what a too-low cap produces is
#: not a shorter record but no record at all. That makes this the wrong instrument for sizing
#: the record and the right one for bounding the bill, and the two numbers here are set
#: accordingly: this is roughly twice :data:`DEFAULT_RECORD_TARGET_TOKENS`, so a model that
#: overshoots its stated target still finishes inside the cap.
DEFAULT_RECORD_MAX_TOKENS: Final[int] = 4_000

#: Reason recorded on the record's messages when this strategy protects them, so a caller
#: reading a conversation back can tell which strategy claimed them.
PRESERVE_REASON: Final[str] = "tool_summary_record"

#: Share of a group's distinctive values the record must quote before the group may be dropped.
#:
#: The number has to sit between two failures. At 1.0 the check is as brittle as the tool-name
#: rule it replaces: one value the model rendered differently -- a number regrouped, a
#: timestamp normalised, an identifier wrapped in quotes the strip below does not remove --
#: keeps a whole group whose content is demonstrably present, and the measured cost of that
#: mistake was compaction falling from 20% to 5-6% on a model whose records were complete. Far
#: below 0.5 the check stops being one: a record that quoted two values from a group of eight
#: would license deleting the other six, which is the silent loss the whole coverage check
#: exists to prevent.
#:
#: 0.8 is the loosest setting that still refuses a record which dropped a quarter of a group,
#: and at the eight values per result these runs were measured on it tolerates exactly one
#: value in eight being unrecognisable. It is a threshold rather than a derivation, and it is a
#: constructor keyword because the right value depends on how many values a workload's results
#: carry: at two values per group the share can only be 0, 0.5 or 1, so a workload like that
#: should set it deliberately rather than inherit this.
DEFAULT_COVERAGE_SHARE: Final[float] = 0.8

#: Shortest token :func:`_distinctive_tokens` will treat as a value worth quoting.
#:
#: Three characters and under is where ordinary prose with a digit in it lives -- "3rd", "v2",
#: "10%", "1)" -- and none of that is a value a later question could depend on. Four is also
#: the point below which substring matching starts producing accidental hits: "a12" occurs
#: inside any longer identifier containing it, so a short token would count itself covered by
#: an unrelated mention.
_MIN_DISTINCTIVE_LENGTH: Final[int] = 4

#: Record length stated in the tool's own description, which is the only channel that makes
#: the model aim for a size.
#:
#: The middleware deliberately sends no message -- an appended instruction would be persisted
#: into the caller's own conversation -- so the description and the ``values`` parameter are
#: the entire prompt, and a target has to be baked into them at construction.
DEFAULT_RECORD_TARGET_TOKENS: Final[int] = 2_000

#: Fraction of the ceiling at which a record is first asked for.
#:
#: One constant read by both halves rather than a default written twice, because the middleware
#: asks and the strategy waits: a caller who moved one without the other would either have the
#: strategy dropping groups before anything had been recorded, or have the middleware recording
#: what nothing was yet willing to drop.
#:
#: **0.8, and it was 0.6.** The trigger is a bet that enough conversation remains to repay a
#: compaction, and 0.6 takes that bet far too early. With the 2,048-token output reservation
#: these runs use, 0.6 of the input budget is 58% of a 60,000-token window: a record is forced,
#: an agent turn is spent, and the cached prefix is broken part-way through a conversation that
#: may well end before it ever needed compacting at all. The break-even derived in
#: :data:`~._anchored.DEFAULT_MIN_GAIN_FRACTION` is the same argument from the other side -- an
#: edit repays itself only over the turns that follow it, so it wants as many of them as
#: possible, but the turns *before* the ceiling is approached are exactly the ones where the
#: horizon is unknown and the compaction may turn out to have bought nothing. Waiting costs
#: nothing until the ceiling is actually in reach; asking early costs a call, a re-read of the
#: whole prefix, and the tool results the record then licenses deleting.
DEFAULT_TRIGGER_FRACTION: Final[float] = 0.8

#: Fraction of the ceiling at which the strategy stops waiting for a record and compacts
#: without one.
#:
#: **0.95, and it was 0.9.** The move is forced by the trigger's. A record arrives one call late
#: by construction: the middleware can only read the history on the way *out* of a call and can
#: only pin the *next* one, so the conversation grows by a whole turn between the ask and the
#: answer -- see :meth:`ToolResultRecallMiddleware.process`. With the trigger at 0.8, a give-up
#: line at 0.9 leaves a single turn's growth of room, and one turn carrying a large tool result
#: crosses it: the strategy then compacts without a record while the record it asked for is
#: still in flight, which is the one outcome this whole design exists to avoid. The gap between
#: the two has to be wide enough for the answer to land in.
#:
#: It cannot simply be raised to 1.0. Past this line the fallback still has to bring the
#: conversation under the ceiling, and a fallback given no headroom has nothing to work in.
DEFAULT_FALLBACK_FRACTION: Final[float] = 0.95

#: What the recall tool is for, as the model reads it.
#:
#: Deliberately not "identifiers and values": that phrasing was fitted to one benchmark's hex
#: codes and would drop prose, findings and conclusions from any real tool output, which is
#: most of what a real tool returns.
RECALL_DESCRIPTION: Final[str] = (
    "Record what must survive from earlier tool results, so it remains available after those "
    "results are removed from the conversation to save space."
)

#: What to put in the ``values`` argument, as the model reads it.
#:
#: The four instructions after the opening sentence partition the content, so that nothing in
#: a tool result falls outside all of them: values that cannot be reconstructed are quoted,
#: findings and conclusions are kept as stated, a summary the tool already wrote is carried
#: over rather than rewritten, and whatever remains is summarised. Removing one of the four
#: opens a gap that the model is then free to drop silently, which is the failure this whole
#: strategy exists to avoid.
RECALL_VALUES_DESCRIPTION: Final[str] = (
    "Everything from earlier tool results, grouped by the tool that produced it, so that "
    "nothing is lost without being noticed. Quote verbatim any value that cannot be "
    "reconstructed or guessed: identifiers, codes, names, numbers, paths, URLs, versions, "
    "states, timestamps. Keep findings and conclusions as they were stated, shortening only "
    "those long enough to need it. Carry over any summary a tool already produced as it "
    "stands, rather than rewriting it. Summarise the remaining content briefly, so its "
    "substance is still represented. Record only what the results actually contained, and "
    "where you must choose, keep exactness over brevity."
)


def find_record_index(messages: Sequence[Message]) -> int | None:
    """Return the index of the newest recall tool result, if one exists.

    Shared by the strategy and the middleware so the two halves cannot disagree about whether
    a record exists -- otherwise one would force a call that was already made, or drop results
    a record never covered.

    The call and its result are matched by ``call_id`` rather than by adjacency, because a
    provider is free to order or batch them differently. A result whose call is absent does not
    count: that is the shape a client-synthesised pair produces, and exactly what breaks on
    routes that track tool calls server-side.

    Args:
        messages: The conversation to search.

    Returns:
        The index of the result message, or None when no complete record exists.
    """
    recall_ids = {
        content.call_id
        for message in messages
        for content in message.contents
        if content.type == "function_call" and content.name == RECALL_TOOL_NAME and content.call_id
    }
    if not recall_ids:
        return None
    newest: int | None = None
    for index, message in enumerate(messages):
        for content in message.contents:
            if content.type != "function_result" or content.call_id not in recall_ids:
                continue
            result = content.result if isinstance(content.result, str) else str(content.result)
            # The marker is what separates a record from a call the model made on its own
            # initiative. Without it an uninvited call would look like a record and the
            # strategy would drop results nothing had preserved.
            if RECORD_MARKER in result:
                newest = index
    return newest


def _record_text(message: Message) -> str:
    """Return the record text carried by a recall tool result message.

    Only results bearing :data:`RECORD_MARKER` are read. A provider may batch several tool
    results into one message, and text from an unrelated result sitting beside the record would
    then count towards coverage without anyone having written it as a record -- which is
    precisely the mistake the coverage check exists to stop.

    Args:
        message: The message at the anchor index.

    Returns:
        The record, or an empty string when the message carries none.
    """
    parts: list[str] = []
    for content in message.contents:
        if content.type != "function_result":
            continue
        result = content.result if isinstance(content.result, str) else str(content.result)
        if RECORD_MARKER in result:
            parts.append(result)
    return "\n".join(parts)


def _called_function_names(messages: Sequence[Message], group: dict[str, Any]) -> set[str]:
    """Return the distinct function names called inside one group's span.

    Args:
        messages: The conversation the span indexes into.
        group: One span from :func:`group_messages`.

    Returns:
        The names, empty when the span holds only results whose declaration sits elsewhere.
    """
    return {
        content.name
        for message in messages[group["start_index"] : group["end_index"] + 1]
        for content in message.contents
        if content.type == "function_call" and content.name
    }


def _group_result_text(messages: Sequence[Message], group: dict[str, Any]) -> str:
    """Return everything the tools in one group returned, concatenated.

    Only ``function_result`` contents are read. The call's arguments are excluded on purpose:
    the model wrote those, so they are reconstructable from the conversation and quoting them
    back proves nothing about whether the *result* survived.

    Args:
        messages: The conversation the span indexes into.
        group: One span from :func:`group_messages`.

    Returns:
        The results' text, empty when the span returned nothing.
    """
    parts: list[str] = []
    for message in messages[group["start_index"] : group["end_index"] + 1]:
        for content in message.contents:
            if content.type == "function_result":
                parts.append(content.result if isinstance(content.result, str) else str(content.result))
    return "\n".join(parts)


def _distinctive_tokens(text: str) -> set[str]:
    """Return the tokens in ``text`` that look like values nothing could reconstruct.

    **The rule.** Split on whitespace; strip punctuation from both ends of each token; keep
    what is left when it is at least :data:`_MIN_DISTINCTIVE_LENGTH` characters long and
    contains at least one digit. Results are lowercased, because the comparison against the
    record is case-insensitive.

    It is deliberately generic and deliberately crude. The temptation is to match this
    benchmark's hex identifiers, and a rule fitted to those would be worthless on the next
    workload -- the same mistake the tool's own description already had to be rewritten out of
    (see :data:`RECALL_VALUES_DESCRIPTION`). A digit is the one signal shared by nearly
    everything :data:`RECALL_VALUES_DESCRIPTION` lists as unreconstructable: identifiers,
    codes, numbers, versions, timestamps, and most paths and URLs that matter.

    **What it cannot see, stated rather than discovered later.**

    - *Alphabetic values.* A name, a status word, a region, a UUID that happens to have no
      digits: none of these are found, so a group whose results hold only those yields nothing
      and falls through to the tool-name test instead. That is the whole reason the fallback in
      :meth:`ToolResultAnchoredSummarizationCompactionStrategy._drop_before` exists.
    - *Ordinary numbers.* Line numbers, counts, prices and dates are collected as though they
      were identifiers. That over-collects, which makes coverage harder to claim and keeps more
      -- the direction this package errs in everywhere.
    - *Its own leftovers.* A result already shortened by the anchored fallback carries that
      strategy's marker, whose character count is a digit-bearing token no record will ever
      quote. It costs the group one token's worth of coverage, in the keeping direction again.
    - *Substrings.* A value is "quoted" when it appears anywhere in the record, so a record
      mentioning ``AB-1234567`` also satisfies a group whose value was ``AB-123456``. Bounded
      by the length floor, and biased toward finding coverage; the alternative, tokenising the
      record too, would miss every value the model wrapped in punctuation it chose itself.

    Args:
        text: Tool result text to read.

    Returns:
        The distinctive tokens, lowercased, without duplicates.
    """
    found: set[str] = set()
    for raw in text.split():
        token = raw.strip(string.punctuation)
        if len(token) >= _MIN_DISTINCTIVE_LENGTH and any(character.isdigit() for character in token):
            found.add(token.lower())
    return found


def _is_recall_group(messages: Sequence[Message], group: dict[str, Any]) -> bool:
    """Return whether a group is itself a record rather than ordinary tool work.

    Shared by the strategy and the middleware for the same reason :func:`find_record_index` is:
    one half must not count a record as work still to be covered while the other treats it as
    the coverage.

    Args:
        messages: The conversation the span indexes into.
        group: One span from :func:`group_messages`.

    Returns:
        True when the span contains a call to the recall tool.
    """
    return RECALL_TOOL_NAME in _called_function_names(messages, group)


def _preserve_records(messages: list[Message]) -> int:
    """Mark every recall record in the conversation as protected from removal, and count them.

    Re-applied on every pass rather than set once, because compaction runs against a freshly
    loaded conversation and the annotations a previous pass wrote are not there when the next
    one starts -- the same reason the framework re-derives its own exclusion flags each time.

    *Every* record, not only the newest. An older record is the sole account of the groups
    behind it, and :meth:`ToolResultAnchoredSummarizationCompactionStrategy._drop_before`
    already refuses to delete one; without this the fallback would shorten it instead, which
    loses the same facts more quietly.

    Both halves of the identity are required. The call must name the recall tool and the result
    must carry :data:`RECORD_MARKER`, matching :func:`find_record_index`, so that an uninvited
    call the gate refused -- which returns ordinary text and preserves nothing -- does not get
    itself protected as though it had recorded something.

    Args:
        messages: The conversation, whose messages are annotated in place.

    Returns:
        How many records the conversation carries. Counted here rather than by a second walk
        because this is already the one place that applies both halves of the identity, and a
        counter disagreeing with what is protected would report a floor the prompt does not
        actually have.
    """
    records = 0
    for group in group_messages(messages):
        if not _is_recall_group(messages, group):
            continue
        members = messages[group["start_index"] : group["end_index"] + 1]
        if not any(_record_text(message) for message in members):
            continue
        records += 1
        for message in members:
            set_preserved(message, preserved=True, reason=PRESERVE_REASON)
    return records


def _droppable_groups_after(messages: list[Message], record_index: int | None) -> int:
    """Count the tool-call groups a record would be asked to cover.

    Args:
        messages: The conversation to measure.
        record_index: Index of the newest record, or None when there is none, in which case
            the count runs from the beginning of the conversation.

    Returns:
        How many non-recall tool-call groups sit after the record.
    """
    boundary = -1 if record_index is None else record_index
    count = 0
    for group in group_messages(messages):
        if group.get("kind") != "tool_call" or group["start_index"] <= boundary:
            continue
        # A record is not work that needs recording. Counting one would make every record
        # bring the next one closer, and a bound of one would force a record on every call.
        if _is_recall_group(messages, group):
            continue
        count += 1
    return count


class RecallGate:
    """One-shot permission for the recall tool.

    The tool cannot be hidden from the model. It has to be registered with the agent for the
    function-invocation layer to execute it, and that layer wraps the middleware layer, so a
    tool supplied per call reaches the model but never the executor: measured, the model's
    call simply went unanswered. A registered tool is advertised on every request, and this
    one was called uninvited on the unpinned follow-up call in every run.

    Permission is therefore separated from visibility. The middleware arms the gate
    immediately before the request it forces, and the tool records only while armed. An
    uninvited call still runs and still answers honestly; it just produces no record.
    """

    def __init__(self) -> None:
        """Start disarmed, so nothing is recorded until something asks for it."""
        self._armed = False

    def arm(self) -> None:
        """Permit the next call to record."""
        self._armed = True

    def take(self) -> bool:
        """Consume the permission.

        Returns:
            True if this call may record. One-shot: a single arming cannot licence a second
            record, which would drop results the first had already replaced.
        """
        armed, self._armed = self._armed, False
        return armed


def make_recall_tool(
    gate: RecallGate | None = None,
    *,
    target_tokens: int | None = DEFAULT_RECORD_TARGET_TOKENS,
) -> Callable[[str], str]:
    """Build the tool :class:`ToolResultAnchoredSummarizationCompactionStrategy` anchors on.

    It echoes what it is given straight back. That is the whole point: the value of the call
    is not what the tool computes but that the model's own recollection ends up in the
    transcript as a tool result, which the provider issued and which survives strategies that
    shed assistant prose.

    The docstring built here is the entire prompt for the record. The middleware sends no
    message -- one appended there would carry no history provider's source tag, so per-call
    persistence would store it and an instruction of ours would surface in the conversation
    the application replays to its user -- so the description, the ``values`` guidance and the
    stated target are the only three things steering what the model writes.

    Args:
        gate: Permission to record. Without one the tool always records, which is only right
            for a caller driving it deliberately.

    Keyword Args:
        target_tokens: Length to aim the record at, stated in the description. ``None`` states
            none. This is the only bound that makes the model *plan* to fit: the middleware's
            cap merely truncates, and truncating a tool call destroys the arguments JSON
            rather than shortening the record it carries.

    Returns:
        A callable named :data:`RECALL_TOOL_NAME`.
    """

    def tool(values: str) -> str:
        if gate is not None and not gate.take():
            return "Not required right now: nothing was recorded, and no results have been removed."
        return (
            f"{RECORD_MARKER} Earlier tool results may have been shortened, and this is their "
            "compaction record. Treat values in this record as authoritative for the tool it "
            "names, and treat information as absent only if it appears nowhere, including "
            f"here.\n{values}"
        )

    tool.__name__ = RECALL_TOOL_NAME
    sections = [RECALL_DESCRIPTION]
    if target_tokens is not None:
        # Spent where it is needed, rather than as a flat budget: the four clauses below are
        # not equally compressible, and a bare length would be read as licence to shorten the
        # verbatim half, which is the half that cannot be rewritten from anything else.
        sections.append(
            f"Aim for about {target_tokens:,} tokens in total, spent on what cannot be "
            "reconstructed rather than on the summaries."
        )
    sections.append(f"Args:\n    values: {RECALL_VALUES_DESCRIPTION}")
    # The whole docstring becomes the tool description: the framework builds the parameter
    # schema from the annotations and does not lift the Args section into it, so the guidance
    # for ``values`` only reaches the model as part of this text.
    tool.__doc__ = "\n\n".join(sections)
    return tool


class ToolResultAnchoredSummarizationCompactionStrategy:
    """Ask the agent to record the facts in a tool call, then drop what is behind it.

    Args:
        max_input_tokens: Ceiling the included prompt must stay under.
        tokenizer: Token counter, shared with whatever measures the result.

    Keyword Args:
        keep_head_groups: Groups at the start never touched, carrying the task and its
            requirements. Dropping these is what makes other strategies lose the labelling
            that gives surviving values their meaning.
        keep_tail_groups: Recent groups the fallback keeps verbatim. Read only by the default
            fallback built below; pass a ``fallback`` of your own and it is that strategy's
            business instead.
        trigger_fraction: Fraction of the ceiling at which the record is first requested.
            Below it nothing happens: a record that is not needed costs an agent turn and
            buys nothing. See :data:`DEFAULT_TRIGGER_FRACTION` for where the default sits.
        fallback_fraction: Fraction of the ceiling at which waiting stops and the conversation
            is compacted without a record. Must be greater than ``trigger_fraction``, and by
            enough for a record asked for at the trigger to arrive before this is crossed --
            it arrives one call late by construction. See :data:`DEFAULT_FALLBACK_FRACTION`.
        coverage_share: Share of a group's distinctive values the record must quote verbatim
            before that group may be deleted. Exposed because the right value depends on how
            many values a workload's results carry, which this module cannot know; see
            :data:`DEFAULT_COVERAGE_SHARE` for where the default sits and why. Zero means any
            group holding a distinctive value at all counts as covered, which restores the
            behaviour this check replaced and is there so the two can be run side by side.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        keep_head_groups: int = 3,
        keep_tail_groups: int = 4,
        trigger_fraction: float = DEFAULT_TRIGGER_FRACTION,
        fallback_fraction: float = DEFAULT_FALLBACK_FRACTION,
        coverage_share: float = DEFAULT_COVERAGE_SHARE,
        fallback: CompactionStrategy | None = None,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range, or the two thresholds are the wrong way
                around -- a fallback at or below the trigger would fire before the model had
                any chance to answer, and the recording step would never happen at all.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if not 0.0 < fallback_fraction <= 1.0:
            raise ValueError("fallback_fraction must be in (0.0, 1.0].")
        if fallback_fraction <= trigger_fraction:
            raise ValueError("fallback_fraction must be greater than trigger_fraction.")
        if keep_head_groups < 0 or keep_tail_groups < 0:
            raise ValueError("keep_head_groups and keep_tail_groups must be >= 0.")
        # Zero is admissible and 1.0 is admissible; a share above 1.0 is not, because no record
        # can quote more of a group's values than the group contains, so the strategy would
        # silently never delete anything and would read as a model that never complied.
        if not 0.0 <= coverage_share <= 1.0:
            raise ValueError("coverage_share must be in [0.0, 1.0].")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.keep_tail_groups = keep_tail_groups
        self.trigger_fraction = trigger_fraction
        self.fallback_fraction = fallback_fraction
        self.coverage_share = coverage_share
        self.fallback = fallback or AnchoredCompactionStrategy(
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
            keep_head_groups=keep_head_groups,
            keep_tail_groups=keep_tail_groups,
        )
        self._records = 0
        self._records_in_conversation = 0
        self._fallbacks = 0
        self._fallbacks_after_record = 0
        # Group ids rather than a running total, so one group examined on ten later passes is
        # one number rather than ten. See :attr:`groups_kept_uncovered`.
        self._uncovered: set[str] = set()

    @property
    def fallbacks_used(self) -> int:
        """Passes that gave up waiting for a record and truncated instead.

        Non-zero means no record ever arrived, and that row is measuring the fallback rather
        than this design. Reported rather than hidden: a strategy that quietly degrades into
        another one produces a number that belongs to neither.

        This path only. A record that did arrive and did not free enough is the same
        degradation reached by the other route, and is counted by
        :attr:`fallbacks_after_record` -- for a while it was counted nowhere, which is how a
        row measuring the fallback came to carry no flag at all.
        """
        return self._fallbacks

    @property
    def fallbacks_after_record(self) -> int:
        """Passes that had a record, dropped what it covered, and fell back anyway.

        A different event from :attr:`fallbacks_used`, which counts the passes that gave up
        waiting for a record that never arrived. This counts the passes where one did arrive
        and did not free enough, so the fallback ran behind it and shortened whatever was
        still in the prompt -- which, since the coverage check went in, is exactly the groups
        the record failed to carry and this strategy had just declined to delete.

        Kept apart from ``fallbacks_used`` because the two ask for opposite responses: no
        record at all is a model that will not comply, while a record that did not free
        enough is a ceiling, a bound, or a record too partial to be worth its size. What the
        two mean for the *row* is the same, and is why this is reported at all: a non-zero
        value says part of what that row measured is the fallback strategy rather than this
        one. Nothing said so until it was counted -- a seed reporting four uncovered groups
        was measured losing the same facts as the control, three messages shorter and 16,617
        tokens lighter, which is shortening rather than deletion and had no flag anywhere.

        Counted per pass, like ``fallbacks_used`` and unlike :attr:`groups_kept_uncovered`:
        each pass shortens whatever is in the prompt at the time rather than taking a second
        look at material already accounted for, so two passes are two losses.
        """
        return self._fallbacks_after_record

    @property
    def records_found(self) -> int:
        """Recall tool results seen in the history. Zero means the model never complied.

        Saturates at one: it answers whether the model ever complied, not how often. How many
        records a conversation ended up carrying is :attr:`records_in_conversation`, and the
        two are different questions now that the middleware may ask more than once.
        """
        return self._records

    @property
    def records_in_conversation(self) -> int:
        """Records the conversation carries, at the most this strategy has seen it hold.

        A count rather than the flag :attr:`records_found` is, because records accumulate and
        nothing removes them. Every record observed is preserved -- neither shortened nor
        dropped, by this strategy or by the fallback behind it -- so each one raises a floor
        under the prompt that no later pass can lower. One is the cost of the design; several
        is a conversation whose unshrinkable part is growing, and a row that says so can be
        told apart from a row whose compaction simply stopped working.

        Deliberately not consolidated. Merging two records would be tempting and wrong: an
        older record is the sole surviving account of the groups behind *it*, so a merge is a
        rewrite of the evidence rather than of the bulk, and a partial merge would lose facts
        with nothing left to trace them to. The count is therefore the whole of the warning.

        A maximum over passes rather than a running total, for the reason
        :attr:`groups_kept_uncovered` is counted by group id: the same conversation is
        re-examined on every later compaction, so a per-pass tally would report one record as
        eighteen.
        """
        return self._records_in_conversation

    @property
    def groups_kept_uncovered(self) -> int:
        """Tool groups a record failed to carry, and which were therefore not dropped.

        Non-zero means the model wrote a partial record and this strategy declined to delete
        what that record does not account for. The row then costs more than a complete record
        would have cost and loses nothing, which is the trade the check makes deliberately;
        :meth:`_drop_before` says why the alternative was silent loss.

        Counted by group id, not per pass. The same group is re-examined on every later
        compaction, so a per-pass tally would report one uncovered group as eighteen -- the
        mistake :attr:`ToolResultRecallMiddleware.records_volunteered` already had to be fixed
        for, and the reason a diagnostic has to be built to be read rather than merely emitted.
        """
        return len(self._uncovered)

    async def __call__(self, messages: list[Message]) -> bool:
        """Request a record, or drop what an existing record covers.

        Returns:
            True if the outgoing messages changed.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        used = included_token_count(messages)
        if used <= int(self.max_input_tokens * self.trigger_fraction):
            return False

        anchor = find_record_index(messages)
        if anchor is not None:
            self._records = max(self._records, 1)
            # Before anything is deleted on the strength of a record, the record is put out of
            # reach of the fallback that runs below. Not folded into ``changed``: annotating a
            # message is not a change to the conversation the model sees, and reporting one
            # would make a pass that did nothing else look as though it had compacted.
            #
            # The count comes back from the same walk. Taken as a maximum because this runs on
            # every later pass over the same conversation; see ``records_in_conversation``.
            self._records_in_conversation = max(self._records_in_conversation, _preserve_records(messages))
            changed = self._drop_before(messages, anchor)
            # Even a good record may not be enough on its own: the groups after it are
            # untouched by design, and they can exceed the ceiling by themselves.
            if included_token_count(messages) > self.max_input_tokens:
                # Counted, and counted apart from the fallback below. This is the other
                # strategy running over what the record did not free -- which now includes
                # every group the record did not carry -- so the row is partly measuring that
                # other strategy. Until this counter existed only the pre-record path
                # incremented anything, so a run that fell back here reported no FALLBACK at
                # all and read as this design working.
                self._fallbacks_after_record += 1
                changed = await self.fallback(messages) or changed
            return changed

        if used < int(self.max_input_tokens * self.fallback_fraction):
            # Still waiting for the middleware's forced call to come back. Nothing may be
            # dropped yet: the record is the only thing that would replace it.
            return False

        # Out of room to keep waiting. A model that has not answered by now may never answer,
        # and the alternative to compacting without a record is a provider error. The tool
        # results are lost either way at this point; at least the conversation survives.
        self._fallbacks += 1
        return await self.fallback(messages)

    def _drop_before(self, messages: list[Message], anchor: int) -> bool:
        """Exclude the tool groups the record demonstrably covers, and only those.

        This used to exclude every tool group ending before the record, on the stated
        assumption that the record had replaced them. The assumption was never checked, and it
        does not hold on every model. Measured: gpt-5.6-luna writes a record covering two of
        six tool groups; gpt-5.4-mini covers all of them. On the first, four groups were
        deleted behind a record that never mentioned them and nothing said so, so the loss
        arrived in the scores as compaction damage rather than as an instrument that had
        stopped early. Raising the response cap, raising the stated target and rewriting the
        prompt were each tried and each measured as a null result -- coverage did not move --
        so the fix has to be here, in what the strategy is willing to delete.

        **Coverage is checked in values, because models do not write tool names.** The first
        version of this check asked whether the record contained the group's function name, on
        the reading that :data:`RECALL_VALUES_DESCRIPTION` asks for the results "grouped by the
        tool that produced it". Measured on both models, that is not the clause they comply
        with. Luna's record reads *"extra0 deployment lookup returned codes: AB-123456, ..."*
        and never writes ``lookup_extra0`` at all; gpt-5.4-mini, whose records carry every value
        from every group, was scored ``UNCOVERED:4`` by the name rule and its compaction fell
        from a 20% reduction to 5-6% in exchange for nothing. A check that penalises the model
        that complied is a net negative, and this one was.

        So the test is the clause the description actually leads with -- "Quote verbatim any
        value that cannot be reconstructed or guessed" -- applied to what the group's tool
        *results* contained. :func:`_distinctive_tokens` finds those values and states the rule
        and its blind spots; a group is covered when the record quotes at least
        ``coverage_share`` of them, case-insensitively. That also dissolves the repeated-name
        ambiguity the count rule below was built for: two calls to one tool return two different
        sets of values, and a record quoting both has demonstrably accounted for both.

        **A group with no distinctive values falls back to the tool-name test.** By this rule
        nothing in such a group is unreconstructable, so the value check has no evidence either
        way -- and the two available shortcuts are both wrong. Calling it covered would let a
        record that mentions nothing delete a group of prose findings, which is the silent loss
        this whole method exists to stop. Calling it uncovered would make every prose-only tool
        permanently undroppable, which is not a conservative choice but a broken one. The name
        rule is a weaker instrument, and a weaker instrument is the right answer where the
        stronger one has nothing to read.

        **Repeated calls to the same tool cannot be told apart by name.** Six calls to
        ``lookup_eu`` produce six groups and one name, and a record grouped by tool mentions
        that name once, so the name alone cannot say which of the six it accounted for. The
        count rule therefore demands as many mentions as there are groups and keeps all of them
        when it does not get them. The demand is counted over every candidate, including those
        the value rule will settle, so a value-covered group raises the bar for a name-checked
        sibling sharing its tool. That over-demands, and it over-demands in the keeping
        direction: the failure it buys is "compacted less than hoped", which shows up as cost
        on a row anyone can read, against "lost facts silently", which shows up as a wrong
        answer with no trace of where the fact went.

        **Only the newest record is read.** Groups an older record covered are checked against
        the newer record's text and kept when it does not carry them, so a run taking several
        records compacts less than one taking a single complete record. Same conservative
        direction, and the older record is itself never dropped, so what it holds stays
        reachable.

        Returns:
            True if anything was excluded.
        """
        groups = group_messages(messages)
        record = _record_text(messages[anchor]).lower()

        candidates: list[tuple[dict[str, Any], set[str], set[str]]] = []
        for position, group in enumerate(groups):
            if position < self.keep_head_groups or group.get("kind") != "tool_call":
                continue
            if group["end_index"] >= anchor:
                continue
            if _is_recall_group(messages, group):
                # An older record. Deleting it would destroy the only surviving account of the
                # groups behind *it* -- the same loss this strategy exists to prevent, one
                # level removed, and quieter, because the newer record looks like coverage.
                continue
            if any_preserved(messages[group["start_index"] : group["end_index"] + 1]):
                # Something else has already declared this group irreplaceable. Not counted as
                # uncovered: it was never a candidate for deletion, so reporting it would put a
                # protected message in a diagnostic that means "the record fell short".
                continue
            candidates.append((
                group,
                _called_function_names(messages, group),
                _distinctive_tokens(_group_result_text(messages, group)),
            ))

        # How many groups each name has to account for, counted over the droppable candidates
        # alone. A group the head protects, or one sitting after the record, is not being
        # replaced by this record and must not raise the bar for the groups that are.
        demand: dict[str, int] = {}
        for _, names, _ in candidates:
            for name in names:
                demand[name] = demand.get(name, 0) + 1
        named = {name for name, needed in demand.items() if record.count(name.lower()) >= needed}

        changed = False
        for group, names, values in candidates:
            if not self._is_covered(record, names=names, values=values, named=named):
                self._uncovered.add(str(group["group_id"]))
                continue
            for message in messages[group["start_index"] : group["end_index"] + 1]:
                changed = set_excluded(message, excluded=True, reason="tool_summary_anchored") or changed
        return changed

    def _is_covered(self, record: str, *, names: set[str], values: set[str], named: set[str]) -> bool:
        """Return whether one group's contents demonstrably survive in ``record``.

        Args:
            record: The record's text, already lowercased.

        Keyword Args:
            names: The functions called inside the group.
            values: The distinctive tokens its results contained, lowercased.
            named: Function names the record mentions as often as they are called.

        Returns:
            True when the group may be deleted.
        """
        if values:
            # ``ceil`` rather than rounding, so the constant is a genuine floor on the share:
            # seven of eight values clears 0.8 and six does not. It also makes 1.0 mean every
            # value and 0.0 mean none, which is what those two ends have to mean for the
            # keyword to be usable as a dial across its whole range.
            quoted = sum(1 for value in values if value in record)
            return quoted >= ceil(len(values) * self.coverage_share)
        # No names either means a span of results whose declaration sits outside it, so there
        # is nothing at all to check the record against. Kept, on the same principle as
        # everything else here.
        return bool(names) and names.issubset(named)

    def _excluded(self, messages: list[Message]) -> int:
        """Return how many messages are currently excluded, for tests and diagnostics."""
        return sum(1 for message in messages if message.additional_properties.get(EXCLUDED_KEY, False))


class ToolResultRecallMiddleware(ChatMiddleware):
    """Force the recall call once, so phase 2 has something to anchor on.

    Args:
        max_input_tokens: Ceiling the prompt must stay under, matching the strategy's.
        tokenizer: Token counter, matching the strategy's.

    Keyword Args:
        arm: Called immediately before the forced request, arming the recall tool for one
            call. The tool is inert otherwise, which is what stops the model producing a
            record on its own initiative -- it cannot be hidden, only disabled.
        trigger_fraction: Fraction of the ceiling at which the record is forced. Comfortably
            below the strategy's fallback threshold, because the decision is made one call
            late -- see :meth:`process`. Defaults to :data:`DEFAULT_TRIGGER_FRACTION`, the
            same constant the strategy defaults to, so the two halves cannot silently disagree
            about when a record is wanted.
        repeat_records: Let the size trigger ask again once there is new tool work to record.
            On by default. Off, it asks exactly once per conversation and never again, which
            is what this did before repeats existed and is the setting a run has to use to be
            comparable with one taken before them.

            Repeating cannot be done by reading size alone, and the gate that used to sit here
            is why: the size that fired the trigger does not go away when a record arrives,
            because the record is added to the conversation rather than subtracted from it. A
            trigger reading size alone would therefore pin every remaining call in the run.
            What re-arms it is new *material* -- see :meth:`_record_due` for the rule, which is
            stated there once and nowhere else.

            This governs the size trigger only. ``max_groups_before_record`` is a caller
            asking for repeats outright, so it keeps forcing them whatever this says.
        record_max_tokens: Cap put on the forced call's response, and on no other call.
            ``None`` leaves whatever cap the run already sets, which is what this did before
            the parameter existed: the record inherited the cap sized for an ordinary answer,
            so a record asked to summarise everything had no bound of its own at all.
        max_groups_before_record: How many tool-call groups one record may be asked to cover
            before another is forced. ``None`` switches this bound off and leaves the size
            trigger as the only thing that asks. It is a second trigger beside
            ``trigger_fraction`` rather than a replacement for it: whichever fires first
            forces the call. ``None`` used to mean "one record per run" as well, which
            conflated a bound with a policy; that half is now ``repeat_records``.

            It exists because coverage does not scale with how much there is to cover.
            Measured: gpt-5.6-luna's record covered two of six tool groups, and raising the
            response cap, raising the stated target and rewriting the prompt each left that
            unchanged. What was still within reach was asking each record for less, which is
            what this bounds. The strategy will now keep whatever a record does not cover, so
            an unbounded ask degrades into compacting almost nothing rather than into losing
            facts -- this is the parameter that buys the compaction back.

            **The count is an approximation, and the direction it errs in is chosen.** This
            middleware holds no reference to the strategy, so it cannot read
            ``keep_head_groups`` and cannot tell a group the strategy protects from one it
            would drop. It counts every non-recall tool-call group after the newest record,
            which over-counts by at most the number of tool groups inside the head -- usually
            none, since the head carries the task rather than tool work. Over-counting forces
            a record slightly early and costs an agent turn; under-counting would let a record
            be asked to cover more than the model will, which is the thing this prevents.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        arm: Callable[[], None],
        trigger_fraction: float = DEFAULT_TRIGGER_FRACTION,
        record_max_tokens: int | None = DEFAULT_RECORD_MAX_TOKENS,
        max_groups_before_record: int | None = None,
        repeat_records: bool = True,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if record_max_tokens is not None and record_max_tokens <= 0:
            raise ValueError("record_max_tokens must be positive, or None to leave the run's cap in place.")
        if max_groups_before_record is not None and max_groups_before_record <= 0:
            raise ValueError(
                "max_groups_before_record must be positive, or None to leave the bound off. "
                "Zero groups per record is a record forced on every call, which is not a bound."
            )
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.arm = arm
        self.trigger_fraction = trigger_fraction
        self.record_max_tokens = record_max_tokens
        self.max_groups_before_record = max_groups_before_record
        self.repeat_records = repeat_records
        self._force_next = False
        self._forced = 0
        self._records_forced = 0
        self._records_volunteered = 0
        self._records_truncated = 0
        self._seen_record = False

    def forget_pending(self) -> None:
        """Drop the decision to force a record on the next call.

        The decision is taken on one call and applied to the next, which makes it part of the
        conversation rather than of this object: a decision taken while the conversation was
        being seeded would fire on the first question asked of the snapshot and on none of the
        others, so that one probe would carry a prompt the rest do not. Restoring the snapshot
        has to restore this too.
        """
        self._force_next = False

    @property
    def forced_calls(self) -> int:
        """How many times the recall tool was forced. Zero means phase 1 never fired."""
        return self._forced

    @property
    def records_forced(self) -> int:
        """Records that appeared on a call this middleware pinned."""
        return self._records_forced

    @property
    def records_volunteered(self) -> int:
        """Records the model produced without being pinned.

        Not a success. A turn's pinned ``tool_choice`` applies only to its first call, so on
        the follow-up after a tool result the model may call any registered tool. A record
        arriving that way is the model choosing to, which is not something a strategy can rely
        on, and reporting it as though the design had worked would be the difference between a
        mechanism and a coincidence.
        """
        return self._records_volunteered

    @property
    def records_truncated(self) -> int:
        """Forced calls the provider cut short because they reached the cap.

        Non-zero means a record may be partial, and a partial record is the one failure mode
        this design does not otherwise show. A tool call cut mid-arguments produces no record
        at all, which is loud: the strategy waits, falls back and flags ``FALLBACK``. A call
        cut just after a closing brace, or repaired by the provider, yields a record that
        parses and looks complete -- and the strategy then drops every tool group behind
        something that covers only part of them, scoring the loss as compaction damage rather
        than as an instrument that ran out of room.
        """
        return self._records_truncated

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        """Force the recall tool when the last call showed the conversation is large enough.

        The decision is made *after* ``call_next`` and applied on the *following* call, which
        is not a convenience. Before the pipeline runs, ``context.messages`` holds only the new
        turn: the history middleware sits deeper and replaces it with the loaded conversation
        during the call. There is no earlier point at which the size of the history can be
        known, so the check reads it on the way out and the option is set on the way in next
        time. The trigger sits well below the strategy's fallback threshold to absorb that
        one-call delay.

        The same delay is why a forced call can be forced again: the record it produced is not
        in ``context.messages`` yet, so the condition that fired still reads as true and the
        next call is pinned too. That is existing behaviour rather than a cost of repeats --
        the middleware has always kept asking until a record appears in the loaded history --
        but once records repeat it recurs once per record instead of once per run.

        Whether the *next* call is pinned is the whole of the decision, and it is taken in
        :meth:`_record_due`, which is the one place the rule is written down.
        """
        forced_this_call = self._force_next
        if forced_this_call:
            # Replaced rather than mutated: options may be shared with the caller's own dict,
            # and pinning a tool choice into it would outlive this call.
            # The tool is offered on this call and no other. Registering it on the agent
            # would put its schema in every request, and its description reads as sensible
            # hygiene right after a lookup -- which is exactly what happened: the model called
            # it unprompted on the unpinned follow-up call, and the run then measured the
            # model's initiative rather than this middleware. Options replace the tool list
            # rather than adding to it, so offering it here also hides everything else, which
            # is harmless on a call whose only purpose is to make this one call.
            self.arm()
            options: dict[str, Any] = {
                **dict(context.options or {}),
                "tool_choice": {"mode": "required", "required_function_name": RECALL_TOOL_NAME},
            }
            if self.record_max_tokens is not None:
                # Only here. The run's own cap is sized for an answer to the user and every
                # other call needs it; this one call writes a record instructed to summarise
                # everything, which is the only place a runaway is affordable at all.
                options["max_tokens"] = self.record_max_tokens
            context.options = options
            self._force_next = False
            self._forced += 1

        await call_next()

        if forced_this_call and isinstance(context.result, ChatResponse) and context.result.finish_reason == "length":
            # Taken from the provider rather than inferred from the record's length: the model
            # is free to write a short record, and a short record is not a cut one. This is
            # the provider saying it stopped generating because it hit the ceiling, which is
            # the only statement that separates the two.
            #
            # A streamed result is not a ChatResponse yet, so it is not read here. Recording is
            # a one-shot forced call whose answer nobody displays, so there is nothing to
            # stream it for -- but a caller that did would lose this count, not get a wrong one.
            self._records_truncated += 1

        messages = list(context.messages)
        if not messages:
            self._force_next = False
            return
        record_index = find_record_index(messages)
        # The transition is tracked on the instance, not read from the messages on the way in.
        # Before the pipeline runs, context.messages holds only the new turn, so a pre-call
        # check reports "no record" on every call and every later call counts as a fresh one --
        # which is how an 18 appeared here for a single record.
        if record_index is not None and not self._seen_record:
            self._seen_record = True
            # Attributed, not merely counted. A record that arrived unpinned came from the
            # model volunteering on a follow-up call, and that is a different claim.
            if forced_this_call:
                self._records_forced += 1
            else:
                self._records_volunteered += 1
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        self._force_next = self._record_due(messages, record_index)

    def _record_due(self, messages: list[Message], record_index: int | None) -> bool:
        """Return whether the next call should be pinned to the recall tool.

        **The rule, written once so nobody has to derive it from three booleans.** Write
        *pending* for the droppable tool work no record accounts for: the non-recall tool-call
        groups after the newest record, or all of them when there is no record yet. Then

        - the group bound asks whenever ``pending`` reaches ``max_groups_before_record``,
          whatever else is true, because setting that bound is asking for repeats outright;
        - the size trigger asks for the *first* record as soon as the prompt passes
          ``trigger_fraction`` of the ceiling;
        - it asks again only when ``pending`` is at least one -- and not at all when
          ``repeat_records`` is off, which is what every run before repeats existed did.

        **Why the second record needs ``pending`` and the first does not.** The size that fires
        the trigger does not go away once a record exists: the record is *added* to the
        conversation, and it is preserved, so the prompt is if anything larger afterwards. A
        repeat reading size alone would therefore stay true for the rest of the run and pin
        every remaining call. That is why the size trigger used to be gated on there being no
        record at all, and it is the regression to watch for when un-gating it: a conversation
        sitting above the trigger with nothing recorded since its last record is *settled* --
        there is nothing a second record could carry that the first does not -- and must be
        left alone until the agent does more tool work.

        Args:
            messages: The loaded conversation, already grouped and token-annotated.
            record_index: Index of the newest record, or None when there is none.

        Returns:
            True when the next call should be forced.
        """
        pending = _droppable_groups_after(messages, record_index)
        if self.max_groups_before_record is not None and pending >= self.max_groups_before_record:
            return True
        if record_index is not None and (not self.repeat_records or pending < 1):
            return False
        return included_token_count(messages) > int(self.max_input_tokens * self.trigger_fraction)
