# Copyright (c) Microsoft. All rights reserved.

"""Tests for running the record strategy and the user-turn strategy over one conversation.

The composition owns no selection rule, so almost nothing here is about which messages get
removed -- that is tested beside each part. What is tested here is the four things composing
adds, each of which fails silently rather than loudly.

**Both halves act, on one pass, judged by one number.** The point of the row is that the two
reach material neither can reach alone, and the failure mode is a pass where one half quietly
did nothing: the conversation is smaller, the row looks like it worked, and it is one of the
single rows under a new name. That is not hypothetical -- it is what the row first shipped as,
because the record phase's removals held the prompt below the line the user phase read. So the
headline test runs on a ceiling whose shared line sits *between* the size the conversation
starts at and the size the record phase leaves: both halves fire when both are judged against
the size the pass began with, and the user half declines the moment anything judges it against
what the record phase left. Reverting the pass to sequential judging is what that test is
written to fail on.

**One line for both halves, and the single rows keep their own.** The composition judges the
user half at the record half's trigger, so a test that is about the shipped row builds it with
that default and a test that is about the row this class used to be passes
``user_trigger_fraction`` explicitly. Both are here, next to each other, because the pair is
the argument: the aligned row fires both halves and the split row starves one. And what is
asserted of the single rows is that neither of them moved -- the same instance, run as its own
row, still reads its own trigger before and after a composition has run it.

**The order is the record phase first, and one of its three reasons is now moot.** The record
strategy's trigger is the lower of the two and the middleware that asks the model for a record
reads the conversation on the next call, so a pass that let the user half shrink the prompt
first would not delay the record -- it would stop it being asked for. That reason is asserted
directly. The reason that is gone is "the phase that removes less goes first, so the user line
survives it": the user line now survives either order by construction, which is what the
headline test pins.

**The starvation counter says which silence is the composition's doing.** A record phase that
removes enough to take the prompt under the user phase's own trigger leaves a row reporting no
user compactions, which is indistinguishable from a band that held nothing. That reading is the
counter's whole job. On the aligned row it must read zero -- the record phase never acts below
the line the user half is consulted at -- and on the split row it must read the run it was
written for, so both are tested.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, ChatResponse, Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
    project_included_messages,
)
from agent_framework_lab_cachebench.compaction._composed import (
    ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy,
)
from agent_framework_lab_cachebench.compaction._preserve import PRESERVE_REASON_KEY, is_preserved
from agent_framework_lab_cachebench.compaction._toolsummary import (
    DEFAULT_TRIGGER_FRACTION,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    ToolResultAnchoredSummarizationCompactionStrategy,
    find_record_index,
)
from agent_framework_lab_cachebench.compaction._usersummary import (
    DEFAULT_USER_TRIGGER_FRACTION,
    SUMMARY_MODE_FOLD,
    USER_SUMMARY_MARKER,
    UserTurnAnchoredSummarizationCompactionStrategy,
)

TOKENIZER = CharacterEstimatorTokenizer()

#: Characters of filler in every user turn and every assistant reply.
_TURN_CHARS = 4_000

#: Characters of filler in every tool result, and small beside a turn on purpose.
#:
#: The tool half of this fixture is about a ninth of it, which is the shape the composition
#: exists for: the record strategy is capped at that ninth however well it works, and the
#: question the row asks is what the two halves reach together. A fixture whose tool payload
#: rivalled its user text would let the record phase alone look like the composition working.
_PAYLOAD_CHARS = 2_000

#: A ceiling every line in play is crossed on, whichever of them a half is judged against.
#:
#: The eight-turn fixture measures 18,937 tokens and the record phase takes it to 16,613. The
#: composed row's shared line is 0.6 of this, 12,000, and the ``user_summary_anchored`` row's
#: own line is 0.8 of it, 16,000: both sizes are above both, so a half that declines here
#: declined for a reason of its own rather than for want of a trigger. The ceiling itself is
#: above the post-record size, so the record phase does not reach for its fallback and no
#: assertion here is about the anchored strategy.
#:
#: **Recompute the sizes whenever a default moves, and check the margins rather than the
#: signs.** A fixture that slips under a trigger does not fail; it asserts against a phase that
#: returned without doing anything, and passes. The fixture-sizing test below is what makes
#: that loud, and it is the only test here that should ever need these numbers rewritten.
_COMPACTING_CEILING = 20_000

#: A ceiling whose *split* user line sits between the fixture's two sizes.
#:
#: 0.8 of 22,000 is 17,600: above the 16,613 the record phase leaves and below the 18,937 the
#: conversation starts at. It is the line the record phase's removals used to take the prompt
#: under within a single pass, and the test on it is that they no longer can -- at either
#: fraction, because the size that decides is read before the record phase runs. The shared line
#: here is 0.6 of it, 13,200, which both sizes clear. 0.9 of it is 19,800, so a record-less pass
#: is still waiting rather than falling back, which is what makes the still-waiting test about
#: the composition and not about the anchored strategy.
_NARROW_CEILING = 22_000

#: A ceiling whose *shared* line sits between the fixture's two sizes, which is the headline.
#:
#: 0.6 of 28,000 is 16,800: above the 16,613 the record phase leaves and below the 18,937 the
#: conversation starts at. A user half judged against the size its pass began with fires here
#: and a user half judged against what the record phase left declines, so this one ceiling is
#: the difference between the composition and a sequential one wearing a shared fraction. 0.8 of
#: it is 22,400, above the whole fixture, which is what lets the same number stand for "and the
#: ``user_summary_anchored`` row, reading its own trigger, does not fire at all".
_SHARED_LINE_CEILING = 28_000

#: A ceiling neither trigger is anywhere near, so a pass over the fixture must do nothing.
#:
#: 0.6 of this is 60,000 against a fixture of about 18,900, which is 32% of the lower of the
#: two lines rather than a value sitting near it.
_IDLE_CEILING = 100_000

#: The ceiling the growing fixture is run against, and the one the live composed row had.
#:
#: Sized so the record phase's trigger (0.6 of it, 12,000) is crossed part-way through a
#: twenty-turn run and its removals then hold the prompt below a *split* user line (0.8 of it,
#: 16,000) for the whole of the rest. That relationship is what the starvation tests are about,
#: and they assert it of the fixture before they assert anything of the strategy. The same
#: fixture run at the shared line is the opposite reading: the user half is consulted on every
#: pass the record phase acts on, and the counter reads zero.
_GROWING_CEILING = 20_000


class _Summarizer:
    """A summarizer that answers from a script and counts what it was asked."""

    def __init__(self, text: str = "The user asked for the earlier things, in order.") -> None:
        self.calls = 0
        self.text = text

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        self.calls += 1
        return ChatResponse(messages=[Message(role="assistant", contents=[self.text])])


class _FailingSummarizer:
    """A summarizer that raises, which is how the user half is made to decline."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        self.calls += 1
        raise RuntimeError("the summarizer is unavailable")


def _tool_group(index: int) -> list[Message]:
    """Return one tool call and its result, carrying a value a record can quote.

    Args:
        index: Numbers the pair, its call id and its tool name.

    Returns:
        The two messages.
    """
    call_id = f"call_{index}"
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": call_id, "name": f"lookup_{index}", "arguments": "{}"}],
            message_id=f"c{index}",
        ),
        Message(
            role="tool",
            contents=[
                {
                    "type": "function_result",
                    "call_id": call_id,
                    "result": f"code_1=CODE-{index} " + "x" * _PAYLOAD_CHARS,
                }
            ],
            message_id=f"r{index}",
        ),
    ]


def _record_messages(values: str) -> list[Message]:
    """Return the matched recall call and result the record phase anchors on.

    A matched pair rather than a bare tool result, because the strategy refuses a record whose
    call the provider never issued -- which is the case that breaks on routes tracking tool
    calls server-side.

    Args:
        values: The record's text, as the model would have written it.

    Returns:
        The two messages.
    """
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": "rec", "name": RECALL_TOOL_NAME, "arguments": "{}"}],
            message_id="rec_call",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": "rec", "result": f"{RECORD_MARKER} {values}"}],
            message_id="rec_res",
        ),
    ]


def _conversation(*, user_turns: int = 8, tool_turns: int = 4, record: str | None = "") -> list[Message]:
    """Return a conversation with material for both halves in it.

    The tool groups are interleaved after the earliest turns rather than gathered at the end,
    so the user band the second phase reads spans them: a fixture whose tool work sat entirely
    behind the band would let a composition that reordered the phases pass anyway.

    Args:
        user_turns: User/assistant pairs to build.
        tool_turns: How many of those pairs are followed by a tool call and its result.
        record: Values for a record appended at the end. ``""`` builds the record that covers
            every tool group; a string builds that record instead; None appends none at all,
            which is the conversation where the record phase is still waiting.

    Returns:
        The messages, opening with the system message a real conversation opens with.
    """
    messages = [Message(role="system", contents=["You are an assistant."], message_id="sys")]
    for index in range(user_turns):
        messages.append(Message(role="user", contents=[f"Turn {index}: " + "u" * _TURN_CHARS], message_id=f"u{index}"))
        messages.append(
            Message(role="assistant", contents=[f"Reply {index}: " + "a" * _TURN_CHARS], message_id=f"a{index}")
        )
        if 1 <= index <= tool_turns:
            messages += _tool_group(index)
    if record is not None:
        messages += _record_messages(record or _covering_record(tool_turns))
    return messages


def _covering_record(tool_turns: int) -> str:
    """Return a record quoting every value the tool groups returned.

    Args:
        tool_turns: How many groups the record accounts for.

    Returns:
        The record's text.
    """
    return " ".join(f"lookup_{index}: CODE-{index}." for index in range(1, tool_turns + 1))


def _size(messages: list[Message]) -> int:
    """Return the included token count, re-read rather than taken from the cached annotations.

    Args:
        messages: The conversation to measure.

    Returns:
        Included tokens.
    """
    annotate_message_groups(messages)
    annotate_token_counts(messages, tokenizer=TOKENIZER, force_retokenize=True)
    return included_token_count(messages)


def _rendered(messages: list[Message]) -> str:
    """Return what the model would be sent, excluded messages left out."""
    parts: list[str] = []
    for message in project_included_messages(messages):
        for content in message.contents:
            result = getattr(content, "result", None)
            text = getattr(content, "text", None)
            parts.append(str(result) if result is not None else (text if text is not None else str(content)))
    return "\n".join(parts)


def _user_texts(messages: list[Message]) -> list[str]:
    """Return the text of every user message still being sent, in order."""
    return [message.text or "" for message in project_included_messages(messages) if message.role == "user"]


def _record_phase(
    ceiling: int = _COMPACTING_CEILING, **kwargs: Any
) -> ToolResultAnchoredSummarizationCompactionStrategy:
    """Return the record-then-drop half, configured as its own row configures it."""
    return ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=ceiling, tokenizer=TOKENIZER, **kwargs)


def _user_phase(
    ceiling: int = _COMPACTING_CEILING, summarizer: Any = None, **kwargs: Any
) -> UserTurnAnchoredSummarizationCompactionStrategy:
    """Return the user-band half, configured as its own row configures it."""
    return UserTurnAnchoredSummarizationCompactionStrategy(
        max_input_tokens=ceiling, tokenizer=TOKENIZER, client=summarizer or _Summarizer(), **kwargs
    )


def _composed(
    ceiling: int = _COMPACTING_CEILING,
    *,
    tool_results: ToolResultAnchoredSummarizationCompactionStrategy | None = None,
    user_turns: UserTurnAnchoredSummarizationCompactionStrategy | None = None,
    user_trigger_fraction: float | None = None,
) -> ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Return both halves composed, each defaulted to its own row's configuration.

    ``user_trigger_fraction`` is left at None by every test that is about the shipped row: the
    default aligns the user half to the record half's trigger, and that alignment is most of
    what is under test here. ``_split_composed`` is the other configuration.
    """
    return ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER,
        tool_results=tool_results or _record_phase(ceiling),
        user_turns=user_turns or _user_phase(ceiling),
        user_trigger_fraction=user_trigger_fraction,
    )


def _split_composed(
    ceiling: int = _COMPACTING_CEILING,
) -> ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Return the composition with its two halves deliberately set apart, 0.6 against 0.8.

    The row this class shipped as, reachable now only by asking for it. Every test of the
    starvation counter builds this, because on the aligned row that counter is zero by
    construction -- and a counter that can only be zero is a counter nothing checks.
    """
    return _composed(ceiling, user_trigger_fraction=DEFAULT_USER_TRIGGER_FRACTION)


#: Characters of tool payload in the growing fixture, which is where its bulk is.
#:
#: Nine times a user turn, so the record phase's removals are large enough to hold the prompt
#: under the user phase's line for the whole run. That is not an extreme: the live run this
#: fixture stands in for used ``--scale-payload``, and its composed row settled at 63% of the
#: window with the user line at 80% of it.
_GROWING_PAYLOAD_CHARS = 9_000

#: Characters in a user turn of the growing fixture.
_GROWING_USER_CHARS = 500

#: Characters in an assistant reply of the growing fixture.
_GROWING_REPLY_CHARS = 1_000


def _growing_turn(index: int) -> list[Message]:
    """Return one turn of the growing fixture: a user turn, a reply, and a tool call with it.

    Args:
        index: Numbers the turn and the call it carries.

    Returns:
        The four messages.
    """
    call_id = f"call_{index}"
    return [
        Message(role="user", contents=[f"Turn {index}: " + "u" * _GROWING_USER_CHARS], message_id=f"u{index}"),
        Message(role="assistant", contents=[f"Reply {index}: " + "a" * _GROWING_REPLY_CHARS], message_id=f"a{index}"),
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": call_id, "name": f"lookup_{index}", "arguments": "{}"}],
            message_id=f"c{index}",
        ),
        Message(
            role="tool",
            contents=[
                {
                    "type": "function_result",
                    "call_id": call_id,
                    "result": f"code_1=CODE-{index} " + "x" * _GROWING_PAYLOAD_CHARS,
                }
            ],
            message_id=f"r{index}",
        ),
    ]


def _numbered_record(indices: list[int], serial: int) -> list[Message]:
    """Return a record covering ``indices``, with ids of its own so several can coexist.

    ``_record_messages`` writes one fixed pair of ids, which is right for a conversation handed
    to one pass and wrong for a run where the model writes a record more than once.

    Args:
        indices: The tool groups the record accounts for.
        serial: Numbers this record's call and result.

    Returns:
        The two messages.
    """
    values = " ".join(f"lookup_{index}: CODE-{index}." for index in indices)
    return [
        Message(
            role="assistant",
            contents=[
                {"type": "function_call", "call_id": f"rec{serial}", "name": RECALL_TOOL_NAME, "arguments": "{}"}
            ],
            message_id=f"rec_call{serial}",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": f"rec{serial}", "result": f"{RECORD_MARKER} {values}"}],
            message_id=f"rec_res{serial}",
        ),
    ]


async def _grow_composed(
    strategy: ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy, turns: int
) -> list[tuple[int, int]]:
    """Run one composed pass per turn over a conversation that never stops growing.

    The recall middleware is stood in for rather than wired: once the prompt is past the record
    phase's trigger and there is tool work no record accounts for, a record covering all of it is
    appended, which is what the middleware's forced call produces one turn later. Nothing here
    depends on the timing of that, and the phase under test reads only whether a record is
    present.

    Args:
        strategy: The composed strategy, called once per turn.
        turns: How many turns to seed.

    Returns:
        One ``(turn, prompt tokens before the pass)`` per turn.
    """
    messages = [Message(role="system", contents=["You are an assistant."], message_id="sys")]
    covered: set[int] = set()
    seen: list[int] = []
    passes: list[tuple[int, int]] = []
    for index in range(turns):
        messages += _growing_turn(index)
        seen.append(index)
        if _size(messages) > strategy.tool_results.max_input_tokens * strategy.tool_results.trigger_fraction and (
            set(seen) - covered
        ):
            messages += _numbered_record(seen, len(covered))
            covered = set(seen)
        before = _size(messages)
        await strategy(messages)
        passes.append((index, before))
    return passes


async def test_two_lines_still_let_the_record_phase_hold_the_prompt_under_the_user_one() -> None:
    """Defect 2, kept as a test of the configuration it belongs to now that it is opt-in.

    Measured live: seed 1 of the composed row reported ``REC:1, RECORDS:1, FORCED:1,
    RECFORCED:1`` with a 63% snapshot, no ``USERCOMPACT`` **and no ``USERSTARVED``**. The user
    half had done nothing and the row's own diagnostic was silent about why.

    The cause is the two triggers rather than anything about the band. The record phase fires at
    0.6 and the user phase at 0.8, so on a workload whose bulk is tool payload the record phase
    removes it while the prompt is still in the 60s and holds it there for the rest of the run.
    The prompt is then never above the user line when a pass *starts*, which is what the first
    version of the counter required: it tested ``before > user_line >= after`` within one pass,
    and that transition never happens.

    The shipped row no longer has two lines, so this builds the split one deliberately -- and
    that is the point of keeping the test. The fix is not that the counter now reads zero; it is
    that the configuration which starves a half is the one a caller has to ask for. A run that
    asks for it still gets the row, and still gets a number saying so.

    So the fixture is asserted to be the live case before anything else: no pass over it ever
    starts above the user line, which is a proof that the first definition could not have counted
    one -- and the counter, which asks whether the prompt would be over the line with what the
    record phase removed out of the user half's reach still in it, counts nearly all of them.
    """
    strategy = _split_composed(_GROWING_CEILING)

    passes = await _grow_composed(strategy, 20)
    user_line = int(_GROWING_CEILING * strategy.user_trigger_fraction)

    assert max(before for _, before in passes) <= user_line, (
        "the fixture has to be the live case: no pass starts above the user line, so the "
        "within-pass transition the counter used to test for cannot happen on any of them"
    )
    assert strategy.records_found == 1, "the record phase is acting, which is what does the holding"
    assert strategy.user_compactions == 0, "and the user half never gets a turn"
    assert strategy.user_passes_starved >= 10, (
        "which is the record phase's doing on most of the run, and has to be said in a number"
    )
    assert strategy.user_passes_below_trigger == len(passes), "every pass ended under the user line"
    assert strategy.tokens_removed_out_of_user_reach > user_line - min(before for _, before in passes[-5:]), (
        "the counterfactual rests on this quantity, so it is checked rather than trusted"
    )
    assert strategy.tokens_removed_out_of_user_reach <= strategy.tokens_removed_by_record_phase, (
        "and it is a part of what the record phase removed rather than a second count beside it"
    )


async def test_a_composed_row_whose_user_half_did_nothing_always_says_why() -> None:
    """The failure the composition exists to avoid, asserted as a property of every pass.

    A composed row that silently degrades to one half is worse than either single row: it is
    smaller than the control, returns True, and reads as a working measurement. The wiring test
    beside this one cannot catch it -- it checks the middleware is attached, and it passed
    throughout the run where this happened.

    So what is asserted is the invariant rather than one outcome: over a run of twenty passes,
    every pass in which the user half did not compact is accounted for by exactly one counter,
    and on the split row the account is not merely "it was under its line" but *why* it was --
    the phase in front of it. A composition that reverted to reporting nothing would leave the
    starvation count at zero and fail here, not in a table six weeks later.

    Built split on purpose. The aligned row has no silent user half to account for on this
    fixture, which is the test one below; this one holds the partition open for the row where
    the silence is real.
    """
    strategy = _split_composed(_GROWING_CEILING)

    passes = await _grow_composed(strategy, 20)

    accounted = (
        strategy.user_compactions
        + strategy.user_passes_declined
        + strategy.user_passes_below_trigger
        + strategy.user_summary_failures
    )
    assert accounted == len(passes), "every pass lands in exactly one of the user half's four outcomes"
    assert strategy.user_compactions == 0, "the user half did nothing on this fixture"
    assert strategy.user_passes_starved > 0, (
        "so something other than 'the conversation was small' has to be saying why, and the "
        "only candidate is the phase in front of it"
    )
    assert strategy.user_passes_starved <= strategy.user_passes_below_trigger, (
        "starvation is a subset of the passes the user half was never consulted on, not a second count beside them"
    )


async def test_the_three_reasons_a_composed_user_half_is_silent_read_differently() -> None:
    """Declined by hysteresis, starved by the record phase, never considered -- from the flags.

    They ask for three different responses: lower ``min_band_share``, align the two triggers (or
    accept that a row asked for two lines is a row whose halves fire apart), or run a longer
    conversation. A row that reported one number for all three would send a reader to the wrong
    knob, and the run that produced this counter reported *no* number for any of them.

    The three are still three after the alignment, which is what this pins. A reader of the
    aligned row who sees ``USERHELD`` must not be able to confuse it with the record phase having
    eaten the prompt, and a reader of a split row who sees ``USERSTARVED`` must not read it as a
    band that was too small.
    """
    never = _composed(_IDLE_CEILING)
    starved = _split_composed(_GROWING_CEILING)
    declined = _composed(
        tool_results=_record_phase(_COMPACTING_CEILING),
        user_turns=_user_phase(_COMPACTING_CEILING, keep_head_user_turns=3, keep_tail_user_turns=4),
    )

    assert await never(_conversation()) is False
    await _grow_composed(starved, 20)
    assert await declined(_conversation()) is True, "the record phase acted; the user half is what did not"

    assert (never.user_passes_below_trigger, never.user_passes_starved, never.user_passes_declined) == (1, 0, 0)
    assert starved.user_passes_starved > 0 and starved.user_passes_declined == 0
    assert declined.user_passes_declined == 1, "over its line, and the band between those anchors is one turn"
    assert (declined.user_passes_starved, declined.user_passes_below_trigger) == (0, 0), (
        "which is not starvation: the prompt was over the user line when the user half read it"
    )


async def test_the_fixture_sits_where_the_four_ceilings_assume_it_does() -> None:
    """A fixture that drifts across a trigger asserts against a phase that did nothing.

    Every test here rests on two sizes -- the conversation's, and the conversation's once the
    record phase has dropped what the record covers -- and on where each ceiling's two lines
    fall relative to them. A drift in any of that turns an assertion about composing into an
    assertion about an early return, which passes. So the arithmetic is checked once, here,
    rather than trusted to the comments on the constants.
    """
    before = _size(_conversation())
    compacted = _conversation()
    assert await _record_phase()(compacted) is True
    after = _size(compacted)

    assert after < before, "the record phase has to remove something or half of this is vacuous"
    assert after > 0.6 * _COMPACTING_CEILING, "the record line stays crossed on the compacting ceiling"
    assert after > 0.8 * _COMPACTING_CEILING, "and so does the user line, which is what lets both halves act"
    assert after <= _COMPACTING_CEILING, "and the record phase never reaches for its fallback there"
    assert after < 0.8 * _NARROW_CEILING < before, "the narrow ceiling's split user line sits between the two sizes"
    assert 0.9 * _NARROW_CEILING > before > 0.6 * _NARROW_CEILING, "where a record-less pass is still waiting"
    assert after <= 0.6 * _SHARED_LINE_CEILING < before, (
        "the headline ceiling's *shared* line sits between them, which is the whole of what "
        "makes that test bite: a user half judged against the size the pass began with fires "
        "there and one judged against what the record phase left does not"
    )
    assert before < 0.8 * _SHARED_LINE_CEILING, (
        "and the user row's own line is above the fixture entirely, so the same ceiling says "
        "'the composed row's user half fires where its own row's would not'"
    )
    assert after <= _SHARED_LINE_CEILING, "no fallback there either"
    assert before < 0.6 * _IDLE_CEILING, "and neither line is anywhere near on the idle one"
    assert before < 0.95 * _COMPACTING_CEILING, (
        "the 0.95 line one test below sets has to sit above the fixture's *uncompacted* size, "
        "because that is the size the user half is judged against: below it the half declines "
        "at its trigger, which is what that test reads"
    )


async def _sizes_at(ceiling: int) -> tuple[int, int, int]:
    """Return what one pass of the composed row, the record row and the user row each leave.

    Args:
        ceiling: The shared ceiling all three are built against.

    Returns:
        Included tokens after the composed pass, after the record pass and after the user pass.
    """
    both = _conversation()
    tool_only = _conversation()
    user_only = _conversation()
    await _composed(ceiling)(both)
    await _record_phase(ceiling)(tool_only)
    await _user_phase(ceiling)(user_only)
    return _size(both), _size(tool_only), _size(user_only)


async def test_both_halves_act_when_only_the_size_the_pass_began_with_clears_the_shared_line() -> None:
    """The headline, on the one ceiling where sequential judging and this pass disagree.

    The shared line here is 16,800. The conversation is 18,937 and the record phase leaves
    16,613, so a user half asked "is the prompt over the line?" *after* the record phase has
    acted is being asked about 16,613 and answers no -- and the composed row is then the record
    row with a summarizer attached, which is what it shipped as, first at two fractions and then
    at one shared fraction judged sequentially. Asked about the size the pass began with it
    answers yes, and that is this test.

    Written to fail on the regression rather than to describe the feature: hand the user phase a
    freshly read count in ``__call__`` and this is a row with ``USERCOMPACT`` at zero and
    ``USERUNDER`` at one, on a conversation whose band is a third of the prompt.
    """
    strategy = _composed(_SHARED_LINE_CEILING)
    messages = _conversation()
    shared_line = int(_SHARED_LINE_CEILING * strategy.user_trigger_fraction)
    entry = _size(_conversation())
    post_record = _conversation()
    assert await _record_phase(_SHARED_LINE_CEILING)(post_record) is True
    assert _size(post_record) <= shared_line < entry, (
        "the fixture has to straddle the line or this proves nothing: the record phase's "
        "removals must be what would take the prompt under the line the user half reads"
    )

    assert await strategy(messages) is True

    assert strategy.records_found == 1, "the record phase acted"
    assert "x" * 100 not in _rendered(messages), "and dropped the tool payload"
    assert strategy.user_compactions == 1, "and the user half acted on the same pass, not the next one"
    assert "Turn 4:" not in _rendered(messages), "and replaced the band"
    assert (strategy.user_passes_below_trigger, strategy.user_passes_declined) == (0, 0), (
        "neither under its line nor holding back: it was consulted, and it acted"
    )
    assert strategy.user_passes_starved == 0, "nothing was taken out of its reach to report"
    assert _size(messages) < _size(post_record), "and the pass removed what the record phase alone does not"


async def test_the_band_is_weighed_against_the_prompt_as_it_now_stands() -> None:
    """The number the pass-entry reading deliberately does *not* decide, and why it matters.

    One reading of the prompt decides whether each half acts. It does not decide what a pass
    costs: ``min_band_share`` weighs the band against the prompt the pass would rewrite, and that
    prompt is the one in front of the user half, after the record phase. Handing the stale entry
    size to that check as well would make every band look like a smaller share than it is and
    decline passes worth running -- the same suppression the pass-entry trigger exists to remove,
    arriving through the hysteresis instead of through the trigger.

    The fixture is sized so the two denominators disagree: the band is 6,174 tokens, which is
    37.2% of the 16,613 the record phase leaves and 32.6% of the 18,937 the conversation starts
    at. At a share of 0.35 the composed row's user half fires and the same strategy alone, on the
    same conversation, declines. That pair is also the replacement for an order argument that
    went stale -- "the phase that removes less goes first" used to be about protecting the user
    half's trigger, and what running the record phase first actually buys now is a band that is a
    larger share of a smaller prompt.

    Self-guarding against fixture drift: a band that fell under 0.35 of the post-record size
    fails the first half of this, and one that rose over 0.35 of the entry size fails the second.
    """
    share = 0.35
    composed_messages = _conversation()
    composed = _composed(user_turns=_user_phase(min_band_share=share))
    alone_messages = _conversation()
    alone = _user_phase(min_band_share=share)

    assert await composed(composed_messages) is True
    assert await alone(alone_messages) is False

    assert composed.user_compactions == 1, "the band clears the share against the prompt the pass would rewrite"
    assert "Turn 4:" not in _rendered(composed_messages)
    assert (alone.user_compactions, alone.user_passes_declined) == (0, 1), (
        "and does not clear it against the larger prompt the same conversation starts at, which "
        "is what the stale reading would have weighed it against"
    )
    assert "Turn 4:" in _rendered(alone_messages)


async def test_the_composed_row_leaves_less_behind_than_either_half_alone() -> None:
    """The claim the row exists to make, in tokens, at both ceilings that fire it.

    A composition that quietly ran one half would still return True and still shrink the
    conversation; what it could not do is beat the row that reaches that half. Asserted at two
    ceilings because they fail differently: at the compacting one all three rows act and the
    composed row has to beat two working rows, and at the headline one the user row does not act
    at all -- which is the cost of aligning downward, stated in tokens rather than in prose.
    """
    composed_low, tool_low, user_low = await _sizes_at(_COMPACTING_CEILING)
    composed_high, tool_high, user_high = await _sizes_at(_SHARED_LINE_CEILING)
    entry = _size(_conversation())

    assert tool_low < entry and user_low < entry, "both single rows act at the compacting ceiling"
    assert composed_low < min(tool_low, user_low), (
        "so the composed row there is beating two rows that each did their own half's work"
    )

    assert tool_high < entry, "the record row acts at the headline ceiling"
    assert user_high == entry, (
        "and the user row does not: its own 0.8 line is above the whole fixture. That is the "
        "price of aligning down -- the composed row's user half fires where its own row's does "
        "not, and the two rows are that much less alike"
    )
    assert composed_high < min(tool_high, user_high)


async def test_the_aligned_row_reports_no_starvation_on_the_run_that_starves_the_split_one() -> None:
    """``USERSTARVED`` reads zero when both halves fire, and it reads zero for a reason.

    The same twenty-pass fixture under both configurations. The split row is the measured defect:
    the record phase acts at 0.6, the prompt never reaches 0.8, and the user half is never
    consulted. The aligned row consults it on every pass the record phase acts on, because that
    is what one line means -- so there is no pass on which the record phase removes anything the
    user half was not offered, the quantity the counter is decided against stays at zero, and so
    does the counter.

    The zero is asserted together with the quantity underneath it on purpose. A counter that
    reads zero because the row worked and a counter that reads zero because it stopped counting
    are the same number, and this package has shipped the second one twice.
    """
    aligned = _composed(_GROWING_CEILING)
    split = _split_composed(_GROWING_CEILING)

    passes = await _grow_composed(aligned, 20)
    await _grow_composed(split, 20)

    assert aligned.records_found == 1, "the record phase acted on the aligned row"
    assert aligned.user_compactions > 0, "and so did the user half, which is the whole claim"
    assert aligned.tokens_removed_by_record_phase > 0, (
        "the record phase removed something, or the zero below is vacuous"
    )
    assert aligned.tokens_removed_out_of_user_reach == 0, (
        "and none of it on a pass the user half was not consulted on, which is why the zero is "
        "structural rather than a property of this fixture"
    )
    assert aligned.user_passes_starved == 0
    assert len(passes) == 20

    assert (split.user_compactions, split.records_found) == (0, 1), "the same run with two lines is the old row"
    assert split.user_passes_starved > 0, "and it still says so"


async def test_a_half_run_as_its_own_row_reads_its_own_trigger_before_and_after_composing() -> None:
    """The alignment is a reading the composition takes, not a setting it writes.

    ``tool_summary_anchored`` and ``user_summary_anchored`` are rows in the same table as the
    composed one, and every archived number for them was produced by a strategy reading its own
    ``trigger_fraction`` off the conversation in front of it. A composition that reconfigured the
    objects it was handed -- or a shared line implemented by assigning one -- would change those
    rows too, silently, and only in runs that happened to select all three.

    So the same instance is run as its own row, then as a phase, then as its own row again. The
    ceiling is the headline one, where its own line is above the fixture and the shared line is
    below it, so the two readings give opposite answers and a leak would be loud.
    """
    user_phase = _user_phase(_SHARED_LINE_CEILING)
    record_phase = _record_phase(_SHARED_LINE_CEILING)
    standalone_first = _conversation()

    assert await user_phase(standalone_first) is False, "0.8 of this ceiling is above the fixture"
    assert user_phase.user_passes_below_trigger == 1
    assert "Turn 4:" in _rendered(standalone_first), "so the band is untouched"

    composed_messages = _conversation()
    assert await _composed(tool_results=record_phase, user_turns=user_phase)(composed_messages) is True
    assert user_phase.user_compactions == 1, "the same object, judged at the record half's line, fires"

    standalone_again = _conversation()
    assert await user_phase(standalone_again) is False, "and is unchanged as its own row afterwards"
    assert user_phase.user_passes_below_trigger == 2
    assert "Turn 4:" in _rendered(standalone_again)
    assert user_phase.trigger_fraction == DEFAULT_USER_TRIGGER_FRACTION, "nothing wrote to the object"

    standalone_record = _conversation()
    assert await record_phase(standalone_record) is True, "and the row the line was borrowed from is itself"
    assert record_phase.trigger_fraction == DEFAULT_TRIGGER_FRACTION


def test_the_composed_row_takes_its_one_line_from_the_record_half() -> None:
    """Which of the two fractions is shared, and what a sweep of it moves.

    Aligning to the record half rather than to the user half is the direction that keeps the
    record askable: the model writes it, it degrades with the bulk it is given, and the
    middleware that asks reads the same prompt -- so the alignment can only go down. Reading it
    off the record half rather than storing a constant is what keeps a ``--trigger-fraction``
    sweep moving both halves of this row together instead of splitting them apart again at every
    value but the default.
    """
    default = _composed()
    swept = _composed(tool_results=_record_phase(trigger_fraction=0.35), user_turns=_user_phase())

    assert default.tool_results.trigger_fraction == DEFAULT_TRIGGER_FRACTION
    assert default.user_trigger_fraction == DEFAULT_TRIGGER_FRACTION, "one line, and it is the record half's"
    assert default.user_turns.trigger_fraction == DEFAULT_USER_TRIGGER_FRACTION, (
        "while the object keeps the fraction its own row is measured at"
    )
    assert swept.user_trigger_fraction == 0.35, "a sweep of the record row's trigger moves both halves of this one"


def test_a_user_trigger_fraction_outside_the_unit_interval_is_refused() -> None:
    """The bounds the user half sets on its own trigger, kept where the line is overridden.

    Zero would judge the user half over on an empty conversation, where the band is empty and
    the only thing a pass can produce is a summarizer call; above one it can never fire, which is
    a composed row whose user half is off with nothing saying so. Refused here for the same two
    reasons the strategy refuses them, because a line supplied from outside is still that line.
    """
    with pytest.raises(ValueError, match="user_trigger_fraction"):
        _composed(user_trigger_fraction=0.0)
    with pytest.raises(ValueError, match="user_trigger_fraction"):
        _composed(user_trigger_fraction=1.5)


async def test_the_two_halves_compact_two_halves_of_one_conversation() -> None:
    """The whole claim, and the only test that puts the three rows side by side.

    Each part alone moves only what it owns: the record phase deletes tool groups and leaves
    every user turn verbatim, and the user-band phase replaces user turns and leaves every tool
    result where it found them. A composition that silently ran only one of them would still
    shrink the conversation, still return True, and still look like a working row -- so what is
    asserted is that the composed pass does *both* things the two single passes each do half
    of.
    """
    tool_only = _conversation()
    user_only = _conversation()
    both = _conversation()

    assert await _record_phase()(tool_only) is True
    assert await _user_phase()(user_only) is True
    assert await _composed()(both) is True

    assert "x" * 100 not in _rendered(tool_only), "the record phase drops the tool payload"
    assert "Turn 4:" in _rendered(tool_only), "and reads no user turn"
    assert "x" * 100 in _rendered(user_only), "the user phase reads no tool result"
    assert "Turn 4:" not in _rendered(user_only), "and replaces the user band"

    rendered = _rendered(both)
    assert "x" * 100 not in rendered, "composed, the tool payload is gone"
    assert "Turn 4:" not in rendered, "and so is the user band"
    assert "lookup_1: CODE-1." in rendered, "the record the deletion was licensed against survives"
    assert _size(both) < min(_size(tool_only), _size(user_only)), (
        "a composition that reached only one half cannot beat the row that reached that half"
    )


async def test_the_record_phase_runs_before_the_user_phase() -> None:
    """The order, asserted where it is decided rather than inferred from an outcome.

    It is not arbitrary, and the reason that survives the shared line is the one about the next
    call rather than this one. Both phases' removals are permanent, so a user phase that ran
    first and removed the majority share of the prompt would hand the middleware -- which reads
    the conversation on the next call, not this pass's entry size -- a prompt already below the
    line that asks for a record at all, and the row's tool half would report a model that never
    complied.

    Spied on ``compact_against`` rather than on ``__call__``, because that is what a pass calls
    now: the composition reads the size and the line once and hands both to each half, and a spy
    on the entry point a *row* uses would record nothing at all.
    """
    order: list[str] = []

    class _RecordSpy(ToolResultAnchoredSummarizationCompactionStrategy):
        async def compact_against(self, messages: list[Message], *, prompt_tokens: int, trigger_tokens: int) -> bool:
            order.append("record")
            return await super().compact_against(messages, prompt_tokens=prompt_tokens, trigger_tokens=trigger_tokens)

    class _UserSpy(UserTurnAnchoredSummarizationCompactionStrategy):
        async def compact_against(self, messages: list[Message], *, prompt_tokens: int, trigger_tokens: int) -> bool:
            order.append("user")
            return await super().compact_against(messages, prompt_tokens=prompt_tokens, trigger_tokens=trigger_tokens)

    strategy = _composed(
        tool_results=_RecordSpy(max_input_tokens=_COMPACTING_CEILING, tokenizer=TOKENIZER),
        user_turns=_UserSpy(max_input_tokens=_COMPACTING_CEILING, tokenizer=TOKENIZER, client=_Summarizer()),
    )

    assert await strategy(_conversation()) is True

    assert order == ["record", "user"]
    assert strategy.strategies == (strategy.tool_results, strategy.user_turns), (
        "the advertised order is what the pass runs, or a caller walking the parts reads a lie"
    )


async def test_even_two_lines_do_not_starve_a_half_within_one_pass() -> None:
    """The ceiling the old order argument was made on, where the argument has stopped applying.

    0.8 of this ceiling, 17,600, sits between the conversation's 18,937 and the 16,613 the record
    phase leaves. On the row this class shipped as that was the whole story: the record phase
    spent the pass, the prompt dropped under 17,600, the user half re-read the size and declined,
    and the pass was reported starved. It is what "the order decides which half acts" meant.

    Pass-entry judging removes that from the split row as well as from the aligned one, and that
    is worth a test of its own rather than a line in a docstring: 18,937 is over both lines when
    the pass begins, so both halves act at either fraction. What is left of starvation is a
    statement across passes -- a record phase whose earlier removals are missing from a later
    pass's entry reading -- and the growing fixture is where that is tested.

    The two rows are asserted to reach the same place here, which is the point: the fraction a
    half is read at stops mattering once every half is read at the same size.
    """
    split = _split_composed(_NARROW_CEILING)
    split_messages = _conversation()
    aligned = _composed(_NARROW_CEILING)
    aligned_messages = _conversation()

    assert await split(split_messages) is True
    assert await aligned(aligned_messages) is True

    assert (split.groups_kept_uncovered, aligned.groups_kept_uncovered) == (0, 0), "the record phase acted on both"
    assert "x" * 100 not in _rendered(split_messages)
    assert "x" * 100 not in _rendered(aligned_messages)

    assert split.user_compactions == 1, "the higher line was over the prompt when the pass began"
    assert aligned.user_compactions == 1
    assert (split.user_passes_starved, aligned.user_passes_starved) == (0, 0), (
        "and nothing was taken out from under either, because the size that decided was read first"
    )
    assert "Turn 4:" not in _rendered(split_messages)
    assert "Turn 4:" not in _rendered(aligned_messages)
    assert _size(split_messages) == _size(aligned_messages), (
        "same conversation, same two halves, same result: the fraction each half was read at "
        "stopped deciding anything the moment both were read at one size"
    )


async def test_a_user_half_that_would_not_have_fired_anyway_is_not_counted_as_starved() -> None:
    """The counter has to mean one thing, or it is a second way of saying "did not compact".

    On a ceiling nothing crosses, the user half declines for its own reasons and the record
    phase removed nothing that could have changed that. Counting it here would put a number on
    every idle pass and leave the flag saying nothing about the order at all.
    """
    strategy = _composed(_IDLE_CEILING)
    messages = _conversation()

    assert await strategy(messages) is False
    assert (strategy.user_compactions, strategy.user_passes_starved) == (0, 0)


async def test_the_user_half_still_fires_while_the_record_half_is_waiting_for_a_record() -> None:
    """A phase that declines must not decline for the phase behind it.

    Before a record exists the record phase's contract is to wait: it is past its trigger,
    under its give-up line, and the only thing that could replace the tool groups has not been
    written yet. That is the ordinary state of the first turns of a run, and a composition that
    treated the first phase's False as the pass's answer would make the user half unreachable
    for exactly as long as the model took to comply.
    """
    strategy = _composed(_NARROW_CEILING)
    messages = _conversation(record=None)

    assert await strategy(messages) is True

    assert (strategy.records_found, strategy.fallbacks_used) == (0, 0), "waiting, not fallen back"
    assert "x" * 100 in _rendered(messages), "so the tool payload is untouched"
    assert strategy.user_compactions == 1
    assert strategy.user_passes_starved == 0, "nothing was removed, so nothing could have starved it"


async def test_configuration_reaches_each_half_without_reaching_the_other() -> None:
    """Two rows' worth of knobs on one object, and the trigger is the only one they share.

    The composed row is only readable beside the two single rows if a sweep of either row's
    flags moves this row's matching half and nothing else. The trigger is the documented
    exception and is tested as such elsewhere; everything else has to stay separate. The two
    settings chosen here are the ones that would be most tempting to unify with it -- a coverage
    share and a pair of user anchors -- and they are asserted on behaviour rather than on
    attributes, because a constructor that stored them and a pass that read one number for both
    would pass an attribute check.
    """
    # A record naming three of the four tool groups, so the fourth is covered at a share of 0
    # and not at the default -- which is what makes the coverage assertion below about the
    # knob rather than about the fixture.
    lenient = _composed(
        tool_results=_record_phase(coverage_share=0.0),
        user_turns=_user_phase(keep_head_user_turns=2, keep_tail_user_turns=3),
    )
    lenient_messages = _conversation(record=_covering_record(3))
    default = _composed()
    default_messages = _conversation(record=_covering_record(3))

    assert await lenient(lenient_messages) is True
    assert await default(default_messages) is True

    assert (lenient.groups_kept_uncovered, default.groups_kept_uncovered) == (0, 1), (
        "the record phase read its own coverage share and nothing the user phase was given"
    )
    assert [text[:7] for text in _user_texts(lenient_messages)] == [
        "Turn 0:",
        "Turn 1:",
        USER_SUMMARY_MARKER[:7],
        "Turn 5:",
        "Turn 6:",
        "Turn 7:",
    ], "and the user phase read its own two anchors rather than the tool half's head count"
    assert (lenient.user_messages_replaced, default.user_messages_replaced) == (3, 6), (
        "the anchors moved one row's half and left the other row's alone"
    )


async def test_a_caller_can_set_the_two_halves_apart_again_and_then_they_fire_apart() -> None:
    """The escape hatch, and the behaviour it restores.

    One line is the default rather than the only option. A caller sweeping the two triggers
    against each other -- which is what produced the finding this class was rewritten for -- has
    to be able to ask for two, and what they then get is a row whose halves fire at their own
    lines. Asserted on behaviour rather than on attributes, because a constructor that stored the
    fraction and a pass that ignored it would satisfy an attribute check.
    """
    strategy = _composed(
        tool_results=_record_phase(trigger_fraction=0.2, fallback_fraction=0.99),
        user_turns=_user_phase(trigger_fraction=0.95),
        user_trigger_fraction=0.95,
    )

    assert strategy.tool_results.trigger_fraction == 0.2
    assert strategy.user_turns.trigger_fraction == 0.95
    assert strategy.user_trigger_fraction == 0.95, "the composition was told to read the higher line"

    messages = _conversation()
    assert await strategy(messages) is True

    assert "x" * 100 not in _rendered(messages), "the low trigger fired the record phase"
    assert strategy.user_compactions == 0, "while the high one left the user phase alone"
    assert strategy.user_passes_starved == 0, "and the user line was never crossed to begin with"


async def test_a_summarizer_that_raises_leaves_the_user_band_and_the_pass_alone() -> None:
    """Degrading safely is each part's contract, and composing must not weaken it.

    The user-band strategy writes nothing until it has a summary in hand, so a summarizer that
    raises leaves a conversation byte-identical to the one it was given. What composing adds is
    a way to lose that: a pass that reported the failure as its own answer, or one whose
    re-annotation between the phases mutated something on the way through, would turn a phase
    that did nothing into a conversation that changed.
    """
    summarizer = _FailingSummarizer()
    strategy = _composed(user_turns=_user_phase(summarizer=summarizer))
    messages = _conversation()

    assert await strategy(messages) is True, "the record phase acted, and that is what True says"

    assert summarizer.calls == 1, "the user phase was reached rather than skipped"
    assert (strategy.user_summary_failures, strategy.user_compactions) == (1, 0)
    assert [text[:7] for text in _user_texts(messages)] == [f"Turn {index}:"[:7] for index in range(8)], (
        "every user turn is still being sent, and none is marked as superseded"
    )
    assert not [
        message
        for message in messages
        if message.role == "user" and message.additional_properties.get(EXCLUDED_KEY, False)
    ]


async def test_an_empty_conversation_is_not_handed_to_either_half() -> None:
    """The cheapest pass there is, and the one a composition is most likely to get wrong.

    Each part returns False on an empty list before reading anything; a composition that
    skipped that check would annotate an empty conversation twice and take a token count of
    nothing to compare against a trigger.
    """
    strategy = _composed()

    assert await strategy([]) is False
    assert (strategy.user_compactions, strategy.records_found) == (0, 0)


async def test_every_counter_of_both_halves_is_readable_off_the_composed_row() -> None:
    """A composed row that reports only half its counters is a row nobody can attribute.

    The flags column is built by reading named attributes off whatever strategy the run
    installed, so a counter that stops at a part is a counter that leaves the table -- and the
    one question a composed row raises above all others is which of its two halves did what.
    """
    strategy = _composed()
    messages = _conversation()

    assert await strategy(messages) is True

    assert strategy.records_found == strategy.tool_results.records_found == 1
    assert strategy.records_in_conversation == strategy.tool_results.records_in_conversation == 1
    assert strategy.fallbacks_used == strategy.tool_results.fallbacks_used == 0
    assert strategy.fallbacks_after_record == strategy.tool_results.fallbacks_after_record == 0
    assert strategy.groups_kept_uncovered == strategy.tool_results.groups_kept_uncovered == 0
    assert strategy.user_compactions == strategy.user_turns.user_compactions == 1
    assert strategy.user_messages_replaced == strategy.user_turns.user_messages_replaced == 6
    assert strategy.user_summary_failures == strategy.user_turns.user_summary_failures == 0
    assert strategy.user_summaries_in_conversation == strategy.user_turns.user_summaries_in_conversation == 1
    assert strategy.user_summary_tokens == strategy.user_turns.user_summary_tokens > 0
    assert strategy.user_folds == strategy.user_turns.user_folds == 0


def test_two_halves_measuring_against_two_ceilings_are_refused() -> None:
    """The order argument is about which fraction is crossed first, which needs one ceiling.

    Two ceilings make "0.6 fires before 0.8" false as often as it is true, and they make the
    starvation counter compare a token count against a line taken from the other half's
    arithmetic. Refused in the constructor rather than documented, because a run built from one
    ``StrategyOptions`` can only reach this by a caller assembling the parts by hand.
    """
    with pytest.raises(ValueError, match="one max_input_tokens"):
        ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
            tokenizer=TOKENIZER,
            tool_results=_record_phase(20_000),
            user_turns=_user_phase(24_000),
        )


def _light_turn(index: int) -> list[Message]:
    """Return a user turn and a short reply, for continuing the fixture without crossing the ceiling.

    The fixture's own turns are 4,000 characters each way, and three crossings of those would
    take the conversation past the compacting ceiling, where the record phase's fallback starts
    shedding assistant replies -- correct, but a different test. These keep the user half
    growing and the prompt under the ceiling.
    """
    return [
        Message(role="user", contents=[f"Turn {index}: " + "u" * 1_500], message_id=f"u{index}"),
        Message(role="assistant", contents=[f"Reply {index}: " + "a" * 500], message_id=f"a{index}"),
    ]


class _RatioSummarizer:
    """A summarizer keeping a stated fraction of what it reads, so a fold can be worth its break."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        body = messages[-1].text or ""
        return ChatResponse(messages=[Message(role="assistant", contents=["s" * int(len(body) * self.ratio)])])


async def test_a_fold_in_the_user_half_leaves_the_record_where_it_was() -> None:
    """A fold rewrites the prefix at the oldest summary's position, and the record sits behind it.

    The record half has one preserved message in the same conversation, and the fold is the one
    thing the user half does that reaches back past its newest boundary. So the whole of what
    composing has to guarantee is asserted here: the record is not excluded, not shortened, not
    moved relative to anything but the one inserted message, still found by ``find_record_index``
    and still preserved under the record half's own reason -- and the shared line is still the
    record half's, judged at pass entry, because the fold sits behind the same trigger check.

    The fixture is driven to a fold and asserts that it got there, since a composed row whose
    user half never folded would pass every invariant here for nothing.
    """
    strategy = _composed(
        user_turns=_user_phase(summarizer=_RatioSummarizer(0.5), summary_mode=SUMMARY_MODE_FOLD),
    )
    messages = _conversation()
    (record,) = (message for message in messages if message.message_id == "rec_res")

    assert await strategy(messages) is True
    for start in (8, 14):
        messages += [message for index in range(start, start + 6) for message in _light_turn(index)]
        assert await strategy(messages) is True
    assert strategy.user_summaries_in_conversation == 3 and strategy.user_folds == 0, "three boundaries first"
    messages += _light_turn(20)
    before = [m.message_id for m in messages if m.role != "user"]

    assert await strategy(messages) is True
    assert strategy.user_folds == 1, "the fixture has to fold or the invariants below are vacuous"

    assert strategy.user_summaries_in_conversation == 1 and strategy.user_summary_tokens > 0
    assert messages[find_record_index(messages) or -1] is record, "the record is still the anchor"
    assert record.additional_properties.get(EXCLUDED_KEY, False) is False
    assert is_preserved(record) and record.additional_properties[PRESERVE_REASON_KEY] == "tool_summary_record"
    assert "lookup_1: CODE-1." in _rendered(messages), "and unshortened"
    assert [m.message_id for m in messages if m.role != "user"] == before, (
        "every message that is not a user turn is where it was, in the order it was"
    )
    assert strategy.records_in_conversation == 1
    assert strategy.fallbacks_after_record == 0, "the prompt stayed under the ceiling, so this is about the fold alone"
    assert strategy.user_trigger_fraction == DEFAULT_TRIGGER_FRACTION, "the alignment is untouched"
    assert strategy.user_passes_starved == 0
