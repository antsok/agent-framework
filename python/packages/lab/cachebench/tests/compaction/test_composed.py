# Copyright (c) Microsoft. All rights reserved.

"""Tests for the record strategy, the user-turn strategy and the last-resort chain over one conversation.

The composition owns no selection rule of its own for either half -- that is tested beside
each part. What is tested here is what composing adds, most of which fails silently rather
than loudly.

**The user half is judged after the record half, and stays idle when the record half was
enough.** This reverses what this file used to pin. It used to assert that both halves are
judged against the size the pass *began* with, so the user half fired even when the record
phase had already taken the prompt under the shared line. The row's design now makes the user
half the second line of defence -- its edits break the cached prefix -- so the headline test
is the opposite one: on a ceiling whose shared line sits between the conversation's size and
the size the record phase leaves, the user half stays idle, and handing it the pass-entry size
is what that test is written to fail on. A second test pins the size it *is* handed.

**The starvation counter is gone, and the tests that read it now read ``USERUNDER``.** They
used to assert that a user half held under its line by the record half was a defect worth its
own number. Under this design it is the row working, and it is counted as the ordinary "under
the line" it is.

**The chain runs in order, only while over the budget, and keeps a replacement only if it is
smaller.** Each step is driven on a fixture sized so that step is the one that brings the
prompt under the budget, and the steps after it are asserted not to have run. A replacement
that comes back no smaller is refused and the old material asserted still standing; the bound
on harder rewrites is asserted at several values; and the exhausted chain is asserted to end in
the fallback, over the budget -- the ``DQ`` the design intends. None of it checks content, and
neither do the tests: the scripted summarizer answers by length, which is all the strategy
may look at.

**A merged record is a record to everything downstream, and an excluded one is not.** The
coverage check, the record count, the newest-record lookup the middleware uses, and the hold
the fallback runs behind are each asserted against a merged record.

**The single rows are unchanged.** ``tool_summary_anchored`` still falls back straight behind
its record, never merges, and reports no repeat request; ``user_summary_anchored`` still
defaults to recompacting and remembers one request.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, ChatResponse, Message
from agent_framework._compaction import (
    EXCLUDE_REASON_KEY,
    EXCLUDED_KEY,
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
    project_included_messages,
)
from agent_framework_lab_cachebench.compaction._anchored import AnchoredCompactionStrategy
from agent_framework_lab_cachebench.compaction._composed import (
    DEFAULT_HARDER_ATTEMPTS,
    DEFAULT_RECORD_MERGE_PROMPT,
    ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy,
    harder_record_prompt,
)
from agent_framework_lab_cachebench.compaction._preserve import PRESERVE_REASON_KEY, is_preserved
from agent_framework_lab_cachebench.compaction._toolsummary import (
    CONSOLIDATE_EXCLUDE_REASON,
    DEFAULT_TRIGGER_FRACTION,
    PRESERVE_REASON_UNCOVERED,
    PRESERVE_REASON_UNRECORDED,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    _droppable_groups_after,
    _hold_unrecorded,
    _preserve_records,
    _record_text,
    active_record_groups,
    build_record_messages,
    find_record_index,
    record_body,
)
from agent_framework_lab_cachebench.compaction._usersummary import (
    DEFAULT_SUMMARY_MODE,
    DEFAULT_USER_FOLD_PROMPT,
    DEFAULT_USER_SUMMARY_PROMPT,
    DEFAULT_USER_TRIGGER_FRACTION,
    FOLD_ID_PREFIX,
    SUMMARY_MODE_BOUNDARY,
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
#: above the post-record size, so the chain never runs and no assertion here is about it.
#:
#: **Recompute the sizes whenever a default moves, and check the margins rather than the
#: signs.** A fixture that slips under a trigger does not fail; it asserts against a phase that
#: returned without doing anything, and passes. The fixture-sizing test below is what makes
#: that loud, and it is the only test here that should ever need these numbers rewritten.
_COMPACTING_CEILING = 20_000

#: A ceiling whose *split* user line sits between the fixture's two sizes.
#:
#: 0.8 of 22,000 is 17,600: above the 16,613 the record phase leaves and below the 18,937 the
#: conversation starts at, so a split row's user half, judged after the record phase, stays idle
#: here. The shared line is 0.6 of it, 13,200, which both sizes clear. 0.9 of it is 19,800, so a
#: record-less pass is still waiting rather than falling back, which is what makes the
#: still-waiting test about the composition and not about the anchored strategy.
_NARROW_CEILING = 22_000

#: A ceiling whose *shared* line sits between the fixture's two sizes, which is the headline.
#:
#: 0.6 of 28,000 is 16,800: above the 16,613 the record phase leaves and below the 18,937 the
#: conversation starts at. A user half judged against what the record phase left stays idle here,
#: and one judged against the size its pass began with would fire, so this one ceiling is the
#: difference between the design and the pass-entry judging it reverses. 0.8 of
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
#: twenty-turn run and its removals then hold the prompt below both the shared line and a split
#: user line (0.8 of it, 16,000) after every pass: the run where the user half is never needed.
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


def _record_messages(values: str, *, call_id: str = "rec") -> list[Message]:
    """Return the matched recall call and result the record phase anchors on.

    A matched pair rather than a bare tool result, because the strategy refuses a record whose
    call the provider never issued -- which is the case that breaks on routes tracking tool
    calls server-side.

    Args:
        values: The record's text, as the model would have written it.

    Keyword Args:
        call_id: Distinguishes one record from another, so a test can put a second record in
            the conversation the way a re-force does.

    Returns:
        The two messages.
    """
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": call_id, "name": RECALL_TOOL_NAME, "arguments": "{}"}],
            message_id=f"{call_id}_call",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": call_id, "result": f"{RECORD_MARKER} {values}"}],
            message_id=f"{call_id}_res",
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
    harder_attempts: int = DEFAULT_HARDER_ATTEMPTS,
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
        harder_attempts=harder_attempts,
    )


def _split_composed(
    ceiling: int = _COMPACTING_CEILING,
) -> ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Return the composition with its two halves deliberately set apart, 0.6 against 0.8.

    The row this class shipped as, reachable now only by asking for it.
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


#: A ceiling whose shared line sits *below* the size the record phase leaves, and whose user
#: row's own line sits above the whole fixture.
#:
#: 0.6 of 25,000 is 15,000, under the 16,613 the record phase leaves, so the composed row's user
#: half is judged over its line after the record phase; 0.8 of it is 20,000, above the 18,937
#: the conversation starts at, so the same object run as its own row does not fire. The two
#: readings give opposite answers here, which is what the test on it needs. The ceiling is above
#: every size, so the chain never runs.
_OWN_LINE_CEILING = 25_000


async def test_a_split_row_whose_record_half_holds_the_prompt_down_reports_the_user_half_as_under() -> None:
    """The run this file used to call starvation, now read as what it is.

    Measured live on the row as it first shipped: the record phase fires at 0.6, the user phase
    at 0.8, the record phase removes the tool payload while the prompt is in the 60s and holds
    it there, and the user half never acts. This file used to count that as a defect with a
    number of its own, ``USERSTARVED``. The row now judges its user half after the record half
    on purpose -- user compaction breaks the cached prefix, and a record half that was enough is
    a record half that spared it -- so the same run is asserted to report the user half as under
    its line on every pass, and nothing more.

    Changed from ``test_two_lines_still_let_the_record_phase_hold_the_prompt_under_the_user_one``,
    which asserted the starvation count.
    """
    strategy = _split_composed(_GROWING_CEILING)

    passes = await _grow_composed(strategy, 20)
    user_line = int(_GROWING_CEILING * strategy.user_trigger_fraction)

    assert max(before for _, before in passes) <= user_line, "no pass starts above the split user line"
    assert strategy.records_found == 1, "the record phase is acting, which is what does the holding"
    assert strategy.user_compactions == 0, "and the user half never needs a turn"
    assert strategy.user_passes_below_trigger == len(passes), "which is counted as under its line, every pass"
    assert not hasattr(strategy, "user_passes_starved"), "and not as starvation, which is retired"


async def test_a_composed_row_whose_user_half_did_nothing_always_says_why() -> None:
    """Every pass in which the user half did not compact is accounted for by exactly one counter.

    A composed row whose user half is silent is now often the design working -- the record half
    was enough -- but it has to *say* that, and the only way a reader of the flags column can
    tell it from a band too small or a summarizer that failed is that each pass lands in exactly
    one of the user half's outcomes. Asserted on the aligned row over the growing run, where the
    record half keeps the prompt under the shared line throughout.

    Changed: the starvation assertions this test carried are gone with the counter.
    """
    strategy = _composed(_GROWING_CEILING)

    passes = await _grow_composed(strategy, 20)

    accounted = (
        strategy.user_compactions
        + strategy.user_passes_declined
        + strategy.user_passes_below_trigger
        + strategy.user_summary_failures
    )
    assert accounted == len(passes), "every pass lands in exactly one of the user half's four outcomes"
    assert strategy.records_found == 1
    assert strategy.user_compactions == 0, "the record half kept the prompt under the shared line"
    assert strategy.user_passes_below_trigger == len(passes), "and the flags say so"


async def test_the_reasons_a_composed_user_half_is_silent_read_differently() -> None:
    """Never over the line, held back by the band share, and a summarizer that failed.

    They ask for different responses -- nothing, lower ``min_band_share``, look at the
    summarizer -- and a row that reported one number for all of them would send a reader to the
    wrong knob. Changed: this used to be four reasons, and the fourth, starvation, is retired.
    """
    never = _composed(_IDLE_CEILING)
    declined = _composed(
        tool_results=_record_phase(_COMPACTING_CEILING),
        user_turns=_user_phase(_COMPACTING_CEILING, keep_head_user_turns=3, keep_tail_user_turns=4),
    )
    failing = _composed(user_turns=_user_phase(summarizer=_FailingSummarizer()))

    assert await never(_conversation()) is False
    assert await declined(_conversation()) is True, "the record phase acted; the user half is what did not"
    assert await failing(_conversation()) is True

    assert (never.user_passes_below_trigger, never.user_passes_declined, never.user_summary_failures) == (1, 0, 0)
    assert (declined.user_passes_below_trigger, declined.user_passes_declined) == (0, 1), (
        "over its line after the record phase, and the band between those anchors is one turn"
    )
    assert (failing.user_passes_below_trigger, failing.user_passes_declined, failing.user_summary_failures) == (
        0,
        0,
        1,
    )


async def test_the_fixture_sits_where_the_ceilings_assume_it_does() -> None:
    """A fixture that drifts across a trigger asserts against a phase that did nothing.

    Every test here rests on two sizes -- the conversation's, and the conversation's once the
    record phase has dropped what the record covers -- and on where each ceiling's lines fall
    relative to them. A drift in any of that turns an assertion about composing into an
    assertion about an early return, which passes. So the arithmetic is checked once, here,
    rather than trusted to the comments on the constants.
    """
    before = _size(_conversation())
    compacted = _conversation()
    assert await _record_phase()(compacted) is True
    after = _size(compacted)

    assert after < before, "the record phase has to remove something or half of this is vacuous"
    assert after > 0.6 * _COMPACTING_CEILING, "the shared line stays crossed on the compacting ceiling"
    assert after > 0.8 * _COMPACTING_CEILING, "and so does the user row's own line"
    assert after <= _COMPACTING_CEILING, "and the chain never runs there"
    assert after < 0.8 * _NARROW_CEILING < before, "the narrow ceiling's split user line sits between the two sizes"
    assert after > 0.6 * _NARROW_CEILING, "while its shared line sits under both"
    assert 0.9 * _NARROW_CEILING > before > 0.6 * _NARROW_CEILING, "where a record-less pass is still waiting"
    assert after <= 0.6 * _SHARED_LINE_CEILING < before, (
        "the headline ceiling's *shared* line sits between them, which is the whole of what makes "
        "that test bite: a user half judged against what the record phase left stays idle there, "
        "and one judged against the size the pass began with would fire"
    )
    assert before < 0.8 * _SHARED_LINE_CEILING, "and the user row's own line is above the fixture entirely"
    assert after > 0.6 * _OWN_LINE_CEILING and before < 0.8 * _OWN_LINE_CEILING, (
        "the own-line ceiling's shared line is under the post-record size and its user row's line over the whole"
    )
    assert before < 0.6 * _IDLE_CEILING, "and neither line is anywhere near on the idle one"


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


async def test_the_user_half_stays_idle_when_the_record_phase_alone_brings_the_prompt_under_the_line() -> None:
    """The headline, on the one ceiling where judging after the record phase and at entry disagree.

    The shared line here is 16,800. The conversation is 18,937 and the record phase leaves
    16,613, so a user half asked "is the prompt over the line?" *after* the record phase has
    acted answers no, and stays idle: tool compaction was enough, and a user pass would have
    broken the cached prefix for nothing. That is the design.

    **This test used to assert the opposite**, as
    ``test_both_halves_act_when_only_the_size_the_pass_began_with_clears_the_shared_line``: that
    the user half fires here because it is judged against the size the pass began with. Written
    now to fail on that: hand the user phase the entry size and it compacts a band on a prompt
    already under the line.
    """
    strategy = _composed(_SHARED_LINE_CEILING)
    messages = _conversation()
    shared_line = int(_SHARED_LINE_CEILING * strategy.user_trigger_fraction)
    entry = _size(_conversation())
    post_record = _conversation()
    assert await _record_phase(_SHARED_LINE_CEILING)(post_record) is True
    assert _size(post_record) <= shared_line < entry, "the fixture has to straddle the line or this proves nothing"

    assert await strategy(messages) is True

    assert strategy.records_found == 1, "the record phase acted"
    assert "x" * 100 not in _rendered(messages), "and dropped the tool payload"
    assert strategy.user_compactions == 0, "and the user half, judged on what that left, stayed idle"
    assert "Turn 4:" in _rendered(messages), "so the band is untouched"
    assert (strategy.user_passes_below_trigger, strategy.user_passes_declined) == (1, 0)
    assert _size(messages) == _size(post_record), "the pass removed exactly what the record phase alone does"


async def test_the_user_half_is_judged_against_the_size_the_record_phase_left() -> None:
    """The number the user half is handed, asserted directly and on every pass of a run.

    The outcome tests above and below read what the user half did; this reads what it was
    asked. A spy records the size each pass hands it and compares it with the size of the
    conversation at that moment, which is after the record phase: equal on every pass, and on
    the passes where the record phase removed something, smaller than the size the pass began
    with. Over the growing run and on the one-pass fixture, so both a pass that compacts and
    passes that do not are covered.
    """
    handed: list[tuple[int, int]] = []

    class _UserSpy(UserTurnAnchoredSummarizationCompactionStrategy):
        async def compact_against(self, messages: list[Message], *, prompt_tokens: int, trigger_tokens: int) -> bool:
            handed.append((prompt_tokens, included_token_count(messages)))
            return await super().compact_against(messages, prompt_tokens=prompt_tokens, trigger_tokens=trigger_tokens)

    growing = _composed(
        _GROWING_CEILING,
        user_turns=_UserSpy(max_input_tokens=_GROWING_CEILING, tokenizer=TOKENIZER, client=_Summarizer()),
    )
    passes = await _grow_composed(growing, 20)
    one = _composed(
        user_turns=_UserSpy(max_input_tokens=_COMPACTING_CEILING, tokenizer=TOKENIZER, client=_Summarizer())
    )
    assert await one(_conversation()) is True

    assert len(handed) == len(passes) + 1
    assert all(given == live for given, live in handed), "the live size after the record phase, every time"
    assert any(given < before for (given, _), (_, before) in zip(handed, passes, strict=False)), (
        "and on some pass that is smaller than the size the pass began with, or the equality is vacuous"
    )
    assert one.user_compactions == 1


async def test_the_user_half_acts_when_the_record_phase_leaves_the_prompt_over_the_line() -> None:
    """The second line of defence, engaging when the first was not enough.

    On the compacting ceiling the record phase leaves 16,613 against a shared line of 12,000, so
    the user half is over its line after the record phase and compacts its band on the same pass.
    """
    strategy = _composed(_COMPACTING_CEILING)
    messages = _conversation()

    assert await strategy(messages) is True

    assert strategy.records_found == 1
    assert strategy.user_compactions == 1
    assert "Turn 4:" not in _rendered(messages)
    assert strategy.last_resort_fallbacks == 0, "under the budget afterwards, so the chain never ran"


async def test_the_composed_row_leaves_less_behind_than_either_half_alone_only_when_it_needs_to() -> None:
    """The claim the row makes, in tokens, at both ceilings.

    At the compacting ceiling all three rows act and the composed row beats both. At the
    headline ceiling the record half alone brings the prompt under the shared line, so the
    composed row leaves exactly what the record row leaves: it declined to spend a user pass it
    did not need. Changed: this used to assert the composed row beat both rows at the headline
    ceiling too, which was pass-entry judging buying a smaller prompt with a broken prefix.
    """
    composed_low, tool_low, user_low = await _sizes_at(_COMPACTING_CEILING)
    composed_high, tool_high, user_high = await _sizes_at(_SHARED_LINE_CEILING)
    entry = _size(_conversation())

    assert tool_low < entry and user_low < entry, "both single rows act at the compacting ceiling"
    assert composed_low < min(tool_low, user_low), "so the composed row there is beating two working rows"

    assert tool_high < entry, "the record row acts at the headline ceiling"
    assert user_high == entry, "and the user row does not: its own 0.8 line is above the whole fixture"
    assert composed_high == tool_high, "and the composed row stops where the record row did, because that was enough"


async def test_a_half_run_as_its_own_row_reads_its_own_trigger_before_and_after_composing() -> None:
    """The alignment is a reading the composition takes, not a setting it writes.

    ``tool_summary_anchored`` and ``user_summary_anchored`` are rows in the same table as the
    composed one, and every archived number for them was produced by a strategy reading its own
    ``trigger_fraction`` off the conversation in front of it. So the same instance is run as its
    own row, then as a phase, then as its own row again, on a ceiling where the two readings give
    opposite answers. Changed: the ceiling moved from the headline one, where the composed row's
    user half now stays idle as well and the test would no longer distinguish anything.
    """
    user_phase = _user_phase(_OWN_LINE_CEILING)
    record_phase = _record_phase(_OWN_LINE_CEILING)
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
    assert user_phase.summary_mode == DEFAULT_SUMMARY_MODE and user_phase.remembered_requests == 1

    standalone_record = _conversation()
    assert await record_phase(standalone_record) is True, "and the row the line was borrowed from is itself"
    assert record_phase.trigger_fraction == DEFAULT_TRIGGER_FRACTION


def test_the_composed_row_takes_its_one_line_from_the_record_half() -> None:
    """Which of the two fractions is shared, and what a sweep of it moves.

    Aligning to the record half rather than to the user half is the direction that keeps the
    record askable: the model writes it, it degrades with the bulk it is given, and the
    middleware that asks reads the same prompt -- so the alignment can only go down.
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
    """The bounds the user half sets on its own trigger, kept where the line is overridden."""
    with pytest.raises(ValueError, match="user_trigger_fraction"):
        _composed(user_trigger_fraction=0.0)
    with pytest.raises(ValueError, match="user_trigger_fraction"):
        _composed(user_trigger_fraction=1.5)


def test_a_negative_number_of_harder_attempts_is_refused() -> None:
    """Zero switches step c off; below zero is not a number of attempts."""
    assert _composed(harder_attempts=0).harder_attempts == 0
    assert _composed().harder_attempts == DEFAULT_HARDER_ATTEMPTS
    with pytest.raises(ValueError, match="harder_attempts"):
        _composed(harder_attempts=-1)


def test_the_composed_row_asks_its_middleware_to_repeat_and_the_record_row_does_not() -> None:
    """Part one of the design: every new batch of tool work is recorded, on this row only.

    The setting belongs to the recall middleware, which the run builds; the composed object
    reports ``repeat_records`` and ``run_live`` reads it -- the wiring is asserted in
    ``test_live``. What is asserted here is that the request is the composed row's and not the
    record strategy's: a record strategy reporting it would switch repeats on for the standalone
    ``tool_summary_anchored`` row, whose default is off.
    """
    strategy = _composed()

    assert strategy.repeat_records is True
    assert getattr(strategy.tool_results, "repeat_records", None) is None, "the standalone record row asks nothing"
    assert getattr(strategy.user_turns, "repeat_records", None) is None


async def test_the_two_halves_compact_two_halves_of_one_conversation() -> None:
    """Each part alone moves only what it owns, and the composed pass does both where both are needed."""
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
    assert _size(both) < min(_size(tool_only), _size(user_only))


async def test_the_record_phase_runs_before_the_user_phase_and_without_its_fallback() -> None:
    """The order, asserted where it is decided rather than inferred from an outcome.

    The record phase first, because a user phase that ran first would hand the middleware a
    prompt already below the line that asks for a record at all. And it is asked *not* to run
    its fallback: on this row the fallback is the end of the chain. Changed: the spy now also
    records the ``fallback_after_record`` it was handed.
    """
    order: list[str] = []
    fallback_flags: list[bool] = []

    class _RecordSpy(ToolResultAnchoredSummarizationCompactionStrategy):
        async def compact_against(
            self,
            messages: list[Message],
            *,
            prompt_tokens: int,
            trigger_tokens: int,
            fallback_after_record: bool = True,
        ) -> bool:
            order.append("record")
            fallback_flags.append(fallback_after_record)
            return await super().compact_against(
                messages,
                prompt_tokens=prompt_tokens,
                trigger_tokens=trigger_tokens,
                fallback_after_record=fallback_after_record,
            )

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
    assert fallback_flags == [False], "the composed row keeps the fallback for the end of its chain"
    assert strategy.strategies == (strategy.tool_results, strategy.user_turns)


async def test_a_split_rows_user_half_is_judged_after_the_record_phase_too() -> None:
    """The ceiling the old order argument was made on, read under the new judging.

    0.8 of this ceiling, 17,600, sits between the conversation's 18,937 and the 16,613 the record
    phase leaves; 0.6 of it, 13,200, sits under both. A split row's user half, judged after the
    record phase against 17,600, stays idle; the aligned row's, against 13,200, compacts. Changed
    from ``test_even_two_lines_do_not_starve_a_half_within_one_pass``, which asserted that both
    rows' user halves fired here because both were judged against the entry size.
    """
    split = _split_composed(_NARROW_CEILING)
    split_messages = _conversation()
    aligned = _composed(_NARROW_CEILING)
    aligned_messages = _conversation()

    assert await split(split_messages) is True
    assert await aligned(aligned_messages) is True

    assert "x" * 100 not in _rendered(split_messages), "the record phase acted on both"
    assert "x" * 100 not in _rendered(aligned_messages)
    assert (split.user_compactions, split.user_passes_below_trigger) == (0, 1), "under the split line afterwards"
    assert aligned.user_compactions == 1, "over the shared line afterwards"
    assert "Turn 4:" in _rendered(split_messages)
    assert "Turn 4:" not in _rendered(aligned_messages)


async def test_an_idle_pass_touches_nothing_and_asks_nothing() -> None:
    """On a ceiling nothing crosses, neither half acts and the chain does not run.

    Changed from ``test_a_user_half_that_would_not_have_fired_anyway_is_not_counted_as_starved``:
    the starvation half of it is retired, and what is left worth asserting is that an idle pass
    spends no summarizer call.
    """
    summarizer = _Summarizer()
    strategy = _composed(_IDLE_CEILING, user_turns=_user_phase(_IDLE_CEILING, summarizer=summarizer))
    messages = _conversation()

    assert await strategy(messages) is False
    assert (strategy.user_compactions, strategy.last_resort_fallbacks, summarizer.calls) == (0, 0, 0)


async def test_the_user_half_still_fires_while_the_record_half_is_waiting_for_a_record() -> None:
    """A phase that waits must not make the phase behind it wait.

    Before a record exists the record phase removes nothing, so the size it leaves is the size
    the pass began with, and the user half is judged against that.
    """
    strategy = _composed(_NARROW_CEILING)
    messages = _conversation(record=None)

    assert await strategy(messages) is True

    assert (strategy.records_found, strategy.fallbacks_used) == (0, 0), "waiting, not fallen back"
    assert "x" * 100 in _rendered(messages), "so the tool payload is untouched"
    assert strategy.user_compactions == 1


async def test_configuration_reaches_each_half_without_reaching_the_other() -> None:
    """Two rows' worth of knobs on one object, and the trigger is the only one they share."""
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
    assert (lenient.user_messages_replaced, default.user_messages_replaced) == (3, 6)


async def test_a_caller_can_set_the_two_halves_apart_again_and_then_they_fire_apart() -> None:
    """The escape hatch, and the behaviour it restores."""
    strategy = _composed(
        tool_results=_record_phase(trigger_fraction=0.2, fallback_fraction=0.99),
        user_turns=_user_phase(trigger_fraction=0.95),
        user_trigger_fraction=0.95,
    )

    assert strategy.user_trigger_fraction == 0.95, "the composition was told to read the higher line"

    messages = _conversation()
    assert await strategy(messages) is True

    assert "x" * 100 not in _rendered(messages), "the low trigger fired the record phase"
    assert strategy.user_compactions == 0, "while the high one left the user phase alone"


async def test_a_summarizer_that_raises_leaves_the_user_band_and_the_pass_alone() -> None:
    """Degrading safely is each part's contract, and composing must not weaken it."""
    summarizer = _FailingSummarizer()
    strategy = _composed(user_turns=_user_phase(summarizer=summarizer))
    messages = _conversation()

    assert await strategy(messages) is True, "the record phase acted, and that is what True says"

    assert summarizer.calls == 1, "the user phase was reached rather than skipped"
    assert (strategy.user_summary_failures, strategy.user_compactions) == (1, 0)
    assert [text[:7] for text in _user_texts(messages)] == [f"Turn {index}:"[:7] for index in range(8)]
    assert not [
        message
        for message in messages
        if message.role == "user" and message.additional_properties.get(EXCLUDED_KEY, False)
    ]


async def test_an_empty_conversation_is_not_handed_to_either_half() -> None:
    """The cheapest pass there is, and the one a composition is most likely to get wrong."""
    strategy = _composed()

    assert await strategy([]) is False
    assert (strategy.user_compactions, strategy.records_found) == (0, 0)


async def test_every_counter_of_both_halves_is_readable_off_the_composed_row() -> None:
    """A composed row that reports only half its counters is a row nobody can attribute."""
    strategy = _composed()
    messages = _conversation()

    assert await strategy(messages) is True

    assert strategy.records_found == strategy.tool_results.records_found == 1
    assert strategy.records_in_conversation == strategy.tool_results.records_in_conversation == 1
    assert strategy.fallbacks_used == strategy.tool_results.fallbacks_used == 0
    assert strategy.fallbacks_after_record == strategy.tool_results.fallbacks_after_record == 0
    assert strategy.groups_kept_uncovered == strategy.tool_results.groups_kept_uncovered == 0
    assert strategy.groups_preserved_uncovered == strategy.tool_results.groups_preserved_uncovered == 0
    assert strategy.user_compactions == strategy.user_turns.user_compactions == 1
    assert strategy.user_messages_replaced == strategy.user_turns.user_messages_replaced == 6
    assert strategy.user_summary_failures == strategy.user_turns.user_summary_failures == 0
    assert strategy.user_summaries_in_conversation == strategy.user_turns.user_summaries_in_conversation == 1
    assert strategy.user_summary_tokens == strategy.user_turns.user_summary_tokens > 0
    assert strategy.user_folds == strategy.user_turns.user_folds == 0
    assert strategy.user_summaries_replayed == strategy.user_turns.user_summaries_replayed == 0
    assert strategy.tokens_removed_by_record_phase > 0


async def test_the_composed_row_holds_what_its_record_missed_asks_again_and_then_preserves() -> None:
    """Both layers run inside the composed row, through the same seam its own row uses."""
    strategy = _composed(tool_results=_record_phase(trigger_fraction=0.1))
    messages = _conversation(record=_covering_record(2))

    assert await strategy(messages) is True

    assert strategy.groups_kept_uncovered == 2
    held = {
        message.message_id
        for message in messages
        if is_preserved(message) and message.additional_properties.get(PRESERVE_REASON_KEY) == PRESERVE_REASON_UNCOVERED
    }
    assert held == {"c3", "r3", "c4", "r4"}, "the two groups the record never quoted, and only those"
    assert strategy.tool_results.take_reforce() is True, "and the record phase asked for another record"
    assert strategy.user_compactions == 1, "while the user half compacted its own half of the same pass"
    assert "code_1=CODE-3 " in _rendered(messages) and "code_1=CODE-4 " in _rendered(messages)

    await strategy(messages)
    messages += _record_messages("nothing to add.", call_id="rec2")
    await strategy(messages)

    assert strategy.groups_preserved_uncovered == 2, "the re-forced record covered nothing, so layer two settled them"
    assert strategy.tool_results.take_reforce() is False
    assert find_record_index(messages) == len(messages) - 1, "the newest record is reachable where it was put"
    preserved = {message.message_id for message in messages if is_preserved(message)}
    assert preserved >= {"rec_call", "rec_res", "rec2_call", "rec2_res"}, "both records stay preserved"
    assert "code_1=CODE-3 " in _rendered(messages) and "code_1=CODE-4 " in _rendered(messages), "still whole"


async def test_the_composed_row_holds_the_tool_groups_behind_its_record_when_its_fallback_runs() -> None:
    """The post-record hold runs inside the composed row too, now at the end of the chain.

    Run 51's shape: tool groups sitting after the record, covered by no record, and a ceiling
    the record phase cannot reach, so the chain gets to its fallback. Those groups are held
    under the record half's third reason and come through whole.
    """
    strategy = _composed(500, tool_results=_record_phase(500, trigger_fraction=0.1, fallback_fraction=0.9))
    messages = _conversation(record=_covering_record(4))
    for index in range(5, 8):
        messages += _tool_group(index)

    assert await strategy(messages) is True

    held = {
        message.message_id
        for message in messages
        if is_preserved(message)
        and message.additional_properties.get(PRESERVE_REASON_KEY) == PRESERVE_REASON_UNRECORDED
    }
    assert held == {"c5", "r5", "c6", "r6", "c7", "r7"}, "every tool group behind the record, and only those"
    for index in range(5, 8):
        assert f"code_1=CODE-{index} " + "x" * _PAYLOAD_CHARS in _rendered(messages), f"lookup_{index} is whole"
    assert strategy.fallbacks_held_after_record == strategy.tool_results.fallbacks_held_after_record == 1
    assert strategy.last_resort_fallbacks == 1, "reached at the end of the chain"
    assert strategy.user_compactions == 1, "while the user half compacted its own half of the same pass"


def test_two_halves_measuring_against_two_ceilings_are_refused() -> None:
    """One shared line, and one budget for the chain, need one ceiling."""
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

    A hand-assembled composition in the fold mode, which the builder no longer produces -- the
    row runs in ``boundary`` and folds only through its chain -- but which a caller may still
    assemble, and whose fold must not disturb the record half's preserved message.
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
    assert [m.message_id for m in messages if m.role != "user"] == before
    assert strategy.records_in_conversation == 1
    assert strategy.last_resort_fallbacks == 0, "the prompt stayed under the ceiling, so this is about the fold alone"
    assert strategy.user_trigger_fraction == DEFAULT_TRIGGER_FRACTION, "the alignment is untouched"


# -- The last-resort chain ---------------------------------------------------------------------

#: Characters of padding in each record and each user summary of the chain fixture.
#:
#: The chain fixture is built so that, once the record phase has dropped the tool payload,
#: almost all of what is left is two records and two standing user summaries -- the four things
#: the chain can rewrite -- and the budget each test sets decides which step is the one that
#: brings the prompt under it.
_CHAIN_PADDING_CHARS = 4_000

#: The ceiling the chain fixture's triggers are taken from when a test does not set one.
_CHAIN_CEILING = 500

#: A merged record that is plainly smaller than the two it replaces, and carries their values.
_SHORT_RECORD = "lookup_1: CODE-1. lookup_2: CODE-2."


def _longer(body: str) -> str:
    """Answer with more than was sent, which the chain must refuse as no smaller."""
    return body + " " + body


def _short_record(body: str) -> str:
    """Answer with :data:`_SHORT_RECORD`, whatever was sent."""
    return _SHORT_RECORD


def _short_summary(body: str) -> str:
    """Answer a fold with one short line, whatever was sent."""
    return "The user set a task and then asked two follow-ups."


class _RoutingSummarizer:
    """A summarizer that answers each kind of request by its own rule, and logs the kinds in order.

    The kind is read off the system prompt, which is the only thing that tells the four requests
    the composed row can make apart: a band summary, a fold, a record merge and a harder
    rewrite. The rules answer by length and nothing else, because length is all the chain may
    judge a replacement by.
    """

    def __init__(
        self,
        *,
        merge: Callable[[str], str] = _longer,
        fold: Callable[[str], str] = _longer,
        harder: Callable[[int, str], str] | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.merge = merge
        self.fold = fold
        self.harder = harder or (lambda attempt, body: _longer(body))
        self.log = log if log is not None else []
        self.prompts: list[str] = []

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        prompt = messages[0].text or ""
        body = messages[-1].text or ""
        self.prompts.append(prompt)
        if prompt == DEFAULT_USER_SUMMARY_PROMPT:
            self.log.append("band")
            text = "The user asked for the earlier things, in order."
        elif prompt == DEFAULT_USER_FOLD_PROMPT:
            self.log.append("fold")
            text = self.fold(body)
        elif prompt == DEFAULT_RECORD_MERGE_PROMPT:
            self.log.append("merge")
            text = self.merge(body)
        else:
            attempt = next(n for n in range(1, 10) if prompt == harder_record_prompt(n))
            self.log.append(f"harder{attempt}")
            text = self.harder(attempt, body)
        return ChatResponse(messages=[Message(role="assistant", contents=[text])])


class _LoggedFallback:
    """The default anchored fallback, logging each time it is run into a shared list."""

    def __init__(self, log: list[str], ceiling: int = _CHAIN_CEILING) -> None:
        self.log = log
        self.inner = AnchoredCompactionStrategy(max_input_tokens=ceiling, tokenizer=TOKENIZER)

    async def __call__(self, messages: list[Message]) -> bool:
        self.log.append("fallback")
        return await self.inner(messages)


def _standing_summary(serial: int) -> Message:
    """Return a user summary as the boundary mode leaves one standing."""
    return Message(
        role="user",
        contents=[f"{USER_SUMMARY_MARKER}\n" + "s" * _CHAIN_PADDING_CHARS],
        message_id=f"user_summary_{serial}",
    )


def _padded_record(serial: int, indices: list[int]) -> list[Message]:
    """Return a record covering ``indices`` and padded with the chain fixture's filler."""
    values = " ".join(f"lookup_{index}: CODE-{index}." for index in indices)
    return _record_messages(f"{values} " + "r" * _CHAIN_PADDING_CHARS, call_id=f"rec{serial}")


def _chain_conversation(*, uncovered: bool = False, trailing: bool = False) -> list[Message]:
    """Return a conversation holding two records and two standing user summaries.

    Its user band is empty in the boundary mode -- the only turn newer than the newest summary
    is the tail -- so the user half declines, and every rewrite a test sees is the chain's.

    Args:
        uncovered: Put a tool group no record quotes in front of the second record, so the
            coverage check holds it.
        trailing: Put a tool group after the second record, which no record covers.

    Returns:
        The messages.
    """
    messages = [Message(role="system", contents=["You are an assistant."], message_id="sys")]
    messages += [
        Message(role="user", contents=["Turn 0: the task."], message_id="u0"),
        Message(role="assistant", contents=["Reply 0: " + "a" * 400], message_id="a0"),
        _standing_summary(0),
        Message(role="assistant", contents=["Reply 1: " + "a" * 400], message_id="a1"),
        *_tool_group(1),
        *_padded_record(1, [1]),
        Message(role="user", contents=["Turn 2: next."], message_id="u2"),
        Message(role="assistant", contents=["Reply 2: " + "a" * 400], message_id="a2"),
        _standing_summary(1),
        Message(role="assistant", contents=["Reply 3: " + "a" * 400], message_id="a3"),
        *_tool_group(2),
    ]
    if uncovered:
        messages += _tool_group(3)
    messages += _padded_record(2, [2])
    if trailing:
        messages += _tool_group(5)
    messages += [
        Message(role="user", contents=["Turn 4: last."], message_id="u4"),
        Message(role="assistant", contents=["Reply 4: " + "a" * 400], message_id="a4"),
    ]
    return messages


def _chain_composed(
    summarizer: _RoutingSummarizer, *, ceiling: int = _CHAIN_CEILING, **kwargs: Any
) -> ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Return the composed row over the chain fixture, its fallback logging into the summarizer's log.

    The record half's trigger is low enough to fire on any budget a test sets, and its give-up
    line is above everything, so no test here is about waiting for a record. The user half runs
    in the boundary mode, as the builder runs it.
    """
    return ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER,
        tool_results=ToolResultAnchoredSummarizationCompactionStrategy(
            max_input_tokens=ceiling,
            tokenizer=TOKENIZER,
            trigger_fraction=0.01,
            fallback_fraction=0.99,
            fallback=_LoggedFallback(summarizer.log, ceiling),
        ),
        user_turns=UserTurnAnchoredSummarizationCompactionStrategy(
            max_input_tokens=ceiling,
            tokenizer=TOKENIZER,
            client=summarizer,
            trigger_fraction=0.01,
            summary_mode=SUMMARY_MODE_BOUNDARY,
        ),
        **kwargs,
    )


async def _post_record_size(**fixture: bool) -> int:
    """Return the chain fixture's size once the record phase has dropped what its records cover."""
    messages = _chain_conversation(**fixture)
    record_phase = _record_phase(_CHAIN_CEILING, trigger_fraction=0.01, fallback_fraction=0.99)
    entry = _size(messages)
    await record_phase.compact_against(messages, prompt_tokens=entry, trigger_tokens=1, fallback_after_record=False)
    return _size(messages)


def _active_record_ids(messages: list[Message]) -> list[str]:
    """Return the message ids of every record still being sent, oldest first."""
    return [str(messages[group["end_index"]].message_id) for group in active_record_groups(messages)]


def _standing_summary_ids(messages: list[Message]) -> list[str]:
    """Return the ids of the user summaries still being sent."""
    return [
        str(message.message_id)
        for message in project_included_messages(messages)
        if message.role == "user" and USER_SUMMARY_MARKER in (message.text or "")
    ]


async def test_the_chain_does_not_run_while_the_prompt_fits() -> None:
    """Every step of the chain is worse than nothing when nothing is needed, so none runs.

    A budget above the post-record size: the record phase acts, the user half declines on its
    empty band, and no merge, fold, rewrite or fallback is asked for.
    """
    summarizer = _RoutingSummarizer(merge=_short_record, fold=_short_summary)
    budget = await _post_record_size() + 100
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation()

    assert await strategy(messages) is True, "the record phase dropped the covered tool groups"

    assert summarizer.log == []
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"]
    assert (strategy.records_merged, strategy.user_summaries_merged, strategy.record_rewrites) == (0, 0, 0)
    assert strategy.last_resort_fallbacks == 0


async def test_the_chain_merges_the_records_first_and_stops_once_the_prompt_fits() -> None:
    """Step a, and the steps after it skipped because it was enough.

    The budget sits between the post-record size and the size a merge leaves, so the merge is
    the step that brings the prompt under it, and the log must end there: no fold, no rewrite, no
    fallback.
    """
    summarizer = _RoutingSummarizer(merge=_short_record, fold=_short_summary)
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation()

    assert await strategy(messages) is True

    assert summarizer.log == ["merge"], "merged first, and nothing after it once the prompt fit"
    assert _size(messages) <= budget
    assert (strategy.records_merged, strategy.record_merges_rejected) == (1, 0)
    assert _active_record_ids(messages) == ["compaction_record_0_result"], "one record, the merged one"
    assert _standing_summary_ids(messages) == ["user_summary_0", "user_summary_1"], "the summaries untouched"
    assert strategy.last_resort_fallbacks == 0


async def test_a_merge_that_is_no_smaller_is_refused_and_the_old_records_stay() -> None:
    """The acceptance rule on step a: no smaller, and the records it would have replaced stand.

    Nothing is mutated on a refusal -- the records keep their preservation and are not excluded
    -- and the chain moves on to the next step rather than stopping.
    """
    summarizer = _RoutingSummarizer(merge=_longer, fold=_short_summary)
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation()

    await strategy(messages)

    assert summarizer.log[:2] == ["merge", "fold"], "refused, and the chain moved on to the fold"
    assert (strategy.records_merged, strategy.record_merges_rejected) == (0, 1)
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"], "both records still being sent"
    for message_id in ("rec1_call", "rec1_res", "rec2_call", "rec2_res"):
        (message,) = (m for m in messages if m.message_id == message_id)
        assert is_preserved(message) and not message.additional_properties.get(EXCLUDED_KEY, False)
    assert not [m for m in messages if str(m.message_id).startswith("compaction_record_")], "nothing inserted"


async def test_a_replacement_the_same_size_as_what_it_replaces_is_refused() -> None:
    """ "Smaller" means smaller: an equal-sized answer is refused, not kept.

    The record here is one the chain itself wrote, so a rewrite carrying the same text builds a
    replacement of exactly the same shape and length -- the one case that tells ``>=`` from
    ``>`` in the rule.
    """
    summarizer = _RoutingSummarizer()
    text = "lookup_1: CODE-1. " + "r" * _CHAIN_PADDING_CHARS
    summarizer.harder = lambda attempt, body: text
    messages = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        Message(role="user", contents=["Turn 0: the task."], message_id="u0"),
        Message(role="assistant", contents=["Reply 0."], message_id="a0"),
        *_tool_group(1),
        *build_record_messages(text, call_id="compaction_record_7"),
        Message(role="user", contents=["Turn 1: last."], message_id="u1"),
    ]
    strategy = _chain_composed(summarizer, harder_attempts=1)

    await strategy(messages)

    assert "harder1" in summarizer.log
    assert (strategy.record_rewrites, strategy.record_rewrites_rejected) == (1, 1)
    assert _active_record_ids(messages) == ["compaction_record_7_result"], "the record it would have replaced stands"


async def test_the_user_summaries_are_merged_second_through_the_user_halfs_own_fold() -> None:
    """Step b, reached because step a was refused, and enough on its own.

    The fold is the user half's own machinery: it counts as a fold there, it is numbered under
    the fold prefix, and the folded summaries are excluded behind it.
    """
    summarizer = _RoutingSummarizer(merge=_longer, fold=_short_summary)
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation()

    assert await strategy(messages) is True

    assert summarizer.log == ["merge", "fold"], "the fold was enough, so no rewrite and no fallback"
    assert _size(messages) <= budget
    assert (strategy.user_summaries_merged, strategy.user_merges_rejected) == (1, 0)
    assert strategy.user_folds == 1, "counted by the user half as the fold it is"
    (folded,) = _standing_summary_ids(messages)
    assert folded.startswith(FOLD_ID_PREFIX)
    assert strategy.last_resort_fallbacks == 0


async def test_a_fold_that_is_no_smaller_is_refused_and_the_summaries_stay() -> None:
    """The acceptance rule on step b, through the user half's seam."""
    summarizer = _RoutingSummarizer(merge=_longer, fold=_longer)
    strategy = _chain_composed(summarizer, harder_attempts=0)
    messages = _chain_conversation()

    await strategy(messages)

    assert (strategy.user_summaries_merged, strategy.user_merges_rejected) == (0, 1)
    assert strategy.user_folds == 0, "a refused fold is not a fold"
    assert _standing_summary_ids(messages) == ["user_summary_0", "user_summary_1"]


async def test_the_record_is_rewritten_harder_third_and_each_attempt_asks_for_more() -> None:
    """Step c, reached because a and b were refused: attempts escalate, and a kept one counts.

    The first rewrite comes back longer and is refused; the second comes back short and is kept,
    which brings the prompt under the budget, so no fallback follows. Both records are handed
    over together, because the merge was refused, and the kept rewrite stands for both.
    """
    summarizer = _RoutingSummarizer(
        merge=_longer, fold=_longer, harder=lambda attempt, body: _longer(body) if attempt == 1 else _SHORT_RECORD
    )
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation()

    assert await strategy(messages) is True

    assert summarizer.log == ["merge", "fold", "harder1", "harder2"]
    assert (strategy.record_rewrites, strategy.record_rewrites_rejected) == (2, 1)
    assert _active_record_ids(messages) == ["compaction_record_2_result"], (
        "one record, the kept rewrite -- the third id minted, since the two refused answers had one each"
    )
    assert _size(messages) <= budget
    assert strategy.last_resort_fallbacks == 0
    first, second = harder_record_prompt(1), harder_record_prompt(2)
    assert "60%" in first and "36%" in second, "each attempt asks for more compression than the last"
    for prompt in (first, second, DEFAULT_RECORD_MERGE_PROMPT):
        assert "verbatim" in prompt and "CODE" not in prompt, "a generic instruction, naming nothing planted"


@pytest.mark.parametrize("attempts", [0, 1, 3])
async def test_the_number_of_harder_attempts_bounds_the_rewrites(attempts: int) -> None:
    """``harder_attempts`` is how many rewrites one pass may ask for, and not one more.

    Every answer is refused, so the prompt stays over the budget throughout and nothing but the
    bound can stop the attempts.
    """
    summarizer = _RoutingSummarizer()
    strategy = _chain_composed(summarizer, harder_attempts=attempts)

    await strategy(_chain_conversation())

    assert [kind for kind in summarizer.log if kind.startswith("harder")] == [
        f"harder{attempt}" for attempt in range(1, attempts + 1)
    ]
    assert strategy.record_rewrites == strategy.record_rewrites_rejected == attempts


async def test_an_exhausted_chain_ends_in_the_fallback_over_the_budget() -> None:
    """Steps a to d in order, and then nothing: the prompt goes out over the limit, which is ``DQ``.

    Every replacement is refused, so each step runs and fails; the fallback runs last, holds the
    tool group no record covers, and sheds narration only; and what is left is still over the
    budget, because nothing more is tried.
    """
    summarizer = _RoutingSummarizer()
    strategy = _chain_composed(summarizer)
    messages = _chain_conversation(trailing=True)

    await strategy(messages)

    assert summarizer.log == ["merge", "fold", "harder1", "harder2", "fallback"], "in order, the fallback last"
    assert strategy.last_resort_fallbacks == 1
    assert strategy.fallbacks_held_after_record == 1, "with the tool group no record covers held"
    assert "code_1=CODE-5 " + "x" * _PAYLOAD_CHARS in _rendered(messages), "which comes through whole"
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"], "and the records untouched"
    assert _size(messages) > _CHAIN_CEILING, "still over the budget: the intended loud failure"
    assert (
        strategy.record_merges_rejected,
        strategy.user_merges_rejected,
        strategy.record_rewrites_rejected,
    ) == (1, 1, DEFAULT_HARDER_ATTEMPTS)


async def test_the_fallback_runs_only_at_the_end_on_the_composed_row_and_straight_behind_the_record_alone() -> None:
    """Where the fallback runs is the one place the two rows differ, and each is asserted.

    On the composed row it runs after the user phase and after the chain's other steps. On the
    standalone record row it runs inside the record pass, straight behind the record, exactly as
    it did -- and that row never merges.
    """
    log: list[str] = []

    class _UserSpy(UserTurnAnchoredSummarizationCompactionStrategy):
        async def compact_against(self, messages: list[Message], *, prompt_tokens: int, trigger_tokens: int) -> bool:
            log.append("user")
            return await super().compact_against(messages, prompt_tokens=prompt_tokens, trigger_tokens=trigger_tokens)

    summarizer = _RoutingSummarizer(log=log)
    composed = ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER,
        tool_results=_record_phase(_CHAIN_CEILING, trigger_fraction=0.01, fallback=_LoggedFallback(log)),
        user_turns=_UserSpy(
            max_input_tokens=_CHAIN_CEILING,
            tokenizer=TOKENIZER,
            client=summarizer,
            trigger_fraction=0.01,
            summary_mode=SUMMARY_MODE_BOUNDARY,
        ),
    )
    await composed(_chain_conversation())
    assert log == ["user", "merge", "fold", "harder1", "harder2", "fallback"]

    alone_log: list[str] = []
    alone = _record_phase(_CHAIN_CEILING, trigger_fraction=0.01, fallback=_LoggedFallback(alone_log))
    messages = _chain_conversation()
    await alone(messages)
    assert alone_log == ["fallback"], "the standalone row falls back inside its own pass"
    assert alone.fallbacks_after_record == 1
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"], "and never merges its records"


async def test_a_merged_record_is_a_record_to_everything_that_reads_records() -> None:
    """The merged record is found, preserved, counted, read for coverage and skipped by the holds.

    One merge, then the readers one by one: the newest-record lookup the strategy and the
    middleware share, the preservation and the count, the coverage check -- which releases and
    drops a group the old records never quoted once the merged record quotes it -- the pending
    work the middleware counts from it, the hold the fallback runs behind, and the body a later
    merge would be handed. And the replaced records are records to none of them.
    """
    merged_text = "lookup_1: CODE-1. lookup_2: CODE-2. lookup_3: CODE-3."
    summarizer = _RoutingSummarizer(merge=lambda body: merged_text)
    budget = await _post_record_size(uncovered=True) - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    messages = _chain_conversation(uncovered=True)

    await strategy(messages)

    assert strategy.records_merged == 1
    index = find_record_index(messages)
    assert index is not None and messages[index].message_id == "compaction_record_0_result"
    merged_call = messages[index - 1]
    assert [content.name for content in merged_call.contents] == [RECALL_TOOL_NAME]
    assert RECORD_MARKER in _record_text(messages[index])
    (group,) = active_record_groups(messages)
    assert record_body(messages, group) == merged_text, "the body a later merge would be handed"
    assert is_preserved(messages[index]) and messages[index].additional_properties[PRESERVE_REASON_KEY] == (
        "tool_summary_record"
    )
    assert _preserve_records(messages) == 1, "counted as one record"
    assert strategy.records_in_conversation == 2, "the peak, which is what the count reports"
    for message_id in ("rec1_call", "rec1_res", "rec2_call", "rec2_res"):
        (old,) = (m for m in messages if m.message_id == message_id)
        assert old.additional_properties[EXCLUDED_KEY] is True
        assert old.additional_properties[EXCLUDE_REASON_KEY] == CONSOLIDATE_EXCLUDE_REASON
        assert not is_preserved(old), "released, so preserved and included keep meaning one thing"
        assert _record_text(old) == "", "and an excluded record carries no record"

    held = [m for m in messages if m.message_id in ("c3", "r3")]
    assert all(m.additional_properties.get(PRESERVE_REASON_KEY) == PRESERVE_REASON_UNCOVERED for m in held), (
        "the group no old record quoted was held on the pass that merged"
    )
    await strategy(messages)
    assert all(m.additional_properties.get(EXCLUDED_KEY) is True for m in held), (
        "and on the next pass the merged record's coverage licenses dropping it"
    )

    assert _droppable_groups_after(messages, find_record_index(messages)) == 0
    messages += _tool_group(9)
    annotate_message_groups(messages)
    annotate_token_counts(messages, tokenizer=TOKENIZER)
    assert _droppable_groups_after(messages, find_record_index(messages)) == 1, "pending work counts from it"
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=budget, tokenizer=TOKENIZER, arm=lambda: None, trigger_fraction=0.01, repeat_records=True
    )
    assert middleware._record_due(messages, find_record_index(messages)) is True, "and the middleware asks again"
    assert _hold_unrecorded(messages) == 1, "the new group is held behind it"
    assert messages[index].additional_properties[PRESERVE_REASON_KEY] == "tool_summary_record", (
        "and the merged record is not relabelled as a held tool group"
    )


def test_an_excluded_record_is_not_a_record() -> None:
    """The rule that makes a replaced record disappear from every reader, on a bare conversation."""
    messages = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        *_record_messages("lookup_1: CODE-1.", call_id="old"),
        *_record_messages("lookup_2: CODE-2.", call_id="new"),
    ]
    annotate_message_groups(messages)
    assert find_record_index(messages) == 4

    for message in messages[3:]:
        message.additional_properties[EXCLUDED_KEY] = True

    assert find_record_index(messages) == 2, "the newest record still being sent"
    assert _record_text(messages[4]) == ""
    assert [str(messages[group["end_index"]].message_id) for group in active_record_groups(messages)] == ["old_res"]
    assert _preserve_records(messages) == 1


async def test_the_other_list_replays_the_merge_rather_than_paying_for_it_again() -> None:
    """The live path compacts the copies sent on a call and then the store, and both get one merge.

    The second view of the same conversation asks the same question, so it gets the same answer
    under the same call id -- the model is sent one merged record and the store holds that one --
    and the summarizer is asked once.
    """
    summarizer = _RoutingSummarizer(merge=_short_record)
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)
    copies = _chain_conversation()
    store = copy.deepcopy(copies)

    await strategy(copies)
    await strategy(store)

    assert summarizer.log == ["merge"], "asked once"
    assert strategy.records_merged == 2, "and kept on both views"
    assert _active_record_ids(copies) == _active_record_ids(store) == ["compaction_record_0_result"]


async def test_a_refused_merge_is_not_paid_for_again_on_the_next_pass() -> None:
    """An identical request refused on one pass is answered from memory on the next."""
    summarizer = _RoutingSummarizer()
    strategy = _chain_composed(summarizer, harder_attempts=0)

    await strategy(_chain_conversation())
    await strategy(_chain_conversation())

    assert summarizer.log.count("merge") == 1
    assert strategy.record_merges_rejected == 2, "refused on both passes"


async def test_a_summarizer_that_fails_a_merge_leaves_the_records_and_moves_on() -> None:
    """A failure is not a refusal: it is counted apart, and the chain carries on to the next step."""
    log: list[str] = []

    class _FailsOnMerge(_RoutingSummarizer):
        async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
            if messages[0].text == DEFAULT_RECORD_MERGE_PROMPT:
                log.append("merge")
                raise RuntimeError("the summarizer is unavailable")
            return await super().get_response(messages, stream=stream, **kwargs)

    summarizer = _FailsOnMerge(log=log)
    strategy = _chain_composed(summarizer, harder_attempts=0)
    messages = _chain_conversation()

    await strategy(messages)

    assert log[:2] == ["merge", "fold"]
    assert (strategy.record_summary_failures, strategy.record_merges_rejected) == (1, 0)
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"]


@pytest.mark.parametrize("answer", ["", "  \t\n  "], ids=["empty", "whitespace"])
async def test_an_empty_merge_answer_is_a_failure_and_never_replaces_the_records(answer: str) -> None:
    """The one answer the size rule alone would accept, and the one that would cost the most.

    Acceptance is size only -- the strategy must not read content -- and an empty answer is
    always smaller than what it replaces. Without the emptiness guard in ``_ask`` a summarizer
    that returned nothing would have every active record excluded in favour of an empty one,
    discarding every value the records held in one step. It is a failure, counted as one, and
    the records stay exactly as they were.
    """
    log: list[str] = []
    summarizer = _RoutingSummarizer(log=log, merge=lambda body: answer)
    strategy = _chain_composed(summarizer, harder_attempts=0)
    messages = _chain_conversation()

    await strategy(messages)

    assert log[0] == "merge"
    assert (strategy.record_summary_failures, strategy.record_merges_rejected) == (1, 0)
    assert strategy.records_merged == 0
    assert _active_record_ids(messages) == ["rec1_res", "rec2_res"]


def test_the_standalone_user_row_keeps_its_defaults() -> None:
    """``user_summary_anchored`` still recompacts by default and remembers one request."""
    strategy = _user_phase()

    assert strategy.summary_mode == DEFAULT_SUMMARY_MODE == "recompact"
    assert strategy.remembered_requests == 1


async def test_an_ask_for_another_record_survives_a_merge_that_lowers_the_record_count() -> None:
    """The record strategy judges an ask by a record count that grew, and a merge lowers the count.

    An ask is made for a group no record quotes; the chain then merges the two records into one;
    the model's answer arrives as a new record that still does not quote it. Judged against the
    count the ask was made at, that arrival would read as no record at all and the group would sit
    held for two more passes; re-based by the merge, it is read as the record that came and covered
    nothing, and the group is settled on the pass it arrives.
    """
    summarizer = _RoutingSummarizer(merge=_short_record)
    strategy = _chain_composed(summarizer, harder_attempts=0)
    messages = _chain_conversation(uncovered=True)

    await strategy(messages)
    assert strategy.records_merged == 1
    assert strategy.groups_kept_uncovered == 1, "the group no record quotes"
    assert strategy.tool_results.take_reforce() is True, "and an ask for another record was made for it"

    messages += _record_messages("nothing to add.", call_id="rec9")
    await strategy(messages)

    assert strategy.groups_preserved_uncovered == 1, "the ask's record arrived, covered nothing, and settled it"


async def test_the_harder_rewrites_stop_as_soon_as_the_prompt_fits() -> None:
    """``harder_attempts`` is a bound, not a quota: a kept rewrite that fits ends step c."""
    summarizer = _RoutingSummarizer(harder=lambda attempt, body: _SHORT_RECORD)
    budget = await _post_record_size() - 500
    strategy = _chain_composed(summarizer, ceiling=budget)

    await strategy(_chain_conversation())

    assert summarizer.log == ["merge", "fold", "harder1"], "the first rewrite fit, so the second was never asked"
    assert (strategy.record_rewrites, strategy.record_rewrites_rejected) == (1, 0)


async def test_a_single_record_is_never_merged_only_rewritten() -> None:
    """Step a needs two records: merging one is a rewrite, which is step c's to ask for."""
    summarizer = _RoutingSummarizer()
    strategy = _chain_composed(summarizer, harder_attempts=1)
    messages = [message for message in _chain_conversation() if message.message_id not in ("rec1_call", "rec1_res")]

    await strategy(messages)

    assert "merge" not in summarizer.log
    assert "harder1" in summarizer.log
    assert (strategy.records_merged, strategy.record_merges_rejected) == (0, 0)


async def test_the_other_list_replays_both_the_band_and_the_fold_of_one_pass() -> None:
    """A pass over the budget may ask the user half for a band and then a fold; both replay.

    The single row remembers one request, which is enough for it; the composed row's user half
    remembers two, as its builder configures it, so the store pass after the copies pass pays
    for neither again.
    """
    summarizer = _RoutingSummarizer()
    messages = _chain_conversation()
    tail = messages[-2:]
    messages[-2:] = [
        Message(role="user", contents=["Turn 5: " + "u" * 8_000], message_id="u5"),
        Message(role="assistant", contents=["Reply 5."], message_id="a5"),
        *tail,
    ]
    strategy = ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER,
        tool_results=_record_phase(_CHAIN_CEILING, trigger_fraction=0.01),
        user_turns=UserTurnAnchoredSummarizationCompactionStrategy(
            max_input_tokens=_CHAIN_CEILING,
            tokenizer=TOKENIZER,
            client=summarizer,
            trigger_fraction=0.01,
            summary_mode=SUMMARY_MODE_BOUNDARY,
            remembered_requests=2,
        ),
        harder_attempts=0,
    )
    store = copy.deepcopy(messages)

    await strategy(messages)
    await strategy(store)

    assert summarizer.log.count("band") == 1 and summarizer.log.count("fold") == 1, summarizer.log
    assert strategy.user_summaries_replayed == 1, "the band replayed; the refused fold is refused again from memory"
