# Copyright (c) Microsoft. All rights reserved.

"""Tests for running the record strategy and the user-turn strategy over one conversation.

The composition owns no selection rule, so almost nothing here is about which messages get
removed -- that is tested beside each part. What is tested here is the three things composing
adds, each of which fails silently rather than loudly.

**Both halves act.** The point of the row is that the two reach material neither can reach
alone, and the failure mode is a pass where one half quietly did nothing: the conversation is
smaller, the row looks like it worked, and it is one of the single rows under a new name. So
the headline test runs each part alone over the same fixture and pins that each alone moves
only its own half.

**The order is the record phase first, and it is load-bearing rather than arbitrary.** The
record strategy's trigger is the lower of the two, and the middleware that asks the model for
a record reads the same size, so a pass that let the user half shrink the prompt first would
not delay the record -- it would stop it being asked for. The order is therefore asserted
directly *and* through the consequence, on a ceiling where the two orders give opposite
answers.

**The token count is what the two phases share, and the starvation counter is what says so.**
A record phase that removes enough to take the prompt under the user phase's own trigger
leaves a row reporting no user compactions, which is indistinguishable from a band that held
nothing. That reading is the counter's whole job, so it is tested at the boundary rather than
in the middle: one ceiling where it fires, one where the user half would not have fired anyway
and it must stay silent.
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
from agent_framework_lab_cachebench.compaction._toolsummary import (
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    ToolResultAnchoredSummarizationCompactionStrategy,
)
from agent_framework_lab_cachebench.compaction._usersummary import (
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

#: A ceiling both triggers are crossed on, and which the record phase's removals leave them
#: crossed on.
#:
#: The eight-turn fixture measures 18,937 tokens and the record phase takes it to 16,613. The
#: user line is 0.8 of this, 16,000, so both sizes are above it and the user half fires whether
#: or not the record phase ran first. The record line is 0.6 of this, 12,000, which both sizes
#: also clear -- and the ceiling itself is above the post-record size, so the record phase does
#: not reach for its fallback and no assertion here is about the anchored strategy.
#:
#: **Recompute the sizes whenever a default moves, and check the margins rather than the
#: signs.** A fixture that slips under a trigger does not fail; it asserts against a phase that
#: returned without doing anything, and passes. The fixture-sizing test below is what makes
#: that loud, and it is the only test here that should ever need these numbers rewritten.
_COMPACTING_CEILING = 20_000

#: A ceiling whose user line sits *between* the fixture's two sizes.
#:
#: 0.8 of 22,000 is 17,600: above the 16,613 the record phase leaves and below the 18,937 the
#: conversation starts at. That is the whole of what this fixture is for, and two tests read it
#: from opposite ends -- the user half is starved by a record phase that ran first, and the
#: user half fires normally when the record phase had no record to act on. 0.9 of it is 19,800,
#: so the record-less case is still waiting rather than falling back, which is what makes the
#: second reading about the composition and not about the anchored strategy.
_NARROW_CEILING = 22_000

#: A ceiling neither trigger is anywhere near, so a pass over the fixture must do nothing.
#:
#: 0.6 of this is 60,000 against a fixture of about 18,900, which is 32% of the lower of the
#: two lines rather than a value sitting near it.
_IDLE_CEILING = 100_000


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
) -> ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Return both halves composed, each defaulted to its own row's configuration."""
    return ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER,
        tool_results=tool_results or _record_phase(ceiling),
        user_turns=user_turns or _user_phase(ceiling),
    )


async def test_the_fixture_sits_where_the_three_ceilings_assume_it_does() -> None:
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
    assert after < 0.8 * _NARROW_CEILING < before, "the narrow ceiling's user line sits between the two sizes"
    assert 0.9 * _NARROW_CEILING > before > 0.6 * _NARROW_CEILING, "where a record-less pass is still waiting"
    assert before < 0.6 * _IDLE_CEILING, "and neither line is anywhere near on the idle one"


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

    It is not arbitrary. The record phase's trigger is the lower of the two and the middleware
    that asks the model for a record reads the same size, so a user phase that ran first and
    removed the majority share of the prompt would not delay the record -- it would take the
    conversation below the line that asks for one at all, and the row's tool half would report
    a model that never complied.
    """
    order: list[str] = []

    class _RecordSpy(ToolResultAnchoredSummarizationCompactionStrategy):
        async def __call__(self, messages: list[Message]) -> bool:
            order.append("record")
            return await super().__call__(messages)

    class _UserSpy(UserTurnAnchoredSummarizationCompactionStrategy):
        async def __call__(self, messages: list[Message]) -> bool:
            order.append("user")
            return await super().__call__(messages)

    strategy = _composed(
        tool_results=_RecordSpy(max_input_tokens=_COMPACTING_CEILING, tokenizer=TOKENIZER),
        user_turns=_UserSpy(max_input_tokens=_COMPACTING_CEILING, tokenizer=TOKENIZER, client=_Summarizer()),
    )

    assert await strategy(_conversation()) is True

    assert order == ["record", "user"]
    assert strategy.strategies == (strategy.tool_results, strategy.user_turns), (
        "the advertised order is what the pass runs, or a caller walking the parts reads a lie"
    )


async def test_the_order_is_what_decides_which_half_acts_on_a_narrow_ceiling() -> None:
    """The consequence of the order, on the ceiling where the two orders disagree.

    The user line sits between the conversation's size and its size once the record phase has
    acted, so exactly one half can fire. Running the record phase first spends the pass on the
    tool half and reports the user half as starved; running it the other way round would spend
    the pass on the user half -- and would then leave the prompt below the record phase's own
    trigger, which is the outcome the order exists to prevent.
    """
    strategy = _composed(_NARROW_CEILING)
    messages = _conversation()

    assert await strategy(messages) is True

    assert strategy.groups_kept_uncovered == 0, "the record phase acted"
    assert "x" * 100 not in _rendered(messages)
    assert strategy.user_compactions == 0, "and left no room for the user phase's own trigger"
    assert strategy.user_passes_starved == 1
    assert "Turn 4:" in _rendered(messages), "so the user band is exactly as it was found"


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
    """Two rows' worth of knobs on one object, and no shared trigger between them.

    The composed row is only readable beside the two single rows if a sweep of either row's
    flags moves this row's matching half and nothing else. The two settings chosen here are the
    ones that would be most tempting to unify -- a coverage share and a pair of user anchors --
    and they are asserted on behaviour rather than on attributes, because a constructor that
    stored them and a pass that read one number for both would pass an attribute check.
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


async def test_the_two_halves_keep_their_own_triggers() -> None:
    """No shared threshold, which is the one thing composing here deliberately does not do.

    ``TokenBudgetComposedStrategy`` gives its parts one ceiling because its family holds size
    fixed to compare orderings of deletion. Here the sizes the two phases reach are the
    measurement, and a shared trigger would fire both halves at one line that neither single
    row is measured at -- so the composed row would stop being comparable with either.
    """
    strategy = _composed(
        tool_results=_record_phase(trigger_fraction=0.2, fallback_fraction=0.99),
        user_turns=_user_phase(trigger_fraction=0.95),
    )

    assert strategy.tool_results.trigger_fraction == 0.2
    assert strategy.user_turns.trigger_fraction == 0.95

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
