# Copyright (c) Microsoft. All rights reserved.

"""Tests for the strategy that summarises the user's own turns.

Three things are checked harder than the rest, because three things are what this design is.

It must touch **only** user turns: the row exists to be compared with the tool-side rows in the
same table, and a strategy that quietly also shed tool results would make both numbers
unreadable.

It must **recompact its own output**, which is the opposite of what the anchored strategy does
with a result it has already shortened. That is the headline behaviour and the one a future
change is most likely to "fix" by refusing, so the test for it asserts the consequence -- the
conversation does not grow across two compactions -- rather than only the mechanics, because
the mechanics can be satisfied by a pass that inserts a second summary beside the first.

And it must **fail safe**: a summarizer that raises or answers with nothing has to leave a
conversation that is byte-identical to the one it was given, since the alternative is a band
excluded on the strength of a replacement that does not exist.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, ChatResponse, Message
from agent_framework._compaction import (
    EXCLUDE_REASON_KEY,
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    SUMMARIZED_BY_SUMMARY_ID_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    SUMMARY_OF_MESSAGE_IDS_KEY,
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
    project_included_messages,
)
from agent_framework_lab_cachebench.compaction._preserve import set_preserved
from agent_framework_lab_cachebench.compaction._usersummary import (
    DEFAULT_KEEP_HEAD_USER_TURNS,
    DEFAULT_KEEP_TAIL_USER_TURNS,
    DEFAULT_USER_TRIGGER_FRACTION,
    EXCLUDE_REASON,
    SUMMARY_ID_PREFIX,
    USER_SUMMARY_MARKER,
    UserTurnAnchoredSummarizationCompactionStrategy,
)

TOKENIZER = CharacterEstimatorTokenizer()

#: Characters of filler in every user turn and every reply, so each message is about the same
#: size and the arithmetic below is exact rather than approximate. An eight-turn conversation
#: built from these measures roughly 16,700 tokens.
_TURN_CHARS = 4_000

#: A ceiling the eight-turn fixture is comfortably over, so the default trigger fires.
#:
#: 0.8 of this is 9,600 against a fixture of about 16,700, which is 174% of the line rather than
#: a value sitting near it. **Recompute both of these whenever a default moves**, and check the
#: margin rather than the sign: a fixture that slips under the trigger does not fail, it asserts
#: against a strategy that returned without doing anything, and passes.
_COMPACTING_CEILING = 12_000

#: A ceiling the same fixture is comfortably under, so nothing fires.
#:
#: 0.8 of this is 80,000 against the same 16,700, so the fixture is at 21% of the line.
_IDLE_CEILING = 100_000


class _Summarizer:
    """A summarizer that answers from a script, and records what it was asked.

    The requests are kept because what reaches the summarizer is part of the contract: it is
    sent the user's turns and nothing else, so a test can prove the assistant's replies and the
    tool output never leave the conversation.
    """

    def __init__(self, text: str = "The user asked for the earlier things, in order.") -> None:
        self.requests: list[list[Message]] = []
        self.text = text

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        self.requests.append(list(messages))
        return ChatResponse(messages=[Message(role="assistant", contents=[self.text])])


class _FailingSummarizer:
    """A summarizer that raises, which is one of the two ways this strategy must do nothing."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        self.calls += 1
        raise RuntimeError("the summarizer is unavailable")


class _EmptySummarizer:
    """A summarizer that answers with whitespace, which is the other way."""

    async def get_response(self, messages: list[Message], *, stream: bool = False, **kwargs: Any) -> ChatResponse:
        return ChatResponse(messages=[Message(role="assistant", contents=["   "])])


def _turn(index: int, *, chars: int = _TURN_CHARS) -> list[Message]:
    """Return one user turn and the assistant's reply to it.

    Args:
        index: Numbers the pair, so ids and text are unique and a test can name one of them.

    Keyword Args:
        chars: Filler in each message.

    Returns:
        The two messages.
    """
    return [
        Message(role="user", contents=[f"Turn {index}: " + "u" * chars], message_id=f"u{index}"),
        Message(role="assistant", contents=[f"Reply {index}: " + "a" * chars], message_id=f"a{index}"),
    ]


def _tool_group(index: int) -> list[Message]:
    """Return a tool call and its result, which this strategy must never read or touch.

    Args:
        index: Numbers the pair.

    Returns:
        The two messages.
    """
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": f"call_{index}", "name": "lookup", "arguments": "{}"}],
            message_id=f"c{index}",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": f"call_{index}", "result": f"TOOL-{index}-VALUE"}],
            message_id=f"r{index}",
        ),
    ]


def _conversation(turns: int, *, first_turn: int = 0, tools_after: int | None = None) -> list[Message]:
    """Return a conversation of ``turns`` user/assistant pairs.

    Args:
        turns: How many pairs to build.

    Keyword Args:
        first_turn: Index the pairs are numbered from, so a second stretch can be appended to
            the first without colliding on ids -- which is what a second crossing of the
            threshold needs.
        tools_after: Index of the turn a tool-call group follows, or None for a conversation
            with no tool work in it at all.

    Returns:
        The messages, opening with the system message a real conversation opens with.
    """
    messages: list[Message] = []
    if first_turn == 0:
        messages.append(Message(role="system", contents=["You are an assistant."], message_id="sys"))
    for offset in range(turns):
        index = first_turn + offset
        messages += _turn(index)
        if tools_after is not None and index == tools_after:
            messages += _tool_group(index)
    return messages


def _included(messages: list[Message]) -> int:
    """Return the token count the strategy itself would read off ``messages``.

    Annotates first, because a token count is cached per message and a conversation that has
    just been mutated carries numbers describing text that no longer exists.

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


def _user_tokens(messages: list[Message]) -> int:
    """Return the tokens the user's own messages contribute to the prompt.

    The half this strategy owns, measured apart from the whole, because the other half grows
    with every turn whatever this does: the assistant's replies are not its to remove, so a
    total cannot say whether a compaction worked.
    """
    return sum(TOKENIZER.count_tokens(text) for text in _user_texts(messages))


def _strategy(summarizer: Any = None, **kwargs: Any) -> UserTurnAnchoredSummarizationCompactionStrategy:
    """Return a strategy over the compacting ceiling unless a test says otherwise."""
    kwargs.setdefault("max_input_tokens", _COMPACTING_CEILING)
    return UserTurnAnchoredSummarizationCompactionStrategy(
        tokenizer=TOKENIZER, client=summarizer or _Summarizer(), **kwargs
    )


def test_the_fixture_sits_where_the_two_ceilings_assume_it_does() -> None:
    """A fixture that drifts under a trigger asserts against a strategy that did nothing.

    Every test here that expects a compaction is worthless if the conversation is below the
    line, and it does not fail when that happens -- it passes, having measured an early return.
    So the two margins are checked once, here, rather than being trusted to a comment that
    nothing reads.
    """
    size = _included(_conversation(8))

    assert size > _COMPACTING_CEILING * DEFAULT_USER_TRIGGER_FRACTION * 1.5, "the fixture must clear the trigger"
    assert size < _IDLE_CEILING * DEFAULT_USER_TRIGGER_FRACTION * 0.5, "and must be nowhere near the idle one"


def test_the_defaults_are_one_turn_at_each_end_and_a_late_trigger() -> None:
    """The three numbers a caller inherits, pinned where a reader of the table can find them.

    They are defaults rather than derivations, and a change to any of them changes what every
    archived row means, so moving one should have to move this line too.
    """
    assert (DEFAULT_KEEP_HEAD_USER_TURNS, DEFAULT_KEEP_TAIL_USER_TURNS) == (1, 1)
    assert DEFAULT_USER_TRIGGER_FRACTION == 0.8


async def test_it_compacts_user_turns_and_leaves_tool_results_and_assistant_messages_alone() -> None:
    """The independence the whole comparison rests on.

    ``tool_summary_anchored`` and the anchored family act on the tool half; this acts on the
    user half; and the table reads the two as separate answers to separate questions. A
    strategy that shed both would make each row's ``snap%`` a sum of two effects with nothing
    saying how it divided -- and it would do it silently, because every assertion about *this*
    half would still pass.
    """
    summarizer = _Summarizer()
    strategy = _strategy(summarizer)
    messages = _conversation(8, tools_after=3)

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "TOOL-3-VALUE" in rendered, "the tool result is not this strategy's to remove"
    assert "Reply 4:" in rendered, "and neither is the assistant's narration"
    assert "Turn 4:" not in rendered, "while the user turn beside them is gone"
    untouched = [
        message
        for message in messages
        if message.role in {"assistant", "tool", "system"}
        and (message.additional_properties.get(EXCLUDED_KEY, False) or _summarized_by(message))
    ]
    assert not untouched, f"these were annotated by a strategy that may only read user turns: {untouched}"
    asked = summarizer.requests[0][-1].text or ""
    assert "Reply 4:" not in asked, "the summarizer is sent the user's turns and nothing else"
    assert "TOOL-3-VALUE" not in asked


async def test_the_first_and_last_user_turns_survive_on_the_defaults() -> None:
    """The task and the live request, which are the two turns that cannot be paraphrased.

    The first carries the requirements every surviving value is interpreted against -- the
    thing truncation measured itself losing, 29 of 53 facts left in a prompt the model could
    not use. The last is the question being asked right now, and a model answering a summary of
    it answers a different question.
    """
    strategy = _strategy()
    messages = _conversation(8)

    assert await strategy(messages) is True
    texts = _user_texts(messages)

    assert len(texts) == 3, f"head, one summary and tail, not {texts}"
    assert texts[0].startswith("Turn 0:")
    assert texts[1].startswith(USER_SUMMARY_MARKER)
    assert texts[2].startswith("Turn 7:")
    assert strategy.user_messages_replaced == 6


async def test_wider_anchors_keep_more_turns_verbatim_at_each_end() -> None:
    """Both ends move, and they move independently.

    One number for both would be the cheaper API and the wrong one: the two ends are protecting
    different things -- the terms of reference at the front, the live request at the back -- so
    the question "what is the opening task statement worth against the recent turns" can only
    be asked by moving them apart.
    """
    strategy = _strategy(keep_head_user_turns=2, keep_tail_user_turns=3)
    messages = _conversation(8)

    assert await strategy(messages) is True
    texts = _user_texts(messages)

    assert [text[:7] for text in texts] == [
        "Turn 0:",
        "Turn 1:",
        USER_SUMMARY_MARKER[:7],
        "Turn 5:",
        "Turn 6:",
        "Turn 7:",
    ]
    assert strategy.user_messages_replaced == 3


async def test_anchors_that_leave_no_band_compact_nothing() -> None:
    """A conversation shorter than its own anchors is a no-op, not an error or a summary of one.

    Asked for a head of four and a tail of four on eight turns, there is nothing in between,
    and the honest answer is to return False: a summary of an empty band would be a summarizer
    call spent on nothing and a message inserted that stands for no message at all.
    """
    summarizer = _Summarizer()
    strategy = _strategy(summarizer, keep_head_user_turns=4, keep_tail_user_turns=4)
    messages = _conversation(8)
    before = list(messages)

    assert await strategy(messages) is False
    assert messages == before
    assert summarizer.requests == []
    assert strategy.user_compactions == 0


async def test_nothing_happens_below_the_trigger() -> None:
    """Under the line the strategy must not even look, let alone spend a summarizer call.

    A compaction that was not needed is the worst trade this package measures: it breaks the
    cached prefix, which is billed at the uncached rate for everything behind the edit, and
    saves tokens on a prompt that already fitted.
    """
    summarizer = _Summarizer()
    strategy = _strategy(summarizer, max_input_tokens=_IDLE_CEILING)
    messages = _conversation(8)
    before = list(messages)

    assert await strategy(messages) is False
    assert messages == before, "an untriggered pass must not even annotate an exclusion"
    assert summarizer.requests == [], "and must not spend a call finding that out"
    assert (strategy.user_compactions, strategy.user_messages_replaced) == (0, 0)


async def test_a_second_crossing_recompacts_the_earlier_summary_together_with_the_new_turns() -> None:
    """The headline behaviour, and the one a later change is most likely to undo.

    ``AnchoredCompactionStrategy._shorten`` deliberately refuses to re-trim a result carrying
    its own removal marker, because its trigger is the band's geometry and re-trimming would
    fire on every pass. This strategy's trigger is a threshold that a compaction moves away
    from, so the same rewrite is rare by construction and re-reading its own summary is what
    stops summaries accumulating. Somebody who knows the first rule and not the second will
    make this strategy skip its own output, and every mechanical assertion below would still
    pass: the second pass would summarise the new turns alone and insert a second summary
    beside the first.

    So the consequence is asserted as well, and it is asserted on the user side of the prompt
    rather than on the whole of it: the assistant's replies are not this strategy's to remove
    and they grow with every turn, so a total would rise across two compactions however well
    the user band was compacted. On the half this strategy owns the arithmetic is exact. With a
    summarizer of fixed size and turns of one size, a conversation that recompacts carries the
    same user-side tokens after the second pass as after the first -- head, one summary, tail --
    while one that refuses carries a whole extra summary, and another on every crossing after
    that.
    """
    strategy = _strategy()
    messages = _conversation(8)

    assert await strategy(messages) is True
    user_after_first = _user_tokens(messages)
    first_summary = next(message for message in messages if (message.message_id or "").startswith(SUMMARY_ID_PREFIX))

    messages += _conversation(6, first_turn=8)
    before_second = _included(messages)
    user_before_second = _user_tokens(messages)

    assert await strategy(messages) is True
    after_second = _included(messages)
    user_after_second = _user_tokens(messages)
    summaries = [message for message in project_included_messages(messages) if _is_summary(message)]

    assert strategy.user_compactions == 2
    assert len(summaries) == 1, f"the earlier summary must be replaced, not joined: {len(summaries)} are being sent"
    assert summaries[0] is not first_summary
    assert first_summary.additional_properties[EXCLUDED_KEY] is True, "the earlier summary was superseded"
    assert first_summary.message_id in _summary_of_message_ids(summaries[0]), (
        "and the new summary has to say so, or nothing records that it stands for the old one"
    )
    assert after_second < before_second, "the second pass has to make the conversation smaller"
    assert user_after_second < user_before_second
    assert len(_user_texts(messages)) == 3, "head, one summary and tail, exactly as after the first pass"
    assert user_after_second <= user_after_first, (
        "a pass that refused to re-read its own summary would leave the user side of the "
        "prompt larger after the second compaction than after the first, by one whole summary"
    )
    # Seven new turns' worth of material -- the six appended plus the summary standing for the
    # first six -- against one message, which is the count a refusal could not produce.
    assert strategy.user_messages_replaced == 7


async def test_a_band_holding_only_its_own_summary_is_left_alone() -> None:
    """Recompaction is licensed by new material, not by the prompt still being large.

    The size that fired the trigger does not go away when the compaction that answered it is
    already in the prompt, so a pass reading size alone would rewrite the same message at the
    same position on every turn for the rest of the run -- which is precisely the thrash
    ``_shorten`` refuses to do, arrived at from the other direction. The middleware in
    ``_toolsummary`` had the same defect and ``_record_due`` is where it was fixed.
    """
    summarizer = _Summarizer()
    strategy = _strategy(summarizer)
    messages = _conversation(8)

    assert await strategy(messages) is True
    assert len(summarizer.requests) == 1

    assert await strategy(messages) is False, "nothing new has been said, so there is nothing to recompact"
    assert len(summarizer.requests) == 1, "and no call may be spent discovering that"
    assert strategy.user_compactions == 1


async def test_a_summarizer_failure_leaves_the_conversation_untouched_and_is_counted() -> None:
    """No replacement means no supersession, and the row has to say the pass was lost.

    Excluding the band first and restoring it on failure is the obvious alternative and is the
    mutate-and-roll-back ``MinimumGainAnchoredCompactionStrategy`` already refused, on the
    ground that a restore missing one field is a silent wrong answer. Here it would be worse
    than silent: the band would be gone and nothing would stand in its place.
    """
    summarizer = _FailingSummarizer()
    strategy = _strategy(summarizer)
    messages = _conversation(8)
    before = list(messages)

    assert await strategy(messages) is False
    assert messages == before
    assert summarizer.calls == 1, "the failure has to be a real attempt, not a refusal to try"
    assert strategy.user_summary_failures == 1
    assert (strategy.user_compactions, strategy.user_messages_replaced) == (0, 0)


async def test_a_summarizer_that_answers_with_nothing_is_the_same_failure() -> None:
    """An empty summary and a raised exception leave the caller with the same thing: no text.

    Treated apart, the empty case is the one that gets through -- it is not an error, so it
    reads as success, and the band would be superseded by a message that says nothing at all.
    """
    strategy = _strategy(_EmptySummarizer())
    messages = _conversation(8)
    before = list(messages)

    assert await strategy(messages) is False
    assert messages == before
    assert strategy.user_summary_failures == 1


async def test_the_replacement_carries_the_frameworks_own_supersession_annotations() -> None:
    """Replace, in the way the framework already means by it, rather than in a private way.

    ``SummarizationStrategy`` links a summary to what it replaced in both directions and
    excludes the originals with a reason, and the framework's own summary reconciliation reads
    those annotations. A strategy that dropped the band instead, or linked it with keys of its
    own, would produce a conversation that is only legible to itself.
    """
    strategy = _strategy()
    messages = _conversation(8)

    assert await strategy(messages) is True
    summary = next(message for message in messages if (message.message_id or "").startswith(SUMMARY_ID_PREFIX))
    replaced = [message for message in messages if message.message_id in {f"u{index}" for index in range(1, 7)}]

    assert _summary_of_message_ids(summary) == [f"u{index}" for index in range(1, 7)]
    assert len(_summary_of_group_ids(summary)) == 6
    for message in replaced:
        assert message.additional_properties[EXCLUDED_KEY] is True
        assert message.additional_properties[EXCLUDE_REASON_KEY] == EXCLUDE_REASON
        assert _summarized_by(message) == summary.message_id
    assert messages.index(summary) < messages.index(replaced[0]), "the summary stands where the band started"


async def test_the_replacement_is_a_user_message() -> None:
    """It stands for user turns, and only a user message can be recompacted as one.

    The framework's summarizer writes an assistant message because it summarises groups of
    every kind. Here that choice would make the summary invisible to this strategy's own
    selection rule, so the first summary would sit in the prompt untouched for the rest of the
    run -- and it would put two assistant messages in a row wherever the band ended just before
    a reply, which several providers reject.
    """
    strategy = _strategy()
    messages = _conversation(8)

    assert await strategy(messages) is True
    summary = next(message for message in messages if (message.message_id or "").startswith(SUMMARY_ID_PREFIX))

    assert summary.role == "user"
    assert USER_SUMMARY_MARKER in (summary.text or ""), (
        "a model shown a silently reduced conversation answers as though it had seen all of it"
    )


async def test_a_preserved_user_turn_is_never_summarised() -> None:
    """``_preserve`` means no strategy may shorten, drop or shed a message, and this drops.

    The mark exists because one strategy was measured destroying another's output. Superseding
    a preserved turn is the same loss by a politer route: the message stops being sent, and
    what stands in its place is a paraphrase written by a different model.
    """
    strategy = _strategy()
    messages = _conversation(8)
    protected = next(message for message in messages if message.message_id == "u3")
    set_preserved(protected, preserved=True, reason="a test")

    assert await strategy(messages) is True

    assert protected.additional_properties.get(EXCLUDED_KEY, False) is False
    assert "Turn 3:" in _rendered(messages)
    assert strategy.user_messages_replaced == 5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"max_input_tokens": 0}, "max_input_tokens", id="ceiling"),
        pytest.param({"keep_head_user_turns": -1}, "keep_head_user_turns", id="head"),
        pytest.param({"keep_tail_user_turns": -1}, "keep_tail_user_turns", id="tail"),
        pytest.param({"trigger_fraction": 0.0}, "trigger_fraction", id="trigger-zero"),
        pytest.param({"trigger_fraction": 1.5}, "trigger_fraction", id="trigger-above-one"),
    ],
)
def test_a_configuration_outside_its_range_is_refused_by_the_constructor(kwargs: dict[str, Any], match: str) -> None:
    """The bound belongs here, where the class is, and not in whatever is building it.

    ``compaction/`` ships without the benchmark, so a check written in the lab's pre-flight
    would leave the class unguarded for every other caller -- and would give a sweep two places
    to disagree about what is legal. The lab's job is only to build every selected strategy
    before a run spends anything, so that these errors arrive at the command line.
    """
    with pytest.raises(ValueError, match=match):
        _strategy(**kwargs)


def _is_summary(message: Message) -> bool:
    """Return whether ``message`` is one of the strategy's summaries.

    Written out here rather than imported, so a test cannot agree with the implementation by
    sharing its mistake: if the strategy stopped recognising its own output, the private helper
    would say so too and the recompaction test above would pass on a conversation carrying two
    summaries.
    """
    return (message.message_id or "").startswith(SUMMARY_ID_PREFIX)


def _summarized_by(message: Message) -> str | None:
    """Return the id of the summary that superseded ``message``, if one did."""
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return annotation.get(SUMMARIZED_BY_SUMMARY_ID_KEY) if isinstance(annotation, dict) else None


def _summary_of_message_ids(message: Message) -> list[str]:
    """Return the message ids a summary says it stands for."""
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return list(annotation.get(SUMMARY_OF_MESSAGE_IDS_KEY, [])) if isinstance(annotation, dict) else []


def _summary_of_group_ids(message: Message) -> list[str]:
    """Return the group ids a summary says it stands for."""
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return list(annotation.get(SUMMARY_OF_GROUP_IDS_KEY, [])) if isinstance(annotation, dict) else []
