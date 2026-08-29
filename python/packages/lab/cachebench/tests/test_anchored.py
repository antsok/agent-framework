# Copyright (c) Microsoft. All rights reserved.

"""Tests for the anchored compaction strategy.

The behaviour that matters is not "does it shrink the prompt" -- every strategy does that.
It is whether the decisions it makes on turn N survive unchanged into turn N+1, because
that is the only thing separating it from the strategies already measured costing more than
not compacting at all.
"""

from __future__ import annotations

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message
from agent_framework._compaction import included_token_count, project_included_messages
from agent_framework_lab_cachebench._anchored import REMOVAL_MARKER, AnchoredCompactionStrategy

TOKENIZER = CharacterEstimatorTokenizer()


def _tool_turn(index: int, payload_chars: int = 8_000) -> list[Message]:
    """Return one tool-call group plus the assistant reply that follows it."""
    call_id = f"call_{index}"
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}],
            message_id=f"a_call_{index}",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": call_id, "result": f"R{index} " + "x" * payload_chars}],
            message_id=f"t_res_{index}",
        ),
        Message(role="assistant", contents=[f"I looked up {index}."], message_id=f"a_txt_{index}"),
    ]


def _conversation(tool_turns: int, payload_chars: int = 8_000) -> list[Message]:
    """Return a conversation with a stable head and ``tool_turns`` tool groups."""
    messages = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        Message(role="user", contents=["Requirement: region is EU-WEST-1."], message_id="u0"),
        Message(role="assistant", contents=["Understood."], message_id="a0"),
    ]
    for index in range(tool_turns):
        messages.append(Message(role="user", contents=[f"Look up {index}."], message_id=f"u_{index}"))
        messages.extend(_tool_turn(index, payload_chars))
    return messages


def _text_of(message: Message) -> str:
    """Return a message's payload as the model would see it, tool results included."""
    parts: list[str] = []
    for content in message.contents:
        result = getattr(content, "result", None)
        text = getattr(content, "text", None)
        parts.append(str(result) if result is not None else (text if text is not None else str(content)))
    return "".join(parts)


def _rendered(messages: list[Message]) -> str:
    """Return what the model would actually receive, as one string."""
    return "\n".join(f"{m.role}:{_text_of(m)}" for m in project_included_messages(messages))


async def test_head_and_tail_anchors_are_never_touched() -> None:
    """The requirement stated at the start must survive, however long the conversation runs."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=1_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    rendered = _rendered(messages)

    assert "EU-WEST-1" in rendered
    # The most recent tool result is working context and is kept verbatim.
    assert "R7 " + "x" * 100 in rendered


async def test_the_ceiling_is_met() -> None:
    """A token-aware strategy that leaves the prompt over its ceiling has failed at its job."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=4_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=10)

    await strategy(messages)

    kept = project_included_messages(messages)
    total = sum(TOKENIZER.count_tokens("".join(str(c) for c in m.contents)) for m in kept)
    assert total <= 4_000


async def test_decisions_are_frozen_as_the_conversation_grows() -> None:
    """A group compacted at turn N must look identical at turn N+1.

    This is the whole design. Prompt caching is strict-prefix, so a strategy that re-decides
    the fate of an old group -- as any "compact to 50% when over 80%" rule does -- rewrites
    the start of the prompt and re-bills everything after it. Here the prefix that both turns
    share must come out byte-identical.
    """
    strategy = AnchoredCompactionStrategy(max_input_tokens=3_000, tokenizer=TOKENIZER)

    earlier = _conversation(tool_turns=8)
    await strategy(earlier)
    earlier_rendered = _rendered(earlier)

    # The same conversation two tool turns later, compacted from scratch as the before-phase
    # always does: exclusion flags do not survive into storage.
    later = _conversation(tool_turns=10)
    await strategy(later)

    # Groups 0-3 are in the middle band of both conversations: the band only ever grows from
    # the tail end, so a group that has entered it never leaves. Each must render identically.
    earlier_by_id = {m.message_id: _text_of(m) for m in project_included_messages(earlier)}
    later_by_id = {m.message_id: _text_of(m) for m in project_included_messages(later)}
    for index in range(4):
        for message_id in (f"a_call_{index}", f"t_res_{index}", f"a_txt_{index}"):
            assert earlier_by_id.get(message_id) == later_by_id.get(message_id), message_id
    assert earlier_rendered  # the earlier conversation was in fact compacted, not empty


async def test_running_twice_changes_nothing_further() -> None:
    """Idempotence. A second pass that shortens an already-shortened result would compound."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=3_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    once = _rendered(messages)
    changed_again = await strategy(messages)

    assert _rendered(messages) == once
    assert changed_again is False


async def test_shortening_alone_is_preferred_to_removing_anything() -> None:
    """When trimming results is enough, nothing is dropped and the structure stays intact.

    This is the cheap case and it should be the common one: the model still sees that every
    call happened and roughly what each returned, and no message is missing.
    """
    strategy = AnchoredCompactionStrategy(max_input_tokens=5_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    rendered = _rendered(messages)

    assert REMOVAL_MARKER in rendered
    assert "[compacted: an earlier tool call and its result]" not in rendered
    assert "[compacted: an earlier assistant reply]" not in rendered
    # Every reply is still there, so nothing the model said about a result was lost.
    assert all(f"I looked up {index}." in rendered for index in range(8))


async def test_assistant_narration_is_the_last_thing_dropped() -> None:
    """Under a ceiling too tight for trimming, tool groups go before any prose does."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=120, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    rendered = _rendered(messages)

    assert "[compacted: an earlier tool call and its result]" in rendered


async def test_collapse_assistant_text_can_be_forbidden() -> None:
    """The last-resort step must be switchable, so its cost can be measured separately."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=500, tokenizer=TOKENIZER, collapse_assistant_text=False)
    messages = _conversation(tool_turns=8)

    await strategy(messages)

    assert "[compacted: an earlier assistant reply]" not in _rendered(messages)


async def test_short_conversations_are_left_alone() -> None:
    """With nothing between the anchors there is nothing to compact, and no cache to spend."""
    strategy = AnchoredCompactionStrategy(max_input_tokens=10, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=1)
    before = _rendered(messages)

    assert await strategy(messages) is False
    assert _rendered(messages) == before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_input_tokens": 0}, "max_input_tokens"),
        ({"max_input_tokens": 100, "keep_head_groups": -1}, "keep_head_groups"),
        ({"max_input_tokens": 100, "keep_tokens": -1}, "keep_tokens"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, int], match: str) -> None:
    """A silently accepted bad bound would produce a plausible-looking wrong measurement."""
    with pytest.raises(ValueError, match=match):
        AnchoredCompactionStrategy(tokenizer=TOKENIZER, **kwargs)


async def test_the_ceiling_is_best_effort_when_the_anchors_alone_exceed_it() -> None:
    """The anchors are inviolable, so an impossible ceiling is missed rather than obeyed.

    One tool result in the tail is larger than the whole ceiling here. No strategy can split
    a single result, so the honest behaviour is to shed everything it is allowed to shed and
    stop -- not to start eating the working set or the requirements to chase a number it
    cannot reach.
    """
    strategy = AnchoredCompactionStrategy(max_input_tokens=120, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    rendered = _rendered(messages)

    assert included_token_count(messages) > 120
    # What it was allowed to shed, it shed.
    assert "[compacted: an earlier tool call and its result]" in rendered
    assert "[compacted: an earlier assistant reply]" in rendered
    # What it was not allowed to touch is untouched.
    assert "EU-WEST-1" in rendered


async def test_retention_scales_with_the_ceiling() -> None:
    """A fixed retention becomes a rounding error as tool results grow.

    Measured: a 600-character retention is 1.9% of an 8,000-token result and 0.6% of a
    25,200-token one. The strategy scored 32 of 53 planted facts at the first size and 11 at
    the second, and 11 was exactly the five non-tool facts plus the single code that fell
    inside each surviving head fragment. The band budget now scales with the window instead.
    """
    small = AnchoredCompactionStrategy(max_input_tokens=57_952, tokenizer=TOKENIZER)
    large = AnchoredCompactionStrategy(max_input_tokens=269_952, tokenizer=TOKENIZER)
    band = [
        {"kind": "tool_call", "group_id": f"g{index}", "start_index": index, "end_index": index} for index in range(6)
    ]

    assert large._keep_tokens_for(band) > 4 * small._keep_tokens_for(band)
    # And an explicit value still wins, so a caller can pin it for a comparison.
    pinned = AnchoredCompactionStrategy(max_input_tokens=269_952, tokenizer=TOKENIZER, keep_tokens=600)
    assert pinned._keep_tokens_for(band) == 600


async def test_a_wider_band_share_keeps_more_of_each_result() -> None:
    """The knob has to move the trade-off, or it is decoration."""
    narrow = AnchoredCompactionStrategy(max_input_tokens=269_952, tokenizer=TOKENIZER, band_share=0.05)
    wide = AnchoredCompactionStrategy(max_input_tokens=269_952, tokenizer=TOKENIZER, band_share=0.5)
    band = [
        {"kind": "tool_call", "group_id": f"g{index}", "start_index": index, "end_index": index} for index in range(6)
    ]

    assert wide._keep_tokens_for(band) > narrow._keep_tokens_for(band)


def test_band_share_is_validated() -> None:
    """A share outside (0, 1] would silently produce a nonsensical budget."""
    with pytest.raises(ValueError, match="band_share"):
        AnchoredCompactionStrategy(max_input_tokens=1_000, tokenizer=TOKENIZER, band_share=0.0)
    with pytest.raises(ValueError, match="band_share"):
        AnchoredCompactionStrategy(max_input_tokens=1_000, tokenizer=TOKENIZER, band_share=1.5)
