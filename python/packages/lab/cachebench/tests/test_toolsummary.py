# Copyright (c) Microsoft. All rights reserved.

"""Tests for the record-then-drop strategy.

Two phases, and the tests separate them: phase 1 must ask without inserting anything into the
cached prefix, and phase 2 must act only on a record the *provider* issued -- never on one the
client invented, which is unsafe on routes that track tool calls server-side.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message
from agent_framework._compaction import project_included_messages
from agent_framework_lab_cachebench._toolsummary import (
    RECALL_TOOL_NAME,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    find_record_index,
)

TOKENIZER = CharacterEstimatorTokenizer()


def _conversation(tool_turns: int, payload_chars: int = 8_000, *, record: str | None = None) -> list[Message]:
    """Return a conversation, optionally with a recall record the agent already made."""
    messages = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        Message(role="user", contents=["Requirement: region is EU-WEST-1."], message_id="u0"),
        Message(role="assistant", contents=["Understood."], message_id="a0"),
    ]
    for index in range(tool_turns):
        call_id = f"call_{index}"
        messages += [
            Message(role="user", contents=[f"Look up {index}."], message_id=f"u_{index}"),
            Message(
                role="assistant",
                contents=[{"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}],
                message_id=f"a_call_{index}",
            ),
            Message(
                role="tool",
                contents=[
                    {"type": "function_result", "call_id": call_id, "result": f"CODE-{index} " + "x" * payload_chars}
                ],
                message_id=f"t_res_{index}",
            ),
        ]
    if record is not None:
        messages += [
            Message(
                role="assistant",
                contents=[{"type": "function_call", "call_id": "rec", "name": RECALL_TOOL_NAME, "arguments": "{}"}],
                message_id="rec_call",
            ),
            Message(
                role="tool",
                contents=[{"type": "function_result", "call_id": "rec", "result": record}],
                message_id="rec_res",
            ),
        ]
    return messages


def _rendered(messages: list[Message]) -> str:
    """Return what the model would receive, tool results included."""
    parts: list[str] = []
    for message in project_included_messages(messages):
        for content in message.contents:
            result = getattr(content, "result", None)
            text = getattr(content, "text", None)
            parts.append(str(result) if result is not None else (text if text is not None else str(content)))
    return "\n".join(parts)


#: A ceiling that puts the eight-turn conversation between the two thresholds, so the default
#: strategy asks and waits rather than giving up. Chosen from the fixture's own size: the
#: conversation is about 16,000 tokens, which is 73% of this, between the 60% trigger and the
#: 90% fallback.
_WAITING_CEILING = 22_000


def _strategy(**kwargs: Any) -> ToolResultAnchoredSummarizationCompactionStrategy:
    kwargs.setdefault("max_input_tokens", _WAITING_CEILING)
    return ToolResultAnchoredSummarizationCompactionStrategy(tokenizer=TOKENIZER, **kwargs)


async def test_phase_two_drops_only_what_precedes_the_record() -> None:
    """The record is what those results were reduced to; everything after it is untouched."""
    strategy = _strategy()
    messages = _conversation(tool_turns=8, record="CODE-0 CODE-1 CODE-2")

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "CODE-0 CODE-1 CODE-2" in rendered, "the record itself survives"
    assert "x" * 100 not in rendered, "the bulk behind it is gone"
    assert "EU-WEST-1" in rendered, "the head anchor is never touched"
    assert strategy.records_found == 1


async def test_a_client_invented_record_is_not_trusted() -> None:
    """Only a result whose call the provider issued counts.

    A tool result with no matching call is what synthesising the pair client-side produces,
    and it is exactly what breaks on routes that track tool calls server-side. The strategy
    must not treat one as a record.
    """
    strategy = _strategy()
    messages = _conversation(tool_turns=8)
    messages.append(
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": "invented", "result": "CODE-0"}],
            message_id="fake",
        )
    )

    await strategy(messages)

    assert strategy.records_found == 0
    assert "CODE-0 " in _rendered(messages), "nothing dropped on the strength of a bare result"


async def test_nothing_happens_below_the_trigger() -> None:
    """A record that is not needed costs an agent turn and buys nothing."""
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=10_000_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert strategy.fallbacks_used == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_input_tokens": 0}, "max_input_tokens"),
        ({"max_input_tokens": 100, "trigger_fraction": 0.0}, "trigger_fraction"),
        ({"max_input_tokens": 100, "keep_head_groups": -1}, "keep_head_groups"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], match: str) -> None:
    """A silently accepted bad bound produces a plausible-looking wrong measurement."""
    with pytest.raises(ValueError, match=match):
        ToolResultAnchoredSummarizationCompactionStrategy(tokenizer=TOKENIZER, **kwargs)


async def test_the_fallback_fires_when_the_record_never_arrives() -> None:
    """Waiting forever means overflowing the window, which is worse than truncating.

    The model may never call the tool. Past the fallback threshold the strategy stops asking
    and compacts without a record: the tool results are lost either way at that point, and
    the alternative is a provider error that loses the whole conversation.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.2
    )
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is True
    assert strategy.fallbacks_used == 1
    assert strategy.records_found == 0
    # It compacted rather than waiting further.
    assert "CODE-0" not in _rendered(messages), "the oldest results are gone"
    assert "EU-WEST-1" in _rendered(messages), "the fallback keeps the head anchor too"


async def test_thresholds_the_wrong_way_around_are_rejected() -> None:
    """A fallback at or below the trigger silently disables the whole design."""
    with pytest.raises(ValueError, match="fallback_fraction"):
        ToolResultAnchoredSummarizationCompactionStrategy(
            max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.8, fallback_fraction=0.8
        )


async def test_a_record_that_does_not_free_enough_still_falls_back() -> None:
    """The groups after the record are untouched by design and can exceed the ceiling alone."""
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=500, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.9
    )
    # The record sits early, so most of the bulk is behind it and survives phase 2.
    messages = _conversation(tool_turns=2, record="CODE-0")
    messages += _conversation(tool_turns=6)[3:]

    assert await strategy(messages) is True
    assert strategy.records_found == 1


class _Recorder:
    """Stands in for the rest of the pipeline, capturing the options a call went out with."""

    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages
        self.seen: list[dict[str, Any]] = []

    async def __call__(self) -> None:
        self.seen.append(dict(self.context.options or {}))
        self.context.messages = self.messages


async def _run(middleware: ToolResultRecallMiddleware, messages: list[Message]) -> dict[str, Any]:
    """Drive one middleware pass and return the options the call went out with."""
    from agent_framework import ChatContext

    context = ChatContext(client=None, messages=[Message(role="user", contents=["q"])], options={"temperature": 0})
    recorder = _Recorder(messages)
    recorder.context = context
    await middleware.process(context, recorder)
    return recorder.seen[0]


async def test_the_middleware_forces_the_call_and_sends_no_message() -> None:
    """Phase 1 must leave no trace in the prompt, only in the options.

    A message appended here carries no history provider's source tag, so per-service-call
    persistence would treat it as new input and store it -- and an instruction of ours would
    then appear in the conversation the application replays to its user.
    """
    middleware = ToolResultRecallMiddleware(max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.1)
    big = _conversation(tool_turns=8)

    first = await _run(middleware, big)
    assert "tool_choice" not in first, "nothing is known about history size before the first call"

    second = await _run(middleware, big)
    assert second["tool_choice"] == {"mode": "required", "required_function_name": RECALL_TOOL_NAME}
    assert middleware.forced_calls == 1
    # The prompt is untouched: no instruction, no extra turn.
    assert all("recall" not in str(m.contents[0]).lower() for m in [Message(role="user", contents=["q"])])


async def test_the_middleware_stops_once_a_record_exists() -> None:
    """Forcing a second record would re-drop what the first already covered."""
    middleware = ToolResultRecallMiddleware(max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.1)

    await _run(middleware, _conversation(tool_turns=8))
    await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))
    after = await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))

    assert "tool_choice" not in after
    assert find_record_index(_conversation(tool_turns=8, record="CODE-0")) is not None


async def test_the_middleware_leaves_small_conversations_alone() -> None:
    """Below the trigger there is nothing to record and nothing to drop."""
    middleware = ToolResultRecallMiddleware(max_input_tokens=10_000_000, tokenizer=TOKENIZER)

    await _run(middleware, _conversation(tool_turns=8))
    after = await _run(middleware, _conversation(tool_turns=8))

    assert "tool_choice" not in after
    assert middleware.forced_calls == 0


async def test_a_single_record_is_attributed_exactly_once() -> None:
    """The transition must be tracked on the instance, not re-read from each prompt.

    Before the pipeline runs, context.messages holds only the new turn, so a pre-call check
    reports "no record" every time and every later call counts as another one. That produced
    RECVOLUNTEERED:18 for a single record in a live run, which turned an attribution into
    noise at exactly the moment it was needed.
    """
    middleware = ToolResultRecallMiddleware(max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.1)
    with_record = _conversation(tool_turns=8, record="CODE-0")

    for _ in range(5):
        await _run(middleware, with_record)

    assert middleware.records_forced + middleware.records_volunteered == 1
