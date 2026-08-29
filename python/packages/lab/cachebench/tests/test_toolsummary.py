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
    MAX_REQUESTS,
    RECALL_INSTRUCTION,
    RECALL_TOOL_NAME,
    ToolResultAnchoredSummarizationCompactionStrategy,
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


def _strategy(**kwargs: Any) -> ToolResultAnchoredSummarizationCompactionStrategy:
    return ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=1_000, tokenizer=TOKENIZER, **kwargs)


async def test_phase_one_appends_the_request_and_drops_nothing() -> None:
    """Nothing may be removed before a record exists, or the values are simply gone."""
    strategy = _strategy()
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert RECALL_INSTRUCTION in rendered
    assert "CODE-0" in rendered, "phase 1 must not drop anything"
    assert strategy.requests_made == 1
    assert strategy.records_found == 0


async def test_the_request_is_appended_not_inserted() -> None:
    """Appending leaves the cached prefix intact; inserting re-bills everything after it."""
    strategy = _strategy()
    messages = _conversation(tool_turns=8)
    before = list(messages)

    await strategy(messages)

    assert messages[: len(before)] == before
    assert RECALL_INSTRUCTION in str(messages[-1].contents[0])


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


async def test_the_request_is_not_repeated_forever() -> None:
    """A model that will not call the tool would otherwise grow every prompt for the whole run."""
    strategy = _strategy()

    for _ in range(MAX_REQUESTS):
        assert await strategy(_conversation(tool_turns=8)) is True
    assert await strategy(_conversation(tool_turns=8)) is False
    assert strategy.requests_made == MAX_REQUESTS


async def test_nothing_happens_below_the_trigger() -> None:
    """A record that is not needed costs an agent turn and buys nothing."""
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=10_000_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert strategy.requests_made == 0


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
