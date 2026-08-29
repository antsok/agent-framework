# Copyright (c) Microsoft. All rights reserved.

"""Tests for the extract-then-drop strategy.

The property that decides whether it is worth anything is not that it shrinks the prompt.
It is that the extract it produces is generated once and re-emitted byte-identically
afterwards. A strategy that re-derives its summary on every model call mutates the prefix
every turn and forfeits the cache, which is what the framework's own SummarizationStrategy
measured at 0% to 59% hit rate against a 93% control.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message
from agent_framework._compaction import project_included_messages
from agent_framework_lab_cachebench._toolsummary import ToolResultAnchoredSummarizationCompactionStrategy

TOKENIZER = CharacterEstimatorTokenizer()


class StubExtractor:
    """A client whose extract text changes on every call, so re-derivation is detectable."""

    def __init__(self, *, fail: bool = False, text: str | None = None) -> None:
        self.calls = 0
        self.fail = fail
        self.text = text
        self.seen: list[str] = []

    async def get_response(self, messages: list[Message], **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("extraction unavailable")
        self.seen.append("".join(str(c) for m in messages for c in m.contents))

        class _Response:
            text = self.text if self.text is not None else f"EXTRACT-{self.calls}"

        return _Response()


def _conversation(tool_turns: int, payload_chars: int = 8_000) -> list[Message]:
    """Return a conversation with a stable head and ``tool_turns`` tool groups."""
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
            Message(role="assistant", contents=[f"Looked up {index}."], message_id=f"a_txt_{index}"),
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


def _strategy(client: StubExtractor, **kwargs: Any) -> ToolResultAnchoredSummarizationCompactionStrategy:
    return ToolResultAnchoredSummarizationCompactionStrategy(
        client=client, max_input_tokens=1_000, tokenizer=TOKENIZER, **kwargs
    )


async def test_the_extract_replaces_the_results_it_covers() -> None:
    """The bulk goes, the values stay, and the model is told the swap happened."""
    client = StubExtractor()
    strategy = _strategy(client)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "EXTRACT-1" in rendered
    assert "removed from this conversation" in rendered
    # The head anchor survives, and so does the most recent result: only the band is dropped.
    assert "EU-WEST-1" in rendered
    assert "CODE-7" in rendered
    assert "CODE-0" not in rendered


async def test_the_extract_is_generated_once_and_reused_verbatim() -> None:
    """The whole design rests on this: one call, then the same bytes forever.

    The stub returns different text on every call, so a strategy that re-derived its extract
    would show EXTRACT-2 on the second pass and mutate the prefix. Caching is not an
    optimisation here, it is the reason the strategy can be cheaper than not compacting.
    """
    client = StubExtractor()
    strategy = _strategy(client)

    first = _conversation(tool_turns=8)
    await strategy(first)
    later = _conversation(tool_turns=10)
    await strategy(later)

    # Extracts accumulate rather than being rewritten: the first is still there, verbatim,
    # alongside a second covering the groups that newly aged into the band. That is the
    # property that matters -- an extract, once made, is never regenerated, so the prefix
    # holding it stays byte-identical while the conversation grows past it.
    assert "EXTRACT-1" in _rendered(later)
    assert "EXTRACT-2" in _rendered(later)
    assert client.calls == 2

    # Growing again must not touch either of them.
    later_still = _conversation(tool_turns=12)
    await strategy(later_still)
    rendered = _rendered(later_still)
    assert "EXTRACT-1" in rendered and "EXTRACT-2" in rendered
    assert client.calls == 3, "one new extract for the newly aged groups, none re-derived"


async def test_a_failed_extraction_keeps_the_tool_results() -> None:
    """Losing the results *and* the extract is strictly worse than not compacting."""
    client = StubExtractor(fail=True)
    strategy = _strategy(client)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert "CODE-0" in _rendered(messages)
    assert strategy.failures == 1


async def test_an_empty_extraction_counts_as_a_failure() -> None:
    """A blank extract would drop the results and replace them with nothing at all."""
    client = StubExtractor(text="   ")
    strategy = _strategy(client)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert "CODE-0" in _rendered(messages)
    assert strategy.failures == 1


async def test_nothing_happens_below_the_trigger() -> None:
    """An extract that is not needed is a model call spent and a prefix mutated for nothing."""
    client = StubExtractor()
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        client=client, max_input_tokens=10_000_000, tokenizer=TOKENIZER
    )
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert client.calls == 0


async def test_extracts_are_never_fed_back_into_later_extracts() -> None:
    """A summary of a summary loses a little more each round and is unbounded."""
    client = StubExtractor()
    strategy = _strategy(client)

    messages = _conversation(tool_turns=8)
    await strategy(messages)
    grown = _conversation(tool_turns=12)
    await strategy(grown)

    assert all("EXTRACT-" not in seen for seen in client.seen)


async def test_the_synthesised_result_has_a_matching_call() -> None:
    """A bare tool result with no request is malformed for most providers."""
    client = StubExtractor()
    strategy = _strategy(client)
    messages = _conversation(tool_turns=8)

    await strategy(messages)
    kept = project_included_messages(messages)

    call_ids = {c.call_id for m in kept for c in m.contents if c.type == "function_call"}
    result_ids = {c.call_id for m in kept for c in m.contents if c.type == "function_result"}
    assert result_ids <= call_ids


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
        ToolResultAnchoredSummarizationCompactionStrategy(client=StubExtractor(), tokenizer=TOKENIZER, **kwargs)
