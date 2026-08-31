# Copyright (c) Microsoft. All rights reserved.

"""Tests for the record-then-drop strategy, its tool, and the middleware that forces it.

Two phases, and the tests separate them: phase 1 must ask without inserting anything into the
cached prefix, and phase 2 must act only on a record the *provider* issued -- never on one the
client invented, which is unsafe on routes that track tool calls server-side.

The tool's own text is tested here as well, because the middleware sends no message: the
description and the ``values`` guidance are the entire prompt for the record, so a clause
lost from them is a class of content silently dropped, with nothing else in the design to
catch it.
"""

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message
from agent_framework._compaction import project_included_messages
from agent_framework_lab_cachebench.compaction._toolsummary import (
    DEFAULT_RECORD_MAX_TOKENS,
    DEFAULT_RECORD_TARGET_TOKENS,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    RecallGate,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    find_record_index,
    make_recall_tool,
)

TOKENIZER = CharacterEstimatorTokenizer()


#: Every arming the middleware performs, so a test can assert it happened exactly when the
#: tool was pinned and never otherwise.
_armings: list[int] = []


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
                contents=[{"type": "function_result", "call_id": "rec", "result": f"{RECORD_MARKER} {record}"}],
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

    def __init__(self, messages: list[Message], finish_reason: str | None = None) -> None:
        self.messages = messages
        self.finish_reason = finish_reason
        self.seen: list[dict[str, Any]] = []

    async def __call__(self) -> None:
        from agent_framework import ChatResponse

        self.seen.append(dict(self.context.options or {}))
        self.context.messages = self.messages
        # What the provider says about why it stopped, which is the only thing that separates
        # a record the model chose to keep short from one it was cut off in the middle of.
        self.context.result = ChatResponse(
            messages=Message(role="assistant", contents=["ok"]), finish_reason=self.finish_reason
        )


async def _run(
    middleware: ToolResultRecallMiddleware, messages: list[Message], finish_reason: str | None = None
) -> dict[str, Any]:
    """Drive one middleware pass and return the options the call went out with."""
    from agent_framework import ChatContext

    context = ChatContext(client=None, messages=[Message(role="user", contents=["q"])], options={"temperature": 0})
    recorder = _Recorder(messages, finish_reason)
    recorder.context = context
    await middleware.process(context, recorder)
    return recorder.seen[0]


async def test_the_middleware_forces_the_call_and_sends_no_message() -> None:
    """Phase 1 must leave no trace in the prompt, only in the options.

    A message appended here carries no history provider's source tag, so per-service-call
    persistence would treat it as new input and store it -- and an instruction of ours would
    then appear in the conversation the application replays to its user.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    first = await _run(middleware, big)
    assert "tool_choice" not in first, "nothing is known about history size before the first call"

    second = await _run(middleware, big)
    assert second["tool_choice"] == {"mode": "required", "required_function_name": RECALL_TOOL_NAME}
    assert len(_armings) == 1, "the tool is armed exactly when it is pinned, and never otherwise"
    assert middleware.forced_calls == 1
    # The prompt is untouched: no instruction, no extra turn.
    assert all("recall" not in str(m.contents[0]).lower() for m in [Message(role="user", contents=["q"])])


async def test_forgetting_the_pending_decision_stops_the_next_call_forcing() -> None:
    """Restoring the snapshot has to clear the middleware's pending decision too.

    The decision to force a record is taken on one call and applied to the next, so it belongs
    to the conversation rather than to the middleware. Left in place, a decision taken while
    the conversation was being seeded fires on the first question asked of the snapshot and on
    none of the others -- one probe carrying a prompt the rest do not, which is exactly the
    difference between probes the snapshot design exists to remove.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    # The first call arms nothing but leaves the middleware intending to force the next one.
    await _run(middleware, big)
    middleware.forget_pending()
    after = await _run(middleware, big)

    assert "tool_choice" not in after
    assert not _armings, "the recall tool was armed on a call the snapshot had reset"
    assert middleware.forced_calls == 0


async def test_the_middleware_stops_once_a_record_exists() -> None:
    """Forcing a second record would re-drop what the first already covered."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )

    await _run(middleware, _conversation(tool_turns=8))
    await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))
    after = await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))

    assert "tool_choice" not in after
    assert find_record_index(_conversation(tool_turns=8, record="CODE-0")) is not None


async def test_the_middleware_leaves_small_conversations_alone() -> None:
    """Below the trigger there is nothing to record and nothing to drop."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=10_000_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1)
    )

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
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    with_record = _conversation(tool_turns=8, record="CODE-0")

    for _ in range(5):
        await _run(middleware, with_record)

    assert middleware.records_forced + middleware.records_volunteered == 1


# region the tool, which is the whole of the prompt


def _recall_exchange(result: str) -> list[Message]:
    """Return a matched recall call and result carrying ``result``."""
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": "r", "name": RECALL_TOOL_NAME, "arguments": "{}"}],
        ),
        Message(role="tool", contents=[{"type": "function_result", "call_id": "r", "result": result}]),
    ]


def test_the_recall_tool_records_only_while_armed() -> None:
    """The tool cannot be hidden, so it has to be inert instead.

    MAF requires it to be registered with the agent: the function-invocation layer wraps the
    middleware layer and builds its tool map first, so a tool supplied through per-call
    options reaches the model but never the executor, and the model's call goes unanswered.
    A registered tool is advertised on every request, and this one was called uninvited on
    every unpinned follow-up call. Permission is therefore separated from visibility.
    """
    gate = RecallGate()
    tool = make_recall_tool(gate)

    uninvited = tool("AA-1")
    gate.arm()
    armed = tool("AA-1")
    reused = tool("AA-2")

    assert RECORD_MARKER not in uninvited
    assert RECORD_MARKER in armed
    assert RECORD_MARKER not in reused, "one arming cannot licence a second record"
    # The scorer must agree, or an uninvited call would look like a record and the strategy
    # would drop results that nothing had preserved.
    assert find_record_index(_recall_exchange(uninvited)) is None
    assert find_record_index(_recall_exchange(armed)) is not None


@pytest.mark.parametrize(
    "clause",
    [
        pytest.param("grouped by the tool that produced it", id="attribution"),
        pytest.param("Quote verbatim any value that cannot be reconstructed or guessed", id="unreconstructable"),
        pytest.param("Keep findings and conclusions as they were stated", id="findings"),
        pytest.param("Carry over any summary a tool already produced as it stands", id="existing-summary"),
        pytest.param("Summarise the remaining content briefly", id="the-rest"),
        pytest.param("keep exactness over brevity", id="tie-break"),
    ],
)
def test_the_tool_asks_for_every_kind_of_content_a_result_can_hold(clause: str) -> None:
    """The four instructions partition the content, and a missing one is a silent hole.

    The description and the ``values`` guidance are the entire prompt: the middleware sends no
    message, because one appended there would be persisted into the caller's own conversation.
    So whatever these do not name is content the model may drop without anything noticing, and
    the earlier text named only "identifiers and values seen in earlier tool results" -- fitted
    to this benchmark's hex codes, and blind to the prose, findings and conclusions that are
    most of what a real tool returns.
    """
    assert clause in (make_recall_tool().__doc__ or "")


def test_the_tool_says_what_the_call_is_for_before_it_says_what_to_pass() -> None:
    """The description has to stand on its own: a model reading the schema sees it first."""
    doc = make_recall_tool().__doc__ or ""

    assert doc.startswith("Record what must survive from earlier tool results")
    assert "after those results are removed from the conversation to save space" in doc


def test_the_target_length_is_stated_when_one_is_given() -> None:
    """The description is the only channel that makes the model plan for a size.

    A ``max_tokens`` cap cannot do it: a model does not shorten to fit one, it writes until it
    is cut, and on a tool call the cut lands inside the arguments JSON -- so a cap set where
    the record should end produces no record rather than a shorter one. The middleware cannot
    send an instruction message either, which leaves this text.
    """
    stated = make_recall_tool(target_tokens=1_500).__doc__ or ""
    default = make_recall_tool().__doc__ or ""
    silent = make_recall_tool(target_tokens=None).__doc__ or ""

    assert "1,500 tokens" in stated
    assert f"{DEFAULT_RECORD_TARGET_TOKENS:,} tokens" in default
    assert "Aim for about" not in silent
    # Stating a length must not cost a clause: the two bounds are independent instructions.
    assert "Quote verbatim any value that cannot be reconstructed or guessed" in silent


def test_the_target_sits_well_under_the_cap() -> None:
    """Overshooting the stated length must not be the same event as being cut off.

    They measure different things -- one is what the model aims for, the other is what the
    provider enforces -- and a default pair close together would make every slightly long
    record a truncated one.
    """
    assert DEFAULT_RECORD_TARGET_TOKENS < DEFAULT_RECORD_MAX_TOKENS


# region bounding the record


async def test_the_cap_is_set_on_the_forced_call_and_on_no_other() -> None:
    """Every other call needs the run's own cap; only this one is asked to write a record.

    Left to inherit ``--answer-max-tokens``, the single call instructed to summarise every
    earlier tool result is the one call in the run with no bound of its own.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        trigger_fraction=0.1,
        record_max_tokens=777,
    )
    big = _conversation(tool_turns=8)
    recorded = _conversation(tool_turns=8, record="CODE-0")

    calls = [
        # Nothing is known about the history before the first call, so it cannot be forced.
        await _run(middleware, big),
        await _run(middleware, big),
        # The middleware keeps asking until a record exists, so the record has to arrive
        # before an unforced call can happen again.
        await _run(middleware, recorded),
        await _run(middleware, recorded),
    ]

    assert [("tool_choice" in options) for options in calls] == [False, True, True, False]
    for options in calls:
        assert ("max_tokens" in options) is ("tool_choice" in options), options
        assert options.get("max_tokens", 777) == 777
        # The call's other options survive: this replaces the option set, it does not discard it.
        assert options["temperature"] == 0


async def test_no_cap_leaves_the_runs_own_ceiling_in_place() -> None:
    """None has to mean what it meant before the parameter existed, or a run cannot opt out."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        trigger_fraction=0.1,
        record_max_tokens=None,
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big)
    forced = await _run(middleware, big)

    assert "tool_choice" in forced
    assert "max_tokens" not in forced


def test_a_cap_that_cannot_hold_a_record_is_refused() -> None:
    """Zero is not "no cap": it is a call that can produce nothing, which None expresses."""
    with pytest.raises(ValueError, match="record_max_tokens"):
        ToolResultRecallMiddleware(max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: None, record_max_tokens=0)


async def test_a_record_cut_short_is_counted_rather_than_read_as_complete() -> None:
    """A partial record is the one failure this design does not otherwise show.

    A tool call cut mid-arguments produces no record at all, which is loud: the strategy waits,
    falls back, and the row carries FALLBACK. A call cut just after a closing brace yields a
    record that parses and looks whole -- and the strategy then drops every tool group behind
    something covering only part of them, so the loss is scored as compaction damage.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big, finish_reason="tool_calls")
    await _run(middleware, big, finish_reason="length")

    assert middleware.forced_calls == 1
    assert middleware.records_truncated == 1


async def test_only_the_forced_call_can_truncate_a_record() -> None:
    """Every call in a long run can hit its own ceiling; only one of them writes the record.

    Counting the rest would put an ordinary long answer in the column that says the record is
    partial, which is the reading the count exists to prevent.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    # Every call stops at its ceiling, forced or not.
    for _ in range(4):
        await _run(middleware, big, finish_reason="length")

    assert middleware.forced_calls > 0
    assert middleware.records_truncated == middleware.forced_calls


async def test_a_forced_call_that_finished_is_not_reported_as_truncated() -> None:
    """A short record is a choice the model is allowed to make, and is not a cut one."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big, finish_reason="stop")
    await _run(middleware, big, finish_reason="tool_calls")

    assert middleware.forced_calls == 1
    assert middleware.records_truncated == 0
