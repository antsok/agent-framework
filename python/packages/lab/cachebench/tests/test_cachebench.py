# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the compaction / prompt-cache benchmark.

Everything here runs offline. The provider call is replaced by a stub so the replay loop,
the prefix oracle, and the aggregation can be verified without spending anything.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from agent_framework import CharacterEstimatorTokenizer, Content, Message, SlidingWindowStrategy
from agent_framework_lab_cachebench import (
    DEFAULT_SYSTEM_TOKENS,
    CallOutcome,
    CellKey,
    ProviderCaller,
    ProviderRuntime,
    build_preset,
    build_strategy,
    build_transcript,
    common_message_prefix,
    parse_provider_selector,
    percentile,
    prompt_cache_key,
    prompt_cache_key_options,
    render_summary_table,
    resolve_context_window,
    run_cell,
    serialize_message,
    strategy_names,
    summarize_cell,
    write_records_jsonl,
    write_summary_csv,
)
from agent_framework_lab_cachebench._runner import _unsupported_option
from agent_framework_lab_cachebench._strategies import StrategyOptions
from agent_framework_lab_cachebench._types import TurnRecord

TOKENIZER = CharacterEstimatorTokenizer()


class StubCaller:
    """Records what it was asked to send and reports fixed usage."""

    def __init__(self, *, cached_tokens: int | None = None) -> None:
        self.calls: list[int] = []
        self.cached_tokens = cached_tokens

    async def __call__(self, messages: Sequence[Message]) -> CallOutcome:
        self.calls.append(len(messages))
        return CallOutcome(
            latency_ms=1.0,
            input_tokens=100,
            cached_tokens=self.cached_tokens,
            output_tokens=5,
        )


def _cell(strategy: str = "none") -> CellKey:
    return CellKey(provider="stub", model="stub-model", transcript="small", strategy=strategy, repeat=1)


# region transcripts


def test_transcript_is_deterministic_for_a_given_salt() -> None:
    first = build_transcript(name="t", turns=5, salt="abc")
    second = build_transcript(name="t", turns=5, salt="abc")
    assert serialize_message(first.system) == serialize_message(second.system)
    for left, right in zip(first.turns, second.turns):
        assert [serialize_message(m) for m in left.request] == [serialize_message(m) for m in right.request]
        assert [serialize_message(m) for m in left.reply] == [serialize_message(m) for m in right.reply]


def test_salt_isolates_the_cache_namespace() -> None:
    first = build_transcript(name="t", turns=2, salt="alpha")
    second = build_transcript(name="t", turns=2, salt="beta")
    left = first.system.contents[0].text or ""
    right = second.system.contents[0].text or ""
    # The salt must land at the very front, since provider caches match on exact prefixes.
    assert left != right
    assert left.startswith("[cachebench:")


def test_salt_does_not_change_the_transcript_size() -> None:
    # Salts of very different lengths must still yield identical token counts: the salt
    # feeds the auto context window, which sets the compaction budgets. A length-sensitive
    # salt makes a strategy retain different amounts for different providers.
    short = build_transcript(name="t", turns=6, salt="a")
    long = build_transcript(name="t", turns=6, salt="a-very-much-longer-cell-salt-string")
    assert len(short.system.contents[0].text or "") == len(long.system.contents[0].text or "")
    assert short.approx_final_prompt_tokens == long.approx_final_prompt_tokens


def test_transcript_emits_tool_call_groups() -> None:
    transcript = build_transcript(name="t", turns=6, salt="s", tool_call_every=3)
    tool_turns = [
        index
        for index, turn in enumerate(transcript.turns, start=1)
        if any(content.type == "function_call" for message in turn.reply for content in message.contents)
    ]
    assert tool_turns == [3, 6]


def test_transcript_messages_differ_between_positions() -> None:
    transcript = build_transcript(name="t", turns=8, salt="s", tool_call_every=0)
    bodies = {serialize_message(turn.request[0]) for turn in transcript.turns}
    # Identical message bodies would let unrelated prefixes match and inflate reuse.
    assert len(bodies) == 8


def test_system_anchor_clears_the_provider_cache_floor() -> None:
    transcript = build_transcript(name="t", turns=1, salt="s")
    assert TOKENIZER.count_tokens(transcript.system.contents[0].text or "") >= 1024


def test_build_preset_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        build_preset("enormous", salt="s")


def test_build_transcript_rejects_zero_turns() -> None:
    with pytest.raises(ValueError, match="turns must be greater than 0"):
        build_transcript(name="t", turns=0, salt="s")


# region prefix oracle


def test_serialize_message_ignores_non_wire_fields() -> None:
    left = Message(role="user", contents=["hello"], message_id="id-one")
    right = Message(role="user", contents=["hello"], message_id="id-two")
    # Distinct message_ids must not make two otherwise identical messages look different,
    # because message_id is never sent to the provider. Counting it would report a broken
    # prefix on every turn and make every strategy look equally cache-hostile.
    assert left.message_id != right.message_id
    assert serialize_message(left) == serialize_message(right)


def test_serialize_message_distinguishes_roles_and_contents() -> None:
    assert serialize_message(Message(role="user", contents=["a"])) != serialize_message(
        Message(role="assistant", contents=["a"])
    )
    assert serialize_message(Message(role="user", contents=["a"])) != serialize_message(
        Message(role="user", contents=["b"])
    )


def test_serialize_message_covers_tool_content() -> None:
    call = Message(role="assistant", contents=[Content.from_function_call(call_id="c1", name="f", arguments="{}")])
    other = Message(role="assistant", contents=[Content.from_function_call(call_id="c1", name="g", arguments="{}")])
    assert serialize_message(call) != serialize_message(other)


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ([], ["a"], 0),
        (["a"], ["a", "b"], 1),
        (["a", "b"], ["a", "b", "c"], 2),
        (["a", "b"], ["a", "x", "c"], 1),
        (["a", "b", "c"], ["a", "b"], 2),
        (["x"], ["y"], 0),
    ],
)
def test_common_message_prefix(previous: list[str], current: list[str], expected: int) -> None:
    assert common_message_prefix(previous, current) == expected


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [
        ([], 0.5, None),
        ([4.0], 0.5, 4.0),
        ([1.0, 2.0, 3.0], 0.5, 2.0),
        ([3.0, 1.0, 2.0], 0.0, 1.0),
        ([1.0, 2.0, 3.0], 1.0, 3.0),
    ],
)
def test_percentile(values: list[float], fraction: float, expected: float | None) -> None:
    assert percentile(values, fraction) == expected


# region replay loop


async def test_uncompacted_replay_never_breaks_the_prefix() -> None:
    transcript = build_transcript(name="t", turns=6, salt="s")
    caller = StubCaller()
    records = await run_cell(
        cell=_cell("none"),
        transcript=transcript,
        strategy=None,
        tokenizer=TOKENIZER,
        caller=caller,
    )
    assert len(records) == 6
    # Without compaction each prompt is a pure extension of the last, so the whole
    # previous prompt stays cacheable.
    assert [record.prefix_broken for record in records] == [False] * 6
    assert records[0].reusable_prefix_tokens_local == 0
    assert all(record.reusable_prefix_tokens_local > 0 for record in records[1:])
    assert caller.calls == sorted(caller.calls), "prompt should grow monotonically"


async def test_compaction_breaks_the_prefix() -> None:
    transcript = build_transcript(name="t", turns=8, salt="s")
    records = await run_cell(
        cell=_cell("sliding_window"),
        transcript=transcript,
        strategy=SlidingWindowStrategy(keep_last_groups=2),
        tokenizer=TOKENIZER,
        caller=StubCaller(),
    )
    # This is the effect the whole benchmark exists to quantify: dropping older groups
    # invalidates the cached prefix.
    assert any(record.prefix_broken for record in records)
    assert max(record.history_messages for record in records) > max(record.sent_messages for record in records)


async def test_on_record_streams_results() -> None:
    seen: list[TurnRecord] = []
    await run_cell(
        cell=_cell(),
        transcript=build_transcript(name="t", turns=3, salt="s"),
        strategy=None,
        tokenizer=TOKENIZER,
        caller=StubCaller(),
        on_record=seen.append,
    )
    assert [record.turn for record in seen] == [1, 2, 3]


async def test_failed_turns_are_recorded_not_raised() -> None:
    class FailingCaller:
        async def __call__(self, messages: Sequence[Message]) -> CallOutcome:
            return CallOutcome(latency_ms=0.0, error="boom")

    records = await run_cell(
        cell=_cell(),
        transcript=build_transcript(name="t", turns=2, salt="s"),
        strategy=None,
        tokenizer=TOKENIZER,
        caller=FailingCaller(),
    )
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=False)
    assert summary.errors == 2
    assert summary.total_input_tokens == 0


# region aggregation


def test_summary_ratios() -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=index,
            history_messages=10,
            sent_messages=8,
            sent_tokens_local=1000,
            reusable_prefix_tokens_local=800,
            prefix_broken=index == 2,
            input_tokens=1000,
            cached_tokens=400,
            output_tokens=10,
            latency_ms=float(index),
        )
        for index in (1, 2)
    ]
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=True)
    assert summary.total_input_tokens == 2000
    assert summary.fresh_input_tokens == 1200
    assert summary.cache_hit_ratio == pytest.approx(0.4)
    assert summary.local_reusable_ratio == pytest.approx(0.8)
    # hit 0.4 over reuse 0.8 — a quotient of fractions, so the local estimator's
    # inflated token counts cancel instead of halving the result.
    assert summary.cache_realization == pytest.approx(0.5)
    assert summary.prefix_breaks == 1
    # 1200 fresh tokens at full price plus 800 cached tokens at a quarter price.
    assert summary.effective_input_tokens(0.25) == pytest.approx(1400)


def test_cached_tokens_are_clamped_to_the_input_they_are_part_of() -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=1,
            history_messages=4,
            sent_messages=4,
            sent_tokens_local=500,
            reusable_prefix_tokens_local=300,
            prefix_broken=False,
            input_tokens=1000,
            cached_tokens=4000,  # upstream inconsistency: more cached than prompt
            latency_ms=1.0,
        )
    ]
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=True)
    assert summary.total_cached_tokens == 1000
    assert summary.cache_hit_ratio == pytest.approx(1.0)
    assert summary.fresh_input_tokens == 0


def test_hit_ratio_suppressed_when_a_turn_omits_the_input_count() -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=1,
            history_messages=4,
            sent_messages=4,
            sent_tokens_local=500,
            reusable_prefix_tokens_local=300,
            prefix_broken=False,
            input_tokens=None,  # provider dropped it on a cache hit
            cached_tokens=800,
            latency_ms=1.0,
        )
    ]
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=True)
    assert summary.turns_missing_input == 1
    # Dividing 800 cached by a denominator the provider never sent would invent a number.
    assert summary.cache_hit_ratio is None


def test_missing_cache_reporting_is_not_a_zero_hit_rate() -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=1,
            history_messages=4,
            sent_messages=4,
            sent_tokens_local=500,
            reusable_prefix_tokens_local=300,
            prefix_broken=False,
            input_tokens=500,
            cached_tokens=None,
            latency_ms=2.0,
        )
    ]
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=False)
    # A provider that reports nothing must not be shown as a 0% hit rate.
    assert summary.cache_hit_ratio is None
    assert summary.cache_realization is None
    assert summary.local_reusable_ratio == pytest.approx(0.6)


# region strategies


def test_every_registered_strategy_builds() -> None:
    options = StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=8000, max_output_tokens=512)
    for name in strategy_names():
        if name == "summarization":
            continue
        strategy = build_strategy(name, options)
        assert (strategy is None) == (name == "none")


def test_summarization_requires_a_client() -> None:
    options = StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=8000, max_output_tokens=512)
    with pytest.raises(ValueError, match="summarizer client"):
        build_strategy("summarization", options)


def test_unknown_strategy_rejected() -> None:
    options = StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=8000, max_output_tokens=512)
    with pytest.raises(KeyError):
        build_strategy("does_not_exist", options)


def test_resolve_context_window_scales_to_the_transcript() -> None:
    # An auto window must sit below the transcript size or compaction never fires and the
    # benchmark measures nothing.
    assert resolve_context_window(40_000) < 40_000
    assert resolve_context_window(40_000) == 24_000
    assert resolve_context_window(40_000, override=9_000) == 9_000


def test_auto_window_floor_keeps_the_system_anchor_inside_the_eviction_budget() -> None:
    # ContextWindowCompactionStrategy evicts at half the input budget. If the anchor does
    # not fit under that half, compaction falls back to evicting the anchor itself and the
    # stable cacheable prefix disappears.
    window = resolve_context_window(100, max_output_tokens=512)
    eviction_budget = (window - 512) * 0.5
    assert eviction_budget > DEFAULT_SYSTEM_TOKENS


# region unsupported request options


def test_unsupported_option_is_extracted_from_a_wrapped_provider_error() -> None:
    # Verbatim from Foundry/gpt-5.6-luna. The framework wraps provider errors in an
    # exception whose args are a tuple, so str() repr's the payload and the quotes around
    # the parameter name arrive backslash-escaped — which silently defeated the first
    # version of this matcher and cost a full 120-call re-run.
    wrapped = Exception(
        "<class 'FoundryChatClient'> service failed to complete the prompt: Error code: 400 - "
        "{'error': {'message': \"Unsupported parameter: 'temperature' is not supported with "
        "this model.\", 'type': 'invalid_request_error', 'param': 'temperature'}}"
    )
    assert _unsupported_option(wrapped) == "temperature"


def test_unsupported_option_ignores_unrelated_failures() -> None:
    assert _unsupported_option(Exception("Error code: 429 - rate limit exceeded")) is None
    assert _unsupported_option(Exception("Error code: 500 - internal")) is None


async def test_caller_drops_a_rejected_option_and_retries() -> None:
    class RejectsTemperature:
        def __init__(self) -> None:
            self.seen: list[dict[str, object]] = []

        async def get_response(self, messages: object, *, options: dict[str, object]) -> object:
            self.seen.append(dict(options))
            if "temperature" in options:
                raise ValueError("Unsupported parameter: 'temperature' is not supported with this model.")
            return type("R", (), {"usage_details": {"input_token_count": 7}, "text": "ok"})()

    client = RejectsTemperature()
    runtime = ProviderRuntime(client=client, model="m", options={"max_tokens": 16, "temperature": 0.0})
    caller = ProviderCaller(runtime)
    outcome = await caller(())
    assert outcome.error is None
    assert outcome.input_tokens == 7
    assert "temperature" in client.seen[0] and "temperature" not in client.seen[1]
    # The retry must not mutate the shared runtime options other cells still use.
    assert "temperature" in runtime.options


# region provider selectors


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("openrouter", ("openrouter", None)),
        ("openrouter:openai/gpt-5.4-mini", ("openrouter", "openai/gpt-5.4-mini")),
        # Model ids legitimately contain colons, so only the first one separates.
        ("ollama:glm-5.2:cloud", ("ollama", "glm-5.2:cloud")),
        ("openrouter:z-ai/glm-5.2:free", ("openrouter", "z-ai/glm-5.2:free")),
        ("mistral:", ("mistral", None)),
    ],
)
def test_parse_provider_selector(selector: str, expected: tuple[str, str | None]) -> None:
    assert parse_provider_selector(selector) == expected


# region prompt cache key


def test_automatic_cache_providers_need_opting_in() -> None:
    # Measured: Mistral and Azure both cache without a key, so none of them send one by
    # default; an older deployment can reject the unknown field.
    for provider in ("mistral", "azure", "openrouter"):
        assert prompt_cache_key_options(provider, "salt-a") == {}


def test_mistral_takes_the_key_as_a_declared_option_not_extra_body() -> None:
    # MistralChatOptions declares prompt_cache_key, so it travels as a plain option;
    # OpenAI-SDK routes need it smuggled through extra_body instead.
    assert prompt_cache_key_options("mistral", "salt-a", enable_optional=True)["prompt_cache_key"].startswith(
        "cachebench-"
    )
    azure = prompt_cache_key_options("azure", "salt-a", enable_optional=True)
    assert azure["extra_body"]["prompt_cache_key"].startswith("cachebench-")


def test_providers_without_the_field_never_get_one() -> None:
    assert prompt_cache_key_options("ollama", "salt-a", enable_optional=True) == {}
    assert prompt_cache_key_options("foundry", "salt-a", enable_optional=True) == {}


def test_cache_key_is_stable_per_salt_and_distinct_across_cells() -> None:
    assert prompt_cache_key("salt-a") == prompt_cache_key("salt-a")
    assert prompt_cache_key("salt-a") != prompt_cache_key("salt-b")


def test_cell_options_do_not_clobber_provider_extra_body() -> None:
    runtime = ProviderRuntime(
        client=object(),
        model="m",
        options={"max_tokens": 16, "extra_body": {"provider": {"order": ["openai"]}}},
    )
    caller = ProviderCaller(runtime, extra_options={"extra_body": {"prompt_cache_key": "k"}})
    # Losing the routing pin would silently reintroduce the confound it exists to remove.
    assert caller.options["extra_body"] == {"provider": {"order": ["openai"]}, "prompt_cache_key": "k"}
    assert caller.options["max_tokens"] == 16
    assert runtime.options["extra_body"] == {"provider": {"order": ["openai"]}}


# region reporting


def test_render_marks_unreported_cache_stats_as_not_available() -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=1,
            history_messages=2,
            sent_messages=2,
            sent_tokens_local=100,
            reusable_prefix_tokens_local=0,
            prefix_broken=False,
            input_tokens=100,
            latency_ms=1.0,
        )
    ]
    table = render_summary_table([summarize_cell(records, cell=_cell(), reports_cache_tokens=False)])
    assert "n/a" in table
    assert "hit%" in table


def test_render_empty() -> None:
    assert render_summary_table([]) == "No results."


def test_outputs_round_trip(tmp_path: Path) -> None:
    records = [
        TurnRecord(
            cell=_cell(),
            turn=1,
            history_messages=2,
            sent_messages=2,
            sent_tokens_local=100,
            reusable_prefix_tokens_local=50,
            prefix_broken=False,
            input_tokens=100,
            cached_tokens=50,
            latency_ms=1.0,
        )
    ]
    records_path = tmp_path / "nested" / "records.jsonl"
    assert write_records_jsonl(records_path, records) == 1
    payload = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["provider"] == "stub"
    assert payload["cached_tokens"] == 50

    summary_path = tmp_path / "nested" / "summary.csv"
    summary = summarize_cell(records, cell=_cell(), reports_cache_tokens=True)
    assert write_summary_csv(summary_path, [summary], cache_read_ratio=0.25) == 1
    header = summary_path.read_text(encoding="utf-8").splitlines()[0]
    assert "cache_hit_ratio" in header
    assert "effective_input_tokens" in header
