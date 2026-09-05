# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the live-agent compaction runner.

Everything here runs offline against a stub chat client. The stub is composed from the same
layers a real provider client uses, so the agent's middleware pipeline, its history
persistence and the compaction hook all execute for real; only the network call is
replaced. That matters because the questions worth testing here are all about *ordering*
between those layers, and a hand-rolled mock that skipped them would answer none of them.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agent_framework import (
    Agent,
    BaseChatClient,
    CharacterEstimatorTokenizer,
    ChatMiddlewareLayer,
    ChatResponse,
    CompactionProvider,
    Content,
    ContextWindowCompactionStrategy,
    FunctionInvocationLayer,
    InMemoryHistoryProvider,
    Message,
    SlidingWindowStrategy,
    TokenBudgetComposedStrategy,
    ToolResultCompactionStrategy,
    TruncationStrategy,
    UsageDetails,
)
from agent_framework_lab_cachebench import (
    AGENT_KINDS,
    FillPlan,
    IdentifiedHistoryProvider,
    LiveOutcome,
    MeteredClient,
    ModelCall,
    ProbeOutcome,
    ProviderRuntime,
    StrategyOptions,
    UsageRecorder,
    build_live_agent,
    build_live_scenario,
    build_recall_scenario,
    build_strategy,
    make_lookup_tool,
    recommend,
    run_live,
    score_samples,
    unretrieved_facts,
    wants_client_side_history,
)
from agent_framework_lab_cachebench._advisor import ModelPricing
from agent_framework_lab_cachebench._live import (
    CONNECTION_ATTEMPTS,
    CONNECTION_BASE_DELAY,
    CONNECTION_MAX_DELAY,
    CONNECTION_MAX_WAIT,
    DEFAULT_COMBINED_REPEATS,
    RATE_LIMIT_ATTEMPTS,
    RATE_LIMIT_BASE_DELAY,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MAX_WAIT,
    RETRIEVAL_GUIDANCE,
    _strategy_notes,
    _turn_text,
    make_scope_tools,
    probe_count,
    recall_record_text,
    resolve_instructions,
    serialize_history,
    snapshot_state,
)
from agent_framework_lab_cachebench._live_cli import (
    CellStats,
    _accuracy_note,
    _aggregate,
    _build_or_exit,
    _control_message_gap,
    _cost,
    _coverage,
    _dump_record,
    _excluded_cells,
    _fill_note,
    _flags,
    _probe_spread,
    _progress,
    _render,
    _row,
    _seed_record,
    _seed_spread,
    _spread,
    _strategy_options,
    _summarizer_cost,
    _to_joint,
    build_parser,
    run_live_comparison,
)
from agent_framework_lab_cachebench._recall import COMBINED_SCOPE
from agent_framework_lab_cachebench._records import (
    SCHEMA_VERSION,
    CellParams,
    SeedRecord,
    append_seed_record,
    group_by_cell,
    read_seed_records,
)
from agent_framework_lab_cachebench.compaction import (
    DEFAULT_RECORD_MAX_TOKENS,
    DEFAULT_RECORD_TARGET_TOKENS,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    AnchoredCompactionStrategy,
    MinimumGainAnchoredCompactionStrategy,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    make_recall_tool,
)

TOKENIZER = CharacterEstimatorTokenizer()
PRICING = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)


def _cell_params(**overrides: Any) -> CellParams:
    """Return the cell parameters a record carries, shaped like the ones a live run writes."""
    defaults: dict[str, Any] = {
        "provider": "stub",
        "model": "stub-model",
        "agent_kind": "plain",
        "context_window": 60_000,
        "fill": 0.0,
        "probe_repeats": 3,
        "repeats": 1,
        "strategies": ("none",),
        "narration": "neutral",
        "fact_placement": "spread",
        "tool_result_tokens": 50,
        "filler_turns": 3,
        "filler_tokens": 50,
        "tool_turns": 6,
        "filler_tool_turns": 0,
        "markers_per_tool": 2,
        "price_input": PRICING.input_per_million,
        "price_cached": PRICING.cached_read_per_million,
        "price_output": PRICING.output_per_million,
    }
    return CellParams(**{**defaults, **overrides})


def _record(
    outcome: LiveOutcome,
    scenario: Any,
    *,
    strategy: str | None = None,
    cell: CellParams | None = None,
    seed: int = 1,
) -> SeedRecord:
    """Score one outcome into the record every table is built from.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Keyword Args:
        strategy: Name the row under, when the test drove the outcome with a different one.
        cell: Cell parameters, defaulting to the shared test cell.
        seed: 1-based seed index.

    Returns:
        The record.
    """
    record = _seed_record(outcome, scenario, PRICING, cell or _cell_params(), seed)
    return replace(record, strategy=strategy) if strategy is not None else record


class StubChatClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]):
    """A chat client composed from the real layers, answering from a script.

    Built the same way ``OpenAIChatClient`` is, so middleware, function invocation and the
    compaction hook all run. ``seen`` records how many messages each request carried, which
    is the ground truth every recorder assertion is checked against.
    """

    def __init__(
        self,
        *,
        tool_turns: Sequence[int] = (),
        usage: UsageDetails | None = None,
        reply: str = "a reply with some body to it",
        obey_tool_choice: bool = False,
        finish_reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the stub.

        Keyword Args:
            tool_turns: Indices of calls that should answer with a tool call instead of text.
            usage: Usage to report on every response.
            reply: What every non-tool call answers with. Settable because a reply is history
                for every later turn, so a test about how large a conversation gets cannot use
                a six-word one.
            obey_tool_choice: Call whatever function the request pinned, as a real model does.
                Off by default so that the tests written against ``tool_turns`` keep choosing
                which calls use a tool; on, the conversation actually gathers every tool result,
                which is the only way an offline test can see how large a real run gets.
            finish_reason: Reported on every response. ``"length"`` is a provider saying it
                stopped because the answer reached its cap, which is the only signal that
                separates a record the model kept short from one it was cut off in.
        """
        super().__init__(**kwargs)
        self.seen: list[int] = []
        self.options_seen: list[dict[str, Any]] = []
        self.tool_turns = set(tool_turns)
        self.usage = usage
        self.reply = reply
        self.obey_tool_choice = obey_tool_choice
        self.finish_reason = finish_reason

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        index = len(self.seen)
        self.seen.append(len(messages))
        self.options_seen.append(dict(options))
        choice = options.get("tool_choice")
        pinned = choice.get("required_function_name") if isinstance(choice, Mapping) else None
        stamp = self._stamp(index, messages)

        async def _go() -> ChatResponse[Any]:
            if index in self.tool_turns or (self.obey_tool_choice and pinned):
                contents: list[Any] = [
                    Content.from_function_call(
                        call_id=f"call_{stamp}",
                        # One no-argument tool per scope: the turn's scope is pinned by which
                        # function is called, not by an argument the model chooses.
                        name=pinned or "lookup_early",
                        arguments="{}",
                    )
                ]
            else:
                contents = [f"{self.reply} #{stamp}"]
            return ChatResponse(
                messages=Message(role="assistant", contents=contents),
                usage_details=self.usage,
                finish_reason=self.finish_reason,
            )

        return _go()

    def _stamp(self, index: int, messages: Sequence[Message]) -> str:
        """Return what makes this call's output distinguishable from every other call's.

        It reaches the reply text and the tool call id, and the replies are the reason it
        exists: the history provider hashes messages to decide what is new, so byte-identical
        replies after the first are dropped as duplicates, and a stub that answered the same
        words every time built a history a third the size it appeared to be.

        Keyed on the call index, which is right for a stub whose every call lands once. A test
        that re-sends a turn keys it on the request instead, so that the same conversation
        produces the same output whichever attempt sent it -- see :class:`_ToolLoopStub`.
        """
        return str(index)

    def service_url(self) -> str:
        """Return a placeholder URL."""
        return "stub://"


def _agent(client: StubChatClient, strategy: Any, recorder: UsageRecorder, **kwargs: Any) -> Agent[Any]:
    return Agent(
        client=client,
        name="test",
        instructions="SYSTEM",
        context_providers=[InMemoryHistoryProvider()],
        compaction_strategy=strategy,
        require_per_service_call_history_persistence=True,
        middleware=[recorder],
        **kwargs,
    )


# region recorder


@pytest.mark.parametrize(
    "strategy",
    [
        None,
        SlidingWindowStrategy(keep_last_groups=2),
        TruncationStrategy(max_n=40, compact_to=20, tokenizer=TOKENIZER),
    ],
    ids=["none", "sliding_window", "truncation"],
)
async def test_recorder_matches_what_the_client_received(strategy: Any) -> None:
    """The recorded prompt must be the prompt that was actually sent.

    This is the property the whole live benchmark rests on. Both ways of getting it wrong
    were measured before it was written correctly: reading ``context.messages`` before the
    call reports every conversation as one message, and reading it after without projecting
    reports that every strategy preserved every fact.
    """
    recorder = UsageRecorder()
    client = StubChatClient()
    agent = _agent(client, strategy, recorder)
    session = agent.create_session()
    for index in range(5):
        await agent.run(f"user message {index}", session=session)

    assert [call.messages_sent for call in recorder.calls] == client.seen


async def test_recorder_sees_loaded_history_not_only_the_new_message() -> None:
    """The pre-compaction count must grow with the conversation.

    The history is loaded by a middleware that runs inside this one and replaces
    ``context.messages``. A reference captured before the call keeps pointing at the
    original one-element list, which silently pins every measurement to 1.
    """
    recorder = UsageRecorder()
    agent = _agent(StubChatClient(), None, recorder)
    session = agent.create_session()
    for index in range(4):
        await agent.run(f"user message {index}", session=session)

    counts = [call.messages_before_compaction for call in recorder.calls]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0], "history never grew, so the stale-reference bug is back"


async def test_recorder_shows_compaction_removing_messages() -> None:
    """A compacting run must record fewer messages sent than the history held."""
    recorder = UsageRecorder()
    agent = _agent(StubChatClient(), SlidingWindowStrategy(keep_last_groups=1), recorder)
    session = agent.create_session()
    for index in range(5):
        await agent.run(f"user message {index}", session=session)

    last = recorder.calls[-1]
    assert last.messages_sent < last.messages_before_compaction


async def test_recorder_captures_usage() -> None:
    """Reported usage must be carried through to the recorded call."""
    usage = UsageDetails(input_token_count=900, output_token_count=40, cache_read_input_token_count=300)
    recorder = UsageRecorder()
    agent = _agent(StubChatClient(usage=usage), None, recorder)
    await agent.run("hello", session=agent.create_session())

    call = recorder.calls[0]
    assert call.input_tokens == 900
    assert call.cached_tokens == 300
    assert call.output_tokens == 40
    assert call.fresh_tokens == 600


async def test_recorder_counts_every_call_in_a_tool_turn() -> None:
    """A turn that calls a tool bills more than one prompt, so it must record more than one."""
    recorder = UsageRecorder()
    scenario = build_recall_scenario(salt="tool", filler_turns=3, filler_tokens=100)
    client = StubChatClient(tool_turns=(0,))
    agent = _agent(client, None, recorder, tools=make_scope_tools(scenario.tool_lookups, 100))
    await agent.run("look up the early deployment facts", session=agent.create_session())

    assert len(recorder.calls) > 1, "the tool round trip was not billed as its own call"


# endregion
# region metering


async def test_metered_client_accumulates_usage() -> None:
    """Summarizer calls must be counted, since the agent's middleware never sees them."""

    class Inner:
        additional_properties: dict[str, Any] = {}

        async def get_response(self, *args: Any, **kwargs: Any) -> ChatResponse[Any]:
            return ChatResponse(
                messages=Message(role="assistant", contents=["summary"]),
                usage_details=UsageDetails(input_token_count=500, output_token_count=80),
            )

    metered = MeteredClient(Inner())
    await metered.get_response([])
    await metered.get_response([])

    assert metered.calls == 2
    assert metered.input_tokens == 1_000
    assert metered.output_tokens == 160
    assert metered.failures == 0


async def test_metered_client_counts_failures_and_reraises() -> None:
    """A failing summarizer must be visible.

    ``SummarizationStrategy`` catches its own errors and returns False, so a broken
    summarizer produces a run that never compacted and therefore scores perfect recall.
    Counting the failure here is the only thing that distinguishes that from a real win.
    """

    class Failing:
        additional_properties: dict[str, Any] = {}

        async def get_response(self, *args: Any, **kwargs: Any) -> ChatResponse[Any]:
            raise RuntimeError("summarizer exploded")

    metered = MeteredClient(Failing())
    with pytest.raises(RuntimeError, match="exploded"):
        await metered.get_response([])

    assert metered.failures == 1
    assert metered.calls == 1


def test_metered_client_forwards_unknown_attributes() -> None:
    """The proxy must behave like the client it wraps for everything it does not record."""

    class Inner:
        additional_properties = {"a": 1}
        model_id = "inner-model"

    metered = MeteredClient(Inner())
    assert metered.model_id == "inner-model"
    assert metered.additional_properties == {"a": 1}


# endregion
# region tool


def test_lookup_tool_returns_the_planted_markers() -> None:
    """The live tool must return the same markers the replayed transcript scripts."""
    scenario = build_recall_scenario(salt="s", filler_turns=3, filler_tokens=100)
    lookup = make_lookup_tool(scenario.tool_lookups)
    region, host = scenario.tool_lookups["early"]

    result = lookup("early")
    assert region in result
    assert host in result


def test_tool_results_are_sized_in_tokens_not_characters() -> None:
    """Tool output must be as large as asked for, in tokens.

    The size was once a character count: a 600-character body is about 76 tokens, so six
    results came to under 2% of a 28,000-token prompt and every tool-oriented strategy had
    essentially nothing to evict. Getting the unit wrong made those strategies look inert.
    """
    scenario = build_live_scenario(salt="x", filler_turns=6, filler_tokens=500)
    lookup = make_lookup_tool(scenario.tool_lookups, 4_000)

    approx_tokens = len(lookup("early")) / 7.9
    assert 3_000 < approx_tokens < 5_000


def test_tool_results_differ_between_scopes() -> None:
    """Each scope must return distinct text.

    Identical results would share a long prefix, letting unrelated messages match by accident
    and inflating measured cache reuse.
    """
    scenario = build_live_scenario(salt="x", filler_turns=6, filler_tokens=500)
    lookup = make_lookup_tool(scenario.tool_lookups, 500)
    results = {scope: lookup(scope) for scope in scenario.tool_lookups}

    assert len(set(results.values())) == len(results)
    for scope, (region, host) in scenario.tool_lookups.items():
        assert region in results[scope]
        assert host in results[scope]


def test_scope_tools_are_one_per_scope_and_take_no_arguments() -> None:
    """Splitting the tool per scope is what makes forcing deterministic.

    ``tool_choice`` can name the function to call but cannot constrain its arguments, so a
    single ``lookup_deployment(scope)`` leaves the choice of scope to the model. Measured:
    forcing a call raised tool use from 4 to 7 per run yet still reached only 3 of 6 scopes,
    one of them called twice.
    """
    import inspect

    scenario = build_live_scenario(salt="s", filler_turns=3, filler_tokens=100, tool_turns=6)
    tools = make_scope_tools(scenario.tool_lookups, 100)

    assert sorted(tool.__name__ for tool in tools) == sorted(f"lookup_{s}" for s in scenario.tool_lookups)
    for tool in tools:
        assert not inspect.signature(tool).parameters, "a scope argument would let the model choose"
    for scope, (region, host) in scenario.tool_lookups.items():
        result = next(tool for tool in tools if tool.__name__ == f"lookup_{scope}")()
        assert region in result
        assert host in result


async def test_forcing_pins_the_exact_tool_per_turn_and_closes_the_rest() -> None:
    """Each tool turn must require its own function, and other turns must allow none.

    Requiring a call only on the wanted turns still lets the model make unwanted ones
    elsewhere: measured at 12 calls against the 6 asked for, doubling the tokens carried on
    one repeat in three.
    """
    scenario = build_live_scenario(salt="pin", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = StubChatClient()
    await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        force_tool_calls=True,
    )

    required = [
        options["tool_choice"]["required_function_name"]
        for options in client.options_seen
        if isinstance(options.get("tool_choice"), dict)
    ]
    assert sorted(set(required)) == sorted(f"lookup_{s}" for s in scenario.tool_lookups)
    assert any(options.get("tool_choice") == "none" for options in client.options_seen)


async def test_forcing_can_be_turned_off() -> None:
    """Leaving the model to decide must remain possible, since that is real agent behaviour."""
    scenario = build_live_scenario(salt="free", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = StubChatClient()
    await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        force_tool_calls=False,
    )

    # The framework supplies its own tool_choice default when tools are present; what must be
    # absent is *forcing* -- a named function, or closing a turn to tools entirely.
    choices = [options.get("tool_choice") for options in client.options_seen]
    assert not any(isinstance(choice, dict) for choice in choices)
    assert "none" not in choices


def test_lookup_tool_handles_an_unknown_scope() -> None:
    """An unknown scope must not raise; the model picking a wrong argument is not a crash."""
    lookup = make_lookup_tool({"early": ("R", "H")})
    assert "Unknown scope" in lookup("nonsense")


# endregion
# region wiring


def test_plain_agent_puts_the_before_phase_on_the_agent_not_the_provider() -> None:
    """The before strategy must not be installed on the CompactionProvider.

    ``CompactionProvider.before_strategy`` is a no-op under per-service-call history
    persistence: the agent skips ``HistoryProvider.before_run``, so the provider only ever
    sees an empty context. Installing it there compacts nothing while looking correct.
    """
    strategy = SlidingWindowStrategy(keep_last_groups=2)
    agent = build_live_agent(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        kind="plain",
        strategy=strategy,
        tokenizer=TOKENIZER,
        tools=[],
        recorder=UsageRecorder(),
        max_context_window_tokens=8_000,
        max_output_tokens=512,
    )

    assert agent.compaction_strategy is strategy
    providers = [p for p in agent.context_providers if isinstance(p, CompactionProvider)]
    assert len(providers) == 1
    assert providers[0].before_strategy is None
    assert providers[0].after_strategy is strategy


def test_plain_agent_installs_no_provider_for_the_control() -> None:
    """The uncompacted control must carry no compaction anywhere."""
    agent = build_live_agent(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        kind="plain",
        strategy=None,
        tokenizer=TOKENIZER,
        tools=[],
        recorder=UsageRecorder(),
        max_context_window_tokens=8_000,
        max_output_tokens=512,
    )

    assert agent.compaction_strategy is None
    assert not [p for p in agent.context_providers if isinstance(p, CompactionProvider)]


class RepeatingStub(StubChatClient):
    """A stub that answers every call with the same words, the way a model answers filler.

    ``StubChatClient`` stamps each reply with the call index precisely so that no two of them
    can collide, which is the one thing a test about message identity has to switch off. The
    live runs it stands in for did collide: the model's acknowledgements of filler turns were
    byte-identical, which is what the content hash could not tell apart.
    """

    def _stamp(self, index: int, messages: Sequence[Message]) -> str:
        """Return nothing, so every reply this stub gives is identical to every other."""
        return ""


@pytest.mark.parametrize("kind", list(AGENT_KINDS))
def test_every_agent_kind_issues_its_messages_an_id(kind: str) -> None:
    """Identity has to be the same on both kinds, since either can be the cell under test.

    The harness builds its own history provider when none is handed to it, so the fix reaching
    only the plain agent would leave every ``--agent harness`` cell -- which is every cell on
    disk -- carrying the defect while the tests said otherwise.
    """
    agent = build_live_agent(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        kind=kind,
        strategy=None,
        tokenizer=TOKENIZER,
        tools=[],
        recorder=UsageRecorder(),
        max_context_window_tokens=8_000,
        max_output_tokens=512,
    )

    providers = [p for p in agent.context_providers if isinstance(p, InMemoryHistoryProvider)]
    assert providers, f"the {kind} agent stores its history somewhere else"
    assert all(isinstance(provider, IdentifiedHistoryProvider) for provider in providers)


@pytest.mark.parametrize("strategy", ["truncation", "context_window"])
async def test_the_control_carries_the_same_conversation_as_a_strategy_row(strategy: str) -> None:
    """Two rows driven from one turn list must reach the same number of messages.

    The control has no strategy, so nothing annotates its messages and nothing gave them ids;
    identity then fell back to ``(role, serialized contents)``, and the model's byte-identical
    replies to filler turns collided and were dropped from the stored history. A strategy row
    was never affected, because compaction stamps everything it touches.

    So the control was measured on a shorter conversation than every row it was the baseline
    for. Recorded at 120,000/0.86: 82 messages against 109, with ``anchored`` -- which planned
    nothing at that cell -- ending 5.4% larger than the baseline it was supposed to equal.
    Every ``vs none`` taken then is a comparison between two workloads.
    """
    scenario = build_live_scenario(salt="identity", filler_turns=3, filler_tokens=50, tool_turns=6)
    peaks: dict[str, int] = {}
    for name in ("none", strategy):
        outcome = await run_live(
            ProviderRuntime(client=RepeatingStub(usage=UsageDetails(input_token_count=100)), model="stub"),
            strategy_name=name,
            options=_options(),
            scenario=scenario,
            probe_repeats=1,
        )
        assert outcome.error is None
        peaks[name] = outcome.messages_peak

    assert peaks["none"] == peaks[strategy], "the control lost messages the strategy row kept"


async def test_the_ids_never_reach_the_provider() -> None:
    """The fix has to move no prompt, or it is a second change riding on the first.

    ``serialize_message`` excludes ``message_id`` by construction, so an id changes no prompt
    text, no token count and no cache prefix. Asserted on the recorded prompts rather than
    argued from the serializer, because the recorded prompts are what every cost column reads.
    """
    scenario = build_live_scenario(salt="identity", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=RepeatingStub(usage=UsageDetails(input_token_count=100)), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=1,
    )

    assert outcome.error is None
    assert "cachebench_" not in outcome.snapshot_prompt
    assert not [call for call in outcome.calls if "cachebench_" in call.prompt_text]


def test_unknown_agent_kind_is_rejected() -> None:
    """An unknown agent kind must fail loudly rather than silently picking a default."""
    with pytest.raises(ValueError, match="Unknown agent kind"):
        build_live_agent(
            ProviderRuntime(client=StubChatClient(), model="stub"),
            kind="nonsense",
            strategy=None,
            tokenizer=TOKENIZER,
            tools=[],
            recorder=UsageRecorder(),
            max_context_window_tokens=8_000,
            max_output_tokens=512,
        )


def test_agent_kinds_are_the_documented_ones() -> None:
    """The advertised kinds must match what the builder accepts."""
    assert AGENT_KINDS == ("plain", "harness")


def test_server_side_history_is_overridden_by_default() -> None:
    """A client that keeps history server-side must be forced to send it.

    Otherwise MAF skips history loading, the agent sends only the new turn, and no strategy
    can compact anything: every row silently measures the same conversation. Measured on
    Foundry, where a 16-turn run reported a one-message prompt on every row.
    """

    class Stateful:
        STORES_BY_DEFAULT = True

    assert wants_client_side_history(Stateful()) is True


def test_stateless_clients_are_left_alone() -> None:
    """A client that already sends history needs no override."""

    class Stateless:
        STORES_BY_DEFAULT = False

    assert wants_client_side_history(Stateless()) is False
    assert wants_client_side_history(object()) is False


def test_server_history_can_be_opted_into() -> None:
    """Opting in must be possible, since measuring the service is a valid question."""

    class Stateful:
        STORES_BY_DEFAULT = True

    assert wants_client_side_history(Stateful(), allow_server_history=True) is False


def test_every_argument_the_runner_reads_is_defined() -> None:
    """Parsing must produce every attribute the run function reads.

    A flag referenced but never declared raises AttributeError only once a live run is
    already under way. That happened: two full runs died on the first call because an
    `add_argument` edit silently failed to apply while the code using it did not.
    """
    args = build_parser().parse_args(["openrouter:some/model"])
    for name in (
        "strategies",
        "agent",
        "repeats",
        "filler_turns",
        "filler_tokens",
        "tool_turns",
        "tool_result_tokens",
        "context_window",
        "max_output_tokens",
        "answer_max_tokens",
        "budget_fraction",
        "band_share",
        "keep_tokens",
        "min_gain_fraction",
        "keep_head_groups",
        "keep_tail_groups",
        "keep_last_groups",
        "keep_last_tool_groups",
        "trigger_fraction",
        "fallback_fraction",
        "coverage_share",
        "no_record_repeats",
        "min_correctness",
        "summarizer_provider",
        "no_force_tool_calls",
        "server_history",
        "no_temperature",
        "tokenizer",
        "show_answers",
        "dump_record",
        "max_groups_before_record",
        "dry_run",
        "fill",
        "tool_share",
        "probe_repeats",
        "combined_repeats",
    ):
        assert hasattr(args, name), f"--{name.replace('_', '-')} is read by the runner but not declared"


def test_the_help_can_actually_be_printed() -> None:
    """``--help`` must not raise, which is not free: argparse %-formats every help string.

    A bare ``%`` in help text makes the whole parser unprintable, and nothing else notices --
    parsing works, runs work, and the CLI is simply undiscoverable. Two help strings quoting
    percentages had broken it, found only because someone ran ``--help``.
    """
    help_text = build_parser().format_help()

    assert "--probe-repeats" in help_text
    assert "--combined-repeats" in help_text


def test_the_help_says_which_of_the_two_payload_flags_wins() -> None:
    """--tool-share and --tool-result-tokens state one quantity two ways.

    A reader who sets both and is not told which is read will believe the run carried the size
    they typed. Both help strings say it, because either one is where they will look.
    """
    help_text = build_parser().format_help()

    assert "--tool-share" in help_text
    assert "wins when both are given" in help_text
    assert "Ignored when" in help_text, "--tool-result-tokens must say it loses"


def _dry_argv(*extra: str) -> list[str]:
    """Return a command line for a dry run at the cell the payload flags were written for.

    120,000 tokens at 0.86 fill is where an absolute 3,500-token payload left
    ``AnchoredCompactionStrategy`` inert, so it is the cell whose numbers mean something.

    Args:
        extra: Flags appended after the defaults, so they win.

    Returns:
        The argument vector.
    """
    return [
        "azure",
        "--price-input",
        "1",
        "--strategies",
        "none",
        "--tokenizer",
        "estimator",
        "--context-window",
        "120000",
        "--fill",
        "0.86",
        "--tool-turns",
        "6",
        "--markers-per-tool",
        "8",
        "--dry-run",
        *extra,
    ]


async def test_the_dry_run_states_the_size_it_derived_and_the_share_it_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nothing else says what a derived payload came out at before money is spent.

    The size is the number the run is about to build with, and under --tool-share it appears
    nowhere the caller typed it. The share beside it is what makes the number checkable: a
    10,000-token result means nothing on its own, and "60.0% against 60% requested" is the
    whole claim.
    """
    await run_live_comparison(build_parser().parse_args(_dry_argv("--tool-share", "0.6")))
    printed = capsys.readouterr().out

    assert "tool share: 60.0% predicted against 60% requested" in printed
    assert re.search(r"6 tool results of ~[\d,]+ tokens", printed), "the derived size must be printed"


async def test_a_dry_run_without_a_share_says_nothing_about_one(capsys: pytest.CaptureFixture[str]) -> None:
    """A line reporting a share of 0 would read as a payload that vanished."""
    await run_live_comparison(build_parser().parse_args(_dry_argv("--tool-result-tokens", "3500")))
    printed = capsys.readouterr().out

    assert "tool share:" not in printed
    assert "6 tool results of ~3,500 tokens" in printed


async def test_the_tool_share_overrides_the_stated_result_size_through_the_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The precedence has to hold where it is actually exercised, not only in the solver.

    The size the run builds with is read in four places -- the sizing print, the scenario, the
    record's cell parameters and run_live itself -- and a flag that wins in the solver while
    one of those still read the raw argument would build one conversation and describe another.
    """
    await run_live_comparison(build_parser().parse_args(_dry_argv("--tool-share", "0.6", "--tool-result-tokens", "50")))
    printed = capsys.readouterr().out

    assert "6 tool results of ~50 tokens" not in printed
    assert "tool share: 60.0% predicted against 60% requested" in printed


async def test_a_tool_share_without_a_fill_target_is_refused() -> None:
    """A share of nothing is not a size, and guessing one would be a silent second cell."""
    argv = _dry_argv("--tool-share", "0.6", "--fill", "0", "--filler-turns", "3", "--filler-tokens", "50")

    with pytest.raises(SystemExit) as error:
        await run_live_comparison(build_parser().parse_args(argv))

    assert "--tool-share is a share of the fill target" in str(error.value)


# endregion
# region strategies


def _options(**kwargs: Any) -> StrategyOptions:
    return StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=32_000, max_output_tokens=2_048, **kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "token_budget_fallback",
        "token_budget_tools_first",
        "token_budget_truncate_first",
        "token_budget_window_first",
    ],
)
def test_token_budget_variants_share_one_ceiling(name: str) -> None:
    """Every composed variant must compact to the same budget.

    That shared ceiling is what makes the family comparable: holding the target size fixed
    means any difference in what survives is attributable to the order of deletion rather
    than to one variant simply trimming harder than another.
    """
    options = _options()
    strategy = build_strategy(name, options)

    assert isinstance(strategy, TokenBudgetComposedStrategy)
    assert strategy.token_budget == options.composed_budget_tokens


def test_token_budget_fallback_composes_nothing() -> None:
    """The family's control must rely purely on the built-in oldest-first fallback."""
    strategy = build_strategy("token_budget_fallback", _options())

    assert isinstance(strategy, TokenBudgetComposedStrategy)
    assert strategy.strategies == []


def test_budget_fraction_moves_the_ceiling() -> None:
    """The shared ceiling must follow the configured fraction of the input budget."""
    tight = build_strategy("token_budget_tools_first", _options(token_budget_fraction=0.25))
    loose = build_strategy("token_budget_tools_first", _options(token_budget_fraction=0.75))

    assert isinstance(tight, TokenBudgetComposedStrategy)
    assert isinstance(loose, TokenBudgetComposedStrategy)
    assert tight.token_budget < loose.token_budget


def test_context_window_matches_the_framework_tool_retention_default() -> None:
    """The harness-default row must retain as many tool groups as the harness does.

    ``create_harness_agent`` passes no ``keep_last_tool_call_groups``, so it inherits the
    framework default of 4. A lab override would make this row harsher than the
    configuration it is supposed to stand for.
    """
    import inspect

    framework_default = (
        inspect.signature(ContextWindowCompactionStrategy.__init__).parameters["keep_last_tool_call_groups"].default
    )
    assert _options().keep_last_tool_call_groups == framework_default

    built = build_strategy("context_window", _options())
    assert isinstance(built, ContextWindowCompactionStrategy)
    assert built.tool_eviction_threshold == ContextWindowCompactionStrategy.DEFAULT_TOOL_EVICTION_THRESHOLD
    assert built.truncation_threshold == ContextWindowCompactionStrategy.DEFAULT_TRUNCATION_THRESHOLD


#: Every strategy knob the command line can set, at a value nothing else in the package uses,
#: so a builder that silently took a constructor default fails the assertion rather than
#: matching it by coincidence.
_TUNED_ARGV = (
    "--keep-head-groups",
    "5",
    "--keep-tail-groups",
    "7",
    "--keep-last-groups",
    "9",
    "--keep-last-tool-groups",
    "2",
    "--keep-tokens",
    "321",
    "--band-share",
    "0.11",
    "--min-gain-fraction",
    "0.13",
    "--trigger-fraction",
    "0.17",
    "--fallback-fraction",
    "0.19",
    "--coverage-share",
    "0.23",
    "--budget-fraction",
    "0.29",
)


def _tuned_options() -> StrategyOptions:
    """Return the options a command line setting every knob produces."""
    return _strategy_options(build_parser().parse_args(["azure", *_TUNED_ARGV]), TOKENIZER)


def test_every_tuning_flag_reaches_the_strategy_that_consumes_it() -> None:
    """A flag that parses and then goes nowhere is worse than no flag at all.

    It is worse because the run reports the value: the cell parameters, the dry-run plan and
    the archived log all quote what was typed, so a sweep across a knob nothing reads produces
    a table of identical rows labelled with different settings, and the conclusion drawn is
    that the knob does not matter. Five of these were unreachable at once --
    ``min_gain_fraction`` is the whole of what separates ``anchored_min_gain`` from
    ``anchored``, and the pair had only ever been compared at one value of it.
    """
    options = _tuned_options()

    anchored = build_strategy("anchored", options)
    assert isinstance(anchored, AnchoredCompactionStrategy)
    assert (anchored.keep_head_groups, anchored.keep_tail_groups) == (5, 7)
    assert (anchored.keep_tokens, anchored.band_share) == (321, 0.11)

    min_gain = build_strategy("anchored_min_gain", options)
    assert isinstance(min_gain, MinimumGainAnchoredCompactionStrategy)
    assert min_gain.min_gain_fraction == 0.13

    summary = build_strategy("tool_summary_anchored", options)
    assert isinstance(summary, ToolResultAnchoredSummarizationCompactionStrategy)
    assert (summary.trigger_fraction, summary.fallback_fraction) == (0.17, 0.19)
    assert (summary.coverage_share, summary.keep_head_groups, summary.keep_tail_groups) == (0.23, 5, 7)

    window = build_strategy("sliding_window", options)
    assert isinstance(window, SlidingWindowStrategy)
    assert window.keep_last_groups == 9

    tools = build_strategy("tool_result", options)
    assert isinstance(tools, ToolResultCompactionStrategy)
    assert tools.keep_last_tool_call_groups == 2

    composed = build_strategy("token_budget_fallback", options)
    assert isinstance(composed, TokenBudgetComposedStrategy)
    assert composed.token_budget == options.composed_budget_tokens


def test_the_anchored_knobs_reach_the_fallback_hiding_inside_the_record_strategy() -> None:
    """``tool_summary_anchored`` falls back to an anchored strategy, and it is a strategy row too.

    Left to its own default, that inner strategy took ``AnchoredCompactionStrategy``'s
    constructor defaults for ``band_share`` and ``keep_tokens`` -- so a sweep across either
    moved every anchored row except the one nested inside this one, on a path this strategy
    takes often enough that ``RECFALLBACK`` has its own flag and its own column.
    """
    summary = build_strategy("tool_summary_anchored", _tuned_options())

    assert isinstance(summary, ToolResultAnchoredSummarizationCompactionStrategy)
    fallback = summary.fallback
    assert isinstance(fallback, AnchoredCompactionStrategy)
    assert (fallback.band_share, fallback.keep_tokens) == (0.11, 321)
    assert (fallback.keep_head_groups, fallback.keep_tail_groups) == (5, 7)


def test_a_retention_of_zero_means_derive_it_rather_than_keep_nothing() -> None:
    """``--keep-tokens 0`` is the absence of a fixed budget, the convention every other 0 uses.

    Read literally it would mean a retention of nothing, which is a different strategy: the
    anchored family would shorten every banded result to its marker. ``--fill 0``,
    ``--record-max-tokens 0`` and ``--max-groups-before-record 0`` all already mean "no bound
    of my own", and one flag reading its zero the other way is the kind of difference nobody
    checks before spending a cell on it.
    """
    default = _strategy_options(build_parser().parse_args(["azure"]), TOKENIZER)
    explicit = _strategy_options(build_parser().parse_args(["azure", "--keep-tokens", "0"]), TOKENIZER)

    assert default.keep_tokens is None
    assert explicit.keep_tokens is None


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        pytest.param(["--band-share", "1.5"], "band_share", id="band-share"),
        pytest.param(["--keep-head-groups", "-1"], "keep_head_groups", id="keep-head-groups"),
        pytest.param(["--min-gain-fraction", "1.0"], "min_gain_fraction", id="min-gain-fraction"),
        pytest.param(["--trigger-fraction", "0"], "trigger_fraction", id="trigger-fraction"),
        pytest.param(["--fallback-fraction", "1.5"], "fallback_fraction", id="fallback-fraction"),
        pytest.param(["--coverage-share", "1.5"], "coverage_share", id="coverage-share"),
        pytest.param(
            ["--trigger-fraction", "0.9", "--fallback-fraction", "0.9"], "fallback_fraction", id="thresholds-equal"
        ),
        pytest.param(
            ["--trigger-fraction", "0.9", "--fallback-fraction", "0.8"], "fallback_fraction", id="thresholds-inverted"
        ),
    ],
)
def test_a_value_outside_a_strategys_range_is_refused_before_anything_is_spent(argv: list[str], match: str) -> None:
    """A bad number has to fail at the command line, not on the first call of a paid cell.

    The ranges themselves are checked in ``compaction/``, which is where they belong -- those
    classes ship without this package -- so this is about *when*, not about a second copy of
    the rule. Before the pre-flight existed a bad ``--band-share`` surfaced on the first seed,
    after the provider was built and the pricing fetched, and under ``--dry-run`` it surfaced
    not at all: the dry run built every strategy from bare defaults and then printed "every
    strategy builds cleanly" about a configuration it was not going to use.
    """
    strategies = ["anchored", "anchored_min_gain", "tool_summary_anchored"]
    options = _strategy_options(build_parser().parse_args(["azure", *argv]), TOKENIZER)

    with pytest.raises(SystemExit) as error:
        _build_or_exit(strategies, options)

    assert match in str(error.value)


async def test_the_dry_run_checks_the_configuration_it_is_printing_a_plan_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ "Every strategy builds cleanly" has to be about this run's flags, not about defaults."""
    argv = ("--strategies", "none,anchored")

    with pytest.raises(SystemExit) as error:
        await run_live_comparison(build_parser().parse_args(_dry_argv(*argv, "--band-share", "1.5")))

    assert "band_share" in str(error.value)

    await run_live_comparison(build_parser().parse_args(_dry_argv(*argv, "--band-share", "0.1")))

    assert "every strategy builds cleanly" in capsys.readouterr().out


# endregion
# region scenario


def test_live_scenario_keeps_the_same_facts_as_replay() -> None:
    """At matching settings the two modes must score identical planted facts.

    Compared at the same ``tool_turns`` on purpose. The live default is deliberately
    tool-heavier than replay's, so that tool-oriented compaction has enough groups to
    engage; what must not differ is the scoring for a given configuration.
    """
    replay = build_recall_scenario(salt="x", filler_turns=6, filler_tokens=4_000, tool_turns=6)
    live = build_live_scenario(salt="x", filler_turns=6, filler_tokens=4_000, tool_turns=6)

    assert live.facts == replay.facts
    assert live.tool_lookups == replay.tool_lookups
    assert live.contradictions == replay.contradictions


def test_live_defaults_are_more_tool_heavy_than_replay() -> None:
    """The live default must plant more tool groups than the replay default."""
    replay = build_recall_scenario(salt="x", filler_turns=6, filler_tokens=100)
    live = build_live_scenario(salt="x", filler_turns=6, filler_tokens=100)

    assert len(live.tool_lookups) > len(replay.tool_lookups)


def test_tool_results_carry_the_majority_of_the_scored_facts() -> None:
    """Most of what correctness scores lives in tool results.

    This is why the tool-group count matters so much: a strategy that touches tool results
    is acting on more than half the evidence the final answer is graded on.
    """
    scenario = build_live_scenario(salt="x", filler_turns=6, filler_tokens=100)
    tool_facts = [fact for fact in scenario.facts if fact.kind == "tool_result"]

    assert len(tool_facts) > len(scenario.facts) / 2


def test_default_tool_turns_exceed_the_frameworks_retention() -> None:
    """The default scenario must let tool-oriented compaction actually fire.

    Those strategies keep the last ``keep_last_tool_call_groups`` groups verbatim. Plant no
    more groups than that and they evict nothing, change no tokens, and score a perfect
    result for doing nothing — which reads as the best row in the table. Measured at 3 groups
    against a retention of 4: two strategies were exact no-ops while carrying 55% of the
    planted facts.
    """
    retained = _options().keep_last_tool_call_groups
    scenario = build_live_scenario(salt="x", filler_turns=6, filler_tokens=100)

    assert len(scenario.tool_lookups) > retained


@pytest.mark.parametrize("tool_turns", [3, 6, 8, 12])
def test_markers_stay_unique_as_the_scenario_grows(tool_turns: int) -> None:
    """Every planted fact must have a distinct, non-degenerate marker.

    Markers were once six-character slices of a single sha256, which runs dry after ten and
    then yields an empty marker. An empty marker is a substring of any answer, so it scores
    as recalled every time and silently inflates the result.
    """
    scenario = build_live_scenario(salt="x", filler_turns=9, filler_tokens=100, tool_turns=tool_turns)
    markers = [fact.marker for fact in scenario.facts]

    assert len(set(markers)) == len(markers)
    assert all(len(marker) >= 6 for marker in markers)
    assert not any(marker.endswith("-") for marker in markers)


def test_live_scenario_moves_the_bulk_to_the_user_turns() -> None:
    """Padding must sit on the user side when the assistant writes its own replies.

    A real model will not emit thousands of filler tokens on request, so leaving the bulk in
    the scripted reply would mean the history never grows and no strategy ever triggers.
    """

    def user_chars(scenario: Any) -> int:
        return sum(
            len(str(content)) for turn in scenario.transcript.turns for m in turn.request for content in m.contents
        )

    replay = build_recall_scenario(salt="x", filler_turns=6, filler_tokens=4_000)
    live = build_live_scenario(salt="x", filler_turns=6, filler_tokens=4_000)

    assert user_chars(live) > user_chars(replay) * 5


# endregion
# region outcome and cost


def _call(sent: int, before: int, *, inp: int = 0, cached: int = 0, out: int = 0) -> ModelCall:
    return ModelCall(
        messages_sent=sent,
        prompt_text="",
        messages_before_compaction=before,
        input_tokens=inp,
        cached_tokens=cached,
        output_tokens=out,
    )


def test_messages_peak_is_measured_before_compaction() -> None:
    """The peak must say how large the history got, not how hard it was trimmed."""
    outcome = LiveOutcome(
        strategy="s",
        calls=(_call(2, 5), _call(2, 9), _call(2, 7)),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=3,
        turns_total=3,
    )

    assert outcome.messages_peak == 9
    assert outcome.messages_left == 2
    assert outcome.messages_dropped == 5


def test_prompt_size_is_reported_in_tokens_not_only_messages() -> None:
    """An in-place rewrite must be visible.

    ``ToolResultCompactionStrategy`` collapses tool results into summaries without excluding
    any message, so the message count is identical before and after while real tokens are
    gone. A table that reports only message counts shows that strategy as having done
    nothing, which is how it was nearly dropped from the comparison.
    """
    outcome = LiveOutcome(
        strategy="tool_result",
        calls=(_call(9, 9, inp=8_000), _call(9, 9, inp=5_000)),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=2,
        turns_total=2,
    )

    assert outcome.messages_left == outcome.messages_peak == 9
    assert outcome.messages_dropped == 0
    assert outcome.prompt_tokens_final == 5_000
    assert outcome.prompt_tokens_peak == 8_000


def test_cost_includes_generation_and_summarizer_charges() -> None:
    """Live cost must price replies and the summarizer's own calls.

    A live run generates real replies, and summarization bills calls the agent never sees.
    Omitting either scores the strategy that spends most to preserve information as though
    preserving it were free.
    """
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.0, output_per_million=10.0)
    base = LiveOutcome(
        strategy="s",
        calls=(_call(1, 1, inp=1_000_000, out=100_000),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )
    with_summary = LiveOutcome(
        strategy="s",
        calls=(_call(1, 1, inp=1_000_000, out=100_000),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
        summarizer_input_tokens=1_000_000,
        summarizer_output_tokens=100_000,
    )

    assert _cost(base, pricing) == pytest.approx(2.0)
    assert _cost(with_summary, pricing) == pytest.approx(4.0)


def test_summarizer_cost_is_zero_for_a_strategy_that_never_summarized() -> None:
    """A non-summarizing strategy must carry no summarizer charge.

    A single shared meter once accumulated one strategy's summarizer spend into every later
    row as a flat addition, which is invisible in a total and inverted the ranking of a whole
    family. Reporting the charge per row is what makes that visible.
    """
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=10.0)
    outcome = LiveOutcome(
        strategy="truncation",
        calls=(_call(1, 1, inp=1_000, out=100),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )

    assert _summarizer_cost(outcome, pricing) == 0.0


def test_total_cost_is_the_agent_plus_its_own_summarizer() -> None:
    """The reported total must decompose exactly into the two halves shown."""
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=10.0)
    outcome = LiveOutcome(
        strategy="summarization",
        calls=(_call(1, 1, inp=2_000_000, out=100_000),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
        summarizer_input_tokens=500_000,
        summarizer_output_tokens=50_000,
    )

    agent_only = (2_000_000 * 1.0 + 100_000 * 10.0) / 1_000_000
    assert _cost(outcome, pricing) == pytest.approx(agent_only + _summarizer_cost(outcome, pricing))
    assert _summarizer_cost(outcome, pricing) == pytest.approx((500_000 * 1.0 + 50_000 * 10.0) / 1_000_000)


def _priced(strategy: str, inp: int) -> LiveOutcome:
    return LiveOutcome(
        strategy=strategy,
        calls=(_call(1, 1, inp=inp),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )


def test_spread_reports_the_gap_between_repeats() -> None:
    """Spread must show how far repeats of one strategy disagree."""
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)
    repeats = [_priced("s", 8_000), _priced("s", 10_000), _priced("s", 12_000)]

    assert _spread([_cost(outcome, pricing) for outcome in repeats]) == pytest.approx(0.4)


def test_spread_is_zero_for_a_single_repeat() -> None:
    """One repeat measures nothing about stability, and must not imply otherwise."""
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)

    assert _spread([_cost(_priced("s", 5_000), pricing)]) == 0.0


def test_cached_tokens_are_discounted() -> None:
    """Cache reads must be billed at the discounted rate, not the full input rate."""
    pricing = ModelPricing(input_per_million=10.0, cached_read_per_million=1.0, output_per_million=0.0)
    outcome = LiveOutcome(
        strategy="s",
        calls=(_call(1, 1, inp=1_000_000, cached=1_000_000),),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )

    assert _cost(outcome, pricing) == pytest.approx(1.0)


# endregion
# region run_live


async def test_a_rejected_option_is_dropped_and_the_run_continues() -> None:
    """A provider that rejects an option must not cost the whole run.

    Two of five models tested reject something: one refuses ``temperature``, another refuses
    any pinned ``tool_choice``. Before this, either produced 42 failed runs and no data.
    """

    class Picky(StubChatClient):
        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if "temperature" in options:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="picky", filler_turns=3, filler_tokens=50, tool_turns=6)
    runtime = ProviderRuntime(client=Picky(), model="stub", options={"temperature": 0.0, "max_tokens": 16})
    outcome = await run_live(runtime, strategy_name="none", options=_options(), scenario=scenario)

    assert outcome.error is None
    assert outcome.dropped_options == ("temperature",)
    assert outcome.turns_completed == outcome.turns_total


async def test_dropping_tool_choice_also_stops_forcing() -> None:
    """A provider that refuses pinned tool choice must fall back to letting the model decide.

    Continuing to send the option after it was refused would fail every remaining turn.
    """

    class NoForcing(StubChatClient):
        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if options.get("tool_choice") not in (None, "auto"):
                raise RuntimeError("Tool choice must be auto")
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="noforce", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=NoForcing(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        force_tool_calls=True,
    )

    assert outcome.error is None
    assert outcome.dropped_options == ("tool_choice",)
    assert outcome.turns_completed == outcome.turns_total


class _Throttled(Exception):
    """A 429 shaped the way a provider SDK raises one, and the way one reached the sweep.

    A status on the error, a response carrying the headers, and text naming both the code and
    the reason -- so a test can take away whichever of the three it wants to prove is enough.
    """

    def __init__(self, retry_after: str | None = None) -> None:
        """Create the refusal.

        Args:
            retry_after: Seconds to advertise in the ``Retry-After`` header, if any.
        """
        super().__init__("Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}")
        self.status_code = 429
        self.response = SimpleNamespace(headers={"Retry-After": retry_after} if retry_after else {})


class _Disconnected(Exception):
    """A dropped connection shaped the way one actually reaches this code.

    Deliberately unhelpful on its own. ``APIConnectionError`` carries no status, and its own
    message is the bare "Connection error.", so the only thing in it that says what happened is
    the transport exception it was raised *from* -- which is why the detection walks the chain.
    The message here names nothing a marker could match, so a test using this fails if the
    detection ever quietly degrades into reading text or class names.
    """

    def __init__(self, cause: BaseException | None = None) -> None:
        """Create the failure.

        Args:
            cause: What the transport raised, defaulting to a reset socket.
        """
        super().__init__("upstream request failed")
        self.__cause__ = cause or ConnectionResetError(104, "the peer went away")


class _Refused(Exception):
    """A refusal the provider decided on, carrying its status the way an SDK error does."""

    def __init__(self, status: int | None, message: str) -> None:
        """Create the refusal.

        Args:
            status: The HTTP status, or None for a provider that only rendered its message.
            message: What the provider said.
        """
        super().__init__(message)
        if status is not None:
            self.status_code = status


class ThrottlingStub(StubChatClient):
    """A stub that refuses its first calls with a 429 and then answers normally."""

    def __init__(self, *, refusals: int, retry_after: str | None = None, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            refusals: How many calls to refuse before answering.
            retry_after: Seconds to advertise in the ``Retry-After`` header of each refusal.
        """
        super().__init__(**kwargs)
        self.refusals = refusals
        self.refused = 0
        self.retry_after = retry_after

    def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
        if self.refused < self.refusals:
            self.refused += 1
            raise _Throttled(self.retry_after)
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


class DisconnectingStub(StubChatClient):
    """A stub whose first calls never come back with an answer, and which then answers."""

    def __init__(self, *, drops: int, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            drops: How many calls to lose before answering.
        """
        super().__init__(**kwargs)
        self.drops = drops
        self.dropped = 0

    def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
        if self.dropped < self.drops:
            self.dropped += 1
            raise _Disconnected
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


class RefusingStub(StubChatClient):
    """A stub that answers every call with the same refusal, as a wall does."""

    def __init__(self, *, status: int | None, message: str, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            status: The HTTP status to carry, or None to carry only the message.
            message: What the provider says.
        """
        super().__init__(**kwargs)
        self.status = status
        self.message = message

    def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
        raise _Refused(self.status, self.message)


class _Waits:
    """A stand-in for ``asyncio.sleep`` that records what it was asked for and returns at once.

    The bound on the retry is five minutes of waiting, and a test that proved it by waiting is
    a test nobody runs. Recording the schedule checks the same thing and checks it exactly: a
    delay that was capped wrongly is visible here and invisible in an elapsed time.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        """Record a wait instead of taking it."""
        self.delays.append(delay)


async def test_a_throttled_call_is_waited_out_and_re_sent() -> None:
    """A transient 429 must cost a wait, not the seed.

    The sweep this was built for lost 100 seeds and EUR 4.17 to exactly this: ``run_live`` had
    a two-attempt loop, but it existed only to drop an option the provider had named, so a
    rate limit fell straight through it and abandoned the whole seed at 0 of 32 turns.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="429", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=ThrottlingStub(refusals=2), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.rate_limit_retries == 2
    assert outcome.throttled_seconds == pytest.approx(sum(waits.delays))
    # Exponential from the base and jittered downwards, so each wait sits inside its own step
    # and is larger than the one before it.
    assert len(waits.delays) == 2
    assert 0 < waits.delays[0] <= RATE_LIMIT_BASE_DELAY
    assert waits.delays[0] < waits.delays[1] <= RATE_LIMIT_BASE_DELAY * 2


async def test_a_limit_that_does_not_lift_fails_the_turn_rather_than_looping() -> None:
    """Retries are for surviving a spike, not for hiding a wall.

    A quota that is genuinely exhausted has to end the seed, and end it in minutes: continuing
    with whatever turns got through would report a cheap, forgetful strategy that was never
    run, and looping would park a multi-hour sweep indefinitely.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="wall", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=ThrottlingStub(refusals=1_000), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is not None
    assert "429" in outcome.error
    assert outcome.turns_completed == 0
    assert outcome.rate_limit_retries == len(waits.delays) == RATE_LIMIT_ATTEMPTS - 1
    assert max(waits.delays) <= RATE_LIMIT_MAX_DELAY
    assert sum(waits.delays) <= RATE_LIMIT_MAX_WAIT
    # And the schedule really would have waited, so it is the injected sleep keeping this test
    # instant rather than the bounds being trivially small.
    assert sum(waits.delays) > RATE_LIMIT_BASE_DELAY


async def test_the_wait_a_provider_asks_for_is_taken_verbatim() -> None:
    """A named ``Retry-After`` is an instruction, not an input to a guess.

    The provider knows when its own window refills. Jittering it downwards, as the invented
    schedule is jittered, spends an attempt against a limit that has not moved.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="ra", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=ThrottlingStub(refusals=1, retry_after="7"), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert waits.delays == [7.0]
    assert outcome.throttled_seconds == 7.0


async def test_a_wait_longer_than_the_quota_window_is_capped() -> None:
    """An hour-long ``Retry-After`` is a daily cap, and parking a sweep on one is not surviving it."""
    waits = _Waits()
    scenario = build_live_scenario(salt="cap", filler_turns=3, filler_tokens=50, tool_turns=6)

    await run_live(
        ProviderRuntime(client=ThrottlingStub(refusals=1, retry_after="3600"), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert waits.delays == [RATE_LIMIT_MAX_DELAY]


async def test_an_error_that_is_not_throttling_is_not_retried() -> None:
    """A deterministic refusal must fail at once.

    The prompt-too-large error used here names the size that was refused, and 274,293 tokens
    contains the digits 429. Retried as though that were a status, a wall costs six attempts
    and minutes of waiting before failing exactly as it would have failed immediately.
    """
    waits = _Waits()

    class TooLarge(StubChatClient):
        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Error code: 400 - supports at most 272000 tokens, got 274293")

    scenario = build_live_scenario(salt="big", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=TooLarge(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is not None
    assert waits.delays == []
    assert outcome.rate_limit_retries == 0


async def test_dropping_an_option_and_waiting_out_a_limit_compose() -> None:
    """Both retries have to survive meeting each other.

    They are nested rather than placed side by side: throttling is waited out inside each
    attempt, so an option rejected on the attempt after two 429s still gets dropped and still
    gets another go. Written flat, whichever loop was outermost would swallow the other -- and
    the outermost one was the option drop, which is how a rate limit reached a handler that
    only knew what to do with an option the provider had named.
    """
    waits = _Waits()

    class Awkward(StubChatClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.refused = 0

        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if self.refused < 2:
                self.refused += 1
                raise _Throttled
            if "temperature" in options:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="both", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=Awkward(), model="stub", options={"temperature": 0.0}),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.dropped_options == ("temperature",)
    assert outcome.rate_limit_retries == 2
    assert len(waits.delays) == 2


def _dangling_calls(messages: Sequence[Message]) -> tuple[str, ...]:
    """Return the function calls a request carries and never answers.

    This is the provider's own check. A call with no result beside it is refused with
    ``400 No tool output found for function call``, and that is exactly what a re-sent turn
    carries when the retry starts from a session the failed attempt has already written to.
    """
    calls: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        for content in message.contents:
            if content.type == "function_call" and content.call_id:
                calls.add(content.call_id)
            elif content.type == "function_result" and content.call_id:
                answered.add(content.call_id)
    return tuple(sorted(calls - answered))


class _ToolLoopStub(StubChatClient):
    """A stub that fails a turn *inside* its tool-calling loop, the way the live sweep failed.

    The interleaving is the whole point, and a 429 on a plain text turn does not reproduce it.
    The model returns a function call; history is persisted per model call, so that assistant
    message is already durable; the follow-up call carrying the tool result is then refused,
    so the result never lands and the session is left holding a call nothing answers.

    The consequence is enforced here too: any request carrying a dangling call is refused with
    the provider's own 400, so a retry that re-sends one fails loudly rather than passing.
    """

    def __init__(
        self,
        *,
        throttle_once: bool = False,
        disconnect_once: bool = False,
        reject_option: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the stub.

        Keyword Args:
            throttle_once: Refuse the first call that carries a tool result with a 429.
            disconnect_once: Lose the first call that carries a tool result, so the connection
                fails in the same place -- between a function call and its result -- where a
                429 produced the provider 400 this restore exists for.
            reject_option: Name of a request option to refuse, and refuse only on a call
                carrying a tool result -- so the option-drop retry meets a half-finished turn
                exactly as the rate limit does.
        """
        super().__init__(obey_tool_choice=True, **kwargs)
        self.requests: list[tuple[Message, ...]] = []
        self.throttle_once = throttle_once
        self.disconnect_once = disconnect_once
        self.reject_option = reject_option
        self.throttled = 0
        self.disconnected = 0
        self.rejected = 0

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        self.requests.append(tuple(messages))
        if dangling := _dangling_calls(messages):
            raise RuntimeError(
                f"Error code: 400 - {{'error': {{'message': 'No tool output found for function call "
                f"{dangling[0]}.', 'type': 'invalid_request_error', 'param': 'input'}}}}"
            )
        after_tool_result = any(
            content.type == "function_result" for message in messages for content in message.contents
        )
        if after_tool_result and self.throttle_once and not self.throttled:
            self.throttled += 1
            raise _Throttled
        if after_tool_result and self.disconnect_once and not self.disconnected:
            self.disconnected += 1
            raise _Disconnected
        if after_tool_result and self.reject_option is not None and self.reject_option in options:
            self.rejected += 1
            raise RuntimeError(f"Unsupported parameter: '{self.reject_option}' is not supported with this model.")
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    def _stamp(self, index: int, messages: Sequence[Message]) -> str:
        """Key the reply and the call id on the request rather than on the call index.

        A retried turn costs an extra call, so index-keyed output would differ from a clean
        run's from that point onwards and the two histories could not be compared at all. The
        request is the same request whichever attempt sent it, and it grows with the
        conversation, so keying on its length keeps replies distinct between turns and
        identical between two runs that reached the same place.
        """
        return str(len(messages))


async def test_a_turn_throttled_inside_the_tool_loop_is_re_sent_from_where_it_started() -> None:
    """``agent.run`` is not idempotent, so a retry has to put the conversation back first.

    Measured live: 7 turns in one cell were refused with "No tool output found for function
    call", every one of them on a row that had been throttled and including the uncompacted
    ``none`` control -- which is what proves it was the retry and not compaction. The turn
    then failed and the seed was abandoned, so the retry written to save seeds was destroying
    them.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="midloop", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = _ToolLoopStub(throttle_once=True)

    outcome = await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert client.throttled == 1, "the stub never reached the call it was written to refuse"
    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.rate_limit_retries == 1
    assert [request for request in client.requests if _dangling_calls(request)] == []


async def test_a_retried_turn_leaves_the_history_a_clean_attempt_would_have_left() -> None:
    """Surviving the limit is not enough: the seed has to be the seed that was asked for.

    A retry that appended to a half-written turn could still complete every turn and report no
    error, having measured a conversation containing a stray call, a repeated question, or an
    orphaned tool result. Every strategy is scored against this history, so a difference here
    is a difference in the measurement, not in its bookkeeping.
    """
    scenario = build_live_scenario(salt="cleanstate", filler_turns=3, filler_tokens=50, tool_turns=6)

    async def _seed(client: _ToolLoopStub) -> LiveOutcome:
        return await run_live(
            ProviderRuntime(client=client, model="stub"),
            strategy_name="none",
            options=_options(),
            scenario=scenario,
            sleep=_Waits(),
        )

    throttled = await _seed(_ToolLoopStub(throttle_once=True))
    clean = await _seed(_ToolLoopStub())

    assert clean.error is None
    assert throttled.error is None
    assert throttled.rate_limit_retries == 1
    assert clean.rate_limit_retries == 0
    assert clean.snapshot_prompt, "the control seeded nothing, so matching it proves nothing"
    assert throttled.snapshot_prompt == clean.snapshot_prompt


async def test_dropping_an_option_inside_the_tool_loop_also_re_sends_from_the_start() -> None:
    """Both retries re-send, so both have to restore, and they have to survive meeting.

    A provider names an unsupported option on the call that carries it, and after a tool
    result that is the second call of the turn -- so this path reaches a half-finished turn
    exactly as throttling does. Here the same turn hits both: a 429 on the follow-up, then the
    refused option on the follow-up of the attempt after it.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="bothmidloop", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = _ToolLoopStub(throttle_once=True, reject_option="temperature")

    outcome = await run_live(
        ProviderRuntime(client=client, model="stub", options={"temperature": 0.0}),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert client.throttled == 1
    assert client.rejected == 1
    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.dropped_options == ("temperature",)
    assert outcome.rate_limit_retries == 1
    assert [request for request in client.requests if _dangling_calls(request)] == []


async def test_a_throttled_probe_is_also_re_sent_from_the_snapshot() -> None:
    """The probe phase restores before its first attempt, which is not the same as before each.

    A probe that calls a tool and is throttled on the follow-up leaves the same dangling call
    a seeding turn does, and the restore that ran before the probe started has already
    happened. Unpinned here, because the pinned ``tool_choice: "none"`` of an ordinary run
    makes tool calls during a probe rare rather than impossible.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="probeloop", filler_turns=3, filler_tokens=50, tool_turns=6)
    seed_calls = len(scenario.transcript.turns) - scenario.answer_turn_count
    client = _ToolLoopStub(throttle_once=True, tool_turns=(seed_calls,))

    outcome = await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        force_tool_calls=False,
        probe_repeats=1,
        sleep=waits,
    )

    assert client.throttled == 1, "the tool call landed somewhere other than the first probe"
    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert [request for request in client.requests if _dangling_calls(request)] == []


async def test_a_dropped_connection_is_waited_out_and_re_sent() -> None:
    """A network blip must cost a wait, not the seed.

    A cell of 30 seed records came back every row ``ERR`` with no turns completed, lost whole
    to ``APIConnectionError``; three cells of an earlier sweep went the same way. The retry
    covered HTTP 429 only, so a request that never arrived failed its turn and abandoned the
    seed, taking every turn already paid for with it.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="drop", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=DisconnectingStub(drops=2), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.connection_retries == 2
    assert outcome.connection_seconds == pytest.approx(sum(waits.delays))
    assert outcome.rate_limit_retries == 0, "a dropped connection was counted as throttling"
    assert outcome.throttled_seconds == 0.0
    # Its own schedule, and a faster one: a quota window refills on a fixed period and cannot
    # be hurried, while the only question a reconnect asks is whether the path is back.
    assert len(waits.delays) == 2
    assert 0 < waits.delays[0] <= CONNECTION_BASE_DELAY < RATE_LIMIT_BASE_DELAY
    assert waits.delays[0] < waits.delays[1] <= CONNECTION_BASE_DELAY * 2


async def test_a_connection_that_stays_down_fails_the_turn_rather_than_looping() -> None:
    """Retries are for surviving a blip, not for hiding an outage.

    A provider that has gone away has to end the seed, and end it in seconds: this is the same
    bargain the rate-limit retry makes, struck at a shorter horizon because there is no window
    to wait out. The bound has to be provable without waiting for it, which is what the
    injected sleep is for -- a delay capped wrongly is visible in the schedule and invisible in
    an elapsed time.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="outage", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=DisconnectingStub(drops=1_000), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is not None
    assert outcome.turns_completed == 0
    assert outcome.connection_retries == len(waits.delays) == CONNECTION_ATTEMPTS - 1
    assert max(waits.delays) <= CONNECTION_MAX_DELAY
    assert sum(waits.delays) <= CONNECTION_MAX_WAIT
    # And the schedule really would have waited, so it is the injected sleep keeping this test
    # instant rather than the bounds being trivially small.
    assert sum(waits.delays) > CONNECTION_BASE_DELAY


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Error code: 400 - supports at most 272000 tokens, got 274293"),
        (401, "Error code: 401 - Access denied due to invalid subscription key"),
        (403, "Error code: 403 - {'error': {'code': 'PermissionDenied'}}"),
        (None, "Error code: 400 - {'error': {'message': 'Invalid value for tool_choice'}}"),
    ],
    ids=["context_length", "auth", "forbidden", "no_status_at_all"],
)
async def test_a_failure_the_provider_decided_on_is_not_re_sent(status: int | None, message: str) -> None:
    """A refusal is an answer, and re-sending a large prompt against one buys nothing.

    This is the failure the ``429``-as-substring bug already caused once: a prompt-too-large
    error names the size it refused, and 274,293 contains those digits, so a wall was retried
    as though it were a spike and cost six attempts and minutes of waiting before failing
    exactly as it would have failed at once. Widening the retry is the same opportunity again,
    against prompts of 50,000 to 230,000 tokens.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="wall", filler_turns=3, filler_tokens=50, tool_turns=6)

    outcome = await run_live(
        ProviderRuntime(client=RefusingStub(status=status, message=message), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is not None
    assert waits.delays == []
    assert outcome.connection_retries == 0
    assert outcome.rate_limit_retries == 0


async def test_a_turn_disconnected_inside_the_tool_loop_is_re_sent_from_where_it_started() -> None:
    """The re-send has to put the conversation back, whatever it was that failed.

    A connection lost between a function call and its result leaves exactly what a 429 there
    leaves: history is persisted per model call, so the assistant's call is already durable and
    the result still in flight is not, and re-sending against that state is refused outright
    with "No tool output found for function call". That was measured at 7 occurrences in one
    cell, on the uncompacted control as well as the compacting rows, and it destroyed the seeds
    the retry existed to save. The new path reuses the same snapshot and restore, and this is
    what says so: the stub answers any request carrying a dangling call with the provider's own
    400, and the finished history has to be the one a clean run would have left.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="dropmidloop", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = _ToolLoopStub(disconnect_once=True)

    async def _seed(stub: _ToolLoopStub) -> LiveOutcome:
        return await run_live(
            ProviderRuntime(client=stub, model="stub"),
            strategy_name="none",
            options=_options(),
            scenario=scenario,
            sleep=waits,
        )

    outcome = await _seed(client)
    clean = await _seed(_ToolLoopStub())

    assert client.disconnected == 1, "the stub never reached the call it was written to drop"
    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.connection_retries == 1
    assert clean.connection_retries == 0
    assert [request for request in client.requests if _dangling_calls(request)] == []
    assert clean.snapshot_prompt, "the control seeded nothing, so matching it proves nothing"
    assert outcome.snapshot_prompt == clean.snapshot_prompt


async def test_a_lost_connection_and_a_rate_limit_keep_their_own_budgets() -> None:
    """One turn may meet both, and neither may spend the other's allowance.

    They are independent events with different schedules, so a turn that reconnected twice
    should still be able to wait out a quota window afterwards -- and it should wait out the
    *first* window with the first wait of the rate-limit schedule, not with wherever the
    reconnections had left a shared counter.
    """
    waits = _Waits()

    class Unlucky(StubChatClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.dropped = 0
            self.refused = 0

        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if self.dropped < 2:
                self.dropped += 1
                raise _Disconnected
            if self.refused < 2:
                self.refused += 1
                raise _Throttled
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="bothkinds", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=Unlucky(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.connection_retries == 2
    assert outcome.rate_limit_retries == 2
    assert len(waits.delays) == 4
    # The third wait is the first throttled one. Shared counters would have made it the fourth
    # step of a single schedule -- four seconds and upwards -- rather than the base again.
    assert waits.delays[2] <= RATE_LIMIT_BASE_DELAY
    assert outcome.connection_seconds == pytest.approx(sum(waits.delays[:2]))
    assert outcome.throttled_seconds == pytest.approx(sum(waits.delays[2:]))


async def test_a_lost_connection_composes_with_dropping_an_option() -> None:
    """The two retries are nested, so a turn that meets both still gets both.

    Written flat, whichever loop was outermost would swallow the other. That is not
    hypothetical: the rate-limit retry was placed beside the option drop first, and a 429 fell
    straight through a handler that only knew what to do with an option the provider had named.
    """
    waits = _Waits()

    class Awkward(StubChatClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.dropped = 0

        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if not self.dropped:
                self.dropped += 1
                raise _Disconnected
            if "temperature" in options:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="dropopt", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=Awkward(), model="stub", options={"temperature": 0.0}),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        sleep=waits,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total
    assert outcome.dropped_options == ("temperature",)
    assert outcome.connection_retries == 1
    assert len(waits.delays) == 1


async def test_every_probe_answer_reaches_the_score() -> None:
    """Every question, asked every time, must reach the scorer.

    With several targeted closing questions the answer is their union, and each is now asked
    repeatedly. Collecting the parts but never joining them leaves the answer empty and every
    fact scores as ignored -- a control that recalls nothing looks like a stable measurement
    rather than a broken one.
    """
    scenario = build_live_scenario(salt="join", filler_turns=3, filler_tokens=50, tool_turns=6)
    assert scenario.answer_turn_count > 1, "the default scenario should close with several questions"

    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=3,
    )

    assert outcome.answer, "no probe answer reached the scorer"
    assert len(outcome.probes) == probe_count(
        scenario.answer_scopes, probe_repeats=3, combined_repeats=DEFAULT_COMBINED_REPEATS
    )
    assert outcome.answer.count("a reply with some body to it") == probe_count(
        scenario.answer_scopes, probe_repeats=3, combined_repeats=DEFAULT_COMBINED_REPEATS
    )


async def test_run_live_reports_a_failed_turn_without_raising() -> None:
    """A failing turn must return a partial result, not throw away the spend already made."""

    class Exploding(StubChatClient):
        def _inner_get_response(self, *, messages: Any, stream: Any, options: Any, **kwargs: Any) -> Any:
            if len(self.seen) >= 2:
                raise RuntimeError("provider fell over")
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)

    scenario = build_live_scenario(salt="fail", filler_turns=3, filler_tokens=50)
    outcome = await run_live(
        ProviderRuntime(client=Exploding(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
    )

    assert outcome.error is not None
    assert "provider fell over" in outcome.error
    assert outcome.turns_completed < outcome.turns_total
    assert outcome.calls, "the calls made before the failure were lost"


async def test_facts_the_agent_never_fetched_are_not_charged_to_compaction() -> None:
    """A tool the agent never calls cannot have been evicted by compaction.

    The tool result only enters the history if the model asks for it. Counting an unasked-for
    fact as lost makes the *uncompacted* control report losing information to compaction,
    which cannot happen, and inflates every strategy's apparent damage by the same amount.
    """
    scenario = build_live_scenario(salt="nf", filler_turns=3, filler_tokens=50, tool_turns=6)
    # A client that never emits a tool call: nothing is ever fetched.
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
    )

    assert outcome.scopes_called == ()
    never_fetched = unretrieved_facts(outcome, scenario)
    tool_facts = [fact for fact in scenario.facts if fact.kind == "tool_result"]
    assert len(never_fetched) == len(tool_facts)


async def test_fetched_scopes_are_recorded() -> None:
    """Scopes the agent did ask for must not be counted as never fetched."""
    scenario = build_live_scenario(salt="f", filler_turns=3, filler_tokens=50, tool_turns=6)
    client = StubChatClient(tool_turns=(1,))
    outcome = await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
    )

    assert "early" in outcome.scopes_called
    never_fetched = {entry.fact.marker for entry in unretrieved_facts(outcome, scenario)}
    assert not never_fetched.intersection(scenario.tool_lookups["early"])


async def test_run_live_drives_every_turn_and_counts_tool_use() -> None:
    """The happy path must run the whole scenario and record the tool calls it made."""
    scenario = build_live_scenario(salt="ok", filler_turns=3, filler_tokens=50)
    client = StubChatClient(tool_turns=(1,), usage=UsageDetails(input_token_count=100, output_token_count=10))
    outcome = await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
    )

    assert outcome.error is None
    assert outcome.turns_completed == outcome.turns_total == len(scenario.transcript.turns)
    assert outcome.tool_calls_made == 1
    assert outcome.input_tokens > 0
    assert len(outcome.calls) > outcome.turns_total, "the tool round trip should add a call"


# endregion


# region probes


def _probe_scenario() -> Any:
    return build_live_scenario(salt="probe", filler_turns=3, filler_tokens=50, tool_turns=6)


class ClosingAnswerStub(StubChatClient):
    """A stub that says something distinctive only when it is asked a closing question.

    The seeding replies have to stay ordinary. A stub that answered with the marker on every
    turn would plant it in the snapshot itself, and a test asking whether the marker reached a
    probe's prompt would then pass for entirely the wrong reason.
    """

    def __init__(self, *, closing: Sequence[str], answer: str, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            closing: The closing question texts, which mark a probe.
            answer: What to reply to those, and to nothing else.
        """
        super().__init__(**kwargs)
        self.closing = tuple(closing)
        self.answer = answer

    def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any], **kwargs: Any
    ) -> Any:
        asked = _turn_text(messages[-1:]) if messages else ""
        # Set before the response is built and read a moment later, when it is awaited. Calls
        # here are strictly sequential, so the two cannot disagree.
        self.reply = self.answer if any(question in asked for question in self.closing) else "an ordinary reply"
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def _closing_questions(scenario: Any) -> list[str]:
    """Return the text of every closing question, which is what marks a call as a probe."""
    turns = scenario.transcript.turns
    return [_turn_text(turn.request) for turn in turns[len(turns) - scenario.answer_turn_count :]]


async def _probed(
    client: StubChatClient, *, scenario: Any = None, strategy: str = "none", repeats: int = 3
) -> tuple[LiveOutcome, Any]:
    """Run one scenario through the seed, snapshot and probe phases.

    Returns:
        The outcome and the scenario it was driven from.
    """
    scenario = scenario if scenario is not None else _probe_scenario()
    outcome = await run_live(
        ProviderRuntime(client=client, model="stub"),
        strategy_name=strategy,
        options=_options(),
        scenario=scenario,
        probe_repeats=repeats,
    )
    assert outcome.error is None
    return outcome, scenario


async def test_each_question_is_asked_its_default_number_of_times() -> None:
    """The default must be several readings of one snapshot, not one.

    A single reading of a two-valued accuracy distribution is a draw, and the whole point of
    separating the two spreads is that there is something to take a spread of.
    """
    scenario = _probe_scenario()
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
    )

    assert outcome.probe_repeats == 3
    assert len(outcome.probes) == probe_count(
        scenario.answer_scopes, probe_repeats=3, combined_repeats=DEFAULT_COMBINED_REPEATS
    )
    for scope in scenario.answer_scopes:
        asked = [probe for probe in outcome.probes if probe.scope == scope]
        wanted = DEFAULT_COMBINED_REPEATS if scope == "*" else 3
        assert [probe.repeat for probe in asked] == list(range(1, wanted + 1)), (
            f"{scope} was not asked its own number of times"
        )


async def test_the_combined_question_keeps_its_own_attempts_when_the_others_drop_to_one() -> None:
    """acc2's repeat count must not follow --probe-repeats.

    One acc1 reading averages every scoped question; one acc2 reading is a single answer. The runs
    that matter set --probe-repeats to 1, the per-scope repeat spread having measured 0 to 2
    points, and acc2 was then one sample per seed against acc1's seven -- which is the whole
    reason it was the noisier of the two.
    """
    scenario = _probe_scenario()
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=1,
    )

    assert outcome.error is None
    assert outcome.combined_repeats == DEFAULT_COMBINED_REPEATS
    combined = [probe for probe in outcome.probes if probe.scope == COMBINED_SCOPE]
    assert [probe.repeat for probe in combined] == list(range(1, DEFAULT_COMBINED_REPEATS + 1)), (
        "the combined question followed --probe-repeats"
    )
    for scope in scenario.answer_scopes:
        if scope != COMBINED_SCOPE:
            asked = [probe for probe in outcome.probes if probe.scope == scope]
            assert [probe.repeat for probe in asked] == [1], f"{scope} was asked more than once"
    # The count the dry run prices the cell on, against the probes the runner actually sent.
    assert probe_count(scenario.answer_scopes, probe_repeats=1, combined_repeats=DEFAULT_COMBINED_REPEATS) == len(
        outcome.probes
    )


async def test_the_combined_repeat_count_is_the_one_that_is_asked_for() -> None:
    """Neither count may be quietly clamped to the other.

    The pair is the point: three probe repeats and two combined attempts is a legitimate cell,
    and a run that silently asked one of them the other's number of times would report a spread
    over material it never gathered.
    """
    scenario = _probe_scenario()
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=3,
        combined_repeats=2,
    )

    assert outcome.error is None
    assert (outcome.probe_repeats, outcome.combined_repeats) == (3, 2)
    asked = {scope: 0 for scope in scenario.answer_scopes}
    for probe in outcome.probes:
        asked[probe.scope] += 1
    assert asked.pop(COMBINED_SCOPE) == 2
    assert set(asked.values()) == {3}


async def test_a_probe_never_sees_another_probes_answer() -> None:
    """No probe's answer may reach any other probe's prompt.

    This is the change. When the closing questions were ordinary turns, each answer re-listed
    codes into the history as assistant text, so the next question was asked from a context
    the previous answer had rewritten -- and survival, scored against that context, credited
    compaction for a code the model had recited two questions earlier. The same strategy read
    53/53 on a run that emitted 10,941 output tokens and 18/53 on one that emitted 4,873.
    """
    scenario = _probe_scenario()
    stub = ClosingAnswerStub(closing=_closing_questions(scenario), answer="ANSWER-MARKER answering now")
    outcome, _ = await _probed(stub, scenario=scenario)

    assert len(outcome.probes) > 1, "one probe cannot contaminate anything, so this proves nothing"
    assert "ANSWER-MARKER" in outcome.answer, "no probe said the marker, so nothing could have leaked"
    for probe in outcome.probes:
        assert "ANSWER-MARKER" not in probe.prompt_text, f"a probe answer reached the {probe.scope} probe"
    assert "ANSWER-MARKER" not in outcome.snapshot_prompt


async def test_every_probe_is_asked_from_the_same_restored_snapshot() -> None:
    """Probe N's prompt must equal probe 1's apart from the question.

    A shallow restore passes every other check here and fails this one: compaction marks its
    exclusions on the message objects themselves, so a snapshot sharing them is rewritten by
    the first probe and every later probe starts from somewhere else.
    """
    outcome, _ = await _probed(StubChatClient())

    # The question is the last message of the prompt; everything before it is the snapshot.
    contexts = {probe.prompt_text.rsplit(chr(10), 1)[0] for probe in outcome.probes}
    questions = {probe.prompt_text.rsplit(chr(10), 1)[1] for probe in outcome.probes}

    assert len(contexts) == 1, "the probes were asked from different contexts"
    assert len(questions) == len({probe.question for probe in outcome.probes})


async def test_the_snapshot_survives_the_probes_that_read_it() -> None:
    """Restoring must not hand the next probe the copy the last one mutated.

    Restoring is a fresh deep copy each time rather than one copy assigned once, because the
    probe about to run will mark exclusions on whatever it is given.
    """
    outcome, _ = await _probed(StubChatClient(), strategy="truncation")
    before = outcome.snapshot_prompt

    assert before, "nothing was snapshotted, so the comparison is vacuous"
    assert all(probe.prompt_text.startswith(before) for probe in outcome.probes), (
        "a probe was asked from a context that was not the snapshot"
    )


async def test_compaction_cannot_accumulate_across_the_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The strategy must meet the same history on every probe.

    The old design let compaction keep firing through the closing turns, so the first scope
    was answered from a fuller context than the last and the combined question from the most
    compacted context of the run. Restoring the snapshot makes that structurally impossible
    rather than switchable, and this is the assertion that says so.
    """
    scenario = _probe_scenario()
    seen: list[int] = []

    async def counting(messages: list[Message]) -> bool:
        seen.append(len(messages))
        return False

    monkeypatch.setattr("agent_framework_lab_cachebench._live.build_strategy", lambda name, options: counting)
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="truncation",
        options=_options(),
        scenario=scenario,
        probe_repeats=3,
    )

    assert outcome.error is None
    # Two invocations per probe: the before phase runs as the agent's compaction_strategy
    # inside the client, and the after phase runs on the provider once the reply is in hand,
    # so it sees one message more. Sliced apart, each must be flat.
    probe_calls = seen[len(seen) - 2 * len(outcome.probes) :]
    assert len(set(probe_calls[0::2])) == 1, f"the before phase saw a moving history: {sorted(set(probe_calls[0::2]))}"
    assert len(set(probe_calls[1::2])) == 1, f"the after phase saw a moving history: {sorted(set(probe_calls[1::2]))}"


async def test_a_strategy_that_acts_again_on_the_snapshot_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring stops compaction accumulating; it does not stop it acting once more.

    Survival is scored against the snapshot, so a probe that was sent less than the snapshot
    would be credited with facts the model never saw. It cannot be prevented -- the strategy
    has to run, or the probe would be answered from a context no row ever had -- so it is
    counted and flagged instead of assumed away.
    """
    scenario = _probe_scenario()
    closing = _closing_questions(scenario)

    async def evicting(messages: list[Message]) -> bool:
        """Leave the seeding alone and evict one message on the way into each probe.

        Quiet during seeding so the snapshot is the whole conversation, which is what makes the
        difference between the snapshot and a probe's prompt attributable to this and nothing
        else.
        """
        asked = _turn_text(messages[-1:]) if messages else ""
        if not any(question in asked for question in closing):
            return False
        for message in messages:
            if not message.additional_properties.get("_excluded"):
                message.additional_properties["_excluded"] = True
                return True
        return False

    monkeypatch.setattr("agent_framework_lab_cachebench._live.build_strategy", lambda name, options: evicting)
    outcome = await run_live(
        ProviderRuntime(client=StubChatClient(), model="stub"),
        strategy_name="truncation",
        options=_options(),
        scenario=scenario,
        probe_repeats=2,
    )

    assert outcome.error is None
    assert outcome.context_drift == len(outcome.probes), "the drift went unnoticed"
    # Still identical to each other, which is the property restoring the snapshot buys: the
    # strategy acts once on each probe rather than once more on each probe than on the last.
    assert len({probe.prompt_text.rsplit(chr(10), 1)[0] for probe in outcome.probes}) == 1


async def test_a_settled_strategy_reports_no_drift() -> None:
    """The usual case must read zero, or the flag says nothing when it appears."""
    outcome, _ = await _probed(StubChatClient(), strategy="none", repeats=2)

    assert outcome.probes
    assert outcome.context_drift == 0


async def test_survival_is_scored_against_the_snapshot_not_the_answers() -> None:
    """A fact the model produced but was never given must not score as surviving.

    The model here quotes every planted code while never calling a single tool, so no tool
    result ever entered the history. Scored against a prompt that had already carried an
    answer, those codes read as preserved by compaction; scored against the snapshot they read
    as what they are, which is a model producing values it was not shown.
    """
    scenario = _probe_scenario()
    codes = " ".join(fact.marker for fact in scenario.facts)
    stub = ClosingAnswerStub(closing=_closing_questions(scenario), answer=f"here they are: {codes}")
    outcome, _ = await _probed(stub, scenario=scenario)
    tool_facts = {fact.marker for fact in scenario.facts if fact.kind == "tool_result"}

    assert outcome.scopes_called == (), "the agent fetched a tool, so this proves nothing"
    assert tool_facts, "the scenario planted no tool facts"
    for sample in score_samples(outcome, scenario):
        for entry in sample:
            if entry.fact.marker in tool_facts:
                assert not entry.survived, f"{entry.fact.marker} was never in the history but scored as surviving"


def test_a_prompt_over_the_tried_limit_disqualifies_the_run() -> None:
    """A row whose prompt exceeded the limit it stands in for is not a row.

    The limit is simulated -- the model itself accepts 272,000 -- so nothing but this check
    enforces it. Without it the 60,000 control ran at 78,003 tokens and was ranked anyway, and
    every "cheaper than not compacting" at that size was measured against a baseline no model
    of that size could have produced.
    """
    within = LiveOutcome(
        strategy="none",
        calls=(_call(2, 2, inp=57_000), _call(2, 2, inp=40_000)),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )
    over = LiveOutcome(
        strategy="none",
        calls=(_call(2, 2, inp=40_000), _call(2, 2, inp=78_003)),
        answer="",
        snapshot_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )

    assert not within.disqualified(60_000)
    # Any call, not only the last: the run overran while it was being seeded and the closing
    # prompts were smaller, which is exactly the shape the 60,000 cell had.
    assert over.disqualified(60_000)


async def test_a_disqualified_cell_is_excluded_from_the_ranking() -> None:
    """A cell that disqualifies at all leaves the ranking rather than being starred.

    Ranked with an asterisk it still sets the baseline every other row is compared against,
    which is the failure the asterisk was supposed to warn about.
    """
    cells = []
    for name, usage in (("none", 1_000), ("truncation", 90_000)):
        outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=usage)), repeats=1)
        cells.append(_aggregate(name, [_record(outcome, scenario, strategy=name)]))

    incomplete, oversized, diverged = _excluded_cells(cells)
    ranked = [cell.strategy for cell in cells if cell.strategy not in incomplete | oversized | diverged]

    assert cells[0].disqualified == 0.0
    assert cells[1].disqualified == 1.0
    assert oversized == {"truncation"}
    assert ranked == ["none"]


async def test_the_dq_flag_says_what_the_dq_column_says() -> None:
    """Two exclusions must not share one name.

    The flag used to read "excluded from the ranking", which is the wider set: a row that
    failed a turn leaves the ranking too. The last sweep printed every row flagged DQ beside a
    dq of 0%, because every row had died on a rate limit and none had oversent -- so the flag
    said the one thing that had not happened. ``dq`` still means what it always meant: the
    share of a cell's seeds that sent a prompt over the tried limit.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    base = _record(outcome, scenario)
    unfinished = _aggregate("none", [replace(base, turns_completed=0, disqualified=False)])
    oversized = _aggregate("none", [replace(base, disqualified=True)])

    assert "EXCL" in _row(unfinished, None, True, 60_000)
    assert "DQ" not in _row(unfinished, None, True, 60_000)
    assert "DQ" in _row(oversized, None, True, 60_000)
    # Both are out of the ranking, and for different reasons: the notes under the table name
    # each of them, so the flag only has to say which kind this row is.
    assert _excluded_cells([unfinished, oversized]) == ({"none"}, {"none"}, set())


async def test_a_throttled_row_says_so_in_the_table() -> None:
    """A run that spent its wall clock backing off is a different run, and had to say so.

    Nothing else in the table can show it. Every column reads the same as an unthrottled run
    except hit%, which can move for a reason that is not compaction's: a cached prefix that
    expired during a minute of waiting is a miss the row would otherwise be charged for.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="flag", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=ThrottlingStub(refusals=2, usage=UsageDetails(input_token_count=1_000)), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=1,
        sleep=waits,
    )
    cell = _aggregate("none", [_record(outcome, scenario)])

    assert cell.rate_limit_retries == 2
    assert cell.throttled_seconds == pytest.approx(sum(waits.delays))
    assert "THROTTLED:2" in _flags(cell, None)

    table = _render(None, [cell], set(), show_answers=False)

    assert "THROTTLED:2" in table
    assert "Throttled:" in table, "the seconds are the point, and the flag cannot carry them"


async def test_a_reconnected_row_says_so_separately_from_a_throttled_one() -> None:
    """The two retries are different events and must not read as one.

    A throttled row waited out a quota window, which is wide enough that its cached prefix may
    have gone with it, so its hit% carries a caveat. A reconnected row waited seconds and
    re-sent the same prefix. One flag for both would put that caveat on every row that merely
    survived a blip, and would hide which of the two a sweep is actually losing time to.
    """
    waits = _Waits()
    scenario = build_live_scenario(salt="flagdrop", filler_turns=3, filler_tokens=50, tool_turns=6)
    outcome = await run_live(
        ProviderRuntime(client=DisconnectingStub(drops=2, usage=UsageDetails(input_token_count=1_000)), model="stub"),
        strategy_name="none",
        options=_options(),
        scenario=scenario,
        probe_repeats=1,
        sleep=waits,
    )
    cell = _aggregate("none", [_record(outcome, scenario)])

    assert cell.connection_retries == 2
    assert cell.connection_seconds == pytest.approx(sum(waits.delays))
    assert cell.rate_limit_retries == 0
    assert "RECONNECTED:2" in _flags(cell, None)
    assert not any(flag.startswith("THROTTLED") for flag in _flags(cell, None))

    table = _render(None, [cell], set(), show_answers=False)

    assert "RECONNECTED:2" in table
    assert "Reconnected:" in table, "the seconds are the point, and the flag cannot carry them"
    assert "Throttled:" not in table


async def test_a_declined_collapse_reaches_the_flags_column() -> None:
    """A row that never fired and a row that fired to no effect must not read the same.

    ``anchored_min_gain`` can leave a conversation untouched for two opposite reasons: there
    was nothing in the band worth shortening, or there was and the saving would not have
    repaid the prompt cache the edit spends. Every other column reads identically in the two
    cases -- same size, same cost, same facts -- so without the count the table cannot say
    which of them the run measured, and the floor is exactly the thing being measured.
    """
    strategy = build_strategy("anchored_min_gain", _options())
    assert strategy is not None
    messages: list[Message] = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        Message(role="user", contents=["Requirement: region is EU-WEST-1."], message_id="u0"),
        Message(role="assistant", contents=["Understood."], message_id="a0"),
    ]
    for index in range(8):
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
                contents=[{"type": "function_result", "call_id": call_id, "result": f"R{index} " + "x" * 5_000}],
                message_id=f"t_res_{index}",
            ),
            Message(role="assistant", contents=[f"I looked up {index}."], message_id=f"a_txt_{index}"),
        ]

    assert await strategy(messages) is False
    notes = _strategy_notes(strategy)

    assert notes == ("NOGAIN:1",)

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = replace(_record(outcome, scenario, strategy="anchored_min_gain"), strategy_notes=notes)
    cell = _aggregate("anchored_min_gain", [record])

    assert "NOGAIN:1" in _flags(cell, None)
    assert "NOGAIN:1" in _render(None, [cell], set(), show_answers=False)


async def test_the_table_renders_every_column_it_declares() -> None:
    """The table must actually print, with a legend entry for every column it shows.

    Nothing else exercises the formatting, and it is entirely f-strings: a mis-specified
    width or a missing key raises only at the end of a paid run, after every call has been
    made. The legend is checked against the header for the same reason a flag list is checked
    against the parser -- a column that appears with no explanation is how "correct" was read
    as the median-cost repeat for three matrices running.
    """
    cells = []
    for name, usage in (("none", 1_000), ("truncation", 900)):
        outcome, scenario = await _probed(
            StubChatClient(usage=UsageDetails(input_token_count=usage, output_token_count=20)), repeats=2
        )
        cells.append(_aggregate(name, [_record(outcome, scenario, strategy=name)]))
    verdict = recommend([_to_joint(cell) for cell in cells])

    table = _render(verdict, cells, set(), show_answers=False)

    header = next(line for line in table.splitlines() if line.strip().startswith("strategy"))
    # Split on the padding between fields, not on spaces: two columns have a space in the name.
    explained = {line.split("=", 1)[0].strip() for line in table.splitlines() if "=" in line}
    for column in re.split(r"\s{2,}", header.strip()):
        if column not in {"strategy", "flags"}:
            # By its first word, since a header cell can carry a "left/peak" qualifier the
            # legend explains in its body rather than in its key.
            assert column in explained or column.split()[0] in explained, f"the {column!r} column has no legend entry"
    assert "per-sample acc1" in table
    assert "per-sample acc2" in table
    assert "VERDICT:" in table


def _body(table: str) -> list[str]:
    """Return the table's rows, the split line included, and nothing above or below them."""
    lines = table.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("strategy")) + 2
    end = next(index for index, line in enumerate(lines[start:], start) if not line.strip())
    return lines[start:end]


def _order(table: str) -> list[str]:
    """Return the strategy of each row, in the order the table printed them."""
    return [line.split()[0] for line in _body(table) if not line.startswith("-")]


def _ranked_cell(record: SeedRecord, *, strategy: str, cost: float, correctness: float) -> Any:
    """Return a one-seed row carrying the cost and the accuracy the ordering is tested on."""
    return _aggregate(strategy, [replace(record, strategy=strategy, cost=cost, correctness_samples=(correctness,))])


async def test_the_input_cost_column_prices_the_prompt_side_and_nothing_else() -> None:
    """The new column has to be the input half of the money, at the cell's own rates.

    It exists because the total is dominated by something compaction does not touch: on a
    clean five-seed control the total moved 38% while input tokens moved 13% and the hit rate
    4 points, all of it output, which is priced 57 times a cache read here. Three of five rows
    then differed from the control by less than the control's own spread. Priced any other way
    -- output folded in, the cached share at the full rate -- it would be a second cost column
    rather than the quiet reading of the one already there.
    """
    outcome, scenario = await _probed(
        StubChatClient(
            usage=UsageDetails(input_token_count=1_000, output_token_count=20, cache_read_input_token_count=300)
        ),
        repeats=1,
    )
    record = _record(outcome, scenario)
    fresh = record.input_tokens - record.cached_tokens
    expected = (fresh * PRICING.input_per_million + record.cached_tokens * PRICING.cached_read_per_million) / 1_000_000

    assert record.cached_tokens > 0, "with nothing cached the discounted rate is never exercised"
    assert record.input_cost == pytest.approx(expected)
    # The whole difference is output and the summarizer, which is what "input alone" means.
    assert record.output_tokens > 0
    assert record.input_cost < record.cost
    assert record.cost - record.input_cost == pytest.approx(
        record.output_tokens * PRICING.output_per_million / 1_000_000 + record.summarizer_cost
    )
    # Derived from tokens and rates the record already carries, so nothing about the format
    # moved and every file on disk gains the column.
    assert "input_cost" not in record.to_dict()
    assert record.schema == SCHEMA_VERSION
    assert SeedRecord.from_dict(record.to_dict()).input_cost == pytest.approx(expected)

    cell = _aggregate("none", [record])

    assert cell.input_cost == pytest.approx(expected)
    # What the table prints is the seeding half of it, because that is the half the ranking is
    # on: the whole-run figure carries twelve re-reads of the snapshot that nobody deploys.
    assert cell.seeding_input_cost is not None
    assert cell.seeding_input_cost < cell.input_cost
    assert f"${cell.seeding_input_cost:.4f}" in _render(None, [cell], set(), show_answers=False)


async def test_the_two_phases_add_back_to_what_the_run_was_billed() -> None:
    """Splitting the money must not create or lose any of it.

    Taken as the whole seed less the probing rather than summed over the seeding calls, so the
    two halves reconcile whatever else the run did. Anything that is neither an answered probe
    nor a seeding turn -- the summarizer, a probe that failed before it produced an outcome --
    lands in the workload, which is the direction that cannot flatter a strategy.
    """
    outcome, scenario = await _probed(
        StubChatClient(
            usage=UsageDetails(input_token_count=1_000, output_token_count=20, cache_read_input_token_count=300)
        ),
        repeats=2,
    )
    record = _record(outcome, scenario)

    assert record.probe_cost is not None
    assert record.seeding_cost is not None
    assert record.probe_cost > 0, "a run that probed for nothing is not measuring the probes"
    assert record.seeding_cost + record.probe_cost == pytest.approx(record.cost)
    assert record.seeding_input_cost is not None
    assert 0 < record.seeding_input_cost < record.input_cost, "the prompt side has to split the same way"
    # Measured off the probes rather than modelled from the last prompt, which is what makes
    # this exact: every probe carries the same snapshot but not at the same hit rate.
    assert record.probe_input_tokens == sum(call.input_tokens for probe in outcome.probes for call in probe.calls)
    assert record.probe_output_tokens == sum(call.output_tokens for probe in outcome.probes for call in probe.calls)
    assert 0 < (record.probe_input_tokens or 0) < record.input_tokens
    assert SeedRecord.from_dict(record.to_dict()).seeding_cost == pytest.approx(record.seeding_cost)


async def test_a_record_written_before_the_phase_split_says_so_rather_than_guessing() -> None:
    """The ten recorded cells counted their calls in one total, and nothing recovers the halves.

    The per-probe prompts are not on those records; twelve probes priced at
    ``prompt_tokens_final`` is a model of the run, not the run. Zero would be worse than a
    guess -- it would hand a reader the correction already applied, and wrong, since every one
    of those runs probed twelve times.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    written = _record(outcome, scenario).to_dict()
    written["schema"] = 3
    for name in ("probe_input_tokens", "probe_cached_tokens", "probe_output_tokens"):
        del written[name]

    old = SeedRecord.from_dict(written)

    assert old.probe_input_tokens is None
    assert (old.probe_cost, old.seeding_cost, old.seeding_input_cost) == (None, None, None)
    assert old.cost > 0, "everything the record did measure still reads"
    assert _aggregate("none", [old]).seeding_cost is None


async def test_a_cheap_lossy_row_ranks_below_a_dearer_faithful_one() -> None:
    """Cost ascending on its own promotes whichever strategy destroyed the most.

    That is not a stray case: the cheapest row of a cell is reliably the one that threw the
    conversation away, and it sorted above a strategy that kept the conversation and cost a
    little more. So the rows that still answer are ranked first, and the reader is not asked
    to infer the difference from a column eleven places to the right.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    record = _record(outcome, scenario)
    control = _ranked_cell(record, strategy="none", cost=0.10, correctness=1.0)
    lossy = _ranked_cell(record, strategy="threw_it_away", cost=0.01, correctness=0.2)
    faithful = _ranked_cell(record, strategy="kept_it", cost=0.05, correctness=0.95)
    cells = [control, lossy, faithful]

    table = _render(None, cells, set(), show_answers=False)

    assert min(cells, key=lambda cell: cell.cost).strategy == "threw_it_away", "the old order put this first"
    assert _order(table) == ["kept_it", "none", "threw_it_away"]
    # Cheapest first inside each group, and the group boundary between them rather than a
    # reader working out which rows are comparable.
    assert [line.startswith("-") for line in _body(table)] == [False, False, True, False]
    assert "Ranking: 2 of 3 rows kept at least 90%" in table
    assert "below 90% of the control's acc1" in table


def _turn_list_cells(record: SeedRecord, control_peak: int) -> list[CellStats]:
    """Return one cell's rows, differing only in how many messages each reached.

    Args:
        record: The record to shape every row from.
        control_peak: Peak message count to give the control.

    Returns:
        The control, a strategy that adds nothing of its own, and one that adds a record and
        the call that fetches it -- which is a legitimate reason to sit above the turn list
        and so the reason the guard reads the leanest row rather than every row.
    """
    peaks = {"none": control_peak, "truncation": 109, "tool_summary_anchored": 121}
    return [_aggregate(name, [replace(record, strategy=name, messages_peak=peak)]) for name, peak in peaks.items()]


async def test_a_control_that_ran_another_conversation_is_flagged_and_out_of_the_ranking() -> None:
    """A baseline carrying fewer messages than its rows is a broken measurement, not a cheap row.

    Compaction excludes and rewrites in place; it never deletes from the stored history, and
    what it adds is its own. So the leanest strategy row is the turn list at its own size and
    the control, which adds nothing, has to equal it. Recorded at 120,000/0.86 it did not: 82
    against 109, which makes every cost figure in that cell a comparison between two different
    conversations. Ranked silently, that reads as compaction being expensive.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    cells = _turn_list_cells(_record(outcome, scenario), control_peak=82)

    assert _control_message_gap(cells) == -27
    assert _excluded_cells(cells) == (set(), set(), {"none"})

    table = _render(None, cells, {"none"}, show_answers=False)
    control_row = next(line for line in _body(table) if line.startswith("none"))

    assert "MSGS:-27" in control_row
    assert "EXCL" in control_row
    assert "CONTROL DIVERGED" in table
    assert "NO VERDICT: the 'none' row ran a different conversation" in table
    # The one column the divergence invalidates, withdrawn rather than printed as a number.
    # These rows measured their phases, so every other money column is a figure and the single
    # question mark on the line is the comparison that has been taken away.
    assert next(line for line in _body(table) if line.startswith("truncation")).count("?") == 1


async def test_a_control_that_matches_its_rows_is_left_alone() -> None:
    """The guard has to be quiet on a sound cell, or it says nothing when it fires.

    The row above the turn list is the point: a strategy that fetches a record back adds a
    call and its result, so equality with *every* row would flag the sound case forever.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    cells = _turn_list_cells(_record(outcome, scenario), control_peak=109)

    assert _control_message_gap(cells) is None
    assert _excluded_cells(cells) == (set(), set(), set())

    table = _render(None, cells, set(), show_answers=False)

    assert not [line for line in _body(table) if "MSGS" in line]
    assert "CONTROL DIVERGED" not in table
    assert "?" not in next(line for line in _body(table) if line.startswith("truncation"))


def _phase_cell(record: SeedRecord, *, strategy: str, cost: float, probe_input: int | None) -> CellStats:
    """Return a one-seed row whose invoice and whose workload can be set against each other.

    Args:
        record: The record to shape.

    Keyword Args:
        strategy: Name the row under.
        cost: What the whole run was billed.
        probe_input: Input tokens the probe phase billed, or None for a record written before
            the phases were counted apart.

    Returns:
        The aggregated row.
    """
    return _aggregate(
        strategy,
        [
            replace(
                record,
                strategy=strategy,
                cost=cost,
                probe_input_tokens=probe_input,
                probe_cached_tokens=None if probe_input is None else 0,
                probe_output_tokens=None if probe_input is None else 0,
            )
        ],
    )


async def test_the_ranking_is_on_the_workload_and_not_on_the_invoice() -> None:
    """A row dearer on the whole run but cheaper on the conversation must rank first.

    Every probe is asked from the restored snapshot, so each one re-sends the snapshot whole,
    and a strategy with a small snapshot collects that discount once per probe -- on a phase no
    deployed agent has, since an agent continues the conversation instead of being interrogated
    twelve times from a frozen state. Ranked on the invoice, the measurement pays a strategy
    for being cheap to measure. Stripping the probes turned one recorded cell's -14.1% into
    -3.5% and reversed the sign on two of three.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    record = _record(outcome, scenario)
    # 1.00 - 0.50 of probing against 1.10 - 0.80: the second is the dearer run and the
    # cheaper conversation, which is the only pair of numbers that can tell the two apart.
    invoice = _phase_cell(record, strategy="cheap_to_measure", cost=1.00, probe_input=500_000)
    workload = _phase_cell(record, strategy="cheap_to_run", cost=1.10, probe_input=800_000)
    cells = [invoice, workload]

    assert invoice.cost < workload.cost
    assert workload.seeding_cost is not None and invoice.seeding_cost is not None
    assert workload.seeding_cost < invoice.seeding_cost
    assert _order(_render(None, cells, set(), show_answers=False)) == ["cheap_to_run", "cheap_to_measure"]


async def test_a_cell_that_cannot_split_its_phases_ranks_on_the_invoice_and_says_so() -> None:
    """The correction cannot be applied to a record that never counted its probing.

    The per-probe prompts are not on those records, and pricing twelve probes at the final
    prompt's size is a model of the run rather than the run. So the ranking falls back to what
    the run was billed -- the older and dirtier order, which is exactly the one the split
    exists to replace -- and every money column that would have described the workload reads
    '?' instead of quietly describing something else.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    record = _record(outcome, scenario)
    invoice = _phase_cell(record, strategy="cheap_to_measure", cost=1.00, probe_input=None)
    workload = _phase_cell(record, strategy="cheap_to_run", cost=1.10, probe_input=None)
    cells = [invoice, workload]

    table = _render(None, cells, set(), show_answers=False)

    assert invoice.seeding_cost is None and invoice.probe_cost is None
    assert _order(table) == ["cheap_to_measure", "cheap_to_run"], "the invoice order, since nothing else is known"
    assert "Ranking: run$, probes and all ascending" in table
    assert "NO PHASE SPLIT" in table
    assert all("NOSPLIT" in line for line in _body(table))
    assert all(line.count("?") >= 4 for line in _body(table)), "a money column read as a number it never measured"


async def test_the_split_holds_when_every_row_clears_and_when_none_does() -> None:
    """A cell where nothing is separated must still say what the order means.

    Both degenerate cases render as one block of rows, and they mean opposite things: every
    strategy usable, or none of them. Left to the line alone the two are indistinguishable,
    which is why the threshold and the count are stated above the table whether or not the
    line is drawn.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    record = _record(outcome, scenario)
    control = _ranked_cell(record, strategy="none", cost=0.10, correctness=1.0)
    faithful = _ranked_cell(record, strategy="kept_it", cost=0.05, correctness=0.95)
    lossy = _ranked_cell(record, strategy="threw_it_away", cost=0.01, correctness=0.2)

    every = _render(None, [control, faithful], set(), show_answers=False)

    assert _order(every) == ["kept_it", "none"]
    assert not [line for line in _body(every) if line.startswith("-")], "there are no two groups to separate"
    assert "Ranking: 2 of 2 rows kept at least 90%" in every

    # A control that scored nothing is a cell in which no row can be shown to have retained
    # anything, the control included: the line then sits above every row rather than vanishing.
    dead = _ranked_cell(record, strategy="none", cost=0.10, correctness=0.0)
    nothing = _render(None, [dead, faithful, lossy], set(), show_answers=False)

    assert _order(nothing) == ["threw_it_away", "kept_it", "none"]
    assert _body(nothing)[0].startswith("-")
    assert "Ranking: 0 of 3 rows kept at least 90%" in nothing


async def test_the_two_spreads_are_reported_apart() -> None:
    """Between-seed and within-seed disagreement must not arrive as one number.

    They are different findings. A strategy that scored 52, 52, 52 and 22 while preserving
    exactly the same 27 facts every time is telling us about the model; one whose seeds
    disagree is telling us about compaction. Reported together, the first reads as the second.
    """
    scenario = _probe_scenario()
    codes = [fact.marker for fact in scenario.facts]

    def outcome_with(answers: Sequence[str]) -> LiveOutcome:
        return LiveOutcome(
            strategy="none",
            calls=(_call(2, 2, inp=100),),
            answer=chr(10).join(answers),
            snapshot_prompt=" ".join(codes),
            tool_calls_made=0,
            turns_completed=1,
            turns_total=1,
            probe_repeats=len(answers),
            probes=tuple(
                ProbeOutcome(scope="", question="q", repeat=index, answer=answer, prompt_text="", calls=())
                for index, answer in enumerate(answers, start=1)
            ),
        )

    # One seed answered thoroughly every time, one that wandered: identical facts in front of
    # the model in both, so all of this belongs to the within-seed column.
    steady = _record(outcome_with([" ".join(codes)] * 3), scenario).correctness_samples
    wandering = _record(
        outcome_with([" ".join(codes), " ".join(codes[:2]), " ".join(codes)]), scenario
    ).correctness_samples

    assert _probe_spread([wandering]) > 0
    assert _seed_spread([wandering]) == 0.0, "one seed cannot show a between-seed spread"
    assert _probe_spread([steady]) == 0.0
    assert _seed_spread([steady, wandering]) > 0


async def test_acc2_is_the_mean_over_every_combined_attempt_of_every_seed() -> None:
    """Every attempt weighs the same, whatever number of them a seed made.

    Not the mean of the seed means: a seed asked the combined question three times would then
    count for as much as one asked it once, and a file merged from a run before the count
    existed and a run after it would be weighted by which run a seed came from.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    base = _record(outcome, scenario)
    three = replace(base, seed=1, combined_samples=(1.0, 0.5, 0.6))
    one = replace(base, seed=2, combined_samples=(0.2,))

    cell = _aggregate("none", [three, one])

    assert cell.combined == pytest.approx(fmean([1.0, 0.5, 0.6, 0.2]))
    assert cell.combined != pytest.approx(fmean([fmean((1.0, 0.5, 0.6)), 0.2])), "the seed means were averaged"
    # The seed read once contributes no spread rather than a spread against a missing value:
    # 50 points is the three-attempt seed alone, which is the only one that was read twice.
    assert cell.combined_spread == pytest.approx(50.0)
    assert cell.combined_samples == ((1.0, 0.5, 0.6), (0.2,))


async def test_a_seed_recorded_before_the_combined_count_still_aggregates_as_one_attempt() -> None:
    """A record on disk holds one combined sample, and it is one answer, not a failed three.

    Read as a third of three it would be scaled down to nothing; read as its own count it is
    exactly the acc2 the file has always rendered. The count itself is recoverable too: back
    then the combined question was one of the closing questions, so it was asked probe_repeats
    times.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    record = replace(_record(outcome, scenario, cell=_cell_params(probe_repeats=2)), combined_samples=(0.37,))
    written = record.to_dict()
    del written["cell"]["combined_repeats"]

    old = SeedRecord.from_dict(written)

    assert old.cell.combined_repeats == 2, "the count a version 2 record implies was not recovered"
    assert old.combined == pytest.approx(0.37)
    cell = _aggregate("none", [old])
    assert cell.combined == pytest.approx(0.37)
    assert cell.combined_spread == 0.0
    row = _row(cell, None, excluded=False, limit=60_000)
    assert " 37%" in row
    assert "per-sample acc2" in _render(None, [cell], set(), show_answers=False)


def test_the_two_accuracy_measures_are_named_acc1_and_acc2() -> None:
    """One run measured two ways has to read as that, in the header and everywhere else.

    ``acc`` and ``all`` named the questions rather than the measures, so nothing in the table
    said the two columns were the same run scored twice -- one over the scoped questions, one
    over the single combined one.
    """
    record = SeedRecord(
        cell=_cell_params(),
        strategy="none",
        seed=1,
        cost=0.01,
        summarizer_cost=0.0,
        input_tokens=1_000,
        cached_tokens=300,
        output_tokens=20,
        probe_input_tokens=400,
        probe_cached_tokens=120,
        probe_output_tokens=8,
        calls=4,
        messages_left=6,
        messages_peak=12,
        prompt_tokens_final=900,
        prompt_tokens_peak=1_000,
        seed_prompt_tokens=800,
        facts_total=53,
        facts_left=53,
        facts_lost=0,
        nofetch=0,
        correctness_samples=(0.9, 1.0),
        ignored_samples=(0, 0),
        combined_samples=(0.4, 0.6, 0.5),
        disqualified=False,
        context_drift=0,
        rate_limit_retries=0,
        throttled_seconds=0.0,
        connection_retries=0,
        connection_seconds=0.0,
        turns_completed=10,
        turns_total=10,
        probe_repeats=2,
        summarizer_calls=0,
        summarizer_failures=0,
        groups_kept_uncovered=0,
        fallbacks_after_record=0,
        records_in_conversation=0,
        strategy_notes=(),
        dropped_options=(),
        answer="",
    )

    table = _render(None, [_aggregate("none", [record])], set(), show_answers=False)

    header = next(line for line in table.splitlines() if line.strip().startswith("strategy"))
    columns = set(re.split(r"\s{2,}", header.strip()))
    assert {"acc1", "acc2", "rep2+-"} <= columns
    assert not columns & {"acc", "all"}, "the old column names are still in the header"
    explained = {line.split("=", 1)[0].strip() for line in table.splitlines() if "=" in line}
    assert {"acc1", "acc2", "rep2+-"} <= explained
    assert "per-sample acc1, one group per seed:" in table
    assert "per-sample acc2, one group per seed:" in table
    assert "[90% 100%]" in table, "the acc1 samples are not the per-scope repeats"
    assert "[40% 60% 50%]" in table, "the acc2 samples are not the combined attempts"


# endregion


def test_retrieval_guidance_can_be_switched_off() -> None:
    """The guidance clause must be removable, or its contribution cannot be measured.

    It is the only sentence suspected of holding the closing answer stable, and the run that
    would settle whether it or the reply cap caused the earlier bimodality needs an arm
    without it.
    """
    with_guidance = resolve_instructions("neutral", retrieval_guidance=True)
    without = resolve_instructions("neutral", retrieval_guidance=False)

    assert RETRIEVAL_GUIDANCE in with_guidance
    assert RETRIEVAL_GUIDANCE not in without
    # The rest of the instructions must survive: an arm that also changed the persona would
    # not isolate the clause.
    assert without and without in with_guidance


def test_retrieval_guidance_is_appended_to_narration_modes_that_lack_it() -> None:
    """Every narration mode gains the clause when asked, not just the neutral one."""
    for mode in ("prompted", "neutral", "suppressed"):
        assert RETRIEVAL_GUIDANCE in resolve_instructions(mode, retrieval_guidance=True)
        assert RETRIEVAL_GUIDANCE not in resolve_instructions(mode, retrieval_guidance=False)


def test_spread_placement_puts_codes_beyond_a_head_truncation() -> None:
    """Spread codes must not all survive the 4,096 characters a collapsed result keeps.

    ``ToolResultCompactionStrategy`` head-truncates, so head-placed codes survive it
    unconditionally and every tool-oriented strategy scores a perfect result for free. That
    made the accuracy column say more about where the scenario put its markers than about
    what compaction preserves.
    """
    codes = tuple(f"AA-{index:04d}" for index in range(8))
    lookups = {"early": codes}

    head = make_scope_tools(lookups, 8_000, narration="neutral", placement="head")[0]()
    spread = make_scope_tools(lookups, 8_000, narration="neutral", placement="spread")[0]()

    assert all(head.index(code) < 4_096 for code in codes)
    assert sum(1 for code in codes if spread.index(code) < 4_096) <= 2
    # Both carry every code, so the two placements differ in position only and the accuracy
    # ceiling before compaction is identical.
    assert all(code in spread for code in codes)


def test_spread_placement_does_not_change_the_result_size() -> None:
    """Placement must not move the cost axis, or the two arms are not comparable."""
    lookups = {"early": tuple(f"AA-{index:04d}" for index in range(8))}
    head = make_scope_tools(lookups, 8_000, narration="neutral", placement="head")[0]()
    spread = make_scope_tools(lookups, 8_000, narration="neutral", placement="spread")[0]()
    assert abs(len(head) - len(spread)) < 0.01 * len(head)


def test_buried_placement_hides_codes_in_prose() -> None:
    """The buried arm must be genuinely harder, or it measures nothing the spread arm does not.

    It is not a broken variant of spread. Retrieval under noise is a real property of a real
    agent, and it is the arm where compaction can score *above* the uncompacted control by
    deleting the haystack the codes were hiding in. Measured on the control: 11 of 53 buried
    against 53 of 53 head-placed.
    """
    codes = tuple(f"AA-{index:04d}" for index in range(8))
    lookups = {"early": codes}

    buried = make_scope_tools(lookups, 8_000, narration="neutral", placement="buried")[0]()
    spread = make_scope_tools(lookups, 8_000, narration="neutral", placement="spread")[0]()

    # Same positions, so the two arms differ in findability alone and not in what a
    # head-truncating strategy would keep.
    assert sum(1 for code in codes if buried.index(code) < 4_096) == 1
    assert sum(1 for code in codes if spread.index(code) < 4_096) == 1
    # Only the spread arm gives each code a line of its own.
    assert sum(1 for line in spread.split("\n") if line.startswith("[record ")) == 8
    assert sum(1 for line in buried.split("\n") if line.startswith("[record ")) == 0
    assert all(code in buried for code in codes)


def test_seed_spread_reads_the_scored_outcome_not_the_raw_run() -> None:
    """Regression: correctness is not an attribute of ``LiveOutcome``.

    Reading it from the raw run passes ruff, pyright and the whole suite, then raises
    ``AttributeError`` on the first live call -- after the run has already been paid for.
    That is the second time a change has died that way, so the computation is a function
    with a test rather than a line inside the run loop.
    """
    scenario = build_live_scenario(salt="range", filler_turns=1, filler_tokens=10, tool_turns=6, markers_per_tool=2)
    codes = [fact.marker for fact in scenario.facts]

    def outcome_with(answer: str) -> LiveOutcome:
        return LiveOutcome(
            strategy="none",
            calls=(_call(2, 2, inp=100),),
            answer=answer,
            snapshot_prompt=" ".join(codes),
            tool_calls_made=0,
            turns_completed=1,
            turns_total=1,
        )

    perfect = outcome_with(" ".join(codes))
    partial = outcome_with(" ".join(codes[: len(codes) // 4]))
    scored = [_record(perfect, scenario).correctness_samples, _record(partial, scenario).correctness_samples]

    assert not hasattr(perfect, "correctness")
    assert _seed_spread(scored) > 0
    # A single seed says nothing about compaction's reliability, and must not claim to.
    assert _seed_spread(scored[:1]) == 0.0


def test_an_unstable_control_disables_the_accuracy_ranking() -> None:
    """A baseline that swings cannot be compared against, and the table must say so.

    Cost has been policed by a spread warning since the beginning. Accuracy was not, and
    three matrices were produced and believed before the gap was noticed: the `correct`
    column shows the median-*cost* repeat, so a control scoring 100, 22 and 22 prints an
    unremarkable 100.
    """
    stable = _accuracy_note({"none": 9.0}, "none", repeats=3)
    unstable = _accuracy_note({"none": 78.0}, "none", repeats=3)
    single = _accuracy_note({"none": 78.0}, "none", repeats=1)

    assert stable == []
    assert any("ACCURACY NOT RANKABLE" in line for line in unstable)
    assert any("78 points" in line for line in unstable)
    # The cost axis survives an unstable accuracy axis and the warning must say which is which.
    assert any("cost columns are unaffected" in line for line in unstable)
    # One repeat measures no stability at all; the cost warning already says so, and a second
    # warning saying the same thing would just be noise.
    assert single == []


def _control_cell(seeded: int) -> dict[str, CellStats]:
    """Return the one row ``_fill_note`` reads, carrying the size the control actually seeded.

    Args:
        seeded: Billed size of the control's last seeding prompt.

    Returns:
        The stats mapping, keyed as the note expects.
    """
    record = SeedRecord(
        cell=_cell_params(),
        strategy="none",
        seed=1,
        cost=0.01,
        summarizer_cost=0.0,
        input_tokens=1_000,
        cached_tokens=300,
        output_tokens=20,
        probe_input_tokens=400,
        probe_cached_tokens=120,
        probe_output_tokens=8,
        calls=4,
        messages_left=6,
        messages_peak=12,
        prompt_tokens_final=900,
        prompt_tokens_peak=1_000,
        seed_prompt_tokens=seeded,
        facts_total=53,
        facts_left=53,
        facts_lost=0,
        nofetch=0,
        correctness_samples=(1.0,),
        ignored_samples=(0,),
        combined_samples=(1.0,),
        disqualified=False,
        context_drift=0,
        rate_limit_retries=0,
        throttled_seconds=0.0,
        connection_retries=0,
        connection_seconds=0.0,
        turns_completed=10,
        turns_total=10,
        probe_repeats=1,
        summarizer_calls=0,
        summarizer_failures=0,
        groups_kept_uncovered=0,
        fallbacks_after_record=0,
        records_in_conversation=0,
        strategy_notes=(),
        dropped_options=(),
        answer="",
    )
    return {"none": _aggregate("none", [record])}


def _shared_plan(**overrides: Any) -> FillPlan:
    """Return a plan that asked for 60% of a 100,000-token target to be tool results."""
    defaults: dict[str, Any] = {
        "filler_turns": 18,
        "filler_tokens": 2_000,
        "context_limit": 120_000,
        "fill_fraction": 0.86,
        "target_tokens": 100_000,
        "predicted_tokens": 100_000,
        "payload_tokens": 62_500,
        "tool_payload_tokens": 60_000,
        "tool_result_tokens": 10_000,
        "tool_share": 0.6,
    }
    return FillPlan(**{**defaults, **overrides})


def test_the_achieved_tool_share_is_reported_beside_the_achieved_fill() -> None:
    """The requested share is an intention; only what the run seeded says what it measured.

    Reported against the same billed denominator as the fill, so the two lines can be read
    together: a share that missed because the conversation did is a different fault from one
    that missed because the results were sized wrong, and the fill line above says which.
    """
    lines = _fill_note(_control_cell(100_000), _shared_plan(), "none")

    assert any("Fill: 100,000 tokens seeded" in line for line in lines)
    assert any("Tool share: 60,000 tokens of tool results is 60.0% of what was seeded" in line for line in lines)
    assert any("against 60% requested, +0.0%" in line for line in lines)
    assert not any("OFF TARGET" in line for line in lines)


def test_a_tool_share_that_missed_is_flagged_the_way_a_fill_that_missed_is() -> None:
    """A payload that is not the share it claims does not compare with cells at other windows.

    Which is the whole purpose of stating it as a share, so it is policed to the same tolerance
    as the fill rather than left in the table as a number nobody checked.
    """
    lines = _fill_note(_control_cell(150_000), _shared_plan(), "none")

    assert any("TOOL SHARE OFF TARGET" in line for line in lines)
    assert any("FILL OFF TARGET" in line for line in lines)


def test_a_run_that_asked_for_no_share_is_told_nothing_about_one() -> None:
    """Every cell recorded so far sized its payload outright, and their notes must not change."""
    lines = _fill_note(_control_cell(100_000), _shared_plan(tool_share=0.0), "none")

    assert any("Fill: " in line for line in lines)
    assert not any("Tool share" in line for line in lines)


def test_narration_probe_declares_every_flag_it_reads() -> None:
    """A sample that reads an undeclared flag passes every check and dies on first use.

    That has happened twice in this package: once for --no-force-tool-calls, once for an
    attribute assumed to exist on LiveOutcome. Both cost a live run to discover. The probe
    is a calibration tool people will point at a new model, so its arguments are pinned.
    """
    from samples.probe_narration import build_parser as narration_parser

    args = narration_parser().parse_args(["foundry:some-model"])
    for flag in (
        "narrations",
        "placements",
        "repeats",
        "agent",
        "context_window",
        "max_output_tokens",
        "answer_max_tokens",
        "markers_per_tool",
        "tool_turns",
        "filler_turns",
        "filler_tokens",
        "tool_result_tokens",
        "no_force_tool_calls",
        "no_temperature",
    ):
        assert hasattr(args, flag), flag
    # The defaults must name real modes, or the probe fails on its own first run.
    assert set(args.narrations.split(",")) <= {"neutral", "prompted", "suppressed"}
    assert set(args.placements.split(",")) <= {"spread", "buried", "head"}


async def test_run_live_forces_client_side_history_without_being_asked() -> None:
    """A Responses-API client must have store=False set by run_live, not by its caller.

    When the service owns the conversation, MAF skips history loading and the agent sends
    only the new turn: compaction has nothing to act on and every setting measures the same
    thing. The live CLI has always forced this. A calibration probe calling run_live directly
    did not, and reported all three narration modes as stable with ranges of 0 to 7 points --
    against 78 points for the same model measured properly -- because the model was reading a
    history the client had never compacted.
    """

    class StoringRuntime:
        """Minimal stand-in exposing what the forcing looks at."""

        STORES_BY_DEFAULT = True

    runtime = SimpleNamespace(client=StoringRuntime(), model="stub", options={})

    assert wants_client_side_history(runtime.client) is True
    assert "store" not in runtime.options

    # run_live sets it before doing anything else; the call is expected to fail past that
    # point because the stub is not a real client, which is enough to prove the ordering.
    with contextlib.suppress(Exception):
        await run_live(
            cast(Any, runtime),
            strategy_name="none",
            options=StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=1_000, max_output_tokens=100),
            scenario=build_live_scenario(salt="store", filler_turns=1, filler_tokens=10, tool_turns=6),
        )

    assert runtime.options.get("store") is False


async def test_server_history_can_still_be_opted_into() -> None:
    """The escape hatch must survive, or --server-history silently stops meaning anything."""

    class StoringRuntime:
        STORES_BY_DEFAULT = True

    runtime = SimpleNamespace(client=StoringRuntime(), model="stub", options={})

    with contextlib.suppress(Exception):
        await run_live(
            cast(Any, runtime),
            strategy_name="none",
            options=StrategyOptions(tokenizer=TOKENIZER, max_context_window_tokens=1_000, max_output_tokens=100),
            scenario=build_live_scenario(salt="store", filler_turns=1, filler_tokens=10, tool_turns=6),
            allow_server_history=True,
        )

    assert "store" not in runtime.options


async def test_the_recall_middleware_forces_the_call_inside_the_real_pipeline() -> None:
    """The middleware must fire when installed on an actual agent, not only in isolation.

    It was verified standalone and then produced a live row showing a record found with zero
    forced calls -- a combination the code should not allow. Isolation tests cannot catch
    that: what matters is whether the middleware sees the loaded history at the point it
    looks, and the history is only assembled inside the pipeline.
    """
    ceiling = 2_000
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=ceiling, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.99
    )
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=ceiling, tokenizer=TOKENIZER, arm=lambda: None, trigger_fraction=0.1
    )
    client = StubChatClient()
    recorder = UsageRecorder()
    agent = build_live_agent(
        cast(Any, SimpleNamespace(client=client, model="stub", options={})),
        kind="harness",
        strategy=strategy,
        tokenizer=TOKENIZER,
        tools=[make_recall_tool()],
        recorder=recorder,
        extra_middleware=[middleware],
        max_context_window_tokens=ceiling,
        max_output_tokens=100,
    )
    session = agent.create_session()

    # Enough turns, each large, that the history passes the trigger.
    for _ in range(4):
        await agent.run("x" * 4_000, session=session)

    forced = [options for options in client.options_seen if "tool_choice" in options]
    assert middleware.forced_calls > 0, "the middleware never fired inside the pipeline"
    assert any(
        options["tool_choice"] == {"mode": "required", "required_function_name": RECALL_TOOL_NAME} for options in forced
    ), "tool_choice never reached the client"


async def test_a_truncated_record_reaches_the_flags_column() -> None:
    """A record cut at the cap must be visible in the table, or it reads as a complete one.

    Every other failure of this strategy is loud. No record at all leaves the row carrying
    FALLBACK, and a tool call cut mid-arguments produces exactly that, because the arguments
    JSON is where the cut lands. A record cut after a closing brace is the quiet one: it
    parses, the strategy anchors on it and drops every tool group behind it, and the part that
    never got written is scored as compaction damage. So the count has to travel all the way
    to the column a reader actually looks at.

    Driven through the real pipeline rather than the middleware alone, because what is being
    checked is that the middleware reads the provider's finish reason off a response the
    pipeline produced -- the same thing that made a standalone-verified middleware report a
    record found with zero forced calls in a live run.
    """
    ceiling = 2_000
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=ceiling, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.99
    )
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=ceiling,
        tokenizer=TOKENIZER,
        arm=lambda: None,
        trigger_fraction=0.1,
        record_max_tokens=512,
    )
    client = StubChatClient(finish_reason="length")
    agent = build_live_agent(
        cast(Any, SimpleNamespace(client=client, model="stub", options={})),
        kind="harness",
        strategy=strategy,
        tokenizer=TOKENIZER,
        tools=[make_recall_tool()],
        recorder=UsageRecorder(),
        extra_middleware=[middleware],
        max_context_window_tokens=ceiling,
        max_output_tokens=100,
    )
    session = agent.create_session()
    for _ in range(4):
        await agent.run("x" * 4_000, session=session)

    assert middleware.forced_calls > 0, "the middleware never fired inside the pipeline"
    assert middleware.records_truncated == middleware.forced_calls
    # The cap rides on the pinned call and on no other, so the rest of the run keeps the
    # ceiling sized for an answer to the user. Pinned means a named function: every call in
    # this run carries a tool_choice, and all but these say "auto".
    caps = {
        isinstance(options.get("tool_choice"), Mapping): options.get("max_tokens") for options in client.options_seen
    }
    assert caps[True] == 512
    assert caps[False] == 100, "the run's own ceiling, unchanged by the record's"

    notes = _strategy_notes(middleware)
    assert f"TRUNCATED:{middleware.forced_calls}" in notes

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = replace(_record(outcome, scenario, strategy="tool_summary_anchored"), strategy_notes=notes)
    cell = _aggregate("tool_summary_anchored", [record])

    assert f"TRUNCATED:{middleware.forced_calls}" in _flags(cell, None)
    assert f"TRUNCATED:{middleware.forced_calls}" in _render(None, [cell], set(), show_answers=False)


async def test_a_run_that_was_not_cut_short_says_nothing() -> None:
    """A clean run must add no flag, or the column stops meaning anything."""
    ceiling = 2_000
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=ceiling, tokenizer=TOKENIZER, arm=lambda: None, trigger_fraction=0.1
    )
    agent = build_live_agent(
        cast(Any, SimpleNamespace(client=StubChatClient(), model="stub", options={})),
        kind="harness",
        strategy=ToolResultAnchoredSummarizationCompactionStrategy(
            max_input_tokens=ceiling, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.99
        ),
        tokenizer=TOKENIZER,
        tools=[make_recall_tool()],
        recorder=UsageRecorder(),
        extra_middleware=[middleware],
        max_context_window_tokens=ceiling,
        max_output_tokens=100,
    )
    session = agent.create_session()
    for _ in range(4):
        await agent.run("x" * 4_000, session=session)

    assert middleware.forced_calls > 0
    assert not [note for note in _strategy_notes(middleware) if note.startswith("TRUNCATED")]


#: A ceiling that leaves the eight-turn fixture between the strategy's two thresholds, so it
#: acts on the record rather than sitting below the trigger or giving up above the fallback.
#: The conversation is about 17,000 tokens, which is 90% of this. It was 22,000 while the
#: thresholds were 0.6 and 0.9; at 0.8 that puts the fixture *below* the trigger, and every
#: test built on it would have asserted against a strategy that did nothing.
_RECORD_CEILING = 19_000


def _tool_conversation(tool_turns: int, *, covered: int) -> list[Message]:
    """Return a conversation whose recall record names only the first ``covered`` tools.

    The live benchmark's own shape: one no-argument tool per scope, named ``lookup_<n>``, so a
    turn can pin exactly which fact it gathers. It has to be that shape here because the
    strategy checks a record against the *names* of the tools it claims to cover, and a
    fixture calling a single tool repeatedly would exercise only the degenerate case.

    Args:
        tool_turns: How many lookup turns to generate.

    Keyword Args:
        covered: How many of them the record names. Fewer than ``tool_turns`` is the measured
            shape of a model that named some tools and stopped.

    Returns:
        The messages, the record last.
    """
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
                contents=[{"type": "function_call", "call_id": call_id, "name": f"lookup_{index}", "arguments": "{}"}],
                message_id=f"a_call_{index}",
            ),
            Message(
                role="tool",
                contents=[{"type": "function_result", "call_id": call_id, "result": f"CODE-{index} " + "x" * 8_000}],
                message_id=f"t_res_{index}",
            ),
        ]
    values = " ".join(f"lookup_{index}: CODE-{index}." for index in range(covered))
    return [
        *messages,
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


async def test_groups_a_record_never_named_reach_the_seed_record_and_the_flags_column() -> None:
    """What the coverage check held back has to be visible, or the row reads as a clean run.

    The strategy now refuses to delete a tool group the record does not mention, which turns a
    silent loss into a cost: the row carries tokens a complete record would have replaced. That
    cost is indistinguishable from the design simply not saving much, and the two call for
    opposite responses -- one says fix the record, the other says the strategy does not pay. So
    the count has to travel from the strategy through the outcome and the seed record to the
    column a reader actually looks at, which is four handoffs and four places to lose it.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=16_000, tokenizer=TOKENIZER)

    assert await strategy(_tool_conversation(6, covered=2)) is True
    assert strategy.groups_kept_uncovered == 4

    notes = _strategy_notes(strategy)
    assert "UNCOVERED:4" in notes

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(
        replace(outcome, strategy_notes=notes, groups_kept_uncovered=strategy.groups_kept_uncovered),
        scenario,
        strategy="tool_summary_anchored",
    )

    assert record.groups_kept_uncovered == 4, "the count must survive scoring, not only the flag string"
    cell = _aggregate("tool_summary_anchored", [record])

    assert "UNCOVERED:4" in _flags(cell, None)
    assert "UNCOVERED:4" in _render(None, [cell], set(), show_answers=False)


async def test_a_record_that_named_every_tool_adds_no_flag_at_all() -> None:
    """A silent check is the point of it, so a complete record must leave the column clean.

    gpt-5.4-mini writes records naming every tool they cover, and on those the check costs
    nothing. A flag that appeared anyway would put a warning on every row of every model that
    complied, which is how a flags column stops being read.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=_RECORD_CEILING, tokenizer=TOKENIZER)

    assert await strategy(_tool_conversation(8, covered=8)) is True
    assert strategy.groups_kept_uncovered == 0
    assert not [note for note in _strategy_notes(strategy) if note.startswith("UNCOVERED")]

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(outcome, scenario, strategy="tool_summary_anchored")
    cell = _aggregate("tool_summary_anchored", [record])

    assert record.groups_kept_uncovered == 0
    assert not [flag for flag in _flags(cell, None) if flag.startswith("UNCOVERED")]


async def test_a_fallback_taken_behind_a_record_reaches_the_seed_record_and_the_flags_column() -> None:
    """A row measuring the fallback strategy has to say so, and this half of it never did.

    ``FALLBACK`` is the flag whose legend says the row is measuring another strategy, and only
    the give-up path ever set it. The path taken here is the other one: a record arrives, the
    strategy anchors on it, drops what it covers, finds the prompt still over the ceiling and
    hands the rest to the fallback -- which sheds the very groups the coverage check had just
    declined to delete. The row then loses facts by shortening rather than by deletion, and
    reported no fallback of any kind: measured, a seed flagged UNCOVERED:4 lost the control's
    facts while sitting three messages shorter and 16,617 tokens lighter.

    So the count has to travel from the strategy through the outcome and the seed record to
    the column a reader looks at, which is the same four handoffs UNCOVERED makes and the same
    four places to lose it.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=500, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.9
    )

    assert await strategy(_tool_conversation(6, covered=2)) is True
    assert strategy.fallbacks_after_record == 1
    assert strategy.fallbacks_used == 0, "the give-up path is a different event and must stay at zero"

    notes = _strategy_notes(strategy)
    assert "RECFALLBACK:1" in notes
    assert "FALLBACK:1" not in notes, "the two counts must not be readable as one another"

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(
        replace(outcome, strategy_notes=notes, fallbacks_after_record=strategy.fallbacks_after_record),
        scenario,
        strategy="tool_summary_anchored",
    )

    assert record.fallbacks_after_record == 1, "the count must survive scoring, not only the flag string"
    cell = _aggregate("tool_summary_anchored", [record])

    assert "RECFALLBACK:1" in _flags(cell, None)
    assert "RECFALLBACK:1" in _render(None, [cell], set(), show_answers=False)


async def test_a_record_that_freed_enough_leaves_the_fallback_flag_off() -> None:
    """The flag has to be silent on the runs where the design worked, or it stops being read.

    A record that covers its groups and frees the room they took is this strategy doing exactly
    what it is for, and a row like that measures nothing but itself. A flag appearing there
    would put a warning on the good case, which is how a flags column comes to be skipped.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=_RECORD_CEILING, tokenizer=TOKENIZER)

    assert await strategy(_tool_conversation(8, covered=8)) is True
    assert strategy.fallbacks_after_record == 0
    assert not [note for note in _strategy_notes(strategy) if note.startswith("RECFALLBACK")]

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(outcome, scenario, strategy="tool_summary_anchored")
    cell = _aggregate("tool_summary_anchored", [record])

    assert record.fallbacks_after_record == 0, "a control run took no record and no fallback behind one"
    assert not [flag for flag in _flags(cell, None) if flag.startswith("RECFALLBACK")]


async def test_the_uncovered_count_survives_the_results_file_and_an_older_record_reads_as_zero(
    tmp_path: Path,
) -> None:
    """The file is where a cell's numbers live, and the schema is what dates them.

    A count that reached the live table but not the file would go missing on exactly the runs
    it matters for: a cell is hours long, and the table people read months later is rebuilt
    from these lines. The older-record half pins the other direction -- a record written before
    the coverage check existed ran under a strategy that dropped every group in front of the
    record regardless, so no group was ever kept for want of coverage and zero is a measurement
    rather than a gap. Refusing those records would throw away every cell already on disk.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(replace(outcome, groups_kept_uncovered=4), scenario, strategy="tool_summary_anchored")
    path = tmp_path / "results.jsonl"
    append_seed_record(path, record)

    (read_back,) = read_seed_records(path)

    assert read_back.groups_kept_uncovered == 4
    assert read_back.schema == SCHEMA_VERSION

    older = {key: value for key, value in record.to_dict().items() if key != "groups_kept_uncovered"}
    older["schema"] = SCHEMA_VERSION - 1

    assert SeedRecord.from_dict(older).groups_kept_uncovered == 0


async def test_the_post_record_fallback_count_survives_the_file_and_an_older_record_reads_as_unknown(
    tmp_path: Path,
) -> None:
    """This count reads back as None where the one beside it reads back as zero, deliberately.

    Both are new fields on the same schema, and their absences mean opposite things. No run
    before the coverage check could keep an uncovered group, so zero groups is what those runs
    did. Every run since this strategy existed *could* fall back behind a record that had not
    freed enough, and none of them counted it when they did -- so a zero here would tell a
    reader that every recorded row stayed the strategy it is named for, which is precisely the
    claim nobody was in a position to make. None is how the record says nobody took the number.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(replace(outcome, fallbacks_after_record=2), scenario, strategy="tool_summary_anchored")
    path = tmp_path / "results.jsonl"
    append_seed_record(path, record)

    (read_back,) = read_seed_records(path)

    assert read_back.fallbacks_after_record == 2
    assert read_back.schema == SCHEMA_VERSION

    older = {key: value for key, value in record.to_dict().items() if key != "fallbacks_after_record"}
    older["schema"] = SCHEMA_VERSION - 1

    assert SeedRecord.from_dict(older).fallbacks_after_record is None, "an uncounted fallback is not a fallback of zero"


#: A ceiling that leaves the two-record fixture between the strategy's thresholds, chosen the
#: same way ``_RECORD_CEILING`` is: four tool turns and two records come to about 8,800 tokens,
#: which is 88% of this.
_TWO_RECORD_CEILING = 10_000


def _two_record_conversation() -> list[Message]:
    """Return a conversation carrying two records, which is what repeats produce.

    Both records cover every tool group, so the strategy has no shortfall to report and the
    only thing left for it to say about this conversation is how many records it is carrying.

    Returns:
        The messages, the newer record last.
    """
    covered = " ".join(f"lookup_{index}: CODE-{index}." for index in range(4))
    return [
        *_tool_conversation(4, covered=4),
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": "rec2", "name": RECALL_TOOL_NAME, "arguments": "{}"}],
            message_id="rec2_call",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": "rec2", "result": f"{RECORD_MARKER} {covered}"}],
            message_id="rec2_res",
        ),
    ]


async def test_the_record_count_reaches_the_seed_record_and_the_flags_column(tmp_path: Path) -> None:
    """A conversation accumulating records has to say so, in the one place a reader looks.

    Records are preserved: unshrinkable, undroppable, counted against the ceiling in full, and
    nothing merges them. So a run that takes three of them carries a floor under its prompt
    that no later pass can lower -- and every other column reads as though compaction were
    still working, because the message count keeps rising and each pass still reports having
    acted. Letting the size trigger ask more than once is what made this possible, so the
    count has to travel the same four handoffs ``groups_kept_uncovered`` does: strategy,
    outcome, seed record, column.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=_TWO_RECORD_CEILING, tokenizer=TOKENIZER
    )

    await strategy(_two_record_conversation())

    assert strategy.records_in_conversation == 2
    notes = _strategy_notes(strategy)
    assert "RECORDS:2" in notes
    assert "REC:1" in notes, "compliance and quantity are different questions and both are shown"

    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(
        replace(outcome, strategy_notes=notes, records_in_conversation=strategy.records_in_conversation),
        scenario,
        strategy="tool_summary_anchored",
    )

    assert record.records_in_conversation == 2, "the count must survive scoring, not only the flag string"
    cell = _aggregate("tool_summary_anchored", [record])

    assert "RECORDS:2" in _flags(cell, None)
    assert "RECORDS:2" in _render(None, [cell], set(), show_answers=False)

    path = tmp_path / "results.jsonl"
    append_seed_record(path, record)
    (read_back,) = read_seed_records(path)

    assert read_back.records_in_conversation == 2
    assert read_back.schema == SCHEMA_VERSION


async def test_a_record_written_before_repeats_existed_reports_no_count_rather_than_one() -> None:
    """The absence is "nobody took this number", which is not the same as "there was one".

    A run before schema 6 could take at most one record, so the temptation is to read the field
    back as 1. But whether it took that one depended on whether the model ever complied, and
    the rows flagged ``FALLBACK`` are precisely the ones where it did not -- so a 1 would credit
    them with a record they never got, and a 0 would deny one to every row that did. Which of
    the two happened is on those records only as a flag, and inferring a count from a flag
    string is not a count anybody took.
    """
    outcome, scenario = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    record = _record(replace(outcome, records_in_conversation=2), scenario, strategy="tool_summary_anchored")

    assert record.records_in_conversation == 2

    older = {key: value for key, value in record.to_dict().items() if key != "records_in_conversation"}
    older["schema"] = SCHEMA_VERSION - 1

    assert SeedRecord.from_dict(older).records_in_conversation is None, "a count nobody took is not a count of one"


class _RecordingStub(StubChatClient):
    """A model that answers its first call with a recall record and plain text after that.

    The base stub answers every call with empty arguments, which is right for the no-argument
    lookup tools and wrong for the recall tool: ``values`` is required, so an empty call is a
    failed invocation rather than a record, and nothing would ever reach the stored history to
    be read back out of it.
    """

    def __init__(self, values: str, **kwargs: Any) -> None:
        """Create the stub.

        Args:
            values: What this model writes into the record.

        Keyword Args:
            kwargs: Passed to the base stub.
        """
        super().__init__(**kwargs)
        self.values = values
        self.recorded = False

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        if self.recorded:
            return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)
        self.recorded = True
        self.seen.append(len(messages))
        self.options_seen.append(dict(options))

        async def _go() -> ChatResponse[Any]:
            return ChatResponse(
                messages=Message(
                    role="assistant",
                    contents=[
                        Content.from_function_call(
                            call_id="call_record",
                            name=RECALL_TOOL_NAME,
                            arguments=json.dumps({"values": self.values}),
                        )
                    ],
                ),
                usage_details=self.usage,
            )

        return _go()


def _recorded_agent(client: StubChatClient) -> Agent[Any]:
    """Return the harness's own agent with the recall tool registered and no strategy.

    The tool is ungated, so it records whatever it is asked to. What is under test is reading
    a record back out of a finished conversation, not the arming that decides when one is
    written, and a gate here would only add a way for the fixture to produce nothing.
    """
    return build_live_agent(
        cast(Any, SimpleNamespace(client=client, model="stub", options={})),
        kind="plain",
        strategy=None,
        tokenizer=TOKENIZER,
        tools=[make_recall_tool()],
        recorder=UsageRecorder(),
        max_context_window_tokens=10_000,
        max_output_tokens=100,
    )


async def test_the_record_dump_writes_the_text_the_model_actually_wrote(tmp_path: Path) -> None:
    """Every count about a record describes it without quoting it, and the record is the question.

    The counters say how many records were found, forced, truncated, and how many groups were
    left uncovered. None of them can say whether a record that named every tool also kept the
    values under those names -- the failure a check by name cannot see, and the one this exists
    to expose. So the dump carries the record verbatim, out of the conversation the probes were
    answered from, into a file named for the strategy and seed that produced it.
    """
    client = _RecordingStub("lookup_early: CODE-AAA1. lookup_late: CODE-ZZZ9.")
    agent = _recorded_agent(client)
    session = agent.create_session()
    await agent.run("record what you have", session=session)

    text = recall_record_text(agent, snapshot_state(session))

    assert RECORD_MARKER in text
    assert "CODE-AAA1" in text, "the values are the whole point of reading it"
    assert "CODE-ZZZ9" in text

    path = _dump_record(tmp_path / "records", "tool_summary_anchored", 2, text)

    assert path is not None
    assert path.name == "tool_summary_anchored-seed2.txt", "a file nobody can attribute is not a diagnostic"
    assert path.read_text(encoding="utf-8") == text


async def test_a_seed_that_took_no_record_leaves_no_file_behind(tmp_path: Path) -> None:
    """An empty file and a record the model wrote as nothing would look identical.

    Four of the five strategies in a cell never take a record at all, so a dump writing one
    file per seed regardless would bury the two or three worth reading under a dozen empty
    ones -- each of which reads as a model that was asked and answered nothing.
    """
    outcome, _ = await _probed(StubChatClient(usage=UsageDetails(input_token_count=1_000)), repeats=1)
    directory = tmp_path / "records"

    assert outcome.record_text == "", "the control takes no record, so run_live must report none"
    assert _dump_record(directory, "none", 1, outcome.record_text) is None
    assert not directory.exists(), "asking for a dump must not litter the disk with empty directories"


async def test_reading_the_record_back_changes_nothing_about_the_run(tmp_path: Path) -> None:
    """The dump is diagnostic, so it must be unable to move a single number in the table.

    Two ways it could: by being computed while the conversation is being had, where it would be
    state threaded through the object under measurement, or by mutating what it reads. It is
    neither -- it reads the finished history and returns a string -- and this pins that, because
    the flag exists to answer a question about a paid-for run and would be worthless if turning
    it on changed the run. The parser half pins the other guarantee: off unless asked for, and
    asking for it creates nothing until there is something to write.
    """
    client = _RecordingStub("lookup_early: CODE-AAA1.")
    agent = _recorded_agent(client)
    session = agent.create_session()
    await agent.run("record what you have", session=session)
    before = serialize_history(agent, session.state)
    calls = len(client.seen)

    first = recall_record_text(agent, session.state)
    second = recall_record_text(agent, session.state)

    assert first == second != ""
    assert serialize_history(agent, session.state) == before, "reading the record rewrote the conversation"
    assert len(client.seen) == calls, "reading the record sent something to the provider"

    default = build_parser().parse_args(["openrouter:some/model"])
    asked = build_parser().parse_args(["openrouter:some/model", "--dump-record", str(tmp_path / "records")])

    assert default.dump_record is None, "the dump must be opt-in"
    assert asked.dump_record == str(tmp_path / "records")
    assert not (tmp_path / "records").exists(), "parsing must not create the directory"


async def test_a_pinned_tool_choice_does_not_survive_the_follow_up_call() -> None:
    """A turn's pinned tool_choice applies to the first call, not to the whole turn.

    This is the only explanation left for a live row that showed a record found with zero
    forced calls: if the follow-up call after a tool result no longer carries the pin, the
    model is free to call any registered tool, including the recall tool. A record produced
    that way is the model volunteering, not the design working, and the two must not be
    reported as the same thing.
    """
    client = StubChatClient(tool_turns=[0])
    recorder = UsageRecorder()
    agent = build_live_agent(
        cast(Any, SimpleNamespace(client=client, model="stub", options={})),
        kind="harness",
        strategy=None,
        tokenizer=TOKENIZER,
        tools=[make_scope_tools({"early": ("AA-0",)}, 100, narration="neutral")[0]],
        recorder=recorder,
        max_context_window_tokens=10_000,
        max_output_tokens=100,
    )
    session = agent.create_session()

    await agent.run(
        "look it up",
        session=session,
        options={"tool_choice": {"mode": "required", "required_function_name": "lookup_early"}},
    )

    pins = [options.get("tool_choice") for options in client.options_seen]
    assert len(pins) >= 2, "a tool turn makes at least two calls"
    assert pins[0] is not None, "the first call carries the pin"
    assert pins[1] is None, "the follow-up call does not, so the model may call anything"


# region the results file


def _live_argv(*extra: str) -> list[str]:
    """Return a command line driving a small comparison the stub can answer.

    Sized down to the smallest cell that still has two strategies and two seeds, since the
    questions here are about what reaches the file rather than about what compaction does.

    Args:
        extra: Further flags, appended after the defaults so they win.

    Returns:
        The argument vector.
    """
    return [
        "azure",
        "--price-input",
        "1",
        "--price-cached",
        "0.1",
        "--price-output",
        "1",
        "--strategies",
        "none,truncation",
        "--repeats",
        "2",
        "--probe-repeats",
        "2",
        "--fill",
        "0",
        "--filler-turns",
        "2",
        "--filler-tokens",
        "50",
        "--tool-result-tokens",
        "50",
        "--context-window",
        "60000",
        "--tokenizer",
        "estimator",
        *extra,
    ]


def _stub_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the live CLI at the stub client, so a whole comparison runs offline."""

    def build(name: str, **_: Any) -> ProviderRuntime:
        usage = UsageDetails(input_token_count=1_000, output_token_count=20, cache_read_input_token_count=300)
        return ProviderRuntime(client=StubChatClient(usage=usage), model="stub-model")

    monkeypatch.setattr("agent_framework_lab_cachebench._live_cli.build_provider", build)


def _table(printed: str) -> str:
    """Return just the rendered table from a run's output, dropping the progress before it."""
    return printed[printed.index("Model:") :]


async def test_the_record_repeat_setting_reaches_the_run_that_installs_the_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reproducibility flag that stops at the parser is worse than not having one.

    The whole value of ``--no-record-repeats`` is that a cell can be put back on the axis runs
    26-39 were measured on. A flag that parsed, appeared in the archived command line, and then
    never reached the middleware would produce a cell labelled single-record that was not one,
    and the comparison it exists for would be made against the wrong thing with nothing saying
    so.
    """
    _stub_provider(monkeypatch)
    live = run_live
    seen: list[bool] = []

    async def capture(*args: Any, **kwargs: Any) -> LiveOutcome:
        seen.append(kwargs["repeat_records"])
        return await live(*args, **kwargs)

    monkeypatch.setattr("agent_framework_lab_cachebench._live_cli.run_live", capture)

    await run_live_comparison(build_parser().parse_args(_live_argv()))
    default = list(seen)
    await run_live_comparison(build_parser().parse_args(_live_argv("--no-record-repeats")))

    assert default and all(default), "repeats are on unless the run asks otherwise"
    assert not any(seen[len(default) :]), "and off for every strategy-seed of a run that does"


async def test_every_finished_seed_is_on_disk_before_the_cell_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cell killed partway must leave behind every seed it had already completed.

    This is the whole reason the file exists. The 60,000/0.86 cell ran all fifteen
    strategy-seeds over three and a half hours, died before printing its table, and left
    nothing at all -- work that had already been paid for. Writing when the cell ends cannot
    survive that, so a record is written and closed as each seed is scored.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "results.jsonl"
    live = run_live
    seeds = 0

    async def dying(*args: Any, **kwargs: Any) -> LiveOutcome:
        nonlocal seeds
        seeds += 1
        if seeds > 3:
            raise KeyboardInterrupt("the process died mid-cell")
        return await live(*args, **kwargs)

    monkeypatch.setattr("agent_framework_lab_cachebench._live_cli.run_live", dying)
    args = build_parser().parse_args(_live_argv("--results-jsonl", str(path)))

    with pytest.raises(KeyboardInterrupt):
        await run_live_comparison(args)

    records = read_seed_records(path)
    assert [(record.strategy, record.seed) for record in records] == [
        ("none", 1),
        ("none", 2),
        ("truncation", 1),
    ], "a completed seed was still in memory when the process died"
    assert all(record.correctness_samples for record in records), "records were written before they were scored"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param([], (DEFAULT_RECORD_MAX_TOKENS, DEFAULT_RECORD_TARGET_TOKENS), id="defaults"),
        pytest.param(["--record-max-tokens", "900", "--record-target-tokens", "450"], (900, 450), id="set"),
        pytest.param(["--record-max-tokens", "0", "--record-target-tokens", "0"], (None, None), id="zero-is-unset"),
    ],
)
async def test_the_record_bounds_reach_the_run(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: tuple[int | None, int | None]
) -> None:
    """Both numbers have to be settable per run, since neither has a knowable right value yet.

    Zero means "no bound of my own" for each: the cap falls back to --answer-max-tokens and
    the description states no target. The same convention as --fill 0, which hands sizing back
    to the manual flags.
    """
    _stub_provider(monkeypatch)
    live = run_live
    seen: list[tuple[int | None, int | None]] = []

    async def capturing(*args: Any, **kwargs: Any) -> LiveOutcome:
        seen.append((kwargs["record_max_tokens"], kwargs["record_target_tokens"]))
        return await live(*args, **kwargs)

    monkeypatch.setattr("agent_framework_lab_cachebench._live_cli.run_live", capturing)

    await run_live_comparison(build_parser().parse_args(_live_argv(*argv)))

    assert seen
    assert set(seen) == {expected}


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param([], None, id="off-by-default"),
        pytest.param(["--max-groups-before-record", "0"], None, id="zero-is-off"),
        pytest.param(["--max-groups-before-record", "3"], 3, id="set"),
    ],
)
async def test_the_group_bound_reaches_the_middleware_that_enforces_it(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: int | None
) -> None:
    """The bound was implemented and unit-tested, and nothing constructed it with a value.

    A parameter no caller sets has never run, whatever its own tests say, so the wiring is the
    thing worth pinning: the flag has to arrive at the one object that acts on it. Zero means
    off, the same convention --record-max-tokens and --fill already use, and off has to be the
    default -- forcing a record every few groups spends an agent turn each time and is a trade
    a run opts into rather than one it discovers.

    Asserted on the middleware's own construction rather than on the run's kwargs, because the
    kwarg was never in doubt: what this covers is the one strategy that builds a middleware at
    all, and the four that must not.
    """
    _stub_provider(monkeypatch)
    seen: list[int | None] = []

    class _Capturing(ToolResultRecallMiddleware):
        """The real middleware, noting what it was built with on the way past."""

        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs["max_groups_before_record"])
            super().__init__(**kwargs)

    monkeypatch.setattr("agent_framework_lab_cachebench._live.ToolResultRecallMiddleware", _Capturing)

    await run_live_comparison(
        build_parser().parse_args(_live_argv("--strategies", "none,tool_summary_anchored", "--repeats", "1", *argv))
    )

    assert seen == [expected], "one middleware, for the one strategy that takes a record"


async def test_the_combined_count_reaches_the_record_and_the_cell_it_identifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--combined-repeats has to arrive in the file, not just in the run.

    Two things read it back. The samples are what acc2 is a mean over, so their count is the
    measurement; the cell key is what decides whether two records belong in one row, and a
    cell asked the combined question once does not average with a cell asked it three times.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "results.jsonl"

    argv = _live_argv("--results-jsonl", str(path), "--strategies", "none", "--repeats", "1")
    await run_live_comparison(build_parser().parse_args([*argv, "--combined-repeats", "2"]))

    (record,) = read_seed_records(path)

    assert record.cell.combined_repeats == 2
    assert len(record.combined_samples) == 2, "the combined question was not asked twice"
    assert len(record.correctness_samples) == 2, "--probe-repeats 2 is what _live_argv asked for"
    # The two counts sit in the cell key, so a differently sampled cell is a different row.
    assert record.cell.key != replace(record.cell, combined_repeats=3).key


async def test_the_table_rebuilt_from_the_file_matches_the_live_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--from-jsonl must reproduce the live table exactly, not approximately.

    A rebuilt table that differs anywhere is a second measurement, and the point of recovering
    a dead cell is that its numbers are the numbers it would have printed. Both paths reach one
    aggregation over one kind of input, and this is what says so.

    The order the rows come out in is part of that. Two rows of equal cost are ordered by name
    precisely so the two paths cannot differ here: the live run builds them in the order the
    strategies ran, and the file is grouped in the order the records were written.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "results.jsonl"

    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(path))))
    live = capsys.readouterr().out
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(path)]))
    rebuilt = capsys.readouterr().out

    assert "VERDICT:" in live
    assert "Ranking: 2 of 2 rows kept at least 90%" in live, "both rows clear the bar, so there is no line"
    assert _table(rebuilt) == _table(live)

    # Again with a bar nothing can clear, so that the split itself is part of what has to
    # match. The threshold rides on the records, and the rebuilt table reads it from there
    # rather than from its own default -- otherwise the two would split at different places
    # while every column in them stayed identical.
    split = tmp_path / "split.jsonl"
    argv = _live_argv("--results-jsonl", str(split), "--min-correctness", "1.5")
    await run_live_comparison(build_parser().parse_args(argv))
    live_split = capsys.readouterr().out
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(split)]))
    rebuilt_split = capsys.readouterr().out

    assert "Ranking: 0 of 2 rows kept at least 150%" in live_split
    assert "below 150% of the control's acc1" in live_split
    assert _table(rebuilt_split) == _table(live_split)


async def test_aggregation_survives_the_round_trip_through_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Records read back must aggregate to the identical cell, field for field.

    Stronger than comparing the rendered tables, which round every number they print: a
    difference in the fourth decimal of a cost, or a tuple that came back as a list, is
    invisible there and is exactly the kind of thing that makes two paths disagree later.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "results.jsonl"
    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(path))))
    capsys.readouterr()

    written = read_seed_records(path)
    again = tmp_path / "again.jsonl"
    for record in written:
        append_seed_record(again, record)
    reread = read_seed_records(again)

    for name in ("none", "truncation"):
        assert _aggregate(name, [record for record in written if record.strategy == name]) == _aggregate(
            name, [record for record in reread if record.strategy == name]
        )


async def test_a_partial_file_renders_and_says_it_is_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cell missing strategies or seeds must render, and must not read as finished.

    Every mean in the table is over whatever is present, and nothing in the table itself
    distinguishes a mean over four strategy-seeds from one over fifteen. So the coverage is
    stated against what the run said it was going to take.
    """
    _stub_provider(monkeypatch)
    full = tmp_path / "full.jsonl"
    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(full))))
    capsys.readouterr()

    partial = tmp_path / "partial.jsonl"
    for record in read_seed_records(full):
        if record.strategy == "none" or record.seed == 1:
            append_seed_record(partial, record)
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(partial)]))
    printed = capsys.readouterr().out

    assert "seeds present: none 2/2, truncation 1/2" in printed
    assert "PARTIAL" in printed
    assert "fewer than the 2 seeds asked for" in printed
    assert "VERDICT:" in printed, "a partial cell that still holds the control can still be ranked"


async def test_a_file_without_the_control_renders_without_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every ranking is relative to the uncompacted control, so without it there is none.

    It must still print what it has. A cell that died before reaching the control is exactly
    the cell whose surviving rows are worth reading, and refusing to show them would repeat
    the loss the file exists to prevent.
    """
    _stub_provider(monkeypatch)
    full = tmp_path / "full.jsonl"
    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(full))))
    capsys.readouterr()

    headless = tmp_path / "headless.jsonl"
    for record in read_seed_records(full):
        if record.strategy != "none":
            append_seed_record(headless, record)
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(headless)]))
    printed = capsys.readouterr().out

    assert "NO VERDICT" in printed
    assert "truncation" in printed
    assert "never recorded a seed (none)" in printed


async def test_two_runs_into_one_path_keep_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Appending must not clobber, and two cells in one file must stay two cells.

    A sweep points every cell at one results file, and a resumed run points at the file it
    already half-filled. Truncating on open would lose the run that was being recovered.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "sweep.jsonl"
    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(path))))
    await run_live_comparison(
        build_parser().parse_args(_live_argv("--results-jsonl", str(path), "--context-window", "40000"))
    )
    capsys.readouterr()

    records = read_seed_records(path)
    cells = group_by_cell(records)
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(path)]))
    printed = capsys.readouterr().out

    assert len(records) == 8, "the second run overwrote the first"
    assert len(cells) == 2, "two context windows are two cells and cannot share a row"
    assert {params.context_window for params, _ in cells} == {40_000, 60_000}
    assert printed.count("Cell: ") == 2
    assert printed.count("VERDICT") == 2, "each cell needs its own table"


async def test_a_resumed_cell_reads_back_as_one_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The strategies a run asked for are intent, not identity.

    Resuming a dead cell for the strategies it never reached writes records naming a different
    ``--strategies`` list. Treating that as a different cell would put the recovered half in a
    table of its own, which is the opposite of recovering it.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "resumed.jsonl"
    await run_live_comparison(
        build_parser().parse_args(_live_argv("--results-jsonl", str(path), "--strategies", "none,truncation"))
    )
    await run_live_comparison(
        build_parser().parse_args(_live_argv("--results-jsonl", str(path), "--strategies", "none,sliding_window"))
    )
    capsys.readouterr()

    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(path)]))
    printed = capsys.readouterr().out

    assert len(group_by_cell(read_seed_records(path))) == 1
    assert "none 4/2, sliding_window 2/2, truncation 2/2" in printed


async def test_each_finished_seed_prints_a_line_of_its_own(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cell that runs for hours has to show progress, or a collapsed row is invisible.

    Cost, facts and accuracy are what move first when a strategy stops preserving anything,
    and they are readable here hours before the table would have printed them.
    """
    _stub_provider(monkeypatch)
    await run_live_comparison(build_parser().parse_args(_live_argv()))
    printed = capsys.readouterr().out

    lines = [line for line in printed.splitlines() if re.search(r" seed \d+/\d+ ", line)]
    assert len(lines) == 4, "one line per strategy-seed"
    assert "none seed 1/2" in lines[0]
    for line in lines:
        assert "$" in line
        assert "facts " in line


def test_a_progress_line_shows_what_a_watcher_needs() -> None:
    """The line must carry the four numbers, and flag a seed that disqualified or waited."""
    record = SeedRecord(
        cell=_cell_params(repeats=3),
        strategy="truncation",
        seed=2,
        cost=0.0412,
        summarizer_cost=0.0,
        input_tokens=1_000,
        cached_tokens=300,
        output_tokens=20,
        probe_input_tokens=400,
        probe_cached_tokens=120,
        probe_output_tokens=8,
        calls=4,
        messages_left=6,
        messages_peak=12,
        prompt_tokens_final=900,
        prompt_tokens_peak=1_000,
        seed_prompt_tokens=800,
        facts_total=53,
        facts_left=27,
        facts_lost=20,
        nofetch=6,
        correctness_samples=(0.5, 0.54),
        ignored_samples=(1, 2),
        combined_samples=(0.4,),
        disqualified=True,
        context_drift=1,
        rate_limit_retries=4,
        throttled_seconds=37.5,
        connection_retries=2,
        connection_seconds=6.0,
        turns_completed=10,
        turns_total=10,
        probe_repeats=2,
        summarizer_calls=0,
        summarizer_failures=0,
        groups_kept_uncovered=0,
        fallbacks_after_record=0,
        records_in_conversation=0,
        strategy_notes=(),
        dropped_options=(),
        answer="",
    )

    line = _progress(record)

    assert "truncation seed 2/3" in line
    assert "$0.0412" in line
    assert "facts 27/53" in line
    assert "acc1 52%" in line
    assert "acc2 40%" in line
    assert "DQ" in line
    assert "DRIFT:1" in line
    assert "THROTTLED:4 (38s)" in line
    assert "RECONNECTED:2 (6s)" in line


@pytest.mark.parametrize("version", [1, SCHEMA_VERSION + 1], ids=["before_the_rebuild", "from_the_future"])
def test_records_from_another_schema_are_refused(version: int) -> None:
    """A reader must be able to tell an old record from a new one.

    The measurement itself has been rebuilt once. Before the seed/snapshot/probe design,
    ``survived`` was scored against a prompt the closing answers had written, and the same
    strategy read 53/53 on one run and 18/53 on another. Averaging records from either side of
    that into one row would be a mean over two different questions, which is why version 1 is
    still refused although version 2 is not.
    """
    payload = {"schema": version, "cell": _cell_params().to_dict(), "strategy": "none"}

    with pytest.raises(ValueError, match="schema"):
        SeedRecord.from_dict(payload)


async def test_a_record_written_before_the_connection_counters_still_reads() -> None:
    """The six cells on disk are version 2, and they are 180 seeds of paid-for measurement.

    Their absent counters are not a gap: the code that wrote them failed the turn on a dropped
    connection instead of re-sending it, so no call was re-sent and none was waited on, and
    zero is what that run did. That is the opposite of the version 1 case above, where the
    numbers on the record are answers to a question this reader no longer asks.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    written = _record(outcome, scenario).to_dict()
    written["schema"] = 2
    del written["connection_retries"]
    del written["connection_seconds"]

    old = SeedRecord.from_dict(written)

    assert (old.connection_retries, old.connection_seconds) == (0, 0.0)
    cell = _aggregate("none", [old])
    assert cell.connection_retries == 0
    assert "RECONNECTED" not in " ".join(_flags(cell, None)), "a run that never reconnected was flagged as having"
    assert "Reconnected:" not in _render(None, [cell], set(), show_answers=False)


def test_the_recorded_cells_on_disk_still_read() -> None:
    """The files this package's results are written up from have to survive a schema change.

    Not a fixture: these are the actual records behind ``RESULTS.md`` and the report, and the
    reader refuses a line rather than skipping it, so a change that makes them unreadable makes
    every recorded cell unrecoverable at once.
    """
    recorded = sorted((Path(__file__).parents[1] / "runs").glob("*.jsonl"))
    assert recorded, "no recorded cells were found, so this passed without reading anything"

    for path in recorded:
        records = read_seed_records(path)
        assert records, f"{path.name} read as empty"
        assert all(record.correctness_samples for record in records), f"{path.name} lost its samples"


def test_the_solved_sizing_survives_the_file() -> None:
    """The fill check has to be reproducible from the file.

    Without the plan, a rebuilt cell cannot say whether it landed on the fraction it is
    labelled with, which is the one thing that makes the fill a variable rather than a wish.
    """
    plan = FillPlan(
        filler_turns=7,
        filler_tokens=2_000,
        context_limit=60_000,
        fill_fraction=0.86,
        target_tokens=51_600,
        predicted_tokens=51_500,
        payload_tokens=30_000,
        tool_payload_tokens=21_000,
    )
    params = _cell_params(plan=plan, fill=0.86)

    assert CellParams.from_dict(params.to_dict()) == params


def test_a_plan_written_before_the_payload_could_be_derived_still_reads() -> None:
    """Six cells on disk have a plan with neither sizing field, and they have to keep opening.

    Their run is not in doubt: the size was stated, which is what a share of 0 means, and it is
    on the cell beside the plan. Demanding the fields here would refuse every record already
    recorded rather than reading it as the measurement it was.
    """
    stored = _cell_params(fill=0.86).to_dict()
    stored["plan"] = {
        "filler_turns": 39,
        "filler_tokens": 1_874,
        "context_limit": 120_000,
        "fill_fraction": 0.86,
        "target_tokens": 103_200,
        "predicted_tokens": 103_213,
        "payload_tokens": 24_497,
        "tool_payload_tokens": 21_967,
    }

    params = CellParams.from_dict(stored)

    assert params.plan is not None
    assert params.tool_share == 0.0
    assert params.plan.tool_share == 0.0


async def test_two_tool_shares_are_two_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two payloads cannot share a row, however the payload was arrived at.

    The derived size is what separates them -- it is in the identity key and two shares cannot
    reach one size without building the same conversation -- and the share is in the key beside
    it so the sweep reads back against the axis it was swept on.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "shares.jsonl"
    argv = _live_argv("--results-jsonl", str(path), "--fill", "0.5", "--repeats", "1", "--strategies", "none")
    await run_live_comparison(build_parser().parse_args([*argv, "--tool-share", "0.2"]))
    await run_live_comparison(build_parser().parse_args([*argv, "--tool-share", "0.4"]))
    capsys.readouterr()

    cells = group_by_cell(read_seed_records(path))

    assert len(cells) == 2, "two shares are two workloads and cannot aggregate into one row"
    shares = {params.tool_share for params, _ in cells}
    assert shares == {0.2, 0.4}
    sizes = {params.tool_result_tokens for params, _ in cells}
    assert len(sizes) == 2, "the derived size must reach the record, or --from-jsonl rebuilds the wrong payload"
    assert 50 not in sizes, "the record kept the --tool-result-tokens the share was supposed to override"


async def test_a_complete_cell_is_not_marked_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A complete cell must say so by saying nothing, or PARTIAL means nothing either."""
    _stub_provider(monkeypatch)
    path = tmp_path / "complete.jsonl"
    await run_live_comparison(build_parser().parse_args(_live_argv("--results-jsonl", str(path))))

    params, records = group_by_cell(read_seed_records(path))[0]
    lines = _coverage(params, records)

    assert any("none 2/2, truncation 2/2" in line for line in lines)
    assert not any("PARTIAL" in line for line in lines)


async def test_a_rebuilt_verdict_applies_the_bar_the_run_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The correctness bar is an input to the verdict, so it has to travel with the records.

    It is the one input that leaves no trace in the columns. A rebuild under the default bar
    would rank rows the original run never ranked while every number above the verdict stayed
    identical, which is the hardest kind of disagreement to notice. Naming the flag on the
    rebuild still overrides it, since asking a measured cell a different question is a
    legitimate thing to want.
    """
    _stub_provider(monkeypatch)
    path = tmp_path / "bar.jsonl"
    await run_live_comparison(
        build_parser().parse_args(_live_argv("--results-jsonl", str(path), "--min-correctness", "0.4"))
    )
    capsys.readouterr()
    seen: list[float] = []

    def capturing(outcomes: Any, *, min_correctness: float) -> Any:
        seen.append(min_correctness)
        return recommend(outcomes, min_correctness=min_correctness)

    monkeypatch.setattr("agent_framework_lab_cachebench._live_cli.recommend", capturing)
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(path)]))
    await run_live_comparison(build_parser().parse_args(["--from-jsonl", str(path), "--min-correctness", "0.9"]))
    capsys.readouterr()

    assert {record.cell.min_correctness for record in read_seed_records(path)} == {0.4}
    assert seen == [0.4, 0.9]


def test_from_jsonl_needs_no_provider() -> None:
    """Rebuilding a table calls nothing, so demanding a provider would misdescribe it."""
    args = build_parser().parse_args(["--from-jsonl", "results.jsonl"])

    assert args.provider is None
    assert args.from_jsonl == "results.jsonl"


async def test_a_run_without_a_provider_is_refused() -> None:
    """Making the provider optional must not make it optional for a run that measures."""
    args = build_parser().parse_args(["--repeats", "1"])

    with pytest.raises(SystemExit, match="provider is required"):
        await run_live_comparison(args)


# endregion


def test_seed_numbering_can_be_offset() -> None:
    """Concurrent invocations of one cell must not build the same conversation.

    The scenario salt is a whole-second timestamp plus the strategy name and the seed number.
    Several single-seed invocations launched together share the first two, so without an
    offset they all seed at 1 and produce byte-identical conversations -- five records that
    look like five seeds and are one, which is exactly the spread the seed axis exists to
    measure being reported as zero.
    """
    args = build_parser().parse_args(["openrouter:some/model"])
    assert args.seed_offset == 0, "the default must not renumber anything"

    offset = build_parser().parse_args(["openrouter:some/model", "--seed-offset", "4"])
    assert offset.seed_offset == 4


async def test_the_single_seed_warning_counts_seeds_not_the_flag() -> None:
    """A merged cell holds more seeds than any one invocation asked for.

    Cells are run as several concurrent single-seed invocations and merged afterwards, so
    ``--repeats`` reads 1 on every record while the cell holds five of them. Reading the
    request rather than the records had a five-seed cell announce that it measured nothing,
    directly under a table showing its seed spread.
    """
    outcome, scenario = await _probed(
        StubChatClient(usage=UsageDetails(input_token_count=1_000, output_token_count=20)), repeats=1
    )
    cells = [
        _aggregate(name, [_record(outcome, scenario, strategy=name, seed=seed) for seed in (1, 2, 3)])
        for name in ("none", "truncation")
    ]
    verdict = recommend([_to_joint(cell) for cell in cells])

    table = _render(verdict, cells, set(), show_answers=False)

    assert all(len(cell.records) == 3 for cell in cells)
    assert "Single seed" not in table, "three seeds were reported as one"
