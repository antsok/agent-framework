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
    TruncationStrategy,
    UsageDetails,
)
from agent_framework_lab_cachebench import (
    AGENT_KINDS,
    FillPlan,
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
    resolve_instructions,
)
from agent_framework_lab_cachebench._live_cli import (
    _accuracy_note,
    _aggregate,
    _cost,
    _coverage,
    _excluded_cells,
    _flags,
    _probe_spread,
    _progress,
    _render,
    _row,
    _seed_record,
    _seed_spread,
    _spread,
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
        "min_correctness",
        "summarizer_provider",
        "no_force_tool_calls",
        "server_history",
        "no_temperature",
        "tokenizer",
        "show_answers",
        "dry_run",
        "fill",
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

    def __init__(self, *, throttle_once: bool = False, reject_option: str | None = None, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            throttle_once: Refuse the first call that carries a tool result with a 429.
            reject_option: Name of a request option to refuse, and refuse only on a call
                carrying a tool result -- so the option-drop retry meets a half-finished turn
                exactly as the rate limit does.
        """
        super().__init__(obey_tool_choice=True, **kwargs)
        self.requests: list[tuple[Message, ...]] = []
        self.throttle_once = throttle_once
        self.reject_option = reject_option
        self.throttled = 0
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

    incomplete, oversized = _excluded_cells(cells)
    ranked = [cell.strategy for cell in cells if cell.strategy not in incomplete | oversized]

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
    assert _excluded_cells([unfinished, oversized]) == ({"none"}, {"none"})


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
    assert f"${expected:.4f}" in _render(None, [cell], set(), show_answers=False)


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
        turns_completed=10,
        turns_total=10,
        probe_repeats=2,
        summarizer_calls=0,
        summarizer_failures=0,
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

    lines = [line for line in printed.splitlines() if " seed " in line and "acc1 " in line]
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
        turns_completed=10,
        turns_total=10,
        probe_repeats=2,
        summarizer_calls=0,
        summarizer_failures=0,
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


def test_records_from_another_schema_are_refused() -> None:
    """A reader must be able to tell an old record from a new one.

    The measurement itself has been rebuilt once. Before the seed/snapshot/probe design,
    ``survived`` was scored against a prompt the closing answers had written, and the same
    strategy read 53/53 on one run and 18/53 on another. Averaging records from either side of
    that into one row would be a mean over two different questions.
    """
    payload = {"schema": SCHEMA_VERSION + 1, "cell": _cell_params().to_dict(), "strategy": "none"}

    with pytest.raises(ValueError, match="schema"):
        SeedRecord.from_dict(payload)


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
