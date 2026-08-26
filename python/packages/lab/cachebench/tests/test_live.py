# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the live-agent compaction runner.

Everything here runs offline against a stub chat client. The stub is composed from the same
layers a real provider client uses, so the agent's middleware pipeline, its history
persistence and the compaction hook all execute for real; only the network call is
replaced. That matters because the questions worth testing here are all about *ordering*
between those layers, and a hand-rolled mock that skipped them would answer none of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    LiveOutcome,
    MeteredClient,
    ModelCall,
    ProviderRuntime,
    StrategyOptions,
    UsageRecorder,
    build_live_agent,
    build_live_scenario,
    build_recall_scenario,
    build_strategy,
    make_lookup_tool,
    run_live,
    unretrieved_facts,
    wants_client_side_history,
)
from agent_framework_lab_cachebench._advisor import ModelPricing
from agent_framework_lab_cachebench._live import make_scope_tools
from agent_framework_lab_cachebench._live_cli import _cost, _representative, _spread, _summarizer_cost

TOKENIZER = CharacterEstimatorTokenizer()


class StubChatClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], BaseChatClient[Any]):
    """A chat client composed from the real layers, answering from a script.

    Built the same way ``OpenAIChatClient`` is, so middleware, function invocation and the
    compaction hook all run. ``seen`` records how many messages each request carried, which
    is the ground truth every recorder assertion is checked against.
    """

    def __init__(self, *, tool_turns: Sequence[int] = (), usage: UsageDetails | None = None, **kwargs: Any) -> None:
        """Create the stub.

        Keyword Args:
            tool_turns: Indices of calls that should answer with a tool call instead of text.
            usage: Usage to report on every response.
        """
        super().__init__(**kwargs)
        self.seen: list[int] = []
        self.options_seen: list[dict[str, Any]] = []
        self.tool_turns = set(tool_turns)
        self.usage = usage

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

        async def _go() -> ChatResponse[Any]:
            if index in self.tool_turns:
                contents: list[Any] = [
                    Content.from_function_call(
                        call_id=f"call_{index}",
                        # One no-argument tool per scope: the turn's scope is pinned by which
                        # function is called, not by an argument the model chooses.
                        name="lookup_early",
                        arguments="{}",
                    )
                ]
            else:
                contents = ["a reply with some body to it"]
            return ChatResponse(
                messages=Message(role="assistant", contents=contents),
                usage_details=self.usage,
            )

        return _go()

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
        final_prompt="",
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
        final_prompt="",
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
        final_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )
    with_summary = LiveOutcome(
        strategy="s",
        calls=(_call(1, 1, inp=1_000_000, out=100_000),),
        answer="",
        final_prompt="",
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
        final_prompt="",
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
        final_prompt="",
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
        final_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )


def test_representative_repeat_is_the_median_not_an_average() -> None:
    """The reported row must be one conversation that actually happened.

    Blending repeats would let the columns contradict each other: an averaged cost against a
    token count from a different run, and a fact count from a third.
    """
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)
    repeats = [_priced("s", 1_000), _priced("s", 9_000), _priced("s", 5_000)]

    assert _representative(repeats, pricing).input_tokens == 5_000


def test_spread_reports_the_gap_between_repeats() -> None:
    """Spread must show how far repeats of one strategy disagree."""
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)
    repeats = [_priced("s", 8_000), _priced("s", 10_000), _priced("s", 12_000)]

    assert _spread(repeats, pricing) == pytest.approx(0.4)


def test_spread_is_zero_for_a_single_repeat() -> None:
    """One repeat measures nothing about stability, and must not imply otherwise."""
    pricing = ModelPricing(input_per_million=1.0, cached_read_per_million=0.1, output_per_million=1.0)

    assert _spread([_priced("s", 5_000)], pricing) == 0.0


def test_cached_tokens_are_discounted() -> None:
    """Cache reads must be billed at the discounted rate, not the full input rate."""
    pricing = ModelPricing(input_per_million=10.0, cached_read_per_million=1.0, output_per_million=0.0)
    outcome = LiveOutcome(
        strategy="s",
        calls=(_call(1, 1, inp=1_000_000, cached=1_000_000),),
        answer="",
        final_prompt="",
        tool_calls_made=0,
        turns_completed=1,
        turns_total=1,
    )

    assert _cost(outcome, pricing) == pytest.approx(1.0)


# endregion
# region run_live


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
