# Copyright (c) Microsoft. All rights reserved.

"""Drive the recall scenario through a real agent instead of a scripted replay.

The replay harness sends every provider a byte-identical prompt, which is what makes its
cross-provider numbers comparable. It buys that by scripting the assistant's replies, and a
scripted reply is not what an agent accumulates: real replies carry information, vary in
length, and are themselves candidates for eviction. Replay also cannot produce a genuine
tool-calling loop, so a turn is always exactly one model call.

This module gives up byte-identity to get those back. The scenario's user turns go to a
real ``Agent`` with a real tool, and the model writes its own replies into the history that
compaction then acts on. The consequence is that live numbers are **within-model only**:
two models write different replies, so their histories diverge from the first turn and
cannot be placed side by side the way replayed ones can.

There is a second consequence, and it is the interesting one. Because each strategy's
history contains that strategy's own replies, a strategy that compacts badly produces a
worse reply, which becomes worse history, which it compacts again. Replay cannot show that
compounding at all; here it is the thing being measured.

Compaction is wired the way ``create_harness_agent`` wires it, which is not the obvious
way. A ``CompactionProvider``'s ``before_strategy`` is a no-op under per-service-call
history persistence: the agent skips ``HistoryProvider.before_run``, so the provider only
ever sees an empty context. The before phase has to travel as the agent's
``compaction_strategy`` instead, which runs per model call inside the client. Only the
after phase belongs on the provider.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from agent_framework import (
    Agent,
    ChatContext,
    ChatMiddleware,
    CompactionProvider,
    InMemoryHistoryProvider,
    Message,
    create_harness_agent,
)

# Exported by ``agent_framework._compaction.__all__`` but not re-exported at package level.
# A chat middleware cannot otherwise see what compaction actually kept.
from agent_framework._compaction import project_included_messages

from ._metrics import serialize_message
from ._recall import FactOutcome, RecallScenario, build_recall_scenario, score_answer
from ._strategies import StrategyOptions, build_strategy

if TYPE_CHECKING:
    from agent_framework import CompactionStrategy, TokenizerProtocol

    from ._providers import ProviderRuntime

__all__ = [
    "AGENT_KINDS",
    "LiveOutcome",
    "MeteredClient",
    "ModelCall",
    "UsageRecorder",
    "build_live_agent",
    "build_live_scenario",
    "make_lookup_tool",
    "run_live",
    "score_live",
]

#: How the agent under test is assembled.
#:
#: ``plain`` builds the smallest agent that still exercises compaction: history, tools and
#: the strategy, nothing else. ``harness`` builds the real ``create_harness_agent``, which
#: is what production code calls, at the cost of adding its own tools and system prompt to
#: every measured prompt.
AGENT_KINDS: Final[tuple[str, ...]] = ("plain", "harness")

_INSTRUCTIONS: Final[str] = (
    "You are a meticulous engineering assistant. Follow every stated requirement exactly. "
    "When the user asks you to look up deployment facts, call the lookup_deployment tool. "
    "Keep replies short unless asked otherwise."
)


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One model call, as the middleware observed it.

    A turn is not a call. A turn that triggers a tool produces at least two, each billed
    separately against a different prompt, so cost has to be summed per call rather than
    per turn.
    """

    messages_sent: int
    """Messages actually sent, after compaction removed what it removed."""
    prompt_text: str
    messages_before_compaction: int
    """Messages the history held when the call started, before compaction ran."""
    input_tokens: int
    cached_tokens: int
    output_tokens: int

    @property
    def fresh_tokens(self) -> int:
        """Input tokens that were not served from the provider's cache."""
        return max(self.input_tokens - self.cached_tokens, 0)


class UsageRecorder(ChatMiddleware):
    """Record what each model call actually sent, and what it was billed for.

    Everything about *when* this reads the prompt is deliberate, because two separate
    layers sit between this middleware and the request that is finally sent.

    On the way in, ``context.messages`` holds only the new user message. The full history
    is loaded further in, by ``PerServiceCallHistoryPersistingMiddleware``, which replaces
    ``context.messages`` with the loaded list. A reference captured before ``call_next``
    therefore points at a stale one-element list, and would report a five-turn conversation
    as a single message.

    Compaction then runs deeper still, inside ``BaseChatClient.get_response``, and it works
    by mutation: ``apply_compaction`` sets exclusion flags on the ``Message`` objects
    themselves rather than shortening the list it was handed. So after the call completes,
    ``context.messages`` holds every message the turn had, each flagged with whether it
    survived, and projecting it reproduces the prompt the model actually received.

    Reading it any earlier reports the history as one message; reading it without projecting
    reports that every strategy preserved every fact. Both were measured before this was
    written the way it is.
    """

    def __init__(self) -> None:
        """Create a recorder holding no calls."""
        self.calls: list[ModelCall] = []

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        """Run the call, then read back what compaction left of the prompt.

        Args:
            context: The chat invocation being wrapped.
            call_next: Continuation that performs the call.
        """
        await call_next()
        # Read after the call, not before: the history middleware replaces this attribute
        # with the loaded history, and compaction then flags that list in place.
        outgoing = list(context.messages)
        sent = project_included_messages(outgoing)
        # UsageDetails is a TypedDict, so it is read with .get() rather than getattr:
        # attribute access on it silently yields None and reports every call as free.
        usage: dict[str, Any] = dict(getattr(context.result, "usage_details", None) or {})
        self.calls.append(
            ModelCall(
                messages_sent=len(sent),
                prompt_text="\n".join(serialize_message(message) for message in sent),
                messages_before_compaction=len(outgoing),
                input_tokens=usage.get("input_token_count") or 0,
                cached_tokens=usage.get("cache_read_input_token_count") or 0,
                output_tokens=usage.get("output_token_count") or 0,
            )
        )


class MeteredClient:
    """Wrap a chat client so calls made outside the agent are still counted.

    This is a proxy rather than a chat client: it forwards everything it does not record,
    and only ``get_response`` is intercepted, because that is the only call
    ``SummarizationStrategy`` makes. Structurally satisfying the client protocol would mean
    reproducing its four overloads for no benefit, so callers cast instead.

    ``SummarizationStrategy`` calls its client directly, so those calls never reach the
    agent's middleware. Left unmetered, the one strategy that spends extra money to do its
    job would be scored as though it were free, which is the same mistake as pricing an
    absent cache-read rate at zero.

    Failures matter as much as tokens. ``SummarizationStrategy`` catches its own errors,
    logs a warning and returns ``False``, so a broken summarizer produces a run with no
    compaction at all and therefore a *perfect* recall score. Counting failures here is
    what stops that being read as a win.
    """

    def __init__(self, inner: Any) -> None:
        """Wrap a client.

        Args:
            inner: The client to delegate to.
        """
        self.inner = inner
        self.calls = 0
        self.failures = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0

    @property
    def additional_properties(self) -> dict[str, Any]:
        """Delegate to the wrapped client, which the chat-client protocol requires."""
        properties: dict[str, Any] = getattr(self.inner, "additional_properties", {})
        return properties

    def __getattr__(self, name: str) -> Any:
        """Forward everything not overridden here to the wrapped client.

        Returns:
            The wrapped client's attribute.
        """
        return getattr(self.inner, name)

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped client, recording usage and failures.

        Returns:
            Whatever the wrapped client returned.

        Raises:
            Exception: Whatever the wrapped client raised, after counting it.
        """
        self.calls += 1
        try:
            response = await self.inner.get_response(*args, **kwargs)
        except Exception:
            self.failures += 1
            raise
        usage: dict[str, Any] = dict(getattr(response, "usage_details", None) or {})
        self.input_tokens += usage.get("input_token_count") or 0
        self.cached_tokens += usage.get("cache_read_input_token_count") or 0
        self.output_tokens += usage.get("output_token_count") or 0
        return response


@dataclass(frozen=True, slots=True)
class LiveOutcome:
    """Everything one live strategy run produced."""

    strategy: str
    calls: tuple[ModelCall, ...]
    answer: str
    final_prompt: str
    tool_calls_made: int
    turns_completed: int
    turns_total: int
    summarizer_calls: int = 0
    summarizer_failures: int = 0
    summarizer_input_tokens: int = 0
    summarizer_output_tokens: int = 0
    error: str | None = None
    replies: tuple[str, ...] = field(default_factory=tuple[str, ...])

    @property
    def input_tokens(self) -> int:
        """Input tokens billed across every model call."""
        return sum(call.input_tokens for call in self.calls)

    @property
    def cached_tokens(self) -> int:
        """Input tokens served from cache across every model call."""
        return sum(call.cached_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        """Output tokens billed across every model call."""
        return sum(call.output_tokens for call in self.calls)

    @property
    def messages_left(self) -> int:
        """Messages in the last prompt sent, after compaction."""
        return self.calls[-1].messages_sent if self.calls else 0

    @property
    def messages_peak(self) -> int:
        """Largest the history ever got, before compaction was applied to it.

        Measured pre-compaction on purpose. The post-compaction peak only says how hard a
        strategy trimmed; this says how much there was to trim, which is the denominator
        that makes ``messages_left`` mean anything.
        """
        return max((call.messages_before_compaction for call in self.calls), default=0)

    @property
    def prompt_tokens_final(self) -> int:
        """Billed size of the last prompt sent.

        Measured in tokens rather than messages because message counts are blind to
        strategies that rewrite content in place. ``ToolResultCompactionStrategy`` collapses
        tool results into summaries without excluding anything, so it leaves the message
        count untouched while removing real tokens — by the message count alone it looks
        like it did nothing at all.
        """
        return self.calls[-1].input_tokens if self.calls else 0

    @property
    def prompt_tokens_peak(self) -> int:
        """Billed size of the largest prompt any single call carried."""
        return max((call.input_tokens for call in self.calls), default=0)

    @property
    def messages_dropped(self) -> int:
        """Messages the final call's compaction removed from the history."""
        return max(self.calls[-1].messages_before_compaction - self.calls[-1].messages_sent, 0) if self.calls else 0

    @property
    def reply_tokens_in_history(self) -> int:
        """Output tokens the model wrote that then became history for later turns.

        The quantity replay cannot produce. Every one of these is a token some later turn
        had to pay to resend, or that compaction had to decide whether to keep.
        """
        return sum(call.output_tokens for call in self.calls[:-1]) if len(self.calls) > 1 else 0


def make_lookup_tool(lookups: Mapping[str, tuple[str, str]], filler: str = "") -> Callable[[str], str]:
    """Build the tool a live agent calls to obtain the planted tool-result facts.

    The replayed transcript scripts these values into a tool-result message. Live, the model
    has to ask for them, so the returned function must hand back the same markers or the two
    modes would be scoring different conversations.

    Args:
        lookups: Scope label mapped to ``(region_code, fallback_host)``.
        filler: Padding appended to each result, matching the replayed result's bulk so that
            tool-eviction strategies have something worth evicting.

    Returns:
        A callable suitable for passing to ``Agent(tools=...)``.
    """

    def lookup_deployment(scope: str) -> str:
        """Look up the deployment facts for one scope of the system.

        Args:
            scope: Which deployment to look up. One of "early", "mid" or "late".

        Returns:
            The region code and fallback host for that scope.
        """
        entry = lookups.get(scope.strip().casefold())
        if entry is None:
            return f"Unknown scope {scope!r}. Valid scopes are: {', '.join(sorted(lookups))}."
        region, host = entry
        return f"region_code={region}; fallback_host={host}; both values must appear in the final report. {filler}"

    return lookup_deployment


def build_live_agent(
    runtime: ProviderRuntime,
    *,
    kind: str,
    strategy: CompactionStrategy | None,
    tokenizer: TokenizerProtocol,
    tools: Sequence[Callable[..., Any]],
    recorder: UsageRecorder,
    max_context_window_tokens: int,
    max_output_tokens: int,
) -> Agent[Any]:
    """Assemble the agent under test, with compaction wired the way the harness wires it.

    Args:
        runtime: The provider's client, model and per-request options.

    Keyword Args:
        kind: One of :data:`AGENT_KINDS`.
        strategy: The compaction strategy, or ``None`` for the uncompacted control.
        tokenizer: Token counter shared with the strategy.
        tools: Tools the agent may call.
        recorder: Middleware capturing prompts and usage.
        max_context_window_tokens: Window the harness variant sizes its default against.
        max_output_tokens: Output reservation.

    Returns:
        The configured agent.

    Raises:
        ValueError: If ``kind`` is not a known agent kind.
    """
    if kind not in AGENT_KINDS:
        raise ValueError(f"Unknown agent kind {kind!r}. Available: {', '.join(AGENT_KINDS)}")

    if kind == "harness":
        # The harness resolves both phases itself from the strategies handed in, so it gets
        # the same object twice. Its optional providers are switched off: each adds tools
        # and system-prompt text to every measured prompt, which would inflate every cell
        # and move the trigger points without saying anything about compaction.
        return create_harness_agent(
            runtime.client,
            name="cachebench",
            agent_instructions=_INSTRUCTIONS,
            tools=list(tools),
            max_context_window_tokens=max_context_window_tokens,
            max_output_tokens=max_output_tokens,
            disable_compaction=strategy is None,
            before_compaction_strategy=strategy,
            after_compaction_strategy=strategy,
            tokenizer=tokenizer,
            disable_todo=True,
            disable_mode=True,
            disable_file_memory=True,
            disable_web_search=True,
            middleware=[recorder],
            default_options=dict(runtime.options),
        )

    history = InMemoryHistoryProvider()
    providers: list[Any] = [history]
    if strategy is not None:
        # before_strategy is deliberately None: on a provider it would never run. The
        # before phase travels as the agent's compaction_strategy below.
        providers.append(
            CompactionProvider(
                before_strategy=None,
                after_strategy=strategy,
                tokenizer=tokenizer,
                history_source_id=history.source_id,
            )
        )
    return Agent(
        client=runtime.client,
        name="cachebench",
        instructions=_INSTRUCTIONS,
        tools=list(tools),
        context_providers=providers,
        compaction_strategy=strategy,
        require_per_service_call_history_persistence=True,
        middleware=[recorder],
        default_options=dict(runtime.options),
    )


def _turn_text(messages: Sequence[Message]) -> str:
    """Flatten a scenario turn's request messages into the text to send."""
    return "\n".join(
        text for message in messages for content in message.contents if (text := getattr(content, "text", None))
    )


async def run_live(
    runtime: ProviderRuntime,
    *,
    strategy_name: str,
    options: StrategyOptions,
    scenario: RecallScenario,
    agent_kind: str = "plain",
    tool_filler: str = "",
) -> LiveOutcome:
    """Run the scenario end to end against a real agent.

    Args:
        runtime: The provider's client, model and per-request options.

    Keyword Args:
        strategy_name: Strategy to install; ``"none"`` for the uncompacted control.
        options: Budget and tokenizer parameters for the strategy.
        scenario: The scenario to drive, built with ``bulk_in_user=True``.
        agent_kind: One of :data:`AGENT_KINDS`.
        tool_filler: Padding appended to each tool result.

    Returns:
        The outcome. A turn that fails sets ``error`` and stops the run rather than raising,
        so a partial result is still reported instead of losing the spend already made.
    """
    strategy = build_strategy(strategy_name, options)
    recorder = UsageRecorder()
    summarizer = options.summarizer if isinstance(options.summarizer, MeteredClient) else None
    lookup = make_lookup_tool(scenario.tool_lookups, tool_filler)
    tool_calls = 0

    def lookup_deployment(scope: str) -> str:
        """Look up the deployment facts for one scope of the system.

        Args:
            scope: Which deployment to look up. One of "early", "mid" or "late".

        Returns:
            The region code and fallback host for that scope.
        """
        nonlocal tool_calls
        tool_calls += 1
        return lookup(scope)

    agent = build_live_agent(
        runtime,
        kind=agent_kind,
        strategy=strategy,
        tokenizer=options.tokenizer,
        tools=[lookup_deployment],
        recorder=recorder,
        max_context_window_tokens=options.max_context_window_tokens,
        max_output_tokens=options.max_output_tokens,
    )
    session = agent.create_session()

    turns = scenario.transcript.turns
    replies: list[str] = []
    error: str | None = None
    completed = 0
    answer = ""
    final_mark = 0
    for index, turn in enumerate(turns):
        final_mark = len(recorder.calls)
        try:
            response = await agent.run(_turn_text(turn.request), session=session)
        except Exception as exc:
            error = f"turn {index + 1}: {type(exc).__name__}: {exc}"
            break
        completed += 1
        text = response.text or ""
        replies.append(text)
        if index == len(turns) - 1:
            answer = text

    # Survival is judged against every prompt sent during the final turn, not just the last
    # one. A turn that calls a tool sends several, and a fact the model saw in any of them
    # was available to it when it wrote the answer.
    final_prompt = "\n".join(call.prompt_text for call in recorder.calls[final_mark:])

    return LiveOutcome(
        strategy=strategy_name,
        calls=tuple(recorder.calls),
        answer=answer,
        final_prompt=final_prompt,
        tool_calls_made=tool_calls,
        turns_completed=completed,
        turns_total=len(turns),
        summarizer_calls=summarizer.calls if summarizer else 0,
        summarizer_failures=summarizer.failures if summarizer else 0,
        summarizer_input_tokens=summarizer.input_tokens if summarizer else 0,
        summarizer_output_tokens=summarizer.output_tokens if summarizer else 0,
        error=error,
        replies=tuple(replies),
    )


def build_live_scenario(*, salt: str, filler_turns: int, filler_tokens: int) -> RecallScenario:
    """Build the scenario in the shape a live run needs.

    Keyword Args:
        salt: Cell-unique string; markers and filler derive from it.
        filler_turns: Padding turns between the planted facts.
        filler_tokens: Approximate size of each filler exchange.

    Returns:
        A scenario whose padding sits in the user turns, because the assistant's replies are
        generated rather than scripted.
    """
    return build_recall_scenario(
        salt=salt,
        filler_turns=filler_turns,
        filler_tokens=filler_tokens,
        bulk_in_user=True,
    )


def score_live(outcome: LiveOutcome, scenario: RecallScenario) -> tuple[FactOutcome, ...]:
    """Score a live outcome's answer against the planted facts.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Returns:
        One outcome per planted fact.
    """
    return score_answer(outcome.answer, scenario.facts, outcome.final_prompt)
