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
from ._recall import (
    FactOutcome,
    RecallScenario,
    build_recall_scenario,
    render_code,
    render_codes,
    score_answer,
    score_scoped,
)
from ._runner import unsupported_option
from ._strategies import StrategyOptions, build_strategy
from ._toolsummary import (
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
)
from ._transcripts import TRUE_CHARS_PER_TOKEN, sized_text

if TYPE_CHECKING:
    from agent_framework import CompactionStrategy, TokenizerProtocol

    from ._providers import ProviderRuntime

__all__ = [
    "AGENT_KINDS",
    "COMPACTION_GUIDANCE",
    "DEFAULT_TOOL_RESULT_TOKENS",
    "NEUTRAL_INSTRUCTIONS",
    "RETRIEVAL_GUIDANCE",
    "TERSE_INSTRUCTIONS",
    "LiveOutcome",
    "MeteredClient",
    "ModelCall",
    "RecallGate",
    "UsageRecorder",
    "build_live_agent",
    "build_live_scenario",
    "make_lookup_tool",
    "make_recall_tool",
    "make_scope_tools",
    "resolve_instructions",
    "run_live",
    "score_combined",
    "score_live",
    "unretrieved_facts",
    "wants_client_side_history",
]

#: How the agent under test is assembled.
#:
#: ``plain`` builds the smallest agent that still exercises compaction: history, tools and
#: the strategy, nothing else. ``harness`` builds the real ``create_harness_agent``, which
#: is what production code calls, at the cost of adding its own tools and system prompt to
#: every measured prompt.
AGENT_KINDS: Final[tuple[str, ...]] = ("plain", "harness")

#: Default size of each tool result, in tokens. Set high on purpose: in a real agent
#: trace tool output is usually the bulk of the context, and a benchmark whose tool
#: results are a rounding error cannot say anything about tool-oriented compaction.
DEFAULT_TOOL_RESULT_TOKENS: Final[int] = 4_000

TERSE_INSTRUCTIONS: Final[str] = (
    "You are a meticulous engineering assistant. Follow every stated requirement exactly. "
    "When the user asks for a deployment lookup, call the matching tool. "
    "Acknowledge each tool result in three words or fewer, and do not restate its values in "
    "that acknowledgement. This applies only to acknowledgements: when you are asked for the "
    "final report, include every value you were asked for, in full."
)
"""Instructions that stop the model narrating tool results back into the conversation.

Separates a strategy's contribution from the model's. When the model restates every value, a
strategy can discard the tool results entirely and still appear to preserve them. The
final-report exemption is load-bearing: without it the model applies the rule to its answer
too, and the recall score measures the instruction rather than the compaction.
"""

COMPACTION_GUIDANCE: Final[str] = (
    "Earlier tool results may have been shortened, summarised or replaced by a compaction "
    "record. Treat values in such a record as authoritative for the tool it names, and treat "
    "information as absent only if it appears nowhere, including there."
)
"""How to read what compaction left behind.

Given to *every* row, including the uncompacted control, which is what keeps it a measurement
device rather than an advantage for the strategies that leave artefacts. Without it a strategy
that removed the original results is scored by a question naming a tool whose result it
deleted, against an instruction inviting the answer "no longer present" -- so the score partly
measures whether the model thought to look at the record rather than whether the record
preserved anything.

Legitimate by the same test applied to the narration guidance: it changes *whether the model
looks*, not *where the information is*, and cannot resurrect a value the record does not
contain. It is inert for the control, which has no artefacts to interpret, and that asymmetry
belongs in any report of the numbers.
"""

RETRIEVAL_GUIDANCE: Final[str] = (
    "When asked for codes or identifiers, quote them exactly as they appear earlier in this "
    "conversation, and list every one you are asked for. If a value is not present in the "
    "conversation, say so plainly for that item instead of guessing or inventing one. " + COMPACTION_GUIDANCE
)
"""The retrieval clause, isolated so a run can measure what it is worth.

Kept separate from the instructions it is appended to because it is the only sentence
suspected of holding the closing answer stable, and a suspicion that cannot be switched off
cannot be tested.
"""

NEUTRAL_INSTRUCTIONS: Final[str] = (
    "You are a meticulous engineering assistant. Follow every stated requirement exactly. " + RETRIEVAL_GUIDANCE
)
"""Agent instructions that say nothing about narrating tool results.

Paired with ``narration="neutral"`` and the harness, this is the configuration a typical
caller gets: the framework's own ``DEFAULT_HARNESS_INSTRUCTIONS`` are then the only thing
telling the model to explain what it learned between tool calls.

The retrieval guidance is deliberate, and it is a different kind of instruction from the
narration guidance. Narration changes *where the information is*, copying tool output into
assistant prose, which lets a strategy delete the original and still appear lossless.
Retrieval guidance only changes *whether the model looks* for what is already there; it
cannot resurrect a fact compaction removed. Only the first kind can mask damage.

It is not, however, what made the closing answer stable. That was the reply cap. Measured on
the uncompacted control asking for all 53 codes at once: without this clause a 900-token cap
scored 33% with 36 facts present but unlisted, and raising the cap to 4,000 scored 100% with
no clause at all. The guidance had been compensating for a truncated answer by pushing codes
ahead of prose. At an adequate cap it changes nothing here, and it is kept for continuity with
the runs already measured rather than because it is doing work.

The "say so plainly" clause guards the other direction: a model that invents a plausible
code would score as recall without the fact ever being in context.
"""

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
    dropped_options: tuple[str, ...] = ()
    """Request options the provider rejected, dropped so the run could continue.

    A run that dropped ``tool_choice`` is not comparable with one that kept it: the model
    picked its own tool calls, so it gathered its own set of facts.
    """
    scopes_called: tuple[str, ...] = ()
    """Tool scopes the agent actually asked for.

    A fact the agent never fetched never entered the history, so compaction cannot have
    evicted it. Without this the uncompacted control reports losing facts to compaction,
    which is not a thing that can happen.
    """
    summarizer_calls: int = 0
    summarizer_failures: int = 0
    #: Free-form notes a strategy chose to report about its own run, shown in the flags
    #: column. A strategy that can silently degrade into a different one has to say so:
    #: this package has twice read a row that scored well for having done nothing.
    strategy_notes: tuple[str, ...] = ()
    #: One reply per closing turn, so each can be scored against the question that asked for
    #: it rather than against all of them joined.
    answers: tuple[str, ...] = ()
    #: The reply to the combined question, scored separately.
    combined_answer: str = ""
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


def _strategy_notes(strategy: Any) -> tuple[str, ...]:
    """Return what a strategy reports about its own run, if it reports anything.

    Read by duck typing rather than by isinstance so a strategy from outside this package can
    surface its own diagnostics without the runner knowing about it. Only counts that are
    non-zero are reported, so a clean run adds no noise to the flags column.

    Args:
        strategy: The strategy that was installed, or None for the control.

    Returns:
        Short tokens for the flags column.
    """
    notes: list[str] = []
    for attribute, label in (
        ("records_found", "REC"),
        ("fallbacks_used", "FALLBACK"),
        ("forced_calls", "FORCED"),
        ("records_forced", "RECFORCED"),
        ("records_volunteered", "RECVOLUNTEERED"),
        ("compactions_skipped", "FROZEN"),
    ):
        value = getattr(strategy, attribute, None)
        if isinstance(value, int) and value:
            notes.append(f"{label}:{value}")
    return tuple(notes)


def wants_client_side_history(client: Any, *, allow_server_history: bool = False) -> bool:
    """Return whether ``store=False`` must be forced so compaction can act.

    Clients on the Responses API keep the conversation server-side. When they do, MAF skips
    ``HistoryProvider.before_run`` entirely -- the comment in the framework is explicit that
    "the service owns loading; the providers are write-only sinks" -- and the agent sends
    only the new turn. A compaction strategy then has nothing to compact, and every setting
    silently measures the same thing.

    Measured on Foundry before this was forced: a 16-turn conversation reported a one-message
    prompt on every row, while the service billed 82,708 input tokens for history the client
    never sent.

    Args:
        client: The chat client under test.

    Keyword Args:
        allow_server_history: Leave the service in charge, accepting that no compaction runs.

    Returns:
        True when ``store=False`` should be forced.
    """
    return bool(getattr(client, "STORES_BY_DEFAULT", False)) and not allow_server_history


def make_lookup_tool(
    lookups: Mapping[str, tuple[str, ...]],
    filler_tokens: int = DEFAULT_TOOL_RESULT_TOKENS,
) -> Callable[[str], str]:
    """Build the tool a live agent calls to obtain the planted tool-result facts.

    The replayed transcript scripts these values into a tool-result message. Live, the model
    has to ask for them, so the returned function must hand back the same markers or the two
    modes would be scoring different conversations.

    Args:
        lookups: Scope label mapped to the verifiable codes it carries.
        filler_tokens: Approximate size of each result, **in tokens**. This is the only thing
            that decides how much context tool output occupies, and therefore whether
            tool-oriented compaction has anything worth evicting. At the ~76 tokens a
            600-character default produced, six results came to under 2% of a 28,000-token
            prompt and ``tool_result`` could move only 1.2% of it.

    Returns:
        A callable suitable for passing to ``Agent(tools=...)``.
    """
    # Distinct text per scope: six identical results would share a prefix and let unrelated
    # messages match by accident, inflating measured cache reuse.
    bodies = {
        scope: sized_text(f"[{scope} deployment notes] ", index * 31 + 7, filler_tokens, TRUE_CHARS_PER_TOKEN)
        for index, scope in enumerate(sorted(lookups))
    }

    def lookup_deployment(scope: str) -> str:
        """Look up the deployment facts for one scope of the system.

        Args:
            scope: Which deployment to look up, such as "early", "mid" or "late".

        Returns:
            The region code and fallback host for that scope.
        """
        key = scope.strip().casefold()
        entry = lookups.get(key)
        if entry is None:
            return f"Unknown scope {scope!r}. Valid scopes are: {', '.join(sorted(lookups))}."
        return f"{render_codes(entry)}; all of these values must appear in the final report. {bodies[key]}"

    return lookup_deployment


def _scope_tool(scope: str, result: str) -> Callable[[], str]:
    """Return one no-argument tool that hands back a fixed result.

    A closure rather than a default argument. Capturing the result as ``def tool(_result=...)``
    puts it in the function signature, and the framework turns the signature into the tool
    schema -- so the whole result body would be advertised to the model as a parameter it
    could set.

    Args:
        scope: The deployment scope this tool reports on.
        result: The text to return.

    Returns:
        A zero-argument callable named ``lookup_<scope>``.
    """

    def tool() -> str:
        return result

    tool.__name__ = f"lookup_{scope}"
    tool.__doc__ = (
        f"Look up the deployment facts for the {scope} deployment."
        + chr(10)
        + chr(10)
        + "Returns:"
        + chr(10)
        + f"    The region code and fallback host for the {scope} deployment."
        + chr(10)
    )
    return tool


class RecallGate:
    """One-shot permission for the recall tool.

    The tool cannot be hidden from the model. It has to be registered with the harness for
    ``FunctionInvocationLayer`` to execute it, and that layer wraps the middleware layer, so a
    tool supplied per call reaches the model but never the executor: measured, the model's
    call simply went unanswered. A registered tool is advertised on every request, and this
    one was called uninvited on the unpinned follow-up call in every run.

    Permission is therefore separated from visibility. The middleware arms the gate
    immediately before the request it forces, and the tool records only while armed. An
    uninvited call still runs and still answers honestly; it just produces no record.
    """

    def __init__(self) -> None:
        """Start disarmed, so nothing is recorded until something asks for it."""
        self._armed = False

    def arm(self) -> None:
        """Permit the next call to record."""
        self._armed = True

    def take(self) -> bool:
        """Consume the permission.

        Returns:
            True if this call may record. One-shot: a single arming cannot licence a second
            record, which would drop results the first had already replaced.
        """
        armed, self._armed = self._armed, False
        return armed


class CompactionSwitch:
    """Whether compaction may still run, and how often it was stopped.

    The closing questions go through the same ``agent.run()`` loop as every other turn, so by
    default the strategy fires before each of them. Three things follow, none of them
    controlled: the first scope is answered from a fuller context than the last; the combined
    question is answered from the most compacted context of the whole run; and a fact can be
    evicted *during* scoring, so ``survived`` is computed against a prompt that is still
    moving. Freezing at the first closing turn holds one context still for all of them, so
    the answers differ only in what they were asked.

    Off by default, because the unfrozen run is the one an agent would actually perform.
    """

    def __init__(self) -> None:
        """Start thawed: nothing is frozen until the runner reaches the closing turns."""
        self._frozen = False
        self._skipped = 0

    @property
    def frozen(self) -> bool:
        """True once the closing turns have begun."""
        return self._frozen

    @property
    def skipped(self) -> int:
        """How many compaction calls the freeze suppressed.

        Zero on a frozen run means the freeze changed nothing, which is worth telling apart
        from the freeze never having been asked for.
        """
        return self._skipped

    def freeze(self) -> None:
        """Stop compaction for the rest of the run."""
        self._frozen = True

    def note_skip(self) -> None:
        """Record one suppressed call."""
        self._skipped += 1


class _FreezableStrategy:
    """A compaction strategy that can be switched off part-way through a run.

    Everything except ``__call__`` reads through to the wrapped strategy, so the counters the
    runner reports and the configuration the middleware copies behave as though it were not
    here. Installed only when the freeze is asked for, which keeps the default path
    byte-identical to the runs already recorded.
    """

    def __init__(self, inner: Any, switch: CompactionSwitch) -> None:
        """Wrap ``inner``, consulting ``switch`` before every call."""
        self._inner = inner
        self._switch = switch

    async def __call__(self, messages: list[Message]) -> bool:
        """Run the wrapped strategy unless the switch is frozen.

        Returns:
            What the wrapped strategy returned, or False when frozen, meaning nothing changed.
        """
        if self._switch.frozen:
            self._switch.note_skip()
            return False
        return await self._inner(messages)

    @property
    def compactions_skipped(self) -> int:
        """Suppressed calls, surfaced in the flags column by :func:`_strategy_notes`."""
        return self._switch.skipped

    def __getattr__(self, name: str) -> Any:
        """Read anything else off the wrapped strategy.

        Reached only for names the wrapper does not define. Goes through ``__dict__`` rather
        than ``self._inner`` so that a lookup arriving before ``__init__`` has finished raises
        rather than recursing.

        Returns:
            The wrapped strategy's attribute.
        """
        return getattr(self.__dict__["_inner"], name)


def make_recall_tool(gate: RecallGate | None = None) -> Callable[[str], str]:
    """Build the tool ``ToolResultAnchoredSummarizationCompactionStrategy`` anchors on.

    It echoes what it is given straight back. That is the whole point: the value of the call
    is not what the tool computes but that the model's own recollection ends up in the
    transcript as a tool result, which the provider issued and which survives strategies that
    shed assistant prose.

    Args:
        gate: Permission to record. Without one the tool always records, which is only right
            for a caller driving it deliberately.

    Returns:
        A callable named :data:`RECALL_TOOL_NAME`.
    """

    def tool(values: str) -> str:
        if gate is not None and not gate.take():
            return "Not required right now: nothing was recorded, and no results have been removed."
        return (
            f"{RECORD_MARKER} Earlier tool results may have been shortened, and this is their "
            "compaction record. Treat values in this record as authoritative for the tool it "
            "names, and treat information as absent only if it appears nowhere, including "
            f"here.\n{values}"
        )

    tool.__name__ = RECALL_TOOL_NAME
    tool.__doc__ = (
        "Record identifiers and values seen in earlier tool results, so they survive when "
        "those results are removed from the conversation to save space.\n\n"
        "Args:\n    values: Every identifier, code and concrete value, verbatim, grouped by "
        "the tool that returned it."
    )
    return tool


def make_scope_tools(
    lookups: Mapping[str, tuple[str, ...]],
    filler_tokens: int = DEFAULT_TOOL_RESULT_TOKENS,
    narration: str = "prompted",
    placement: str = "spread",
) -> list[Callable[[], str]]:
    """Build one no-argument tool per scope, so the wrong scope cannot be requested.

    A single ``lookup_deployment(scope)`` tool leaves the choice of scope to the model, and
    ``tool_choice="required"`` cannot constrain an argument -- only which function is called.
    Measured on one model: forcing a call raised tool use from 4 to 7 calls per run but it
    still reached only 3 of 6 scopes, calling one twice and skipping another. Splitting the
    tool per scope makes ``required_function_name`` sufficient to pin exactly which fact the
    turn gathers.

    Args:
        lookups: Scope label mapped to the verifiable codes it carries.
        filler_tokens: Approximate size of each result, in tokens.
        narration: Whether the result text asks the model to restate its values.
        placement: ``"spread"`` distributes the codes on their own labelled lines;
            ``"buried"`` distributes them inline in prose, where finding them is itself part
            of the task; ``"head"`` puts them all at the front, which lets a head-truncating
            strategy preserve every fact for free.

    Returns:
        One callable per scope, named ``lookup_<scope>``.
    """
    tools: list[Callable[[], str]] = []
    for index, scope in enumerate(sorted(lookups)):
        codes = lookups[scope]
        body = sized_text(f"[{scope} deployment notes] ", index * 31 + 7, filler_tokens, TRUE_CHARS_PER_TOKEN)
        # This trailing instruction is what makes the model restate the values in its own
        # reply. Dropping it leaves the facts only in the tool result.
        preamble = "" if narration != "prompted" else "all of these values must appear in the final report. "
        if placement == "head":
            result = f"{render_codes(codes)}; {preamble}{body}"
        else:
            result = f"{preamble}{_spread_codes(codes, body, labelled=placement == 'spread')}"
        tools.append(_scope_tool(scope, result))
    return tools


def _spread_codes(codes: Sequence[str], body: str, *, labelled: bool) -> str:
    """Distribute labelled codes evenly through a tool result instead of heading it.

    Placement decides what a size-reducing strategy can destroy.
    ``ToolResultCompactionStrategy`` head-truncates a collapsed result at 4,096 characters,
    so codes sitting at the front survive that cut unconditionally no matter how large the
    result is. That flatters every tool-oriented strategy, and the flattery grows with the
    result size: at 25,200 tokens a head-placed code set is 0.6% of the text and 100% of the
    scored content.

    Spreading them makes the result behave like a real one, where the useful line is as
    likely to be in the middle as at the top.

    Args:
        codes: The verifiable codes this tool result carries.
        body: Filler text to distribute them through.

    Keyword Args:
        labelled: Give each code its own line with a ``[record N]`` prefix. Unlabelled, the
            codes go inline in running prose, which is a materially harder task: two
            independent controls read the first code of each tool result and none of the
            other seven, scoring exactly 11 of 53 both times.

    Returns:
        The body with one code inserted before each of ``len(codes)`` evenly spaced
        segments, at a word boundary so no code is glued to a partial word.
    """
    if not codes:
        return body
    words = body.split(" ")
    # One segment per code, so the first code stays near the front and the last sits near the
    # end. Anything less even would leave a head-heavy result and reproduce the problem.
    step = max(len(words) // len(codes), 1)
    parts: list[str] = []
    for index, code in enumerate(codes):
        start = index * step
        end = (index + 1) * step if index + 1 < len(codes) else len(words)
        segment = " ".join(words[start:end])
        if labelled:
            parts.append(f"\n[record {index + 1}] {render_code(index, code)}\n{segment}")
        else:
            parts.append(f"{render_code(index, code)}; {segment} ")
    return "".join(parts)


def build_live_agent(
    runtime: ProviderRuntime,
    *,
    kind: str,
    strategy: CompactionStrategy | None,
    tokenizer: TokenizerProtocol,
    tools: Sequence[Callable[..., Any]],
    recorder: UsageRecorder,
    extra_middleware: Sequence[Any] = (),
    instructions: str = _INSTRUCTIONS,
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
        extra_middleware: Further middleware a strategy needs, such as the one that forces the
            recall call. Installed after the recorder so the recorder still sees every call.
        instructions: System instructions for the agent.
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
            agent_instructions=instructions,
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
            middleware=[recorder, *extra_middleware],
            # Deliberately empty: every option travels per turn instead. An option baked
            # in here cannot be dropped when a provider rejects it without rebuilding the
            # agent, which would discard the session the conversation lives in.
            default_options={},
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
        instructions=instructions,
        tools=list(tools),
        context_providers=providers,
        compaction_strategy=strategy,
        require_per_service_call_history_persistence=True,
        middleware=[recorder, *extra_middleware],
        default_options={},
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
    tool_result_tokens: int = DEFAULT_TOOL_RESULT_TOKENS,
    force_tool_calls: bool = True,
    narration: str = "prompted",
    retrieval_guidance: bool = True,
    fact_placement: str = "spread",
    allow_server_history: bool = False,
    freeze_during_answers: bool = False,
) -> LiveOutcome:
    """Run the scenario end to end against a real agent.

    Args:
        runtime: The provider's client, model and per-request options.

    Keyword Args:
        strategy_name: Strategy to install; ``"none"`` for the uncompacted control.
        options: Budget and tokenizer parameters for the strategy.
        scenario: The scenario to drive, built with ``bulk_in_user=True``.
        agent_kind: One of :data:`AGENT_KINDS`.
        tool_result_tokens: Approximate size of each tool result, in tokens.
        narration: How hard the scenario and instructions push the model to restate tool
            values. See :data:`_INSTRUCTIONS_BY_NARRATION`.
        force_tool_calls: Set ``tool_choice='required'`` on the turns that ask for a
            lookup. Without it a model that ignores the instruction gathers fewer facts
            and carries fewer tokens, which moves both axes for reasons unrelated to
            compaction: measured at 3 of 6 scopes reached and a 33% input swing between
            identical runs on one model, against 6 of 6 and 8% on another.
        retrieval_guidance: Append the clause telling the model to quote every identifier it
            is asked for. Dropping it measures the model's own willingness to enumerate,
            which is a different thing from what compaction left behind.
        fact_placement: Where the verifiable codes sit inside each tool result. ``"spread"``
            distributes them; ``"head"`` reproduces the earlier runs, in which every code sat
            inside the first 4,096 characters and so survived head-truncating compaction
            unconditionally.
        allow_server_history: Leave a Responses-API client in charge of the conversation,
            accepting that no compaction runs. Off by default, and forced here rather than
            left to the caller: a calibration probe that forgot it reported every narration
            mode as stable, because the service was feeding the model a history the client
            had never compacted.
        freeze_during_answers: Stop compacting once the closing questions begin, so every
            closing answer is written from the same history. See :class:`CompactionSwitch`
            for what this controls for. Off by default, because an agent in use compacts
            while it answers, and the default row should be that agent.

    Returns:
        The outcome. A turn that fails sets ``error`` and stops the run rather than raising,
        so a partial result is still reported instead of losing the spend already made.
    """
    if wants_client_side_history(runtime.client, allow_server_history=allow_server_history):
        runtime.options["store"] = False
    strategy = build_strategy(strategy_name, options)
    recorder = UsageRecorder()
    summarizer = options.summarizer if isinstance(options.summarizer, MeteredClient) else None
    tool_calls = 0
    scopes_called: list[str] = []

    def _wrap(scope: str, inner: Callable[[], str]) -> Callable[[], str]:
        def recorded() -> str:
            nonlocal tool_calls
            tool_calls += 1
            scopes_called.append(scope)
            return inner()

        recorded.__name__ = inner.__name__
        recorded.__doc__ = inner.__doc__
        return recorded

    scope_tools = [
        _wrap(name.removeprefix("lookup_"), fn)
        for fn in make_scope_tools(
            scenario.tool_lookups, tool_result_tokens, narration=narration, placement=fact_placement
        )
        if (name := fn.__name__)
    ]

    # The recall tool is registered only for the strategy that asks the model to call it.
    # Adding it to every row would put an extra tool in every prompt and give unrelated
    # strategies something new to call, which is a difference between rows that has nothing
    # to do with compaction.
    # Flipped by the loop below at the first closing turn. The middleware consults it too:
    # forcing a fresh record mid-answer would move the history the answers are written from,
    # which is the one thing the freeze exists to hold still.
    switch = CompactionSwitch()
    recall_middleware: ToolResultRecallMiddleware | None = None
    if isinstance(strategy, ToolResultAnchoredSummarizationCompactionStrategy):
        gate = RecallGate()
        # Registered like any other tool, because the harness must know it to run it, and
        # inert until the middleware arms it, because it cannot be hidden from the model.
        scope_tools = [*scope_tools, make_recall_tool(gate)]
        recall_middleware = ToolResultRecallMiddleware(
            max_input_tokens=strategy.max_input_tokens,
            tokenizer=options.tokenizer,
            arm=gate.arm,
            trigger_fraction=strategy.trigger_fraction,
            paused=(lambda: switch.frozen) if freeze_during_answers else None,
        )

    # Wrapped only when the freeze is asked for, so an ordinary run is exactly the run every
    # recorded result was produced by.
    installed: Any = strategy
    if strategy is not None and freeze_during_answers:
        installed = _FreezableStrategy(strategy, switch)

    agent = build_live_agent(
        runtime,
        kind=agent_kind,
        strategy=installed,
        tokenizer=options.tokenizer,
        tools=scope_tools,
        recorder=recorder,
        extra_middleware=[recall_middleware] if recall_middleware else [],
        instructions=resolve_instructions(narration, retrieval_guidance=retrieval_guidance),
        max_context_window_tokens=options.max_context_window_tokens,
        max_output_tokens=options.max_output_tokens,
    )
    session = agent.create_session()

    turns = scenario.transcript.turns
    # Every closing turn is scored. With several targeted questions the answer is their union,
    # and survival is judged against the union of the prompts that carried them.
    first_answer_turn = len(turns) - max(scenario.answer_turn_count, 1)
    answer_parts: list[str] = []
    replies: list[str] = []
    error: str | None = None
    completed = 0
    answer = ""
    final_mark = 0
    forced: dict[int, str] = dict(scenario.tool_turn_scopes) if force_tool_calls else {}
    dropped: list[str] = []
    for index, turn in enumerate(turns):
        if freeze_during_answers and index == first_answer_turn:
            # Frozen on the way in to the first closing turn, not after it: the strategy runs
            # before the model call, so freezing afterwards would already have let it act on
            # the turn it was meant to protect.
            switch.freeze()
        final_mark = len(recorder.calls)
        response = None
        # Two attempts. Providers differ in which request options they accept, and one that
        # rejects an option names it. Dropping that option and retrying is what lets a model
        # with an unusual surface be measured at all instead of returning an empty run:
        # measured on two of five models, one rejecting temperature and one rejecting any
        # pinned tool choice.
        for _ in range(2):
            # Per-turn options carry the runtime's own options too: this replaces the
            # per-call option set rather than adding to it.
            turn_options: dict[str, Any] = {k: v for k, v in runtime.options.items() if k not in dropped}
            if "tool_choice" not in dropped:
                if index in forced:
                    # Name the function, not just "required". Requiring *a* call still lets
                    # the model pick the scope, and it picks wrong: measured reaching 3 of 6
                    # scopes while calling one of them twice.
                    turn_options["tool_choice"] = {
                        "mode": "required",
                        "required_function_name": f"lookup_{forced[index]}",
                    }
                elif force_tool_calls:
                    # Every other turn is closed to tools. Pinning only the wanted calls
                    # still leaves the model free to make unwanted ones: measured at 12 calls
                    # against the 6 asked for, on one repeat in three.
                    turn_options["tool_choice"] = "none"
            try:
                response = await agent.run(_turn_text(turn.request), session=session, options=turn_options)
                break
            except Exception as exc:
                option = unsupported_option(exc)
                if option is None or option in dropped:
                    error = f"turn {index + 1}: {type(exc).__name__}: {exc}"
                    break
                dropped.append(option)
                if option == "tool_choice":
                    forced = {}
        if response is None:
            if error is None:
                error = f"turn {index + 1}: no response"
            break
        completed += 1
        text = response.text or ""
        replies.append(text)
        if index >= first_answer_turn:
            answer_parts.append(text)

    # Survival is judged against every prompt sent during the final turn, not just the last
    # one. A turn that calls a tool sends several, and a fact the model saw in any of them
    # was available to it when it wrote the answer.
    final_prompt = "\n".join(call.prompt_text for call in recorder.calls[final_mark:])
    answer = chr(10).join(answer_parts)
    # The combined turn is the last one when the scenario has scopes; it is scored on its own
    # so a failure to assemble everything cannot drag down the per-scope figure, and a good
    # per-scope figure cannot hide a failure to assemble.
    combined = answer_parts[-1] if scenario.answer_scopes[-1:] == ("*",) and answer_parts else ""

    return LiveOutcome(
        strategy=strategy_name,
        calls=tuple(recorder.calls),
        answer=answer,
        final_prompt=final_prompt,
        tool_calls_made=tool_calls,
        turns_completed=completed,
        turns_total=len(turns),
        dropped_options=tuple(dropped),
        scopes_called=tuple(scopes_called),
        summarizer_calls=summarizer.calls if summarizer else 0,
        answers=tuple(answer_parts),
        combined_answer=combined,
        summarizer_failures=summarizer.failures if summarizer else 0,
        strategy_notes=_strategy_notes(installed) + _strategy_notes(recall_middleware),
        summarizer_input_tokens=summarizer.input_tokens if summarizer else 0,
        summarizer_output_tokens=summarizer.output_tokens if summarizer else 0,
        error=error,
        replies=tuple(replies),
    )


def unretrieved_facts(outcome: LiveOutcome, scenario: RecallScenario) -> tuple[FactOutcome, ...]:
    """Return the facts the agent never fetched, so compaction never had them.

    Scored separately from compaction damage on purpose. A tool result only enters the
    history if the model chooses to call that tool; if it never does, the markers it would
    have carried were never in the conversation at all. Counting those as evicted makes the
    uncompacted control appear to lose information, and inflates the apparent damage of every
    strategy by the same amount.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Returns:
        One outcome per fact that was never retrieved.
    """
    called = set(outcome.scopes_called)
    missing_markers = {
        marker for scope, pair in scenario.tool_lookups.items() if scope.casefold() not in called for marker in pair
    }
    return tuple(
        FactOutcome(fact=fact, survived=False, recalled=fact.appears_in(outcome.answer))
        for fact in scenario.facts
        if fact.marker in missing_markers
    )


def build_live_scenario(
    *,
    salt: str,
    filler_turns: int,
    filler_tokens: int,
    tool_turns: int = 6,
    narration: str = "prompted",
    markers_per_tool: int = 2,
    filler_tool_turns: int = 0,
    subset_questions: bool = True,
) -> RecallScenario:
    """Build the scenario in the shape a live run needs.

    Keyword Args:
        salt: Cell-unique string; markers and filler derive from it.
        filler_turns: Padding turns between the planted facts.
        filler_tokens: Approximate size of each filler exchange.
        tool_turns: Tool-call groups to plant. Defaults above the framework's
            ``keep_last_tool_call_groups`` of 4, so that tool-oriented strategies
            actually engage instead of scoring a perfect result for doing nothing.
        narration: How hard the scenario pushes the model to restate tool values.
        filler_tool_turns: Extra tool calls whose results carry no codes, so that adding
            calls does not also add values to remember.
        markers_per_tool: Verifiable codes each tool result carries.
        subset_questions: Close with several targeted questions rather than one sweeping
            one. On by default: the sweeping form measures stamina, not retrieval.

    Returns:
        A scenario whose padding sits in the user turns, because the assistant's replies are
        generated rather than scripted.
    """
    return build_recall_scenario(
        salt=salt,
        filler_turns=filler_turns,
        filler_tokens=filler_tokens,
        bulk_in_user=True,
        tool_turns=tool_turns,
        filler_tool_turns=filler_tool_turns,
        narration=narration,
        markers_per_tool=markers_per_tool,
        subset_questions=subset_questions,
    )


def score_live(outcome: LiveOutcome, scenario: RecallScenario) -> tuple[FactOutcome, ...]:
    """Score a live outcome against the planted facts, per scope where the scenario allows.

    Scoped scoring matches each fact only against the reply to the question that asked for
    it. Joining the replies first lets a code answered under the wrong heading count as
    recalled, which measures whether the value was emitted rather than whether it was
    attributed -- and a strategy that keeps values while losing the labelling that says which
    tool returned them then scores like one that kept both.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Returns:
        One outcome per planted fact.
    """
    if scenario.answer_scopes and outcome.answers:
        return score_scoped(outcome.answers, scenario.answer_scopes, scenario.facts, outcome.final_prompt)
    return score_answer(outcome.answer, scenario.facts, outcome.final_prompt)


def score_combined(outcome: LiveOutcome, scenario: RecallScenario) -> float:
    """Return the share of planted facts the single combined answer contained.

    The per-scope questions ask for eight values each from a nearby part of the conversation.
    This one asks for all 53 at once from a context they are scattered through, which is a
    materially harder task and the one a real user is more likely to pose.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Returns:
        A fraction, or 0.0 when the scenario has no combined question.
    """
    if not outcome.combined_answer or not scenario.facts:
        return 0.0
    found = sum(1 for fact in scenario.facts if fact.appears_in(outcome.combined_answer))
    return found / len(scenario.facts)


_INSTRUCTIONS_BY_NARRATION: Final[dict[str, str]] = {
    "prompted": _INSTRUCTIONS,
    "neutral": NEUTRAL_INSTRUCTIONS,
    "suppressed": TERSE_INSTRUCTIONS,
}


def resolve_instructions(narration: str, *, retrieval_guidance: bool = True) -> str:
    """Return the agent instructions for a narration mode.

    Args:
        narration: One of the keys of :data:`_INSTRUCTIONS_BY_NARRATION`.

    Keyword Args:
        retrieval_guidance: Append :data:`RETRIEVAL_GUIDANCE`, which tells the model to quote
            every identifier it is asked for. Dropping it measures how much of the closing
            answer is the model's own willingness to enumerate rather than what compaction
            left behind.

    Returns:
        The instructions to install on the agent.

    Raises:
        KeyError: If ``narration`` is not a known mode.
    """
    base = _INSTRUCTIONS_BY_NARRATION[narration]
    if retrieval_guidance:
        return base if RETRIEVAL_GUIDANCE in base else f"{base} {RETRIEVAL_GUIDANCE}"
    return base.replace(RETRIEVAL_GUIDANCE, "").strip()
