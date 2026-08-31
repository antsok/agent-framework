# Copyright (c) Microsoft. All rights reserved.

"""Have the agent record the facts, then drop everything behind the record.

Every truncating strategy measured here fails the same way, and the failure is arithmetic
rather than tuning. Head-and-tail retention keeps the first and last f/2 of a result, so ``n``
values spread evenly through it -- sitting 1/n apart -- survive only when f exceeds 2/n. With
eight values per result that needs more than 25% of it retained, which is not compaction.
Measured: at 30% retention 3 of 8 codes survived, at 0.9% just 1.

Truncation preserves *positions*. What is needed is something that preserves *information*,
after which the bulk it came from is genuinely redundant and can be dropped outright rather
than sampled.

**Four parts, and all four are the design rather than harness around it.** The strategy
cannot work without a tool for the model to call (:func:`make_recall_tool`) or without
something keeping that tool inert when nobody asked for a record (:class:`RecallGate`), so
both live here beside the middleware that arms them.

**Two phases, split across a middleware and this strategy.**

1. :class:`ToolResultRecallMiddleware` forces ``tool_choice`` to the recall tool on one call.
   It sends no message at all: the tool's own description already says what to pass, and the
   schema travels on every request anyway. Nothing is added to the prompt, and nothing extra
   reaches the caller's stored history.
2. The model makes the call, the agent executes it, and the result is persisted through the
   ordinary path. On a later pass this strategy finds that real tool result in the loaded
   history and drops every tool group in front of it.

Sending no message matters for more than tokens. A message appended here carries no history
provider's source tag, so the per-service-call persistence would treat it as new input and
store it -- and an instruction of ours would show up in the conversation the application
replays to its user. Forcing the option leaves no such trace. The tool call and its result do
appear, which is correct: they are a real record of what the agent did.

**Why not synthesise the tool call directly.** A ``call_id`` invented by the client is only
safe when the client owns the conversation. Responses-API routes with ``store=True`` track
tool calls server-side, so a fabricated call is unknown to the service or mismatched against
it, and Gemini's thought signatures behave similarly. Letting the provider issue the call
avoids the problem entirely, at the price of one extra agent turn.

**Why a tool result rather than an assistant message.** A tool result is data. Assistant prose
is the first thing a size-pressed strategy sheds -- this package's own anchored strategy sheds
it as a last resort -- so a record written as narration would be eligible for exactly the step
that destroys it.

**Why forcing beats asking.** Asking required the model to choose, which meant the benchmark
could not pin ``tool_choice`` -- and unpinned, the uncompacted control's cost varied by 102%
between identical runs while the strategy's row gathered eight fewer facts than the control.
Forcing the call keeps every other turn pinned, so the comparison stays measurable, and makes
phase 1 deterministic rather than a compliance rate to be estimated.

**How the record is bounded, and why it takes two numbers rather than one.** The instructions
ask for everything, so the record wants to grow. A ``max_tokens`` cap does not answer that: a
model does not plan to fit a cap, it writes until it is cut, and on a *tool call* the cut
lands inside the arguments JSON, so a cap set where the record should end produces no record
instead of a shorter one. The two bounds therefore do different jobs.
:data:`DEFAULT_RECORD_TARGET_TOKENS` is stated in the tool's own description, which is the
only channel that reaches the model before it writes, since the middleware sends no message.
:data:`DEFAULT_RECORD_MAX_TOKENS` is set on the forced call alone and is roughly twice the
target, so it bounds the bill without ever being the thing that stops the writing. When it is
the thing that stops it, ``ToolResultRecallMiddleware.records_truncated`` says so.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Final

from agent_framework import ChatContext, ChatMiddleware, ChatResponse, Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    annotate_message_groups,
    annotate_token_counts,
    group_messages,
    included_token_count,
    set_excluded,
)

from ._anchored import AnchoredCompactionStrategy

if TYPE_CHECKING:
    from agent_framework import CompactionStrategy, TokenizerProtocol

__all__ = [
    "DEFAULT_RECORD_MAX_TOKENS",
    "DEFAULT_RECORD_TARGET_TOKENS",
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "RecallGate",
    "ToolResultAnchoredSummarizationCompactionStrategy",
    "ToolResultRecallMiddleware",
    "find_record_index",
    "make_recall_tool",
]

#: Name of the tool the agent must call. The strategy looks for this name in the history, so
#: the tool the caller registers has to match it.
RECALL_TOOL_NAME: Final[str] = "recall_earlier_tool_results"

#: Prefix the recall tool puts on a result that counts as a record.
#:
#: The tool cannot be hidden from the model. Tools passed through per-call options reach the
#: model but not the executor -- ``FunctionInvocationLayer`` wraps ``ChatMiddlewareLayer``, so
#: it has already built its tool map by the time a middleware could add one, and the model's
#: call goes unanswered. A registered tool is therefore advertised on every request, and this
#: one was called unprompted on the unpinned follow-up call in every run.
#:
#: So the tool stays visible and becomes *inert* instead: it records only while the middleware
#: has armed it, and a result without this marker is not a record. The model may still call
#: it; calling it uninvited simply achieves nothing.
RECORD_MARKER: Final[str] = "[recorded by compaction]"

#: Hard ceiling put on the forced call's response, so a runaway record cannot cost more than
#: intended.
#:
#: A cap is not a plan. The model does not shorten to fit one; it writes until it is cut, and
#: on a *tool call* the cut lands inside the arguments JSON, so what a too-low cap produces is
#: not a shorter record but no record at all. That makes this the wrong instrument for sizing
#: the record and the right one for bounding the bill, and the two numbers here are set
#: accordingly: this is roughly twice :data:`DEFAULT_RECORD_TARGET_TOKENS`, so a model that
#: overshoots its stated target still finishes inside the cap.
DEFAULT_RECORD_MAX_TOKENS: Final[int] = 4_000

#: Record length stated in the tool's own description, which is the only channel that makes
#: the model aim for a size.
#:
#: The middleware deliberately sends no message -- an appended instruction would be persisted
#: into the caller's own conversation -- so the description and the ``values`` parameter are
#: the entire prompt, and a target has to be baked into them at construction.
DEFAULT_RECORD_TARGET_TOKENS: Final[int] = 2_000

#: What the recall tool is for, as the model reads it.
#:
#: Deliberately not "identifiers and values": that phrasing was fitted to one benchmark's hex
#: codes and would drop prose, findings and conclusions from any real tool output, which is
#: most of what a real tool returns.
RECALL_DESCRIPTION: Final[str] = (
    "Record what must survive from earlier tool results, so it remains available after those "
    "results are removed from the conversation to save space."
)

#: What to put in the ``values`` argument, as the model reads it.
#:
#: The four instructions after the opening sentence partition the content, so that nothing in
#: a tool result falls outside all of them: values that cannot be reconstructed are quoted,
#: findings and conclusions are kept as stated, a summary the tool already wrote is carried
#: over rather than rewritten, and whatever remains is summarised. Removing one of the four
#: opens a gap that the model is then free to drop silently, which is the failure this whole
#: strategy exists to avoid.
RECALL_VALUES_DESCRIPTION: Final[str] = (
    "Everything from earlier tool results, grouped by the tool that produced it, so that "
    "nothing is lost without being noticed. Quote verbatim any value that cannot be "
    "reconstructed or guessed: identifiers, codes, names, numbers, paths, URLs, versions, "
    "states, timestamps. Keep findings and conclusions as they were stated, shortening only "
    "those long enough to need it. Carry over any summary a tool already produced as it "
    "stands, rather than rewriting it. Summarise the remaining content briefly, so its "
    "substance is still represented. Record only what the results actually contained, and "
    "where you must choose, keep exactness over brevity."
)


def find_record_index(messages: Sequence[Message]) -> int | None:
    """Return the index of the newest recall tool result, if one exists.

    Shared by the strategy and the middleware so the two halves cannot disagree about whether
    a record exists -- otherwise one would force a call that was already made, or drop results
    a record never covered.

    The call and its result are matched by ``call_id`` rather than by adjacency, because a
    provider is free to order or batch them differently. A result whose call is absent does not
    count: that is the shape a client-synthesised pair produces, and exactly what breaks on
    routes that track tool calls server-side.

    Args:
        messages: The conversation to search.

    Returns:
        The index of the result message, or None when no complete record exists.
    """
    recall_ids = {
        content.call_id
        for message in messages
        for content in message.contents
        if content.type == "function_call" and content.name == RECALL_TOOL_NAME and content.call_id
    }
    if not recall_ids:
        return None
    newest: int | None = None
    for index, message in enumerate(messages):
        for content in message.contents:
            if content.type != "function_result" or content.call_id not in recall_ids:
                continue
            result = content.result if isinstance(content.result, str) else str(content.result)
            # The marker is what separates a record from a call the model made on its own
            # initiative. Without it an uninvited call would look like a record and the
            # strategy would drop results nothing had preserved.
            if RECORD_MARKER in result:
                newest = index
    return newest


class RecallGate:
    """One-shot permission for the recall tool.

    The tool cannot be hidden from the model. It has to be registered with the agent for the
    function-invocation layer to execute it, and that layer wraps the middleware layer, so a
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


def make_recall_tool(
    gate: RecallGate | None = None,
    *,
    target_tokens: int | None = DEFAULT_RECORD_TARGET_TOKENS,
) -> Callable[[str], str]:
    """Build the tool :class:`ToolResultAnchoredSummarizationCompactionStrategy` anchors on.

    It echoes what it is given straight back. That is the whole point: the value of the call
    is not what the tool computes but that the model's own recollection ends up in the
    transcript as a tool result, which the provider issued and which survives strategies that
    shed assistant prose.

    The docstring built here is the entire prompt for the record. The middleware sends no
    message -- one appended there would carry no history provider's source tag, so per-call
    persistence would store it and an instruction of ours would surface in the conversation
    the application replays to its user -- so the description, the ``values`` guidance and the
    stated target are the only three things steering what the model writes.

    Args:
        gate: Permission to record. Without one the tool always records, which is only right
            for a caller driving it deliberately.

    Keyword Args:
        target_tokens: Length to aim the record at, stated in the description. ``None`` states
            none. This is the only bound that makes the model *plan* to fit: the middleware's
            cap merely truncates, and truncating a tool call destroys the arguments JSON
            rather than shortening the record it carries.

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
    sections = [RECALL_DESCRIPTION]
    if target_tokens is not None:
        # Spent where it is needed, rather than as a flat budget: the four clauses below are
        # not equally compressible, and a bare length would be read as licence to shorten the
        # verbatim half, which is the half that cannot be rewritten from anything else.
        sections.append(
            f"Aim for about {target_tokens:,} tokens in total, spent on what cannot be "
            "reconstructed rather than on the summaries."
        )
    sections.append(f"Args:\n    values: {RECALL_VALUES_DESCRIPTION}")
    # The whole docstring becomes the tool description: the framework builds the parameter
    # schema from the annotations and does not lift the Args section into it, so the guidance
    # for ``values`` only reaches the model as part of this text.
    tool.__doc__ = "\n\n".join(sections)
    return tool


class ToolResultAnchoredSummarizationCompactionStrategy:
    """Ask the agent to record the facts in a tool call, then drop what is behind it.

    Args:
        max_input_tokens: Ceiling the included prompt must stay under.
        tokenizer: Token counter, shared with whatever measures the result.

    Keyword Args:
        keep_head_groups: Groups at the start never touched, carrying the task and its
            requirements. Dropping these is what makes other strategies lose the labelling
            that gives surviving values their meaning.
        trigger_fraction: Fraction of the ceiling at which the record is first requested.
            Below it nothing happens: a record that is not needed costs an agent turn and
            buys nothing.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        keep_head_groups: int = 3,
        keep_tail_groups: int = 4,
        trigger_fraction: float = 0.6,
        fallback_fraction: float = 0.9,
        fallback: CompactionStrategy | None = None,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range, or the two thresholds are the wrong way
                around -- a fallback at or below the trigger would fire before the model had
                any chance to answer, and the recording step would never happen at all.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if not 0.0 < fallback_fraction <= 1.0:
            raise ValueError("fallback_fraction must be in (0.0, 1.0].")
        if fallback_fraction <= trigger_fraction:
            raise ValueError("fallback_fraction must be greater than trigger_fraction.")
        if keep_head_groups < 0 or keep_tail_groups < 0:
            raise ValueError("keep_head_groups and keep_tail_groups must be >= 0.")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.keep_tail_groups = keep_tail_groups
        self.trigger_fraction = trigger_fraction
        self.fallback_fraction = fallback_fraction
        self.fallback = fallback or AnchoredCompactionStrategy(
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
            keep_head_groups=keep_head_groups,
            keep_tail_groups=keep_tail_groups,
        )
        self._records = 0
        self._fallbacks = 0

    @property
    def fallbacks_used(self) -> int:
        """Passes that gave up waiting and truncated instead.

        Non-zero means the record arrived too late to help, or never arrived, and that row is
        measuring the fallback rather than this design. Reported rather than hidden: a
        strategy that quietly degrades into another one produces a number that belongs to
        neither.
        """
        return self._fallbacks

    @property
    def records_found(self) -> int:
        """Recall tool results seen in the history. Zero means the model never complied."""
        return self._records

    async def __call__(self, messages: list[Message]) -> bool:
        """Request a record, or drop what an existing record covers.

        Returns:
            True if the outgoing messages changed.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        used = included_token_count(messages)
        if used <= int(self.max_input_tokens * self.trigger_fraction):
            return False

        anchor = find_record_index(messages)
        if anchor is not None:
            self._records = max(self._records, 1)
            changed = self._drop_before(messages, anchor)
            # Even a good record may not be enough on its own: the groups after it are
            # untouched by design, and they can exceed the ceiling by themselves.
            if included_token_count(messages) > self.max_input_tokens:
                changed = await self.fallback(messages) or changed
            return changed

        if used < int(self.max_input_tokens * self.fallback_fraction):
            # Still waiting for the middleware's forced call to come back. Nothing may be
            # dropped yet: the record is the only thing that would replace it.
            return False

        # Out of room to keep waiting. A model that has not answered by now may never answer,
        # and the alternative to compacting without a record is a provider error. The tool
        # results are lost either way at this point; at least the conversation survives.
        self._fallbacks += 1
        return await self.fallback(messages)

    def _drop_before(self, messages: list[Message], anchor: int) -> bool:
        """Exclude every tool group that ends before the record.

        The record is what those groups have been reduced to, so they are redundant rather
        than merely old. Everything from the record onward is left alone, and so is the head,
        which carries the requirements that give the recorded values their meaning.

        Returns:
            True if anything was excluded.
        """
        groups = group_messages(messages)
        changed = False
        for position, group in enumerate(groups):
            if position < self.keep_head_groups or group.get("kind") != "tool_call":
                continue
            if group["end_index"] >= anchor:
                continue
            for message in messages[group["start_index"] : group["end_index"] + 1]:
                changed = set_excluded(message, excluded=True, reason="tool_summary_anchored") or changed
        return changed

    def _excluded(self, messages: list[Message]) -> int:
        """Return how many messages are currently excluded, for tests and diagnostics."""
        return sum(1 for message in messages if message.additional_properties.get(EXCLUDED_KEY, False))


class ToolResultRecallMiddleware(ChatMiddleware):
    """Force the recall call once, so phase 2 has something to anchor on.

    Args:
        max_input_tokens: Ceiling the prompt must stay under, matching the strategy's.
        tokenizer: Token counter, matching the strategy's.

    Keyword Args:
        arm: Called immediately before the forced request, arming the recall tool for one
            call. The tool is inert otherwise, which is what stops the model producing a
            record on its own initiative -- it cannot be hidden, only disabled.
        trigger_fraction: Fraction of the ceiling at which the record is forced. Comfortably
            below the strategy's fallback threshold, because the decision is made one call
            late -- see :meth:`process`.
        record_max_tokens: Cap put on the forced call's response, and on no other call.
            ``None`` leaves whatever cap the run already sets, which is what this did before
            the parameter existed: the record inherited the cap sized for an ordinary answer,
            so a record asked to summarise everything had no bound of its own at all.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        arm: Callable[[], None],
        trigger_fraction: float = 0.6,
        record_max_tokens: int | None = DEFAULT_RECORD_MAX_TOKENS,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if record_max_tokens is not None and record_max_tokens <= 0:
            raise ValueError("record_max_tokens must be positive, or None to leave the run's cap in place.")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.arm = arm
        self.trigger_fraction = trigger_fraction
        self.record_max_tokens = record_max_tokens
        self._force_next = False
        self._forced = 0
        self._records_forced = 0
        self._records_volunteered = 0
        self._records_truncated = 0
        self._seen_record = False

    def forget_pending(self) -> None:
        """Drop the decision to force a record on the next call.

        The decision is taken on one call and applied to the next, which makes it part of the
        conversation rather than of this object: a decision taken while the conversation was
        being seeded would fire on the first question asked of the snapshot and on none of the
        others, so that one probe would carry a prompt the rest do not. Restoring the snapshot
        has to restore this too.
        """
        self._force_next = False

    @property
    def forced_calls(self) -> int:
        """How many times the recall tool was forced. Zero means phase 1 never fired."""
        return self._forced

    @property
    def records_forced(self) -> int:
        """Records that appeared on a call this middleware pinned."""
        return self._records_forced

    @property
    def records_volunteered(self) -> int:
        """Records the model produced without being pinned.

        Not a success. A turn's pinned ``tool_choice`` applies only to its first call, so on
        the follow-up after a tool result the model may call any registered tool. A record
        arriving that way is the model choosing to, which is not something a strategy can rely
        on, and reporting it as though the design had worked would be the difference between a
        mechanism and a coincidence.
        """
        return self._records_volunteered

    @property
    def records_truncated(self) -> int:
        """Forced calls the provider cut short because they reached the cap.

        Non-zero means a record may be partial, and a partial record is the one failure mode
        this design does not otherwise show. A tool call cut mid-arguments produces no record
        at all, which is loud: the strategy waits, falls back and flags ``FALLBACK``. A call
        cut just after a closing brace, or repaired by the provider, yields a record that
        parses and looks complete -- and the strategy then drops every tool group behind
        something that covers only part of them, scoring the loss as compaction damage rather
        than as an instrument that ran out of room.
        """
        return self._records_truncated

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        """Force the recall tool when the last call showed the conversation is large enough.

        The decision is made *after* ``call_next`` and applied on the *following* call, which
        is not a convenience. Before the pipeline runs, ``context.messages`` holds only the new
        turn: the history middleware sits deeper and replaces it with the loaded conversation
        during the call. There is no earlier point at which the size of the history can be
        known, so the check reads it on the way out and the option is set on the way in next
        time. The trigger sits well below the strategy's fallback threshold to absorb that
        one-call delay.
        """
        forced_this_call = self._force_next
        if forced_this_call:
            # Replaced rather than mutated: options may be shared with the caller's own dict,
            # and pinning a tool choice into it would outlive this call.
            # The tool is offered on this call and no other. Registering it on the agent
            # would put its schema in every request, and its description reads as sensible
            # hygiene right after a lookup -- which is exactly what happened: the model called
            # it unprompted on the unpinned follow-up call, and the run then measured the
            # model's initiative rather than this middleware. Options replace the tool list
            # rather than adding to it, so offering it here also hides everything else, which
            # is harmless on a call whose only purpose is to make this one call.
            self.arm()
            options: dict[str, Any] = {
                **dict(context.options or {}),
                "tool_choice": {"mode": "required", "required_function_name": RECALL_TOOL_NAME},
            }
            if self.record_max_tokens is not None:
                # Only here. The run's own cap is sized for an answer to the user and every
                # other call needs it; this one call writes a record instructed to summarise
                # everything, which is the only place a runaway is affordable at all.
                options["max_tokens"] = self.record_max_tokens
            context.options = options
            self._force_next = False
            self._forced += 1

        await call_next()

        if forced_this_call and isinstance(context.result, ChatResponse) and context.result.finish_reason == "length":
            # Taken from the provider rather than inferred from the record's length: the model
            # is free to write a short record, and a short record is not a cut one. This is
            # the provider saying it stopped generating because it hit the ceiling, which is
            # the only statement that separates the two.
            #
            # A streamed result is not a ChatResponse yet, so it is not read here. Recording is
            # a one-shot forced call whose answer nobody displays, so there is nothing to
            # stream it for -- but a caller that did would lose this count, not get a wrong one.
            self._records_truncated += 1

        messages = list(context.messages)
        if not messages:
            self._force_next = False
            return
        if find_record_index(messages) is not None:
            # The transition is tracked on the instance, not read from the messages on the way
            # in. Before the pipeline runs, context.messages holds only the new turn, so a
            # pre-call check reports "no record" on every call and every later call counts as
            # a fresh one -- which is how an 18 appeared here for a single record.
            if not self._seen_record:
                self._seen_record = True
                # Attributed, not merely counted. A record that arrived unpinned came from the
                # model volunteering on a follow-up call, and that is a different claim.
                if forced_this_call:
                    self._records_forced += 1
                else:
                    self._records_volunteered += 1
            self._force_next = False
            return
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        self._force_next = included_token_count(messages) > int(self.max_input_tokens * self.trigger_fraction)
