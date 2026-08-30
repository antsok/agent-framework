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
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Final

from agent_framework import ChatContext, ChatMiddleware, Message
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
    "RECALL_TOOL_NAME",
    "RECORD_MARKER",
    "ToolResultAnchoredSummarizationCompactionStrategy",
    "ToolResultRecallMiddleware",
    "find_record_index",
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
        paused: Consulted before each call; while it returns True nothing is forced. The
            runner uses it to hold the conversation still for the closing questions, where a
            fresh record would change the history the answers are being scored against.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        arm: Callable[[], None],
        trigger_fraction: float = 0.6,
        paused: Callable[[], bool] | None = None,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.arm = arm
        self.trigger_fraction = trigger_fraction
        self._paused = paused
        self._force_next = False
        self._forced = 0
        self._records_forced = 0
        self._records_volunteered = 0
        self._seen_record = False

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
        if self._paused is not None and self._paused():
            # Nothing is forced and nothing is re-armed while paused, including the pending
            # decision from the last call: a record arriving now would replace tool results
            # part-way through the questions that are scoring them.
            self._force_next = False
            await call_next()
            return

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
            context.options = {
                **dict(context.options or {}),
                "tool_choice": {"mode": "required", "required_function_name": RECALL_TOOL_NAME},
            }
            self._force_next = False
            self._forced += 1

        await call_next()

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
