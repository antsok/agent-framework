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

**Two phases, using the agent's own tool loop.**

1. Over the trigger and with no record yet, the strategy appends an instruction to the
   outgoing prompt asking the model to call a recall tool with every identifier it has seen.
   The instruction is appended, never inserted, so the cached prefix is untouched.
2. The model makes that call, the agent executes it, and the result is persisted through the
   ordinary path. On a later pass the strategy finds the *real* tool result in the loaded
   history and drops every tool group in front of it.

The working copy the strategy mutates is discarded after each call, so phase 1 cannot leave
itself a note -- but it does not need to. What persists is the tool call the model made, which
the agent stored like any other.

**Why not synthesise the tool call directly.** A ``call_id`` invented by the client is only
safe when the client owns the conversation. Responses-API routes with ``store=True`` track
tool calls server-side, so a fabricated call is unknown to the service or mismatched against
it, and Gemini's thought signatures behave similarly. Letting the provider issue the call
avoids the problem entirely, at the price of one extra agent turn.

**Why a tool result rather than an assistant message.** A tool result is data. Assistant prose
is the first thing a size-pressed strategy sheds -- this package's own anchored strategy sheds
it as a last resort -- so a record written as narration would be eligible for exactly the step
that destroys it.

**What this costs to measure.** The model has to *choose* to call the recall tool, so a run
using this strategy cannot pin ``tool_choice``. Pinning is what makes rows comparable, so a
matrix containing this strategy must be unpinned throughout and is comparable within itself
only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from agent_framework import Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    annotate_message_groups,
    annotate_token_counts,
    group_messages,
    included_token_count,
    set_excluded,
)

if TYPE_CHECKING:
    from agent_framework import TokenizerProtocol

__all__ = [
    "RECALL_INSTRUCTION",
    "RECALL_TOOL_NAME",
    "ToolResultAnchoredSummarizationCompactionStrategy",
]

#: Name of the tool the agent must call. The strategy looks for this name in the history, so
#: the tool the caller registers has to match it.
RECALL_TOOL_NAME: Final[str] = "recall_earlier_tool_results"

#: What phase 1 appends. Deliberately narrow: "summarise" invites prose, and prose is where
#: identifiers go to die. It asks for verbatim values and forbids inventing one, because a
#: fabricated identifier would score as preserved information while being made up.
RECALL_INSTRUCTION: Final[str] = (
    f"Before you continue, call the {RECALL_TOOL_NAME} tool exactly once. Pass it every "
    "identifier, code, reference and concrete value that appeared in the tool results earlier "
    "in this conversation, verbatim and grouped under the tool that returned it. Do not "
    "paraphrase a value, do not invent one, and do not leave one out because it looks "
    "unimportant. Those results are about to be removed from the conversation to save space, "
    "and what you pass to this tool is the only copy that will remain."
)

#: How many consecutive passes may ask for the record before the strategy gives up. A model
#: that will not call the tool would otherwise have the instruction appended to every prompt
#: for the rest of the run, growing it rather than shrinking it.
MAX_REQUESTS: Final[int] = 3


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
        trigger_fraction: float = 0.6,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if keep_head_groups < 0:
            raise ValueError("keep_head_groups must be >= 0.")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.trigger_fraction = trigger_fraction
        self._requests = 0
        self._records = 0

    @property
    def requests_made(self) -> int:
        """Passes that asked the model to record. Compare with ``records_found``."""
        return self._requests

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
        if included_token_count(messages) <= int(self.max_input_tokens * self.trigger_fraction):
            return False

        anchor = self._anchor_index(messages)
        if anchor is None:
            return self._request_record(messages)

        self._records = max(self._records, 1)
        return self._drop_before(messages, anchor)

    def _anchor_index(self, messages: list[Message]) -> int | None:
        """Return the index of the newest recall tool result, if the model has made one.

        The call and its result are matched by ``call_id`` rather than by adjacency, because
        a provider is free to order or batch them differently.

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
                if content.type == "function_result" and content.call_id in recall_ids:
                    newest = index
        return newest

    def _request_record(self, messages: list[Message]) -> bool:
        """Append the instruction that makes the model produce a record.

        Appended rather than inserted: adding to the end leaves the cached prefix intact,
        while inserting anywhere earlier would re-bill everything after the insertion point.

        Returns:
            True if an instruction was added.
        """
        if self._requests >= MAX_REQUESTS:
            return False
        self._requests += 1
        messages.append(Message(role="user", contents=[RECALL_INSTRUCTION]))
        return True

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
