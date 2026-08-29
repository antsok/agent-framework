# Copyright (c) Microsoft. All rights reserved.

"""Lift the facts out of tool results, then compact hard behind the lift.

Every truncating strategy measured here fails the same way, and the failure is arithmetic
rather than tuning. Head-and-tail retention keeps the first and last f/2 of a result, so
``n`` values spread evenly through it -- sitting 1/n apart -- survive only when f exceeds
2/n. With eight values per result that needs more than 25% of it retained, which is not
compaction. Measured: at 30% retention 3 of 8 codes survived, at 0.9% just 1.

Truncation preserves *positions*. What is needed is something that preserves *information*,
after which the bulk it came from is genuinely redundant and can be discarded outright rather
than sampled.

So: ask the model to extract the values from the tool results, keep that extract as a tool
result of its own, and drop the originals behind it.

**Why a tool result and not an assistant message.** A tool result is data. Assistant prose is
the first thing a size-pressed strategy sheds -- this package's own anchored strategy sheds it
as a last resort -- so a summary written as narration is eligible for exactly the step that
destroys it. As a result it sits in the same class as the values it replaced.

**Why the extract is computed once and then frozen.** A compaction strategy is handed a fresh
copy of the history on every model call, so a strategy that re-derives its summary each time
emits different text each time, mutating the prefix and forfeiting the cache. That is visible
in the framework's own ``SummarizationStrategy``, which measured hit rates of 0% to 59% where
the uncompacted control held 93%. Here each extract is generated once, cached on the strategy
instance, and re-emitted byte-identically forever after. Extracts accumulate rather than being
rewritten, so an extract that exists never changes again.

**What this cannot do.** It cannot ask the agent to make the tool call and wait for the reply.
A ``CompactionStrategy`` is invoked inside a single model call and returns a bool; it has no
way to suspend, let the agent take a turn, and resume. Nor can it leave itself a note in the
history, because the copy it mutates is discarded after the call. The extraction therefore
runs as the strategy's own model call, the way ``SummarizationStrategy`` runs its own, and the
synthesised call/result pair is what the *next* prompt carries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from agent_framework import Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    SUMMARY_OF_MESSAGE_IDS_KEY,
    annotate_message_groups,
    annotate_token_counts,
    group_messages,
    included_token_count,
    set_excluded,
)

if TYPE_CHECKING:
    from agent_framework import TokenizerProtocol
    from agent_framework._clients import SupportsChatGetResponse

__all__ = ["EXTRACTION_INSTRUCTION", "ToolResultAnchoredSummarizationCompactionStrategy"]

#: What the extraction call asks for. Deliberately narrow: "summarise" invites prose, and
#: prose is where identifiers go to die. It asks for verbatim values and for an explicit
#: admission when there are none, because an extract that quietly invents a plausible
#: identifier would score as preserved information while being fabricated.
EXTRACTION_INSTRUCTION: Final[str] = (
    "Below are results from earlier tool calls in a conversation. Extract every identifier, "
    "code, reference and concrete value they contain, verbatim and exactly as written, "
    "grouped under the tool that returned it. Do not paraphrase a value, do not invent one, "
    "and do not omit one because it looks unimportant. If a tool returned no such values, "
    "write that tool's name followed by 'no values'. Output only the list."
)

#: Prefix of the ``message_id`` on every message this strategy synthesises.
MARKER_ID_PREFIX: Final[str] = "tool_extract_"

#: Name the synthesised tool call carries. It appears in the prompt, so it should read as
#: what it is rather than as an internal detail.
EXTRACT_TOOL_NAME: Final[str] = "recall_earlier_tool_results"


class ToolResultAnchoredSummarizationCompactionStrategy:
    """Extract the values from old tool results, keep the extract, drop the results.

    Args:
        client: Chat client used for the extraction call. It sees the tool results, so it
            must be trusted with the same data as the agent itself.
        max_input_tokens: Ceiling the included prompt must stay under.
        tokenizer: Token counter, shared with whatever measures the result.

    Keyword Args:
        keep_head_groups: Groups at the start never touched, carrying the task and its
            requirements.
        keep_tail_groups: Recent groups kept verbatim, the working set.
        trigger_fraction: Fraction of the ceiling at which extraction first runs. Below it
            nothing happens at all: an extract that is not needed is a model call spent and a
            prefix mutated for nothing.
        extract_max_tokens: Cap on each extract, so the thing replacing the bulk cannot itself
            become bulk.
    """

    def __init__(
        self,
        *,
        client: SupportsChatGetResponse[Any],
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        keep_head_groups: int = 3,
        keep_tail_groups: int = 4,
        trigger_fraction: float = 0.8,
        extract_max_tokens: int = 2_000,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If a bound is out of range.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if keep_head_groups < 0 or keep_tail_groups < 0:
            raise ValueError("keep_head_groups and keep_tail_groups must be >= 0.")
        self.client = client
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.keep_tail_groups = keep_tail_groups
        self.trigger_fraction = trigger_fraction
        self.extract_max_tokens = extract_max_tokens
        # Append-only. Each entry is (call_id, covered group ids, extract text) and is never
        # rewritten once created, which is what keeps the prefix byte-identical across turns.
        self._extracts: list[tuple[str, tuple[str, ...], str]] = []
        self._failures = 0

    @property
    def failures(self) -> int:
        """Extraction calls that raised. A run with failures compacted less than it claims."""
        return self._failures

    async def __call__(self, messages: list[Message]) -> bool:
        """Extract, then drop what was extracted.

        Returns:
            True if any message was excluded or added.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        if included_token_count(messages) <= int(self.max_input_tokens * self.trigger_fraction):
            return False

        groups = group_messages(messages)
        band = self._band(groups)
        covered = {group_id for _, ids, _ in self._extracts for group_id in ids}
        pending = [group for group in band if group.get("kind") == "tool_call" and group["group_id"] not in covered]
        if pending and not await self._extract(messages, pending):
            return False

        return self._apply(messages, groups)

    def _band(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the groups eligible for extraction, between the two anchors.

        Returns:
            The middle groups, oldest first, excluding anything this strategy synthesised so
            that its own extracts are never fed back into a later extract.
        """
        real = [group for group in groups if not str(group.get("group_id", "")).startswith(MARKER_ID_PREFIX)]
        tail_start = len(real) - self.keep_tail_groups
        if tail_start <= self.keep_head_groups:
            return []
        return real[self.keep_head_groups : tail_start]

    async def _extract(self, messages: list[Message], pending: list[dict[str, Any]]) -> bool:
        """Run one extraction call over the pending groups and record the result.

        Returns:
            True if an extract was produced. False when the call failed, in which case
            nothing is dropped: losing the results *and* the extract would be strictly worse
            than not compacting.
        """
        payload: list[str] = []
        for group in pending:
            for message in messages[group["start_index"] : group["end_index"] + 1]:
                for content in message.contents:
                    if content.type == "function_result":
                        result = content.result if isinstance(content.result, str) else str(content.result)
                        payload.append(result)
        if not payload:
            return False

        prompt = [
            Message(role="system", contents=[EXTRACTION_INSTRUCTION]),
            Message(role="user", contents=["\n\n---\n\n".join(payload)]),
        ]
        try:
            response = await self.client.get_response(prompt, max_tokens=self.extract_max_tokens)
        except Exception:
            # Swallowed on purpose, and counted. A failed extraction must not take the tool
            # results with it, and it must not be silent either: a row that quietly stopped
            # compacting would otherwise read as a strategy that preserves information well.
            self._failures += 1
            return False

        text = str(response.text or "").strip()
        if not text:
            self._failures += 1
            return False
        call_id = f"{MARKER_ID_PREFIX}{len(self._extracts)}"
        self._extracts.append((call_id, tuple(group["group_id"] for group in pending), text))
        return True

    def _apply(self, messages: list[Message], groups: list[dict[str, Any]]) -> bool:
        """Exclude every extracted group and put the extracts in their place.

        Returns:
            True if the message list changed.
        """
        by_id = {group["group_id"]: group for group in groups}
        changed = False
        insert_at: dict[str, int] = {}
        for call_id, covered, _ in self._extracts:
            positions = [by_id[group_id]["start_index"] for group_id in covered if group_id in by_id]
            if positions:
                insert_at[call_id] = min(positions)
            for group_id in covered:
                group = by_id.get(group_id)
                if group is None:
                    continue
                for message in messages[group["start_index"] : group["end_index"] + 1]:
                    changed = set_excluded(message, excluded=True, reason="tool_extract") or changed

        # Inserted last and in reverse position order, so no insertion moves a position still
        # to be used. The same ordering bug cost this package a debugging round already.
        for call_id, covered, text in sorted(self._extracts, key=lambda item: -insert_at.get(item[0], 0)):
            if call_id not in insert_at or any(m.message_id == f"{call_id}_result" for m in messages):
                continue
            messages[insert_at[call_id] : insert_at[call_id]] = self._pair(call_id, covered, text)
            changed = True
        return changed

    def _pair(self, call_id: str, covered: tuple[str, ...], text: str) -> list[Message]:
        """Return the synthesised call/result pair carrying one extract.

        A bare tool result with no matching call is malformed for most providers, so the
        request is synthesised alongside it. Both ids are derived from the extract's index
        rather than from a counter or a clock, so the pair is byte-identical on every later
        turn.

        Returns:
            The assistant request and the tool result, in order.
        """
        annotation = {SUMMARY_OF_GROUP_IDS_KEY: list(covered), SUMMARY_OF_MESSAGE_IDS_KEY: []}
        return [
            Message(
                role="assistant",
                contents=[
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": EXTRACT_TOOL_NAME,
                        "arguments": "{}",
                    }
                ],
                message_id=f"{call_id}_call",
                additional_properties={GROUP_ANNOTATION_KEY: annotation},
            ),
            Message(
                role="tool",
                contents=[
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "result": (
                            "Values recovered from earlier tool results, which have since been "
                            f"removed from this conversation to save space:\n{text}"
                        ),
                    }
                ],
                message_id=f"{call_id}_result",
                additional_properties={GROUP_ANNOTATION_KEY: annotation},
            ),
        ]

    def _excluded(self, messages: list[Message]) -> int:
        """Return how many messages are currently excluded, for tests and diagnostics."""
        return sum(1 for message in messages if message.additional_properties.get(EXCLUDED_KEY, False))
