# Copyright (c) Microsoft. All rights reserved.

"""A compaction strategy designed against what the benchmark measured.

Every strategy in the framework was measured costing more than not compacting at all, and
the runs say why in enough detail to design against it. Three constraints fell out:

**1. The decision must not depend on the current size.** Prompt caching is strict-prefix: a
mutation at position K re-bills everything after it. A rule like "when over 80% of the
budget, compact down to 50%" re-decides the whole history every time it trips, so the same
old group is rewritten differently at turn 12 and turn 15 and the cache is lost from the
head each time. Measured: ``truncation`` and ``context_window`` hold 60-83% hit rates where
the uncompacted control holds 93%. A rule that depends only on a group's *position* produces
byte-identical output for the same prefix on every later turn, so the prefix stays cached.

**2. Mutations must march forward, never backward.** ``SlidingWindowStrategy`` drops the
oldest group each turn, which changes the *start* of the prompt every time; it measured a
1-9% hit rate, the worst of anything tested. Collapsing from the front once and leaving it
collapsed means each turn only invalidates the small suffix that newly aged out.

**3. What gets shed matters more than how much.** ``truncation`` left 29 of 53 planted facts
in the prompt and the model used none of them, because the codes survived while the turns
saying which deployment each belonged to were deleted. Facts near the start of a conversation
-- requirements, corrections, the actual task -- are cheap to keep and expensive to lose.

The strategy that follows keeps a fixed head and a fixed tail verbatim and collapses the band
between them by a position-only rule, shedding in a defensible order: tool results first,
then the tool call requests that produced them, and assistant narration only as a last
resort. It never touches user turns.

**What it does not do is invent information.** Any strategy that reduces size destroys what
it removes. This one does not preserve every fact and does not claim to; it preserves the
head, guarantees a ceiling that token-blind strategies cannot, and pays as little cache as
the mechanism allows.
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

__all__ = ["DEFAULT_KEEP_CHARS", "MARKER_ID_PREFIX", "REMOVAL_MARKER", "AnchoredCompactionStrategy"]

#: Reason recorded on every message this strategy excludes, so a caller inspecting the
#: history can tell our removals apart from the framework's.
EXCLUDE_REASON: Final[str] = "anchored_compaction"

#: Characters of a collapsed tool result that survive, taken from the head and the tail in
#: equal measure. Head-only retention is what the framework does; keeping both ends is a
#: better general policy for logs and documents, where the conclusion is as often at the
#: bottom as the top. It is *not* enough to preserve values scattered through the middle, and
#: no retention budget of this size would be.
DEFAULT_KEEP_CHARS: Final[int] = 600

#: Marks where a tool result was cut. It serves two purposes: a model shown a truncated
#: document with no sign of truncation answers as though it had seen all of it, and the
#: marker is how a later pass recognises its own earlier work and leaves it alone.
REMOVAL_MARKER: Final[str] = "... removed by compaction"

#: Prefix of the ``message_id`` given to every note this strategy leaves behind.
MARKER_ID_PREFIX: Final[str] = "anchored_"

#: Ceiling on shed passes. Two is enough in practice; the bound exists so a mistake in the
#: termination condition cannot become an infinite loop inside a chat client.
_MAX_SHED_PASSES: Final[int] = 4


def _is_marker(message: Message) -> bool:
    """Return whether ``message`` is a note this strategy left in place of a dropped group."""
    return bool(message.message_id and message.message_id.startswith(MARKER_ID_PREFIX))


class AnchoredCompactionStrategy:
    """Collapse the middle of a conversation by a rule that never revisits its own decisions.

    Args:
        max_input_tokens: Ceiling the included prompt must stay under. This is the model's
            real input limit, not its advertised context window: on GPT-5-class deployments
            those differ by 128,000 tokens and configuring the larger one puts every
            threshold above what the service will accept.
        tokenizer: Token counter, shared with whatever measures the result.

    Keyword Args:
        keep_head_groups: Message groups at the start that are never touched. These carry the
            task and its requirements, which every deleting strategy measured so far throws
            away first and which are the cheapest facts in the conversation to keep.
        keep_tail_groups: Recent groups kept verbatim. The working set: too small and the
            model loses the thread of what it is doing, too large and each new turn shifts a
            large block and re-bills it.
        keep_chars: Characters of a collapsed tool result to retain, split between its head
            and its tail.
        collapse_assistant_text: Allow assistant narration in the middle band to be dropped
            when tool shedding is not enough. Last resort, because narration is often where
            a tool's values ended up after the model restated them.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        keep_head_groups: int = 3,
        keep_tail_groups: int = 4,
        keep_chars: int = DEFAULT_KEEP_CHARS,
        collapse_assistant_text: bool = True,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If any bound is negative or the ceiling is not positive.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if keep_head_groups < 0 or keep_tail_groups < 0:
            raise ValueError("keep_head_groups and keep_tail_groups must be >= 0.")
        if keep_chars < 0:
            raise ValueError("keep_chars must be >= 0.")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.keep_tail_groups = keep_tail_groups
        self.keep_chars = keep_chars
        self.collapse_assistant_text = collapse_assistant_text

    async def __call__(self, messages: list[Message]) -> bool:
        """Compact in place and report whether anything changed.

        Returns:
            True if any message was excluded or replaced.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)

        groups = group_messages(messages)
        band = self._middle_band(messages, groups)
        if not band:
            return False

        # Shortening results first is what keeps the conversation legible: the model still
        # sees that each call happened and roughly what it returned. Only when that is not
        # enough does anything get removed outright.
        changed = self._collapse_tool_results(messages, band)
        if changed:
            # Token counts are cached per message inside the group annotations, so shortening
            # a result in place leaves the cached number describing text that no longer
            # exists. Without this the ceiling check below reads the original sizes and sheds
            # groups that were already small enough.
            annotate_token_counts(messages, tokenizer=self.tokenizer, force_retokenize=True)
        # Shedding is repeated because each pass adds notes of its own, which can leave the
        # prompt fractionally over the ceiling it just tried to meet. Bounded: every pass
        # removes at least one group, and there are finitely many.
        for _ in range(_MAX_SHED_PASSES):
            if included_token_count(messages) <= self.max_input_tokens:
                break
            shed = self._shed(messages, "tool_call", "[compacted: an earlier tool call and its result]")
            if self.collapse_assistant_text and included_token_count(messages) > self.max_input_tokens:
                shed = self._shed(messages, "assistant_text", "[compacted: an earlier assistant reply]") or shed
            changed = shed or changed
            if not shed:
                break
        return changed

    def _middle_band(self, messages: list[Message], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the groups between the two anchors.

        The band is decided from group *positions* alone. Nothing about the current token
        count enters, which is what makes a group's fate identical on every later turn and
        so keeps the prefix byte-identical for the cache.

        Returns:
            The middle groups, oldest first. Empty when the conversation is still short
            enough that the anchors cover all of it.
        """
        # Notes left by an earlier pass are skipped before the anchors are counted. They sit
        # where a real group used to, so counting them would shift the head and tail by one
        # per note and hand a different band back on every pass -- which would make the
        # strategy's own output change what it does next, the exact instability it exists to
        # remove.
        real = [group for group in groups if not self._is_marker_group(messages, group)]
        tail_start = len(real) - self.keep_tail_groups
        if tail_start <= self.keep_head_groups:
            return []
        return real[self.keep_head_groups : tail_start]

    @staticmethod
    def _is_marker_group(messages: list[Message], group: dict[str, Any]) -> bool:
        """Return whether every message in ``group`` is a note this strategy inserted."""
        members = messages[group["start_index"] : group["end_index"] + 1]
        return bool(members) and all(_is_marker(message) for message in members)

    def _collapse_tool_results(self, messages: list[Message], band: list[dict[str, Any]]) -> bool:
        """Shrink every tool result in the band, in place, keeping both of its ends.

        Rewrites content rather than excluding the message, so the tool-call structure stays
        intact: the model still sees that a call happened and what it returned a little of,
        which reads far better than a hole where a result used to be.

        Returns:
            True if any result was shortened.
        """
        changed = False
        for group in band:
            if group.get("kind") != "tool_call":
                continue
            for message in messages[group["start_index"] : group["end_index"] + 1]:
                if message.additional_properties.get(EXCLUDED_KEY, False):
                    continue
                for content in message.contents:
                    if content.type != "function_result":
                        continue
                    text = content.result if isinstance(content.result, str) else str(content.result)
                    shortened = self._shorten(text)
                    if shortened != text:
                        content.result = shortened
                        changed = True
        return changed

    def _shorten(self, text: str) -> str:
        """Return ``text`` reduced to ``keep_chars``, taken from both ends.

        Returns:
            The text unchanged when it already fits, otherwise its head and tail joined by a
            marker that says how much was removed. The marker matters: a model shown a
            truncated document with no sign of truncation will answer as though it saw all
            of it.
        """
        # The marker check is what makes this idempotent. The replacement is longer than
        # keep_chars by the width of the marker itself, so a second pass would shorten it
        # again and a third again -- each one a fresh mutation at the same position, which is
        # precisely the cache behaviour this strategy exists to avoid.
        if len(text) <= self.keep_chars or REMOVAL_MARKER in text:
            return text
        half = self.keep_chars // 2
        removed = len(text) - 2 * half
        return f"{text[:half]}\n[{REMOVAL_MARKER}: {removed:,} characters]\n{text[-half:]}"

    def _shed(self, messages: list[Message], kind: str, note: str) -> bool:
        """Exclude whole groups of one kind from the band, oldest first, until the ceiling is met.

        Exclusion and insertion are separated on purpose. Excluding leaves every index in the
        band valid, so the selection loop can re-measure after each step; the notes are then
        inserted in reverse index order, where each insertion cannot disturb the position of
        one still to come. Doing both in one forward pass silently shifts every later group by
        the number of notes already inserted.

        Args:
            messages: The message list, mutated in place.
            kind: Group kind to shed, one of ``"tool_call"`` or ``"assistant_text"``.
            note: Text left in place of each dropped group.

        Returns:
            True if any group was dropped.
        """
        band = self._middle_band(messages, group_messages(messages))
        dropped: list[dict[str, Any]] = []
        for group in band:
            if group.get("kind") != kind:
                continue
            if included_token_count(messages) <= self.max_input_tokens:
                break
            members = messages[group["start_index"] : group["end_index"] + 1]
            if all(message.additional_properties.get(EXCLUDED_KEY, False) for message in members):
                continue
            # A note left by an earlier phase is an assistant message, so the assistant-text
            # phase would otherwise shed the very markers the tool phase just inserted --
            # leaving a silent hole instead of a stated one, and undoing the only signal the
            # model has that something was removed.
            if all(_is_marker(message) for message in members):
                continue
            for message in members:
                set_excluded(message, excluded=True, reason=EXCLUDE_REASON)
            dropped.append(group)

        for group in reversed(dropped):
            self._insert_note(messages, group, note)
        if dropped:
            # Each note is itself a message with a cost. Counting it only on the next call
            # would let this one stop just above its ceiling and look as though it had met
            # it -- and on the following turn the strategy would shed one more group,
            # changing a decision it had already made.
            annotate_token_counts(messages, tokenizer=self.tokenizer)
        return bool(dropped)

    def _insert_note(self, messages: list[Message], group: dict[str, Any], note: str) -> None:
        """Leave one deterministic marker where a dropped group used to be.

        The ``message_id`` is derived from the group, not from a counter or a timestamp, so
        the replacement is byte-identical on every later turn. A marker that varied would be
        a cache mutation in its own right, which is the failure this whole strategy is built
        to avoid. The framework's own summary insertion uses the same convention.
        """
        marker_id = f"{MARKER_ID_PREFIX}{group['group_id']}"
        if any(message.message_id == marker_id for message in messages):
            return
        members = messages[group["start_index"] : group["end_index"] + 1]
        messages.insert(
            group["start_index"],
            Message(
                role="assistant",
                contents=[note],
                message_id=marker_id,
                additional_properties={
                    GROUP_ANNOTATION_KEY: {
                        SUMMARY_OF_MESSAGE_IDS_KEY: [m.message_id for m in members if m.message_id],
                        SUMMARY_OF_GROUP_IDS_KEY: [group["group_id"]],
                    }
                },
            ),
        )
