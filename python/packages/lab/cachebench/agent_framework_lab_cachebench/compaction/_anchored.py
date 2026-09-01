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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from agent_framework import Content, Message
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

__all__ = [
    "DEFAULT_KEEP_TOKENS",
    "DEFAULT_MIN_GAIN_FRACTION",
    "MARKER_ID_PREFIX",
    "REMOVAL_MARKER",
    "AnchoredCompactionStrategy",
    "MinimumGainAnchoredCompactionStrategy",
]

#: Reason recorded on every message this strategy excludes, so a caller inspecting the
#: history can tell our removals apart from the framework's.
EXCLUDE_REASON: Final[str] = "anchored_compaction"

#: Tokens of a collapsed tool result that survive, taken from the head and the tail in equal
#: measure. Head-only retention is what the framework does; keeping both ends is a better
#: general policy for logs and documents, where the conclusion is as often at the bottom as
#: the top.
#:
#: It is *not* enough to preserve values scattered through the middle, and arithmetic says so
#: rather than taste: n values spread evenly through a result sit 1/n apart, so a head slice
#: of f/2 captures the second one only when f exceeds 2/n. With 8 values that means retaining
#: over 25% of the result to keep more than one of them. Head-and-tail retention cannot
#: preserve uniformly distributed information at any budget worth calling compaction.
DEFAULT_KEEP_TOKENS: Final[int] = 150

#: Fraction of the ceiling the collapsed middle band may occupy, shared between its tool
#: results. A fixed per-result budget cannot work: measured at a 60,000-token window a
#: 600-character retention is 0.9% of the result, and at 272,000 it is 0.3%. The strategy
#: scored 32 of 53 facts in the first case and 11 in the second -- 11 being exactly the five
#: non-tool facts plus the one code per result that fell inside the surviving head fragment.
DEFAULT_BAND_SHARE: Final[float] = 0.25

#: Refinement passes when converting a token budget into a character offset. Two is enough:
#: the first estimate uses the text's own measured ratio, so it is already close, and each
#: pass only shrinks. Bounded because the tokenizer is called on large strings.
_FIT_PASSES: Final[int] = 2

#: Marks where a tool result was cut. It serves two purposes: a model shown a truncated
#: document with no sign of truncation answers as though it had seen all of it, and the
#: marker is how a later pass recognises its own earlier work and leaves it alone.
REMOVAL_MARKER: Final[str] = "... removed by compaction"

#: Prefix of the ``message_id`` given to every note this strategy leaves behind.
MARKER_ID_PREFIX: Final[str] = "anchored_"

#: Ceiling on shed passes. Two is enough in practice; the bound exists so a mistake in the
#: termination condition cannot become an infinite loop inside a chat client.
_MAX_SHED_PASSES: Final[int] = 4

#: Share of the currently included prompt a collapse must remove before it is worth making.
#:
#: Derived rather than chosen. Prompt caching is strict-prefix, so an edit at position K makes
#: the provider re-read everything behind K once at the uncached price; the edit then saves
#: the tokens it removed on every turn that follows, at the cached price. Write ``R`` for the
#: tokens removed, ``B`` for the included tokens sitting behind the edit, ``T`` for the turns
#: still to come, and ``p`` and ``c`` for the uncached and cached prices. On the turn after
#: the edit the compacted arm pays ``(B - R) * p`` where the uncompacted arm pays ``B * c``,
#: and on each of the ``T`` turns after that it pays ``R * c`` less. So the edit repays itself
#: when ``T * R * c > (B - R) * p - B * c``, which rearranges to::
#:
#:     R > B * (p - c) / (p + T * c)
#:
#: The first term on the right of that inequality is ``(B - R) * p - B * c`` and not
#: ``(B - R) * (p - c)``: the tokens removed are not re-sent, so they are not re-read at the
#: cached price either, and writing it the other way drops an ``R * c`` and overstates the
#: floor by about 3%.
#:
#: At the measured prices -- 0.66 and 0.07 per million -- and at the cell where the anchored
#: row was measured (``B`` about 40,000 tokens behind the edit, ``T`` about 20 turns left) that
#: is ``R > 11,456`` tokens against a 52,322-token snapshot: 21.9% of the included prompt,
#: rounded up here so the floor is never *below* the break-even it is derived from.
#:
#: What the anchored row actually removed at that cell was 263 tokens,
#: 0.5%, and it cost 11% more than not compacting at all -- 46,471 tokens re-read at full
#: price to save 263, which is 177 to 1 against.
#:
#: ``T`` is the term nobody knows at decision time, and it divides: ten remaining turns need
#: 35% and forty need 13%. This default is the twenty-turn figure, so a caller who expects
#: shorter conversations should raise it rather than trust it.
DEFAULT_MIN_GAIN_FRACTION: Final[float] = 0.23


def _is_marker(message: Message) -> bool:
    """Return whether ``message`` is a note this strategy left in place of a dropped group."""
    return bool(message.message_id and message.message_id.startswith(MARKER_ID_PREFIX))


@dataclass(frozen=True, slots=True)
class _Shortening:
    """One tool result a collapse would rewrite, and what rewriting it would save.

    ``saved_tokens`` is measured on the result text rather than on the serialized message, so
    it omits the few tokens of JSON envelope that the rewrite does not change. The two agree
    to within a token per result, and the text is what the reduction is actually made of.
    """

    content: Content
    text: str
    saved_tokens: int


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
        keep_tokens: Tokens of a collapsed tool result to retain, split between its head and
            its tail. ``None`` derives it from ``band_share``, which is what makes the
            retention scale with the window instead of shrinking to nothing as results grow.
            Counted with the tokenizer rather than converted from characters: a fixed
            characters-per-token guess was wrong by a factor of two on this workload, which
            both wasted budget and made the reported retention wrong.
        band_share: Fraction of ``max_input_tokens`` the whole collapsed band may occupy,
            divided evenly between the tool results in it. Raising it keeps more of each
            result and saves less.
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
        keep_tokens: int | None = None,
        band_share: float = DEFAULT_BAND_SHARE,
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
        if keep_tokens is not None and keep_tokens < 0:
            raise ValueError("keep_tokens must be >= 0.")
        if not 0.0 < band_share <= 1.0:
            raise ValueError("band_share must be in (0.0, 1.0].")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.keep_head_groups = keep_head_groups
        self.keep_tail_groups = keep_tail_groups
        self.keep_tokens = keep_tokens
        self.band_share = band_share
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
        return self._apply_shortenings(self._plan_shortenings(messages, band))

    def _plan_shortenings(self, messages: list[Message], band: list[dict[str, Any]]) -> list[_Shortening]:
        """Return the rewrites a collapse would make, without making any of them.

        Split out from applying them so a subclass can price a collapse before it happens.
        Nothing here writes: ``_shorten`` is a pure function of the text and the budget, so
        the plan *is* what the collapse does and the two cannot drift apart. A dry run that
        instead mutated and rolled back would have to unwind ``additional_properties``
        exactly, and one flag missed there is a wrong measurement that looks like a right one.

        Args:
            messages: The message list, read but not modified.
            band: The middle groups, as returned by :meth:`_middle_band`.

        Returns:
            One entry per tool result whose text would change, in band order.
        """
        budget = self._keep_tokens_for(band)
        plan: list[_Shortening] = []
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
                    shortened = self._shorten(text, budget)
                    if shortened == text:
                        continue
                    saved = self.tokenizer.count_tokens(text) - self.tokenizer.count_tokens(shortened)
                    plan.append(_Shortening(content=content, text=shortened, saved_tokens=saved))
        return plan

    @staticmethod
    def _apply_shortenings(plan: list[_Shortening]) -> bool:
        """Write a plan out, in place.

        Args:
            plan: What :meth:`_plan_shortenings` returned.

        Returns:
            True if the plan held anything at all.
        """
        for item in plan:
            item.content.result = item.text
        return bool(plan)

    def _keep_tokens_for(self, band: list[dict[str, Any]]) -> int:
        """Return how many characters each collapsed tool result may keep.

        An explicit ``keep_tokens`` is honoured verbatim. Otherwise the band as a whole gets
        ``band_share`` of the ceiling and the tool results in it divide that evenly, so the
        retention grows with the window rather than becoming a rounding error against it.

        Args:
            band: The middle groups, as returned by :meth:`_middle_band`.

        Returns:
            A token budget per collapsed result, never below a floor that still carries a
            recognisable fragment.
        """
        if self.keep_tokens is not None:
            return self.keep_tokens
        tool_groups = sum(1 for group in band if group.get("kind") == "tool_call")
        if tool_groups == 0:
            return DEFAULT_KEEP_TOKENS
        share = self.max_input_tokens * self.band_share / tool_groups
        return max(int(share), DEFAULT_KEEP_TOKENS)

    def _shorten(self, text: str, budget: int) -> str:
        """Return ``text`` reduced to about ``budget`` tokens, taken from both ends.

        Args:
            text: The tool result to shorten.
            budget: Tokens the result may keep in total, split between its two ends.

        Returns:
            The text unchanged when it already fits, otherwise its head and tail joined by a
            marker that says how much was removed. The marker matters: a model shown a
            truncated document with no sign of truncation will answer as though it saw all
            of it.
        """
        # The marker check is what makes this idempotent. The replacement carries the marker's
        # own tokens on top of the budget, so a second pass would shorten it again and a third
        # again -- each one a fresh mutation at the same position, which is precisely the cache
        # behaviour this strategy exists to avoid.
        if REMOVAL_MARKER in text or self.tokenizer.count_tokens(text) <= budget:
            return text
        half = max(budget // 2, 1)
        head_chars = self._fit(text, half, from_end=False)
        tail_chars = self._fit(text, half, from_end=True)
        head, tail = text[:head_chars], text[len(text) - tail_chars :]
        removed = len(text) - head_chars - tail_chars
        return f"{head}\n[{REMOVAL_MARKER}: {removed:,} characters]\n{tail}"

    def _fit(self, text: str, tokens: int, *, from_end: bool) -> int:
        """Return how many characters from one end of ``text`` are worth about ``tokens``.

        The tokenizer counts but cannot slice, so the offset has to be measured. Starting from
        the text's own characters-per-token ratio lands within a few percent immediately. A
        fixed ratio does not: using 4 where the real value was 7.9 made the strategy keep half
        the budget it was entitled to, and made every retention figure reported from it wrong
        by the same factor.

        Args:
            text: The text to measure into.
            tokens: Target token count for the slice.

        Keyword Args:
            from_end: Measure a suffix rather than a prefix.

        Returns:
            A character count whose slice is at or just under ``tokens``.
        """
        total = max(self.tokenizer.count_tokens(text), 1)
        chars = min(int(len(text) * tokens / total), len(text))
        for _ in range(_FIT_PASSES):
            if chars <= 0:
                return 0
            piece = text[-chars:] if from_end else text[:chars]
            counted = self.tokenizer.count_tokens(piece)
            if counted <= tokens:
                return chars
            chars = int(chars * tokens / max(counted, 1))
        return max(chars, 0)

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


class MinimumGainAnchoredCompactionStrategy(AnchoredCompactionStrategy):
    """The anchored strategy, refusing any collapse too small to repay the cache it spends.

    Anchored compaction is cheap per edit but not free, and at one measured cell it was
    almost entirely cost. At a 60,000-token window, 0.86 fill and 3,500-token tool results,
    the ``anchored`` row removed **263 tokens** -- 0.5% of a 52,322-token snapshot -- and came
    out 11% more expensive than not compacting at all. Its prompt-cache hit rate fell from
    92% to 88%, which is 46,471 tokens re-read at the uncached price. Nothing was wrong with
    *what* it shortened. The edit was simply too small to be worth making, at 177 to 1
    against, and the strategy had no way to notice that because it never asked.

    This one asks. Before any result is rewritten it prices the whole collapse against the
    break-even in :data:`DEFAULT_MIN_GAIN_FRACTION`, and when the projected reduction falls
    under that floor it leaves the conversation exactly as it found it and counts the refusal.
    Everything else -- the anchors, the position-only band, the shed order, the markers -- is
    inherited unchanged, so a run of this row beside ``anchored`` measures the floor and
    nothing else.

    **The projection is dry, not undone.** It is :meth:`_plan_shortenings`, the same call the
    collapse itself makes, so the number the floor is compared against is the reduction the
    collapse would produce rather than an estimate of it. Compaction records its decisions by
    mutating ``additional_properties`` in place; a projection that mutated and rolled back
    would have to unwind every one of those flags, and a single missed flag is a silent wrong
    measurement rather than a failure.

    **The floor does not apply when the prompt will not fit.** Over the ceiling, shortening is
    not an optimisation whose saving has to beat a cache cost -- it is what keeps the
    conversation admissible at all, and declining it would only push the work onto the shed
    step, which drops whole groups instead of trimming them. So the floor governs the case
    the measurement was about, a prompt that already fits and is being tidied, and the
    last-resort shedding behind it is untouched.

    Keyword Args:
        min_gain_fraction: Share of the currently included prompt a collapse must be
            projected to remove before it is allowed to happen. Zero disables the floor,
            which makes this row identical to ``anchored``. See
            :data:`DEFAULT_MIN_GAIN_FRACTION` for where the default comes from and for the
            one term in it -- the turns remaining -- that no strategy can know.

    See :class:`AnchoredCompactionStrategy` for every other parameter.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        keep_head_groups: int = 3,
        keep_tail_groups: int = 4,
        keep_tokens: int | None = None,
        band_share: float = DEFAULT_BAND_SHARE,
        collapse_assistant_text: bool = True,
        min_gain_fraction: float = DEFAULT_MIN_GAIN_FRACTION,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If ``min_gain_fraction`` is negative or at least 1.0 -- a floor of one
                whole prompt can never be met, so the strategy would silently never act -- or
                if any bound the anchored strategy validates is out of range.
        """
        super().__init__(
            max_input_tokens=max_input_tokens,
            tokenizer=tokenizer,
            keep_head_groups=keep_head_groups,
            keep_tail_groups=keep_tail_groups,
            keep_tokens=keep_tokens,
            band_share=band_share,
            collapse_assistant_text=collapse_assistant_text,
        )
        if not 0.0 <= min_gain_fraction < 1.0:
            raise ValueError("min_gain_fraction must be in [0.0, 1.0).")
        self.min_gain_fraction = min_gain_fraction
        self._declined = 0

    @property
    def declined_collapses(self) -> int:
        """Passes that had a collapse available and refused it as too small to pay for itself.

        Read by the runner and surfaced in the table's flags column, because the two rows this
        strategy can produce are otherwise indistinguishable there: a run with nothing to
        compact and a run that decided compacting was not worth it both report no reduction,
        and they are opposite findings. A non-zero count says the floor is what is being
        measured; a zero count on a row that also removed nothing says the conversation never
        gave it anything to remove.
        """
        return self._declined

    def _collapse_tool_results(self, messages: list[Message], band: list[dict[str, Any]]) -> bool:
        """Collapse the band's tool results, unless doing so would not pay for itself.

        Relies on the token annotations :meth:`AnchoredCompactionStrategy.__call__` refreshes
        before it calls this, which is the only caller.

        Args:
            messages: The message list, mutated in place only if the collapse goes ahead.
            band: The middle groups, as returned by :meth:`_middle_band`.

        Returns:
            True if any result was shortened.
        """
        plan = self._plan_shortenings(messages, band)
        if not plan:
            return False
        included = included_token_count(messages)
        # Over the ceiling the collapse is not being judged on its saving: it is the cheapest
        # way left to make the conversation fit, and refusing it here would hand the work to
        # the shed step, which removes whole groups rather than trimming them.
        if included > self.max_input_tokens:
            return self._apply_shortenings(plan)
        if sum(item.saved_tokens for item in plan) < int(included * self.min_gain_fraction):
            self._declined += 1
            return False
        return self._apply_shortenings(plan)
