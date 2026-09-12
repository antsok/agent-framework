# Copyright (c) Microsoft. All rights reserved.

"""Summarise the user's own turns, and re-summarise the summary when it fills up again.

**The half of the conversation nothing here was touching.** Every other strategy in this
subpackage sheds tool output: :class:`~._anchored.AnchoredCompactionStrategy` shortens tool
results, :class:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy` replaces
whole tool groups with a record of them, and both of them keep user turns verbatim on the
stated ground that the turns are what give surviving values their meaning. That is true of the
*first* user turn, which carries the task and the requirements, and it is true of the *last*,
which is what the model is being asked to do now. It is not true of the seventy in between.

Measured on this benchmark's own sizing -- run 43, a 170,000-token cell filled to 90% -- the
seeded conversation divides as **57% user-turn text, 28% assistant replies and 14% tool
results**: 87,551 tokens of user turns against 21,978 of tool payload across 72 seeded turns.
A strategy that may only touch the tool half is working on a seventh of the prompt, which is
why the anchored rows in that run land at 72% of the window where the uncompacted control
lands at 85%. The user half was never a tuning question; nothing was allowed to look at it.

**What this does.** Past ``trigger_fraction`` of the ceiling it takes the user turns between a
fixed head and a fixed tail, sends them to a summarizer, and puts the summary back in their
place as a single user message. Tool-call groups, tool results and assistant messages are not
read, not annotated and not excluded -- see :meth:`UserTurnAnchoredSummarizationCompactionStrategy._band`,
which is the one place the selection rule is written down. That independence is not tidiness:
the point of the row is to be comparable with the tool-side rows in the same table, and a
strategy that shed both halves would answer neither question.

**It recompacts its own output, and that is the opposite of what ``_shorten`` does.**
:meth:`~._anchored.AnchoredCompactionStrategy._shorten` refuses to touch a tool result that
already carries :data:`~._anchored.REMOVAL_MARKER`, and the comment there says why: the
replacement carries the marker's own tokens on top of the budget, so a second pass would
shorten it again and a third again, each one a fresh mutation at the same position, and a
strict-prefix cache is re-billed from that position every time. That reasoning is sound *for
that strategy*, because its trigger is the band's geometry: a result sits in the band from the
turn it ages out of the tail until the end of the run, so "trim whatever is in the band" is an
instruction that fires on every single pass.

Here the trigger is a threshold that the compaction itself moves away from. A pass only runs
when the prompt is past ``trigger_fraction`` of the ceiling, and a pass that runs removes
tens of thousands of tokens, so the next pass cannot happen until the conversation has grown
all of that back. Between two compactions the prefix is therefore byte-identical on every
turn, which is the property the anchored design is built around; what changes is how often the
prefix is rewritten at all, and at the default trigger on a conversation of this shape that is
once or twice in a run rather than once per turn. So the same argument that makes re-trimming
wrong in ``_shorten`` makes re-summarising affordable here, and the two were decided
differently on purpose rather than by one of them forgetting the other.

Refusing to recompact is not a neutral alternative either. The summary is a user message, so a
strategy that would not re-read its own output would have to keep every earlier summary
alongside every new one -- exactly the accumulation
:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.records_in_conversation`
exists to report, where each preserved record raises a floor under the prompt that no later
pass can lower. Recompaction is what stops the floor rising.

**It will not run on nothing new.** A pass whose band holds only this strategy's own earlier
summary would rewrite one message at one position and free almost nothing, which is precisely
the thrash ``_shorten`` refuses. So the band has to contain at least one turn that is not a
summary -- the same shape as
:meth:`~._toolsummary.ToolResultRecallMiddleware._record_due`, which re-arms its trigger on new
material rather than on size, and for the same reason: the size that fired the trigger does not
go away by itself.

**The replacement is a user message, where the framework's own summarizer writes an assistant
one.** ``SummarizationStrategy`` summarises whole groups of every kind, so its output belongs to
neither speaker and assistant is the neutral choice. This replaces user turns only, and the
replacement has to be readable *as* those turns on the next pass: a summary written as
assistant prose would be invisible to the selection rule above, so nothing could ever
recompact it, and the first summary would sit in the prompt for the rest of the run. It also
keeps the conversation's shape legal -- an assistant message inserted between a user turn and
the assistant reply to it puts two assistant messages in a row, which several providers reject.

**How it marks what it replaced is the framework's mechanism and not a new one.**
``SummarizationStrategy`` inserts its summary at the first index it superseded, annotates the
summary with :data:`~agent_framework._compaction.SUMMARY_OF_MESSAGE_IDS_KEY` and
:data:`~agent_framework._compaction.SUMMARY_OF_GROUP_IDS_KEY`, writes the reverse link onto
each superseded message, and excludes them with a reason. All five steps are repeated here
verbatim, so a conversation compacted by this strategy reads back through the same trace
metadata as one compacted by the framework's. Replacing rather than deleting is what makes the
recompaction honest as well: the band that the second pass reads is a summary that still
*stands for* the turns behind it, so nothing is lost by a route nobody can follow.

**Retention of the turns' content is explicitly not measured here.** The question this row
answers is how much of a conversation is user-side and therefore how much a strategy that may
touch it can remove -- ``snap%`` in the benchmark's table. What the summariser managed to keep
is a separate question, and one the recall instrument cannot ask of filler turns that carry no
planted facts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from agent_framework import Message
from agent_framework._compaction import (
    EXCLUDED_KEY,
    GROUP_ANNOTATION_KEY,
    SUMMARIZED_BY_SUMMARY_ID_KEY,
    SUMMARY_OF_GROUP_IDS_KEY,
    SUMMARY_OF_MESSAGE_IDS_KEY,
    annotate_message_groups,
    annotate_token_counts,
    group_messages,
    included_token_count,
    set_excluded,
)

from ._preserve import any_preserved

if TYPE_CHECKING:
    from agent_framework import TokenizerProtocol
    from agent_framework._clients import SupportsChatGetResponse

__all__ = [
    "DEFAULT_KEEP_HEAD_USER_TURNS",
    "DEFAULT_KEEP_TAIL_USER_TURNS",
    "DEFAULT_USER_SUMMARY_PROMPT",
    "DEFAULT_USER_TRIGGER_FRACTION",
    "EXCLUDE_REASON",
    "SUMMARY_ID_PREFIX",
    "USER_SUMMARY_MARKER",
    "UserTurnAnchoredSummarizationCompactionStrategy",
]

logger = logging.getLogger(__name__)

#: Fraction of the ceiling at which the user band is summarised.
#:
#: **0.8, and the number is the whole of what makes recompaction affordable.** This strategy
#: rewrites the prefix at one position and is willing to rewrite it again later, which is the
#: behaviour :meth:`~._anchored.AnchoredCompactionStrategy._shorten` refuses; what buys it is
#: that a pass cannot follow another pass until the conversation has grown back everything the
#: first one removed. A trigger low enough to fire on a conversation that has barely grown
#: would collapse that argument and produce exactly the per-turn mutation the anchored design
#: exists to avoid.
#:
#: It sits above :data:`~._toolsummary.DEFAULT_TRIGGER_FRACTION`, which is 0.6, and the two are
#: different decisions rather than an inconsistency. That one asks a *model* for a record and
#: has to ask early, because the record degrades with the bulk it is given to read; this one
#: asks a summarizer for prose whose quality is not what the row measures, and pays instead in
#: a broken cached prefix, so it wants to fire as late as it can while still leaving the
#: conversation room to continue.
DEFAULT_USER_TRIGGER_FRACTION: Final[float] = 0.8

#: User turns kept verbatim at the start.
#:
#: One, because one is what is load-bearing: the opening turn carries the task and the
#: requirements, and every deleting strategy measured in this package throws them away first --
#: truncation left 29 of 53 planted facts in the prompt and the model used none of them,
#: because the codes survived while the turns saying which deployment each belonged to did not.
#: The turns after it are the conversation, not its terms of reference, and summarising them is
#: the point of the strategy rather than a cost of it.
DEFAULT_KEEP_HEAD_USER_TURNS: Final[int] = 1

#: User turns kept verbatim at the end.
#:
#: One, and for a reason that does not generalise upward: the last user turn is the live
#: request, and a model answering a summary of the question it was just asked is answering the
#: wrong question. A second-to-last turn has no such claim -- it has already been answered --
#: so a larger tail buys nothing and costs the band its newest and most quotable material.
DEFAULT_KEEP_TAIL_USER_TURNS: Final[int] = 1

#: Marks the message this strategy leaves in place of the turns it replaced.
#:
#: Two jobs, as :data:`~._anchored.REMOVAL_MARKER` has two. It tells the model that what it is
#: reading stands for turns that are no longer present, which a model shown a silently reduced
#: conversation cannot know; and it is how a later pass recognises its own earlier output,
#: which decides whether there is new material to compact at all.
USER_SUMMARY_MARKER: Final[str] = "[earlier turns in this conversation, compacted]"

#: Prefix of the ``message_id`` given to every summary this strategy inserts.
#:
#: The identification is doubled deliberately: the id is what the framework's own trace
#: metadata is keyed on, and the marker above is what survives a round trip through a store
#: that assigns its own ids. Either one alone has a failure mode in which the strategy stops
#: recognising its own output and starts accumulating summaries instead of replacing them.
SUMMARY_ID_PREFIX: Final[str] = "user_summary_"

#: Reason recorded on the turns this strategy supersedes.
#:
#: Named after the framework's own ``"summarized"``, and distinct from it, so a conversation
#: read back says which of the two strategies claimed a message.
EXCLUDE_REASON: Final[str] = "user_turn_summarized"

#: What the summarizer is asked for.
#:
#: Deliberately not the framework's ``DEFAULT_SUMMARIZATION_PROMPT``, which asks for "the
#: entire conversation" in "no more than five sentences". Both halves are wrong here. The
#: input is one side of the conversation rather than all of it, so a prompt describing both
#: invites the model to invent the assistant's half; and a fixed five-sentence bound is a
#: length, which is the one thing a summary of seventy turns must be allowed to vary.
DEFAULT_USER_SUMMARY_PROMPT: Final[str] = (
    "You are compacting a conversation to save space. Below are the user's own turns, in "
    "order, with the assistant's replies and any tool output removed. Rewrite them as a "
    "single, much shorter account of what the user asked for, in the order they asked for "
    "it. Keep every requirement, constraint, correction and preference exactly as stated, "
    "and quote verbatim any value that could not be reconstructed or guessed: identifiers, "
    "codes, names, numbers, paths, URLs, versions, states, timestamps. Drop pleasantries, "
    "restatements and anything a later turn superseded. Write about the user in the third "
    "person and add nothing that is not in the text you were given."
)


def _mark_summarized_by(message: Message, summary_id: str) -> None:
    """Record on ``message`` which summary now stands for it.

    This is ``agent_framework._compaction._set_group_summarized_by_summary_id`` written out
    rather than called. The key, the value and the placement are the framework's, because a
    caller reading the conversation back -- including the framework's own summary
    reconciliation -- looks for exactly this annotation; what is not borrowed is the helper,
    which is private even by the standard of a module that is already private, and this
    subpackage is meant to be lifted out and pinned against a framework version rather than to
    follow one symbol by symbol. Reaching one level deeper for four lines would buy a
    dependency that a rename could break silently.

    The guard is the framework's too: the annotation is normally the mapping ``group_messages``
    wrote, but a message arriving with something else under that key must not have it mutated
    in place.

    Args:
        message: The superseded message, annotated in place.
        summary_id: Id of the summary that replaced it.
    """
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    if not isinstance(annotation, dict):
        annotation = {}
        message.additional_properties[GROUP_ANNOTATION_KEY] = annotation
    annotation[SUMMARIZED_BY_SUMMARY_ID_KEY] = summary_id


def _is_summary(message: Message) -> bool:
    """Return whether ``message`` is a summary this strategy wrote.

    Two tests rather than one. The id is what the framework's summary annotations key on and is
    the cheaper check, but a message id is assigned by whoever stores the conversation and a
    round trip is free to replace it; the marker travels inside the text and cannot be lost
    without losing the message. Getting this wrong in the false direction is not a crash, which
    is why it is worth doubling: the strategy simply stops recognising its own output, treats
    every pass as new material, and accumulates summaries it believes are turns.

    Args:
        message: The message to inspect.

    Returns:
        True when this strategy wrote it.
    """
    if message.message_id and message.message_id.startswith(SUMMARY_ID_PREFIX):
        return True
    return USER_SUMMARY_MARKER in (message.text or "")


def _format_turns(turns: list[Message]) -> str:
    """Return the user turns as the numbered transcript the summarizer reads.

    Numbered because order is part of what has to survive: a later turn may correct an earlier
    one, and a summary that cannot tell which came first will keep the correction and the thing
    it corrected as though both still stood.

    Args:
        turns: The user messages about to be replaced, in conversation order.

    Returns:
        One line per turn.
    """
    return "\n".join(f"{index}. {message.text or ''}" for index, message in enumerate(turns, start=1))


class UserTurnAnchoredSummarizationCompactionStrategy:
    """Replace the user turns between a fixed head and tail with one summary of them.

    Args:
        max_input_tokens: Ceiling the included prompt is measured against. This is the model's
            real input limit rather than its advertised context window; the two differ by
            128,000 tokens on GPT-5-class deployments, and configuring the larger one puts the
            trigger above what the service will accept.
        tokenizer: Token counter, shared with whatever measures the result.

    Keyword Args:
        client: Chat client that writes the summary. **Security:** its output permanently
            replaces the user's own turns in the conversation and is trusted from then on like
            any other message, so point this only at a service trusted as much as the primary
            model -- the same indirect-prompt-injection caveat the framework's
            ``SummarizationStrategy`` carries, and it applies harder here, because the text it
            replaces is the user's.
        keep_head_user_turns: User turns at the start left verbatim. See
            :data:`DEFAULT_KEEP_HEAD_USER_TURNS`.
        keep_tail_user_turns: User turns at the end left verbatim. See
            :data:`DEFAULT_KEEP_TAIL_USER_TURNS`.
        trigger_fraction: Fraction of ``max_input_tokens`` the included prompt must pass before
            anything happens. Below it the strategy returns without reading the conversation's
            shape at all: a compaction that was not needed spends a summarizer call and breaks
            a cached prefix for nothing. See :data:`DEFAULT_USER_TRIGGER_FRACTION`.
        prompt: What the summarizer is asked for. See :data:`DEFAULT_USER_SUMMARY_PROMPT`.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        tokenizer: TokenizerProtocol,
        client: SupportsChatGetResponse[Any],
        keep_head_user_turns: int = DEFAULT_KEEP_HEAD_USER_TURNS,
        keep_tail_user_turns: int = DEFAULT_KEEP_TAIL_USER_TURNS,
        trigger_fraction: float = DEFAULT_USER_TRIGGER_FRACTION,
        prompt: str | None = None,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If the ceiling is not positive, either anchor is negative, or the
                trigger is outside ``(0.0, 1.0]``. A trigger of zero would fire on an empty
                conversation, where the band is empty and the only thing a pass can produce is
                a summarizer call; above one it can never fire, which is a row that silently
                measures the uncompacted control under another name.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if keep_head_user_turns < 0 or keep_tail_user_turns < 0:
            raise ValueError("keep_head_user_turns and keep_tail_user_turns must be >= 0.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.client = client
        self.keep_head_user_turns = keep_head_user_turns
        self.keep_tail_user_turns = keep_tail_user_turns
        self.trigger_fraction = trigger_fraction
        self.prompt = prompt or DEFAULT_USER_SUMMARY_PROMPT
        self._compactions = 0
        self._replaced = 0
        self._failures = 0

    @property
    def user_compactions(self) -> int:
        """Passes that replaced a band of user turns with a summary.

        The count that says whether this row is measuring the design at all. Zero means the
        conversation never passed ``trigger_fraction``, or passed it with nothing between the
        anchors, and either way the row is the uncompacted control wearing another name --
        which is the reading this package has twice had to add a counter to rule out.

        Read it against :attr:`user_messages_replaced` rather than alone. One compaction that
        replaced seventy turns and seven that replaced ten each cost very different amounts of
        cache: every pass re-bills the prompt from its own edit to the end, so the number of
        passes is the number of times that was paid.
        """
        return self._compactions

    @property
    def user_messages_replaced(self) -> int:
        """User turns the most recent compaction superseded.

        The most recent, not the total, and the distinction is the same one
        :attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.groups_kept_uncovered`
        draws: this answers how much of the conversation the strategy is standing in for *as it
        now stands*, which is what the prompt's size is made of. A running total would double
        count by construction, because a later pass reads the earlier pass's summary and
        replaces it again -- so a run that compacted seventy turns, then recompacted the
        summary of them plus ten more, would report eighty-one turns replaced out of eighty-one
        that ever existed, while the prompt carries one message.

        It is not reset by a pass that declines, because what it describes is the summary
        sitting in the prompt and that summary is still standing for those turns. Zero
        therefore means one thing only: no compaction has happened, which is the same reading
        :attr:`user_compactions` gives and is why the two are reported together.
        """
        return self._replaced

    @property
    def user_summary_failures(self) -> int:
        """Passes where the summarizer raised, or returned nothing, and the band was left alone.

        Non-zero means part of what this row measured is a summarizer that was not answering,
        and the row is then the uncompacted control for those passes -- reported rather than
        hidden for the reason
        :attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_used`
        is: a strategy that quietly degrades into a different one produces a number belonging
        to neither.

        Counted on both failures, because they are the same event to the conversation. The
        framework's own strategy separates them only in its log line and treats an empty
        summary exactly as it treats an exception, for the same reason: what the caller has in
        either case is no replacement text, and the one thing that must not happen is
        superseding the turns anyway.
        """
        return self._failures

    async def __call__(self, messages: list[Message]) -> bool:
        """Summarise the user band when the prompt is over the trigger.

        Nothing is mutated until the summary is in hand. That ordering is the whole of the
        "degrade safely" contract: exclusion flags and the summary's back-references are
        written in one step after the call returns text, so a summarizer that raises, times out
        or answers with whitespace leaves a conversation that is byte-identical to the one it
        was given. The alternative -- excluding first and restoring on failure -- is the
        mutate-and-roll-back that ``MinimumGainAnchoredCompactionStrategy`` already refused to
        do for its projection, on the ground that a rollback missing one field is a silent
        wrong answer rather than a loud one.

        Args:
            messages: The conversation, mutated in place.

        Returns:
            True if the outgoing messages changed. False does not imply the prompt now fits:
            this strategy has no shed step and no fallback, so a conversation whose bulk is tool
            output is one it can legitimately leave over the ceiling. Compose it with a
            strategy that sheds if that matters.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        if included_token_count(messages) <= int(self.max_input_tokens * self.trigger_fraction):
            return False

        band = self._band(messages)
        if not band:
            return False

        summary = await self._summarize([messages[span["start_index"]] for span in band])
        if summary is None:
            return False

        self._replace(messages, band, summary)
        return True

    def _band(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Return the user groups this pass may replace, oldest first.

        **User groups and nothing else.** ``group_messages`` gives every user message a group of
        its own with ``kind == "user"``, so the selection needs no heuristics: tool-call groups,
        their results and assistant narration are simply never candidates. That is what keeps
        this row independent of the tool-side rows it is meant to be compared with, and it is
        the reason the rule lives in one method rather than being spread across the caller.

        **Excluded turns are not candidates.** A message another strategy has already dropped is
        not in the prompt, so superseding it would cost a summarizer call to remove nothing --
        and it would put a message that is not being sent into this pass's replaced count, where
        it would read as compaction this strategy performed.

        **Preserved turns are not candidates either, and do not count towards the anchors.** The
        mark in :mod:`._preserve` means a message no strategy may shorten, drop or shed, and a
        superseded message is dropped in every sense that matters. Leaving it out of the band
        entirely -- rather than out of the head and tail arithmetic as well -- would be the
        subtler bug: a preserved last turn would otherwise take the tail's place and let the
        live request be summarised.

        **At least one candidate must not be this strategy's own summary.** A band holding only
        the previous summary is a band whose replacement frees nothing, and rewriting one
        message at one position for nothing is exactly the thrash
        :meth:`~._anchored.AnchoredCompactionStrategy._shorten` refuses to do. This is the same
        shape as :meth:`~._toolsummary.ToolResultRecallMiddleware._record_due`, which re-arms on
        new material rather than on size, and for the same reason: the size that fired the
        trigger does not go away when the compaction that answered it is already in the prompt.

        Args:
            messages: The conversation, already grouped.

        Returns:
            The spans to replace, empty when this pass must do nothing.
        """
        turns = [
            span
            for span in group_messages(messages)
            if span.get("kind") == "user"
            and not messages[span["start_index"]].additional_properties.get(EXCLUDED_KEY, False)
            and not any_preserved(messages[span["start_index"] : span["end_index"] + 1])
        ]
        last = len(turns) - self.keep_tail_user_turns
        if last <= self.keep_head_user_turns:
            return []
        band = turns[self.keep_head_user_turns : last]
        if all(_is_summary(messages[span["start_index"]]) for span in band):
            return []
        return band

    async def _summarize(self, turns: list[Message]) -> str | None:
        """Return the summary of ``turns``, or None when the summarizer did not produce one.

        Args:
            turns: The user messages about to be replaced, in conversation order.

        Returns:
            The summary text, stripped, or None on either failure.
        """
        try:
            response = await self.client.get_response(
                [
                    Message(role="system", contents=[self.prompt]),
                    Message(role="user", contents=[_format_turns(turns)]),
                ],
                stream=False,
            )
        except Exception as error:
            # Broad on purpose, and the framework's own summarizing strategy is broad in the
            # same place. A compaction strategy runs inside the chat client's own call path, so
            # anything this does not catch fails the user's turn -- which trades a conversation
            # that is merely too long for one that does not happen.
            logger.warning("Skipping user-turn compaction: summary generation failed (%s).", error)
            self._failures += 1
            return None
        summary = response.text.strip() if response.text else ""
        if not summary:
            logger.warning("Skipping user-turn compaction: the summarizer returned no text.")
            self._failures += 1
            return None
        return summary

    def _replace(self, messages: list[Message], band: list[dict[str, Any]], summary: str) -> None:
        """Put one user message in place of the band, linked to what it supersedes.

        Every step here is ``SummarizationStrategy``'s, performed in its order: the summary
        carries the ids of the messages and the groups it stands for, each superseded message
        carries the summary's id back, the supersession is recorded as an exclusion with a
        reason, the message is inserted at the first index it replaced, and the groups are
        re-annotated from there. Matching it is worth more than tidiness -- a caller reading a
        compacted conversation back, or the framework's own summary reconciliation, finds the
        annotations it already knows how to follow rather than a second convention that means
        the same thing.

        The insertion index is taken from the band's own first span rather than by searching for
        the message, because messages compare by value: two user turns with the same text are
        equal, and a search would find the earlier one and insert the summary in front of a turn
        the head was protecting.

        The id is numbered by compaction rather than by conversation length, which is what the
        framework uses. Length is not unique across passes here: this strategy's own summary is
        superseded by the next one, so a conversation can be compacted at the same length twice
        and the second summary would claim the first one's id, silently pointing every
        back-reference at the wrong message.

        Args:
            messages: The conversation, mutated in place.
            band: The spans being replaced, oldest first.
            summary: The summarizer's text.
        """
        summary_id = f"{SUMMARY_ID_PREFIX}{self._compactions}"
        replaced = [messages[span["start_index"]] for span in band]
        summary_message = Message(
            role="user",
            contents=[f"{USER_SUMMARY_MARKER}\n{summary}"],
            message_id=summary_id,
            additional_properties={
                GROUP_ANNOTATION_KEY: {
                    SUMMARY_OF_MESSAGE_IDS_KEY: [message.message_id for message in replaced if message.message_id],
                    SUMMARY_OF_GROUP_IDS_KEY: [str(span["group_id"]) for span in band],
                }
            },
        )
        for message in replaced:
            _mark_summarized_by(message, summary_id)
            set_excluded(message, excluded=True, reason=EXCLUDE_REASON)
        insertion_index = int(band[0]["start_index"])
        messages.insert(insertion_index, summary_message)
        annotate_message_groups(messages, from_index=insertion_index, force_reannotate=False)
        self._compactions += 1
        self._replaced = len(replaced)
