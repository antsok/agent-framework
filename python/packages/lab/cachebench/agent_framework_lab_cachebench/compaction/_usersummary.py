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

**The threshold was argued to be self-limiting. Measurement says it is not, and this module
used to say the opposite.** The argument that stood here was that the trigger is a threshold
the compaction itself moves away from: a pass only runs past ``trigger_fraction`` of the
ceiling, a pass removes tens of thousands of tokens, so the next pass cannot happen until the
conversation has grown all of that back -- once or twice in a run rather than once per turn.
Measured on gpt-5.6-luna at a 170,000-token window, 0.9 fill, scaled payload, this row
reported ``USERCOMPACT:31 USERREPLACED:2`` on one seed and ``USERCOMPACT:30 USERREPLACED:2``
on another: about one pass per turn, each replacing the previous summary and one new turn,
with the cache hit rate down to 53% and 61% against the uncompacted control's 95%.

The argument has two holes and either one is enough on its own.

- **The band is not the prompt.** It is the user turns between the anchors, and the assistant
  replies and tool results around them are not this strategy's to touch. On the run above the
  band was about 28% of the prompt, so a pass that removed *all* of it need not take the
  prompt back under the line -- and on a workload whose bulk is tool output it certainly does
  not. "A compaction moves away from its own trigger" is true only while the band is most of
  what there is to remove, which is not the ordinary case.
- **After the first pass the band is not even that.** What is left between the anchors is this
  strategy's own summary plus whatever turns arrived since, a fraction of a percent of the
  prompt. Every later pass then rewrites the prefix at the summary's position to free almost
  nothing, which is exactly the thrash ``_shorten`` refuses.

The size that fired the trigger does not go away by itself -- the sentence
:meth:`~._toolsummary.ToolResultRecallMiddleware._record_due` is built around -- and the
"something in the band is not my own summary" rule below is satisfied by every new user turn,
so above the trigger the condition stays true for the rest of the run.

**Hysteresis, and it is one rule: a pass has to be worth what a pass costs.** The band must be
worth at least ``min_band_share`` of the included prompt before anything is summarised: see
:data:`DEFAULT_MIN_BAND_SHARE` for the number, and
:meth:`UserTurnAnchoredSummarizationCompactionStrategy._worth_compacting` for the rule and for
the two shapes of hysteresis it was chosen over. That is what bounds the firing count, and the
bound is geometric rather than a cap: the band can only regrow from user turns added since the
last pass, so for the band to be worth a share ``f`` of the prompt again the prompt itself must
have grown by a factor of at least ``1 / (1 - f)``. A conversation that grows from the size of
its first compaction to ``k`` times that size therefore compacts at most
``ceil(log(k) / log(1 / (1 - f)))`` times -- one pass per 11% of prompt growth at the default,
so seven over a conversation that doubles, against one per turn before. What it costs is stated
with the constant: a band that never clears the share is a band never compacted at all, and
:attr:`UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined` is what says so
rather than leaving the row looking like the uncompacted control.

So the same argument that makes re-trimming wrong in ``_shorten`` makes re-summarising
affordable here only once a pass is required to be worth something, and the two are now
decided by one rule read from opposite ends rather than by one of them forgetting the other.

Refusing to recompact is not a neutral alternative either. The summary is a user message, so a
strategy that would not re-read its own output would have to keep every earlier summary
alongside every new one -- exactly the accumulation
:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.records_in_conversation`
exists to report, where each preserved record raises a floor under the prompt that no later
pass can lower. Recompaction is what stops the floor rising.

**It will not run on nothing new.** A pass whose band holds only this strategy's own earlier
summary would rewrite one message at one position and free exactly nothing, which is precisely
the thrash ``_shorten`` refuses. So the band has to contain at least one turn that is not a
summary -- the same shape as
:meth:`~._toolsummary.ToolResultRecallMiddleware._record_due`, which re-arms its trigger on new
material rather than on size. That rule is necessary and, as the measurement above shows, not
sufficient: one new turn satisfies it, so it is the ``f = 0`` corner of the share rule and both
are checked in the one place.

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
    "DEFAULT_MIN_BAND_SHARE",
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
#: **0.8, and it decides when compaction starts, not how often it happens.** This constant used
#: to be described as the whole of what makes recompaction affordable, on the argument that a
#: pass cannot follow another pass until the conversation has grown back everything the first
#: one removed. That argument is wrong whenever the band is a minority of the prompt, which is
#: the ordinary case, and the row measured 30 passes in a run because of it -- see the module
#: docstring. What bounds the passes is :data:`DEFAULT_MIN_BAND_SHARE`; this number only decides
#: how large the prompt is before the first one.
#:
#: Lowering it does not now produce per-turn mutation, because the share rule is what refuses
#: that; it produces a first compaction on a smaller prompt, which is the thing the flag is for.
#:
#: It sits above :data:`~._toolsummary.DEFAULT_TRIGGER_FRACTION`, which is 0.6, and the two are
#: different decisions rather than an inconsistency. That one asks a *model* for a record and
#: has to ask early, because the record degrades with the bulk it is given to read; this one
#: asks a summarizer for prose whose quality is not what the row measures, and pays instead in
#: a broken cached prefix, so it wants to fire as late as it can while still leaving the
#: conversation room to continue.
DEFAULT_USER_TRIGGER_FRACTION: Final[float] = 0.8

#: Share of the included prompt the band must be worth before a pass may run.
#:
#: **0.1, and it is the hysteresis.** Without it the strategy fires once per turn for the rest
#: of a run that stays above the trigger, because the band it reads after its first pass is its
#: own summary plus the turns arrived since -- 0.4% to 0.9% of the prompt on the fixture in
#: ``tests``, and ``USERREPLACED:2`` on the live run in the module docstring. Each of those
#: passes spends a summarizer call and re-bills the prompt from the summary's position to the
#: end of the conversation in order to free a few hundred tokens.
#:
#: **Where 0.1 comes from: this package's own break-even, at this benchmark's own length.**
#: :data:`~._anchored.DEFAULT_MIN_GAIN_FRACTION` derives when an edit repays the prefix it
#: breaks -- ``R > B * (p - c) / (p + T * c)`` for ``R`` tokens removed, ``B`` tokens behind the
#: edit, ``T`` turns still to come and the measured prices ``p = 0.66`` and ``c = 0.07`` per
#: million. This strategy's edit sits at the band's first position, just behind the head turn,
#: so ``B`` is very nearly the whole prompt and the share is a share of the prompt. Solved for
#: ``T`` instead of for ``R``, a band worth ``f`` of the prompt repays itself within
#: ``((p - c) / f - p) / c`` turns: **75 turns at 0.1**, which is the length of the
#: conversations this benchmark seeds (72 filler turns at the 170,000-token cell). So a pass
#: this rule permits can repay inside the run it is part of, and the 0.9% passes measured above
#: would have needed about 900 turns to.
#:
#: **Why not 0.29, which is the same formula.** That is the twenty-turns-remaining figure, and
#: at this row's geometry it refuses the 28% band the live run had -- the row would then be the
#: uncompacted control, which is not a fix for a strategy that fires too often. It is also a
#: floor this package deliberately did *not* make a default: it ships as
#: :class:`~._anchored.MinimumGainAnchoredCompactionStrategy`, a separate row measured against
#: its own parent, because a break-even floor changes what a strategy measures rather than only
#: how often it fires. This constant is sized to remove passes that cannot repay under any
#: conversation length anyone here runs, and to leave the rest to that row.
#:
#: **What it bounds.** A band regrows only from user turns added since the last pass, so for the
#: band to be worth ``f`` of the prompt again the prompt must have grown by a factor of at least
#: ``1 / (1 - f)`` -- 1.111 here. Compactions over a conversation growing from ``P`` to ``kP``
#: are therefore at most ``ceil(log(k) / log(1.111))``: seven on a conversation that doubles
#: after its first compaction, three on one that grows by a third, against one per turn before.
#:
#: **What it costs.** A band that never reaches a tenth of the prompt is never compacted, and on
#: a workload whose bulk is tool output that can be every band in a run. The row is then the
#: uncompacted control on its user half -- but it says so, in
#: :attr:`UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined`, which is the
#: difference between this and the silent degradation the counters in this package exist to
#: rule out. ``0.0`` restores the old behaviour exactly, so the two can be run side by side.
DEFAULT_MIN_BAND_SHARE: Final[float] = 0.1

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
        min_band_share: Share of the included prompt the band must be worth before a pass may
            run. This is the hysteresis: without it the strategy fires once per turn for the
            rest of a run that stays above the trigger. ``0.0`` restores that behaviour, which
            is what every run before this one measured. See :data:`DEFAULT_MIN_BAND_SHARE`.
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
        min_band_share: float = DEFAULT_MIN_BAND_SHARE,
        prompt: str | None = None,
    ) -> None:
        """Validate and store the configuration.

        Raises:
            ValueError: If the ceiling is not positive, either anchor is negative, the trigger
                is outside ``(0.0, 1.0]``, or the band share is outside ``[0.0, 1.0)``. A
                trigger of zero would fire on an empty conversation, where the band is empty
                and the only thing a pass can produce is a summarizer call; above one it can
                never fire, which is a row that silently measures the uncompacted control under
                another name. A band share of one demands a band that is the whole prompt, which
                is the same never-fires row by the other route; zero is legal and is the
                behaviour this class had before the share existed, kept so the two can be run
                side by side.
        """
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")
        if keep_head_user_turns < 0 or keep_tail_user_turns < 0:
            raise ValueError("keep_head_user_turns and keep_tail_user_turns must be >= 0.")
        if not 0.0 < trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be in (0.0, 1.0].")
        if not 0.0 <= min_band_share < 1.0:
            raise ValueError("min_band_share must be in [0.0, 1.0).")
        self.max_input_tokens = max_input_tokens
        self.tokenizer = tokenizer
        self.client = client
        self.keep_head_user_turns = keep_head_user_turns
        self.keep_tail_user_turns = keep_tail_user_turns
        self.trigger_fraction = trigger_fraction
        self.min_band_share = min_band_share
        self.prompt = prompt or DEFAULT_USER_SUMMARY_PROMPT
        self._compactions = 0
        self._replaced = 0
        self._failures = 0
        self._below_trigger = 0
        self._declined = 0

    @property
    def user_compactions(self) -> int:
        """Passes that replaced a band of user turns with a summary.

        The count that says whether this row is measuring the design at all. Zero means the row
        is the uncompacted control wearing another name -- which is the reading this package has
        twice had to add a counter to rule out -- and which of the three ways it got there is
        said by :attr:`user_passes_below_trigger`, :attr:`user_passes_declined` and
        :attr:`user_summary_failures`, exactly one of which is non-zero on such a row.

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

    @property
    def user_passes_below_trigger(self) -> int:
        """Passes that returned at the trigger check, having read nothing but the prompt's size.

        The ordinary state of a conversation that has not grown yet, and the reason it is
        counted at all is that ``user_compactions == 0`` has several causes and a reader needs
        to tell them apart from the numbers rather than from a story. This one says the
        conversation never reached the line; :attr:`user_passes_declined` says it did and the
        band was not worth a pass; :attr:`user_summary_failures` says the band was and the
        summarizer was not. Together with :attr:`user_compactions` the four partition every
        pass over a non-empty conversation, so a row where the user half did nothing always has
        exactly one non-zero number saying why.

        On a composed row this number is the one
        :attr:`~._composed.ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.user_passes_starved`
        subdivides: of the passes counted here, the starved ones are those the phase in front
        took below the line.
        """
        return self._below_trigger

    @property
    def user_passes_declined(self) -> int:
        """Passes over the trigger where the band was not worth what a pass costs.

        The hysteresis counter, and the price of having one. Every pass counted here is a pass
        the old behaviour would have spent: a summarizer call, a rewritten prefix billed from
        the summary's position to the end of the conversation, and a few hundred tokens freed.
        Thirty of them is what the live run in the module docstring measured.

        It is one number for three refusals, because they are one statement -- there was not
        enough here to be worth a pass. The band was empty (the anchors cover the conversation),
        or it held only this strategy's own earlier summary (nothing new has been said), or it
        was worth less than ``min_band_share`` of the prompt (something new has been said and it
        is not enough). :meth:`_worth_compacting` is where all three are written down.

        **Read it as a ratio against :attr:`user_compactions`, not alone.** Non-zero beside a
        non-zero compaction count is the mechanism working. Non-zero beside *zero* compactions
        is a row whose user half never acted, and the share is then too high for that workload's
        band -- which is a configuration to change, not a strategy that failed, and it is
        visible here rather than looking like a conversation that never grew.
        """
        return self._declined

    async def __call__(self, messages: list[Message]) -> bool:
        """Summarise the user band when the prompt is over the trigger and the band is worth it.

        Two conditions, and the second one is why this is not once per turn: the trigger says
        the prompt is large enough to act on, and :meth:`_worth_compacting` says this band is
        worth the pass. Each refusal increments the counter that names it, so the four outcomes
        -- under the line, declined, summarizer failed, compacted -- partition the passes and a
        row that did nothing says which.

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
        prompt_tokens = included_token_count(messages)
        if prompt_tokens <= int(self.max_input_tokens * self.trigger_fraction):
            self._below_trigger += 1
            return False

        band = self._band(messages)
        if not self._worth_compacting(messages, band, prompt_tokens):
            self._declined += 1
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

        **Whether the band is worth replacing is not decided here.** This returns what a pass
        *may* touch; :meth:`_worth_compacting` decides whether a pass runs at all. The two were
        one method until the band's own size had to be weighed, and separating them is what
        keeps the selection rule readable as a selection rule -- and puts all three reasons a
        pass declines in one place, behind one counter.

        Args:
            messages: The conversation, already grouped.

        Returns:
            The spans this pass may replace, empty when the anchors leave nothing between them.
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
        return turns[self.keep_head_user_turns : last]

    def _worth_compacting(self, messages: list[Message], band: list[dict[str, Any]], prompt_tokens: int) -> bool:
        """Return whether this band is worth what a pass costs.

        **The hysteresis, in the one place it is written down.** A pass spends a summarizer call
        and rewrites the prompt at the band's first position, which re-bills everything from
        there to the end of the conversation at the uncached rate. Three refusals, which are one
        statement -- there is not enough here to pay for that:

        - **An empty band.** The anchors cover the whole conversation, so there is nothing
          between them to stand for.
        - **A band holding only this strategy's own earlier summary.** Replacing it frees
          exactly nothing: one message is rewritten at one position and the prompt is the size
          it was. This is :meth:`~._toolsummary.ToolResultRecallMiddleware._record_due`'s rule,
          re-arming on new material rather than on size.
        - **A band worth less than ``min_band_share`` of the included prompt.** The rule the
          other two are corners of, and the one the measurement forced. The condition that fires
          a pass is the prompt's size, and the prompt does not shrink to the size of the band --
          so once the band is down to the previous summary plus a turn or two, every new turn
          makes the second rule true again while freeing a fraction of a percent. Thirty passes
          in a run, measured; see the module docstring and :data:`DEFAULT_MIN_BAND_SHARE`.

        **The band is measured, not counted.** A minimum number of new *turns* since the last
        pass would bound nothing: it divides the firing count by a constant and leaves it
        growing with the conversation. A minimum *growth of the prompt* would bound it, but on
        growth this strategy may not touch -- a run whose bulk is tool output would re-arm the
        band without adding anything to it. Tokens in the band against tokens in the prompt is
        the one measure that is both what a pass would free and what it would cost.

        The share is taken against the prompt as the trigger read it, before anything was
        removed, and the summary's own size is not deducted from the band: it is not known until
        the summarizer has answered, and it is one message against a band that has to clear a
        fifth of the prompt to get here.

        Args:
            messages: The conversation, already grouped and token-annotated.
            band: The spans :meth:`_band` selected.
            prompt_tokens: Included tokens, as the trigger check read them.

        Returns:
            True when the pass may go on to spend a summarizer call.
        """
        if not band:
            return False
        replaced = [messages[span["start_index"]] for span in band]
        if all(_is_summary(message) for message in replaced):
            return False
        return included_token_count(replaced) >= prompt_tokens * self.min_band_share

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
