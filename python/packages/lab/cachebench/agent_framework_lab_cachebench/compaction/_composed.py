# Copyright (c) Microsoft. All rights reserved.

"""Compact the tool half and the user half of one conversation, in that order.

**Two strategies, neither of which can reach the other's material.**
:class:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy` replaces tool groups
with a record of them and never reads a user turn;
:class:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy` summarises user turns
and never reads a tool group. Each is therefore bounded by the share of the conversation it is
allowed to touch, and each says so in its own module: the record strategy is capped at the tool
payload, and the user-turn strategy's offline replay put its floor at "the assistant replies and
the tool payload it may not touch". ``STRATEGIES.md`` names composing the two as the obvious
next measurement and states that it is not made there.

This is that composition. What it reaches is the question a run of it would answer, and nothing
in this module is a report of one: no row produced by this class has been measured at the time
it was written, so every claim below is about mechanism -- which messages each phase selects,
which number each phase reads -- and not about what the pair removed.

**Order: the record phase first, then the user-turn phase.** Three reasons, the first of which
is the expensive one to get wrong.

1. *The record has to be asked for before the bulk degrades it.*
   :data:`~._toolsummary.DEFAULT_TRIGGER_FRACTION` is 0.6 against
   :data:`~._usersummary.DEFAULT_USER_TRIGGER_FRACTION`'s 0.8, and the two constants document
   why they differ: the record strategy asks a *model* to write down what a set of tool results
   contained, and that record is measured degrading with the bulk it is given to read, so it
   fires early; the user-turn strategy asks a summarizer for prose whose quality is not what its
   row measures and pays only in a broken cached prefix, so it fires as late as it can. A
   composition that let the user phase act first would hand the record phase -- and, worse, the
   middleware that does the asking -- a conversation already shrunk below the line that is
   supposed to trigger them. That is not a delay. It is the ask never being made.
2. *The phase that can remove less goes first.* The record phase is capped at the tool share of
   the conversation and the user phase at the user share, and on the sizing this package
   measures those are not close: the user half is most of the prompt and the tool half a
   seventh of it. A pass ordered the other way would routinely take the prompt from over the
   user line to under the record phase's lower one in a single step, which silences the record
   half every pass; ordered this way the removal is small enough that the user line usually
   survives it, and when it does not, the counter below says so instead of the user half merely
   looking inert.
3. *The record phase's fallback decides from group positions.* When a record does not free
   enough, :class:`~._anchored.AnchoredCompactionStrategy` runs behind it and reads a band
   defined by counting groups from each end. Running it after the user phase would have it
   count a band containing this run's summary message rather than the user turns it replaced,
   so the fallback's geometry -- and therefore what the ``tool_summary_anchored`` row means --
   would differ between the composed row and the row it is meant to be compared with.

**Both phases run on a pass where both want to act.** They are not alternated across turns and
neither vetoes the other: the pass calls the record phase, re-reads the conversation, and calls
the user-turn phase, and it reports a change when either reported one. That is safe because
they select disjoint messages -- ``group_messages`` gives a user message a group of kind
``user`` and a call-and-result pair a group of kind ``tool_call``, and each phase's selection
rule names exactly one of those kinds -- so there is no message both could claim and no order in
which one could supersede what the other had already superseded.

**The interference is the token accounting, and it is not removed, it is made visible.** Both
phases begin by reading :func:`~agent_framework._compaction.included_token_count` and comparing
it with their own trigger, so the second phase sees a number the first phase moved. Three
things are done about that and a fourth is deliberately not done:

- The conversation is re-annotated between the phases, forced, when the first phase changed
  anything. Token counts are cached per message inside the group annotations, and the record
  phase's fallback rewrites tool results *in place*; without the re-read the user phase would
  threshold against text that no longer exists. This is
  :meth:`~._anchored.AnchoredCompactionStrategy.__call__`'s own reasoning, one level up.
- The order above is chosen so the moved number moves as little as it can.
- :attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.user_passes_starved`
  counts the passes where the record phase's removals are why the prompt is under the user
  phase's own line. It is the one reading of "the user half did nothing" that is not about the
  user half at all. It was written as a within-pass transition -- above the line when the pass
  started, at or below it when the record phase finished -- and that definition could not see
  the case it was written for: the record phase's trigger is *lower* than the user phase's, so
  it takes the prompt down before the user line is ever reached and holds it there, and the
  transition never happens. Measured: seed 1 of the run in ``STATE.md`` reported ``snap 63%``,
  no ``USERCOMPACT`` and no ``USERSTARVED``. It now counts against what the record phase has
  removed over the whole run rather than over one pass, which makes the old reading a special
  case of the new one.

**The user half is silent for four different reasons, and the flags say which.** This is the
failure the composition exists to avoid -- a row that is one of its halves under a new name --
so "did not compact" is not one state here but four, each with its own counter:
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_below_trigger`
is never considered, ``user_passes_starved`` is the subset of those the record phase caused,
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined` is
considered and held back by the user phase's own hysteresis, and
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summary_failures` is
a summarizer that did not answer. A reader who has only the flags column can tell the four
apart, which is the whole point of having them.
- What is *not* done is overriding either phase's trigger. Both keep their own configuration,
  including the two thresholds, because the composed row only means anything beside the two
  single rows if its halves fire where those rows' halves fire. Collapsing them into one shared
  trigger -- the shape :class:`~agent_framework._compaction.TokenBudgetComposedStrategy` uses,
  where every variant compacts to one ceiling -- answers a different question: that family holds
  size fixed to compare orderings of deletion, and here the sizes the two phases reach are the
  measurement rather than a nuisance to be normalised away.

**It has no ceiling and no fallback of its own.** Returning False does not mean the prompt now
fits, exactly as it does not for either part: the record phase carries its own fallback behind
its own ceiling, the user phase has none by design, and adding a third shed step here would put
a removal in the composed row that neither single row can make, which is the one thing that
would stop the three being readable together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_framework._compaction import (
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
)

from ._toolsummary import ToolResultAnchoredSummarizationCompactionStrategy
from ._usersummary import UserTurnAnchoredSummarizationCompactionStrategy

if TYPE_CHECKING:
    from agent_framework import Message, TokenizerProtocol

__all__ = ["ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy"]

#: One of the two phases, which is what walking :attr:`.strategies` hands back.
_Phase = ToolResultAnchoredSummarizationCompactionStrategy | UserTurnAnchoredSummarizationCompactionStrategy


class ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Run the record strategy and then the user-turn strategy over one conversation.

    Keyword Args:
        tokenizer: Token counter, and it must be the one both phases were given. This class
            re-reads the conversation between them, so a count taken here with a different
            counter would hand the second phase a number its own trigger was not set against.
        tool_results: The record-then-drop strategy, configured as its own row configures it.
            Its recall middleware is wired by the caller and is not optional: without it the
            model is never pinned to the recall tool, no record is written, and this phase can
            only ever wait and then fall back. See ``_live.run_live``, which discovers this
            object inside this one for exactly that reason.
        user_turns: The user-band summarising strategy, configured as its own row configures
            it, summarizer client included.

    Raises:
        ValueError: If the two phases measure against different ceilings. Each phase's trigger
            is a fraction of its own ``max_input_tokens``, and the whole of the order argument
            in this module is about which of the two fractions is crossed first -- which is a
            statement about fractions only while they are fractions of one number. Two ceilings
            would also make ``user_passes_starved`` compare a count against a line taken from
            somewhere else.
    """

    def __init__(
        self,
        *,
        tokenizer: TokenizerProtocol,
        tool_results: ToolResultAnchoredSummarizationCompactionStrategy,
        user_turns: UserTurnAnchoredSummarizationCompactionStrategy,
    ) -> None:
        """Validate the pair and store it."""
        if tool_results.max_input_tokens != user_turns.max_input_tokens:
            raise ValueError(
                "tool_results and user_turns must share one max_input_tokens: "
                f"{tool_results.max_input_tokens} and {user_turns.max_input_tokens} are two ceilings, "
                "and the trigger fractions are then not comparable."
            )
        self.tokenizer = tokenizer
        self.tool_results = tool_results
        self.user_turns = user_turns
        self.max_input_tokens = tool_results.max_input_tokens
        self._starved = 0
        self._removed_by_record = 0

    @property
    def strategies(self) -> tuple[_Phase, ...]:
        """The phases, in the order :meth:`__call__` runs them.

        One source of truth for the order, and the attribute name is
        :class:`~agent_framework._compaction.TokenBudgetComposedStrategy`'s so that anything
        walking a composition to find a part -- ``_live``'s middleware wiring is the caller that
        matters -- reads one name rather than a name per composing class.
        """
        return (self.tool_results, self.user_turns)

    @property
    def records_found(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.records_found`."""
        return self.tool_results.records_found

    @property
    def records_in_conversation(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.records_in_conversation`."""
        return self.tool_results.records_in_conversation

    @property
    def fallbacks_used(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_used`."""
        return self.tool_results.fallbacks_used

    @property
    def fallbacks_after_record(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_after_record`."""
        return self.tool_results.fallbacks_after_record

    @property
    def groups_kept_uncovered(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.groups_kept_uncovered`."""
        return self.tool_results.groups_kept_uncovered

    @property
    def user_compactions(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_compactions`."""
        return self.user_turns.user_compactions

    @property
    def user_messages_replaced(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_messages_replaced`."""
        return self.user_turns.user_messages_replaced

    @property
    def user_summary_failures(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summary_failures`."""
        return self.user_turns.user_summary_failures

    @property
    def user_passes_below_trigger(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_below_trigger`.

        On a composed row this is the number :attr:`user_passes_starved` subdivides: of the
        passes where the user phase never reached its line, the starved ones are those the
        record phase's removals account for and the rest are a conversation that was not big
        enough on its own.
        """
        return self.user_turns.user_passes_below_trigger

    @property
    def user_passes_declined(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined`."""
        return self.user_turns.user_passes_declined

    @property
    def user_passes_starved(self) -> int:
        """Passes where the record phase's removals are why the user phase never reached its line.

        The one number composing produces that neither part can report, and the only reading of
        a silent user half that is not about the user half. Every other explanation for
        ``user_compactions == 0`` is visible on the user phase itself -- a conversation that
        never grew past the trigger (:attr:`user_passes_below_trigger`), a band not worth a pass
        (:attr:`user_passes_declined`), a summarizer that raised
        (:attr:`user_summary_failures`) -- and all of them are statements about the user side.
        This one says the user side was never consulted, because the phase in front of it had
        already taken the prompt below the threshold it fires on.

        **It used to be a per-pass transition, and that made it silent exactly where it was
        needed.** The test was ``before > user_line >= after`` within one pass: the prompt had
        to be above the user line when the pass started and at or below it when the record phase
        finished. On the run this counter was written for -- gpt-5.6-luna, a 170,000-token
        window at 0.9 fill, seed 1 -- the composed row reported ``REC:1, RECORDS:1, FORCED:1,
        RECFORCED:1``, a 63% snapshot, no ``USERCOMPACT`` and **no ``USERSTARVED``**. The reason
        is structural rather than incidental: the record phase's trigger is 0.6 and the user
        phase's is 0.8, so the record phase removes the tool payload while the prompt is still
        in the 60s and holds it there for the rest of the run. The prompt is then never above
        the user line at the *start* of a pass, the transition can never be observed, and a row
        whose user half was starved on every pass reported starvation on none.

        **What is counted now is the counterfactual, and it is carried across passes.** This
        class accumulates the tokens the record phase has removed over the run and counts a pass
        as starved when the prompt is at or below the user line *and* would have been above it
        with those tokens still in the conversation. On the pass where a transition does happen
        that is the old test -- the removal of that pass is part of the total -- so every pass
        the old definition counted is still counted, and the passes it could not see are now
        counted too.

        The total is the sum of what each pass's record phase removed, measured as the drop in
        the included token count across the call and floored at zero: the phase inserts notes of
        its own when its fallback runs, and a pass whose notes outweighed its removals has
        removed nothing rather than a negative amount. The counterfactual it stands for is "the
        same conversation, compacted by the user phase exactly as it was, with the record phase
        never run" -- which is the ``user_summary_anchored`` row the composed row is read
        against, and so the right question to be asking.

        Counted per pass rather than as a flag, for the reason
        :attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_after_record`
        is: each pass is its own event, the conversation grows back between them, and one
        starved pass out of one is a different row from one out of twenty.

        It is reported rather than prevented. Preventing it would mean running the user phase
        against a trigger this class had overridden, and then the composed row's user half would
        be firing where no ``user_summary_anchored`` row ever fires -- which takes the pair of
        rows the composition exists to be compared with and makes them incomparable.

        Zero on a row whose user half did compact is the ordinary case and says nothing beyond
        "the order cost nothing here".
        """
        return self._starved

    @property
    def tokens_removed_by_record_phase(self) -> int:
        """Tokens the record phase has removed from the prompt over this run.

        The quantity :attr:`user_passes_starved` is decided against, exposed because a counter
        derived from a hidden number is a counter nobody can check. It is a running total of
        per-pass drops in the included token count, so it describes the conversation as it now
        stands: the record phase's exclusions are permanent, and nothing here puts them back.

        Not a flag on the row. It is a token count rather than an event count, it moves with the
        workload rather than with the strategy, and the flags column is read for the latter.
        """
        return self._removed_by_record

    async def __call__(self, messages: list[Message]) -> bool:
        """Run the record phase, then the user-turn phase.

        Neither phase is called conditionally on the other. Each is trusted to read its own
        trigger and decline, because that decision is the thing its own row measures and a copy
        of it here would be a second place for the two to disagree about when a strategy fires.

        What this class does instead of deciding for them is *attribute* a decline. The user
        phase counts its own three refusals; this counts the one it cannot see, which is the
        record phase in front of it having taken the prompt below the line the user phase reads.

        Args:
            messages: The conversation, mutated in place.

        Returns:
            True if either phase changed the outgoing messages. False does not imply the prompt
            now fits: neither phase promises that, and this adds no shed step of its own.
        """
        if not messages:
            return False
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        before = included_token_count(messages)

        changed = await self.tool_results(messages)
        after = before
        if changed:
            # Forced, because the record phase's fallback shortens tool results in place and
            # the counts are cached per message: an incremental re-read would leave the user
            # phase thresholding against text that is no longer in the conversation. Skipped
            # when nothing changed, which is the same trade ``_anchored.__call__`` makes.
            annotate_message_groups(messages)
            annotate_token_counts(messages, tokenizer=self.tokenizer, force_retokenize=True)
            after = included_token_count(messages)
            # Floored at zero: the fallback inserts notes where it shed a group, so a pass can
            # end larger than it started, and a pass that removed nothing has removed nothing
            # rather than a negative amount that would then pay for a later pass's starvation.
            self._removed_by_record += max(before - after, 0)

        # Read on every pass and not only on one that changed something. The record phase's
        # exclusions are permanent, so the pass that made them is not the only pass they starve;
        # on the live row that produced this counter it was never the transition pass at all,
        # because the record phase acts at 0.6 and the user line is at 0.8. See
        # ``user_passes_starved``.
        user_line = int(self.user_turns.max_input_tokens * self.user_turns.trigger_fraction)
        if after <= user_line < after + self._removed_by_record:
            self._starved += 1

        return await self.user_turns(messages) or changed
