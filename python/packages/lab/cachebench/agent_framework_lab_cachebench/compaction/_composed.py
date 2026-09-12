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
  counts the passes where the record phase's removals took the prompt from above the user
  phase's own line to at or below it. Zero is the ordinary case and the counter is then silent;
  non-zero is the one reading of "the user half did nothing" that is not about the user half at
  all, and it would otherwise be indistinguishable from a conversation with nothing in its band.
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
    def user_passes_starved(self) -> int:
        """Passes where the record phase's removals dropped the prompt under the user phase's line.

        The one number composing produces that neither part can report, and the only reading of
        a silent user half that is not about the user half. Every other explanation for
        ``user_compactions == 0`` is visible elsewhere -- a conversation that never grew past
        the trigger, a band holding nothing but an earlier summary, a summarizer that raised
        (:attr:`user_summary_failures`) -- and all of them are statements about the user side.
        This one says the user side was never consulted, because the phase in front of it had
        already taken the prompt below the threshold it fires on.

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

    async def __call__(self, messages: list[Message]) -> bool:
        """Run the record phase, then the user-turn phase.

        Neither phase is called conditionally on the other. Each is trusted to read its own
        trigger and decline, because that decision is the thing its own row measures and a copy
        of it here would be a second place for the two to disagree about when a strategy fires.

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
        if changed:
            # Forced, because the record phase's fallback shortens tool results in place and
            # the counts are cached per message: an incremental re-read would leave the user
            # phase thresholding against text that is no longer in the conversation. Skipped
            # when nothing changed, which is the same trade ``_anchored.__call__`` makes.
            annotate_message_groups(messages)
            annotate_token_counts(messages, tokenizer=self.tokenizer, force_retokenize=True)
            user_line = int(self.user_turns.max_input_tokens * self.user_turns.trigger_fraction)
            if before > user_line >= included_token_count(messages):
                self._starved += 1

        return await self.user_turns(messages) or changed
