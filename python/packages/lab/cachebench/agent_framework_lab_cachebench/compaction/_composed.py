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
in this module is a report of one: no row produced by this class has been measured since it was
given the shared line described below, so every claim here is about mechanism -- which messages
each phase selects, which number each phase reads -- and not about what the pair removed.

**One reading of the prompt decides both halves, and one line is what both are judged against.**
Those are two decisions, forced by the same measurement, so they are stated together.

The row this class first shipped as was structurally tool-only. The record phase fires at
:data:`~._toolsummary.DEFAULT_TRIGGER_FRACTION`, 0.6, and the user phase at
:data:`~._usersummary.DEFAULT_USER_TRIGGER_FRACTION`, 0.8; the record phase goes first, removes
the tool payload while the prompt is still in the 60s of the ceiling, and holds it there for the
rest of the run. The prompt then never reaches 0.8 at all, so the user half is never consulted:
gpt-5.6-luna at a 170,000-token window, 0.9 fill, seed 1, reported ``REC:1, RECORDS:1, FORCED:1,
RECFORCED:1``, a 63% snapshot and no ``USERCOMPACT`` -- a composed row that was
``tool_summary_anchored`` under a longer name.

*Giving the two halves the same fraction does not on its own fix that*, and this class does not
pretend it does. With both lines at 0.6 and the phases still judged one after the other, the
record phase acts first, takes the prompt below the shared line, and the user phase declines on
its own re-test: the same inertness with a different number on it. What makes a shared line mean
anything is that **both halves are judged against the size the prompt had when the pass began**.
:meth:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.__call__` reads
:func:`~agent_framework._compaction.included_token_count` once, before either phase runs, and
hands that one number to both: a half that would have fired on the pass-entry size fires,
whatever the other half has already removed.

**Judging a half against a stale size cannot let it claim the other's material.** The two
selection rules name disjoint kinds -- ``group_messages`` gives a user message a group of kind
``user`` and a call-and-result pair a group of kind ``tool_call``, and
:meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy._band` takes only the
first while the record phase takes only the second -- so a phase acting on a number a moment out
of date still reads its own half of the conversation and nothing else. What the staleness does
cost is this: the second half may act when the prompt is *already* under the line, freeing
tokens a strategy reading the live size would have left alone, and spending a summarizer call
and a rewritten prefix to do it. That is the deliberate choice being made. The alternative is
the row measured above, where the second half never acts at all, and a pass that removes
slightly more than it had to is a cheaper defect than a row that silently measures one half.

**The shared line is the record phase's own, and the alignment runs in that direction only.**
``user_trigger_fraction`` defaults to None, which means the user half is judged at
``tool_results.trigger_fraction`` -- 0.6 by default, and whatever a sweep of that flag has moved
it to, so the composed row's two halves cannot drift apart under a sweep of the record row's
trigger. Aligning the other way, holding the record phase back to 0.8, is the one direction that
is not safe: the record is written by a *model* asked to read the tool payload, it is measured
degrading with the bulk it is given, and the middleware that does the asking reads the same
prompt -- so a record asked for late is a record asked for on more material, and a record asked
for after the prompt has been cut below the line is a record never asked for. Aligning down
costs the user half an earlier first compaction than its own row takes; aligning up costs the
record its quality and possibly its existence. A caller who wants the two apart again passes an
explicit ``user_trigger_fraction``, which restores the two-line row this class shipped as --
kept, like every other restored default in this package, so that the two can be run side by side.

Neither sub-strategy's own trigger is touched by any of this. ``tool_summary_anchored`` and
``user_summary_anchored`` as separate rows read their own ``trigger_fraction`` off the
conversation in front of them, exactly as they always did; the composition supplies both numbers
through :meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.compact_against`
rather than by reconfiguring the objects it was handed.

**Order: the record phase first, then the user-turn phase.** Three reasons were given for it.
Judging both halves at pass entry makes one of them moot, which is said here rather than left
standing.

1. *The record has to be asked for before the bulk degrades it.* Survives, and it is still the
   expensive one to get wrong -- but it is about the conversation's size *between* passes rather
   than within one. Both phases' removals are permanent, so a user phase that ran first would
   hand the middleware a conversation already shrunk below the line that asks for a record at
   all, on this call and on every call after it. Pass-entry judging does nothing about that: it
   governs which phase acts within a pass, not what the next call sees.
2. *The phase that can remove less goes first.* **Moot as stated, and replaced.** The reason
   given was that the record phase's smaller removal usually leaves the user line intact for the
   phase behind it -- which is now true by construction, at either order, because the user half
   no longer reads a size the record phase has moved. What the order still buys is close to the
   opposite of that reason: the record phase's removal makes the prompt smaller, so the band the
   user phase weighs is a *larger* share of it, and
   :data:`~._usersummary.DEFAULT_MIN_BAND_SHARE` is cleared more easily than it would be the
   other way round. The hysteresis reads the live prompt on purpose; see
   :meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.compact_against`.
3. *The record phase's fallback decides from group positions.* Survives untouched, because it is
   about the conversation's shape rather than its size. When a record does not free enough,
   :class:`~._anchored.AnchoredCompactionStrategy` runs behind it and reads a band defined by
   counting groups from each end. Running it after the user phase would have it count a band
   containing this run's summary message rather than the user turns it replaced, so the
   fallback's geometry -- and therefore what the ``tool_summary_anchored`` row means -- would
   differ between the composed row and the row it is meant to be compared with.

**Both phases run on a pass where both want to act.** They are not alternated across turns and
neither vetoes the other: the pass calls the record phase, re-reads the conversation, and calls
the user-turn phase, and it reports a change when either reported one. That is safe because they
select the two disjoint kinds above, so there is no message both could claim and no order in
which one could supersede what the other had already superseded.

**The conversation is still re-annotated between the phases, and now for one reason rather than
two.** It used to be re-read because the user phase thresholded against the result; it no longer
does. It is re-read because the record phase's fallback rewrites tool results *in place* and
token counts are cached per message, so the band the user phase weighs -- and the drop this class
attributes to the record phase -- would otherwise be measured against text that is no longer in
the conversation. Forced, and skipped when nothing changed, which is the trade
:meth:`~._anchored.AnchoredCompactionStrategy.__call__` makes one level up.

**The user half is silent for four different reasons, and the flags say which.** This is the
failure the composition exists to avoid -- a row that is one of its halves under a new name --
so "did not compact" is not one state here but four, each with its own counter:
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_below_trigger`
is never considered,
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.user_passes_starved` is the
subset of those the record phase caused,
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined` is
considered and held back by the user phase's own hysteresis, and
:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summary_failures` is
a summarizer that did not answer. A reader who has only the flags column can tell the four
apart, which is the whole point of having them. On an aligned row the starvation count is zero
by construction, so a silent user half there is always the user half's own doing -- which is
what makes the other three readable as instructions to change something.

**The user half's summary mode is the user half's own, and neither boundary mode can reach the
record.** ``user_turns`` decides whether it recompacts its summary, leaves it standing as a
boundary, or folds the standing ones, and it decides that from its own configuration: nothing
here reads or sets it, so the mode reaches this row through the same builder that reaches the
single row. What composing has to guarantee is that a boundary or a fold cannot disturb the
record half's one preserved message, and it cannot. The user half reads user groups alone in
every mode; a fold excludes only the standing summaries and inserts one message where the
oldest of them stood; and the record -- a tool result, marked preserved under the record half's
own reason -- is neither read, excluded, nor moved relative to anything but that one insertion,
exactly as it is not by an ordinary pass. Nor does any of it touch the shared line: the fold sits
inside :meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.compact_against`,
behind the same trigger check :meth:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.__call__`
hands the pass-entry size to.

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
            takes the one reading both phases are judged against, so a count taken here with a
            different counter would judge both halves against a number neither trigger was set
            against.
        tool_results: The record-then-drop strategy, configured as its own row configures it.
            Its recall middleware is wired by the caller and is not optional: without it the
            model is never pinned to the recall tool, no record is written, and this phase can
            only ever wait and then fall back. See ``_live.run_live``, which discovers this
            object inside this one for exactly that reason.
        user_turns: The user-band summarising strategy, configured as its own row configures
            it, summarizer client included.
        user_trigger_fraction: Fraction of the shared ceiling the user half is judged at *inside
            this composition*. None, the default, means ``tool_results.trigger_fraction``: one
            line for both halves, moving with whatever the record row's trigger was swept to.
            An explicit fraction sets the two apart again -- pass ``user_turns.trigger_fraction``
            to restore the two-line row this class shipped as. The object handed in is not
            reconfigured either way, so the same instance run as its own row still reads its own
            trigger. See the module docstring for why the alignment runs downward.

    Raises:
        ValueError: If the two phases measure against different ceilings, or if an explicit
            ``user_trigger_fraction`` is outside ``(0.0, 1.0]``. Each phase's trigger is a
            fraction of its own ``max_input_tokens``, and one shared line is a line only while
            the two fractions are fractions of one number; two ceilings would also make
            ``user_passes_starved`` compare a count against a line taken from somewhere else. A
            fraction of zero would fire the user half on an empty conversation and one above one
            could never fire it, which are the bounds
            :class:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy` sets on its
            own trigger for the same two reasons.
    """

    def __init__(
        self,
        *,
        tokenizer: TokenizerProtocol,
        tool_results: ToolResultAnchoredSummarizationCompactionStrategy,
        user_turns: UserTurnAnchoredSummarizationCompactionStrategy,
        user_trigger_fraction: float | None = None,
    ) -> None:
        """Validate the pair and store it."""
        if tool_results.max_input_tokens != user_turns.max_input_tokens:
            raise ValueError(
                "tool_results and user_turns must share one max_input_tokens: "
                f"{tool_results.max_input_tokens} and {user_turns.max_input_tokens} are two ceilings, "
                "and the trigger fractions are then not comparable."
            )
        if user_trigger_fraction is not None and not 0.0 < user_trigger_fraction <= 1.0:
            raise ValueError("user_trigger_fraction must be in (0.0, 1.0].")
        self.tokenizer = tokenizer
        self.tool_results = tool_results
        self.user_turns = user_turns
        self.max_input_tokens = tool_results.max_input_tokens
        #: The line the user half is judged at in this composition, defaulted to the record
        #: half's so that the two cannot drift apart under a sweep of the record row's trigger.
        self.user_trigger_fraction = (
            tool_results.trigger_fraction if user_trigger_fraction is None else user_trigger_fraction
        )
        self._starved = 0
        self._removed_by_record = 0
        self._removed_out_of_reach = 0

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
    def user_summaries_in_conversation(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summaries_in_conversation`."""
        return self.user_turns.user_summaries_in_conversation

    @property
    def user_summary_tokens(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summary_tokens`."""
        return self.user_turns.user_summary_tokens

    @property
    def user_folds(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_folds`."""
        return self.user_turns.user_folds

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
        enough on its own. On an aligned row the second subset is empty, and this number is
        then exactly the passes the prompt never reached the shared line on.
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

        **Zero by construction on an aligned row, which is the point of aligning.** With one
        line for both halves the record phase can only act on a pass whose entry size is *above*
        that line, and a pass above the line is a pass the user half is consulted on -- so
        nothing the record phase removes is ever removed out of the user half's reach, the
        quantity below stays at zero, and so does this. A non-zero value on a row built with the
        default ``user_trigger_fraction`` is therefore a defect report rather than a
        configuration note, and should be read as one.

        **What it counts, on a row whose halves were deliberately set apart.** A pass is starved
        when the size the pass was judged against is at or below the user line, *and* would have
        been above it with :attr:`tokens_removed_out_of_user_reach` still in the conversation.
        The counterfactual it stands for is "the same conversation, compacted by the user phase
        exactly as it was, with the record phase never run" -- which is the
        ``user_summary_anchored`` row the composed row is read against, and so the right
        question to be asking.

        **It is a statement across passes and never within one, which it did not used to be.**
        A pass is judged by the size it began with, so the record phase cannot take the prompt
        out from under the user half's decision on the pass it acts on: whatever it removes
        there, the user half has already been offered the conversation that contained it. Only
        removals made on *earlier* passes are missing from a later pass's entry reading, and only
        those are counted here. A row with two lines and a single pass therefore reports nothing,
        where the shipped row's first version reported a starved half on exactly that case.

        **It has been three definitions, and the first two are why the third is worded as it
        is.** It began as a within-pass transition -- above the line when the pass started, at or
        below it when the record phase finished -- and could not see the case it was written
        for: the record phase's trigger was *lower* than the user phase's, so it took the prompt
        down before the user line was ever reached and held it there, and the transition never
        happened. Measured: seed 1 of the run in ``STATE.md`` reported ``snap 63%``, no
        ``USERCOMPACT`` and no ``USERSTARVED``. It then counted against everything the record
        phase had removed over the whole run, which saw that case -- and, once the halves shared
        a line, saw a second case that is not starvation at all: an ordinary quiet pass after a
        successful compaction, where the prompt is small *because the row worked*. Restricting
        the total to what was removed while the user half could not be consulted keeps the first
        reading and drops the second.

        Counted per pass rather than as a flag, for the reason
        :attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_after_record`
        is: each pass is its own event, the conversation grows back between them, and one
        starved pass out of one is a different row from one out of twenty.

        Reported rather than prevented, on a row whose halves were set apart deliberately: the
        prevention is the aligned default, and a caller who has asked for two lines has asked
        for the row this counts.
        """
        return self._starved

    @property
    def tokens_removed_by_record_phase(self) -> int:
        """Tokens the record phase has removed from the prompt over this run.

        A running total of per-pass drops in the included token count, so it describes the
        conversation as it now stands: the record phase's exclusions are permanent, and nothing
        here puts them back. Read beside :attr:`user_messages_replaced` it says how the composed
        row's two halves divided the work.

        Not a flag on the row. It is a token count rather than an event count, it moves with the
        workload rather than with the strategy, and the flags column is read for the latter.
        """
        return self._removed_by_record

    @property
    def tokens_removed_out_of_user_reach(self) -> int:
        """The part of that total the user half was never in a position to see.

        The quantity :attr:`user_passes_starved` is decided against, exposed because a counter
        derived from a hidden number is a counter nobody can check. It is the sum of the drops
        made on passes whose entry size was at or below the user line -- passes where the user
        half declined at its trigger, so the tokens the record phase took were tokens that could
        never have carried the prompt over that line.

        Zero on an aligned row, for the reason :attr:`user_passes_starved` is: the record phase
        acts only above the shared line, and above the line the user half is consulted.

        Floored at zero per pass. The record phase's fallback inserts notes where it shed a
        group, so a pass can end larger than it started, and a pass that removed nothing has
        removed nothing rather than a negative amount that would then pay for a later pass's
        starvation.
        """
        return self._removed_out_of_reach

    async def __call__(self, messages: list[Message]) -> bool:
        """Run the record phase, then the user-turn phase, both judged at the size on entry.

        The prompt is measured once, before either phase runs, and that single reading is what
        each phase's trigger is compared with. Neither phase is called conditionally on the
        other and neither re-reads the size to decide: the record phase cannot move the number
        the user phase is judged by, which is the whole of what makes one shared line more than
        a renamed version of the two-line row. What a phase does *after* it has decided to act
        is read off the conversation as it now stands -- the band's share of the prompt, the
        ceiling the record phase's fallback is reached for -- because those are questions about
        what a pass would cost and what still does not fit.

        Each phase is still trusted to decide whether to act, because that decision is the thing
        its own row measures and a copy of it here would be a second place for the two to
        disagree. What this class supplies is the pair of numbers the decision is taken against.

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
        entry_tokens = included_token_count(messages)
        user_line = int(self.max_input_tokens * self.user_trigger_fraction)

        # Decided before the record phase runs, and so against what it removed on *earlier*
        # passes only. This pass's removal comes out of the prompt after the entry reading was
        # taken, so it cannot be the reason that reading was under the line -- adding it would
        # report a pass as starved by a removal it had not yet made. Across passes it is a
        # different statement, and the true one: the record phase's exclusions are permanent, so
        # what it took while the user half was declining at its trigger is missing from every
        # entry reading after it.
        if entry_tokens <= user_line < entry_tokens + self._removed_out_of_reach:
            self._starved += 1

        changed = await self.tool_results.compact_against(
            messages,
            prompt_tokens=entry_tokens,
            trigger_tokens=int(self.max_input_tokens * self.tool_results.trigger_fraction),
        )
        if changed:
            # Forced, because the record phase's fallback shortens tool results in place and
            # the counts are cached per message: an incremental re-read would leave the user
            # phase weighing its band against text that is no longer in the conversation, and
            # would misattribute the drop below. Skipped when nothing changed, which is the same
            # trade ``_anchored.__call__`` makes.
            annotate_message_groups(messages)
            annotate_token_counts(messages, tokenizer=self.tokenizer, force_retokenize=True)
            removed = max(entry_tokens - included_token_count(messages), 0)
            self._removed_by_record += removed
            if entry_tokens <= user_line:
                # The user half declines at its trigger on this pass, so what the record phase
                # took here is material it will never be offered. Above the line it is offered
                # everything, whatever the record phase removed first -- which is why an aligned
                # row can never add to this. See ``user_passes_starved``.
                self._removed_out_of_reach += removed

        compacted = await self.user_turns.compact_against(
            messages, prompt_tokens=entry_tokens, trigger_tokens=user_line
        )
        return compacted or changed
