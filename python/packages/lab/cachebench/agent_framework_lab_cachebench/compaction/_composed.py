# Copyright (c) Microsoft. All rights reserved.

"""Compact the tool half, then the user half, then run a last-resort chain while still over.

**Two strategies, neither of which can reach the other's material.**
:class:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy` replaces tool groups
with a record of them and never reads a user turn;
:class:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy` summarises user turns
and never reads a tool group. Each is therefore bounded by the share of the conversation it is
allowed to touch, and each says so in its own module. This is their composition, and it is
three things in order.

1. **The record half records every new batch of tool results and never re-summarises a
   record.** Its recall middleware asks for a further record whenever tool work no record
   covers has accumulated past the trigger -- the composed object's ``repeat_records`` is what
   ``_live.run_live`` reads to switch that on for this row -- and the pass drops what each
   record covers. Existing records are left as they are.
2. **The user half acts only if the record half was not enough.** It is judged at the same
   line, against the prompt *as the record half left it*, and it runs in the boundary mode, so
   a summary it wrote is kept as a boundary rather than re-summarised on the next pass. While
   a record is due and still has time to arrive it is not judged at all: see below.
3. **A last-resort chain, only while the prompt is still over the input budget.** Merge the
   records into one; merge the user summaries into one; rewrite the record harder, up to
   ``harder_attempts`` times; then the record half's fallback, which may drop narration only.
   If the prompt is still over after that, nothing more is done: it goes out over the limit and
   the row reads ``DQ``, which is the intended loud failure.

**The user half waits for a record that is due, because the two halves compact at different
speeds.** The record half compacts in two steps: on the pass where the prompt crosses the line
it can only ask -- its middleware pins a *later* call, the model writes the record there, and
only the pass after that drops what the record covers. The user half compacts in one. Judged
on the asking pass, it saw a prompt the record half had not yet touched and acted at once; when
summarising the user turns alone got back under the line, the middleware, which reads the
prompt on each call's way out, never saw it over the line again and never asked. Measured on
gpt-5.6-luna at 200,000 tokens, 0.9 fill and a 0.8 trigger: no record and one user compaction
on five seeds of five -- the layering inverted, and the cache-breaking half doing all the work.
So on a pass over the line where the record half has tool work a record is due for, or has
already asked for, the user half holds; it acts on the pass that sees the record arrive, if the
prompt is still over the line after the record's drops, or after
:data:`_RECORD_WAIT_RESPONSES` model responses with no record, so a model that never records
cannot leave the conversation uncompacted. With nothing pending it acts as before. The rule,
and the guards on it, are on
:meth:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy._holds_for_record`;
``USERWAIT`` counts the passes held. At 0.6 of 120,000 tokens earlier runs still got a record,
because summarising the user turns there did not get back under the line; a late trigger makes
the inversion more likely rather than causing it.

**Judged after the record phase, which reverses what this module used to argue.** Commit
``4be71a904`` made both halves judge against one reading of the prompt taken at pass entry, on
the argument that a user half judged after the record phase "declines on its own re-test" and
the row is ``tool_summary_anchored`` under a longer name. That argument was made on the
benchmark's short, bounded conversation, where a user half that never acts looks like a row
that measured one half. In the intended design it is not a defect. User compaction rewrites a
message just behind the head, and so breaks nearly the whole cached prefix; it is the second
line of defence, and if tool compaction alone brings the prompt under the line, staying idle
is the correct outcome rather than a starved one. Pass-entry judging instead made the user half
act on passes where the prompt was already under the line after the record phase -- a
summarizer call and a broken prefix bought for nothing -- which this module used to call "the
deliberate choice being made". It is withdrawn. Run 47's composed row, measured at 170,000
tokens and 0.9 fill, ran the pass-entry version; its numbers describe that design and not this
one.

**``USERSTARVED`` and ``tokens_removed_out_of_user_reach`` are retired, not redefined.** Both
existed to report the user half being held idle by the record half's removals as the composition
failing. Under this design that idleness is the composition working, and it is already counted
where it belongs: a pass the user half declined at its trigger is ``USERUNDER``, whether the
conversation had not grown or the record half had brought it under the line. A redefinition
would have had to count "the record half was enough" under a name that says something went
wrong, so the counter and the quantity it was decided against are gone from the class and from
the flags column; records already on disk keep whatever notes they were written with.
``tokens_removed_by_record_phase`` stays, because how the two halves divided the work is still
worth reading.

**One line for both halves, and it is the record half's.** ``user_trigger_fraction`` defaults to
None, which means ``tool_results.trigger_fraction`` -- 0.6 by default, and whatever a sweep of
that flag has moved it to, so the two halves cannot drift apart under a sweep of the record
row's trigger. Aligning the other way, holding the record phase back to the user row's 0.8, is
not safe: the record is written by a *model* asked to read the tool payload, it degrades with
the bulk it is given, and a record asked for late is a record asked for on more material. A
caller who wants two lines passes an explicit ``user_trigger_fraction``. Neither sub-strategy's
own trigger is touched: the composition supplies the line through
:meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.compact_against` rather
than by reconfiguring the object it was handed, so ``user_summary_anchored`` run as its own row
reads its own ``trigger_fraction`` exactly as before.

**How this row configures its halves, and what it leaves alone.** Repeated records and the
boundary mode are this row's configuration, not new defaults for the objects: the recall
middleware's ``repeat_records`` stays off for ``tool_summary_anchored`` unless
``--record-repeats`` is passed, and the user-turn strategy's default mode stays
``recompact`` for ``user_summary_anchored``. Repeats are requested through
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.repeat_records`, which the
run reads when it wires the middleware; the boundary mode is set by ``_strategies``' builder for
this row, and this class does not insist on it -- a caller assembling the parts by hand in the
recompacting mode gets a user half whose summary never stands beside another, so the chain's
user merge never has two to merge and is skipped.

**Order: the record phase, the user phase, then the chain.** The record phase first for two
reasons. The record has to be asked for before the bulk degrades it, and both phases' removals
are permanent, so a user phase that ran first would hand the middleware a conversation already
shrunk below the line that asks for a record at all. And the user phase is the one whose action
breaks the cache, so it is the one that should act only on what the other left. The chain comes
last because every step in it is worse than doing nothing when nothing is needed: each spends a
summarizer call and rewrites a message the cached prefix runs through, so each runs only on a
live reading over the budget, re-read before every step.

**The fallback now runs at the end of the chain, which moves one of this module's older
reasons.** It used to run inside the record phase, before the user phase, on the stated ground
that the fallback counts its band in groups from each end and would otherwise count this row's
summary message rather than the user turns it replaced -- so the fallback's geometry would
differ from the ``tool_summary_anchored`` row's. In this row it now does differ, and that is
accepted: the fallback is the last resort here rather than the first, and behind a record it may
take narration and nothing else (:func:`~._toolsummary._hold_unrecorded`, since ``00061004b``),
so the difference is confined to which assistant replies it sheds. The standalone row keeps its
fallback exactly where and how it was. The give-up fallback the record half takes when no
record ever arrived is not part of the chain and still runs inside the record phase.

**The chain, and why each step is where it is.**

a. *Merge the active records into one*, when there are at least two -- one is a rewrite, which
   is step c. The merged record replaces them in place and is a record to everything that reads
   records: see :meth:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.consolidate_records`.
   It is an ordinary assistant message carrying the record marker, never a tool call: see
   :func:`~._toolsummary.build_record_message`.
b. *Merge the user summaries into one*, when there are at least two, through the user half's own
   fold machinery (:meth:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.fold_if_smaller`).
   After the records because a record merge rewrites the prompt from the oldest record, which
   sits later than the oldest user summary, so it re-bills less.
c. *Rewrite the record harder*, up to ``harder_attempts`` times, each asking for more
   compression than the last. See :data:`DEFAULT_HARDER_ATTEMPTS` for the number.
d. *The record half's fallback*, with every tool group no record covers held, so it may drop
   narration only.
e. Nothing. The prompt goes out over the limit and the row reads ``DQ``.

**Acceptance is generic, and deliberately so.** A merged or rewritten record, or a merged user
summary, is kept if it is non-empty and smaller, in tokens, than what it replaces; otherwise the
old ones stay and the chain moves on. Nothing is checked against the old records, against the
tool results, or against anything the benchmark plants. A real deployment has no planted facts,
and an overlap check fitted to this benchmark's exact codes would either reject correct
paraphrase on real content or pass a lossy summary that happened to keep the codes. "Smaller" is
measured like for like: both sides in the form a written record takes, so a rewrite does not pass
merely because that form drops the second copy a record the model made carries in its call's
arguments -- see :func:`~._toolsummary.build_record_message`. What a merge
loses is measured by the benchmark, through ``facts`` and ``acc1``, not guessed at here. The
record's coverage check is a different thing and is untouched: it decides what a record
*licenses deleting*, which is a question about tool results still in the prompt; this decides
whether a rewrite of records already standing is worth keeping, which is a question about size.

**The records are merged by the summarizer client the user half already has, not by an agent
turn.** The first record has to come from the agent's own model, because only that model has the
tool payload in its context. A merge needs only the records, which are short, and the moment it
is wanted is the moment the prompt is over the budget -- so an agent turn pinned to the recall
tool would be a call made *with* that over-budget prompt, which is the call the chain exists to
avoid. The summarizer sees the records alone. What it writes goes in as an ordinary assistant
message carrying the record marker rather than as a recall call: this used to mint a call id and
synthesise the call, and run 59 measured Foundry refusing the first request that carried one with
``400 invalid_payload`` -- see :func:`~._toolsummary.build_record_message`.

**The other list.** The live path runs this over the copies sent on a call and then over the
store, and a request asked on the first is replayed on the second rather than paid for twice:
the chain remembers its record requests for one pass beyond the one they were made on, and the
user half's builder for this row keeps its last two requests, because a pass over the budget
may ask it for both a band and a fold.

**Counters.** Every counter of both halves is readable off this object, so the flags column --
built by duck typing -- says which half did what. The chain adds one per step:
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.records_merged` and
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.record_merges_rejected`,
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.user_summaries_merged` and
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.user_merges_rejected`,
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.record_rewrites` and
:attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.record_rewrites_rejected`,
and :attr:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy.last_resort_fallbacks`
-- so a row says how far down the chain it went.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final, Literal

from agent_framework import Message
from agent_framework._compaction import (
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
)

from ._toolsummary import (
    RECORD_MARKER,
    ToolResultAnchoredSummarizationCompactionStrategy,
    active_record_groups,
    build_record_message,
    find_record_index,
    record_body,
)
from ._usersummary import UserTurnAnchoredSummarizationCompactionStrategy

if TYPE_CHECKING:
    from agent_framework import TokenizerProtocol

__all__ = [
    "DEFAULT_HARDER_ATTEMPTS",
    "DEFAULT_RECORD_MERGE_PROMPT",
    "ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy",
    "harder_record_prompt",
]

logger = logging.getLogger(__name__)

#: One of the two phases, which is what walking :attr:`.strategies` hands back.
_Phase = ToolResultAnchoredSummarizationCompactionStrategy | UserTurnAnchoredSummarizationCompactionStrategy

#: What one attempt to replace the records ended as.
_Consolidation = Literal["accepted", "rejected", "failed"]

#: Rewrites of the record the chain may try on one pass, each asking for more compression.
#:
#: **Two.** Each attempt costs a summarizer call and, if kept, rewrites a preserved message the
#: cached prefix runs through, so the number is a bound on spend and on breaks rather than a
#: patience setting. One is the rewrite that pays most: the first request is the one that finds
#: the wording a record can lose. The second is there because the first may be kept and still not
#: be enough, and the second asks for markedly less. A third would ask a record of values to fall
#: to about a fifth of its length while keeping every value verbatim, which is below the floor the
#: values themselves set on a record written to carry them; beyond that point attempts are
#: rejected as no smaller, or kept by dropping what the instruction says to keep, and neither is
#: worth a call. Zero switches the step off. The bound is per pass: a prompt that stays over the
#: budget meets the chain again on the next one, where a request identical to one already refused
#: is answered from memory rather than paid for again.
DEFAULT_HARDER_ATTEMPTS: Final[int] = 2

#: How much shorter each successive rewrite is asked to be, as a share of the record's length.
#:
#: Attempt ``k`` asks for about ``0.6 ** k`` of it: 60%, then 36%. A share rather than a token
#: count, because the summarizer can judge a proportion of the text in front of it and cannot
#: count tokens; and stated as an aim, because a record that meets it by dropping values is the
#: failure the instruction beside it is there to forbid.
_HARDER_RATIO: Final[float] = 0.6

#: Model responses the user half waits through for a record that is due, before acting anyway.
#:
#: **Two, and it is the re-force layer's number for the re-force layer's reason**, counted in a
#: different unit. The wait begins on the pass that finds the prompt over the line with a record
#: due. The recall middleware decides on the exit of that pass's call and pins the call after
#: it, so the first response the wait sees is the deciding call's own, and it cannot be the
#: record. The second is the pinned call's. If the model wrote the record, the follow-up call's
#: pass finds it and ends the wait as a completed cycle before the count is read. If the model
#: did anything else -- ignored the pin, was cut off, had the option refused -- the count
#: reaches two on the next pass, the record was not written, and the user half acts. Waiting
#: longer would keep the prompt over the line on the evidence of an ask that has already failed;
#: waiting less would give up on a pass that could not have seen the record. See
#: ``_toolsummary._REFORCE_ARRIVAL_PASSES``, whose argument this is.
#:
#: **Responses rather than passes, because the live path runs two passes per call.** The
#: strategy runs over the copies sent on a call and again over the store after the turn, so a
#: bound of two *passes* would be spent on the crossing turn's own two lists and expire on the
#: pinned call's pass, before the model had written anything. A response is counted off the
#: conversation itself -- an assistant message, see :func:`_responses` -- so the store pass and
#: the next call's copy pass read the same number for the same point in the conversation, and
#: one crossing cannot be counted twice.
_RECORD_WAIT_RESPONSES: Final[int] = 2

#: What the summarizer is asked when the chain merges the active records into one.
#:
#: Generic on purpose, in the sense the acceptance rule is: nothing in it names this benchmark's
#: codes or scopes. Values are to be copied verbatim because a record is exactly the place where a
#: value that cannot be reconstructed survives; wording is to be cut because that is what a merge
#: can remove without losing anything a later question could need.
DEFAULT_RECORD_MERGE_PROMPT: Final[str] = (
    "The numbered texts below are compaction records: each is an account of tool results an "
    "agent received earlier in a conversation, written so those results could be removed. "
    "Write one record that replaces all of them. Copy every identifier, code, number, date, "
    "name and other exact value verbatim, from every record; do not paraphrase, round, "
    "abbreviate or drop any of them. Remove what the records repeat and cut wording, not "
    "values. Reply with the record text only."
)


def harder_record_prompt(attempt: int) -> str:
    """Return what the summarizer is asked on the ``attempt``-th rewrite of the record.

    Each attempt asks for a smaller share of the record's length -- see :data:`_HARDER_RATIO` --
    and keeps the same generic rule about what may not be cut. Several records are handed over
    together when an earlier merge was refused, and the instruction covers that case rather than
    leaving a rewrite of several to be read as a rewrite of the first.

    Args:
        attempt: One for the first rewrite on a pass, two for the second, and so on.

    Returns:
        The system prompt.
    """
    percent = round(100 * _HARDER_RATIO**attempt)
    return (
        "Rewrite the compaction record below so that it is at most about "
        f"{percent}% of its current length. Keep every identifier, code, number, date, name and "
        "other exact value verbatim: do not paraphrase, round, abbreviate or drop any of them. "
        "Cut wording instead -- repetition, explanation, connecting prose and formatting. If "
        "several numbered records are given, write one record that carries the values of all of "
        "them. Reply with the record text only."
    )


def _responses(messages: list[Message]) -> int:
    """Count the assistant messages in the conversation: the clock the user half's wait runs on.

    Excluded messages count too, because the clock measures how far the conversation has gone
    and not what is sent. Nothing needs subtracting. A record the model writes is an assistant
    message, but the pass that sees it ends the wait as an arrival before the count is read. And
    a compaction's own insertions -- the fallback's notes, a merged record -- are made only on
    passes the wait does not hold: a held pass is under the give-up line, so neither fallback
    runs, and under the budget, so the chain does not, and any pass that does not hold ends the
    wait.
    """
    return sum(1 for message in messages if message.role == "assistant")


def _newest_record_identity(messages: list[Message]) -> str:
    """Return what identifies the newest record, or an empty string when there is none.

    The record the record half anchors on, found the way it finds it
    (:func:`~._toolsummary.find_record_index`), and named by something the copies sent on a call
    and the store after it both carry -- not a position, which differs between the two lists.

    A record the model made is named by its call id, which the provider issued. A record the
    chain wrote has no call id -- it is an ordinary message, see
    :func:`~._toolsummary.build_record_message` -- so it is named by its text. The two lists
    agree on that text because the store pass replays the answer the copies pass was given
    (:meth:`ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy._ask`). And a change of
    text is always a new record: the chain writes one only in place of every standing record,
    and only when it is smaller than them, so a written record never repeats the one straight
    before it. The two forms are prefixed apart so neither can read as the other.
    """
    index = find_record_index(messages)
    if index is None:
        return ""
    message = messages[index]
    for content in message.contents:
        if content.type == "function_result" and RECORD_MARKER in str(content.result):
            return f"call:{content.call_id or ''}"
    return f"text:{message.text}"


class ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy:
    """Run the record strategy, then the user-turn strategy, then a last-resort chain.

    Keyword Args:
        tokenizer: Token counter, and it must be the one both phases were given. Every size this
            class reads -- the entry reading, the post-record reading the user half is judged
            against, and the live readings the chain runs on -- is taken with it.
        tool_results: The record-then-drop strategy, configured as its own row configures it.
            Its recall middleware is wired by the caller and is not optional: without it the
            model is never pinned to the recall tool, no record is written, and this phase can
            only ever wait and then fall back. See ``_live.run_live``, which discovers this
            object inside this one for exactly that reason, and which reads
            :attr:`repeat_records` off this one to switch the middleware's repeats on.
        user_turns: The user-band summarising strategy, summarizer client included. Its client
            also writes the chain's merged and rewritten records; see the module docstring for
            why. The builder for this row runs it in the boundary mode.
        user_trigger_fraction: Fraction of the shared ceiling the user half is judged at *inside
            this composition*. None, the default, means ``tool_results.trigger_fraction``: one
            line for both halves, moving with whatever the record row's trigger was swept to.
            The object handed in is not reconfigured either way.
        harder_attempts: Rewrites of the record the chain may try on one pass once merging has
            not brought the prompt under the budget. See :data:`DEFAULT_HARDER_ATTEMPTS`; zero
            switches the step off.
        merge_prompt: What the summarizer is asked when the records are merged. See
            :data:`DEFAULT_RECORD_MERGE_PROMPT`.

    Raises:
        ValueError: If the two phases measure against different ceilings, if an explicit
            ``user_trigger_fraction`` is outside ``(0.0, 1.0]``, or if ``harder_attempts`` is
            negative. Each phase's trigger is a fraction of its own ``max_input_tokens``, and
            one shared line is a line only while the two fractions are fractions of one number;
            the chain's budget is that number too. A fraction of zero would fire the user half on
            an empty conversation and one above one could never fire it, which are the bounds
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
        harder_attempts: int = DEFAULT_HARDER_ATTEMPTS,
        merge_prompt: str | None = None,
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
        if harder_attempts < 0:
            raise ValueError("harder_attempts must be >= 0.")
        self.tokenizer = tokenizer
        self.tool_results = tool_results
        self.user_turns = user_turns
        self.max_input_tokens = tool_results.max_input_tokens
        #: The line the user half is judged at in this composition, defaulted to the record
        #: half's so that the two cannot drift apart under a sweep of the record row's trigger.
        self.user_trigger_fraction = (
            tool_results.trigger_fraction if user_trigger_fraction is None else user_trigger_fraction
        )
        self.harder_attempts = harder_attempts
        self.merge_prompt = merge_prompt or DEFAULT_RECORD_MERGE_PROMPT
        self._removed_by_record = 0
        self._records_merged = 0
        self._record_merges_rejected = 0
        self._user_summaries_merged = 0
        self._user_merges_rejected = 0
        self._record_rewrites = 0
        self._record_rewrites_rejected = 0
        self._record_summary_failures = 0
        self._last_resort_fallbacks = 0
        # The user half's wait for the record half; see ``_holds_for_record``. All of it is read
        # off the conversation or kept here, never in message annotations, because the copies'
        # annotations do not reach the store the next pass runs over.
        self._user_passes_waited = 0
        # The response count the current wait began at, or None when none is running.
        self._wait_since: int | None = None
        # The newest record's identity as the last pass saw it -- see ``_newest_record_identity``;
        # a change is a record arriving.
        self._anchor = ""
        # No new wait may begin at or before this response count: set when a record arrives.
        self._quiet_through = -1
        # The anchor a wait ran out behind, or None. No new wait begins behind it.
        self._declined_behind: str | None = None
        # Record requests answered on this pass and on the one before, keyed by the exact
        # request. Two generations, because the pass that replays is the one straight after the
        # pass that asked; see ``_ask``.
        self._answers: dict[tuple[str, str], str] = {}
        self._previous_answers: dict[tuple[str, str], str] = {}

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
    def repeat_records(self) -> bool:
        """Whether the recall middleware should ask again for new tool work. Always True here.

        The first part of this row's design: every new batch of tool results is recorded, and
        the covered results dropped, rather than one record being written and the rest of the
        run's tool work left to the fallback. The setting belongs to the middleware, which the
        run builds; this is how the row says it wants it, and ``_live.run_live`` turns it on for
        any strategy reporting True here whatever ``--record-repeats`` says. The standalone
        ``tool_summary_anchored`` row reports nothing, so its default is untouched.
        """
        return True

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
    def fallbacks_held_after_record(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.fallbacks_held_after_record`."""
        return self.tool_results.fallbacks_held_after_record

    @property
    def groups_kept_uncovered(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.groups_kept_uncovered`."""
        return self.tool_results.groups_kept_uncovered

    @property
    def groups_preserved_uncovered(self) -> int:
        """:attr:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.groups_preserved_uncovered`."""
        return self.tool_results.groups_preserved_uncovered

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
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_folds`.

        On this row the user half runs in the boundary mode and never folds on its own, so every
        fold counted here is one the chain asked for -- :attr:`user_summaries_merged` -- or a
        replay of one on the other list.
        """
        return self.user_turns.user_folds

    @property
    def user_summaries_replayed(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summaries_replayed`."""
        return self.user_turns.user_summaries_replayed

    @property
    def user_summary_failures(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_summary_failures`."""
        return self.user_turns.user_summary_failures

    @property
    def user_passes_below_trigger(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_below_trigger`.

        On this row it includes every pass where the record half alone brought the prompt under
        the shared line, which is the user half staying idle by design. It used to be subdivided
        by a starvation count that read that case as a defect; see the module docstring for why
        that count is retired.
        """
        return self.user_turns.user_passes_below_trigger

    @property
    def user_passes_declined(self) -> int:
        """:attr:`~._usersummary.UserTurnAnchoredSummarizationCompactionStrategy.user_passes_declined`."""
        return self.user_turns.user_passes_declined

    @property
    def user_passes_waited(self) -> int:
        """Passes where the user half was over the line and held back for a record that was due.

        This row's own count, not the user half's: on these passes the user half was not asked
        at all, so none of its counters moved. Counted per pass, as ``USERUNDER`` is, so one
        wait on the live path usually reads three or four -- the copy and store passes of the
        crossing turn and the pinned call's pass -- and a wait that ended in a record reads
        beside a ``RECORDS`` that grew, where one that ran out reads beside the ``USERCOMPACT``
        that followed it. ``USERWAIT`` in the flags. See :meth:`_holds_for_record`.
        """
        return self._user_passes_waited

    @property
    def tokens_removed_by_record_phase(self) -> int:
        """Tokens the record phase has removed from the prompt over this run.

        A running total of per-pass drops in the included token count, taken straight after the
        record phase and before the user phase or the chain, so it is the record phase's own
        work: the fallback it used to run inline now runs at the end of the chain and is not in
        it. Read beside :attr:`user_messages_replaced` it says how the two halves divided the
        work. Floored at zero per pass, because the give-up fallback inserts notes where it shed
        a group and a pass can end larger than it started.

        Not a flag on the row: it is a token count rather than an event count, it moves with the
        workload rather than with the strategy, and the flags column is read for the latter.
        """
        return self._removed_by_record

    @property
    def records_merged(self) -> int:
        """Passes where the chain merged the active records into one, and kept the merge.

        Step a. Each is a summarizer call and a rewrite of the prompt from the oldest record, and
        a floor under the prompt lowered from several records to one. ``RECMERGE`` in the flags.
        """
        return self._records_merged

    @property
    def record_merges_rejected(self) -> int:
        """Merges of the records that came back no smaller than the records, and were discarded.

        Step a refused on the generic rule: the old records stayed and the chain moved on. A
        replay of a refused merge on the other list is refused again and counted again, since it
        is the same decision taken on the second view of the conversation. ``RECMERGEREJ``.
        """
        return self._record_merges_rejected

    @property
    def user_summaries_merged(self) -> int:
        """Passes where the chain folded the standing user summaries into one, and kept it.

        Step b. Also counted by the user half as a fold, which is what the machinery is.
        ``USERMERGE`` in the flags.
        """
        return self._user_summaries_merged

    @property
    def user_merges_rejected(self) -> int:
        """Folds the chain asked for that came back no smaller, and were discarded. ``USERMERGEREJ``."""
        return self._user_merges_rejected

    @property
    def record_rewrites(self) -> int:
        """Harder rewrites of the record the chain tried, kept or not. ``RECHARDER`` in the flags.

        Step c, counted per attempt, so a row reading ``RECHARDER:2`` on a pass-bounded
        ``harder_attempts`` of two used every attempt it had. A summarizer that did not answer
        still used an attempt.
        """
        return self._record_rewrites

    @property
    def record_rewrites_rejected(self) -> int:
        """Of :attr:`record_rewrites`, the ones that came back no smaller. ``RECHARDERREJ``."""
        return self._record_rewrites_rejected

    @property
    def record_summary_failures(self) -> int:
        """Record merges and rewrites where the summarizer raised or answered with nothing.

        The conversation is left as it was, exactly as on the user half's failures, and the chain
        moves on. ``RECSUMMFAIL`` in the flags; not a seed-record column, because the run-level
        summarizer failure count already carries the calls.
        """
        return self._record_summary_failures

    @property
    def last_resort_fallbacks(self) -> int:
        """Passes on which the chain reached step d and ran the record half's fallback.

        Counts the fallback being *run*, where ``RECFALLBACK`` counts it changing something and
        ``RECHELD`` counts tool groups being held from it: non-zero says everything above it in
        the chain was tried and was not enough. Beside ``DQ`` it is step e, the intended loud
        failure. ``LASTFALLBACK`` in the flags.
        """
        return self._last_resort_fallbacks

    async def __call__(self, messages: list[Message]) -> bool:
        """Run the record phase, then the user phase on what it left, then the chain if still over.

        The record phase is judged against the size the pass began with, at the record half's
        trigger. The user phase is judged against the size the record phase left, at the shared
        line, so it acts only when tool compaction was not enough -- and not at all while a record
        is due and still has time to arrive, because the record half compacts in two steps and a
        pass that only asked for one has not yet shown what tool compaction can do: see
        :meth:`_holds_for_record`. What either does once it has decided to act -- the band's
        share of the prompt, and so on -- is read off the conversation as it now stands. The
        chain then reads the live size against the input budget before each step and stops as
        soon as the prompt fits.

        Args:
            messages: The conversation, mutated in place.

        Returns:
            True if anything changed the outgoing messages. False does not imply the prompt now
            fits: after the last step of the chain nothing more is tried.
        """
        if not messages:
            return False
        self._previous_answers, self._answers = self._answers, {}
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        entry_tokens = included_token_count(messages)

        changed = await self.tool_results.compact_against(
            messages,
            prompt_tokens=entry_tokens,
            trigger_tokens=int(self.max_input_tokens * self.tool_results.trigger_fraction),
            fallback_after_record=False,
        )
        if changed:
            # Forced, because the give-up fallback the record phase may still take rewrites tool
            # results in place and the counts are cached per message: an incremental re-read
            # would judge the user phase against text that is no longer in the conversation.
            annotate_message_groups(messages)
            annotate_token_counts(messages, tokenizer=self.tokenizer, force_retokenize=True)
            self._removed_by_record += max(entry_tokens - included_token_count(messages), 0)

        prompt_tokens = included_token_count(messages)
        compacted = False
        if not self._holds_for_record(messages, entry_tokens=entry_tokens, prompt_tokens=prompt_tokens):
            compacted = await self.user_turns.compact_against(
                messages,
                prompt_tokens=prompt_tokens,
                trigger_tokens=int(self.max_input_tokens * self.user_trigger_fraction),
            )
        chained = await self._last_resort(messages)
        return changed or compacted or chained

    def _holds_for_record(self, messages: list[Message], *, entry_tokens: int, prompt_tokens: int) -> bool:
        """Return whether the user half must stay idle on this pass because a record is on its way.

        **Why there is a wait at all.** The record half compacts in two steps and the user half
        in one. On the pass where the prompt first crosses the line, the record half can only
        ask: its middleware pins a later call, the model writes the record there, and only the
        pass after that drops what the record covers. The user half, judged on that first pass,
        sees a prompt the record half has not yet touched, acts at once, and -- when summarising
        the user turns alone gets back under the line -- takes the prompt below the trigger the
        middleware reads, so the record is never asked for. Measured on gpt-5.6-luna at 200,000
        tokens, 0.9 fill and a 0.8 trigger: no record on five seeds of five and one user
        compaction on each, which is this row with its layers the wrong way round.

        **The rule.** The user half holds on a pass where the prompt, as the record phase left
        it, is over the shared line and over the record half's own trigger, and the record half
        has work pending -- :meth:`~._toolsummary.ToolResultAnchoredSummarizationCompactionStrategy.record_pending`,
        which is the middleware's own count of tool work no record covers, or an outstanding
        re-force ask. It stops holding when either

        - the record arrives -- the newest record changes, which on the pass that sees it means
          ``_drop_before`` has just applied it -- and the user half is then judged, on that pass,
          against the prompt the record left; or
        - :data:`_RECORD_WAIT_RESPONSES` responses pass without one, and the user half is judged
          as though there had been no wait. No new wait then begins behind the same newest
          record: the model has had its chance at it, and a model that never records must not
          leave the user half idling one bound in every three responses.

        With nothing pending the record half cannot help, and nothing is held; nor under the
        record half's own trigger, which a caller's explicit ``user_trigger_fraction`` can put
        above the user line, because the middleware does not ask there. A wait is one unbroken
        run of held passes: any pass that does not hold ends it.

        **Two more guards, each for a way the wait could do harm.** None at or past the record half's
        give-up line, read on the pass-entry size the record half itself judged: that line is
        how long this row may wait for a record at all, and past it the record half has stopped
        waiting and shed tool results, so the user half must not wait longer than it does. It
        also keeps every held pass under the input budget, so the chain never runs on one. And
        none for one response after a record arrives: the live path runs this over a call's
        copies and then over the store, and a user half that acted on the copies must act on the
        store too -- where it replays the same summary -- or the store keeps the band the model
        was already sent without, and the next call is sent it again.

        **State that survives the two lists.** Everything is read off the conversation -- the
        response count, the newest record's identity (:func:`_newest_record_identity`) -- or kept
        on this object, and nothing in a
        message annotation, because the copies' annotations never reach the store. A
        conversation that has gone backwards past the wait's start, which is what restoring a
        snapshot for a probe looks like, drops the wait rather than reading a negative count.

        Args:
            messages: The conversation, grouped and token-annotated.

        Keyword Args:
            entry_tokens: The prompt as the pass began, which the record half judged.
            prompt_tokens: The prompt as the record phase left it, which the user half is judged on.

        Returns:
            True to keep the user half idle on this pass.
        """
        clock = _responses(messages)
        anchor = _newest_record_identity(messages)
        if anchor != self._anchor:
            # The wait itself ends below: nothing holds within one response of an arrival.
            self._anchor = anchor
            self._quiet_through = clock + 1
        elif self._wait_since is not None and clock < self._wait_since:
            self._wait_since = None
        holding = (
            prompt_tokens > int(self.max_input_tokens * self.user_trigger_fraction)
            and prompt_tokens > int(self.max_input_tokens * self.tool_results.trigger_fraction)
            and entry_tokens < int(self.max_input_tokens * self.tool_results.fallback_fraction)
            and clock > self._quiet_through
            and anchor != self._declined_behind
            and self.tool_results.record_pending(messages)
        )
        if not holding:
            self._wait_since = None
            return False
        if self._wait_since is None:
            self._wait_since = clock
        if clock - self._wait_since >= _RECORD_WAIT_RESPONSES:
            self._wait_since = None
            self._declined_behind = anchor
            return False
        self._user_passes_waited += 1
        return True

    async def _last_resort(self, messages: list[Message]) -> bool:
        """Run the chain's steps in order, each only while the prompt is still over the budget.

        Args:
            messages: The conversation, mutated in place.

        Returns:
            True if any step changed the outgoing messages.
        """
        if not self._over(messages):
            return False
        changed = await self._merge_records(messages)
        if self._over(messages):
            changed = await self._merge_user_summaries(messages) or changed
        if self._over(messages):
            changed = await self._rewrite_records(messages) or changed
        if self._over(messages) and find_record_index(messages) is not None:
            # Only behind a record. With none, the record phase has already taken its give-up
            # fallback on this pass -- a prompt over the budget is over the give-up line -- and
            # that path is not the chain's to repeat.
            self._last_resort_fallbacks += 1
            changed = await self.tool_results.fall_back_after_record(messages) or changed
        return changed

    def _over(self, messages: list[Message]) -> bool:
        """Return whether the prompt, as it now stands, is over the input budget."""
        annotate_message_groups(messages)
        annotate_token_counts(messages, tokenizer=self.tokenizer)
        return included_token_count(messages) > self.max_input_tokens

    async def _merge_records(self, messages: list[Message]) -> bool:
        """Step a: merge the active records into one, when there are at least two."""
        groups = active_record_groups(messages)
        if len(groups) < 2:
            return False
        outcome = await self._consolidate(messages, groups, prompt=self.merge_prompt)
        if outcome == "accepted":
            self._records_merged += 1
            return True
        if outcome == "rejected":
            self._record_merges_rejected += 1
        return False

    async def _merge_user_summaries(self, messages: list[Message]) -> bool:
        """Step b: fold the standing user summaries into one, when there are at least two."""
        outcome = await self.user_turns.fold_if_smaller(messages)
        if outcome == "folded":
            self._user_summaries_merged += 1
            return True
        if outcome == "rejected":
            self._user_merges_rejected += 1
        return False

    async def _rewrite_records(self, messages: list[Message]) -> bool:
        """Step c: rewrite the record harder, up to :attr:`harder_attempts` times on this pass.

        Every active record is handed over, which is one record whenever step a succeeded or
        there was only one, and several when a merge was refused -- in which case a rewrite is a
        harder merge, and the prompt says so. Each attempt is judged against the records as they
        stand when it is made, so a kept attempt raises the bar for the next.
        """
        changed = False
        for attempt in range(1, self.harder_attempts + 1):
            if attempt > 1 and not self._over(messages):
                break
            groups = active_record_groups(messages)
            if not groups:
                break
            self._record_rewrites += 1
            outcome = await self._consolidate(messages, groups, prompt=harder_record_prompt(attempt))
            if outcome == "accepted":
                changed = True
            elif outcome == "rejected":
                self._record_rewrites_rejected += 1
        return changed

    async def _consolidate(
        self, messages: list[Message], groups: list[dict[str, Any]], *, prompt: str
    ) -> _Consolidation:
        """Ask for one record in place of ``groups``, and put it there if it is smaller.

        The acceptance rule is the module docstring's, and nothing else: non-empty -- a
        summarizer that answers with nothing is a failure -- and fewer tokens than the records it
        replaces. Both sides are measured in the form the replacement is inserted in: the
        candidate as built, and each replaced record rebuilt from its body the same way, rather
        than the replaced messages themselves. A record the model made carries its text twice,
        once in the call's arguments, and a candidate measured against that would come out at
        about half the size with its text unchanged -- see
        :func:`~._toolsummary.build_record_message`.

        Args:
            messages: The conversation, mutated in place when the replacement is kept.
            groups: The active records, oldest first.

        Keyword Args:
            prompt: What the summarizer is asked.

        Returns:
            Whether the replacement was kept, refused as no smaller, or never arrived.
        """
        transcript = "\n".join(
            f"{number}. {record_body(messages, group)}" for number, group in enumerate(groups, start=1)
        )
        text = await self._ask(prompt=prompt, transcript=transcript)
        if text is None:
            return "failed"
        candidate = [build_record_message(text)]
        replaced = [build_record_message(record_body(messages, group)) for group in groups]
        annotate_token_counts(candidate, tokenizer=self.tokenizer, force_retokenize=True)
        annotate_token_counts(replaced, tokenizer=self.tokenizer, force_retokenize=True)
        if included_token_count(candidate) >= included_token_count(replaced):
            return "rejected"
        self.tool_results.consolidate_records(messages, groups, text)
        return "accepted"

    async def _ask(self, *, prompt: str, transcript: str) -> str | None:
        """Return the summarizer's answer, asking only if the request is new.

        Remembered for this pass and the next, so the second list the live path runs on replays
        the text the first list was given -- the model is sent one merged record, the store holds
        the same one, and :func:`_newest_record_identity` reads the same name off both -- and a
        request refused as no smaller is not paid for again while it keeps being asked. A failure
        is not remembered, for the reason the user half gives: the next view should ask again.

        Keyword Args:
            prompt: The system prompt.
            transcript: The numbered records.

        Returns:
            The text, or None when the summarizer raised or said nothing.
        """
        key = (prompt, transcript)
        remembered = self._answers.get(key) or self._previous_answers.get(key)
        if remembered is not None:
            self._answers[key] = remembered
            return remembered
        try:
            response = await self.user_turns.client.get_response(
                [Message(role="system", contents=[prompt]), Message(role="user", contents=[transcript])],
                stream=False,
            )
        except Exception as error:
            # Broad on purpose, as the user half's is: this runs inside the chat client's own
            # call path, so anything not caught here fails the user's turn.
            logger.warning("Skipping record consolidation: summary generation failed (%s).", error)
            self._record_summary_failures += 1
            return None
        text = response.text.strip() if response.text else ""
        if not text:
            logger.warning("Skipping record consolidation: the summarizer returned no text.")
            self._record_summary_failures += 1
            return None
        self._answers[key] = text
        return text
