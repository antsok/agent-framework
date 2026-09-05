# Copyright (c) Microsoft. All rights reserved.

"""Per-seed results, written when they exist rather than when the cell ends.

A cell is five strategies times ``--repeats`` seeds and runs for hours, and its table was
printed only once every one of them had finished. So anything that stopped the process in
between threw away every seed that had already completed and already been paid for. That is
not hypothetical: the 60,000/0.86 cell ran all fifteen strategy-seeds over three and a half
hours, died before printing, and left nothing at all behind.

A seed's result is therefore appended here the moment it is scored. The record carries the
scored numbers rather than a reference to the objects that produced them, because those
objects are what the run cannot keep: the scenario is salted per seed and dies with the
process, and re-scoring later would need it. What is stored is what the table reads, so a
table rebuilt from the file is the same aggregation over the same inputs as the live one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from statistics import fmean
from typing import TYPE_CHECKING, Any, Final

from ._advisor import ModelPricing
from ._fill import FillPlan
from ._summary import DEFAULT_MIN_CORRECTNESS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "SCHEMA_VERSION",
    "CellParams",
    "SeedRecord",
    "append_seed_record",
    "group_by_cell",
    "read_seed_records",
]

#: Format of the records this module writes.
#:
#: Bumped whenever a field changes meaning, not merely whenever one is added: the point is
#: that a reader can tell a record describing the current measurement from one describing an
#: older one. Runs before this file existed have no records at all, and runs from before the
#: seed/snapshot/probe rebuild measured `survived` against a different prompt, so mixing their
#: numbers into one table would produce a mean over two different questions.
#:
#: 2 adds the throttling counters. Additive, and bumped anyway: a version 1 record cannot say
#: whether it was throttled or merely never asked, and reading its absent counters as zero
#: would put "not measured" and "did not happen" in the same column.
#:
#: ``combined_repeats`` was added later without a bump, which is the same rule applied to the
#: opposite case: what a version 2 record did is not in doubt. The combined question was one
#: of the closing questions then, so it was asked exactly ``probe_repeats`` times, and
#: :meth:`CellParams.from_dict` fills that in. There is no "not measured" to confuse with a
#: measurement, and the samples on the record say the same thing by their count.
#:
#: 3 adds the connection-retry counters, and this time the version is what separates a record
#: whose run could re-send a dropped call from one whose run could not.
#:
#: 4 adds the probe phase's own token counts, which split ``cost`` into the workload and the
#: instrument. Bumped for the version 2 reason and not the ``combined_repeats`` one: what a
#: version 3 record spent on probing is not merely absent, it is unknowable from the record,
#: and reading the absent counts as zero would report every one of those runs as having probed
#: for free -- which would put the whole of a twelve-probe discount into the seeding half and
#: make the correction this field exists for read as already applied.
#:
#: 5 adds ``groups_kept_uncovered``, and the version is what separates a run whose strategy
#: could decline to drop an unrecorded tool group from one whose strategy could not. Before
#: it, ``tool_summary_anchored`` dropped every tool group in front of the record on the
#: assumption that the record had replaced them, so no group was ever kept for want of
#: coverage and zero is what those runs did. That is the version 3 case exactly, not the
#: version 4 one: the number is knowable and it is zero, rather than unknowable and reported
#: as zero. The bump is there so a reader can tell which of the two zeroes a row is showing.
#:
#: 5 also adds ``fallbacks_after_record``, and deliberately without a further bump. There is no
#: version 5 record for a 6 to date it against: the bump above is unreleased, and every recorded
#: cell on disk is version 2, 3 or 4. What separates the two fields is not the version but what
#: their absence means, and the values say that themselves -- ``groups_kept_uncovered`` reads
#: back as 0, which is what those runs did, and ``fallbacks_after_record`` reads back as
#: ``None``, because those runs *could* fall back behind a record and counted nothing when they
#: did. That is the version 4 case: not absent but unknowable, and a 0 would tell a reader that
#: every older row stayed the strategy it is named for.
SCHEMA_VERSION: Final[int] = 5

#: Versions this reader accepts, which is not only the current one.
#:
#: Version 2 is readable because its two absent counters are a measurement rather than a gap: a
#: version 2 record was written by code that failed the turn on a connection error instead of
#: re-sending it, so zero re-sends is what happened, and :meth:`SeedRecord.from_dict` fills it
#: in as such. That is the opposite of the version 1 case, which is still refused -- those
#: records predate the seed/snapshot/probe rebuild and scored ``survived`` against a different
#: prompt, so their accuracy columns are answers to another question.
#:
#: Refusing version 2 instead would have thrown away the six recorded cells on disk, 180 seeds
#: of paid-for measurement, to avoid a column of zeroes that are true.
#:
#: Versions 2 and 3 are readable in the same spirit, and their probe counts come back as
#: ``None`` rather than as zero. Everything those records measured they still measure; the one
#: thing they cannot say is how their cost divided between seeding and probing, and ``None`` is
#: how a reader is told that instead of being handed a number nobody took.
#:
#: Version 4 joins them on the version 2 argument rather than the version 3 one: what its runs
#: did about uncovered tool groups is not in doubt, because their code had no way to keep one.
#:
#: What none of the three can say is how often a fallback ran *behind* a record, so
#: ``fallbacks_after_record`` comes back as ``None`` for all of them, on the probe-count
#: argument rather than the retry one: that path existed in every one of those runs and
#: incremented nothing when it was taken.
_READABLE_SCHEMAS: Final[frozenset[int]] = frozenset({2, 3, 4, SCHEMA_VERSION})

#: The parameters that make two records the same cell, and so aggregable into one row.
#:
#: Deliberately excludes ``strategies`` and ``repeats``, which say what a run *intended* to
#: measure rather than what it measured: a cell abandoned after three strategies and resumed
#: for the other two is one cell, and has to aggregate as one. Includes the prices, because
#: two runs priced differently produce costs that cannot go in one column.
#:
#: ``tool_share`` is here although ``tool_result_tokens`` already carries what it derived, and
#: so already separates two workloads that differ: two shares reaching one per-result size
#: would have to build the same conversation. It is kept for the reason ``fill`` is, which is
#: likewise nearly implied by ``filler_turns`` and ``filler_tokens`` -- the key states the
#: parameter that was set and not only the number it produced, so a file holding a sweep can
#: be read back against the sweep's own axes.
_CELL_KEY_FIELDS: Final[tuple[str, ...]] = (
    "provider",
    "model",
    "agent_kind",
    "context_window",
    "fill",
    "probe_repeats",
    "combined_repeats",
    "narration",
    "fact_placement",
    "tool_result_tokens",
    "tool_share",
    "filler_turns",
    "filler_tokens",
    "tool_turns",
    "filler_tool_turns",
    "markers_per_tool",
    "price_input",
    "price_cached",
    "price_output",
)


def _plan_from_dict(data: Mapping[str, Any]) -> FillPlan:
    """Rebuild the solved sizing from its serialized form.

    Field by field rather than by splatting the mapping, so a file carrying a key this version
    does not know about is refused here instead of raising somewhere less obvious.

    Args:
        data: The ``plan`` object from a record's cell parameters.

    Returns:
        The plan.
    """
    return FillPlan(
        filler_turns=int(data["filler_turns"]),
        filler_tokens=int(data["filler_tokens"]),
        context_limit=int(data["context_limit"]),
        fill_fraction=float(data["fill_fraction"]),
        target_tokens=int(data["target_tokens"]),
        predicted_tokens=int(data["predicted_tokens"]),
        payload_tokens=int(data["payload_tokens"]),
        tool_payload_tokens=int(data["tool_payload_tokens"]),
        # The two sizing fields are read leniently where the rest are not, because a plan
        # written before the payload could be derived has neither and its run is not in doubt:
        # the size was stated, which is what a share of 0 means, and it is on the cell beside
        # this plan. Demanding them here would refuse every record already on disk.
        tool_result_tokens=int(data.get("tool_result_tokens", 0)),
        tool_share=float(data.get("tool_share", 0.0)),
    )


@dataclass(frozen=True, slots=True)
class CellParams:
    """What was being measured, carried on every record rather than written once as a header.

    A header is a thing a partial file can be missing, and a file that needs one to be read is
    back to losing the seeds a crash left behind. Repeating the parameters per line also keeps
    two runs appended to one path separable, since the cell is read off the line rather than
    off the file.
    """

    provider: str
    model: str
    agent_kind: str
    context_window: int
    """The limit the cell stands in for, which is simulated and enforced in our own code."""
    fill: float
    """Share of that limit the seeded conversation was sized to reach, 0 for manual sizing."""
    probe_repeats: int
    repeats: int
    """Seeds per strategy the run set out to take.

    Stored so that a short cell reads as unfinished rather than as a cell whose strategies
    happened to disagree. Without it, three seeds of a five-seed cell are indistinguishable
    from a complete three-seed one, and every spread in the table is a spread over less
    material than it claims.
    """
    strategies: tuple[str, ...]
    """The strategies the run set out to measure, for the same reason."""
    narration: str
    fact_placement: str
    tool_result_tokens: int
    """Size each tool result was built to, whether stated or derived from ``tool_share``.

    What the run did, not what it was asked for, so a table rebuilt from the file describes
    the workload that was measured however it came to be chosen.
    """
    filler_turns: int
    filler_tokens: int
    tool_turns: int
    filler_tool_turns: int
    markers_per_tool: int
    price_input: float
    price_cached: float
    price_output: float
    combined_repeats: int = 1
    """Times the combined question was asked per seed: the denominator of ``acc2``.

    Part of what makes two records the same cell, for the same reason ``probe_repeats`` is:
    an ``acc2`` averaged over three attempts per seed and one averaged over a single attempt
    are readings of different precision and do not belong in one column.

    Defaulted to 1 for a record written before the combined question had its own count, where
    it was one of the closing questions and so asked ``probe_repeats`` times --
    :meth:`from_dict` fills that in, so an old file reads as the measurement it was.
    """
    tool_share: float = 0.0
    """Share of the fill target the tool results were sized to reach, 0 when the size was stated.

    The request beside its result: ``tool_result_tokens`` above is what this derived. Both are
    recorded because neither reconstructs the other -- the size cannot say what fraction of the
    context it was meant to be, and the share cannot be turned back into a size without the
    window, the fill and the tool-group count.

    Defaulted to 0 for a record written before the payload could be derived, where the size was
    stated outright and 0 is what that means.
    """
    min_correctness: float = DEFAULT_MIN_CORRECTNESS
    """The correctness bar the verdict applied.

    Recorded because it is the one input to the verdict that is not otherwise on the record: a
    table rebuilt under a different bar would rank rows the original never ranked while every
    column above the verdict stayed identical, which is the hardest kind of disagreement to see.
    """
    plan: FillPlan | None = None
    """The solved sizing, when ``--fill`` was used.

    Kept whole rather than reduced to its target, so the achieved fill can be checked against
    it from the file exactly as the live run checks it.
    """

    @property
    def key(self) -> tuple[Any, ...]:
        """Return what identifies this cell, for grouping records that belong in one table."""
        return tuple(getattr(self, name) for name in _CELL_KEY_FIELDS)

    @property
    def pricing(self) -> ModelPricing:
        """Return the rates the costs on these records were computed at."""
        return ModelPricing(
            input_per_million=self.price_input,
            cached_read_per_million=self.price_cached,
            output_per_million=self.price_output,
        )

    @property
    def label(self) -> str:
        """Return a one-line description of the cell, for a file holding more than one."""
        fill = f"fill {self.fill:.0%}" if self.fill > 0 else "fill manual"
        # The share, when there was one, sits beside the size rather than replacing it: the
        # size is what the conversation carried and the share is why it was that size, and a
        # sweep across window sizes needs both to be readable as one axis.
        payload = f"payload {self.tool_result_tokens:,}x{self.tool_turns}"
        if self.tool_share > 0:
            payload += f" at share {self.tool_share:.0%}"
        return (
            f"{self.provider}:{self.model}  agent {self.agent_kind}  "
            f"window {self.context_window:,}  {fill}  {payload}  "
            f"probes {self.probe_repeats}  combined {self.combined_repeats}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of these parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CellParams:
        """Rebuild parameters from a mapping read back out of a record.

        Args:
            data: One record's ``cell`` object.

        Returns:
            The parameters.
        """
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        values["strategies"] = tuple(values.get("strategies") or ())
        # A record written before the combined question had its own count asked it as one of
        # the closing questions, so it was asked exactly probe_repeats times. Taking the
        # field's own default of 1 instead would report a three-attempt cell as a single
        # attempt, and taking today's default of 3 would report one attempt as three that
        # mostly failed. Neither is what the file says; this is.
        values.setdefault("combined_repeats", int(values.get("probe_repeats", 1)))
        plan: dict[str, Any] | None = values.get("plan")
        values["plan"] = _plan_from_dict(plan) if plan is not None else None
        return cls(**values)


#: Fields stored as JSON arrays, which come back as lists and have to be re-tupled.
#: A list here would compare unequal to the tuple the live path produces, which is exactly
#: the kind of difference that makes "the file reproduces the table" untestable.
_TUPLE_FIELDS: Final[tuple[str, ...]] = (
    "correctness_samples",
    "ignored_samples",
    "combined_samples",
    "strategy_notes",
    "dropped_options",
)


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """One seed of one strategy, reduced to everything a table needs and nothing else.

    Not a serialized ``LiveOutcome``. The outcome carries every prompt of every call, which is
    the bulk of a run and none of what the table reads, and it is unscored -- rebuilding a
    table from it would mean re-scoring against a scenario whose markers are salted per seed
    and which is gone the moment the process is. Scoring happens once, here, and both the live
    table and one rebuilt months later aggregate the result of it.
    """

    cell: CellParams
    strategy: str
    seed: int
    """1-based index of this seed within its strategy."""
    cost: float
    """The whole seed: seeding, every probe, and the strategy's own summarizer calls.

    What the run was billed, which is not what a strategy is ranked on: see
    :attr:`seeding_cost` for the half of it that is the workload.
    """
    summarizer_cost: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    probe_input_tokens: int | None
    """Input tokens the probe phase billed, of ``input_tokens``.

    The instrument's share. Every probe re-sends the whole snapshot, so a strategy with a small
    snapshot is discounted once per probe on a phase no deployed agent has: an agent continues
    the conversation, it is not interrogated twelve times from a frozen state. Recorded so the
    ranking can be taken on the other half.

    ``None`` on a record written before version 4, where the phases were never counted apart
    and no arithmetic over the stored fields can separate them: the per-probe prompts are not
    on the record, and pricing twelve of them at ``prompt_tokens_final`` is a model of the run
    rather than the run. Those records rank on their whole cost and say so.
    """
    probe_cached_tokens: int | None
    """How many of those the provider served from cache, or ``None`` before version 4."""
    probe_output_tokens: int | None
    """Output tokens the probe phase billed, or ``None`` before version 4."""
    calls: int
    messages_left: int
    messages_peak: int
    prompt_tokens_final: int
    prompt_tokens_peak: int
    seed_prompt_tokens: int
    """Billed size of the last seeding prompt: the achieved fill."""
    facts_total: int
    facts_left: int
    """Planted facts that survived compaction into the snapshot, which is recall's ceiling."""
    facts_lost: int
    """Facts compaction removed: the total, less what survived, less what was never fetched.

    Stored per seed although the table derives its own ``lost`` column from the cell's means.
    The two agree whenever every seed planted the same number of facts, which is every cell
    this package builds; they can differ only if a seed's scenario had a different size.
    """
    nofetch: int
    """Facts the agent never fetched, so compaction never had them to lose."""
    correctness_samples: tuple[float, ...]
    """One reading per probe repeat, each scored against the same snapshot.

    A tuple rather than a mean, because the mean of the seed and the spread within it are
    different findings and the cell needs both. Flattened across seeds, these are also what
    the verdict ranks on.
    """
    ignored_samples: tuple[int, ...]
    """Facts still in the snapshot the model did not use, per probe repeat."""
    combined_samples: tuple[float, ...]
    """Share of all planted values present in the single combined answer, per repeat."""
    disqualified: bool
    """Whether any call sent a prompt larger than the limit this cell stands in for."""
    context_drift: int
    rate_limit_retries: int
    """Calls this seed re-sent after the provider refused them for rate reasons."""
    throttled_seconds: float
    """Seconds this seed spent waiting those refusals out.

    Recorded beside the count because the two answer different questions. The count says the
    limit was met; this says how much of the seed's wall clock went into meeting it, which is
    what makes a cache hit rate comparable with a cell that never waited at all.
    """
    connection_retries: int
    """Calls this seed re-sent because the request never came back with an answer.

    Its own field rather than added to the throttled count, because the reader's next question
    differs. A throttled seed waited out a quota window, which is wide enough that the prompt
    cache may have gone with it; a reconnected seed waited seconds and re-sent the same prefix.
    Summed into one column, every reconnected row would carry a caveat that belongs to the
    other failure.
    """
    connection_seconds: float
    """Seconds this seed spent waiting for the provider to answer again."""
    turns_completed: int
    turns_total: int
    probe_repeats: int
    summarizer_calls: int
    summarizer_failures: int
    groups_kept_uncovered: int
    """Tool groups the recall record never mentioned, so the strategy kept them instead of dropping them.

    The cost of a partial record, in the one unit that says how partial it was. A row carrying
    this spent tokens it should not have had to spend and lost nothing, which is the trade
    ``tool_summary_anchored`` makes on purpose; the alternative it declined was deleting tool
    results nothing had preserved and reporting that as compaction damage in the accuracy
    columns, where no reader could trace it back.

    Stored as a number beside the ``UNCOVERED:<n>`` flag on ``strategy_notes`` because the two
    are read for different things. The flag says a row is affected; this can be meaned across
    the seeds of a cell, which is what answers how much the coverage check is holding back.

    Zero for every strategy that keeps no such count, and zero on a record written before
    schema 5 -- see :data:`SCHEMA_VERSION` for why that zero is a measurement rather than a gap.
    """
    fallbacks_after_record: int | None
    """Compaction passes where a record existed and the strategy fell back anyway.

    The other way ``tool_summary_anchored`` stops being itself, and the quieter one: the
    fallback shortens the tool results a partial record left in the prompt, so the row keeps
    its messages and loses its values. ``strategy_notes`` carries it as the ``RECFALLBACK:<n>``
    flag; the number is here so a cell can be meaned on it and asked how often a record failed
    to free enough, rather than only whether one ever did.

    ``None`` on a record written before the counter existed, which is every readable schema
    below 5. Not zero: that path ran in those runs and incremented nothing, so the count is
    unknowable rather than known to be nought -- the probe-token case, not the retry-counter
    one. Zero from a live run means what it says, including on the four strategies that take no
    record at all.
    """
    strategy_notes: tuple[str, ...]
    dropped_options: tuple[str, ...]
    answer: str
    error: str | None = None
    schema: int = SCHEMA_VERSION

    @property
    def correctness(self) -> float:
        """Mean correctness over this seed's probe repeats: the seed's ``acc1``."""
        return fmean(self.correctness_samples) if self.correctness_samples else 0.0

    @property
    def combined(self) -> float:
        """Mean over this seed's combined attempts: the seed's ``acc2``.

        Over the attempts the seed actually made, so a record written when the combined
        question was asked once reads as that one answer rather than as a third of three.
        """
        return fmean(self.combined_samples) if self.combined_samples else 0.0

    @property
    def hit_rate(self) -> float | None:
        """Share of input tokens served from the provider's cache, None when nothing was billed."""
        return self.cached_tokens / self.input_tokens if self.input_tokens > 0 else None

    @property
    def input_cost(self) -> float:
        """What this seed's prompt side cost: uncached plus cached, with output left out.

        Derived rather than stored, so no record already on disk is missing it and the schema
        does not move: the tokens and the rates are both here, and the arithmetic is the same
        one ``cost`` uses for its input half.
        """
        return self.cell.pricing.input_cost(self.input_tokens, self.cached_tokens)

    @property
    def probe_cost(self) -> float | None:
        """What the probing cost: the instrument, and the half a deployed agent never pays.

        Returns:
            The cost, or None when this record was written before the phases were counted
            apart, where no honest number exists to return.
        """
        if self.probe_input_tokens is None or self.probe_cached_tokens is None or self.probe_output_tokens is None:
            return None
        pricing = self.cell.pricing
        return (
            pricing.input_cost(self.probe_input_tokens, self.probe_cached_tokens)
            + self.probe_output_tokens * pricing.output_per_million / 1_000_000
        )

    @property
    def seeding_cost(self) -> float | None:
        """What the conversation cost: the workload, and the number a strategy is judged on.

        Taken as the whole seed less the probing rather than summed over the seeding calls, so
        that the two halves add back to ``cost`` exactly whatever else the run did. Anything
        that is neither an answered probe nor a seeding turn -- the summarizer's own calls, and
        a probe that failed before it produced an outcome -- therefore lands here. The
        summarizer belongs here on its merits: it runs while the conversation is being had, and
        a deployed agent pays for it.

        Returns:
            The cost, or None when the phases were never counted apart.
        """
        probing = self.probe_cost
        return None if probing is None else self.cost - probing

    @property
    def seeding_input_cost(self) -> float | None:
        """What the conversation's prompt side cost, output and the probes both left out.

        The quiet reading of the workload, for the reason :attr:`input_cost` is the quiet
        reading of the whole seed: output is priced many times a cache read here, so a reply
        the model ran long on moves a total further than compaction does.

        Returns:
            The cost, or None when the phases were never counted apart.
        """
        if self.probe_input_tokens is None or self.probe_cached_tokens is None:
            return None
        return self.cell.pricing.input_cost(
            max(self.input_tokens - self.probe_input_tokens, 0),
            max(self.cached_tokens - self.probe_cached_tokens, 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedRecord:
        """Rebuild a record from one parsed line.

        Args:
            data: The parsed JSON object.

        Returns:
            The record.

        Raises:
            ValueError: If the record was written by a different schema version, or is missing
                fields this one requires.
        """
        version = data.get("schema")
        if version not in _READABLE_SCHEMAS:
            readable = ", ".join(str(number) for number in sorted(_READABLE_SCHEMAS))
            raise ValueError(
                f"schema {version!r}, but this reader understands {readable}. Records from "
                "another version describe a different measurement and must not be averaged with these."
            )
        known = {field.name for field in fields(cls)} - {"cell"}
        values = {key: value for key, value in data.items() if key in known}
        for name in _TUPLE_FIELDS:
            values[name] = tuple(values.get(name) or ())
        # A version 2 record ran under code that had no connection retry at all: a dropped call
        # failed its turn on the spot. So nothing was re-sent and nothing was waited, and zero
        # is what that run did rather than a number this reader could not find. Left as a
        # required field with no default, so the live path cannot forget to record it and
        # quietly inherit the same zeroes for a run that could have re-sent.
        values.setdefault("connection_retries", 0)
        values.setdefault("connection_seconds", 0.0)
        # None, not zero, and for the opposite reason to the two above. Those record something
        # the run could not do; these record something the run did and did not count, and every
        # one of those runs probed twelve times. Zero here would hand a reader the exact
        # correction this field exists to make, already applied, and wrong.
        for name in ("probe_input_tokens", "probe_cached_tokens", "probe_output_tokens"):
            values.setdefault(name, None)
        # Zero, and for the connection-retry reason rather than the probe-token one. A record
        # written before schema 5 ran under a strategy that dropped every tool group in front
        # of the record without checking what the record covered, so no group was kept for want
        # of coverage: zero groups is what that run did, not a number this reader could not
        # find. Required with no default on the class for the same reason as the retry
        # counters, so a live run cannot silently inherit the same zero.
        values.setdefault("groups_kept_uncovered", 0)
        # None, and for the probe-token reason rather than the one directly above. Every run
        # since this strategy existed could fall back behind a record that had not freed
        # enough; none of them counted it, so what an older record did on that path cannot be
        # recovered from the record. Zero would say the row stayed the strategy it is named
        # for, which is the exact claim the counter was added because nobody could make.
        values.setdefault("fallbacks_after_record", None)
        try:
            return cls(cell=CellParams.from_dict(data["cell"]), **values)
        except (KeyError, TypeError) as error:
            raise ValueError(f"record is not readable: {error}") from error


def append_seed_record(path: Path, record: SeedRecord) -> None:
    """Append one seed's record to a JSON Lines file, closing the file again immediately.

    Opened and closed per record rather than held open across the cell. A handle held open
    buffers, and the record sitting in that buffer is exactly the one an interruption takes --
    which is the whole failure this file exists to prevent. Appending rather than truncating is
    load-bearing for the same reason: a run resumed after a crash has to extend the file, and
    two cells written to one path have to both survive.

    Args:
        path: Destination file. Parent directories are created as needed.
        record: The seed to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_seed_records(path: Path) -> tuple[SeedRecord, ...]:
    """Read every record in a JSON Lines file.

    Args:
        path: The file to read.

    Returns:
        The records, in the order they were written.

    Raises:
        ValueError: If any line is not a record this reader understands. Refused rather than
            skipped: a line that cannot be read is a seed that was paid for, and dropping it
            silently would leave a mean over fewer seeds than the table claims. A run killed
            mid-write can leave the last line half-finished, and that one line is safe to
            delete by hand.
    """
    records: list[SeedRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(SeedRecord.from_dict(json.loads(line)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}, line {number}: {error}") from error
    return tuple(records)


def group_by_cell(records: Sequence[SeedRecord]) -> list[tuple[CellParams, tuple[SeedRecord, ...]]]:
    """Group records into the cells they were measured in.

    In first-appearance order rather than sorted, so a file written by a sweep reads back in
    the order the sweep ran.

    Args:
        records: Records from one or more cells.

    Returns:
        One entry per cell: its parameters, taken from the first record that named it, and
        every record belonging to it.
    """
    grouped: dict[tuple[Any, ...], list[SeedRecord]] = {}
    params: dict[tuple[Any, ...], CellParams] = {}
    for record in records:
        key = record.cell.key
        params.setdefault(key, record.cell)
        grouped.setdefault(key, []).append(record)
    return [(params[key], tuple(entries)) for key, entries in grouped.items()]
