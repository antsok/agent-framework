# Copyright (c) Microsoft. All rights reserved.

"""Command line entry point for the live-agent compaction comparison."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Final, cast

from ._advisor import ModelPricing, fetch_openrouter_pricing
from ._fill import FillPlan, plan_fill
from ._live import (
    AGENT_KINDS,
    DEFAULT_PROBE_REPEATS,
    DEFAULT_TOOL_RESULT_TOKENS,
    LiveOutcome,
    MeteredClient,
    build_live_scenario,
    run_live,
    score_combined_samples,
    score_samples,
    unretrieved_facts,
    wants_client_side_history,
)
from ._providers import build_provider, parse_provider_selector, provider_names
from ._recall import RecallScenario, RecallScore
from ._records import CellParams, SeedRecord, append_seed_record, group_by_cell, read_seed_records
from ._strategies import StrategyOptions, build_strategy, needs_summarizer, strategy_names
from ._summary import DEFAULT_MIN_CORRECTNESS, JointOutcome, JointVerdict, recommend
from ._tokenizers import TOKENIZER_NAMES, build_tokenizer

if TYPE_CHECKING:
    from agent_framework._clients import SupportsChatGetResponse

__all__ = ["build_parser", "main", "run_live_comparison"]

#: Every strategy that needs no extra client, in a deliberate order: the control, then the
#: single-mechanism strategies, then the composed family that all share one token ceiling.
_DEFAULT_STRATEGIES = (
    "none,truncation,sliding_window,tool_result,selective_tool_call,"
    "context_window,context_window_aggressive,"
    "token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first"
)

#: How far the achieved fill may sit from the target before the cell stops being the cell it
#: claims to be. The one term the analytic sizing cannot compute is the model's own replies,
#: so some deviation is expected; beyond this the fill fraction is no longer the variable it
#: is being read as.
FILL_TOLERANCE: Final[float] = 0.05


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        A parser for the live comparison.
    """
    parser = argparse.ArgumentParser(
        prog="cachebench-live",
        description=(
            "Compare compaction strategies against a real agent that generates its own replies "
            "and calls a real tool. The conversation is seeded, snapshotted, and then probed: "
            "every closing question is asked from the snapshot rather than appended to the "
            "conversation, so no answer contaminates another. Within-model only: real replies "
            "differ per model, so these numbers do not compare across models."
        ),
    )
    parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider or provider:model. Omitted only with --from-jsonl, which runs nothing.",
    )
    parser.add_argument("--strategies", default=_DEFAULT_STRATEGIES, help=f"Available: {','.join(strategy_names())}")
    parser.add_argument("--agent", default="plain", choices=list(AGENT_KINDS), help="How to assemble the agent.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Seeds per strategy: whole conversations, driven from scratch. This is the axis "
            "that measures compaction's own reliability, since a different seed puts the facts "
            "in a different place relative to a retention boundary. 3 or more is what makes a "
            "ranking defensible."
        ),
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Number the seeds from here instead of 1. The seed number goes into the scenario "
            "salt, so two invocations of one cell that both start at seed 1 build byte-identical "
            "conversations -- the salt's other term is a whole-second timestamp, which concurrent "
            "processes share. Offsetting is what makes several single-seed invocations of the "
            "same cell into different seeds rather than one seed measured repeatedly, which is "
            "the difference between measuring compaction's reliability and not."
        ),
    )
    parser.add_argument(
        "--probe-repeats",
        type=int,
        default=DEFAULT_PROBE_REPEATS,
        help=(
            "Times each closing question is asked of the same snapshot. The facts and their "
            "positions are identical across these, so whatever they disagree about is the "
            "model's own willingness to enumerate rather than anything compaction did. "
            "Default %(default)s."
        ),
    )
    parser.add_argument(
        "--fill",
        type=float,
        default=0.70,
        help=(
            "Share of --context-window the seeded conversation is sized to reach, measured on "
            "an uncompacted run. Solved analytically from the payload and filler sizes, so the "
            "user-side turn list is identical across strategies without having to run one "
            "first. The filler is the dial and the payload is held fixed, which is what makes "
            "this 'how much irrelevant context surrounds a fixed set of facts'. Pass 0 to size "
            "manually from --filler-turns and --filler-tokens instead. Default %(default)s."
        ),
    )
    parser.add_argument(
        "--filler-turns",
        type=int,
        default=6,
        help=(
            "Padding turns between planted facts. Ignored unless --fill is 0, where the sizing "
            "is manual: with a fill fraction the count is solved for."
        ),
    )
    parser.add_argument(
        "--filler-tokens",
        type=int,
        default=2_000,
        help=(
            "Size of each filler turn. Under --fill this is the size the solver aims to keep "
            "them near while it picks how many there are, so that a long conversation is many "
            "ordinary turns rather than a handful of implausibly large ones."
        ),
    )
    parser.add_argument(
        "--tool-result-tokens",
        type=int,
        default=DEFAULT_TOOL_RESULT_TOKENS,
        help=(
            "Approximate size of each tool result, in tokens. Part of the payload, which is a "
            "run-level parameter: vary it between runs and compare across them, never inside "
            "one matrix, or the fill fraction stops meaning what it says."
        ),
    )
    parser.add_argument(
        "--narration",
        default="neutral",
        choices=["prompted", "neutral", "suppressed"],
        help=(
            "How hard the scenario pushes the model to restate tool values. 'neutral' says "
            "nothing either way, leaving the framework's own guidance as the only driver -- "
            "the configuration a typical caller gets. Default neutral."
        ),
    )
    parser.add_argument(
        "--fact-placement",
        default="spread",
        choices=["spread", "buried", "head"],
        help=(
            "Where the verifiable codes sit inside each tool result, which decides what is "
            "being measured. 'spread' puts each on its own labelled line, so the score is how "
            "much compaction preserved. 'buried' puts them inline in prose, so the score is "
            "retrieval under noise as well -- a real property, and one where compaction can "
            "score above the uncompacted control by deleting the haystack. 'head' puts them "
            "all at the front, inside the 4,096 characters a collapsed tool result keeps, so "
            "every tool-oriented strategy preserves them for free; it reproduces runs 7 to 9."
        ),
    )
    parser.add_argument(
        "--no-retrieval-guidance",
        action="store_true",
        help=(
            "Drop the clause telling the model to quote every identifier it is asked for. "
            "Measures how much of the closing answer is the model's willingness to enumerate "
            "rather than what compaction left behind. Off by default. Note that dropping it "
            "is only safe with an adequate --answer-max-tokens: at 900 the control scored "
            "33%% without the clause and 100%% with it, which measures the cap, not retrieval."
        ),
    )
    parser.add_argument(
        "--sweeping-question",
        action="store_true",
        help=(
            "Close with one question demanding every code at once, instead of several "
            "targeted ones. Needs a large --answer-max-tokens: enumerating 53 codes is "
            "~640 tokens before prose, and a truncated answer is scored as lost facts."
        ),
    )
    parser.add_argument(
        "--markers-per-tool",
        type=int,
        default=2,
        help=(
            "Verifiable codes each tool result carries. Two is easy for a model to echo into "
            "its reply, which lets narration preserve what a strategy discards. More codes "
            "raise the resolution of the accuracy measure and make narration a weaker substitute."
        ),
    )
    parser.add_argument(
        "--filler-tool-turns",
        type=int,
        default=0,
        help=(
            "Extra tool calls whose results carry no codes. Adds calls and bulk without "
            "adding anything to remember, which is what separates 'the agent made more calls' "
            "from 'the agent has more values to recall'. Without them, raising --tool-turns "
            "moves both at once and no comparison across it is honest."
        ),
    )
    parser.add_argument(
        "--tool-turns",
        type=int,
        default=6,
        help=(
            "Tool-call groups to plant. Must exceed the strategies' keep_last_tool_call_groups "
            "(4) or tool-oriented compaction never fires. Default 6."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32_000,
        help=(
            "The context limit this run stands in for: the fill fraction is a share of it, the "
            "strategies budget against it, and any call whose prompt exceeds it disqualifies "
            "that row. The limit is simulated -- the model itself accepts far more -- so it has "
            "to be enforced here or a row that a model this size would have refused is ranked "
            "anyway. That happened: the 60,000 control ran at 78,003 tokens and every "
            "'cheaper than not compacting' at that size was measured against it."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2_048,
        help=(
            "The model's maximum output tokens per response, which is subtracted from "
            "--context-window to give the input budget every threshold is a fraction of. "
            "This is the model's ceiling, not the size of reply you want: setting it too low "
            "inflates the budget and can push a strategy's trigger above what the service "
            "will accept, which disables compaction with no warning."
        ),
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=4_000,
        help=(
            "Cap sent as max_tokens on every request, which the provider honours exactly. "
            "Must comfortably exceed the longest closing answer: at roughly 12 tokens per "
            "labelled code, enumerating 53 of them costs ~640 tokens before any prose, and a "
            "truncated answer is scored as lost facts and reads as compaction damage. Raising "
            "it is nearly free, since replies average ~150 tokens and models do not pad to "
            "the cap."
        ),
    )
    parser.add_argument(
        "--keep-last-tool-groups",
        type=int,
        default=4,
        help=(
            "Tool-call groups the tool-oriented strategies retain verbatim. The framework "
            "default is 4; with fewer groups than that in the scenario they collapse nothing "
            "at all. Lower it to make them do real work."
        ),
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.5,
        help="Fraction of the input budget the token_budget_* family compacts down to. Default 0.5.",
    )
    parser.add_argument(
        "--min-correctness",
        type=float,
        default=None,
        help=(
            "Fraction of the control's correctness a strategy must retain to be eligible. "
            f"Defaults to {DEFAULT_MIN_CORRECTNESS}, and under --from-jsonl to whatever the run "
            "that wrote the records used, so a rebuilt verdict is the verdict that was measured."
        ),
    )
    parser.add_argument("--summarizer-provider", default=None, help="Provider for summarization strategies.")
    parser.add_argument("--price-input", type=float, default=None, help="Input price per million tokens.")
    parser.add_argument("--price-cached", type=float, default=None, help="Cached-read price per million tokens.")
    parser.add_argument("--price-output", type=float, default=None, help="Output price per million tokens.")
    parser.add_argument("--tokenizer", default="tiktoken", choices=list(TOKENIZER_NAMES), help="Token counter.")
    parser.add_argument(
        "--no-force-tool-calls",
        action="store_true",
        help=(
            "Let the model decide its own tool calls. Needed for routes that reject a pinned "
            "tool_choice, and it must then be set for the whole run: a run where some rows were "
            "pinned and others were not is comparing different conversations."
        ),
    )
    parser.add_argument(
        "--server-history",
        action="store_true",
        help=(
            "Let the service keep the conversation server-side. Compaction then has nothing to "
            "act on, because the agent only sends the new turn. Off by default so that what is "
            "measured is actually compaction."
        ),
    )
    parser.add_argument("--no-temperature", action="store_true", help="Omit temperature for models that reject it.")
    parser.add_argument("--show-answers", action="store_true", help="Print each final answer in full.")
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help=(
            "Append one JSON record per seed to this file, as each seed is scored rather than "
            "when the cell finishes. A cell is every strategy times --repeats seeds and can run "
            "for hours; without this, anything that stops the process before the table prints "
            "discards every seed already completed and already paid for. The file is appended "
            "to, never truncated, so a resumed run extends it. Rebuild the table with --from-jsonl."
        ),
    )
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help=(
            "Render the table and verdict from a --results-jsonl file instead of running "
            "anything. Handles a file whose cells are incomplete, and states which strategies "
            "and how many seeds each cell holds, so a partial result cannot be read as a "
            "finished one."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and its rough size, call nothing.")
    return parser


def _resolve_pricing(args: argparse.Namespace, provider: str, model: str) -> ModelPricing:
    """Resolve pricing from the command line or OpenRouter's catalogue.

    Returns:
        The model's rates.

    Raises:
        SystemExit: If prices are neither supplied nor discoverable.
    """
    if args.price_input is not None:
        return ModelPricing(
            input_per_million=args.price_input,
            cached_read_per_million=args.price_cached if args.price_cached is not None else args.price_input,
            output_per_million=args.price_output if args.price_output is not None else args.price_input,
        )
    if provider == "openrouter":
        try:
            return fetch_openrouter_pricing(model)
        except (KeyError, OSError) as error:
            raise SystemExit(f"Could not fetch pricing for {model!r}: {error}. Pass --price-input.") from error
    raise SystemExit(f"--price-input is required for provider {provider!r} (only OpenRouter pricing is auto-fetched).")


def _cost(outcome: LiveOutcome, pricing: ModelPricing) -> float:
    """Return what one live run cost, seeding and probes together.

    One number, not two. The seeding spend and what each probe added are the same money spent
    answering the same question, and a table that reports them apart invites reading the cheap
    half: a strategy that seeds cheaply and then needs an enormous prompt to answer anything is
    not a cheap strategy.

    Summarization additionally bills calls the agent never sees; those are added here so that
    the strategy which spends money to preserve information is not scored as though preserving
    it were free.
    """
    fresh = max(outcome.input_tokens - outcome.cached_tokens, 0)
    agent_cost = (
        fresh * pricing.input_per_million
        + outcome.cached_tokens * pricing.cached_read_per_million
        + outcome.output_tokens * pricing.output_per_million
    ) / 1_000_000
    return agent_cost + _summarizer_cost(outcome, pricing)


def _summarizer_cost(outcome: LiveOutcome, pricing: ModelPricing) -> float:
    """Return what a strategy's own summarization calls cost.

    Reported as its own column rather than folded silently into the total. A shared meter
    once leaked one strategy's summarizer spend into every later row as a flat addition,
    which a single total cannot show but a per-row column makes obvious.
    """
    return (
        outcome.summarizer_input_tokens * pricing.input_per_million
        + outcome.summarizer_output_tokens * pricing.output_per_million
    ) / 1_000_000


def _sample_scores(outcome: LiveOutcome, scenario: RecallScenario) -> tuple[RecallScore, ...]:
    """Score every independent reading of one seed's snapshot.

    Each probe repeat is scored on its own, against the same snapshot. That is what makes the
    two spreads separable: everything these disagree about happened after the conversation
    stopped changing.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.

    Returns:
        One score per repeat that answered.
    """
    answered = [repeat for repeat in range(1, outcome.probe_repeats + 1) if outcome.sample(repeat)[1]]
    return tuple(
        RecallScore(
            outcomes=facts,
            answer=chr(10).join(outcome.sample(repeat)[1]) if answered else outcome.answer,
            messages_left=outcome.messages_left,
            messages_total=outcome.messages_peak,
            contradictions=scenario.contradictions,
            error=outcome.error,
        )
        for repeat, facts in zip(answered or [1], score_samples(outcome, scenario), strict=False)
    )


def _seed_record(
    outcome: LiveOutcome,
    scenario: RecallScenario,
    pricing: ModelPricing,
    cell: CellParams,
    seed: int,
) -> SeedRecord:
    """Reduce one finished seed to the durable record everything downstream reads.

    This is where scoring happens, and it happens once. The live table and a table rebuilt
    from the file months later are the same aggregation over the same records, so the two
    cannot quietly disagree about what a cell means -- there is only one path, and this is
    its input.

    Scoring needs the scenario, whose markers are salted per seed, so it has to happen while
    the seed is still in hand. That is also why the record stores results rather than the run:
    the scenario is gone the moment the process is.

    Note that correctness comes off the *scored* result and is not an attribute of the run.
    Reading it from ``LiveOutcome`` passes ruff, pyright and the whole suite, then raises
    ``AttributeError`` on the first live call, after the run has been paid for -- which has
    happened twice, and is why this is a function with a test rather than a line in the loop.

    Args:
        outcome: The finished run.
        scenario: The scenario it was driven from.
        pricing: Rates to cost it at.
        cell: The parameters the seed was measured under.
        seed: 1-based index of this seed within its strategy.

    Returns:
        The record, ready to be appended to the results file.
    """
    scores = _sample_scores(outcome, scenario)
    facts_total = len(scores[0].outcomes) if scores else 0
    facts_left = scores[0].facts_left if scores else 0
    nofetch = len(unretrieved_facts(outcome, scenario))
    return SeedRecord(
        cell=cell,
        strategy=outcome.strategy,
        seed=seed,
        cost=_cost(outcome, pricing),
        summarizer_cost=_summarizer_cost(outcome, pricing),
        input_tokens=outcome.input_tokens,
        cached_tokens=outcome.cached_tokens,
        output_tokens=outcome.output_tokens,
        calls=len(outcome.calls),
        messages_left=outcome.messages_left,
        messages_peak=outcome.messages_peak,
        prompt_tokens_final=outcome.prompt_tokens_final,
        prompt_tokens_peak=outcome.prompt_tokens_peak,
        seed_prompt_tokens=outcome.seed_prompt_tokens,
        facts_total=facts_total,
        # Survival is a property of the snapshot, which every probe was answered from, so it
        # is the same in every sample of a seed and the first one speaks for all of them.
        facts_left=facts_left,
        facts_lost=max(facts_total - facts_left - nofetch, 0),
        nofetch=nofetch,
        correctness_samples=tuple(score.correctness_score for score in scores),
        ignored_samples=tuple(score.ignored_by_model for score in scores),
        combined_samples=score_combined_samples(outcome, scenario),
        disqualified=outcome.disqualified(cell.context_window),
        context_drift=outcome.context_drift,
        rate_limit_retries=outcome.rate_limit_retries,
        throttled_seconds=outcome.throttled_seconds,
        turns_completed=outcome.turns_completed,
        turns_total=outcome.turns_total,
        probe_repeats=outcome.probe_repeats,
        summarizer_calls=outcome.summarizer_calls,
        summarizer_failures=outcome.summarizer_failures,
        strategy_notes=outcome.strategy_notes,
        dropped_options=outcome.dropped_options,
        answer=outcome.answer,
        error=outcome.error,
    )


def _seed_spread(samples: Sequence[Sequence[float]]) -> float:
    """Return the points between the least and most correct seed.

    Compaction's own reliability. A different seed is a different conversation, so this is
    where "the strategy cleared a retention boundary this time and not last time" shows up:
    measured at 78 points for one strategy while the uncompacted control moved 7.

    Args:
        samples: One group of per-repeat correctness readings per seed.

    Returns:
        The gap in percentage points, or 0.0 for a single seed, where nothing is known.
    """
    if len(samples) < 2:
        return 0.0
    means = [fmean(seed or (0.0,)) for seed in samples]
    return (max(means) - min(means)) * 100


def _probe_spread(samples: Sequence[Sequence[float]]) -> float:
    """Return the average points between the least and most correct probe repeat within a seed.

    The model's own enumeration variance, and nothing else: the repeats averaged here were all
    answered from one restored snapshot, so the facts in front of the model and their positions
    were identical. Reported beside the between-seed spread because the two used to arrive as a
    single number, and a strategy that scored 52, 52, 52 and 22 with exactly 27 facts preserved
    every time was indistinguishable from one that had lost different facts each time.

    Averaged over seeds rather than maximised, so one unlucky seed does not stand for all of
    them; the between-seed column is where an unlucky seed belongs.

    Args:
        samples: One group of per-repeat correctness readings per seed.

    Returns:
        The mean within-seed gap in percentage points, or 0.0 when each seed was read once.
    """
    ranges = [(max(seed) - min(seed)) * 100 for seed in samples if len(seed) > 1]
    return fmean(ranges) if ranges else 0.0


def _spread(costs: Sequence[float]) -> float:
    """Return the relative gap between the cheapest and dearest seed.

    Zero for a single seed, which is exactly when nothing is known about stability, so the
    report says so rather than showing a reassuring 0%.
    """
    median = sorted(costs)[len(costs) // 2]
    return (max(costs) - min(costs)) / median if len(costs) > 1 and median > 0 else 0.0


@dataclass(frozen=True, slots=True)
class CellStats:
    """One strategy's cell, aggregated over its seeds and their probe repeats.

    Every figure here is a mean over the cell rather than one chosen run. The table used to
    show the median-*cost* seed on every column, which is a defensible choice for cost and an
    arbitrary draw for accuracy: with a two-valued accuracy distribution it reported whichever
    of the two values happened to sit on the median cost.
    """

    strategy: str
    records: tuple[SeedRecord, ...]
    """The seeds this row is a mean over.

    Records rather than runs, so that the row a live cell prints and the row rebuilt from the
    results file are produced by one function from one kind of input. Anything the table needs
    that is not here is a way for the two to disagree.
    """
    cost: float
    cost_spread: float
    summarizer_cost: float
    input_tokens: float
    cached_tokens: float
    output_tokens: float
    calls: float
    messages_left: float
    messages_peak: float
    prompt_tokens_final: float
    prompt_tokens_peak: float
    seed_prompt_tokens: float
    facts_left: float
    facts_total: int
    nofetch: float
    ignored: float
    correctness: float
    seed_spread: float
    probe_spread: float
    combined: float
    disqualified: float
    """Share of this cell's seeds that sent a prompt larger than the tried limit."""
    rate_limit_retries: int
    """Calls this cell re-sent after the provider refused them for rate reasons."""
    throttled_seconds: float
    """Seconds this cell spent waiting those refusals out.

    Summed over its seeds rather than averaged: this is time the cell took, and a sweep
    reading its logs back wants the total it paid, not a per-seed rate.
    """
    samples: tuple[tuple[float, ...], ...]
    """Per-sample correctness: one tuple per seed, one value per probe repeat."""

    @property
    def hit_rate(self) -> float | None:
        """Share of input tokens served from the provider's cache."""
        return self.cached_tokens / self.input_tokens if self.input_tokens > 0 else None


def _aggregate(strategy: str, records: Sequence[SeedRecord]) -> CellStats:
    """Reduce every seed of one strategy to the row the table shows.

    The only aggregation in the package. A live run reaches it through records it has just
    written; ``--from-jsonl`` reaches it through records it has just read; there is no second
    implementation for the two to drift apart in.

    Args:
        strategy: The strategy these seeds measured.
        records: Every seed of it, in any order.

    Returns:
        The aggregated cell.

    Raises:
        ValueError: If no seeds were supplied, since a row is a mean over something.
    """
    if not records:
        raise ValueError(f"No seeds recorded for {strategy!r}; a row is a mean over at least one.")
    samples = tuple(record.correctness_samples for record in records)
    flat = [value for seed in samples for value in seed]
    ignored = [float(value) for record in records for value in record.ignored_samples]
    combined = [value for record in records for value in record.combined_samples]
    return CellStats(
        strategy=strategy,
        records=tuple(records),
        cost=fmean(record.cost for record in records),
        cost_spread=_spread([record.cost for record in records]),
        summarizer_cost=fmean(record.summarizer_cost for record in records),
        input_tokens=fmean(record.input_tokens for record in records),
        cached_tokens=fmean(record.cached_tokens for record in records),
        output_tokens=fmean(record.output_tokens for record in records),
        calls=fmean(record.calls for record in records),
        messages_left=fmean(record.messages_left for record in records),
        messages_peak=fmean(record.messages_peak for record in records),
        prompt_tokens_final=fmean(record.prompt_tokens_final for record in records),
        prompt_tokens_peak=fmean(record.prompt_tokens_peak for record in records),
        seed_prompt_tokens=fmean(record.seed_prompt_tokens for record in records),
        facts_left=fmean(record.facts_left for record in records),
        # The largest, not the mean: every seed of a cell plants the same number of facts, so
        # a smaller one is a seed that failed before scoring rather than an easier scenario.
        facts_total=max((record.facts_total for record in records), default=0),
        nofetch=fmean(record.nofetch for record in records),
        ignored=fmean(ignored or [0.0]),
        correctness=fmean(flat) if flat else 0.0,
        seed_spread=_seed_spread(samples),
        probe_spread=_probe_spread(samples),
        combined=fmean(combined) if combined else 0.0,
        disqualified=fmean(1.0 if record.disqualified else 0.0 for record in records),
        rate_limit_retries=sum(record.rate_limit_retries for record in records),
        throttled_seconds=sum(record.throttled_seconds for record in records),
        samples=samples,
    )


def _excluded_cells(cells: Sequence[CellStats]) -> tuple[set[str], set[str]]:
    """Return the cells that did not finish, and the cells that overran the tried limit.

    Disqualified rather than starred. A row whose prompt exceeded the limit it stands in for
    is not a slightly worse row: it is a row a model of that size would have refused. Ranking
    against one is what made every "+18% versus not compacting" at 60,000 tokens a comparison
    with a baseline that ran at 78,003 and could not have existed.

    A run that stopped early is excluded for the opposite reason: it spent almost nothing and
    answered almost nothing, so it ranks as "100% cheaper" for having died.

    Args:
        cells: Every aggregated cell.

    Returns:
        The names that did not finish, and the names that were disqualified.
    """
    incomplete = {
        cell.strategy for cell in cells if any(record.turns_completed < record.turns_total for record in cell.records)
    }
    oversized = {cell.strategy for cell in cells if cell.disqualified > 0}
    return incomplete, oversized


def _to_joint(stats: CellStats) -> JointOutcome:
    """Convert an aggregated cell into the shape the joint verdict already understands.

    The verdict ranks on ``correctness``, which here is the mean over every sample of every
    seed. ``score`` carries no outcomes: every count the table prints comes from
    :class:`CellStats`, and ``JointOutcome`` consults its score only when there are no samples
    to rank on -- which cannot happen here, since scoring yields at least one reading even for
    a seed that never answered. The field exists for the replay paths, which read a cell once.
    """
    return JointOutcome(
        strategy=stats.strategy,
        cost=stats.cost,
        input_tokens=round(stats.input_tokens),
        cached_tokens=round(stats.cached_tokens),
        messages_left=round(stats.messages_left),
        # The peak, not an uncompacted total: with real replies there is no single "what it
        # would have been" shared across rows, and the peak is what this run actually reached.
        messages_total=round(stats.messages_peak),
        score=RecallScore(
            outcomes=(),
            answer=stats.records[0].answer,
            messages_left=round(stats.messages_left),
            messages_total=round(stats.messages_peak),
            error=stats.records[0].error,
        ),
        correctness_samples=tuple(value for seed in stats.samples for value in seed),
    )


#: Correctness range, in points, above which the accuracy column cannot rank anything.
#: Measured on the uncompacted control at 60,000 tokens: 78 points under the harness's own
#: default narration guidance, 9 with narration suppressed and 15 with it demanded. A control
#: that swings by more than this is choosing between two behaviours, not measuring one.
MAX_USABLE_CORRECTNESS_RANGE: Final[float] = 20.0


def _accuracy_note(correctness_range: dict[str, float], control: str, repeats: int) -> list[str]:
    """Return a warning when the control's own correctness is too unstable to rank against.

    The cost axis has been policed by :func:`_stability_note` since the beginning; the
    accuracy axis was not, and it silently produced three unusable matrices. The accuracy
    column is a mean, which is honest about the middle and says nothing about the shape: a
    control scoring 100, 22 and 22 prints an unremarkable 48 unless something says otherwise.

    Args:
        correctness_range: Points between the least and most correct seed, per strategy.
        control: Name of the uncompacted baseline.
        repeats: Seeds per strategy.

    Returns:
        Zero or two lines, matching the shape of the cost warning.
    """
    if repeats < 2:
        return []
    swing = correctness_range.get(control, 0.0)
    if swing <= MAX_USABLE_CORRECTNESS_RANGE:
        return []
    return [
        "",
        (
            f"ACCURACY NOT RANKABLE: seeds of the uncompacted control varied by {swing:.0f} "
            f"points, over the {MAX_USABLE_CORRECTNESS_RANGE:.0f}-point limit. Nothing can be "
            "compared against a baseline that unstable. The cost columns are unaffected."
        ),
    ]


def _fill_note(stats: dict[str, CellStats], plan: FillPlan | None, control: str) -> list[str]:
    """Return what the uncompacted run actually filled, and a warning if it missed.

    The fill fraction is only a variable if the conversation lands on it. It is measured on
    the uncompacted control because that is the one row whose context is whatever the
    conversation put there; every other row is by definition somewhere below it.

    Args:
        stats: Aggregated cells.
        plan: The sizing that was solved for, or None when sizing was manual.
        control: Name of the uncompacted baseline.

    Returns:
        One line, plus a second when the deviation is outside the tolerance.
    """
    if plan is None or control not in stats:
        return []
    achieved = stats[control].seed_prompt_tokens
    if achieved <= 0:
        return ["", "FILL UNKNOWN: the provider reported no prompt sizes, so the achieved fill cannot be checked."]
    deviation = (achieved - plan.target_tokens) / plan.target_tokens
    lines = [
        "",
        (
            f"Fill: {achieved:,.0f} tokens seeded against a target of {plan.target_tokens:,} "
            f"({plan.fill_fraction:.0%} of {plan.context_limit:,}), {deviation:+.1%}. "
            f"Payload {plan.payload_tokens:,} tokens, of which {plan.tool_payload_tokens:,} is tool results."
        ),
    ]
    if abs(deviation) > FILL_TOLERANCE:
        lines.append(
            f"FILL OFF TARGET: {deviation:+.1%} is outside the {FILL_TOLERANCE:.0%} tolerance, so this "
            "cell is not the fill fraction it is labelled with and does not sit on the same axis as "
            "the others. The replies are the one term the sizing cannot compute; adjust "
            "--filler-tokens or re-solve against a measured reply size."
        )
    return lines


def _throttle_note(cells: Sequence[CellStats]) -> list[str]:
    """Return the lines reporting what throttling cost this cell in time.

    The flag says a row met a rate limit; this says how much of the row's wall clock went
    into it. Worth its own lines because the retries are invisible in every other column
    while being able to move one of them: a prompt cache that expired during a minute of
    backoff is a miss the hit-rate column reads as compaction breaking the prefix.

    Args:
        cells: The rows, in the order they appear in the table.

    Returns:
        Zero lines when nothing was throttled, otherwise a heading and one line per row.
    """
    throttled = [cell for cell in cells if cell.rate_limit_retries]
    if not throttled:
        return []
    return [
        "",
        "Throttled: the provider refused these calls for rate reasons and they were re-sent",
        "after a wait. The measurement is unchanged; the wall clock is not, and neither is the",
        "cache hit rate if a prefix expired while a call was waiting.",
        *(
            f"  {cell.strategy:<28}{cell.rate_limit_retries} retries, {cell.throttled_seconds:,.0f}s waiting"
            for cell in throttled
        ),
    ]


def _stability_note(verdict: JointVerdict, spread: dict[str, float], repeats: int) -> list[str]:
    """Return a warning when the recommendation's margin is inside the measured noise.

    A ranking is only worth reporting if the gap between the options is larger than the gap
    between repeats of the same option. Live cost was measured swinging about 20% on
    identical configuration, mostly from reply length, which is wider than most of the
    differences between strategies.
    """
    if repeats < 2:
        return ["", "Single seed: nothing here measures compaction's own reliability. Re-run with --repeats 3."]
    chosen, base = verdict.chosen, verdict.baseline
    if chosen.strategy == base.strategy or base.cost <= 0:
        return []
    margin = abs(base.cost - chosen.cost) / base.cost
    worst = max(spread.get(chosen.strategy, 0.0), spread.get(base.strategy, 0.0))
    if worst > margin:
        note = (
            f"NOT SUPPORTED: repeats of one strategy varied by {worst:.0%}, wider than the "
            f"{margin:.0%} gap this recommendation rests on. Treat the cost ranking as unresolved."
        )
        return ["", note]
    return []


def _flags(stats: CellStats, control: CellStats | None) -> list[str]:
    """Return the short tokens the flags column carries for one row.

    Args:
        stats: The row.
        control: The uncompacted baseline, or None when the file being read does not hold it.
    """
    flags: list[str] = []
    # A row that gathered a different set of facts than the control is not comparable to
    # it on either axis: it has a different denominator for correctness and a different
    # token volume for cost. Measured at 25% more input for runs that fetched every tool.
    if control is not None and round(stats.nofetch) != round(control.nofetch):
        flags.append("FETCH")
    dropped = {option for record in stats.records for option in record.dropped_options}
    if dropped:
        flags.append("NO:" + ",".join(sorted(option[:4] for option in dropped)))
    if any(record.error for record in stats.records):
        flags.append("ERR")
    drift = sum(record.context_drift for record in stats.records)
    if drift:
        flags.append(f"DRIFT:{drift}")
    if stats.rate_limit_retries:
        flags.append(f"THROTTLED:{stats.rate_limit_retries}")
    failures = sum(record.summarizer_failures for record in stats.records)
    if failures:
        flags.append(f"S{failures}")
    for note in sorted({note for record in stats.records for note in record.strategy_notes}):
        flags.append(note)
    incomplete = [record for record in stats.records if record.turns_completed < record.turns_total]
    if incomplete:
        flags.append(f"{incomplete[0].turns_completed}/{incomplete[0].turns_total}t")
    return flags


def _row(stats: CellStats, control: CellStats | None, excluded: bool) -> str:
    """Render one strategy's line of the table.

    Args:
        stats: The row.
        control: The uncompacted baseline, or None when the file being read does not hold it,
            in which case both relative columns read as unknown rather than being computed
            against whichever row happened to be first.
        excluded: Whether this row is out of the ranking.
    """
    if control is None or stats.strategy == control.strategy or control.cost <= 0:
        cost_delta = "-"
    else:
        cost_delta = f"{stats.cost / control.cost - 1:+.0%}"
    if control is None or stats.strategy == control.strategy or control.correctness <= 0:
        relative = "-"
    else:
        relative = f"{stats.correctness / control.correctness:.0%}"
    hit = "n/a" if stats.hit_rate is None else f"{stats.hit_rate:.0%}"
    flags = _flags(stats, control)
    # ``DQ`` is the dq column crossing zero and nothing else. It used to be "excluded from the
    # ranking", which is a wider set: a row that failed a turn is excluded too, and the last
    # sweep printed rows flagged DQ beside a dq of 0% because every one of them had died on a
    # rate limit. Two exclusions with one name make the flag unreadable exactly when it
    # matters, so the other reason has its own token.
    if stats.disqualified > 0:
        flags.insert(0, "DQ")
    elif excluded:
        flags.insert(0, "EXCL")
    summ = stats.summarizer_cost
    lost = max(stats.facts_total - stats.facts_left - stats.nofetch, 0.0)
    return (
        f"{stats.strategy:<28}{f'{stats.messages_left:.0f}/{stats.messages_peak:.0f}':>9}"
        f"{f'{stats.prompt_tokens_final:,.0f}/{stats.prompt_tokens_peak:,.0f}':>16}"
        f"{stats.calls:>7.0f}{stats.input_tokens:>12,.0f}{hit:>6}"
        f"{stats.output_tokens:>10,.0f}{'$' + format(stats.cost, '.4f'):>10}"
        f"{stats.cost_spread:>6.0%}"
        f"{('-' if not summ else '$' + format(summ, '.4f')):>8}{cost_delta:>9}"
        f"{f'{stats.facts_left:.0f}/{stats.facts_total}':>9}{lost:>6.0f}"
        f"{stats.nofetch:>8.0f}{stats.ignored:>8.0f}"
        f"{stats.correctness:>8.0%}{'*' if control is not None and stats.strategy == control.strategy else ' '}"
        f"{f'{stats.seed_spread:.0f}pp':>7}{f'{stats.probe_spread:.0f}pp':>7}"
        f"{stats.combined:>5.0%} {relative:>8}{stats.disqualified:>5.0%}"
        f"{(','.join(flags) or '-'):>10}"
    )


_LEGEND: Final[tuple[str, ...]] = (
    "msgs      = messages in a probe's prompt, out of the most any call carried. Every probe",
    "            is asked from the same restored snapshot, so this no longer drifts downwards",
    "            through the questions the way it did when they were ordinary turns",
    "tok       = billed tokens in that same prompt, and at the peak. Watch this rather than",
    "            msgs: a strategy that rewrites content in place removes tokens without",
    "            removing messages, and msgs cannot see it",
    "calls     = model calls, seeding and probes together",
    "in        = input tokens billed across the whole run",
    "hit%      = share of those served from the provider's cache. Compaction breaks the",
    "            cached prefix by construction, so this is what it gives up to save tokens",
    "out       = output tokens billed across the whole run. Its own column because a total",
    "            driven by how much the model wrote is a different finding from one driven",
    "            by how much context it was sent, and one number cannot show which",
    "cost      = the whole run: seeding, plus what every probe added, summed. Not split into",
    "            a seed and a probe column, because half of it is not a price anyone pays",
    "+-        = spread between the cheapest and dearest seed. A gap smaller than this is not",
    "            a result. 0% with one seed means stability is unknown, not that it is stable",
    "summ$     = what this strategy's own summarization calls cost, of that total",
    "vs none   = against the uncompacted control: the first is cost, the second accuracy.",
    "            Read them together or not at all -- cheaper and less correct is not a saving",
    "facts     = planted facts surviving compaction into the snapshot: recall's ceiling.",
    "            Scored against the snapshot, which is exactly the context every probe was",
    "            answered from. Scored against a closing prompt instead, this was circular:",
    "            each answer re-listed codes into the history, so a code compaction had",
    "            destroyed came back because the model had recited it two questions earlier",
    "lost      = compaction removed it, so the model could not use it  <- the damage",
    "nofetch   = the agent never called that tool, so the fact never entered the history at",
    "            all. Not compaction damage: an uncompacted run shows these too",
    "ignored   = still in the snapshot but unused: the model's failing, not compaction's",
    "acc       = mean share of checks passed across every probe repeat of every seed, each",
    "            reply scored only against the values its own question asked for",
    "seed+-    = points between the least and most correct seed. Different conversations, so",
    "            this is compaction's own reliability: whether it cleared a retention",
    "            boundary this time and not last time",
    "rep+-     = points between the least and most correct repeat *within* one seed, averaged",
    "            over seeds. Identical facts in identical positions, so this is the model's",
    "            willingness to enumerate and nothing else. The two used to arrive as one",
    "            number, and a strategy scoring 52, 52, 52 and 22 with exactly 27 facts",
    "            preserved every time read the same as one that lost different facts each time",
    "all       = share of all planted values present in the combined answer, where the model",
    "            is asked for everything at once. Asked from the snapshot like every other",
    "            probe, so it is no longer penalised for having been asked last",
    "dq        = share of this cell's seeds that sent a prompt larger than the tried limit.",
    "            The limit is simulated, so it is enforced here or not at all. A cell that",
    "            disqualifies at all is excluded from the ranking rather than starred: a row",
    "            a model that size would have refused is not a baseline for anything",
    "flags     = DQ the dq column above is not zero, so this row sent a prompt a model of",
    "            this size would have refused. EXCL out of the ranking for the other reason:",
    "            it did not finish its turns. The two used to share the name DQ, which is how",
    "            a table came to show rows flagged DQ beside a dq of 0%. ERR failed turn,",
    "            THROTTLED:<n> calls re-sent after the provider refused them for rate",
    "            reasons; the seconds spent waiting are printed below the table, and they",
    "            matter because a cached prefix that expired during a wait is a miss the hit%",
    "            column charges to compaction. DRIFT:<n> probes whose prompt was not the",
    "            snapshot verbatim, because the strategy acted again on the restored state.",
    "            Those probes saw slightly less than survival was scored against, so a row",
    "            carrying this overstates what reached the model. S<n> summarizer failures,",
    "            <n>/<n>t turns",
    "            completed, REC:<n> records the strategy found, FORCED:<n> times it asked for",
    "            one, FALLBACK:<n> times it gave up and compacted another way. A row with",
    "            FALLBACK is measuring that other strategy, not the one named. NO:<opt> the",
    "            provider rejected that option so it was dropped; a run that dropped",
    "            tool_choice chose its own tool calls and is not comparable with one that did",
    "            not. FETCH this row gathered a different set of facts than the control",
)


def _render(
    verdict: JointVerdict | None,
    cells: Sequence[CellStats],
    excluded: set[str],
    control: str = "none",
    *,
    show_answers: bool,
) -> str:
    """Render cost and correctness side by side, then the recommendation.

    Everything about the run itself -- the model, the rates, the sizing, how many seeds were
    asked for -- is read off the cells rather than passed in beside them. A caller cannot then
    label a table with a model or a price the numbers were not produced under, which is the
    one way a rebuilt table could have lied while every column in it was correct.

    Args:
        verdict: The recommendation, or None when the records hold no admissible control and
            there is therefore nothing to recommend against.
        cells: The rows, in the order they should appear.
        excluded: Strategies that are out of the ranking.
        control: Name of the uncompacted baseline.

    Keyword Args:
        show_answers: Print each cell's first answer in full.

    Returns:
        The rendered table.
    """
    baseline = next((cell for cell in cells if cell.strategy == control), None)
    cell_params = cells[0].records[0].cell
    pricing = cell_params.pricing
    header = (
        f"{'strategy':<28}{'msgs':>9}{'tok left/peak':>16}{'calls':>7}{'in':>12}{'hit%':>6}"
        f"{'out':>10}{'cost':>10}{'+-':>6}{'summ$':>8}{'vs none':>9}"
        f"{'facts':>9}{'lost':>6}{'nofetch':>8}{'ignored':>8}{'acc':>9}{'seed+-':>7}{'rep+-':>7}"
        f"{'all':>6}{'vs none':>9}{'dq':>5}{'flags':>10}"
    )
    lines = [
        "",
        (f"Model: {cell_params.model}   agent: {cell_params.agent_kind}   probe repeats: {cell_params.probe_repeats}"),
        (
            f"Pricing: ${pricing.input_per_million:.2f}/M in, "
            f"${pricing.cached_read_per_million:.3f}/M cached, ${pricing.output_per_million:.2f}/M out"
        ),
        "",
        header,
        "-" * len(header),
    ]
    lines.extend(_row(cell, baseline, cell.strategy in excluded) for cell in cells)
    lines += ["", *_LEGEND, "", "per-sample correctness, one group per seed:"]
    for cell in cells:
        groups = "  ".join("[" + " ".join(f"{value:.0%}" for value in seed) + "]" for seed in cell.samples if seed)
        lines.append(f"  {cell.strategy:<28}{groups}")
    lines += _fill_note({cell.strategy: cell for cell in cells}, cell_params.plan, control)
    lines += _throttle_note(cells)
    if verdict is None:
        lines += [
            "",
            (
                f"NO VERDICT: these records hold no admissible {control!r} row, and every ranking "
                "here is relative to one. The columns above still describe what was measured."
            ),
        ]
    else:
        lines += [
            "",
            f"VERDICT: {verdict.recommended}",
            verdict.rationale,
            *_stability_note(verdict, {cell.strategy: cell.cost_spread for cell in cells}, cell_params.repeats),
            *_accuracy_note({cell.strategy: cell.seed_spread for cell in cells}, control, cell_params.repeats),
        ]
    failed = [cell.strategy for cell in cells if any(record.summarizer_failures for record in cell.records)]
    if failed:
        lines += [
            "",
            f"WARNING: the summarizer failed for {', '.join(failed)}. SummarizationStrategy swallows",
            "those errors and skips compaction, so those rows describe a run that barely compacted",
            "and their high correctness is not evidence that summarization preserves information.",
        ]
    if show_answers:
        for cell in cells:
            lines += ["", f"--- {cell.strategy} ---", cell.records[0].answer or "(no answer)"]
    return "\n".join(lines)


def _plan_or_exit(args: argparse.Namespace, tokenizer: Any) -> FillPlan | None:
    """Solve the fill sizing, or exit explaining why this cell cannot be built.

    Returns:
        The plan, or None when --fill 0 asked for manual sizing.

    Raises:
        SystemExit: If the payload does not fit inside the target.
    """
    if args.fill <= 0:
        return None
    try:
        return plan_fill(
            tokenizer=tokenizer,
            context_limit=args.context_window,
            fill_fraction=args.fill,
            tool_turns=args.tool_turns,
            filler_tool_turns=args.filler_tool_turns,
            markers_per_tool=args.markers_per_tool,
            tool_result_tokens=args.tool_result_tokens,
            narration=args.narration,
            fact_placement=args.fact_placement,
            retrieval_guidance=not args.no_retrieval_guidance,
            subset_questions=not args.sweeping_question,
            filler_turn_tokens=args.filler_tokens,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def _progress(record: SeedRecord) -> str:
    """Return the line printed the moment a seed lands.

    A cell prints its table only at the end and takes hours to get there, so without this the
    only difference between a run that is working and one whose rows have collapsed is elapsed
    time. Cost, facts and accuracy are the three that move first: a strategy that has stopped
    preserving anything shows it here, hours before the table would.

    Args:
        record: The seed that just finished.

    Returns:
        One line, already indented to sit under the strategy heading.
    """
    parts = [
        f"   {record.strategy} seed {record.seed}/{record.cell.repeats}",
        f"${record.cost:.4f}",
        f"facts {record.facts_left}/{record.facts_total}",
        f"acc {record.correctness:.0%}",
    ]
    if record.disqualified:
        parts.append("DQ")
    if record.context_drift:
        parts.append(f"DRIFT:{record.context_drift}")
    if record.rate_limit_retries:
        parts.append(f"THROTTLED:{record.rate_limit_retries} ({record.throttled_seconds:,.0f}s)")
    if record.error:
        parts.append(record.error)
    return "  ".join(parts)


def _exclusion_notes(incomplete: set[str], oversized: set[str], limit: int) -> list[str]:
    """Return the lines naming what was dropped from the ranking, and why.

    Args:
        incomplete: Strategies that did not finish their turns.
        oversized: Strategies that overran the tried limit.
        limit: The context limit the cell stands in for.

    Returns:
        Zero or more lines.
    """
    lines: list[str] = []
    if incomplete:
        lines += ["", f"Excluded from the verdict ({len(incomplete)} did not finish): " + ", ".join(sorted(incomplete))]
    if oversized:
        lines += [
            "",
            f"Excluded from the verdict ({len(oversized)} exceeded the {limit:,}-token "
            "limit this run stands in for): " + ", ".join(sorted(oversized)),
        ]
    return lines


def _coverage(cell: CellParams, records: Sequence[SeedRecord]) -> list[str]:
    """Return what a cell read back from file actually holds, and whether that is all of it.

    A file is written seed by seed precisely so that an interrupted cell keeps what it had, so
    an incomplete cell is the normal case here rather than the exception. Every mean in the
    table below is over whatever is present, and the difference between a mean over fifteen
    strategy-seeds and one over four is invisible in the table itself -- so it is stated here,
    against what the run said it was going to take.

    Args:
        cell: The cell's parameters.
        records: Its records.

    Returns:
        The heading, what is present, and a PARTIAL line when something is missing.
    """
    seeds: dict[str, list[int]] = {}
    for record in records:
        seeds.setdefault(record.strategy, []).append(record.seed)
    # Intent is unioned over the records rather than read off the first, because a cell
    # abandoned partway and resumed for the rest is written by two runs that each asked for
    # part of it. Taking the first record's list would report the resumed half as unwanted.
    intended = sorted({name for record in records for name in record.cell.strategies} | set(seeds))
    wanted = max(record.cell.repeats for record in records)
    present = ", ".join(f"{name} {len(seeds.get(name, ()))}/{wanted}" for name in intended)
    lines = ["", f"Cell: {cell.label}", f"  seeds present: {present}"]
    missing = [name for name in intended if name not in seeds]
    short = [name for name, found in seeds.items() if len(found) < wanted]
    if not missing and not short:
        return lines
    detail: list[str] = []
    if missing:
        detail.append(f"{len(missing)} of {len(intended)} strategies never recorded a seed ({', '.join(missing)})")
    if short:
        detail.append(f"{', '.join(sorted(short))} recorded fewer than the {wanted} seeds asked for")
    lines.append(f"  PARTIAL: {'; '.join(detail)}. Every mean below is over what is present.")
    return lines


def _cells_from_records(records: Sequence[SeedRecord]) -> list[CellStats]:
    """Aggregate one cell's records into rows, cheapest first.

    Args:
        records: Every record of one cell.

    Returns:
        One row per strategy present, ordered as the live table orders them.
    """
    by_strategy: dict[str, list[SeedRecord]] = {}
    for record in records:
        by_strategy.setdefault(record.strategy, []).append(record)
    cells = [_aggregate(strategy, seeds) for strategy, seeds in by_strategy.items()]
    cells.sort(key=lambda cell: cell.cost)
    return cells


def _render_from_records(args: argparse.Namespace) -> int:
    """Rebuild the table from a results file, running nothing.

    Args:
        args: Parsed command line arguments.

    Returns:
        A process exit code.

    Raises:
        SystemExit: If the file cannot be read or holds no records.
    """
    path = Path(args.from_jsonl)
    if not path.is_file():
        raise SystemExit(f"No results file at {path}.")
    try:
        records = read_seed_records(path)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not records:
        raise SystemExit(f"{path} holds no records.")

    groups = group_by_cell(records)
    print(f"{len(records)} seed records from {path}, in {len(groups)} cell(s).")
    for cell_params, cell_records in groups:
        cells = _cells_from_records(cell_records)
        incomplete, oversized = _excluded_cells(cells)
        excluded = incomplete | oversized
        for line in _coverage(cell_params, cell_records):
            print(line)
        for line in _exclusion_notes(incomplete, oversized, cell_params.context_window):
            print(line)
        ranked = [_to_joint(cell) for cell in cells if cell.strategy not in excluded]
        verdict: JointVerdict | None = None
        if any(outcome.strategy == "none" for outcome in ranked):
            try:
                # The bar the run set, unless this invocation names one: a rebuilt verdict
                # that silently applied a different threshold would rank rows the original
                # never ranked, while every column above it stayed identical.
                verdict = recommend(
                    ranked,
                    min_correctness=(
                        cell_params.min_correctness if args.min_correctness is None else args.min_correctness
                    ),
                )
            except ValueError as error:
                print(f"Cannot summarize: {error}")
        print(_render(verdict, cells, excluded, show_answers=args.show_answers))
    return 0


async def run_live_comparison(args: argparse.Namespace) -> int:
    """Run every selected strategy against a live agent and print the comparison.

    Args:
        args: Parsed command line arguments.

    Returns:
        A process exit code.

    Raises:
        SystemExit: If the arguments do not describe a runnable cell.
    """
    if args.from_jsonl is not None:
        return _render_from_records(args)
    if args.provider is None:
        raise SystemExit("A provider is required, unless --from-jsonl is rebuilding a table from a results file.")
    provider, model_override = parse_provider_selector(args.provider)
    if provider not in provider_names():
        raise SystemExit(f"Unknown provider {provider!r}. Available: {', '.join(provider_names())}")
    strategies = [entry.strip() for entry in args.strategies.split(",") if entry.strip()]
    if "none" not in strategies:
        raise SystemExit("The 'none' control must be included; every comparison is relative to it.")

    min_correctness = DEFAULT_MIN_CORRECTNESS if args.min_correctness is None else args.min_correctness
    tokenizer = build_tokenizer(args.tokenizer)
    retained = args.keep_last_tool_groups
    plan = _plan_or_exit(args, tokenizer)
    filler_turns = plan.filler_turns if plan else args.filler_turns
    filler_tokens = plan.filler_tokens if plan else args.filler_tokens
    probe = build_live_scenario(
        salt="probe",
        filler_turns=filler_turns,
        filler_tokens=1,
        tool_turns=args.tool_turns,
        filler_tool_turns=args.filler_tool_turns,
    )
    planted_groups = len(probe.tool_lookups)
    # Every tool-oriented strategy keeps the last `retained` groups verbatim. With no more
    # groups than that, it evicts nothing, changes no tokens, and scores a perfect result for
    # having done nothing at all -- which reads as the best row in the table. Measured: at 3
    # groups against a retention of 4, tool_result and selective_tool_call were exact no-ops
    # while carrying 55% of the planted facts.
    tool_strategies_inert = planted_groups <= retained
    if needs_summarizer(strategies) and args.summarizer_provider is None and not args.dry_run:
        raise SystemExit("Summarization strategies require --summarizer-provider.")

    if args.dry_run:
        scenario = build_live_scenario(
            salt="dry",
            filler_turns=filler_turns,
            filler_tokens=filler_tokens,
            tool_turns=args.tool_turns,
            markers_per_tool=args.markers_per_tool,
            filler_tool_turns=args.filler_tool_turns,
            narration=args.narration,
            subset_questions=not args.sweeping_question,
        )
        questions = max(scenario.answer_turn_count, 1)
        probes = questions * args.probe_repeats
        print(f"strategies: {len(strategies)}  turns: {len(scenario.transcript.turns)}  facts: {len(scenario.facts)}")
        print(f"tool-call groups: {planted_groups} planted, {retained} retained by tool-oriented strategies")
        if plan is not None:
            print(
                f"fill: {plan.predicted_tokens:,} predicted against {plan.target_tokens:,} target "
                f"({plan.fill_fraction:.0%} of {plan.context_limit:,}), {plan.deviation:+.1%}"
            )
            print(
                f"sizing: {plan.filler_turns} filler turns of ~{plan.filler_tokens:,} tokens; "
                f"payload {plan.payload_tokens:,} tokens, tool results {plan.tool_payload_tokens:,}"
            )
        else:
            print(f"fill: manual, {filler_turns} filler turns of ~{filler_tokens:,} tokens")
        print(f"probes: {questions} questions x {args.probe_repeats} repeats = {probes} per seed")
        seed_calls = len(scenario.transcript.turns) - questions
        total_calls = len(strategies) * args.repeats * (seed_calls + probes)
        print(f"model calls: >= {total_calls} (more whenever a tool is used)")
        if plan is not None:
            # Every probe carries the whole snapshot, so the probes cost the full prompt each
            # while the seeding averages about half of it. Worth printing before anything is
            # spent: raising --probe-repeats multiplies the expensive half, not the cheap one.
            seeding = seed_calls * plan.predicted_tokens // 2
            probing = probes * plan.predicted_tokens
            per_run = seeding + probing
            print(
                f"prompt tokens: ~{per_run * len(strategies) * args.repeats:,} in total, "
                f"~{per_run:,} per strategy-seed (~{seeding:,} seeding, ~{probing:,} probing). "
                "Cache reads take most of this off; probing is the half --probe-repeats scales."
            )
        for name in strategies:
            build_strategy(name, StrategyOptions(tokenizer, args.context_window, args.max_output_tokens))
        print("every strategy builds cleanly")
        return 0

    runtime = build_provider(
        provider,
        temperature=None if args.no_temperature else 0.0,
        response_max_tokens=args.answer_max_tokens,
        model=model_override,
    )
    pricing = _resolve_pricing(args, provider, runtime.model)
    if plan is not None:
        # Printed on every run, not only the dry one. These logs are archived and read back
        # months later against runs made with different sizing, and a cell that cannot say
        # what it was aiming at cannot be placed on an axis with the others.
        print(
            f"sizing: {plan.filler_turns} filler turns of ~{plan.filler_tokens:,} tokens, "
            f"predicting {plan.predicted_tokens:,} against a target of {plan.target_tokens:,} "
            f"({plan.fill_fraction:.0%} of {plan.context_limit:,}); payload {plan.payload_tokens:,} "
            f"tokens, of which {plan.tool_payload_tokens:,} is tool results.",
            flush=True,
        )

    # Clients on the Responses API keep the conversation server-side. When they do, the agent
    # sends only the new turn and MAF skips HistoryProvider.before_run entirely -- the history
    # never reaches the outgoing messages, so a compaction strategy has nothing to compact and
    # every setting silently measures the same thing. Measured on Foundry before this was
    # forced: a 16-turn conversation reported a one-message prompt on every row.
    stores_by_default = bool(getattr(runtime.client, "STORES_BY_DEFAULT", False))
    if wants_client_side_history(runtime.client, allow_server_history=args.server_history):
        # run_live forces this itself; setting it here too keeps the note honest about what
        # the run will actually do.
        runtime.options["store"] = False
        print(
            f"note: {runtime.model} keeps history server-side by default. Forcing store=False so "
            "the history is sent by the client and compaction actually applies.",
            flush=True,
        )
    elif stores_by_default:
        print(
            "WARNING: --server-history means the service owns the conversation. The agent sends "
            "only the new turn, so no strategy can compact anything and every row will match the "
            "control. This measures the service, not compaction.",
            flush=True,
        )
    summarizer_client: Any = None
    if args.summarizer_provider is not None:
        sum_provider, sum_model = parse_provider_selector(args.summarizer_provider)
        summarizer_client = build_provider(
            sum_provider, temperature=0.0, response_max_tokens=1_024, model=sum_model
        ).client

    cell_params = CellParams(
        provider=provider,
        model=runtime.model,
        agent_kind=args.agent,
        context_window=args.context_window,
        fill=args.fill,
        probe_repeats=args.probe_repeats,
        repeats=args.repeats,
        strategies=tuple(strategies),
        narration=args.narration,
        fact_placement=args.fact_placement,
        tool_result_tokens=args.tool_result_tokens,
        filler_turns=filler_turns,
        filler_tokens=filler_tokens,
        tool_turns=args.tool_turns,
        filler_tool_turns=args.filler_tool_turns,
        markers_per_tool=args.markers_per_tool,
        price_input=pricing.input_per_million,
        price_cached=pricing.cached_read_per_million,
        price_output=pricing.output_per_million,
        min_correctness=min_correctness,
        plan=plan,
    )
    results_path = Path(args.results_jsonl) if args.results_jsonl is not None else None

    cells: list[CellStats] = []
    for name in strategies:
        print(f"-> {name}", flush=True)
        # A fresh meter per strategy. Sharing one accumulates every earlier strategy's
        # summarizer spend into every later row: measured as a flat +$0.0172 on all five
        # strategies that happened to run after 'summarization', which is invisible in a
        # total and inverted the ranking of the whole token_budget family.
        seeds: list[SeedRecord] = []
        for repeat in range(args.seed_offset + 1, args.seed_offset + args.repeats + 1):
            if args.repeats > 1 or args.seed_offset:
                print(f"   seed {repeat}", flush=True)
            summarizer = MeteredClient(summarizer_client) if summarizer_client is not None else None
            scenario = build_live_scenario(
                salt=f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{repeat}",
                filler_turns=filler_turns,
                filler_tokens=filler_tokens,
                tool_turns=args.tool_turns,
                markers_per_tool=args.markers_per_tool,
                filler_tool_turns=args.filler_tool_turns,
                narration=args.narration,
                subset_questions=not args.sweeping_question,
            )
            options = StrategyOptions(
                tokenizer=tokenizer,
                max_context_window_tokens=args.context_window,
                max_output_tokens=args.max_output_tokens,
                token_budget_fraction=args.budget_fraction,
                keep_last_tool_call_groups=args.keep_last_tool_groups,
                # A recording proxy, not a client: see MeteredClient for why it is cast.
                summarizer=cast("SupportsChatGetResponse[Any] | None", summarizer),
            )
            outcome = await run_live(
                runtime,
                strategy_name=name,
                options=options,
                scenario=scenario,
                agent_kind=args.agent,
                tool_result_tokens=args.tool_result_tokens,
                force_tool_calls=not args.no_force_tool_calls,
                narration=args.narration,
                retrieval_guidance=not args.no_retrieval_guidance,
                fact_placement=args.fact_placement,
                probe_repeats=args.probe_repeats,
            )
            # Scored, written and reported here rather than when the cell ends. A seed that
            # has been paid for is durable the moment it exists, and the line that follows is
            # the only sign of progress a cell gives in the hours before its table.
            record = _seed_record(outcome, scenario, pricing, cell_params, repeat)
            if results_path is not None:
                append_seed_record(results_path, record)
            seeds.append(record)
            print(_progress(record), flush=True)
        cells.append(_aggregate(name, seeds))

    # A run that stopped early spent almost nothing and answered almost nothing. Ranking it
    # produces "100% cheaper" for a strategy that simply died, and counts it as clearing the
    # correctness bar because a near-zero control makes every ratio look enormous.
    incomplete, oversized = _excluded_cells(cells)
    if all(any(record.error for record in cell.records) for cell in cells):
        first = next(record.error for cell in cells for record in cell.records if record.error)
        raise SystemExit(
            f"Every strategy failed. First error: {first}"
            + chr(10)
            + "No comparison is possible; nothing below would mean anything."
        )
    if not any(cell.cost > 0 for cell in cells):
        raise SystemExit("No strategy reported any billed tokens, so there is nothing to compare.")

    excluded = incomplete | oversized
    for line in _exclusion_notes(incomplete, oversized, args.context_window):
        print(line)
    cells.sort(key=lambda cell: cell.cost)
    ranked = [_to_joint(cell) for cell in cells if cell.strategy not in excluded]

    if not any(outcome.strategy == "none" for outcome in ranked):
        reason = "exceeded the tried limit" if "none" in oversized else "did not finish"
        raise SystemExit(
            f"The uncompacted control {reason}, so there is no admissible baseline at "
            f"{args.context_window:,} tokens and nothing can be compared against it. That is itself "
            "the finding for this cell: lower --fill, or raise --context-window to a size the "
            "conversation fits in."
        )
    try:
        verdict = recommend(ranked, min_correctness=min_correctness)
    except ValueError as error:
        raise SystemExit(f"Cannot summarize: {error}") from error
    if tool_strategies_inert:
        affected = ", ".join(cell.strategy for cell in cells if "tool" in cell.strategy) or "the tool strategies"
        print()
        print(
            f"WARNING: {planted_groups} tool-call groups were planted but tool-oriented strategies "
            f"retain the last {retained}, so {affected} evicted nothing. Their scores measure "
            "a no-op, not information preservation. Raise --tool-turns above the retention."
        )
    print(_render(verdict, cells, excluded, show_answers=args.show_answers))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the live comparison.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    return asyncio.run(run_live_comparison(build_parser().parse_args(argv)))
