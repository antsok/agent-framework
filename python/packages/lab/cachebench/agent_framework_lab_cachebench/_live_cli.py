# Copyright (c) Microsoft. All rights reserved.

"""Command line entry point for the live-agent compaction comparison."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ._advisor import ModelPricing, fetch_openrouter_pricing
from ._live import AGENT_KINDS, LiveOutcome, MeteredClient, build_live_scenario, run_live, score_live
from ._providers import build_provider, parse_provider_selector, provider_names
from ._recall import RecallScenario, RecallScore
from ._strategies import StrategyOptions, build_strategy, strategy_names
from ._summary import DEFAULT_MIN_CORRECTNESS, JointOutcome, JointVerdict, recommend, relative_correctness
from ._tokenizers import TOKENIZER_NAMES, build_tokenizer
from ._transcripts import filler_text

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


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        A parser for the live comparison.
    """
    parser = argparse.ArgumentParser(
        prog="cachebench-live",
        description=(
            "Compare compaction strategies against a real agent that generates its own replies "
            "and calls a real tool. Reports cost and correctness per strategy. Within-model "
            "only: real replies differ per model, so these numbers do not compare across models."
        ),
    )
    parser.add_argument("provider", help="Provider or provider:model.")
    parser.add_argument("--strategies", default=_DEFAULT_STRATEGIES, help=f"Available: {','.join(strategy_names())}")
    parser.add_argument("--agent", default="plain", choices=list(AGENT_KINDS), help="How to assemble the agent.")
    parser.add_argument("--filler-turns", type=int, default=6, help="Padding turns between planted facts.")
    parser.add_argument("--filler-tokens", type=int, default=4_000, help="Approximate size of each filler turn.")
    parser.add_argument(
        "--tool-turns",
        type=int,
        default=6,
        help=(
            "Tool-call groups to plant. Must exceed the strategies' keep_last_tool_call_groups "
            "(4) or tool-oriented compaction never fires. Default 6."
        ),
    )
    parser.add_argument("--context-window", type=int, default=32_000, help="Simulated context window.")
    parser.add_argument("--max-output-tokens", type=int, default=2_048, help="Output reservation for budget math.")
    parser.add_argument("--answer-max-tokens", type=int, default=900, help="Cap on generated replies.")
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.5,
        help="Fraction of the input budget the token_budget_* family compacts down to. Default 0.5.",
    )
    parser.add_argument(
        "--min-correctness",
        type=float,
        default=DEFAULT_MIN_CORRECTNESS,
        help="Fraction of the control's correctness a strategy must retain to be eligible.",
    )
    parser.add_argument("--summarizer-provider", default=None, help="Provider for summarization strategies.")
    parser.add_argument("--price-input", type=float, default=None, help="Input price per million tokens.")
    parser.add_argument("--price-cached", type=float, default=None, help="Cached-read price per million tokens.")
    parser.add_argument("--price-output", type=float, default=None, help="Output price per million tokens.")
    parser.add_argument("--tokenizer", default="tiktoken", choices=list(TOKENIZER_NAMES), help="Token counter.")
    parser.add_argument("--no-temperature", action="store_true", help="Omit temperature for models that reject it.")
    parser.add_argument("--show-answers", action="store_true", help="Print each final answer in full.")
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
    """Return what one live run cost, including the summarizer's own calls.

    A live run generates real replies, so unlike the replay benchmarks its output tokens are
    a real charge rather than a rounding error. Summarization additionally bills calls the
    agent never sees; those are added here so the strategy that spends money to preserve
    information is not scored as though preserving it were free.
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


def _to_joint(outcome: LiveOutcome, scenario: RecallScenario, pricing: ModelPricing) -> JointOutcome:
    """Convert a live outcome into the shape the joint verdict already understands."""
    return JointOutcome(
        strategy=outcome.strategy,
        cost=_cost(outcome, pricing),
        input_tokens=outcome.input_tokens,
        cached_tokens=outcome.cached_tokens,
        messages_left=outcome.messages_left,
        # The peak, not an uncompacted total: with real replies there is no single "what it
        # would have been" shared across rows, and the peak is what this run actually reached.
        messages_total=outcome.messages_peak,
        score=RecallScore(
            outcomes=score_live(outcome, scenario),
            answer=outcome.answer,
            messages_left=outcome.messages_left,
            messages_total=outcome.messages_peak,
            contradictions=scenario.contradictions,
            error=outcome.error,
        ),
    )


def _render(
    verdict: JointVerdict,
    live: dict[str, LiveOutcome],
    pricing: ModelPricing,
    model: str,
    agent_kind: str,
    *,
    show_answers: bool,
) -> str:
    """Render cost and correctness side by side, then the recommendation."""
    base = verdict.baseline
    header = (
        f"{'strategy':<28}{'msgs':>9}{'tok left/peak':>16}{'calls':>7}{'in':>9}{'hit%':>6}"
        f"{'out':>8}{'cost':>10}{'summ$':>8}{'vs none':>9}{'correct':>9}{'vs none':>9}{'flags':>8}"
    )
    lines = [
        "",
        f"Model: {model}   agent: {agent_kind}",
        (
            f"Pricing: ${pricing.input_per_million:.2f}/M in, "
            f"${pricing.cached_read_per_million:.3f}/M cached, ${pricing.output_per_million:.2f}/M out"
        ),
        "",
        header,
        "-" * len(header),
    ]
    for outcome in verdict.outcomes:
        run = live[outcome.strategy]
        summ = _summarizer_cost(run, pricing)
        cost_delta = "-" if outcome.strategy == base.strategy else f"{(outcome.cost - base.cost) / base.cost:+.0%}"
        rel = "-" if outcome.strategy == base.strategy else f"{relative_correctness(outcome, base):.0%}"
        hit = "n/a" if outcome.hit_rate is None else f"{outcome.hit_rate:.0%}"
        flags: list[str] = []
        if run.error:
            flags.append("ERR")
        if run.summarizer_failures:
            flags.append(f"S{run.summarizer_failures}")
        if run.turns_completed < run.turns_total:
            flags.append(f"{run.turns_completed}/{run.turns_total}t")
        lines.append(
            f"{outcome.strategy:<28}{f'{run.messages_left}/{run.messages_peak}':>9}"
            f"{f'{run.prompt_tokens_final:,}/{run.prompt_tokens_peak:,}':>16}"
            f"{len(run.calls):>7}{run.input_tokens:>9,}{hit:>6}"
            f"{run.output_tokens:>8,}{'$' + format(outcome.cost, '.4f'):>10}"
            f"{('-' if not summ else '$' + format(summ, '.4f')):>8}{cost_delta:>9}"
            f"{outcome.correctness:>8.0%}{'*' if outcome.strategy == base.strategy else ' '}{rel:>9}"
            f"{(','.join(flags) or '-'):>8}"
        )
    lines += [
        "",
        "msgs      = messages in the final prompt, out of the most any call carried",
        "tok       = billed tokens in that same final prompt, and at the peak. Watch this",
        "            rather than msgs: a strategy that rewrites content in place removes",
        "            tokens without removing messages, and msgs cannot see it",
        "calls     = model calls, which exceed turns whenever the agent used a tool",
        "in/out    = tokens billed across the whole run, replies included",
        "cost      = the whole conversation, cached reads discounted, summ$ included",
        "summ$     = what this strategy's own summarization calls cost, of that total",
        "correct   = share of correctness checks the final answer passed (* marks the control)",
        "flags     = ERR failed turn, S<n> summarizer failures, <n>/<n>t turns completed",
        "",
        f"VERDICT: {verdict.recommended}",
        verdict.rationale,
    ]
    failed = [name for name, run in live.items() if run.summarizer_failures]
    if failed:
        lines += [
            "",
            f"WARNING: the summarizer failed for {', '.join(failed)}. SummarizationStrategy swallows",
            "those errors and skips compaction, so those rows describe a run that barely compacted",
            "and their high correctness is not evidence that summarization preserves information.",
        ]
    if show_answers:
        for outcome in verdict.outcomes:
            lines += ["", f"--- {outcome.strategy} ---", live[outcome.strategy].answer or "(no answer)"]
    return "\n".join(lines)


async def run_live_comparison(args: argparse.Namespace) -> int:
    """Run every selected strategy against a live agent and print the comparison.

    Args:
        args: Parsed command line arguments.

    Returns:
        A process exit code.
    """
    provider, model_override = parse_provider_selector(args.provider)
    if provider not in provider_names():
        raise SystemExit(f"Unknown provider {provider!r}. Available: {', '.join(provider_names())}")
    strategies = [entry.strip() for entry in args.strategies.split(",") if entry.strip()]
    if "none" not in strategies:
        raise SystemExit("The 'none' control must be included; every comparison is relative to it.")

    tokenizer = build_tokenizer(args.tokenizer)
    retained = StrategyOptions(tokenizer, args.context_window, args.max_output_tokens).keep_last_tool_call_groups
    probe = build_live_scenario(
        salt="probe",
        filler_turns=args.filler_turns,
        filler_tokens=1,
        tool_turns=args.tool_turns,
    )
    planted_groups = len(probe.tool_lookups)
    # Every tool-oriented strategy keeps the last `retained` groups verbatim. With no more
    # groups than that, it evicts nothing, changes no tokens, and scores a perfect result for
    # having done nothing at all -- which reads as the best row in the table. Measured: at 3
    # groups against a retention of 4, tool_result and selective_tool_call were exact no-ops
    # while carrying 55% of the planted facts.
    tool_strategies_inert = planted_groups <= retained
    needs_summarizer = any("summar" in name for name in strategies)
    if needs_summarizer and args.summarizer_provider is None and not args.dry_run:
        raise SystemExit("Summarization strategies require --summarizer-provider.")

    if args.dry_run:
        scenario = build_live_scenario(
            salt="dry",
            filler_turns=args.filler_turns,
            filler_tokens=args.filler_tokens,
            tool_turns=args.tool_turns,
        )
        user_tokens = (
            sum(len(str(content)) for turn in scenario.transcript.turns for m in turn.request for content in m.contents)
            / 4
        )
        print(f"strategies: {len(strategies)}  turns: {len(scenario.transcript.turns)}  facts: {len(scenario.facts)}")
        print(f"tool-call groups: {planted_groups} planted, {retained} retained by tool-oriented strategies")
        print(f"user-side prompt material: ~{user_tokens:,.0f} tokens per run, growing each turn")
        print(f"model calls: >= {len(strategies) * len(scenario.transcript.turns)} (more whenever a tool is used)")
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
    summarizer_client: Any = None
    if args.summarizer_provider is not None:
        sum_provider, sum_model = parse_provider_selector(args.summarizer_provider)
        summarizer_client = build_provider(
            sum_provider, temperature=0.0, response_max_tokens=1_024, model=sum_model
        ).client

    live: dict[str, LiveOutcome] = {}
    joint: list[JointOutcome] = []
    for name in strategies:
        print(f"-> {name}", flush=True)
        # A fresh meter per strategy. Sharing one accumulates every earlier strategy's
        # summarizer spend into every later row: measured as a flat +$0.0172 on all five
        # strategies that happened to run after 'summarization', which is invisible in a
        # total and inverted the ranking of the whole token_budget family.
        summarizer = MeteredClient(summarizer_client) if summarizer_client is not None else None
        scenario = build_live_scenario(
            salt=f"{time.strftime('%Y%m%d-%H%M%S')}-{name}",
            filler_turns=args.filler_turns,
            filler_tokens=args.filler_tokens,
            tool_turns=args.tool_turns,
        )
        options = StrategyOptions(
            tokenizer=tokenizer,
            max_context_window_tokens=args.context_window,
            max_output_tokens=args.max_output_tokens,
            token_budget_fraction=args.budget_fraction,
            # A recording proxy, not a client: see MeteredClient for why it is cast.
            summarizer=cast("SupportsChatGetResponse[Any] | None", summarizer),
        )
        outcome = await run_live(
            runtime,
            strategy_name=name,
            options=options,
            scenario=scenario,
            agent_kind=args.agent,
            tool_filler=filler_text(7, 600),
        )
        live[name] = outcome
        joint.append(_to_joint(outcome, scenario, pricing))
        if outcome.error:
            print(f"   {outcome.error}", flush=True)

    try:
        verdict = recommend(joint, min_correctness=args.min_correctness)
    except ValueError as error:
        raise SystemExit(f"Cannot summarize: {error}") from error
    if tool_strategies_inert:
        affected = ", ".join(name for name in live if "tool" in name) or "the tool-oriented strategies"
        print()
        print(
            f"WARNING: {planted_groups} tool-call groups were planted but tool-oriented strategies "
            f"retain the last {retained}, so {affected} evicted nothing. Their scores measure "
            "a no-op, not information preservation. Raise --tool-turns above the retention."
        )
    print(_render(verdict, live, pricing, runtime.model, args.agent, show_answers=args.show_answers))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the live comparison.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    return asyncio.run(run_live_comparison(build_parser().parse_args(argv)))
