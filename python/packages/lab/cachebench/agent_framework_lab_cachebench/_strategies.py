# Copyright (c) Microsoft. All rights reserved.

"""The registry of strategies under test, and the parameters they are all built from.

This is benchmark configuration rather than a strategy of its own, which is why it stays in
the lab while :mod:`.compaction` -- the strategies written here, meant to leave for a
repository of their own -- does not. What it does is put ours and the framework's behind one
name each, so ``--strategies`` selects between them on equal terms and every row is built
from the same :class:`StrategyOptions`.

Each entry wraps a strategy from ``agent_framework`` or from :mod:`.compaction` so that the
benchmark can select it by name. ``context_window`` is the strategy the agent harness
installs by default when
``create_harness_agent`` is given ``max_context_window_tokens``; the ``*_aggressive`` and
``*_lazy`` variants are the same strategy at different trigger thresholds and exist to
answer whether compacting early and often costs more in lost cache reads than it saves in
prompt tokens.

The ``token_budget_*`` family is different in kind from the rest. Every other entry decides
*when* to compact from its own trigger, so different strategies leave prompts of different
sizes and a comparison between them confounds "trimmed harder" with "trimmed smarter". The
composed variants all compact down to one shared ceiling and differ only in the order they
delete things, which holds size fixed and isolates the choice of what to discard.
``token_budget_fallback`` composes nothing at all, so its removals are pure oldest-first
eviction: the floor any ordering has to beat to be worth its complexity.

Budgets are sized relative to the transcript rather than to a model's real context window.
A 20-turn transcript never approaches a 128k window, so a real window would mean no
strategy ever fires and the benchmark would measure nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from agent_framework import (
    CompactionStrategy,
    ContextWindowCompactionStrategy,
    SelectiveToolCallCompactionStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    TokenizerProtocol,
    ToolResultCompactionStrategy,
    TruncationStrategy,
)

from .compaction import (
    DEFAULT_BAND_SHARE,
    DEFAULT_COVERAGE_SHARE,
    DEFAULT_FALLBACK_FRACTION,
    DEFAULT_MIN_GAIN_FRACTION,
    DEFAULT_TRIGGER_FRACTION,
    AnchoredCompactionStrategy,
    MinimumGainAnchoredCompactionStrategy,
    ToolResultAnchoredSummarizationCompactionStrategy,
)

if TYPE_CHECKING:
    from agent_framework._clients import SupportsChatGetResponse

__all__ = [
    "STRATEGY_BUILDERS",
    "StrategyOptions",
    "build_strategy",
    "resolve_context_window",
    "strategy_names",
]

# Fraction of a transcript's fully-replayed prompt size used as the simulated context
# window when the caller does not pass one. Below 1.0 so that compaction is guaranteed to
# trigger part-way through every preset.
_AUTO_WINDOW_FRACTION: Final[float] = 0.6
# The floor has to leave the system anchor inside even the most aggressive phase's budget.
# ContextWindowCompactionStrategy evicts tool results at 0.5 of the input budget, so a
# window that puts half the budget below the anchor's size drives the strict fallback into
# evicting the anchor itself — measured on the `small` preset at a 2,048 floor, where the
# prompt collapsed to 47 tokens on turn 1. That destroys the stable prefix whose
# cacheability is the entire subject of the benchmark.
_MIN_AUTO_WINDOW_TOKENS: Final[int] = 4_096


@dataclass(frozen=True, slots=True)
class StrategyOptions:
    """Parameters shared by every strategy builder.

    Every field here is a knob the CLI can set, and that is the point of the type: a builder
    reading a constructor default instead of a field makes that parameter unreachable from a
    sweep, which is how ``min_gain_fraction`` and the two record thresholds came to be
    unsettable while the rows that depend on them were being compared.

    The ranges are not validated here. Each strategy validates its own, in ``compaction/``,
    which is where the constraint belongs -- those classes ship without this package -- and
    duplicating the checks would give a sweep two places to disagree about what is legal. What
    this package owes instead is that every selected strategy is *built* before a run spends
    anything, so a bad value fails at the command line rather than on the first paid call; see
    ``_live_cli._build_or_exit``.
    """

    tokenizer: TokenizerProtocol
    max_context_window_tokens: int
    max_output_tokens: int
    keep_last_groups: int = 6
    keep_last_tool_call_groups: int = 4
    keep_head_groups: int = 3
    keep_tail_groups: int = 4
    keep_tokens: int | None = None
    """Tokens of a collapsed tool result the anchored family retains, head and tail together.

    ``None`` derives it from ``band_share`` and the result's position in the band, which is
    what makes retention scale with the window instead of shrinking to nothing as results grow.
    A number fixes it, which is the older behaviour and is worth being able to reproduce: the
    two answer different questions about the same row.
    """
    band_share: float = DEFAULT_BAND_SHARE
    min_gain_fraction: float = DEFAULT_MIN_GAIN_FRACTION
    """Break-even floor under every collapse ``anchored_min_gain`` would make.

    The one setting that row exists to measure, and it was unreachable: the builder took the
    constructor's default, so the pair ``anchored``/``anchored_min_gain`` could only ever be
    compared at one value of the thing that separates them.
    """
    trigger_fraction: float = DEFAULT_TRIGGER_FRACTION
    """Share of the input budget at which ``tool_summary_anchored`` asks for its record.

    Reaches both halves of that strategy from here: the run hands it to the strategy, and the
    middleware takes the strategy's own value rather than a second copy, so the ask and the
    wait cannot be configured apart.
    """
    fallback_fraction: float = DEFAULT_FALLBACK_FRACTION
    """Share at which ``tool_summary_anchored`` stops waiting and compacts without a record.

    Must exceed ``trigger_fraction``; the strategy raises ``ValueError`` when it does not.
    """
    coverage_share: float = DEFAULT_COVERAGE_SHARE
    """Share of a group's distinctive values a record must quote before the group may be cut.

    The dial on the coverage check, whose default is a threshold rather than a derivation and
    whose right value depends on how many values a workload's results carry.
    """
    token_budget_fraction: float = 0.5
    summarizer: SupportsChatGetResponse[Any] | None = None

    @property
    def input_budget_tokens(self) -> int:
        """Tokens available for input once the output reservation is deducted."""
        return self.max_context_window_tokens - self.max_output_tokens

    @property
    def composed_budget_tokens(self) -> int:
        """Token ceiling every ``token_budget_*`` variant compacts down to.

        Shared across the variants on purpose. They differ only in the order they delete
        things, so holding the ceiling fixed is what makes their correctness scores
        comparable: any difference is attributable to *what* each discarded, not how much.
        """
        return max(int(self.input_budget_tokens * self.token_budget_fraction), 1)


def resolve_context_window(
    transcript_tokens: int,
    *,
    override: int | None = None,
    max_output_tokens: int = 512,
) -> int:
    """Return the simulated context window to compact against.

    Args:
        transcript_tokens: Approximate prompt size of the fully replayed transcript.

    Keyword Args:
        override: Explicit window size. When given, it is used verbatim.
        max_output_tokens: Output reservation, used only to enforce a sane lower bound.

    Returns:
        A window size that guarantees compaction triggers part-way through the transcript.
    """
    if override is not None:
        return override
    scaled = int(transcript_tokens * _AUTO_WINDOW_FRACTION)
    return max(scaled, _MIN_AUTO_WINDOW_TOKENS, max_output_tokens * 2)


def _build_none(options: StrategyOptions) -> CompactionStrategy | None:
    """Return no strategy, establishing the uncompacted baseline."""
    return None


def _context_window(options: StrategyOptions, *, eviction: float, truncation: float) -> CompactionStrategy:
    """Return the harness default strategy at explicit trigger thresholds."""
    return ContextWindowCompactionStrategy(
        max_context_window_tokens=options.max_context_window_tokens,
        max_output_tokens=options.max_output_tokens,
        tokenizer=options.tokenizer,
        tool_eviction_threshold=eviction,
        truncation_threshold=truncation,
        keep_last_tool_call_groups=options.keep_last_tool_call_groups,
    )


def _build_context_window(options: StrategyOptions) -> CompactionStrategy:
    """Return the harness default: shipped thresholds of 0.5 and 0.8.

    This row is meant to stand for what ``create_harness_agent`` actually installs, so its
    ``keep_last_tool_call_groups`` has to match the framework's default of 4 rather than
    being set locally. The harness passes no value, so it inherits that default; a lab
    override would quietly make this row harsher than the configuration it claims to
    represent, and every conclusion drawn about "the shipped default" would be about
    something else.
    """
    return _context_window(options, eviction=0.5, truncation=0.8)


def _build_context_window_aggressive(options: StrategyOptions) -> CompactionStrategy:
    """Return the harness default compacting early, at 0.3 and 0.5 of the input budget."""
    return _context_window(options, eviction=0.3, truncation=0.5)


def _build_context_window_lazy(options: StrategyOptions) -> CompactionStrategy:
    """Return the harness default compacting late, at 0.7 and 0.95 of the input budget."""
    return _context_window(options, eviction=0.7, truncation=0.95)


def _build_anchored(options: StrategyOptions) -> CompactionStrategy:
    """Return the lab's own strategy, designed against what the other rows measured.

    Its ceiling is the full input budget rather than a fraction of it, because unlike the
    threshold-driven strategies it does not need headroom to trip: it collapses the middle
    band from the first turn there is one, and only removes groups outright when shortening
    has not brought the prompt under. See :mod:`.compaction._anchored`.
    """
    return AnchoredCompactionStrategy(
        max_input_tokens=options.input_budget_tokens,
        tokenizer=options.tokenizer,
        keep_head_groups=options.keep_head_groups,
        keep_tail_groups=options.keep_tail_groups,
        keep_tokens=options.keep_tokens,
        band_share=options.band_share,
    )


def _build_anchored_no_assistant(options: StrategyOptions) -> CompactionStrategy:
    """Return the anchored strategy forbidden from touching assistant narration.

    Pairs with ``anchored`` to isolate the last-resort step. With the harness's default
    instructions the model restates tool values in its prose, so that prose can be the only
    surviving copy of a result that has already been shortened; this row measures what
    dropping it costs.
    """
    return AnchoredCompactionStrategy(
        max_input_tokens=options.input_budget_tokens,
        tokenizer=options.tokenizer,
        keep_head_groups=options.keep_head_groups,
        keep_tail_groups=options.keep_tail_groups,
        keep_tokens=options.keep_tokens,
        band_share=options.band_share,
        collapse_assistant_text=False,
    )


def _build_anchored_min_gain(options: StrategyOptions) -> CompactionStrategy:
    """Return the anchored strategy with a break-even floor under every collapse.

    Pairs with ``anchored`` to measure one setting: whether declining collapses too small to
    repay the prompt cache they invalidate is worth the information they would have removed.
    What the pair has actually shown so far is that the floor's effect reverses between cells
    -- the floored row kept fewer facts than its parent at one tool share and more at another --
    which the review of 2 September traced to retention being path-dependent rather than to the
    floor. Both defects are fixed; the pair has not been re-measured since. See
    :mod:`.compaction._anchored`.
    """
    return MinimumGainAnchoredCompactionStrategy(
        max_input_tokens=options.input_budget_tokens,
        tokenizer=options.tokenizer,
        keep_head_groups=options.keep_head_groups,
        keep_tail_groups=options.keep_tail_groups,
        keep_tokens=options.keep_tokens,
        band_share=options.band_share,
        min_gain_fraction=options.min_gain_fraction,
    )


def _build_tool_summary_anchored(options: StrategyOptions) -> CompactionStrategy:
    """Return the record-then-drop strategy.

    Needs no summarizer client of its own: the recording is done by the agent's own model
    through a tool call the provider issues. That is also why a run using it cannot pin
    ``tool_choice`` -- the model has to be free to choose the recall tool.

    The fallback is built here rather than left to the strategy's own default. The default is
    the same object with the same head and tail, but it takes the anchored strategy's *own*
    defaults for ``band_share`` and ``keep_tokens``, so a sweep moving either of those moved
    every anchored row except the one hiding inside this one -- and this row falls back often
    enough that the difference is measured rather than theoretical.
    """
    return ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=options.input_budget_tokens,
        tokenizer=options.tokenizer,
        keep_head_groups=options.keep_head_groups,
        keep_tail_groups=options.keep_tail_groups,
        trigger_fraction=options.trigger_fraction,
        fallback_fraction=options.fallback_fraction,
        coverage_share=options.coverage_share,
        fallback=_build_anchored(options),
    )


def _build_truncation(options: StrategyOptions) -> CompactionStrategy:
    """Return oldest-first truncation triggering at 80% of the input budget."""
    budget = options.input_budget_tokens
    return TruncationStrategy(
        max_n=max(int(budget * 0.8), 1),
        compact_to=max(int(budget * 0.5), 1),
        tokenizer=options.tokenizer,
    )


def _build_sliding_window(options: StrategyOptions) -> CompactionStrategy:
    """Return a fixed-size window over the most recent message groups."""
    return SlidingWindowStrategy(keep_last_groups=options.keep_last_groups)


def _build_tool_result(options: StrategyOptions) -> CompactionStrategy:
    """Return tool-result eviction, which rewrites history in place instead of dropping it."""
    return ToolResultCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups)


def _build_selective_tool_call(options: StrategyOptions) -> CompactionStrategy:
    """Return selective removal of older tool-call groups."""
    return SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups)


def _build_summarization(options: StrategyOptions) -> CompactionStrategy:
    """Return LLM summarization of older turns.

    Raises:
        ValueError: If no summarizer client was configured.
    """
    if options.summarizer is None:
        raise ValueError(
            "The 'summarization' strategy needs a summarizer client. "
            "Pass --summarizer-provider to select one, or drop this strategy from the run."
        )
    return SummarizationStrategy(
        client=options.summarizer,
        target_count=options.keep_last_groups,
        tokenizer=options.tokenizer,
    )


def _truncation_at(options: StrategyOptions, budget: int) -> CompactionStrategy:
    """Return truncation targeting a composed strategy's budget."""
    return TruncationStrategy(max_n=budget, compact_to=max(int(budget * 0.8), 1), tokenizer=options.tokenizer)


def _composed(options: StrategyOptions, parts: list[CompactionStrategy]) -> CompactionStrategy:
    """Return an ordered composition run against the shared token ceiling.

    ``TokenBudgetComposedStrategy`` runs each part in turn, re-counting tokens after every
    one and stopping as soon as the ceiling is met. Whatever the parts fail to remove, its
    built-in fallback removes by evicting oldest groups. That fallback is why every variant
    lands at the same size, and why the interesting difference between them is which
    messages they chose to spend the budget on.
    """
    return TokenBudgetComposedStrategy(
        token_budget=options.composed_budget_tokens,
        tokenizer=options.tokenizer,
        strategies=parts,
    )


def _build_token_budget_fallback(options: StrategyOptions) -> CompactionStrategy:
    """Return the composed strategy with no parts at all.

    The control for the whole ``token_budget_*`` family: every removal is done by the
    built-in oldest-first fallback. A variant that cannot beat this is contributing
    nothing over plain age-ordered eviction at the same size.
    """
    return _composed(options, [])


def _build_token_budget_tools_first(options: StrategyOptions) -> CompactionStrategy:
    """Return a composition that sheds tool bulk before it sheds history."""
    return _composed(
        options,
        [
            ToolResultCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups),
            SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups),
            _truncation_at(options, options.composed_budget_tokens),
        ],
    )


def _build_token_budget_truncate_first(options: StrategyOptions) -> CompactionStrategy:
    """Return a composition that sheds age before it sheds tool bulk.

    The mirror of ``token_budget_tools_first``. Same parts, opposite order, same ceiling,
    so the pair isolates whether ordering alone changes what survives.
    """
    return _composed(
        options,
        [
            _truncation_at(options, options.composed_budget_tokens),
            ToolResultCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups),
        ],
    )


def _build_token_budget_window_first(options: StrategyOptions) -> CompactionStrategy:
    """Return a composition that applies a hard recency window before trimming by tokens."""
    return _composed(
        options,
        [
            SlidingWindowStrategy(keep_last_groups=options.keep_last_groups),
            _truncation_at(options, options.composed_budget_tokens),
        ],
    )


def _build_token_budget_summarize(options: StrategyOptions) -> CompactionStrategy:
    """Return a composition that summarizes rather than deletes once tool bulk is gone.

    The only variant that can carry information past the ceiling instead of dropping it,
    and the only one that spends money to do so.

    Raises:
        ValueError: If no summarizer client was configured.
    """
    if options.summarizer is None:
        raise ValueError(
            "The 'token_budget_summarize' strategy needs a summarizer client. "
            "Pass --summarizer-provider to select one, or drop this strategy from the run."
        )
    return _composed(
        options,
        [
            ToolResultCompactionStrategy(keep_last_tool_call_groups=options.keep_last_tool_call_groups),
            SummarizationStrategy(
                client=options.summarizer,
                target_count=options.keep_last_groups,
                tokenizer=options.tokenizer,
            ),
        ],
    )


STRATEGY_BUILDERS: Final[dict[str, Callable[[StrategyOptions], CompactionStrategy | None]]] = {
    "none": _build_none,
    "context_window": _build_context_window,
    "context_window_aggressive": _build_context_window_aggressive,
    "context_window_lazy": _build_context_window_lazy,
    "truncation": _build_truncation,
    "anchored": _build_anchored,
    "tool_summary_anchored": _build_tool_summary_anchored,
    "anchored_no_assistant": _build_anchored_no_assistant,
    "anchored_min_gain": _build_anchored_min_gain,
    "sliding_window": _build_sliding_window,
    "tool_result": _build_tool_result,
    "selective_tool_call": _build_selective_tool_call,
    "summarization": _build_summarization,
    "token_budget_fallback": _build_token_budget_fallback,
    "token_budget_tools_first": _build_token_budget_tools_first,
    "token_budget_truncate_first": _build_token_budget_truncate_first,
    "token_budget_window_first": _build_token_budget_window_first,
    "token_budget_summarize": _build_token_budget_summarize,
}


#: Strategies that cannot be built without a summarizer client. Named explicitly rather than
#: detected by looking for "summar" in the name: that convention silently required a client
#: for a strategy that does its recording through the agent's own tool loop, and would just as
#: silently fail to require one for a summarizing strategy named otherwise.
STRATEGIES_NEEDING_SUMMARIZER: Final[frozenset[str]] = frozenset({"summarization", "token_budget_summarize"})


def needs_summarizer(names: Iterable[str]) -> bool:
    """Return whether any of ``names`` requires a summarizer client.

    Args:
        names: Strategy names selected for a run.

    Returns:
        True when at least one needs a client.
    """
    return any(name in STRATEGIES_NEEDING_SUMMARIZER for name in names)


def strategy_names() -> list[str]:
    """Return every selectable strategy name."""
    return list(STRATEGY_BUILDERS)


def build_strategy(name: str, options: StrategyOptions) -> CompactionStrategy | None:
    """Build the named strategy.

    Args:
        name: One of the keys of ``STRATEGY_BUILDERS``.
        options: Shared budget and tokenizer parameters.

    Returns:
        The strategy, or ``None`` for the uncompacted ``none`` baseline.

    Raises:
        KeyError: If ``name`` is not a known strategy.
    """
    if name not in STRATEGY_BUILDERS:
        raise KeyError(f"Unknown strategy {name!r}. Known strategies: {sorted(STRATEGY_BUILDERS)}")
    return STRATEGY_BUILDERS[name](options)
