# Copyright (c) Microsoft. All rights reserved.

"""The compaction strategies under test.

Each entry wraps a strategy from ``agent_framework`` so that the benchmark can select it
by name. ``context_window`` is the strategy the agent harness installs by default when
``create_harness_agent`` is given ``max_context_window_tokens``; the ``*_aggressive`` and
``*_lazy`` variants are the same strategy at different trigger thresholds and exist to
answer whether compacting early and often costs more in lost cache reads than it saves in
prompt tokens.

Budgets are sized relative to the transcript rather than to a model's real context window.
A 20-turn transcript never approaches a 128k window, so a real window would mean no
strategy ever fires and the benchmark would measure nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from agent_framework import (
    CompactionStrategy,
    ContextWindowCompactionStrategy,
    SelectiveToolCallCompactionStrategy,
    SlidingWindowStrategy,
    SummarizationStrategy,
    TokenizerProtocol,
    ToolResultCompactionStrategy,
    TruncationStrategy,
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
    """Parameters shared by every strategy builder."""

    tokenizer: TokenizerProtocol
    max_context_window_tokens: int
    max_output_tokens: int
    keep_last_groups: int = 6
    keep_last_tool_call_groups: int = 2
    summarizer: SupportsChatGetResponse[Any] | None = None

    @property
    def input_budget_tokens(self) -> int:
        """Tokens available for input once the output reservation is deducted."""
        return self.max_context_window_tokens - self.max_output_tokens


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
    """Return the harness default at its shipped thresholds of 0.5 and 0.8."""
    return _context_window(options, eviction=0.5, truncation=0.8)


def _build_context_window_aggressive(options: StrategyOptions) -> CompactionStrategy:
    """Return the harness default compacting early, at 0.3 and 0.5 of the input budget."""
    return _context_window(options, eviction=0.3, truncation=0.5)


def _build_context_window_lazy(options: StrategyOptions) -> CompactionStrategy:
    """Return the harness default compacting late, at 0.7 and 0.95 of the input budget."""
    return _context_window(options, eviction=0.7, truncation=0.95)


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


STRATEGY_BUILDERS: Final[dict[str, Callable[[StrategyOptions], CompactionStrategy | None]]] = {
    "none": _build_none,
    "context_window": _build_context_window,
    "context_window_aggressive": _build_context_window_aggressive,
    "context_window_lazy": _build_context_window_lazy,
    "truncation": _build_truncation,
    "sliding_window": _build_sliding_window,
    "tool_result": _build_tool_result,
    "selective_tool_call": _build_selective_tool_call,
    "summarization": _build_summarization,
}


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
