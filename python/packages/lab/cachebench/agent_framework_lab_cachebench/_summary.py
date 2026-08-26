# Copyright (c) Microsoft. All rights reserved.

"""Judge compaction on both axes at once: what it costs and what it destroys.

Measuring the two separately invites the wrong conclusion. The cost sweep says a strategy
is cheap; the recall probe says it is lossy; neither on its own tells you whether to use
it. Worse, running them on different conversations means the numbers describe different
workloads and cannot honestly be placed in the same table.

So both are measured on one conversation: every turn is actually sent, which gives real
token usage across the whole session, and the final turn is scored for whether the model
could still use the facts planted throughout. The recommendation is then the obvious one —
the cheapest strategy that does not degrade the answer — with correctness judged relative
to an uncompacted control rather than against a presumed-perfect 100%, because the model
sometimes overlooks a fact even when everything is in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from ._recall import RecallScore

__all__ = ["JointOutcome", "JointVerdict", "recommend"]

#: Fraction of the control's correctness a strategy must retain to be considered usable.
#: Judged relative to the control because an uncompacted run does not always score 100%:
#: a model that overlooks one fact on its own would otherwise make every strategy look
#: worse than it is.
DEFAULT_MIN_CORRECTNESS: Final[float] = 0.9


@dataclass(frozen=True, slots=True)
class JointOutcome:
    """One strategy measured on both axes over the same conversation."""

    strategy: str
    cost: float
    input_tokens: int
    cached_tokens: int
    messages_left: int
    messages_total: int
    score: RecallScore

    @property
    def correctness(self) -> float:
        """Share of correctness checks the final answer passed."""
        return self.score.correctness_score

    @property
    def hit_rate(self) -> float | None:
        """Share of input tokens served from the provider's cache."""
        if self.input_tokens <= 0:
            return None
        return self.cached_tokens / self.input_tokens


@dataclass(frozen=True, slots=True)
class JointVerdict:
    """The recommendation across both axes."""

    recommended: str
    baseline: JointOutcome
    chosen: JointOutcome
    outcomes: tuple[JointOutcome, ...]
    min_correctness: float
    rationale: str

    @property
    def saving_fraction(self) -> float:
        """What the recommendation saves against not compacting."""
        if self.baseline.cost <= 0:
            return 0.0
        return (self.baseline.cost - self.chosen.cost) / self.baseline.cost


def relative_correctness(outcome: JointOutcome, baseline: JointOutcome) -> float:
    """Return a strategy's correctness as a fraction of the uncompacted control's.

    Args:
        outcome: The strategy being judged.
        baseline: The uncompacted control.

    Returns:
        The ratio, or 0.0 when the control itself scored nothing.
    """
    if baseline.correctness <= 0:
        return 0.0
    return outcome.correctness / baseline.correctness


def recommend(
    outcomes: list[JointOutcome],
    *,
    baseline: str = "none",
    min_correctness: float = DEFAULT_MIN_CORRECTNESS,
) -> JointVerdict:
    """Recommend the cheapest strategy that keeps the answer intact.

    Cost alone would pick whichever strategy trims hardest, which is reliably the one that
    destroys the most information. Correctness alone would always pick the control. The
    useful question is the constrained one: among the options that still answer correctly,
    which is cheapest.

    Args:
        outcomes: Every strategy measured on the same conversation.

    Keyword Args:
        baseline: The uncompacted control strategy.
        min_correctness: Fraction of the control's correctness a strategy must retain to
            be eligible.

    Returns:
        The verdict.

    Raises:
        ValueError: If no outcomes were supplied or the control is missing.
    """
    if not outcomes:
        raise ValueError("No measured strategies to compare.")
    by_name = {outcome.strategy: outcome for outcome in outcomes}
    if baseline not in by_name:
        raise ValueError(f"Baseline strategy {baseline!r} is missing; measured: {sorted(by_name)}")

    base = by_name[baseline]
    eligible = [
        outcome
        for outcome in outcomes
        if outcome.strategy != baseline and relative_correctness(outcome, base) >= min_correctness
    ]
    cheapest_overall = min((o for o in outcomes if o.strategy != baseline), key=lambda o: o.cost, default=base)

    if not eligible:
        chosen = base
        rationale = (
            f"No compaction strategy retained {min_correctness:.0%} of the control's correctness. "
            f"The cheapest, {cheapest_overall.strategy!r}, saves "
            f"{(base.cost - cheapest_overall.cost) / base.cost:.0%} but answers at "
            f"{relative_correctness(cheapest_overall, base):.0%} of the control. Compact only to avoid "
            "overflowing the context window, not to save money."
        )
    else:
        chosen = min(eligible, key=lambda outcome: outcome.cost)
        if chosen.cost >= base.cost:
            chosen = base
            rationale = (
                "Every strategy that keeps the answer intact also costs more than not compacting. "
                "Leave it off unless the conversation would overflow the window."
            )
        else:
            saving = (base.cost - chosen.cost) / base.cost
            rationale = (
                f"{chosen.strategy!r} is {saving:.0%} cheaper than not compacting while answering at "
                f"{relative_correctness(chosen, base):.0%} of the control. "
                f"{len(eligible)} of {len(outcomes) - 1} strategies cleared the correctness bar."
            )

    return JointVerdict(
        recommended=chosen.strategy,
        baseline=base,
        chosen=chosen,
        outcomes=tuple(sorted(outcomes, key=lambda outcome: outcome.cost)),
        min_correctness=min_correctness,
        rationale=rationale,
    )
