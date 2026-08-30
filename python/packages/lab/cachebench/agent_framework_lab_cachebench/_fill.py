# Copyright (c) Microsoft. All rights reserved.

"""Size the seeded conversation to a stated fraction of a stated context limit.

A cell in this matrix is "fill fraction X of a tried limit L". X decides how much irrelevant
context surrounds a fixed set of facts, so it is only a variable if the conversation actually
lands on X x L -- and only a *clean* variable if the payload it surrounds is held constant
while it moves.

The sizing is analytic rather than adaptive. The generator knows its own payload and filler
sizes and holds a tokenizer, so it can solve for the filler directly instead of running a
conversation, measuring it and adjusting. That is not merely cheaper: an adaptive loop would
have to calibrate against one strategy's run and then apply the result to the others, which
makes the user-side turn list depend on the order the strategies were measured in. Solving it
up front keeps that list identical across strategies by construction.

What cannot be solved for is the model's own replies, which are written live and are the one
term here that is assumed rather than computed. That is why the achieved fill is recorded on
the uncompacted run and checked against the target, rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Final

from ._live import DEFAULT_TOOL_RESULT_TOKENS, build_live_scenario, make_scope_tools, resolve_instructions
from ._recall import RecallScenario

if TYPE_CHECKING:
    from agent_framework import TokenizerProtocol

__all__ = ["ASSUMED_REPLY_TOKENS", "FillPlan", "plan_fill"]

ASSUMED_REPLY_TOKENS: Final[int] = 150
"""Tokens assumed for each reply the model writes during seeding.

The one term in the estimate that cannot be computed, because the replies are the model's.
150 is what this workload was measured averaging. It matters less than it looks: on a
40-turn seed it is about 6,000 tokens, so a reply that is half or double this moves the
achieved fill by a few points -- which is exactly why :func:`plan_fill` records the target and
the runner reports the deviation instead of assuming the target was hit.
"""

_FUNCTION_CALL_TOKENS: Final[int] = 20
"""Serialized size of one assistant function-call message.

``{"call_id": ..., "name": "lookup_early", "arguments": "{}"}`` and nothing else: the scope is
pinned by which function is called, so no arguments travel.
"""

#: Filler turns per section, and there are three sections. The scenario generator places
#: ``filler_turns // 3`` pairs in each of them, so the count only moves in steps of three and
#: asking for anything else silently rounds down.
_SECTIONS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class FillPlan:
    """How to build a seeded conversation that lands on a stated share of a context limit."""

    filler_turns: int
    filler_tokens: int
    context_limit: int
    """The limit this cell stands in for. Simulated: the model itself accepts 272,000."""
    fill_fraction: float
    target_tokens: int
    predicted_tokens: int
    """What the plan expects the seeded conversation to reach, replies included."""
    payload_tokens: int
    """Everything the conversation costs before any filler: the instructions, the turns that
    plant facts, every tool result, and the replies. The floor the fill has to clear."""
    tool_payload_tokens: int
    """The tool results alone, which is the part varied between runs and compared across them."""

    @property
    def deviation(self) -> float:
        """Predicted fill against the target, as a signed fraction of the target."""
        return (self.predicted_tokens - self.target_tokens) / self.target_tokens if self.target_tokens else 0.0


def _tool_result_texts(
    scenario: RecallScenario,
    *,
    tool_result_tokens: int,
    narration: str,
    fact_placement: str,
) -> list[str]:
    """Return what each lookup tool actually hands back.

    Built from the same factory the run uses rather than from the scenario's scripted tool
    results, which live runs never send. Sizing against the scripted ones would miss the
    payload by whatever the two differ by, and they differ by the whole placement scheme.

    Returns:
        One result string per scope, in scope order.
    """
    return [tool() for tool in make_scope_tools(scenario.tool_lookups, tool_result_tokens, narration, fact_placement)]


def _measure(
    scenario: RecallScenario,
    *,
    tokenizer: TokenizerProtocol,
    instructions: str,
    tool_result_tokens: int,
    narration: str,
    fact_placement: str,
    reply_tokens: int,
) -> tuple[int, int]:
    """Estimate the seeded conversation's size, and the tool results' share of it.

    Counts the material this package generates -- instructions, user turns, tool results, the
    function-call messages -- and assumes a flat size for the replies. It does not model the
    provider's per-message framing or the tool schemas, which together run a few hundred
    tokens: two orders below the tolerance the achieved fill is judged against, and both
    included in the achieved figure anyway.

    The closing questions are excluded. They are asked of the snapshot, one at a time, so they
    are not part of the context whose size is being set.

    Args:
        scenario: The scenario to size.

    Keyword Args:
        tokenizer: Token counter, the same one the strategies budget with.
        instructions: The agent instructions, which travel with every request.
        tool_result_tokens: Requested size of each tool result.
        narration: How hard the tool result asks for its values to be restated.
        fact_placement: Where the codes sit inside each result.
        reply_tokens: Assumed size of each reply the model writes.

    Returns:
        The estimated total, and the tool results' contribution to it.
    """
    turns = scenario.transcript.turns
    seed_turns = turns[: len(turns) - max(scenario.answer_turn_count, 1)]
    total = tokenizer.count_tokens(instructions)
    for turn in seed_turns:
        for message in turn.request:
            for content in message.contents:
                text = getattr(content, "text", None)
                if text:
                    total += tokenizer.count_tokens(text)
    total += len(seed_turns) * reply_tokens

    results = _tool_result_texts(
        scenario, tool_result_tokens=tool_result_tokens, narration=narration, fact_placement=fact_placement
    )
    by_scope = dict(zip(sorted(scenario.tool_lookups), results, strict=False))
    tool_total = 0
    for index, scope in scenario.tool_turn_scopes.items():
        if index >= len(seed_turns):
            continue
        tool_total += tokenizer.count_tokens(by_scope.get(scope, "")) + _FUNCTION_CALL_TOKENS
    return total + tool_total, tool_total


def plan_fill(
    *,
    tokenizer: TokenizerProtocol,
    context_limit: int,
    fill_fraction: float,
    salt: str = "plan",
    tool_turns: int = 6,
    filler_tool_turns: int = 0,
    markers_per_tool: int = 2,
    tool_result_tokens: int = DEFAULT_TOOL_RESULT_TOKENS,
    narration: str = "neutral",
    fact_placement: str = "spread",
    retrieval_guidance: bool = True,
    subset_questions: bool = True,
    filler_turn_tokens: int = 2_000,
    reply_tokens: int = ASSUMED_REPLY_TOKENS,
) -> FillPlan:
    """Solve for the filler that makes a seeded conversation reach ``fill_fraction`` of a limit.

    Filler is the dial and the payload is fixed. The number of filler turns is the coarse
    setting and their size the fine one, in that order on purpose: holding the size near a
    realistic 2,000 tokens and adding turns is what a longer conversation looks like, whereas
    holding the count and inflating each turn produces a handful of 40,000-token messages that
    no strategy would meet in practice.

    Keyword Args:
        tokenizer: Token counter, the same one the strategies budget with.
        context_limit: The limit this cell stands in for.
        fill_fraction: Share of that limit the seeded conversation should reach.
        salt: Cell-unique string for the probe scenarios built while solving. Immaterial to
            the answer, since markers are fixed-width whatever the salt.
        tool_turns: Tool-call groups to plant.
        filler_tool_turns: Extra lookups whose results carry no codes.
        markers_per_tool: Verifiable codes each tool result carries.
        tool_result_tokens: Requested size of each tool result.
        narration: How hard the scenario pushes the model to restate tool values.
        fact_placement: Where the codes sit inside each result.
        retrieval_guidance: Whether the instructions carry the retrieval clause, which is part
            of every prompt and so part of the size.
        subset_questions: Whether the run closes with several targeted questions.
        filler_turn_tokens: Size each filler turn should sit near. The solver picks how many
            turns from this and then adjusts their size to land exactly.
        reply_tokens: Assumed size of each reply. See :data:`ASSUMED_REPLY_TOKENS`.

    Returns:
        The plan, including what it predicts and what it is aiming at.

    Raises:
        ValueError: If the fraction is out of range, or if the payload alone does not fit
            inside the target -- in which case this cell cannot be built at all and no amount
            of adjusting the filler will change that.
    """
    if not 0.0 < fill_fraction <= 1.0:
        raise ValueError(f"fill_fraction must be in (0.0, 1.0]; got {fill_fraction}.")
    target = round(context_limit * fill_fraction)
    instructions = resolve_instructions(narration, retrieval_guidance=retrieval_guidance)

    def size(groups: int, filler_tokens: int) -> tuple[int, int]:
        """Build one candidate conversation and estimate it.

        Returns:
            The estimated total, and the tool results' contribution to it.
        """
        scenario = build_live_scenario(
            salt=salt,
            filler_turns=groups * _SECTIONS,
            filler_tokens=max(filler_tokens, 1),
            tool_turns=tool_turns,
            filler_tool_turns=filler_tool_turns,
            markers_per_tool=markers_per_tool,
            narration=narration,
            subset_questions=subset_questions,
        )
        return _measure(
            scenario,
            tokenizer=tokenizer,
            instructions=instructions,
            tool_result_tokens=tool_result_tokens,
            narration=narration,
            fact_placement=fact_placement,
            reply_tokens=reply_tokens,
        )

    # The extra lookups and the code-free asides are placed *inside* the filler sections, one
    # per section iteration, so too few filler turns silently drops them: 16 tool turns asked
    # for with the default 6 filler turns yielded 9, and the payload was quietly a different
    # payload. The floor here is what it takes to place every one of them.
    per_section = max(tool_turns - _SECTIONS, filler_tool_turns, 0)
    minimum_groups = max(ceil(per_section / _SECTIONS), 1)

    floor_tokens, tool_tokens = size(minimum_groups, 1)
    if floor_tokens >= target:
        raise ValueError(
            f"The payload does not fit in this cell: {floor_tokens:,} tokens of instructions, "
            f"fact-bearing turns and tool results against a target of {target:,} "
            f"({fill_fraction:.0%} of {context_limit:,}). The payload is held fixed across the "
            "matrix and must fit inside the smallest cell, so reduce --tool-turns, "
            "--tool-result-tokens or --markers-per-tool rather than raising the fill here, "
            "which would make this cell incomparable with the others."
        )

    groups = max(minimum_groups, round((target - floor_tokens) / (filler_turn_tokens * _SECTIONS)))
    pairs = groups * _SECTIONS
    empty_tokens, _ = size(groups, 1)
    # Two points on a line: the filler is generated word by word to a character target, so its
    # token count is proportional to the size asked for but not equal to it. Measuring the
    # ratio is what keeps this correct under a tokenizer other than the one it was written
    # against -- the character estimator counts this vocabulary at twice tiktoken's rate.
    probe_size = 500
    probed_tokens, _ = size(groups, probe_size)
    per_token = max((probed_tokens - empty_tokens) / (pairs * probe_size), 1e-6)
    filler_tokens = max(round((target - empty_tokens) / (pairs * per_token)), 1)

    predicted, tool_tokens = size(groups, filler_tokens)
    # One correction, because the two-point fit is not exact and a small systematic error on a
    # 230,000-token conversation is worth a second build. A third pass has never moved it.
    if predicted != target and pairs:
        filler_tokens = max(filler_tokens + round((target - predicted) / (pairs * per_token)), 1)
        predicted, tool_tokens = size(groups, filler_tokens)

    return FillPlan(
        filler_turns=groups * _SECTIONS,
        filler_tokens=filler_tokens,
        context_limit=context_limit,
        fill_fraction=fill_fraction,
        target_tokens=target,
        predicted_tokens=predicted,
        payload_tokens=floor_tokens,
        tool_payload_tokens=tool_tokens,
    )
