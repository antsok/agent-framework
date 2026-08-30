# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the analytic fill sizing.

The sizing is a claim about a conversation that has not been run yet, so it is checked here
against a stub that bills what it was actually sent. That closes the loop offline: the plan
counts the material the generator is about to produce, the stub counts the material the agent
actually sent, and the two are independent walks over the same conversation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message, UsageDetails
from agent_framework_lab_cachebench import (
    FillPlan,
    ProviderRuntime,
    StrategyOptions,
    build_live_scenario,
    plan_fill,
    run_live,
)
from agent_framework_lab_cachebench._fill import _FUNCTION_CALL_TOKENS
from agent_framework_lab_cachebench._live_cli import FILL_TOLERANCE
from agent_framework_lab_cachebench._transcripts import filler_text
from test_live import StubChatClient

TOKENIZER = CharacterEstimatorTokenizer()

#: A reply large enough to matter in the arithmetic. The replies are the one term the sizing
#: cannot compute, so a stub that answered in three words would let a plan that ignored them
#: pass.
REPLY = filler_text(11, 600)
REPLY_TOKENS = TOKENIZER.count_tokens(REPLY)


class SizingStubClient(StubChatClient):
    """A stub that bills what it was sent, counted the way the plan counts it.

    Not the same code as the plan: this walks the messages the agent actually assembled, while
    the plan walks the scenario it is about to generate. A sizing error shows up as a
    disagreement between the two.
    """

    def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any], **kwargs: Any
    ) -> Any:
        total = TOKENIZER.count_tokens(str(options.get("instructions") or ""))
        for message in messages:
            for content in message.contents:
                if content.type == "function_call":
                    total += _FUNCTION_CALL_TOKENS
                elif content.type == "function_result":
                    total += TOKENIZER.count_tokens(str(content.result))
                elif text := getattr(content, "text", None):
                    total += TOKENIZER.count_tokens(text)
        # Set before the response is built, and read when it is awaited a moment later. Calls
        # are strictly sequential here, so there is no window in which the two disagree.
        self.usage = UsageDetails(input_token_count=total, output_token_count=REPLY_TOKENS)
        return super()._inner_get_response(messages=messages, stream=stream, options=options, **kwargs)


def _plan(fill: float, **kwargs: Any) -> FillPlan:
    settings: dict[str, Any] = {
        "tool_turns": 6,
        "tool_result_tokens": 500,
        "filler_turn_tokens": 800,
        "reply_tokens": REPLY_TOKENS,
        **kwargs,
    }
    return plan_fill(tokenizer=TOKENIZER, context_limit=40_000, fill_fraction=fill, **settings)


def test_the_plan_predicts_the_target_it_was_given() -> None:
    """The arithmetic must close before anything is run.

    A plan that misses here misses by the same amount on every strategy, so the cell is
    labelled with a fill fraction it never reached and sits on a different axis from its
    neighbours without anything saying so.
    """
    plan = _plan(0.5)

    assert plan.target_tokens == 20_000
    assert abs(plan.deviation) < 0.01


async def test_the_seeded_conversation_lands_on_the_target() -> None:
    """The conversation an uncompacted run actually sends must reach the planned size.

    This is the claim the fill fraction rests on, and it is checked end to end rather than
    against the plan's own arithmetic: the stub counts the assembled prompt, which is a
    different walk over the conversation than the one that sized it.
    """
    plan = _plan(0.5)
    scenario = build_live_scenario(
        salt="fill",
        filler_turns=plan.filler_turns,
        filler_tokens=plan.filler_tokens,
        tool_turns=6,
        narration="neutral",
    )
    outcome = await run_live(
        ProviderRuntime(client=SizingStubClient(reply=REPLY, obey_tool_choice=True), model="stub"),
        strategy_name="none",
        options=StrategyOptions(TOKENIZER, 40_000, 2_048),
        scenario=scenario,
        tool_result_tokens=500,
        narration="neutral",
        probe_repeats=1,
    )

    deviation = (outcome.seed_prompt_tokens - plan.target_tokens) / plan.target_tokens
    assert abs(deviation) <= FILL_TOLERANCE, f"seeded {outcome.seed_prompt_tokens:,} against {plan.target_tokens:,}"


def test_filler_is_the_dial_and_the_payload_is_fixed() -> None:
    """Raising the fill must add filler and leave the planted material alone.

    The fill fraction only means "how much irrelevant context surrounds a fixed set of facts"
    if the facts and their carriers are the same in every cell. A sizing that reached its
    target by enlarging the tool results would be measuring two things at once and reporting
    one number.
    """
    low, high = _plan(0.5), _plan(0.8)

    assert high.filler_turns * high.filler_tokens > low.filler_turns * low.filler_tokens
    assert high.payload_tokens == low.payload_tokens
    assert high.tool_payload_tokens == low.tool_payload_tokens


def test_a_payload_larger_than_the_smallest_cell_is_rejected() -> None:
    """A payload that does not fit cannot be filled around, and must say so.

    Silently building the cell anyway produces a conversation that overshoots its own target
    before a single filler turn is added, which reads in the table as a strategy failing to
    compact rather than as a matrix that cannot be built.
    """
    with pytest.raises(ValueError) as error:
        plan_fill(
            tokenizer=TOKENIZER,
            context_limit=10_000,
            fill_fraction=0.5,
            tool_turns=8,
            tool_result_tokens=4_000,
            reply_tokens=REPLY_TOKENS,
        )

    message = str(error.value)
    assert "payload does not fit" in message
    assert "--tool-result-tokens" in message, "the error must name what to change"
    assert "5,000" in message, "the error must state the target it did not fit inside"


def test_the_plan_leaves_room_for_every_tool_group_it_planted() -> None:
    """Extra lookups live inside the filler sections, so too little filler drops them.

    Measured before this was enforced: 16 tool turns asked for with the default 6 filler turns
    yielded 9, and the payload was quietly a different payload from the one the run was
    labelled with.
    """
    plan = plan_fill(
        tokenizer=TOKENIZER,
        context_limit=200_000,
        fill_fraction=0.5,
        tool_turns=16,
        filler_tool_turns=4,
        tool_result_tokens=500,
        reply_tokens=REPLY_TOKENS,
    )
    scenario = build_live_scenario(
        salt="groups", filler_turns=plan.filler_turns, filler_tokens=plan.filler_tokens, tool_turns=16
    )

    assert len(scenario.tool_lookups) == 16


def test_an_impossible_fill_fraction_is_refused() -> None:
    """A fraction outside (0, 1] describes no cell, and would silently size to nonsense."""
    with pytest.raises(ValueError, match="fill_fraction"):
        _plan(1.5)
