# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the compaction information-loss probe."""

from __future__ import annotations

import pytest
from agent_framework import CharacterEstimatorTokenizer, SlidingWindowStrategy, apply_compaction
from agent_framework_lab_cachebench import (
    Contradiction,
    PlantedFact,
    RecallScore,
    build_recall_scenario,
    score_answer,
)
from agent_framework_lab_cachebench._metrics import serialize_message

TOKENIZER = CharacterEstimatorTokenizer()


def test_scenario_plants_all_three_kinds_in_order() -> None:
    scenario = build_recall_scenario(salt="s", filler_turns=4, filler_tokens=100)
    kinds = [fact.kind for fact in scenario.facts]
    assert kinds.count("requirement") == 3
    assert kinds.count("correction") == 2
    assert kinds.count("tool_result") == 6
    # Requirements come first and the correction sits mid-conversation.
    assert kinds.index("requirement") < kinds.index("correction")


def test_markers_are_unique_per_salt() -> None:
    first = {fact.marker for fact in build_recall_scenario(salt="a", filler_turns=2, filler_tokens=50).facts}
    second = {fact.marker for fact in build_recall_scenario(salt="b", filler_turns=2, filler_tokens=50).facts}
    # "streaming pipeline" is shared prose; every generated code must differ.
    assert len(first & second) == 1


def test_final_turn_asks_for_everything_and_is_not_scripted() -> None:
    scenario = build_recall_scenario(salt="s", filler_turns=2, filler_tokens=50)
    final = scenario.transcript.turns[-1]
    assert final.reply == ()
    assert "verbatim" in (final.request[0].contents[0].text or "")


def test_scoring_separates_compaction_damage_from_model_failure() -> None:
    facts = (
        PlantedFact("RQ-AAA", "requirement", 1, "kept and used"),
        PlantedFact("RQ-BBB", "requirement", 1, "kept but unused"),
        PlantedFact("RQ-CCC", "requirement", 1, "evicted and unused"),
        PlantedFact("RQ-DDD", "requirement", 1, "evicted but produced anyway"),
    )
    prompt = "context contains RQ-AAA and RQ-BBB"
    answer = "answer cites RQ-AAA and RQ-DDD"
    outcomes = {outcome.fact.marker: outcome.verdict for outcome in score_answer(answer, facts, prompt)}
    assert outcomes["RQ-AAA"] == "ok"
    assert outcomes["RQ-BBB"] == "ignored_by_model"
    assert outcomes["RQ-CCC"] == "lost_to_compaction"
    # Produced without being in context: suspicious, and must not be counted as damage.
    assert outcomes["RQ-DDD"] == "recalled_without_context"


def test_matching_is_case_insensitive() -> None:
    fact = PlantedFact("RQ-AbCdEf", "requirement", 1, "")
    assert fact.appears_in("we satisfied rq-abcdef today")


def test_score_aggregates() -> None:
    facts = (
        PlantedFact("A", "requirement", 1, ""),
        PlantedFact("B", "requirement", 1, ""),
        PlantedFact("C", "tool_result", 5, ""),
    )
    score = RecallScore(
        outcomes=score_answer("A only", facts, "A and B are here"),
        answer="A only",
        messages_left=4,
        messages_total=9,
    )
    assert score.recalled == 1
    assert score.facts_left == 2
    assert score.ignored_by_model == 1  # B was available and unused
    assert score.lost_to_compaction == 1  # C was evicted
    assert score.recall_rate == pytest.approx(1 / 3)
    assert len(score.by_kind("requirement")) == 2


async def test_compaction_actually_evicts_planted_facts() -> None:
    # The probe is only meaningful if a strategy can genuinely remove a planted fact.
    scenario = build_recall_scenario(salt="s", filler_turns=8, filler_tokens=2_000)
    history = [scenario.transcript.system]
    for turn in scenario.transcript.turns[:-1]:
        history.extend(turn.request)
        await apply_compaction(history, strategy=SlidingWindowStrategy(keep_last_groups=3), tokenizer=TOKENIZER)
        history.extend(turn.reply)
    history.extend(scenario.transcript.turns[-1].request)
    projected = await apply_compaction(history, strategy=SlidingWindowStrategy(keep_last_groups=3), tokenizer=TOKENIZER)
    prompt = "\n".join(serialize_message(message) for message in projected)
    outcomes = score_answer("", scenario.facts, prompt)
    assert any(not outcome.survived for outcome in outcomes), "a narrow window must drop early facts"


async def test_uncompacted_control_keeps_every_fact() -> None:
    scenario = build_recall_scenario(salt="s", filler_turns=4, filler_tokens=500)
    history = [scenario.transcript.system]
    for turn in scenario.transcript.turns:
        history.extend(turn.request)
        history.extend(turn.reply)
    prompt = "\n".join(serialize_message(message) for message in history)
    # Without compaction nothing may be missing, or the probe would blame compaction for
    # facts the scenario never actually planted.
    assert all(outcome.survived for outcome in score_answer("", scenario.facts, prompt))


def test_tool_lookups_span_the_whole_conversation() -> None:
    # Three lookups: early, mid and late. The early one is what separates age-based
    # trimming from tool-specific eviction — with only recent lookups the tool column read
    # 4/4 for every strategy and measured nothing.
    scenario = build_recall_scenario(salt="s", filler_turns=9, filler_tokens=100)
    tool_turns = sorted({fact.turn for fact in scenario.facts if fact.kind == "tool_result"})
    assert len(tool_turns) == 3
    total = len(scenario.transcript.turns)
    assert tool_turns[0] <= 3, "the first lookup must sit in the oldest part of the history"
    assert tool_turns[-1] >= total - 2, "the last lookup must sit just before the question"
    correction_turn = next(fact.turn for fact in scenario.facts if fact.kind == "correction")
    assert tool_turns[0] < correction_turn < tool_turns[-1]


def test_each_lookup_carries_two_distinct_facts() -> None:
    scenario = build_recall_scenario(salt="s", filler_turns=6, filler_tokens=100)
    markers = [fact.marker for fact in scenario.facts if fact.kind == "tool_result"]
    assert len(markers) == len(set(markers)) == 6


PIPELINE = Contradiction("batch pipeline", "streaming pipeline", "retracted choice")


@pytest.mark.parametrize(
    ("answer", "asserted"),
    [
        ("We will use the streaming pipeline.", False),
        # Naming both is correct: the answer explains what was rejected.
        ("Not the batch pipeline — we switched to the streaming pipeline.", False),
        # Naming only the retracted one is the failure this catches.
        ("We will use the batch pipeline.", True),
        ("No pipeline mentioned at all.", False),
    ],
)
def test_contradiction_detects_only_the_retracted_assertion(answer: str, asserted: bool) -> None:
    assert PIPELINE.asserted_in(answer) is asserted


def test_full_recall_is_not_correctness() -> None:
    # The distinction the report hinges on: an answer can contain every planted marker and
    # still state the decision that was explicitly retracted.
    facts = (PlantedFact("RQ-1", "requirement", 1, ""), PlantedFact("FB-1", "correction", 5, ""))
    answer = "Requirement RQ-1, reference FB-1, proceeding with the batch pipeline."
    score = RecallScore(
        outcomes=score_answer(answer, facts, "RQ-1 FB-1"),
        answer=answer,
        messages_left=5,
        messages_total=5,
        contradictions=(PIPELINE,),
    )
    assert score.recall_rate == 1.0
    assert score.asserted_superseded is True
    assert score.is_correct is False


def test_correct_requires_both_full_recall_and_no_retraction() -> None:
    facts = (PlantedFact("RQ-1", "requirement", 1, ""),)
    good = RecallScore(
        outcomes=score_answer("RQ-1, streaming pipeline", facts, "RQ-1"),
        answer="RQ-1, streaming pipeline",
        messages_left=3,
        messages_total=3,
        contradictions=(PIPELINE,),
    )
    assert good.is_correct is True
    missing = RecallScore(
        outcomes=score_answer("streaming pipeline", facts, "RQ-1"),
        answer="streaming pipeline",
        messages_left=3,
        messages_total=3,
        contradictions=(PIPELINE,),
    )
    assert missing.is_correct is False


def test_scenario_supplies_the_pipeline_contradiction() -> None:
    scenario = build_recall_scenario(salt="s", filler_turns=3, filler_tokens=50)
    assert any(entry.superseded == "batch pipeline" for entry in scenario.contradictions)


def test_correctness_score_is_graded_not_binary() -> None:
    facts = tuple(PlantedFact(f"RQ-{i}", "requirement", 1, "") for i in range(3))
    answer = "RQ-0 and RQ-1 only, streaming pipeline"
    score = RecallScore(
        outcomes=score_answer(answer, facts, "RQ-0 RQ-1 RQ-2"),
        answer=answer,
        messages_left=3,
        messages_total=3,
        contradictions=(PIPELINE,),
    )
    # Three facts plus one retraction check: two facts used, retraction avoided.
    assert score.checks_total == 4
    assert score.checks_passed == 3
    assert score.correctness_score == pytest.approx(0.75)
    assert score.is_correct is False


def test_a_retracted_assertion_lowers_correctness_below_recall() -> None:
    facts = (PlantedFact("RQ-1", "requirement", 1, ""),)
    answer = "RQ-1, proceeding with the batch pipeline."
    score = RecallScore(
        outcomes=score_answer(answer, facts, "RQ-1"),
        answer=answer,
        messages_left=3,
        messages_total=3,
        contradictions=(PIPELINE,),
    )
    # Recall is perfect; correctness is not. That gap is the whole point of the column.
    assert score.recall_rate == 1.0
    assert score.correctness_score == pytest.approx(0.5)
    assert score.is_correct is False


def test_perfect_score_requires_every_check() -> None:
    facts = (PlantedFact("RQ-1", "requirement", 1, ""),)
    answer = "RQ-1, streaming pipeline"
    score = RecallScore(
        outcomes=score_answer(answer, facts, "RQ-1"),
        answer=answer,
        messages_left=3,
        messages_total=3,
        contradictions=(PIPELINE,),
    )
    assert score.correctness_score == 1.0
    assert score.is_correct is True
