# Copyright (c) Microsoft. All rights reserved.

"""Tests for the record-then-drop strategy, its tool, and the middleware that forces it.

Two phases, and the tests separate them: phase 1 must ask without inserting anything into the
cached prefix, and phase 2 must act only on a record the *provider* issued -- never on one the
client invented, which is unsafe on routes that track tool calls server-side.

The tool's own text is tested here as well, because the middleware sends no message: the
description and the ``values`` guidance are the entire prompt for the record, so a clause
lost from them is a class of content silently dropped, with nothing else in the design to
catch it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Message
from agent_framework._compaction import project_included_messages
from agent_framework_lab_cachebench.compaction._anchored import REMOVAL_MARKER
from agent_framework_lab_cachebench.compaction._preserve import is_preserved, set_preserved
from agent_framework_lab_cachebench.compaction._toolsummary import (
    DEFAULT_COVERAGE_SHARE,
    DEFAULT_RECORD_MAX_TOKENS,
    DEFAULT_RECORD_TARGET_TOKENS,
    RECALL_TOOL_NAME,
    RECORD_MARKER,
    RecallGate,
    ToolResultAnchoredSummarizationCompactionStrategy,
    ToolResultRecallMiddleware,
    _distinctive_tokens,
    find_record_index,
    make_recall_tool,
)

TOKENIZER = CharacterEstimatorTokenizer()


#: Every arming the middleware performs, so a test can assert it happened exactly when the
#: tool was pinned and never otherwise.
_armings: list[int] = []


def _record_messages(values: str, *, call_id: str = "rec") -> list[Message]:
    """Return a matched recall call and result carrying ``values`` as a record.

    Separate from the conversation builder so a test can put two records in one conversation,
    which is what bounding the ask per record produces and what the strategy has to survive.

    Args:
        values: The record's text, as the model would have written it.

    Keyword Args:
        call_id: Distinguishes one record from another, and is what makes the pair count as
            provider-issued rather than client-invented.

    Returns:
        The two messages.
    """
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": call_id, "name": RECALL_TOOL_NAME, "arguments": "{}"}],
            message_id=f"{call_id}_call",
        ),
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": call_id, "result": f"{RECORD_MARKER} {values}"}],
            message_id=f"{call_id}_res",
        ),
    ]


def _conversation(
    tool_turns: int,
    payload_chars: int = 8_000,
    *,
    record: str | None = None,
    first_turn: int = 0,
    tool_name: str | None = None,
    result_values: Callable[[int], str] | None = None,
) -> list[Message]:
    """Return a conversation, optionally with a recall record the agent already made.

    Each turn calls a tool of its own, named ``lookup_<n>``. That is the live benchmark's own
    shape -- one no-argument tool per scope, so ``tool_choice`` can pin exactly which fact a
    turn gathers -- and it matters here because the strategy checks a record against the names
    of the tools it claims to cover. A fixture calling one tool eight times would exercise only
    the degenerate case, which ``tool_name`` builds deliberately instead.

    Args:
        tool_turns: How many lookup turns to generate.
        payload_chars: Size of each tool result's filler.

    Keyword Args:
        record: Values for a record appended after the turns, or None for no record.
        first_turn: Index the turns are numbered from, so two stretches can be concatenated
            without colliding on message ids, call ids or tool names.
        tool_name: One name shared by every turn, instead of a name per turn.
        result_values: What each turn's result carries in front of the filler, as a function of
            the turn index, instead of the default ``CODE-<n>``. Coverage is now decided on the
            values a result contains, so a test has to be able to say what those are -- and, by
            returning text with no digit in it, to build a result that yields no values at all
            and so falls through to the tool-name rule.

    Returns:
        The messages.
    """
    messages = [
        Message(role="system", contents=["You are an assistant."], message_id="sys"),
        Message(role="user", contents=["Requirement: region is EU-WEST-1."], message_id="u0"),
        Message(role="assistant", contents=["Understood."], message_id="a0"),
    ]
    for offset in range(tool_turns):
        index = first_turn + offset
        call_id = f"call_{index}"
        values = result_values(index) if result_values else f"CODE-{index}"
        messages += [
            Message(role="user", contents=[f"Look up {index}."], message_id=f"u_{index}"),
            Message(
                role="assistant",
                contents=[
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name or f"lookup_{index}",
                        "arguments": "{}",
                    }
                ],
                message_id=f"a_call_{index}",
            ),
            Message(
                role="tool",
                contents=[
                    {
                        "type": "function_result",
                        "call_id": call_id,
                        "result": f"{values} " + "x" * payload_chars,
                    }
                ],
                message_id=f"t_res_{index}",
            ),
        ]
    if record is not None:
        messages += _record_messages(record)
    return messages


def _covering_record(tool_turns: int, *, first_turn: int = 0) -> str:
    """Return a record that names every tool it covers, as the tool's own guidance asks.

    ``RECALL_VALUES_DESCRIPTION`` asks for the results "grouped by the tool that produced it",
    so this is what a compliant record looks like, and it is what the strategy checks against.

    Args:
        tool_turns: How many turns the record accounts for.

    Keyword Args:
        first_turn: Index those turns are numbered from.

    Returns:
        The record's text.
    """
    return " ".join(f"lookup_{index}: CODE-{index}." for index in range(first_turn, first_turn + tool_turns))


def _rendered(messages: list[Message]) -> str:
    """Return what the model would receive, tool results included."""
    parts: list[str] = []
    for message in project_included_messages(messages):
        for content in message.contents:
            result = getattr(content, "result", None)
            text = getattr(content, "text", None)
            parts.append(str(result) if result is not None else (text if text is not None else str(content)))
    return "\n".join(parts)


#: A ceiling that puts the eight-turn conversation between the two thresholds, so the default
#: strategy asks and waits rather than giving up. Chosen from the fixture's own size: the
#: conversation is about 16,000 tokens, which is 73% of this, between the 60% trigger and the
#: 90% fallback.
_WAITING_CEILING = 22_000


def _strategy(**kwargs: Any) -> ToolResultAnchoredSummarizationCompactionStrategy:
    kwargs.setdefault("max_input_tokens", _WAITING_CEILING)
    return ToolResultAnchoredSummarizationCompactionStrategy(tokenizer=TOKENIZER, **kwargs)


async def test_phase_two_drops_only_what_precedes_the_record() -> None:
    """The record is what those results were reduced to; everything after it is untouched."""
    strategy = _strategy()
    messages = _conversation(tool_turns=8, record=_covering_record(8))

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "lookup_0: CODE-0." in rendered, "the record itself survives"
    assert "x" * 100 not in rendered, "the bulk behind it is gone"
    assert "EU-WEST-1" in rendered, "the head anchor is never touched"
    assert strategy.records_found == 1


async def test_a_client_invented_record_is_not_trusted() -> None:
    """Only a result whose call the provider issued counts.

    A tool result with no matching call is what synthesising the pair client-side produces,
    and it is exactly what breaks on routes that track tool calls server-side. The strategy
    must not treat one as a record.
    """
    strategy = _strategy()
    messages = _conversation(tool_turns=8)
    messages.append(
        Message(
            role="tool",
            contents=[{"type": "function_result", "call_id": "invented", "result": "CODE-0"}],
            message_id="fake",
        )
    )

    await strategy(messages)

    assert strategy.records_found == 0
    assert "CODE-0 " in _rendered(messages), "nothing dropped on the strength of a bare result"


async def test_nothing_happens_below_the_trigger() -> None:
    """A record that is not needed costs an agent turn and buys nothing."""
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(max_input_tokens=10_000_000, tokenizer=TOKENIZER)
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is False
    assert strategy.fallbacks_used == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_input_tokens": 0}, "max_input_tokens"),
        ({"max_input_tokens": 100, "trigger_fraction": 0.0}, "trigger_fraction"),
        ({"max_input_tokens": 100, "keep_head_groups": -1}, "keep_head_groups"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], match: str) -> None:
    """A silently accepted bad bound produces a plausible-looking wrong measurement."""
    with pytest.raises(ValueError, match=match):
        ToolResultAnchoredSummarizationCompactionStrategy(tokenizer=TOKENIZER, **kwargs)


async def test_the_fallback_fires_when_the_record_never_arrives() -> None:
    """Waiting forever means overflowing the window, which is worse than truncating.

    The model may never call the tool. Past the fallback threshold the strategy stops asking
    and compacts without a record: the tool results are lost either way at that point, and
    the alternative is a provider error that loses the whole conversation.
    """
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.2
    )
    messages = _conversation(tool_turns=8)

    assert await strategy(messages) is True
    assert strategy.fallbacks_used == 1
    assert strategy.fallbacks_after_record == 0, "no record arrived, so this cannot be the post-record path"
    assert strategy.records_found == 0
    # It compacted rather than waiting further.
    assert "CODE-0" not in _rendered(messages), "the oldest results are gone"
    assert "EU-WEST-1" in _rendered(messages), "the fallback keeps the head anchor too"


async def test_thresholds_the_wrong_way_around_are_rejected() -> None:
    """A fallback at or below the trigger silently disables the whole design."""
    with pytest.raises(ValueError, match="fallback_fraction"):
        ToolResultAnchoredSummarizationCompactionStrategy(
            max_input_tokens=1_000, tokenizer=TOKENIZER, trigger_fraction=0.8, fallback_fraction=0.8
        )


async def test_a_record_that_does_not_free_enough_still_falls_back() -> None:
    """The groups after the record are untouched by design and can exceed the ceiling alone."""
    strategy = ToolResultAnchoredSummarizationCompactionStrategy(
        max_input_tokens=500, tokenizer=TOKENIZER, trigger_fraction=0.1, fallback_fraction=0.9
    )
    # The record sits early, so most of the bulk is behind it and survives phase 2. The second
    # stretch is numbered on from the first: reusing the numbers would give two groups the same
    # message ids, and the grouper derives its group ids from those.
    messages = _conversation(tool_turns=2, record=_covering_record(2))
    messages += _conversation(tool_turns=6, first_turn=2)[3:]

    assert await strategy(messages) is True
    assert strategy.records_found == 1
    assert strategy.fallbacks_after_record == 1
    assert strategy.fallbacks_used == 0, "the give-up path is the one the record made unnecessary"


async def test_a_fallback_taken_behind_a_record_is_counted_apart_from_one_taken_without_one() -> None:
    """A pass that anchored on a record and fell back anyway used to report no fallback at all.

    Only the give-up path incremented a counter, and it is not the path taken here: this one
    finds a record, drops what the record covers, sees the prompt still over the ceiling, and
    hands the rest to the fallback -- which shortens tool results in place, including the very
    groups the coverage check has just refused to delete. So the row is measuring the fallback
    strategy over most of its material while its flags column says nothing at all.

    Measured on a live seed reporting UNCOVERED:4: it lost the same facts as the control while
    sitting three messages shorter and 16,617 tokens lighter, which is shortening rather than
    deletion. The count is separate from ``fallbacks_used`` because the two events differ --
    no record ever arrived, against one that arrived and did not free enough -- and joined to
    it in meaning, because either says part of the row belongs to another strategy.
    """
    strategy = _strategy(max_input_tokens=500, trigger_fraction=0.1, fallback_fraction=0.9)
    messages = _conversation(tool_turns=6, record=_covering_record(2))

    assert await strategy(messages) is True

    assert strategy.fallbacks_after_record == 1
    assert strategy.fallbacks_used == 0, "the give-up path never ran: a record was there and was anchored on"
    assert strategy.groups_kept_uncovered == 4, "and those four are what the fallback then went to work on"
    rendered = _rendered(messages)
    assert "CODE-2" not in rendered, "the fallback shed a group the coverage check had just kept"
    assert "CODE-3" not in rendered, "and another, which is the loss the count exists to make visible"


# region coverage, which is what a record is allowed to delete


async def test_a_partial_record_leaves_the_groups_it_never_named_in_place() -> None:
    """A record covering two groups of six may not delete the other four.

    This is the measured shape of gpt-5.6-luna: asked to record everything from six tool
    groups, it wrote about two. The strategy used to exclude all six anyway, on the stated
    assumption that a record replaces whatever precedes it, so four groups were deleted with
    nothing preserving them and nothing reporting it -- the loss then arrived in the scores as
    compaction damage rather than as an instrument that had stopped early. Raising the response
    cap, raising the stated target and rewriting the prompt were each measured and each changed
    nothing, which is why the check has to live in the strategy.
    """
    strategy = _strategy(max_input_tokens=16_000)
    messages = _conversation(tool_turns=6, record=_covering_record(2))

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "CODE-0 x" not in rendered, "the record names lookup_0, so its bulk is genuinely redundant"
    assert "CODE-1 x" not in rendered, "and lookup_1"
    for index in range(2, 6):
        assert f"CODE-{index} x" in rendered, f"lookup_{index} is unmentioned, so its group stays whole"
    assert strategy.groups_kept_uncovered == 4
    assert strategy.fallbacks_used == 0, "keeping more is not the same as failing to compact at all"


async def test_a_complete_record_still_drops_every_group_it_covers() -> None:
    """The good-model path must not pay for the bad one.

    gpt-5.4-mini writes records that name every tool they cover, and on those the coverage
    check costs nothing: it is meant to be silent whenever the record did what it was asked.
    A check that also held back complete records would trade a rare silent loss for a constant
    one, which is the regression this pins.
    """
    strategy = _strategy()
    messages = _conversation(tool_turns=8, record=_covering_record(8))

    assert await strategy(messages) is True

    assert "x" * 100 not in _rendered(messages), "every group the record named is gone"
    assert strategy.groups_kept_uncovered == 0
    assert strategy.fallbacks_used == 0
    assert strategy.fallbacks_after_record == 0, "a record that freed enough needs no fallback behind it"


async def test_a_newer_record_never_drops_an_older_one() -> None:
    """The only surviving account of what an old record covered is the old record.

    Bounding what one record must cover means a run takes several, and each ends up behind the
    next. Excluding a group merely because it precedes the newest record would delete the
    record before it: the same loss this strategy exists to prevent, one level removed and
    quieter, because a newer record that names the tools looks exactly like coverage.
    """
    strategy = _strategy(max_input_tokens=11_000)
    messages = _conversation(tool_turns=2, record=f"older record. {_covering_record(2)}")
    messages += _conversation(tool_turns=2, first_turn=2)[3:]
    messages += _record_messages(f"newer record. {_covering_record(4)}", call_id="rec2")

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "older record." in rendered, "the newer record must not delete the older one"
    assert "newer record." in rendered
    assert "x" * 100 not in rendered, "the bulk the newer record does cover is still dropped"
    assert strategy.fallbacks_used == 0, "the fallback sheds records of its own, so it must not be what ran"


@pytest.mark.parametrize(
    ("record", "kept"),
    [
        pytest.param("lookup: both deployments were healthy.", 2, id="one-mention-for-two-groups"),
        pytest.param("lookup: healthy. lookup: healthy.", 0, id="one-mention-each"),
    ],
)
async def test_a_name_shared_by_two_valueless_groups_needs_a_mention_for_each(record: str, kept: int) -> None:
    """Two calls to one tool are two groups and one name, which the name alone cannot resolve.

    A record grouped by tool -- which is what the tool's own guidance asks for -- names that
    tool once however many times it was called, so a single mention cannot say whether both
    calls were accounted for or only one. The count rule resolves the ambiguity by keeping
    both, and only a record that mentions the name once per group buys the drop. It errs
    towards keeping on purpose: the cost of that choice is tokens, and the cost of the other
    choice was a fact that vanished with no trace of where it went.

    The results here carry no digit anywhere, so they yield no distinctive values and the name
    rule is what decides them. That is now the *only* way to reach this rule, and it is why the
    ambiguity it works around is a smaller problem than it was: two calls to one tool return
    two different sets of values, so wherever there are values at all they say which call was
    accounted for and the name never has to.
    """
    strategy = _strategy(max_input_tokens=5_000)
    messages = _conversation(
        tool_turns=2, tool_name="lookup", record=record, result_values=lambda _: "the deployment is healthy"
    )

    await strategy(messages)
    rendered = _rendered(messages)

    assert strategy.groups_kept_uncovered == kept
    assert strategy.fallbacks_used == 0
    assert ("the deployment is healthy x" in rendered) is (kept == 2)


async def test_values_settle_two_calls_to_one_tool_that_the_name_cannot() -> None:
    """The name rule's own hard case dissolves once coverage is read off the values.

    Six calls to ``lookup_eu`` are six groups and one name, and a record grouped by tool writes
    that name once, so the count rule above has to keep all six unless the model repeats
    itself. The values do not have that problem: each call returned different ones, and a
    record quoting both sets has demonstrably accounted for both calls -- without ever writing
    the tool's name, which is the thing the models under measurement do not do.
    """
    strategy = _strategy(max_input_tokens=5_000)
    messages = _conversation(
        tool_turns=2,
        tool_name="lookup",
        result_values=lambda index: f"AB-10000{index} CD-20000{index}",
        record="first call returned AB-100000 and CD-200000; second returned AB-100001 and CD-200001.",
    )

    assert await strategy(messages) is True

    assert strategy.groups_kept_uncovered == 0, "one mention of the name, and both groups still accounted for"
    assert "AB-100000 x" not in _rendered(messages)
    assert "AB-100001 x" not in _rendered(messages)


def _codes(index: int) -> list[str]:
    """Return one deployment lookup's worth of identifiers, eight of them, as the live tool returns.

    Eight is the number the recall measurements were run at, and it is the number the default
    coverage share was chosen against: at eight values it tolerates exactly one being
    unrecognisable in the record.
    """
    return [f"{'ABCDEFGH'[index]}{'BCDEFGHI'[index]}-{123456 + offset:06d}" for offset in range(8)]


def _prose_record(index: int) -> str:
    """Return the record gpt-5.6-luna actually writes for one lookup.

    Verbatim in shape: the scope named in prose, every code quoted, and the function name
    ``lookup_<n>`` nowhere in the sentence. That is not a defective record -- it is a complete
    one, written the way models write -- and the tool-name rule scored it as covering nothing.
    """
    return f"extra{index} deployment lookup returned codes: " + ", ".join(_codes(index)) + "."


def _bulk_of(index: int) -> str:
    """Return a string that appears in one lookup's result and nowhere else, filler included."""
    return f"{_codes(index)[-1]} x"


async def test_a_record_quoting_a_groups_values_covers_it_though_it_never_names_the_tool() -> None:
    """The luna case, which the rule this replaces got exactly backwards.

    Asked to record six tool groups, gpt-5.6-luna writes prose: *"extra0 deployment lookup
    returned codes: AB-123456, ..."*. Every identifier is there. The string ``lookup_extra0``
    is not, and the tool-name rule therefore refused to drop a group whose entire content the
    record was carrying. The same rule scored ``UNCOVERED:4`` against gpt-5.4-mini, whose
    records are complete, and its compaction fell from a 20% reduction to 5-6% in exchange for
    nothing at all -- a check that penalises the model that complied.

    ``RECALL_VALUES_DESCRIPTION`` leads with "Quote verbatim any value that cannot be
    reconstructed or guessed", and that is what is tested now.
    """
    strategy = _strategy(max_input_tokens=16_000)
    record = " ".join(_prose_record(index) for index in range(2))
    messages = _conversation(tool_turns=6, result_values=lambda index: " ".join(_codes(index)), record=record)

    assert "lookup_0" not in record, "the fixture is only the luna case if the tool name is genuinely absent"
    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert _bulk_of(0) not in rendered, "the record carries every code this group returned, so its bulk can go"
    assert _bulk_of(1) not in rendered
    for index in range(2, 6):
        assert _bulk_of(index) in rendered, f"lookup_{index} is in no record, so its group stays whole"
    assert strategy.groups_kept_uncovered == 4


async def test_a_record_that_names_a_tool_but_quotes_none_of_its_values_covers_nothing() -> None:
    """The other half of the same correction, and the reason the name was never the contract.

    A record can name every tool it was asked about and still have thrown away everything they
    returned. Under the name rule that record licensed deleting all six groups; under this one
    it licenses deleting none, because nothing it contains could answer a later question about
    what those calls found.
    """
    strategy = _strategy(max_input_tokens=16_000)
    messages = _conversation(
        tool_turns=6,
        result_values=lambda index: " ".join(_codes(index)),
        record="lookup_0, lookup_1, lookup_2, lookup_3, lookup_4 and lookup_5 all returned deployment codes.",
    )

    assert await strategy(messages) is False, "a record that quotes nothing buys nothing"

    assert strategy.groups_kept_uncovered == 6
    assert all(_bulk_of(index) in _rendered(messages) for index in range(6))


async def test_a_group_with_no_values_to_quote_falls_back_to_the_tool_name() -> None:
    """Where the value rule has nothing to read, the weaker rule decides -- not a default.

    A result holding only prose yields no distinctive values, so by this rule nothing in it is
    unreconstructable and the record cannot be checked against it. Both shortcuts are wrong.
    Calling such a group covered would let a record that mentions nothing delete a tool's
    findings, which is the silent loss the check exists to prevent; calling it uncovered would
    make every prose-only tool permanently undroppable, which is not conservatism but a broken
    strategy. So the name rule runs, and here it splits the six groups the way it should: the
    two the record names go, the four it does not stay.
    """
    strategy = _strategy(max_input_tokens=16_000)
    messages = _conversation(
        tool_turns=6,
        result_values=lambda _: "the deployment is healthy",
        record="lookup_0 and lookup_1 both reported healthy deployments.",
    )

    assert await strategy(messages) is True

    assert strategy.groups_kept_uncovered == 4, "neither automatically covered nor automatically uncovered"


@pytest.mark.parametrize(
    ("share", "kept"),
    [
        pytest.param(1.0, 2, id="every-value-or-nothing"),
        pytest.param(DEFAULT_COVERAGE_SHARE, 2, id="the-default-refuses-three-of-four"),
        pytest.param(0.75, 0, id="exactly-the-share-that-was-quoted"),
        pytest.param(0.0, 0, id="any-value-at-all"),
    ],
)
async def test_the_coverage_share_decides_how_much_of_a_group_must_be_quoted(share: float, kept: int) -> None:
    """The dial has to move the trade-off across its whole range, or it is decoration.

    Each group returns four values and the record quotes three of them, so the boundary sits at
    exactly 0.75. The share is compared with ``ceil``, which is what makes it a genuine floor:
    at the 0.8 default a group of four needs all four, because three is 0.75 and 0.75 is less
    than 0.8. The two ends have to mean what they say as well -- 1.0 every value, 0.0 any value
    at all, the latter being the behaviour this check replaced and the row someone comparing
    the two would want to run.
    """
    strategy = _strategy(max_input_tokens=5_000, coverage_share=share)
    messages = _conversation(
        tool_turns=2,
        result_values=lambda index: f"V{index}-1000 V{index}-2000 V{index}-3000 V{index}-4000",
        record=" ".join(f"V{index}-1000, V{index}-2000, V{index}-3000" for index in range(2)),
    )

    await strategy(messages)

    assert strategy.groups_kept_uncovered == kept
    assert ("V0-4000 x" in _rendered(messages)) is (kept == 2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            'Deployment "AB-123456", version v1.2.3, updated 2026-08-31T09:00:00Z.',
            {"ab-123456", "v1.2.3", "2026-08-31t09:00:00z"},
            id="quoted-versioned-and-timestamped",
        ),
        pytest.param("The region is EU-WEST and the status is healthy.", set(), id="no-digit-no-value"),
        pytest.param("The 3rd of v2 at 10% (1).", set(), id="too-short-to-be-a-value"),
    ],
)
def test_the_value_rule_finds_what_cannot_be_reconstructed_and_leaves_prose_alone(
    text: str, expected: set[str]
) -> None:
    """The rule is stated so it can be argued with, and this is the statement executed.

    Whitespace-delimited tokens, punctuation stripped from both ends, kept when at least four
    characters remain and one of them is a digit. Deliberately not fitted to this benchmark's
    hex codes: a rule that was would need refitting for every workload, which is the mistake
    the recall tool's own description already had to be rewritten out of. The cost is the
    second case -- an alphabetic value is invisible to it -- which is why a group yielding
    nothing falls back to the tool name rather than being ruled either way.
    """
    assert _distinctive_tokens(text) == expected


# region protecting the record from the strategy behind it


#: Padding that makes the record big enough for the fallback to want to trim it. The anchored
#: strategy's per-result floor is 150 tokens and this takes the record to about 590, so a
#: record left unprotected is cut rather than merely eligible to be.
_RECORD_PADDING = " ".join(f"deployment {index} returned code QQ-{100_000 + index}." for index in range(60))


async def test_the_record_survives_a_fallback_that_shortens_and_sheds_everything_else() -> None:
    """The headline regression: the strategy used to destroy the one thing it exists to produce.

    Phase 2 deletes tool groups *because* the record replaced them. When the record does not
    free enough on its own, what remains goes to ``fallback`` -- by default
    ``AnchoredCompactionStrategy``, which shortens tool results and then sheds whole tool
    groups. The record is a tool result. Nothing in that strategy had ever heard of one, so it
    trimmed the record like any other bulk, and every deletion the record had licensed lost its
    only surviving copy.

    Measured on a live seed: a record holding four lookups' worth of values, thirty-two
    identifiers, reached the prompt the questions were answered from carrying two. 16,617
    tokens gone while three messages left, which is shortening rather than deletion, and no
    counter in the run was looking at anything but message counts.

    The fixture puts the record early enough to sit in the fallback's middle band -- the only
    place it can be touched -- and then squeezes the ceiling until the fallback runs hard.
    """
    strategy = _strategy(max_input_tokens=500, trigger_fraction=0.1, fallback_fraction=0.9)
    record = f"{_covering_record(2)} {_RECORD_PADDING}"
    messages = _conversation(tool_turns=2, record=record)
    messages += _conversation(tool_turns=6, first_turn=2)[3:]

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert strategy.fallbacks_after_record == 1, "the fixture only means anything if the fallback ran"
    assert f"{RECORD_MARKER} {record}" in rendered, "the record is intact to its last character"
    assert REMOVAL_MARKER not in rendered.split(RECORD_MARKER)[1], "and carries no trim marker of its own"
    assert "CODE-4 x" not in rendered, "while the fallback took the groups around it apart"


async def test_every_record_is_marked_protected_including_the_ones_a_newer_record_supersedes() -> None:
    """An older record is the sole account of the groups behind it, so it is protected too.

    ``_drop_before`` already refuses to *delete* an older record. Without the mark the fallback
    would shorten it instead, which loses the same facts more quietly -- and the mark has to be
    re-applied on every pass, because compaction runs against a freshly loaded conversation and
    the annotations of the previous pass are not in it.
    """
    strategy = _strategy(max_input_tokens=11_000)
    messages = _conversation(tool_turns=2, record=f"older record. {_covering_record(2)}")
    messages += _conversation(tool_turns=2, first_turn=2)[3:]
    messages += _record_messages(f"newer record. {_covering_record(4)}", call_id="rec2")

    await strategy(messages)

    assert {message.message_id for message in messages if is_preserved(message)} == {
        "rec_call",
        "rec_res",
        "rec2_call",
        "rec2_res",
    }


async def test_a_group_another_strategy_protected_is_neither_dropped_nor_reported_uncovered() -> None:
    """Phase 2 deletes, so it honours the mark too -- and does not confuse it with a shortfall.

    The mark means "this is the only surviving copy of something", and a record naming the tool
    that produced it does not change that. Counting the skip as ``groups_kept_uncovered`` would
    be worse than not skipping: that number is read as "the record fell short", and a protected
    group says nothing at all about the record.
    """
    strategy = _strategy(max_input_tokens=16_000)
    messages = _conversation(tool_turns=6, record=_covering_record(6))
    for message in messages:
        if message.message_id in {"a_call_1", "t_res_1"}:
            set_preserved(message, preserved=True, reason="test")

    assert await strategy(messages) is True
    rendered = _rendered(messages)

    assert "CODE-1 x" in rendered, "the protected group survives a record that covers it"
    assert "CODE-0 x" not in rendered, "while its unprotected neighbour does not"
    assert strategy.groups_kept_uncovered == 0, "protected is not the same as the record falling short"


async def test_an_uninvited_recall_call_is_not_protected_as_though_it_had_recorded_anything() -> None:
    """The gate refuses calls nobody asked for, and what it returns is not a record.

    The recall tool cannot be hidden from the model, so it is advertised on every request and
    was called unprompted on unpinned follow-ups in every early run. Protecting the result of
    such a call would give a message that preserves nothing the standing of one that preserves
    everything, and would hand the fallback one more thing it may not touch for no gain at all.
    """
    strategy = _strategy()
    messages = _conversation(tool_turns=8, record=_covering_record(8))
    messages += _record_messages("Not required right now: nothing was recorded.", call_id="uninvited")
    for message in messages:
        if message.message_id == "uninvited_res":
            message.contents[0].result = "Not required right now: nothing was recorded."

    await strategy(messages)

    assert {message.message_id for message in messages if is_preserved(message)} == {"rec_call", "rec_res"}


class _Recorder:
    """Stands in for the rest of the pipeline, capturing the options a call went out with."""

    def __init__(self, messages: list[Message], finish_reason: str | None = None) -> None:
        self.messages = messages
        self.finish_reason = finish_reason
        self.seen: list[dict[str, Any]] = []

    async def __call__(self) -> None:
        from agent_framework import ChatResponse

        self.seen.append(dict(self.context.options or {}))
        self.context.messages = self.messages
        # What the provider says about why it stopped, which is the only thing that separates
        # a record the model chose to keep short from one it was cut off in the middle of.
        self.context.result = ChatResponse(
            messages=Message(role="assistant", contents=["ok"]), finish_reason=self.finish_reason
        )


async def _run(
    middleware: ToolResultRecallMiddleware, messages: list[Message], finish_reason: str | None = None
) -> dict[str, Any]:
    """Drive one middleware pass and return the options the call went out with."""
    from agent_framework import ChatContext

    context = ChatContext(client=None, messages=[Message(role="user", contents=["q"])], options={"temperature": 0})
    recorder = _Recorder(messages, finish_reason)
    recorder.context = context
    await middleware.process(context, recorder)
    return recorder.seen[0]


async def test_the_middleware_forces_the_call_and_sends_no_message() -> None:
    """Phase 1 must leave no trace in the prompt, only in the options.

    A message appended here carries no history provider's source tag, so per-service-call
    persistence would treat it as new input and store it -- and an instruction of ours would
    then appear in the conversation the application replays to its user.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    first = await _run(middleware, big)
    assert "tool_choice" not in first, "nothing is known about history size before the first call"

    second = await _run(middleware, big)
    assert second["tool_choice"] == {"mode": "required", "required_function_name": RECALL_TOOL_NAME}
    assert len(_armings) == 1, "the tool is armed exactly when it is pinned, and never otherwise"
    assert middleware.forced_calls == 1
    # The prompt is untouched: no instruction, no extra turn.
    assert all("recall" not in str(m.contents[0]).lower() for m in [Message(role="user", contents=["q"])])


async def test_forgetting_the_pending_decision_stops_the_next_call_forcing() -> None:
    """Restoring the snapshot has to clear the middleware's pending decision too.

    The decision to force a record is taken on one call and applied to the next, so it belongs
    to the conversation rather than to the middleware. Left in place, a decision taken while
    the conversation was being seeded fires on the first question asked of the snapshot and on
    none of the others -- one probe carrying a prompt the rest do not, which is exactly the
    difference between probes the snapshot design exists to remove.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    # The first call arms nothing but leaves the middleware intending to force the next one.
    await _run(middleware, big)
    middleware.forget_pending()
    after = await _run(middleware, big)

    assert "tool_choice" not in after
    assert not _armings, "the recall tool was armed on a call the snapshot had reset"
    assert middleware.forced_calls == 0


async def test_the_middleware_stops_once_a_record_exists() -> None:
    """Forcing a second record would re-drop what the first already covered."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )

    await _run(middleware, _conversation(tool_turns=8))
    await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))
    after = await _run(middleware, _conversation(tool_turns=8, record="CODE-0"))

    assert "tool_choice" not in after
    assert find_record_index(_conversation(tool_turns=8, record="CODE-0")) is not None


async def test_the_middleware_leaves_small_conversations_alone() -> None:
    """Below the trigger there is nothing to record and nothing to drop."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=10_000_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1)
    )

    await _run(middleware, _conversation(tool_turns=8))
    after = await _run(middleware, _conversation(tool_turns=8))

    assert "tool_choice" not in after
    assert middleware.forced_calls == 0


async def test_a_single_record_is_attributed_exactly_once() -> None:
    """The transition must be tracked on the instance, not re-read from each prompt.

    Before the pipeline runs, context.messages holds only the new turn, so a pre-call check
    reports "no record" every time and every later call counts as another one. That produced
    RECVOLUNTEERED:18 for a single record in a live run, which turned an attribution into
    noise at exactly the moment it was needed.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    with_record = _conversation(tool_turns=8, record="CODE-0")

    for _ in range(5):
        await _run(middleware, with_record)

    assert middleware.records_forced + middleware.records_volunteered == 1


# region the tool, which is the whole of the prompt


def _recall_exchange(result: str) -> list[Message]:
    """Return a matched recall call and result carrying ``result``."""
    return [
        Message(
            role="assistant",
            contents=[{"type": "function_call", "call_id": "r", "name": RECALL_TOOL_NAME, "arguments": "{}"}],
        ),
        Message(role="tool", contents=[{"type": "function_result", "call_id": "r", "result": result}]),
    ]


def test_the_recall_tool_records_only_while_armed() -> None:
    """The tool cannot be hidden, so it has to be inert instead.

    MAF requires it to be registered with the agent: the function-invocation layer wraps the
    middleware layer and builds its tool map first, so a tool supplied through per-call
    options reaches the model but never the executor, and the model's call goes unanswered.
    A registered tool is advertised on every request, and this one was called uninvited on
    every unpinned follow-up call. Permission is therefore separated from visibility.
    """
    gate = RecallGate()
    tool = make_recall_tool(gate)

    uninvited = tool("AA-1")
    gate.arm()
    armed = tool("AA-1")
    reused = tool("AA-2")

    assert RECORD_MARKER not in uninvited
    assert RECORD_MARKER in armed
    assert RECORD_MARKER not in reused, "one arming cannot licence a second record"
    # The scorer must agree, or an uninvited call would look like a record and the strategy
    # would drop results that nothing had preserved.
    assert find_record_index(_recall_exchange(uninvited)) is None
    assert find_record_index(_recall_exchange(armed)) is not None


@pytest.mark.parametrize(
    "clause",
    [
        pytest.param("grouped by the tool that produced it", id="attribution"),
        pytest.param("Quote verbatim any value that cannot be reconstructed or guessed", id="unreconstructable"),
        pytest.param("Keep findings and conclusions as they were stated", id="findings"),
        pytest.param("Carry over any summary a tool already produced as it stands", id="existing-summary"),
        pytest.param("Summarise the remaining content briefly", id="the-rest"),
        pytest.param("keep exactness over brevity", id="tie-break"),
    ],
)
def test_the_tool_asks_for_every_kind_of_content_a_result_can_hold(clause: str) -> None:
    """The four instructions partition the content, and a missing one is a silent hole.

    The description and the ``values`` guidance are the entire prompt: the middleware sends no
    message, because one appended there would be persisted into the caller's own conversation.
    So whatever these do not name is content the model may drop without anything noticing, and
    the earlier text named only "identifiers and values seen in earlier tool results" -- fitted
    to this benchmark's hex codes, and blind to the prose, findings and conclusions that are
    most of what a real tool returns.
    """
    assert clause in (make_recall_tool().__doc__ or "")


def test_the_tool_says_what_the_call_is_for_before_it_says_what_to_pass() -> None:
    """The description has to stand on its own: a model reading the schema sees it first."""
    doc = make_recall_tool().__doc__ or ""

    assert doc.startswith("Record what must survive from earlier tool results")
    assert "after those results are removed from the conversation to save space" in doc


def test_the_target_length_is_stated_when_one_is_given() -> None:
    """The description is the only channel that makes the model plan for a size.

    A ``max_tokens`` cap cannot do it: a model does not shorten to fit one, it writes until it
    is cut, and on a tool call the cut lands inside the arguments JSON -- so a cap set where
    the record should end produces no record rather than a shorter one. The middleware cannot
    send an instruction message either, which leaves this text.
    """
    stated = make_recall_tool(target_tokens=1_500).__doc__ or ""
    default = make_recall_tool().__doc__ or ""
    silent = make_recall_tool(target_tokens=None).__doc__ or ""

    assert "1,500 tokens" in stated
    assert f"{DEFAULT_RECORD_TARGET_TOKENS:,} tokens" in default
    assert "Aim for about" not in silent
    # Stating a length must not cost a clause: the two bounds are independent instructions.
    assert "Quote verbatim any value that cannot be reconstructed or guessed" in silent


def test_the_target_sits_well_under_the_cap() -> None:
    """Overshooting the stated length must not be the same event as being cut off.

    They measure different things -- one is what the model aims for, the other is what the
    provider enforces -- and a default pair close together would make every slightly long
    record a truncated one.
    """
    assert DEFAULT_RECORD_TARGET_TOKENS < DEFAULT_RECORD_MAX_TOKENS


# region bounding the record


async def test_the_cap_is_set_on_the_forced_call_and_on_no_other() -> None:
    """Every other call needs the run's own cap; only this one is asked to write a record.

    Left to inherit ``--answer-max-tokens``, the single call instructed to summarise every
    earlier tool result is the one call in the run with no bound of its own.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        trigger_fraction=0.1,
        record_max_tokens=777,
    )
    big = _conversation(tool_turns=8)
    recorded = _conversation(tool_turns=8, record="CODE-0")

    calls = [
        # Nothing is known about the history before the first call, so it cannot be forced.
        await _run(middleware, big),
        await _run(middleware, big),
        # The middleware keeps asking until a record exists, so the record has to arrive
        # before an unforced call can happen again.
        await _run(middleware, recorded),
        await _run(middleware, recorded),
    ]

    assert [("tool_choice" in options) for options in calls] == [False, True, True, False]
    for options in calls:
        assert ("max_tokens" in options) is ("tool_choice" in options), options
        assert options.get("max_tokens", 777) == 777
        # The call's other options survive: this replaces the option set, it does not discard it.
        assert options["temperature"] == 0


async def test_no_cap_leaves_the_runs_own_ceiling_in_place() -> None:
    """None has to mean what it meant before the parameter existed, or a run cannot opt out."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        trigger_fraction=0.1,
        record_max_tokens=None,
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big)
    forced = await _run(middleware, big)

    assert "tool_choice" in forced
    assert "max_tokens" not in forced


def test_a_cap_that_cannot_hold_a_record_is_refused() -> None:
    """Zero is not "no cap": it is a call that can produce nothing, which None expresses."""
    with pytest.raises(ValueError, match="record_max_tokens"):
        ToolResultRecallMiddleware(max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: None, record_max_tokens=0)


async def test_a_record_cut_short_is_counted_rather_than_read_as_complete() -> None:
    """A partial record is the one failure this design does not otherwise show.

    A tool call cut mid-arguments produces no record at all, which is loud: the strategy waits,
    falls back, and the row carries FALLBACK. A call cut just after a closing brace yields a
    record that parses and looks whole -- and the strategy then drops every tool group behind
    something covering only part of them, so the loss is scored as compaction damage.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big, finish_reason="tool_calls")
    await _run(middleware, big, finish_reason="length")

    assert middleware.forced_calls == 1
    assert middleware.records_truncated == 1


async def test_only_the_forced_call_can_truncate_a_record() -> None:
    """Every call in a long run can hit its own ceiling; only one of them writes the record.

    Counting the rest would put an ordinary long answer in the column that says the record is
    partial, which is the reading the count exists to prevent.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    # Every call stops at its ceiling, forced or not.
    for _ in range(4):
        await _run(middleware, big, finish_reason="length")

    assert middleware.forced_calls > 0
    assert middleware.records_truncated == middleware.forced_calls


async def test_a_forced_call_that_finished_is_not_reported_as_truncated() -> None:
    """A short record is a choice the model is allowed to make, and is not a cut one."""
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: _armings.append(1), trigger_fraction=0.1
    )
    big = _conversation(tool_turns=8)

    await _run(middleware, big, finish_reason="stop")
    await _run(middleware, big, finish_reason="tool_calls")

    assert middleware.forced_calls == 1
    assert middleware.records_truncated == 0


# region bounding what one record must cover


async def test_the_group_bound_forces_a_record_below_the_token_trigger() -> None:
    """Coverage does not scale with how much there is to cover, so the ask has to be bounded.

    Measured: gpt-5.6-luna's record covered two of six tool groups, and raising the response
    cap, raising the stated target and rewriting the prompt each left that unchanged. What was
    still within reach was asking each record for less. The ceiling here is far too large for
    the token trigger to fire at sixteen thousand tokens, so nothing but the group bound can
    have forced this call.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=10_000_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        max_groups_before_record=2,
    )
    big = _conversation(tool_turns=8)

    first = await _run(middleware, big)
    second = await _run(middleware, big)

    assert "tool_choice" not in first, "nothing is known about the history before the first call"
    assert second["tool_choice"] == {"mode": "required", "required_function_name": RECALL_TOOL_NAME}
    assert middleware.forced_calls == 1
    assert len(_armings) == 1, "the tool is armed exactly when it is pinned"


async def test_the_group_bound_asks_again_once_the_next_groups_have_accumulated() -> None:
    """One record covers a bounded stretch, and the stretch after it needs its own.

    The token trigger cannot be the thing that asks for a second record: the size that fired it
    does not go away when the record arrives, so it would pin every remaining call in the run.
    The group bound is what re-arms, and it counts from the newest record rather than from the
    start -- so a conversation that has done no tool work since its record is left alone.
    """
    _armings.clear()
    middleware = ToolResultRecallMiddleware(
        max_input_tokens=10_000_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        max_groups_before_record=2,
    )
    settled = _conversation(tool_turns=8, record=_covering_record(8))
    grown = [*settled, *_conversation(tool_turns=2, first_turn=8)[3:]]

    await _run(middleware, settled)
    quiet = await _run(middleware, settled)
    await _run(middleware, grown)
    again = await _run(middleware, grown)

    assert "tool_choice" not in quiet, "no work has happened since the record, so nothing is due"
    assert "tool_choice" in again, "two groups have, so the next record is"
    assert middleware.forced_calls == 1


async def test_without_the_group_bound_the_middleware_asks_once_and_stops() -> None:
    """None has to mean what it meant before the parameter existed, on both halves.

    Off, tool groups are not a trigger at all -- eight of them below the token threshold force
    nothing -- and a record that exists ends the asking for good. Passing None explicitly has
    to be identical to not passing it, or a run cannot opt out of the new behaviour.
    """
    _armings.clear()
    unbounded = ToolResultRecallMiddleware(
        max_input_tokens=10_000_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        max_groups_before_record=None,
    )
    big = _conversation(tool_turns=8)

    await _run(unbounded, big)

    assert "tool_choice" not in await _run(unbounded, big), "eight groups trigger nothing when the bound is off"
    assert unbounded.forced_calls == 0
    assert not _armings

    asking = ToolResultRecallMiddleware(
        max_input_tokens=1_000,
        tokenizer=TOKENIZER,
        arm=lambda: _armings.append(1),
        trigger_fraction=0.1,
        max_groups_before_record=None,
    )
    recorded = _conversation(tool_turns=8, record=_covering_record(8))

    await _run(asking, big)
    forced = await _run(asking, big)
    await _run(asking, recorded)
    settled = await _run(asking, recorded)

    assert "tool_choice" in forced, "the token trigger still fires"
    assert "tool_choice" not in settled, "and still stops for good once a record exists"


@pytest.mark.parametrize("bound", [0, -1])
def test_a_group_bound_that_can_never_hold_a_record_is_refused(bound: int) -> None:
    """Zero groups per record is a record forced on every call, which is not a bound.

    A silently accepted bad bound produces a plausible-looking wrong measurement: the run would
    spend an agent turn on a record before every single call and report the result as this
    design's cost.
    """
    with pytest.raises(ValueError, match="max_groups_before_record"):
        ToolResultRecallMiddleware(
            max_input_tokens=1_000, tokenizer=TOKENIZER, arm=lambda: None, max_groups_before_record=bound
        )
