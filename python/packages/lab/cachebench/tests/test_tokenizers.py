# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the token counters, chiefly the encrypted-reasoning correction.

The oracle throughout is the framework's own serializer: a message that carries a
``protected_data`` payload must count exactly as the same message built without one, so the
expected value is ``_serialize_message`` of the payload-free twin, counted by the unwrapped
tokenizer. That pins the re-serialization to the framework's, not merely to "smaller".
"""

from __future__ import annotations

import base64
import contextlib
import json
import random
from typing import Any

import pytest
from agent_framework import CharacterEstimatorTokenizer, Content, Message, included_token_count
from agent_framework._compaction import _serialize_message, annotate_token_counts
from agent_framework_lab_cachebench import _tokenizers
from agent_framework_lab_cachebench._tokenizers import (
    REASONING_TOKENS_KEY,
    TOKENIZER_NAMES,
    ProtectedDataStrippingTokenizer,
    TiktokenTokenizer,
    build_tokenizer,
    stamp_reasoning_tokens,
)

ESTIMATOR = CharacterEstimatorTokenizer()


def _blob(size: int = 1200, *, seed: int = 1) -> str:
    """Return base64 of ``size`` random bytes, the shape of an encrypted reasoning payload."""
    rng = random.Random(seed)
    return base64.b64encode(rng.randbytes(size)).decode()


def _reasoning_message(*blobs: str | None, stamps: dict[int, Any] | None = None) -> Message:
    """Return an assistant message with one reasoning content per entry, then visible text.

    A ``None`` entry is a reasoning content without a payload; ``stamps`` maps a content index
    to a value for :data:`REASONING_TOKENS_KEY`.
    """
    contents = [
        Content.from_text_reasoning(id=f"rs_{index}", text="", protected_data=blob) for index, blob in enumerate(blobs)
    ]
    for index, value in (stamps or {}).items():
        contents[index].additional_properties[REASONING_TOKENS_KEY] = value
    # Non-ASCII on purpose: a re-serialization that escaped it would count the escapes.
    contents.append(Content.from_text(text="Réponse: the pipeline keeps the streaming group — 日本語 too."))
    return Message("assistant", contents, message_id="m-1")


class _SpyTokenizer:
    """Estimator that records the strings it was asked to count."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def count_tokens(self, text: str) -> int:
        self.seen.append(text)
        return ESTIMATOR.count_tokens(text)


def _bases() -> list[Any]:
    """Return the counters to check, tiktoken only where it is installed; the estimator always runs."""
    bases: list[Any] = [CharacterEstimatorTokenizer()]
    with contextlib.suppress(RuntimeError):
        bases.append(TiktokenTokenizer())
    return bases


@pytest.mark.parametrize("base", _bases(), ids=lambda base: type(base).__name__)
def test_a_payload_counts_at_the_visible_text_size_not_the_payloads(base: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    blob = _blob()
    with_blobs = _reasoning_message(blob, _blob(seed=2))
    without_blobs = _reasoning_message(None, None)
    parses: list[str] = []
    real_loads = json.loads
    monkeypatch.setattr(_tokenizers.json, "loads", lambda text: parses.append(text) or real_loads(text))

    counted = ProtectedDataStrippingTokenizer(base).count_tokens(_serialize_message(with_blobs))

    assert counted == base.count_tokens(_serialize_message(without_blobs))
    # Both payloads went, not only the first: one payload alone costs more than the whole message now.
    assert counted < base.count_tokens(blob)
    # Parsed once, and only because a payload was there.
    assert len(parses) == 1


def test_counting_leaves_the_message_and_what_is_sent_untouched() -> None:
    blob = _blob()
    message = _reasoning_message(blob)
    before = _serialize_message(message)
    wrapped = ProtectedDataStrippingTokenizer(ESTIMATOR)

    annotate_token_counts([message], tokenizer=wrapped)

    assert message.contents[0].protected_data == blob
    assert _serialize_message(message) == before
    # The framework read back the corrected count, not the payload-inclusive one.
    assert included_token_count([message]) == ESTIMATOR.count_tokens(_serialize_message(_reasoning_message(None)))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('"protected_data" is a phrase here, not a field', id="prose"),
        pytest.param('["protected_data", 1]', id="json-list"),
        pytest.param('{"protected_data": "QUJD"}', id="json-without-contents"),
        pytest.param('{"contents": {"protected_data": "QUJD"}}', id="contents-not-a-list"),
        pytest.param(
            '{"contents": [{"arguments": {"protected_data": "QUJD"}, "type": "function_call"}], "role": "assistant"}',
            id="nested-in-arguments",
        ),
    ],
)
def test_strings_that_are_not_a_message_with_a_payload_count_unchanged(text: str) -> None:
    spy = _SpyTokenizer()

    assert ProtectedDataStrippingTokenizer(spy).count_tokens(text) == ESTIMATOR.count_tokens(text)
    assert spy.seen == [text]


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            Message("user", [Content.from_text(text="Keep the RQ-AAA requirement.")], message_id="u"), id="user"
        ),
        pytest.param(
            Message("tool", [Content.from_function_result(call_id="c1", result="rows: 1, 2, 3")], message_id="t"),
            id="tool-result",
        ),
        pytest.param(
            Message("assistant", [Content.from_text(text='It said "protected_data" in the doc.')], message_id="a"),
            id="assistant-quoting-the-field-name",
        ),
        pytest.param(_reasoning_message(None), id="reasoning-without-payload"),
    ],
)
def test_a_message_without_a_payload_counts_exactly_as_before(
    message: Message, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _serialize_message(message)
    spy = _SpyTokenizer()
    monkeypatch.setattr(_tokenizers.json, "loads", lambda _: pytest.fail("parsed a message with no payload"))

    assert ProtectedDataStrippingTokenizer(spy).count_tokens(text) == ESTIMATOR.count_tokens(text)
    # The very same string object reached the wrapped tokenizer: no re-serialization happened.
    assert spy.seen[0] is text


def test_base64_costs_what_the_fix_says_it_costs() -> None:
    pytest.importorskip("tiktoken", reason="tiktoken not installed")
    raw = TiktokenTokenizer()
    wrapped = ProtectedDataStrippingTokenizer(raw)
    blob = _blob()
    with_blob = _serialize_message(_reasoning_message(blob))
    without_blob = _serialize_message(_reasoning_message(None))

    old_charge = raw.count_tokens(with_blob) - raw.count_tokens(without_blob)
    new_charge = wrapped.count_tokens(with_blob) - wrapped.count_tokens(without_blob)

    # What the old counting charged for the payload: ~0.68 o200k tokens per base64 character,
    # three to four times what the provider bills for the reasoning it encrypts.
    assert 0.6 * len(blob) <= old_charge <= 0.75 * len(blob)
    assert new_charge == 0


def test_a_stamped_reasoning_count_is_counted_in_the_payloads_place() -> None:
    wrapped = ProtectedDataStrippingTokenizer(ESTIMATOR)
    plain = ESTIMATOR.count_tokens(_serialize_message(_reasoning_message(None, None)))

    stamped = _serialize_message(_reasoning_message(_blob(), _blob(seed=2), stamps={0: 300}))
    assert wrapped.count_tokens(stamped) == plain + 300

    # Stamps on several payloads add up; the stamps themselves are never counted.
    split = _serialize_message(_reasoning_message(_blob(), _blob(seed=2), stamps={0: 200, 1: 100}))
    assert wrapped.count_tokens(split) == plain + 300

    # A stamp that is not a positive integer is dropped, not counted.
    for value in (True, -5, "300", 2.5):
        odd = _serialize_message(_reasoning_message(_blob(), _blob(seed=2), stamps={0: value}))
        assert wrapped.count_tokens(odd) == plain, value


def test_stamp_reasoning_tokens_marks_the_first_payload_only() -> None:
    message = _reasoning_message(None, _blob(), _blob(seed=2))
    other = Message("assistant", [Content.from_text(text="no reasoning here")], message_id="o")

    assert stamp_reasoning_tokens([other, message], 312) is True
    assert REASONING_TOKENS_KEY not in message.contents[0].additional_properties
    assert message.contents[1].additional_properties[REASONING_TOKENS_KEY] == 312
    assert REASONING_TOKENS_KEY not in message.contents[2].additional_properties
    assert REASONING_TOKENS_KEY not in other.contents[0].additional_properties
    # The payload is still there to be replayed; only the count moved.
    assert message.contents[1].protected_data == _blob()

    assert stamp_reasoning_tokens([other], 312) is False
    assert stamp_reasoning_tokens([message], 0) is False


def test_build_tokenizer_wraps_every_name_and_keeps_the_names() -> None:
    assert TOKENIZER_NAMES == ("estimator", "tiktoken")
    for name in TOKENIZER_NAMES:
        if name == "tiktoken":
            pytest.importorskip("tiktoken", reason="tiktoken not installed")
        tokenizer = build_tokenizer(name)
        assert isinstance(tokenizer, ProtectedDataStrippingTokenizer)
        assert isinstance(tokenizer.base, CharacterEstimatorTokenizer if name == "estimator" else TiktokenTokenizer)
    with pytest.raises(KeyError):
        build_tokenizer("bpe")
