# Copyright (c) Microsoft. All rights reserved.

"""Token counters used for compaction budgets and the local prefix oracle.

The default ``CharacterEstimatorTokenizer`` assumes 4 chars/token over serialized JSON,
which runs roughly 2x a real BPE count for this benchmark's content. That is harmless when
comparing strategies at small sizes, but at 100k-plus prompts it moves a compaction
threshold by six figures — so large runs should count real tokens.

Whichever counter a run selects, :func:`build_tokenizer` hands it back inside
:class:`ProtectedDataStrippingTokenizer`, so that encrypted reasoning payloads the client
replays are not charged as if they were prompt text. That class documents the upstream
defect it compensates for and the accounting it settles on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Final, cast

from agent_framework import CharacterEstimatorTokenizer, Message, TokenizerProtocol

__all__ = [
    "REASONING_TOKENS_KEY",
    "TOKENIZER_NAMES",
    "ProtectedDataStrippingTokenizer",
    "TiktokenTokenizer",
    "build_tokenizer",
    "stamp_reasoning_tokens",
]

TOKENIZER_NAMES: Final[tuple[str, ...]] = ("estimator", "tiktoken")

#: ``additional_properties`` key on a content that a run may stamp with the reasoning token
#: count the provider reported for the response whose ``protected_data`` the content carries
#: (``UsageDetails["reasoning_output_token_count"]``). :class:`ProtectedDataStrippingTokenizer`
#: counts that many tokens in the payload's place; :func:`stamp_reasoning_tokens` writes it.
REASONING_TOKENS_KEY: Final[str] = "reasoning_output_token_count"

#: The field as the framework's serializer emits it, quotes included. JSON escapes the quotes
#: of any occurrence inside a string value, so a serialized message contains this sequence
#: only where a ``protected_data`` key is present. Its absence, one substring scan, is what
#: lets every message without a payload skip the parse entirely.
_PROTECTED_DATA_MARKER: Final[str] = '"protected_data"'


class TiktokenTokenizer:
    """Exact BPE token counts, so budgets and reuse are measured in real tokens."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        """Create a tokenizer.

        Args:
            encoding: A ``tiktoken`` encoding name. ``o200k_base`` covers current OpenAI
                models and is a reasonable proxy for other vendors' counts.

        Raises:
            RuntimeError: If ``tiktoken`` is not installed.
        """
        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError("The 'tiktoken' tokenizer requires the tiktoken package.") from error
        self._encoding = tiktoken.get_encoding(encoding)

    def count_tokens(self, text: str) -> int:
        """Return the exact number of BPE tokens in ``text``."""
        return len(self._encoding.encode(text))


class ProtectedDataStrippingTokenizer:
    """Count a serialized message as the provider bills it, without its encrypted reasoning.

    **The defect this compensates for, upstream.** ``Content.to_dict`` captures
    ``protected_data``, and ``agent_framework._compaction._serialize_content`` pops
    ``raw_representation`` and ``items`` from that dict but not ``protected_data``, so
    ``agent_framework._compaction._serialize_message`` hands the tokenizer a string that still
    carries every encrypted reasoning payload the client will replay. Base64 tokenizes at
    about 0.68 o200k tokens per character against about 0.2 for prose, so a reasoning trace
    the provider bills at roughly 300 tokens is counted locally at 900 to 1,200. Measured
    across the archive that put gpt-5.6-luna's local count at 1.2 to 1.45 times its billed
    prompt, and every threshold in this package is a fraction of that count: on the same cell
    and code, gpt-5.4-mini (which does not reason) trips the anchored fallback at 0.93-0.95
    of the billed ceiling while luna trips it at 0.61-0.64. Remove this wrapper when upstream
    drops ``protected_data`` from the serialization; nothing else in the package depends on it.

    **What it does.** The framework calls ``count_tokens`` with the serialized string, so the
    wrapper can only change what is counted, never what is sent. When the string contains a
    ``protected_data`` field it is parsed once, the field is removed from every entry of the
    ``contents`` list, the result is re-serialized with the same call ``_serialize_message``
    makes (``ensure_ascii=False, sort_keys=True, default=str``), and that string is what the
    wrapped tokenizer counts. A string with no such field, or one that does not parse as a
    message with a ``contents`` list, is passed to the wrapped tokenizer unchanged, so every
    message without a payload -- every message on a model that does not reason, and every
    user, tool and plain assistant message on one that does -- counts exactly as before and
    existing cells stay comparable.

    **The accounting.** Dropping the payload counts replayed reasoning at zero, and the
    provider does not bill it at zero: it bills the decrypted reasoning, about 300 tokens per
    assistant call on luna. Nothing on the message says how many that is. The OpenAI and
    Foundry clients put ``reasoning_output_token_count`` on the response's usage, not on the
    content, and a reasoning content's ``additional_properties`` carry only ``reasoning_text``
    and ``summary``. So the wrapper trades the over-count for an under-count. The residual is
    the sum of the decrypted reasoning sizes in the prompt, and its direction is low: on the
    fixed-payload 60k cell with ~38 assistant calls in the prompt that is ~11k tokens, about
    19% of the prompt, so a trigger labelled 0.80 fires near 0.99 of billed. Zero is still the
    better of the two numbers the instrument can produce on its own: the error is a third of
    the one it replaces (the over-count was 45% on that cell) and it is bounded by what the
    provider actually bills rather than by the length of a base64 encoding of it. It is not
    small, which is why :data:`REASONING_TOKENS_KEY` exists. A run that stamps the response's
    reasoning token count on the content carrying the payload (:func:`stamp_reasoning_tokens`)
    has that many tokens counted in the payload's place, the stamp itself is not counted, and
    the residual falls to the framing the provider puts around a replayed reasoning item, a
    few tokens per call. Until a run stamps, the wrapper counts zero and says so here.

    **Cost.** A message without a payload pays one substring scan and the wrapped count of
    the original string object: 1.5 microseconds on a 14k-char tool result, invisible next to
    tiktoken's 600 for the same string. A message with a payload pays a ``json.loads``, a
    ``json.dumps`` and the wrapped count of a shorter string: on a 4.5k-char assistant message
    carrying a 1.6k-char payload that is 17 microseconds on top of the estimator, and under
    tiktoken a net saving of about 200, because encoding the base64 cost more than parsing it
    away. Nothing is parsed twice.
    """

    def __init__(self, base: TokenizerProtocol) -> None:
        """Wrap a token counter.

        Args:
            base: The counter that measures the stripped string; whichever the run selected.
        """
        self.base = base

    def count_tokens(self, text: str) -> int:
        """Return the wrapped count of ``text`` less its ``protected_data`` fields.

        Args:
            text: The string to count, normally a message as ``_serialize_message`` emits it.

        Returns:
            The wrapped tokenizer's count of the re-serialized message, plus any stamped
            reasoning token counts; or its count of ``text`` itself when nothing was stripped.
        """
        if _PROTECTED_DATA_MARKER not in text:
            return self.base.count_tokens(text)
        stripped, declared = _strip_protected_data(text)
        if stripped is None:
            return self.base.count_tokens(text)
        return self.base.count_tokens(stripped) + declared


def _strip_protected_data(text: str) -> tuple[str | None, int]:
    """Return ``text`` re-serialized without its ``protected_data`` fields, and the stamps' total.

    Args:
        text: A string that contains the ``protected_data`` marker.

    Returns:
        ``(None, 0)`` when ``text`` is not a serialized message with a ``contents`` list, or
        when no entry of that list carries the field, so the caller counts ``text`` unchanged.
        Otherwise the re-serialized message and the sum of :data:`REASONING_TOKENS_KEY`
        stamps found beside the removed fields, which are removed with them.
    """
    try:
        payload: Any = json.loads(text)
    except ValueError:
        return None, 0
    if not isinstance(payload, dict):
        return None, 0
    contents: Any = cast("dict[str, Any]", payload).get("contents")
    if not isinstance(contents, list):
        return None, 0
    stripped = False
    declared = 0
    for entry in cast("list[Any]", contents):
        if not isinstance(entry, dict) or "protected_data" not in entry:
            continue
        content = cast("dict[str, Any]", entry)
        del content["protected_data"]
        stripped = True
        properties = content.get("additional_properties")
        if isinstance(properties, dict):
            stamp = cast("dict[str, Any]", properties).pop(REASONING_TOKENS_KEY, None)
            if isinstance(stamp, int) and not isinstance(stamp, bool):
                declared += max(stamp, 0)
    if not stripped:
        return None, 0
    # The call ``_serialize_message`` makes, so this is the string the framework would have
    # produced for a message that never carried the payload.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), declared


def stamp_reasoning_tokens(messages: Iterable[Message], reasoning_tokens: int) -> bool:
    """Record a response's reasoning token count on the content carrying its encrypted payload.

    The count goes under :data:`REASONING_TOKENS_KEY` in the ``additional_properties`` of the
    first content in ``messages`` whose ``protected_data`` is set. A response normally carries
    one payload; when it carries several the total sits on the first, which is what
    :class:`ProtectedDataStrippingTokenizer` counts, since it sums stamps across contents.

    This changes the message but not what is sent: the OpenAI and Foundry reasoning
    conversion reads ``status``, ``reasoning_text`` and ``encrypted_content`` from those
    properties and nothing else. It does change the message's serialized string, for the
    framework's count and for the lab's prefix oracle alike, from the first prompt the
    message appears in onwards, so a stamped message is as cacheable as an unstamped one.

    Args:
        messages: The response's messages. The in-memory history provider stores the
            response's own ``Content`` objects, so a stamp applied after the call reaches
            the replayed message.
        reasoning_tokens: ``UsageDetails["reasoning_output_token_count"]`` for the response.

    Returns:
        Whether a content was stamped: False when ``reasoning_tokens`` is not positive or no
        content carries a payload.
    """
    if reasoning_tokens <= 0:
        return False
    for message in messages:
        for content in message.contents:
            if content.protected_data is not None:
                content.additional_properties[REASONING_TOKENS_KEY] = reasoning_tokens
                return True
    return False


def build_tokenizer(name: str) -> TokenizerProtocol:
    """Build a token counter by name.

    Args:
        name: One of :data:`TOKENIZER_NAMES`.

    Returns:
        The token counter, wrapped in :class:`ProtectedDataStrippingTokenizer`.

    Raises:
        KeyError: If ``name`` is not a known tokenizer.
    """
    base: TokenizerProtocol
    if name == "estimator":
        base = CharacterEstimatorTokenizer()
    elif name == "tiktoken":
        base = TiktokenTokenizer()
    else:
        raise KeyError(f"Unknown tokenizer {name!r}. Known tokenizers: {list(TOKENIZER_NAMES)}")
    return ProtectedDataStrippingTokenizer(base)
