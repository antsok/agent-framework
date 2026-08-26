# Copyright (c) Microsoft. All rights reserved.

"""Token counters used for compaction budgets and the local prefix oracle.

The default ``CharacterEstimatorTokenizer`` assumes 4 chars/token over serialized JSON,
which runs roughly 2x a real BPE count for this benchmark's content. That is harmless when
comparing strategies at small sizes, but at 100k-plus prompts it moves a compaction
threshold by six figures — so large runs should count real tokens.
"""

from __future__ import annotations

from typing import Final

from agent_framework import CharacterEstimatorTokenizer, TokenizerProtocol

__all__ = ["TOKENIZER_NAMES", "TiktokenTokenizer", "build_tokenizer"]

TOKENIZER_NAMES: Final[tuple[str, ...]] = ("estimator", "tiktoken")


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


def build_tokenizer(name: str) -> TokenizerProtocol:
    """Build a token counter by name.

    Args:
        name: One of :data:`TOKENIZER_NAMES`.

    Returns:
        The token counter.

    Raises:
        KeyError: If ``name`` is not a known tokenizer.
    """
    if name == "estimator":
        return CharacterEstimatorTokenizer()
    if name == "tiktoken":
        return TiktokenTokenizer()
    raise KeyError(f"Unknown tokenizer {name!r}. Known tokenizers: {list(TOKENIZER_NAMES)}")
