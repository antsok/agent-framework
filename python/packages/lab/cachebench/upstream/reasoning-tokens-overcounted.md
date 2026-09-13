<!--
Draft of an upstream issue for microsoft/agent-framework. Not filed.

Filing notes (not part of the issue body):
- Use the "Python Bug Report" template (.github/ISSUE_TEMPLATE/python-issue.yml).
  It sets the title prefix "Python: [Bug]: ", the "Python" label and the bug type.
  The headings below are the template's fields, in the template's order.
- CONTRIBUTING.md asks for a search of existing issues first. Terms:
  protected_data, encrypted_content, compaction token count, annotate_token_counts.
- Every file/line reference was verified on 2026-09-13 against upstream main at
  e2f7db207 (#7968). Re-check line numbers if main has moved before filing.
- The reproduction output below is from a run on 2026-09-13 with the script as
  shown. Re-run and paste fresh output if anything changed.
- The body names no provider endpoint, resource, project, deployment or account.
-->

# Python: [Bug]: Compaction token counter charges encrypted reasoning payloads (`protected_data`) as prompt text

## Description

The compaction token counter serialises each message to JSON and tokenises the string. That JSON includes `Content.protected_data`, the opaque, provider-encrypted reasoning payload that reasoning models return and that clients replay on later calls. The model never tokenises that blob and its length has no relation to what the provider bills for it, yet it is counted as prompt text, at base64's token density: about 0.68 tokens per character under `o200k_base`, three to four times the density of English prose. Every token-aware compaction strategy therefore measures a prompt larger than the one the model receives, by an amount that grows with each retained reasoning item.

Affected: anyone using `agent_framework` compaction with a tokenizer (the default `CharacterEstimatorTokenizer` included) together with a client that stores encrypted reasoning on `text_reasoning` content. Today that is the OpenAI Responses client (`encrypted_content`) and clients built on it, the Anthropic client (thinking `signature`), the Gemini client (`thought_signature`) and the OpenAI Chat Completions client (`reasoning_details`).

### What happens

1. `Content.to_dict` (`python/packages/core/agent_framework/_types.py`, line 1366; the field list at line 1370) captures `protected_data`.
2. `_serialize_content` (`python/packages/core/agent_framework/_compaction.py`, lines 697-703) calls `to_dict(exclude_none=True)` and removes `raw_representation` and `items` so that provider objects and mirrored payloads are not counted. It does not remove `protected_data`.
3. `_serialize_message` (same file, lines 706-715) JSON-dumps the result. `annotate_token_counts` (line 743) and `annotate_message_groups` with a tokenizer (line 688) hand that string to `tokenizer.count_tokens`.
4. `included_token_count` (line 779) sums the per-message counts, and every token-aware strategy gates on it: `TruncationStrategy.max_n` (line 916); `ContextWindowCompactionStrategy`'s 0.5 and 0.8 fractions of `max_context_window_tokens - max_output_tokens` (lines 1835-1837, checked at 1868 and 1873); `TokenBudgetComposedStrategy`; and `SummarizationStrategy`'s selection of what to summarise.

Where the blob comes from, on the OpenAI Responses path: `_parse_response_from_openai` stores a reasoning item's `encrypted_content` as `protected_data` on a `text_reasoning` content (`python/packages/openai/agent_framework_openai/_chat_client.py`, lines 2769-2794; `_parse_chunk_from_openai` does the same for streaming at lines 3286, 3299 and 3468). When a request does not use service-side storage, `_prepare_reasoning_items_for_openai` (line 1824, called from line 1597) reads it back and replays it as `encrypted_content`. So on precisely the configuration in which framework compaction runs at all (`store=False`; with service-side storage the history is not the client's to compact), every assistant turn from a reasoning model carries a blob, and it is counted on every compaction pass for as long as the message is retained. The same blob is also read from `additional_properties["encrypted_content"]` (line 1839), and `to_dict` captures `additional_properties` too.

Other producers of `protected_data`: Anthropic, `python/packages/anthropic/agent_framework_anthropic/_chat_client.py` lines 1530 and 1538; Gemini, `python/packages/gemini/agent_framework_gemini/_chat_client.py` line 1214; OpenAI Chat Completions, `python/packages/openai/agent_framework_openai/_chat_completion_client.py` lines 856 and 903, which store `json.dumps(reasoning_details)` and so may hold readable reasoning text as well as encrypted parts.

### Expected

Two messages that differ only in whether a `text_reasoning` content carries `protected_data` should count the same, just as two messages that differ only in `raw_representation` do. The provider does not tokenise ciphertext, and whatever it bills for a replayed reasoning item is not a function of the blob's character length.

### Actual

The blob is counted verbatim (the base64 alphabet needs no JSON escaping), at about 0.68 tokens per character with `o200k_base` and 0.25 with the default estimator. Output below.

## Code Sample

No provider, no network. `tiktoken` is optional; the default tokenizer shows the same effect.

```python
"""Offline reproduction: the compaction token counter charges for ``protected_data``."""

import base64
import json
import random

from agent_framework import (
    CharacterEstimatorTokenizer,
    Content,
    Message,
    annotate_message_groups,
    included_token_count,
)

ANSWER = (
    "Good bug reports make it easier for maintainers to verify and root cause the "
    "underlying problem. The better a bug report, the faster the problem will be "
    "resolved. Ideally, a bug report should contain a high-level description of the "
    "problem, a minimal reproduction, a description of the expected behavior "
    "contrasted with the actual behavior observed, and information on the environment."
)


def fake_encrypted_content(n_bytes: int, seed: int = 0) -> str:
    """Stand-in for a provider's encrypted reasoning payload: base64 of ciphertext."""
    return base64.b64encode(random.Random(seed).randbytes(n_bytes)).decode("ascii")


def assistant_turn(blob: str | None) -> list[Message]:
    return [
        Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(id="rs_1", text="", protected_data=blob),
                Content.from_text(ANSWER),
            ],
        )
    ]


def count(messages: list[Message], tokenizer) -> int:
    annotate_message_groups(messages, tokenizer=tokenizer)
    return included_token_count(messages)


class TiktokenTokenizer:
    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def main() -> None:
    tokenizers: dict[str, object] = {"CharacterEstimatorTokenizer (default)": CharacterEstimatorTokenizer()}
    try:
        tokenizers["tiktoken o200k_base"] = TiktokenTokenizer()
    except ImportError:
        pass

    for name, tokenizer in tokenizers.items():
        print(f"== {name} ==")
        base = count(assistant_turn(None), tokenizer)
        print(f"{'blob chars':>10}  {'count':>7}  {'delta':>7}  {'tokens/char':>11}")
        print(f"{0:>10}  {base:>7}  {0:>7}  {'-':>11}")
        for n_bytes in (768, 3072, 12288):
            blob = fake_encrypted_content(n_bytes)
            total = count(assistant_turn(blob), tokenizer)
            delta = total - base
            print(f"{len(blob):>10}  {total:>7}  {delta:>7}  {delta / len(blob):>11.3f}")
        print()

    if "tiktoken o200k_base" in tokenizers:
        enc = tokenizers["tiktoken o200k_base"]._enc  # type: ignore[attr-defined]
        prose_tokens = len(enc.encode(ANSWER))
        blob = fake_encrypted_content(12288)
        blob_tokens = len(enc.encode(blob))
        print("== o200k_base token density ==")
        print(f"English prose: {prose_tokens} tokens / {len(ANSWER)} chars = {prose_tokens / len(ANSWER):.3f} tokens/char")
        print(f"base64 blob:   {blob_tokens} tokens / {len(blob)} chars = {blob_tokens / len(blob):.3f} tokens/char")
        print(f"ratio: {(blob_tokens / len(blob)) / (prose_tokens / len(ANSWER)):.2f}x")
        # The base64 alphabet needs no JSON escaping, so the blob is counted verbatim.
        assert json.dumps(blob) == f'"{blob}"'


if __name__ == "__main__":
    main()
```

Output on `main` at e2f7db207, Python 3.13.12, tiktoken 0.13.0:

```
== CharacterEstimatorTokenizer (default) ==
blob chars    count    delta  tokens/char
         0      146        0            -
      1024      407      261        0.255
      4096     1175     1029        0.251
     16384     4247     4101        0.250

== tiktoken o200k_base ==
blob chars    count    delta  tokens/char
         0      129        0            -
      1024      833      704        0.688
      4096     2934     2805        0.685
     16384    11313    11184        0.683

== o200k_base token density ==
English prose: 70 tokens / 385 chars = 0.182 tokens/char
base64 blob:   11178 tokens / 16384 chars = 0.682 tokens/char
ratio: 3.75x
```

A 1,024-character blob adds 704 tokens under `o200k_base` (261 under the default estimator) to a 129-token message, and the effect is linear in blob length. For scale, the same tokenizer counts English prose at 0.18-0.22 tokens per character (0.182 for the paragraph in the script; 0.214-0.221 over this repository's `CONTRIBUTING.md` and ADR 0019), so each character of blob is charged three to four times what a character of real text would be, on top of being charged at all.

## Error Messages / Stack Traces

None. Nothing raises; the counts are wrong.

## Package Versions

agent-framework-core 1.16.0 and agent-framework-openai 1.14.1, built from `main` at e2f7db207. Not a regression tied to a release: `_serialize_content` has never excluded `protected_data` (introduced with the compaction module in #4469; the `items` exclusion was added in #4331), and the OpenAI Responses client has stored `encrypted_content` on `protected_data` since #7233, and in `additional_properties` before that, which `to_dict` captures as well.

## Python Version

Python 3.13.12 on Windows 11 x64. Nothing here is platform-specific.

## Additional Context

### Impact

- Every threshold is a fraction of the inflated count, so compaction fires earlier than configured. The error is the sum, over retained reasoning items, of the tokens counted for each blob, so it grows with the number of turns and with how much the model reasons. It is a per-message offset, not a constant factor, so it cannot be corrected by scaling a threshold or wrapping the tokenizer.
- Compaction that fires early is not free. Truncation and tool-result eviction discard context the model could still have used and break the provider's cached prompt prefix; `SummarizationStrategy` makes paid model calls. `ContextWindowCompactionStrategy` can also log "Compaction could not fit protected messages within the input budget" (line 1878) for a prompt that fits.
- The `compaction_included_tokens_before` and `_after` figures logged by `_run_compaction_strategy` (line 1587) report the inflated numbers, so the telemetry hides the problem rather than showing it.

A field observation, from a benchmark harness in my fork that drives a real `Agent` through these primitives with `tiktoken` `o200k_base` as the tokenizer. Same code and configuration in every record: a simulated 60,000-token window with a 2,048-token output reserve (input budget 57,952) and a fallback gated at 0.9 of that budget (52,156 tokens) on the framework's `included_token_count`. Only the model differs, between two models of one family from one provider.

- With the model that returned no reasoning items, the fallback fired when the billed prompt, the provider's own input-token count for that call, stood at 0.92-0.95 of the input budget (4 records). The local and billed counts agree.
- With the model that returns encrypted reasoning, it fired at 0.58-0.64 of the input budget (10 records). The local count had reached the 52,156-token gate while the provider was billing 33,881-37,106 tokens for the same prompt, so the counter was running at least 1.4-1.5x the billed size at that moment.

### Proposed fix

**Option A, one line.** In `_serialize_content`, exclude `protected_data` alongside `raw_representation` and `items`:

```python
def _serialize_content(content: Content) -> dict[str, Any]:
    return content.to_dict(exclude_none=True, exclude={"raw_representation", "items", "protected_data"})
```

`to_dict`'s `exclude` parameter is already used this way in `_tools.py` line 839. (`raw_representation` is not in `to_dict`'s field list at all, so the existing pop of it is defensive; `items` is the live precedent.)

The trade: a replayed encrypted reasoning item then counts as zero, while the provider bills the decrypted reasoning tokens when it is replayed, so the local count becomes an under-estimate by that amount. That under-count is bounded by the real reasoning tokens. The current over-count is bounded by nothing the framework can see: three to four times prose density, times a ciphertext length the provider chooses. There is one route where A makes things less accurate than today: on the OpenAI Chat Completions client, `protected_data` holds `json.dumps(reasoning_details)`, which can include readable reasoning text that the provider may re-tokenise on replay, and A stops counting that text.

**Option B, count at billed size.** Providers report reasoning tokens per response, and the Responses client already maps `output_tokens_details.reasoning_tokens` to `UsageDetails.reasoning_output_token_count` (`_chat_client.py` lines 3544-3547). If each client attached that number to the `text_reasoning` content it creates (for example in `additional_properties`), `_serialize_content` could substitute it for the blob. That is the accurate number, but it needs a convention every client follows, a hook in the serialiser, per-item attribution when a response carries more than one reasoning item, and confirmation per provider that replayed reasoning is billed at that size on every call, which I have not verified from provider documentation.

I would take Option A now, with the test below, and treat Option B as a follow-up if maintainers want the accuracy. A removes an unbounded error in the direction that costs users context and money, leaves a bounded one, and matches how the serialiser already treats the other non-text fields.

### Tests

No existing test would have caught this. The only test that exercises `_serialize_message` is `test_serialize_message_preserves_non_ascii_for_token_count` (`python/packages/core/tests/core/test_compaction.py`, line 1999), and nothing asserts what `_serialize_content` excludes; the `raw_representation` and `items` exclusions are untested too.

This test fails on current `main` at its first assertion and passes with Option A:

```python
from agent_framework import CharacterEstimatorTokenizer, Content, Message
from agent_framework._compaction import _serialize_message


def test_serialize_message_excludes_protected_data_from_token_count() -> None:
    """Encrypted reasoning payloads are opaque to the model's tokenizer and are not
    prompt text; they must not inflate the compaction token estimate."""
    blob = "A" * 4000
    with_blob = Message(
        role="assistant",
        contents=[Content.from_text_reasoning(id="rs_1", text="", protected_data=blob)],
    )
    without_blob = Message(
        role="assistant",
        contents=[Content.from_text_reasoning(id="rs_1", text="")],
    )
    tokenizer = CharacterEstimatorTokenizer()

    assert blob not in _serialize_message(with_blob)
    assert tokenizer.count_tokens(_serialize_message(with_blob)) == tokenizer.count_tokens(
        _serialize_message(without_blob)
    )
```

A companion test asserting that `items` is excluded from the count would close the same gap for the existing exclusion.
