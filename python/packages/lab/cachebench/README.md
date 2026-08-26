# cachebench — compaction vs. prompt caching

Measures what Agent Framework's compaction strategies cost you in provider prompt-cache
hits, across providers, at mid and large context sizes.

## Why this exists

Provider prompt caches match on **exact prefixes**. Every compaction strategy in
`agent_framework._compaction` works by excluding or rewriting messages *inside* an
existing history. So compaction breaks the cached prefix by construction — the only
questions are how badly, how often, and whether the prompt tokens it saves are worth more
than the cache reads it destroys.

That trade-off is not obvious in either direction:

- Compacting **more** shrinks every prompt but re-breaks the prefix, forcing a full
  re-prefill at full price.
- Compacting **less** keeps the cache warm but sends more tokens, most of them discounted.

There is a cadence that minimises real cost, and it differs per provider because cache
discounts, minimum cacheable sizes, and TTLs differ.

## How it measures

Conversations are **scripted, not live**. Each turn appends fixed request messages,
compaction runs over the history exactly as `CompactionProvider.before_run` would, the
projection goes to the provider, and then a *scripted* reply is appended — the model's real
answer is discarded. That is what lets every provider and every strategy replay a
byte-identical conversation, which is the only way cross-provider numbers mean anything.

Two independent measurement channels:

| Channel | Source | Available on |
|---|---|---|
| **Reported** | `cache_read_input_token_count` in `UsageDetails`, which Agent Framework already normalises across providers | Providers that report it |
| **Local prefix oracle** | This package recomputes how much of each prompt stayed byte-identical to the previous one | Always |

The oracle is the theoretical ceiling: a provider can never serve more cache than the
prefix that survived. Comparing the two gives `real%` — how much of the reusable prefix the
provider actually delivered. On providers that report nothing, the oracle plus latency is
all you get, and the tool says so rather than printing a misleading 0%.

Matching is at **message granularity**: a message that changed at all contributes zero
reusable tokens. The oracle therefore never overstates reuse.

## Provider support

| Provider | Cache reporting | Engages | Notes |
|---|---|---|---|
| `azure` | yes | automatic | 1,024-token minimum, 128-token increments before GPT-5.6. TTL 5–10 min idle, 1 hour absolute. Cache reads discounted ~50%. |
| `openrouter` | yes | automatic | Returns `cached_tokens` and `cache_discount`, via the standard `agent_framework_openai` path. **Pin `OPENROUTER_PROVIDER_ORDER`** — otherwise routing changes upstream between turns and you are measuring the router, not the cache. |
| `mistral` | yes | automatic, **intermittent** | Caches with no `prompt_cache_key` at all. But engagement is erratic — see below — so a single repeat is noise. Cache reads billed at **10%** of input; pass `--cache-read-ratio 0.1`. |
| `foundry` | unknown | — | Depends on the deployed model. |
| `ollama` | **no** | automatic, but invisible | Caching demonstrably happens and is never reported. Judge Ollama by `reuse%` and latency only. |

Ollama measured directly against `ollama.com` on 2026-08-25 with a 6,000-token shared
prefix, across `glm-5.2`, `minimax-m3`, `gpt-oss:120b` and `mistral-large-3:675b`, on both
`/api/chat` and `/v1/chat/completions`: **no cache field on any of them**, and
`prompt_eval_count` stayed pinned at the full prompt size on every call. Yet `glm-5.2` went
4,078 ms cold → 1,157 ms → 1,056 ms warm on that identical prefix. Prefix KV reuse is real
there; the usage payload just never mentions it. Two independent filters would hide it even
if the server did send one: the local daemon reshapes cloud responses (it drops
`prompt_eval_duration`, `load_duration` and `eval_duration` on `:cloud` models), and the
`ollama` SDK's `ChatResponse` is a closed pydantic model that discards unknown fields.
Re-check with `samples/probe_ollama_usage.py`, which bypasses the SDK.

> **Mistral engages caching intermittently.** Measured 2026-08-25 on
> `mistral-large-latest`, three identical 6k-token-prefix calls reported
> `cached_tokens` of 0, 0, 6000 without a key — and 0, 6000, 0 with one. Across two full
> `mid` sweeps the same cell swung from 42.1% to 29.0% hit rate on byte-identical input.
> Treat any single-repeat Mistral number as noise: use `--repeats` and read the spread,
> not the value. `prompt_cache_key` is *not* the switch — it made no measurable
> difference — so no provider sends one unless you pass `--prompt-cache-key`.

Environment variables per provider:

```bash
# azure — direct Azure OpenAI deployment (distinct from the foundry project route)
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_COMPLETION_MODEL
# foundry — also needs a working DefaultAzureCredential (`az login`, or a managed
# identity when deployed); a project endpoint alone is rejected by the client
FOUNDRY_PROJECT_ENDPOINT, FOUNDRY_MODEL
# openrouter — model must be a real slug, e.g. z-ai/glm-5.2; check /api/v1/models
OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_PROVIDER_ORDER  # e.g. "openai"
# mistral — MISTRAL_MODEL is accepted as a fallback
MISTRAL_API_KEY, MISTRAL_CHAT_MODEL
# ollama (cloud) — model drops the ":cloud" suffix on the direct API
OLLAMA_MODEL, OLLAMA_HOST=https://ollama.com, OLLAMA_API_KEY
```

A provider that fails to construct is skipped with a warning rather than aborting the
sweep, so one missing credential does not cost you every other provider's cells.

## Usage

Validate the matrix and see prompt sizes without spending anything:

```bash
cachebench --dry-run --providers azure --sizes mid,large --strategies none,context_window,truncation
```

A cheap live sweep (the defaults: one provider, `mid`, four strategies, one repeat):

```bash
cachebench --providers azure
```

Compare models on the same provider with `provider:model` — cache behaviour varies by model
family at least as much as it varies by provider:

```bash
cachebench --providers "openrouter:openai/gpt-5.4-mini,openrouter:z-ai/glm-5.2,foundry:gpt-5.6-luna"
```

The full cross-provider comparison:

```bash
cachebench \
  --providers azure,mistral,openrouter,ollama \
  --sizes mid,large \
  --strategies none,context_window,context_window_aggressive,context_window_lazy,truncation,sliding_window,tool_result \
  --repeats 3
```

Results are printed as a table and written to `--out` as per-turn JSONL plus a summary CSV.

### Controlling spend

Output is capped at `--response-max-tokens 16` because answers are discarded — you are only
paying for prompts. Cost scales with `sizes` × `strategies` × `providers` × `repeats`, and
`--dry-run` reports exactly how many prompt tokens a live run would send. Start there.

`mid` is ~20 turns and ~50 messages; `large` is ~100 turns and ~270 messages, and costs
roughly 20× more per cell.

## Reading the output

A real `mid` sweep on `mistral-large-latest`, two repeats, cache reads priced at 10%:

```text
provider  size  strategy        turns  sent_tok  in_tok  cached  hit%  reuse%  real%  breaks  eff_in@0.1
mistral   mid   none            20     93,786    48,723  20,512  42.1  91.3    46.1   0       30,262
mistral   mid   none            20     93,786    48,723  14,112  29.0  91.3    31.7   0       36,022
mistral   mid   truncation      20     42,106    22,721   7,392  32.5  79.2    41.1   5       16,068
mistral   mid   truncation      20     42,106    22,721   6,560  28.9  79.2    36.5   5       16,817
mistral   mid   context_window  20     31,668    17,419   6,720  38.6  78.0    49.5  12       11,371
mistral   mid   context_window  20     31,668    17,419   5,488  31.5  78.0    40.4  12       12,480
```

Two things to read off it. First, `in_tok` is **identical across repeats** for each
strategy — that is the byte-identical replay working, and it is what makes the varying
`cached` column attributable to the provider rather than to the harness. Second, on this
provider compaction wins decisively on cost: `context_window` lands at roughly a third of
the baseline's effective input despite breaking the prefix 12 times, because Mistral only
realises 30–50% of the reusable prefix anyway. The lost discount is smaller than the saved
tokens. On a provider that reliably realises ~100%, that arithmetic can invert — which is
the whole reason to measure per provider rather than reason about it.

- `sent_tok` — total prompt tokens the strategy actually sent across the session.
- `reuse%` — share left byte-identical to the previous prompt. **The cache ceiling.**
- `hit%` — what the provider actually served from cache.
- `real%` — `hit%` ÷ `reuse%`, a quotient of two fractions. Below 100% means misses
  compaction does *not* explain: eviction, TTL expiry, minimum-size floors, intermittent
  engagement, or (on OpenRouter) upstream re-routing. It is deliberately *not*
  `cached ÷ reusable_tokens`: those totals use different tokenizers (provider vs. local
  estimator, which runs ~2× higher), and dividing them directly halves the answer.
- `breaks` — turns where the prompt was not a pure extension of the previous one. Each one
  is a forced re-prefill.
- `no_in` — turns that reported cached tokens but no input count. Some providers drop
  `input_token_count` on a cache hit; when this is non-zero, `hit%` is suppressed rather
  than divided by a denominator the provider never sent.
- `eff_in@0.25` — fresh tokens plus cached tokens priced at `--cache-read-ratio`. Set this
  to your provider's actual cache-read discount to compare strategies on real cost.

The baseline to compare against is always `none`: it sends the most tokens but breaks the
prefix zero times.

## Experimental controls

These matter, and the tool enforces them:

- **Cache namespace isolation.** Each cell gets a unique salt at the very front of the
  system message, so cells cannot serve each other cache hits.
- **Turn 1 is always a cache write, never a read.** It is included in totals because a real
  session pays for it too.
- **Sequential execution.** Overlapping cells would contend for the same cache and rate
  limits.
- **The system anchor is sized above 1,024 tokens** so that prompts clear the provider
  minimum from turn 1. Otherwise early turns report zero cached tokens for reasons that
  have nothing to do with compaction.
- **Opt-in caching is opted into.** Mistral gets a per-cell `prompt_cache_key` derived from
  the cell salt — stable across the cell's turns, distinct between cells. Without it the
  provider simply never caches and the whole row is a false negative.
- **Cached tokens are clamped to the input count** they are a subset of, and turns that
  report cache reads without an input count are excluded from `hit%`. Both guard against
  ratios above 100% that read as a broken benchmark rather than as upstream inconsistency.
- **Simulated context window.** Budgets default to 60% of a transcript's fully-replayed
  size rather than the model's real window — a 20-turn transcript never approaches 128k, so
  a real window would mean no strategy ever fires. Override with `--context-window`.

## Compaction can shrink prompts out of cacheable range

The most surprising measured result, on `foundry` / `gpt-5.4-mini`:

```
strategy         per-turn input tokens        cached
none             668 → 4,096 (growing)        0 until turn 7, then 1280/1792/2304/…
truncation       667–1,400 (oscillating)      0 on every turn   (9/20 turns below 1,024)
context_window   666–850  (pinned)            0 on every turn  (20/20 turns below 1,024)
```

Compaction did not break the cache here — it shrank prompts **below the provider's minimum
cacheable size**, so caching never engaged at all. The uncompacted control proves the
mechanism is size and not compaction: with no compaction whatsoever, the same model still
reported 0 cached at 1,216 / 1,325 / 1,434 tokens, and only began caching at 1,769. Every
compacted prompt sat below that.

The practical consequence: on a provider with a 1,024-token floor, a strategy aggressive
enough to hold prompts near that floor forfeits caching entirely. Compare `eff_in` rather
than `hit%` before concluding it was worth it, and consider raising the compaction budget
so prompts stay comfortably above the floor.

### Scope: stateless routes only

Every provider here is driven statelessly — the full projected message list goes up on each
turn. `FoundryChatClient` carries no `conversation_id` / `store` / `previous_response_id`,
so the Foundry rows measure the stateless path.

Routes where the **service** owns the conversation (Responses-style APIs, hosted agent
threads) are a different regime and are out of scope. There, the client uploads only a delta
and the service maintains a stable prefix of its own — so local compaction works against it,
rewriting history the service was already caching. Applications that let the service own
context should expect the opposite conclusion from the one this benchmark reaches, and the
useful comparison there is compaction-on versus compaction-off, not strategy versus strategy.

## Caveats

- Token counts used locally come from `CharacterEstimatorTokenizer` (4 chars/token), so
  `sent_tok` and `reuse%` are estimates. Provider-reported `in_tok` and `cached` are exact.
  Ratios between strategies are reliable; absolute local token counts are not.
- Cache TTLs are minutes. A long sweep may see later cells behave differently from earlier
  ones purely through cache pressure. Use `--repeats` and compare variance.
- `real%` above 1.0 is possible and means the provider served cache beyond what the
  message-granularity oracle predicted — usually partial-message token-level matching.

## Development

```bash
cd python/packages/lab
poe test-cachebench
```

Tests are offline; the provider call is stubbed.
