# Measured results

Live-agent runs (`cachebench-live`). Each row of each table is the **median of 3 repeats**;
the `+-` column is the spread between cheapest and dearest repeat, and a ranking is only
meaningful where the gap between strategies exceeds it.

## Shared configuration

Unless a run says otherwise:

| Setting | Value |
| --- | --- |
| Scenario | 16 turns, 17 planted facts, 6 tool-call groups |
| Tool output | ~4,000 tokens per result, ~49% of all material |
| Filler | ~2,000 tokens per padding turn |
| Simulated context window | 32,000, minus 2,048 reserved for the reply |
| Working budget | 29,952 tokens |
| Tool calls | pinned (see below) |
| Repeats | 3 |
| Temperature | 0 |

**Tool pinning** matters for comparability. Each tool turn forces its own no-argument tool
via `tool_choice={"mode": "required", "required_function_name": ...}`, and every other turn
is closed with `tool_choice="none"`. Without it, models gather different numbers of facts
between runs, which moves both axes for reasons unrelated to compaction. Runs marked
*unpinned* below predate this and are not comparable with the pinned ones.

## Runs

| # | Model | Route | API | Pinned | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `openai/gpt-5.6-luna` | OpenRouter | Chat Completions | no | `none` |
| 2 | `gpt-5.4-mini` | Azure Foundry | Responses | yes | `none` |

---

### Run 1 — `openai/gpt-5.6-luna` via OpenRouter (unpinned)

Pricing $0.20/M in, $0.02/M cached (10x), $1.20/M out (6x).

| strategy | in | hit% | cost | +- | vs none | lost | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_budget_tools_first | 160,488 | 53% | $0.0196 | 20% | -32% | 13 | 28% |
| token_budget_truncate_first | 153,900 | 43% | $0.0218 | 4% | -25% | 13 | 28% |
| context_window_aggressive | 108,154 | 8% | $0.0231 | 6% | -20% | 15 | 17% |
| truncation | 264,292 | 69% | $0.0240 | 25% | -17% | 11 | 28% |
| token_budget_fallback | 164,010 | 38% | $0.0243 | 10% | -16% | 13 | 28% |
| context_window | 164,055 | 38% | $0.0244 | 11% | -16% | 13 | 28% |
| token_budget_window_first | 164,727 | 34% | $0.0256 | 10% | -11% | 13 | 28% |
| **none** | **710,278** | **92%** | **$0.0289** | **21%** | — | **0** | **100%** |
| token_budget_summarize | 157,392 | 33% | $0.0320 | 11% | +10% | 9 | 39% |
| sliding_window | 191,725 | 7% | $0.0395 | 2% | +36% | 15 | 17% |
| selective_tool_call | 566,308 | 77% | $0.0397 | 4% | +37% | 0 | 100% |
| tool_result | 579,677 | 78% | $0.0398 | 2% | +38% | 0 | 100% |
| summarization | 170,878 | 9% | $0.0493 | 6% | +70% | 6 | 56% |

Effective input (volume adjusted for the cache discount), against `none`:
`tool_result` **+41%** on 18% fewer tokens, `selective_tool_call` **+42%** on 20% fewer.
Break-even needed a **42%** cut at 78% hit rate; they delivered 18%.

*Caveat: unpinned. The agent gathered all 17 facts here, but call counts varied, which is
part of why `none` carries a 21% spread.*

---

### Run 2 — `gpt-5.4-mini` via Azure Foundry (pinned)

Pricing EUR 0.66/M in, 0.07/M cached (9.4x), 3.96/M out (6x). Every run gathered 17/17
facts with 0 unfetched, so all rows describe the same conversation.

| strategy | in | hit% | cost | +- | vs none | lost | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_window_aggressive | 148,685 | 47% | $0.0613 | 10% | -13% | 13 | 28% |
| **none** | **433,829** | **89%** | **$0.0702** | **8%** | — | **0** | **100%** |
| token_budget_tools_first | 205,931 | 53% | $0.0768 | 21% | +9% | 13 | 17% |
| token_budget_truncate_first | 229,435 | 57% | $0.0785 | 12% | +12% | 13 | 28% |
| token_budget_window_first | 197,605 | 47% | $0.0813 | 12% | +16% | 13 | 28% |
| truncation | 345,178 | 73% | $0.0855 | 21% | +22% | 4 | 56% |
| sliding_window | 140,985 | 4% | $0.0940 | 9% | +34% | 15 | 17% |
| token_budget_summarize | 186,719 | 48% | $0.0956 | 11% | +36% | 11 | 39% |
| context_window | 230,409 | 42% | $0.0993 | 12% | +41% | 13 | 28% |
| token_budget_fallback | 225,892 | 36% | $0.1059 | 1% | +51% | 13 | 28% |
| tool_result | 437,288 | 71% | $0.1143 | 68% | +63% | 0 | 100% |
| selective_tool_call | 468,607 | 70% | $0.1257 | 34% | +79% | 0 | 100% |
| summarization | 130,434 | 0% | $0.1268 | 5% | +81% | 1 | 56% |
| context_window_lazy | 309,358 | 42% | $0.1367 | 24% | +95% | 11 | 39% |

Effective input against `none`: `tool_result` **+80%**, `selective_tool_call` **+98%**,
`context_window_aggressive` **-3%**. Break-even needs a **44%** cut at 71% hit rate and
**65%** at 47%.

**Observation unique to this run:** `tool_result` and `selective_tool_call` sent *more*
total tokens than `none` (+1%, +8%) despite ending with smaller prompts (32,988 vs 39,870).
Each took an extra model call, and at this conversation size one extra call costs more than
the trimming saves. Shrinking the prompt does not always reduce spend.

**`summarization` reached a 0% hit rate** — not low, zero. It rewrites history thoroughly
enough that nothing is reusable, then pays for its own summary calls on top.

---

## Observations holding across runs

**The correctness result is a cliff, not a gradient.** Strategies that never evict messages
(`none`, `tool_result`, `selective_tool_call`) lost **0** of 17 facts in every run.
Strategies that evict lost **4 to 15**. Nothing lands in between.

**No strategy has been both cheaper and as accurate**, on any model, provider or API tested.
The only cheaper options destroy most of what the agent was told.

**The governing quantity is the cache discount, not prompt size.** Compaction pays only when
the volume cut exceeds the discount forfeited:

> `T2/T1 < (1 - h1(1-d)) / (1 - h2(1-d))`

where `h` is cache hit rate and `d` is the cached-read price as a fraction of input price.
At the 10x discount common to these models, that means cutting **42-65%** of tokens just to
break even — and strategies cutting that hard are exactly the ones that lose the facts.

**The shipped harness default (`context_window`, 0.5/0.8) has never won.** +16% on run 1,
+41% on run 2, losing 13 of 17 facts in both.
