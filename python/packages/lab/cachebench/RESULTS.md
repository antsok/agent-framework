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
| 1 | `openai/gpt-5.6-luna` | OpenRouter | Chat Completions | yes | `none` |
| 2 | `gpt-5.4-mini` | Azure Foundry | Responses | yes | `none` |
| 3 | `gpt-5.6-luna` | Azure Foundry | Responses | yes | `none` |
| 4 | `z-ai/glm-5.3-flash` | OpenRouter | Chat Completions | no | *partly usable* |
| 5 | `z-ai/glm-5.2` | OpenRouter / Crusoe | Chat Completions | no | `none` (5x discount) |
| 6 | `z-ai/glm-5.2` | OpenRouter / BaseTen | Chat Completions | no | `none` (10x discount) |

---

### Run 1 — `openai/gpt-5.6-luna` via OpenRouter (pinned, backend pinned)

Pricing $0.20/M in, $0.02/M cached (10x), $1.20/M out (6x). Backend pinned to OpenAI with
`allow_fallbacks: false` — this model has **five OpenRouter backends spanning $0.100-$0.400/M
input**, so unpinned routing can move cost 4x on its own. Every run gathered 17/17 facts with
0 unfetched, no errors.

| strategy | in | hit% | cost | +- | vs none | lost | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_budget_truncate_first | 156,393 | 54% | $0.0191 | 28% | -11% | 13 | 28% |
| token_budget_tools_first | 160,996 | 48% | $0.0212 | 23% | -2% | 15 | 17% |
| **none** | **450,590** | **90%** | **$0.0216** | **9%** | — | **0** | **100%** |
| context_window_aggressive | 113,571 | 9% | $0.0236 | 4% | +9% | 15 | 17% |
| truncation | 247,146 | 65% | $0.0247 | 4% | +14% | 13 | 28% |
| token_budget_window_first | 170,597 | 34% | $0.0268 | 14% | +24% | 15 | 17% |
| token_budget_fallback | 169,143 | 22% | $0.0302 | 12% | +40% | 13 | 28% |
| sliding_window | 144,408 | 1% | $0.0315 | 1% | +46% | 15 | 17% |
| tool_result | 435,044 | 78% | $0.0319 | 12% | +47% | 0 | 100% |
| context_window | 188,808 | 23% | $0.0332 | 16% | +54% | 13 | 28% |
| selective_tool_call | 436,465 | 78% | $0.0333 | 15% | +54% | 0 | 100% |
| token_budget_summarize | 158,759 | 39% | $0.0360 | 11% | +67% | 0 | 67% |
| context_window_lazy | 245,506 | 27% | $0.0408 | 19% | +89% | 13 | 28% |
| summarization | 135,744 | 2% | $0.0442 | 19% | +105% | 7 | 61% |

Effective input against `none`: `tool_result` **+51%** and `selective_tool_call` **+52%**, both
on 3% *fewer* tokens; `token_budget_truncate_first` **-6%** on 65% fewer.

**Same model on two routes, both pinned — an unusually direct comparison:**

| | OpenRouter | Foundry |
| --- | ---: | ---: |
| `none` cost | $0.0216 | $0.0215 |
| `none` hit rate | 90% | 91% |
| `tool_result` vs none | +47% | +34% |
| `context_window` vs none | +54% | +23% |
| `context_window` hit rate | 23% | **38%** |

The uncompacted baseline is the same to within 0.5%. The *compaction penalties* differ, and
the hit-rate column says why: **Foundry's cache survives compaction better than OpenRouter's**
(38% vs 23% under the same strategy). The direction of every conclusion is identical; the
magnitude is route-dependent.

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

### Run 3 — `gpt-5.6-luna` via Azure Foundry (pinned)

Pricing $0.20/M in, $0.02/M cached (10x), $1.20/M out (6x); OpenRouter list rates used as a
stand-in, since only the ratios affect the relative figures. `temperature` is rejected by
this deployment and was dropped (`NO:temp` on every row, applied uniformly). Every run
gathered 17/17 facts with 0 unfetched.

**The tightest measurement in the set**: the control varied by 1% across three repeats, so
the cost differences below are unusually trustworthy.

| strategy | in | hit% | cost | +- | vs none | lost | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_budget_tools_first | 159,921 | 65% | $0.0167 | 19% | -22% | 13 | 28% |
| token_budget_truncate_first | 164,176 | 65% | $0.0182 | 15% | -15% | 13 | 28% |
| **none** | **453,004** | **91%** | **$0.0215** | **1%** | — | **0** | **100%** |
| context_window_aggressive | 123,374 | 20% | $0.0229 | 10% | +6% | 15 | 17% |
| truncation | 261,730 | 72% | $0.0233 | 13% | +8% | 10 | 33% |
| token_budget_window_first | 170,460 | 40% | $0.0251 | 1% | +17% | 15 | 17% |
| context_window | 179,512 | 38% | $0.0265 | 18% | +23% | 13 | 28% |
| token_budget_fallback | 181,009 | 39% | $0.0270 | 27% | +25% | 13 | 28% |
| tool_result | 419,919 | 78% | $0.0288 | 33% | +34% | 0 | 100% |
| selective_tool_call | 421,435 | 78% | $0.0306 | 14% | +42% | 0 | 100% |
| sliding_window | 144,743 | 1% | $0.0316 | 0% | +47% | 15 | 17% |
| token_budget_summarize | 165,895 | 48% | $0.0319 | 5% | +48% | 6 | 56% |
| context_window_lazy | 253,705 | 25% | $0.0428 | 31% | +99% | 13 | 28% |
| summarization | 135,247 | 1% | $0.0440 | 10% | +104% | 7 | 50% |

Effective input against `none`: `tool_result` and `selective_tool_call` both **+53%** on 7%
*fewer* tokens; `token_budget_tools_first` **-19%** on 65% fewer. Break-even needs a **39%**
cut at 78% hit rate, **56%** at 65%, **72%** at 38%.

**Same model as run 1, different route and API, and it agrees**: information-preserving
strategies cost 34-42% more, the correctness cliff is identical, and the only cheaper options
lose 13 of 17 facts.

---

### Run 4 — `z-ai/glm-5.3-flash` via OpenRouter, Z.AI pinned (cost axis only)

Pricing $0.075/M in, $0.015/M cached (**5x**), $0.25/M out. Forcing is unavailable: every
backend returns 404 for any `tool_choice` other than `auto`, despite all six advertising
`tool_choice` support. So the run is unpinned throughout, which at least makes its rows
comparable with each other.

**Only six of fourteen strategies completed.** Every strategy built on
`TokenBudgetComposedStrategy` — all `token_budget_*` and all three `context_window*` — failed
at turn 2 with a 400. This is not a provider fault: Z.AI, Novita and GMICloud all reject it,
with different error text. At small scale the same strategies succeed, and their projected
message shapes and annotations are byte-identical to `truncation`, which works. Unresolved.

**The correctness axis is unusable.** The control scored 6%, with all 17 facts present and
unused. That is this model's answer discipline, not compaction.

The cost axis, for the six that completed:

| strategy | in | hit% | cost | vs none | lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| truncation | 322,640 | 70% | $0.0118 | **-44%** | 3 |
| sliding_window | 245,614 | 15% | $0.0171 | -19% | 15 |
| **none** | 947,039 | 90% | $0.0210 | — | 0 |
| selective_tool_call | 686,993 | 69% | $0.0241 | +15% | 0 |
| tool_result | 740,922 | 70% | $0.0258 | +23% | 0 |
| summarization | 220,588 | 17% | $0.0313 | +49% | 3 |

---

### Runs 5 and 6 — `z-ai/glm-5.2`: the discount tested under control

Every other run confounds the cache discount with the model. This pair does not.
**OpenRouter serves glm-5.2 on two backends at the same $1.40/M input and $4.40/M output,
differing only in the cached-read price**: Crusoe at $0.26 (5x) and BaseTen at $0.14 (10x).
Same model, same weights, same input and output rates. The discount is the only variable.

Both runs: 0 errors, 17/17 facts, control 100% correct, control spread 1% and 10%. Unpinned
tools (glm-5.2 also returns 404 for forcing), 10 strategies.

| strategy | 5x (Crusoe) | 10x (BaseTen) | lost |
| --- | ---: | ---: | ---: |
| token_budget_tools_first | -41% | -22% | 13 |
| context_window_aggressive | -41% | -19% | 13 |
| truncation | -35% | -19% | 11-13 |
| context_window | -35% | -7% | 13 |
| token_budget_fallback | -23% | +0% | 13 |
| sliding_window | -21% | +9% | 15 |
| summarization | -11% | +22% | 13 |
| **none** | — | — | **0** |
| selective_tool_call | **+2%** | **+22%** | 0 |
| tool_result | **+3%** | **+24%** | 0 |

**Halving the discount moved every strategy against compaction, and none moved the other
way.** The information-preserving pair goes from roughly free (+2%, +3%) to clearly expensive
(+22%, +24%). `truncation`'s saving halves.

**The prediction was made before the second run.** From the 5x run's own token counts and hit
rates, the formula projected `tool_result` at +17% and `truncation` at -24% under a 10x
discount. Measured: **+24%** and **-19%**. Both errors sit inside the run-to-run spread this
model shows (2-26%), and every strategy moved in the predicted direction.

This is the strongest evidence in the set that **the cached-read price, not the strategy and
not the model, decides whether compaction pays.**

Note also that `none` remained the *only* setting keeping all 17 facts in both runs, at both
discounts. The discount moves the cost axis. It does not touch the correctness axis.

---

## Models that could not be measured

**`google/gemini-3.7-flash` — excluded.** Its turns fail partway through a conversation
with `Invalid thought signature.` (Google, HTTP 400). Gemini's reasoning models require the
thought signature attached to earlier assistant turns to survive intact into later requests,
and that is exactly what compaction rewrites. Simple calls succeed, and calls with tools
succeed; the failure appears once a multi-turn history is replayed. Upstream rate limiting
(HTTP 429 from Google AI Studio) was also observed on the same route.

This is not a harness defect and repeats would not fix it. It is a real constraint worth
knowing: **client-side compaction and reasoning models that sign their thoughts are not
straightforwardly compatible.** Any strategy that rewrites or drops an assistant turn risks
invalidating the signature chain, which fails the request outright rather than degrading the
answer.

## Observations holding across runs

**The correctness result is a cliff, not a gradient.** Strategies that never evict messages
(`none`, `tool_result`, `selective_tool_call`) lost **0** of 17 facts in every run.
Strategies that evict lost **4 to 15**. Nothing lands in between.

**No strategy has been both cheaper and as accurate**, on any model, provider or API tested.
The only cheaper options destroy most of what the agent was told.

**The cost result depends on the cache discount, and only on that.** At the 10x discount
shared by luna and 5.4-mini, compaction loses. At glm-5.3-flash's 5x, `truncation` saves 51%.
The break-even formula predicts which side a model falls on before running it — the discount
and the achieved hit rate are the only inputs it needs.

**The governing quantity is the cache discount, not prompt size.** Compaction pays only when
the volume cut exceeds the discount forfeited:

> `T2/T1 < (1 - h1(1-d)) / (1 - h2(1-d))`

where `h` is cache hit rate and `d` is the cached-read price as a fraction of input price.
At the 10x discount common to these models, that means cutting **42-65%** of tokens just to
break even — and strategies cutting that hard are exactly the ones that lose the facts.

**The shipped harness default (`context_window`, 0.5/0.8) has never won.** +16% on run 1,
+41% on run 2, losing 13 of 17 facts in both.
