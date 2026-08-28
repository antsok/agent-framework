# Measured results

Live-agent runs (`cachebench-live`). Each row of each table is the **median of 3 repeats**;
the `+-` column is the spread between cheapest and dearest repeat, and a ranking is only
meaningful where the gap between strategies exceeds it.

**Source data.** Runs 7 onward keep their verbatim output and exact invocation in
[`runs/`](runs/), one pair of files per run. The tables below are a selection of those
columns, copied rather than recomputed, so any figure here can be checked against the log of
the same run number. Runs 1-6 predate that practice and exist only as the tables below.

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
| 7 | `gpt-5.4-mini` | Azure Foundry | Responses | yes | `none` (60K window, harness) |
| 8 | `gpt-5.4-mini` | Azure Foundry | Responses | yes | `none` (120K window, harness) |
| 9a | `gpt-5.4-mini` | Azure Foundry | Responses | yes | *aborted — 400K window does not exist* |
| 9 | `gpt-5.4-mini` | Azure Foundry | Responses | yes | `none` (272K real limit, 86% full) |

Runs 7 to 9 are a **separate experiment** on the harness agent at wider windows, with 53
planted facts instead of 17. The shared configuration above does not describe them; their own
section below does. Runs 7 and 8 hold the conversation at ~165% of the budget so that
compaction is forced to fire; Run 9 is the opposite regime, with the conversation fitting
inside the window.

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

## Window sweep on the harness agent (Runs 7 and 8)

Runs 1-6 all used the `plain` agent at a 32,000-token window with 17 planted facts. Runs 7
and 8 re-ask the question on the **harness** agent — `create_harness_agent`, the wiring a
typical caller gets, including the framework's own narration guidance — at wider windows and
with a much finer accuracy measure.

**Everything below differs from Runs 1-6 except the model and the route**, so these are a
separate experiment, not two more rows in the cross-model table:

| | Runs 1-6 | Runs 7-8 |
| --- | --- | --- |
| Agent | `plain` | **`harness`** |
| Planted facts | 17 | **53** (8 codes per tool result) |
| Turns | 16 | **22** |
| Closing question | one sweeping question | **7 targeted questions**, union covers all 53 |
| System prompt | no retrieval guidance | **retrieval guidance added** |

The last two rows are why this experiment exists. Under the single sweeping question and no
retrieval guidance, identical repeats of the *uncompacted control* returned 42 of 53 and 5 of
53 — a control that unstable makes every accuracy number in the table meaningless. Adding the
retrieval guidance fixed it; splitting the question did not (both forms score 53/53 once the
guidance is present). The targeted form is kept anyway as insurance at larger fact counts.
`--sweeping-question` restores the old behaviour.

**Material is scaled with the window on purpose.** Holding the conversation fixed while
widening the window would leave every strategy inert — nothing to evict, and a tool-oriented
strategy that evicts nothing scores a perfect result for doing nothing, which is a fault this
harness has already produced once. Both runs therefore hold the *pressure* constant at the
same ~165% of budget as Runs 1-6, and vary only the absolute window.

| | Run 7 | Run 8 |
| --- | ---: | ---: |
| Window / reply reservation | 60,000 / 2,048 | 120,000 / 2,048 |
| Working budget | 57,952 | 117,952 |
| Filler per padding turn | 4,000 | 8,000 |
| Tool result size | 8,000 | 16,000 |
| Material (tool share) | ~96,600 (50%) | ~192,000 (50%) |
| Material vs budget | 167% | 163% |

Exact invocation, Run 7. Run 8 differs only in the three sizing flags above.

```sh
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"

cachebench_live foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,truncation,sliding_window,tool_result,selective_tool_call,context_window,context_window_aggressive,context_window_lazy,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize \
  --summarizer-provider foundry:gpt-5.4-mini \
  --repeats 3 \
  --context-window 60000 --max-output-tokens 2048 \
  --markers-per-tool 8 --tool-turns 6 \
  --filler-tokens 4000 --tool-result-tokens 8000 \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
```

Authentication is `DefaultAzureCredential` against an `az login` session, and the deployment
is `gpt-5.4-mini` on GlobalStandard. Prices are the same EUR rates as Run 2 — 0.66/M in,
0.07/M cached (9.4x), 3.96/M out — so the costs are on the same scale as Run 2's even though
the conversations are not the same.

### Run 7 — 60,000-token window, harness agent

Wall clock 54 minutes for the full 14x3 matrix. **Control: 53/53 facts, 100% correct, 0
unfetched, +-5%.** This is the first run in the series whose accuracy baseline is stable
enough to rank against.

| strategy | peak tok | hit% | in | cost | +- | vs none | facts | lost | ignored | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| truncation | 42,855 | 83% | 713,463 | $0.1352 | 26% | -17% | 29/53 | 24 | 24 | 11% |
| context_window_aggressive | 17,301 | 56% | 388,128 | $0.1385 | 2% | -15% | 8/53 | 45 | 0 | 17% |
| token_budget_tools_first | 25,613 | 67% | 523,828 | $0.1459 | 13% | -10% | 16/53 | 37 | 16 | 2% |
| token_budget_window_first | 32,927 | 66% | 505,106 | $0.1495 | 6% | -8% | 16/53 | 37 | 0 | 31% |
| token_budget_truncate_first | 25,823 | 67% | 528,565 | $0.1512 | 3% | -7% | 16/53 | 37 | 8 | 17% |
| **none** | **78,307** | **93%** | **1,309,273** | **$0.1621** | **5%** | — | **53/53** | **0** | **0** | **100%** |
| sliding_window | 21,233 | 7% | 289,092 | $0.1880 | 6% | +16% | 0/53 | 53 | 0 | 2% |
| token_budget_fallback | 25,802 | 62% | 600,575 | $0.1885 | 10% | +16% | 16/53 | 37 | 0 | 31% |
| token_budget_summarize | 26,411 | 59% | 462,546 | $0.1925 | 5% | +19% | 34/53 | 19 | 0 | 65% |
| context_window | 26,937 | 60% | 597,304 | $0.1963 | 7% | +21% | 21/53 | 32 | 0 | 41% |
| selective_tool_call | 61,330 | 84% | 1,129,087 | $0.1978 | 58% | +22% | 53/53 | 0 | 0 | 100% |
| tool_result | 63,274 | 84% | 1,156,724 | $0.2076 | 28% | +28% | 53/53 | 0 | 0 | 100% |
| context_window_lazy | 39,574 | 67% | 836,137 | $0.2368 | 1% | +46% | 24/53 | 29 | 0 | 46% |
| summarization | 17,236 | 0% | 268,482 | $0.2514 | 4% | +55% | 26/53 | 27 | 0 | 50% |

Verdict `none`, for the seventh time: the only two settings that keep the answer intact both
cost more than not compacting.

**Four rows are not rankable on cost.** A gap smaller than the spread is not a result, and
here that disqualifies `truncation` (-17% at +-26%), `token_budget_tools_first` (-10% at
+-13%), `selective_tool_call` (+22% at +-58%) and `tool_result` (+28% at +-28%). Note *which*
rows those are: the two accuracy-preserving strategies are precisely the two whose cost cannot
be pinned down. They behaved the same way on this model at 32,000 (+-68% and +-34% in Run 2),
so it is a property of `gpt-5.4-mini` on this route rather than a fluke. Rows that are safely
rankable: `context_window_aggressive` (-15%, +-2%), `token_budget_truncate_first` (-7%,
+-3%), `context_window_lazy` (+46%, +-1%), `summarization` (+55%, +-4%).

**The break-even formula predicted the cost column before it was read.** Using
`d = 0.070/0.66 = 0.106` with each row's own hit rate and token count:

| strategy | tokens vs none | hit rate | predicted | measured |
| --- | ---: | ---: | ---: | ---: |
| `truncation` | -45% | 93% -> 83% | -17% | **-17%** |
| `tool_result` | -12% | 93% -> 84% | +30% | **+28%** |
| `selective_tool_call` | -14% | 93% -> 84% | +27% | +22% |

At 93% -> 84% the cut required to break even is **32%**; `tool_result` cut 12%. This is the
third independent confirmation of the formula, and the first at a window other than 32,000.

**New finding: surviving compaction is not the same as being usable.** The `facts` column
counts planted codes still present in the closing prompts, and it has been read as the ceiling
on accuracy. Run 7 shows the ceiling is not tight:

| strategy | facts surviving | ignored | correct |
| --- | ---: | ---: | ---: |
| `truncation` | 29/53 | **24** | 11% |
| `token_budget_tools_first` | 16/53 | **16** | 2% |
| `token_budget_summarize` | 34/53 | 0 | 65% |

`truncation` left 29 codes in front of the model and the model used none of them. The codes
survive as strings while the surrounding turns saying *which deployment each belongs to* are
deleted, so a question about the staging deployment cannot be answered from a bare list of
identifiers. `token_budget_summarize` keeps 13 more facts than `context_window` and scores 24
points better, for the same reason in reverse: a summary preserves the labelling.

The practical consequence is that **fact-survival counts overstate what message-deleting
strategies preserve**, and the two axes have to be measured separately. Earlier runs could not
show this, because their controls were never stable enough to trust the accuracy column.

### Run 8 — 120,000-token window, harness agent

Wall clock 1 h 34 min. Same instrument as Run 7, with the material doubled so the pressure on
the window is unchanged. **Control: 53/53 facts, 100% correct, 0 unfetched, +-7%**, peak
prompt 150,112 tokens — still well inside this model's real 400,000-token limit, so `none`
is a genuinely available option here and not an artefact of a small target.

| strategy | peak tok | hit% | in | cost | +- | vs none | facts | lost | ignored | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context_window_aggressive | 33,574 | 69% | 761,131 | $0.2006 | 1% | -33% | 16/53 | 37 | 0 | 31% |
| token_budget_truncate_first | 57,655 | 75% | 1,013,942 | $0.2328 | 6% | -23% | 19/53 | 34 | 8 | 22% |
| token_budget_tools_first | 64,953 | 71% | 951,341 | $0.2406 | 4% | -20% | 16/53 | 37 | 16 | 2% |
| token_budget_window_first | 58,272 | 77% | 1,133,946 | $0.2443 | 0% | -19% | 24/53 | 29 | 24 | 2% |
| truncation | 91,456 | 83% | 1,513,282 | $0.2690 | 2% | -11% | 29/53 | 24 | 8 | 41% |
| **none** | **150,112** | **93%** | **2,498,779** | **$0.3009** | **7%** | — | **53/53** | **0** | **0** | **100%** |
| context_window | 58,541 | 69% | 1,287,983 | $0.3421 | 36% | +14% | 27/53 | 26 | 8 | 37% |
| sliding_window | 41,014 | 6% | 553,299 | $0.3557 | 6% | +18% | 0/53 | 45 | 0 | 17% |
| selective_tool_call | 117,970 | 85% | 2,182,901 | $0.3670 | 4% | +22% | 53/53 | 0 | 16 | 70% |
| tool_result | 118,114 | 84% | 2,168,925 | $0.3681 | 5% | +22% | 53/53 | 0 | 0 | 100% |
| token_budget_fallback | 58,075 | 62% | 1,318,701 | $0.3966 | 17% | +32% | 25/53 | 28 | 9 | 31% |
| token_budget_summarize | 58,423 | 60% | 1,287,404 | $0.4267 | 35% | +42% | 21/53 | 32 | 0 | 41% |
| context_window_lazy | 75,529 | 67% | 1,608,395 | $0.4401 | 11% | +46% | 35/53 | 18 | 0 | 67% |
| summarization | 132,022 | 51% | 2,105,284 | $0.7972 | 59% | +165% | 47/53 | 6 | 5 | 80% |

Verdict `none` again. Only four settings score 70% or better, and three of them cost more:
`tool_result` (+22%, 100%), `selective_tool_call` (+22%, 70%), `summarization` (+165%, 80%).

**The measurement is much tighter here than at 60,000.** Only `context_window` (+14% at
+-36%), `token_budget_summarize` (+42% at +-35%) and `summarization` (+165% at +-59%) are
disqualified by spread. In particular `tool_result` lands at +-5% and `selective_tool_call` at
+-4%, where at 60,000 they were +-28% and +-58% — so this run, unlike Run 7, can actually
rank the accuracy-preserving strategies on cost.

**Formula check, second window.** Same `d = 0.106`:

| strategy | tokens vs none | hit rate | predicted | measured |
| --- | ---: | ---: | ---: | ---: |
| `context_window_aggressive` | -70% | 93% -> 69% | -31% | **-33%** |
| `truncation` | -39% | 93% -> 83% | -7% | -11% |
| `tool_result` | -13% | 93% -> 84% | +28% | +22% |
| `selective_tool_call` | -13% | 93% -> 85% | +24% | **+22%** |
| `context_window_lazy` | -36% | 93% -> 67% | +53% | +46% |
| `summarization` | -16% | 93% -> 51% | +172% | **+165%** |

Every prediction is within 7 points and **all six err in the same direction**, which is
explained rather than hand-waved: the formula models input tokens only, and every compacting
strategy also generates fewer output tokens than `none` (2,775-4,191 against 4,058) at 6x the
input price. The residual is the output term the formula omits.

### What the two windows say together

Runs 7 and 8 differ only in window size and a proportional material scale, so they can be
compared directly with each other. Run 2 is shown for context but used a different agent,
fact count and question format, so its column is not a clean third point.

| | Run 2 — 32K *(plain, 17 facts)* | Run 7 — 60K | Run 8 — 120K |
| --- | ---: | ---: | ---: |
| `none` peak prompt | 39,870 | 78,307 | 150,112 |
| `none` hit rate | 89% | 93% | 93% |
| `tool_result` vs none | +63% *(+-68%)* | +28% *(+-28%)* | **+22%** *(+-5%)* |
| `truncation` vs none | +22% | -17% *(+-26%)* | **-11%** *(+-2%)* |
| cheapest setting | -13% | -15% | **-33%** |
| best accuracy under `none` | 100% (`tool_result`) | 100% (`tool_result`) | 100% (`tool_result`) |

Two things move with the window, in opposite directions:

- **Keeping the answer intact gets cheaper.** The `tool_result` penalty falls from +63% to
  +22%. Its hit rate barely moves (71% -> 84%), but the prefix it preserves is a larger share
  of a larger prompt, so less is re-billed.
- **Throwing the answer away gets much cheaper.** The cheapest setting improves from -13% to
  -33%, because at 120,000 there is far more absolute volume to delete.

So the **gap between "cheap and wrong" and "correct and expensive" widens with scale**: at
32,000 the two ends of the table were 76 points apart, at 120,000 they are 55 points apart on
cost but the accuracy difference is 69 points (31% against 100%). Bigger windows do not make
compaction safer; they make the wrong choice cheaper and therefore more tempting.

The conclusion does not change at any window tested: **no setting beat `none` on both axes.**

**`selective_tool_call` regressed on accuracy at scale** — 53/53 facts surviving but 70%
correct, with 16 ignored, where at 60,000 the same strategy scored 100%. This is the
survives-but-unused effect from Run 7 appearing in a strategy that deletes no messages at all.
With 16,000-token tool results, collapsing an old tool-call group removes enough surrounding
context that codes still present in the prompt stop being attributable to a deployment. It is
the clearest single demonstration that the `facts` column is a ceiling and not a prediction.

### Run 9a — aborted: `gpt-5.4-mini` does not have a 400,000-token input window

The intent was to measure the regime Runs 7 and 8 do not cover: the conversation staying
*inside* the window rather than overflowing it. The model card says 400,000 tokens, so the run
was configured with `--context-window 400000` and material sized to about 85% of it.

**Four of the fourteen strategies died with HTTP 400 `context_length_exceeded`,** and which
four is the whole finding:

| strategy | first threshold at a 397,952 budget | failed at |
| --- | ---: | --- |
| `none` | never compacts | turn 13, all 3 repeats |
| `truncation` | `max_n` = 0.8 -> **318,361** | turn 13, 13, 12 |
| `context_window_lazy` | tool eviction 0.7 -> **278,566** | turn 13, all 3 repeats |
| `summarization` | none — keeps the last N *groups* | turn 15, all 3 repeats |

**The three token-driven failures are exactly the strategies whose first threshold exceeds
272,000.** Everything that fires at or below 50% of the budget survived, including the shipped
`context_window`, whose tool eviction runs at 0.5 -> 198,976. `context_window_lazy` differs
from it only in thresholds, and that difference is the whole distance between working and
failing.

`truncation` failed at **the same turn as `none`**, which is the proof it never compacted
once: the service refused the request long before the local count reached its trigger.

`summarization` is the exception that is not about thresholds. `SummarizationStrategy` is
count-based — it keeps the last N message groups with no token target at all — so it fired
every turn and still could not get under the limit, because at this scale two surviving tool
results are 50,000 tokens on their own. It failed two turns later than the rest, which is what
compaction that runs but cannot keep up looks like.

Local token counts for the conversation, which the failures bracket precisely:

| turn | cumulative prompt | outcome |
| ---: | ---: | --- |
| 12 | 253,266 | accepted |
| 13 | 271,291 | **rejected** |

A direct probe on the same deployment, one prompt per size, pins it:

| prompt | result |
| ---: | --- |
| 260,278 | accepted — service counted 260,284 |
| 270,288 | accepted — service counted 270,294 |
| 275,292 | **`context_length_exceeded`** |
| 290,309 | **`context_length_exceeded`** |

So the advertised 400,000 is **total** context: **272,000 input + 128,000 output**, the
documented GPT-5-class split. There is no configuration in which 400,000 tokens of history
reach this model.

**The error names the offending parameter.** HTTP 400,
`{"type": "invalid_request_error", "code": "context_length_exceeded", "param": "input",
"message": "Your input exceeds the context window of this model."}` — `param: input`, not the
request as a whole.

**And the two limits are independent, not a pool you can allocate between.** Every probe above
ran with `max_tokens=16`, so a tiny output cap was already in force when 275,292 tokens of
input were refused. Varying the cap at fixed input changes nothing:

| input | `max_tokens` | result |
| ---: | ---: | --- |
| 270,293 | 16 | accepted |
| 270,293 | 100,000 | accepted |
| 270,293 | 128,000 | accepted |
| ~280,000 | 16 | **`context_length_exceeded`** |

A large output cap does not consume input allowance, and a tiny one does not buy any. The
service also does not range-check `max_tokens` — 200,000 is accepted without complaint on a
short prompt — so it is a generation cap, not a reservation.

**This matters for how MAF's budget is configured.** `input_budget =
max_context_window_tokens - max_output_tokens` models a *shared* pool, and this model does not
have one. Passing the advertised 400,000 is therefore wrong regardless of what the output
argument is set to. The correct value for `max_context_window_tokens` is **the model's input
limit**, 272,000, whenever a provider states input and output ceilings separately; the
subtraction then just adds a safety margin for the reply.

**The real cause was the output reservation, and it is worth being precise about it.** Every
threshold in MAF is a fraction of an *input budget*, and both layers compute that budget the
same way:

```python
# agent_framework/_compaction.py, ContextWindowCompactionStrategy.__init__
input_budget         = max_context_window_tokens - max_output_tokens
tool_eviction_tokens = int(input_budget * tool_eviction_threshold)   # default 0.5
truncation_tokens    = int(input_budget * truncation_threshold)      # default 0.8
```

This run passed `max_context_window_tokens=400_000` with `max_output_tokens=2_048`, giving an
input budget of **397,952** — 46% above what the service will accept. Had it passed the
model's real output cap instead, the arithmetic would have landed exactly on the limit:

| configuration | input budget | eviction (0.5) | truncation (0.8) |
| --- | ---: | ---: | ---: |
| 400,000 window, **2,048** reservation | 397,952 | 198,976 | **318,361 — unreachable** |
| 400,000 window, **128,000** cap | **272,000** | 136,000 | 217,600 — reachable |

`400,000 - 128,000 = 272,000` is the measured limit to the token. **MAF's formula is correct;
the argument it was given was not.** `max_output_tokens` is documented as *"Maximum output
tokens per response"* — the model's ceiling — not the size of reply you intend to ask for.
Under-setting it inflates the input budget by the difference, and on a GPT-5-class model that
difference is 126,000 tokens, which is enough to push `TruncationStrategy` and any
`ContextWindowCompactionStrategy` above 0.68 out of reach.

The failure mode is still worth knowing, because nothing reports it: the agent runs, compacts
nothing, and dies of an HTTP 400 rather than a large bill. It belongs with the harness
installing no strategy at all when the token arguments are omitted — same class, in that a
plausible configuration silently produces no compaction.

Note which rows survived: everything triggering at or below 0.5 of the inflated budget, which
is why the shipped `context_window` completed and `context_window_lazy` at 0.7 did not. That
is not a defence of the shipped thresholds so much as a demonstration of how much headroom a
wrong reservation eats.

Runs 7, 8 and 9 are unaffected. They set a deliberately *simulated* window with the same
2,048 reservation, and their budgets — 57,952, 117,952 and 269,952 — all sit far below the
272,000 hard limit, so every threshold was reachable. Run 9 is in fact equivalent to the
correctly configured real-world case: its 232,748-token peak is 86% of its 269,952 budget and
86% of the model's true 272,000 input limit alike.

#### The reservation is arithmetic; the cap is a request option, and they are separate

`max_output_tokens` on a compaction strategy only subtracts. Whether the model is actually
held to it depends on a different value reaching the request. Measured on this deployment,
one prompt asking for a long essay:

| request | output produced |
| --- | ---: |
| `max_tokens=32` | exactly 32 |
| `max_tokens=200` | exactly 200 |
| **omitted** | **1,444** |

So the cap works and is honoured exactly — and **there is no modest default**. Omit it and the
model generates whatever it likes, bounded only by its 128,000-token output ceiling.

`create_harness_agent` connects the two, but weakly:

```python
# agent_framework/_harness/_agent.py:633
if max_output_tokens is not None:
    default_opts.setdefault("max_tokens", max_output_tokens)
```

`setdefault`, so a caller who supplies their own `max_tokens` — in `default_options` or in
per-run options — silently keeps the reservation and the cap out of step. **A hand-built
`Agent` carrying a `ContextWindowCompactionStrategy` gets no cap at all**, and then the
reservation is fiction: the strategy compacts to leave room for 2,048 tokens of reply and the
model is free to emit 40,000, overflowing the window the strategy exists to protect.

These runs sit on the safe side of that. The reservation was 2,048 while the request cap was
900 (`--answer-max-tokens`, applied in `_providers.py:93` and travelling per turn), so more
was reserved than could ever be used. The 1,148-token gap understates the input budget by 2%
at 60,000 and 0.4% at 272,000, which changes nothing. Replies averaged ~147 tokens against
that 900-token cap, so nothing was truncated — the control scoring 53/53 is the proof.

**One caveat this raises about earlier work.** The withdrawn *sweeping question* variant asked
for all 53 codes in a single reply. At roughly 12 tokens per labelled code that is ~640 tokens
before any prose, against a 900-token cap — close enough that truncation is a live alternative
explanation for the 42/53 and 5/53 results previously attributed solely to missing retrieval
guidance. The two cannot be separated from the data already collected. Re-running that variant
with a raised cap would settle it, and the targeted-question format now in use is nowhere near
the cap either way.

Note also what is *not* at fault. The bundled `tiktoken` counter agreed with the service to
within **6 tokens on a 270,294-token prompt** (0.002%), so local counting is sound; the error
was entirely in the assumed window.

The corrected run configures the real input limit instead:

```sh
  --context-window 272000 --max-output-tokens 2048 \
  --filler-tokens 12600 --tool-result-tokens 25200
```

which puts the material at ~228,400 tokens and the peak prompt at ~233,000, or **86% of the
272,000 limit** — the 80-90% band the sweep was asking for, measured against the window the
model actually has.

### Run 9 — 86% of the real limit, the conversation fitting inside the window

Wall clock 1 h 45 min, no failed turns, no dropped options. **Control: 53/53 facts, 100%
correct, 0 unfetched, +-0%** — the tightest control in the series, and its 94% hit rate is
the highest.

This is the regime Runs 1-8 never covered. There the conversation ran at ~165% of the budget,
so compaction was forced to fire and the only question was what it cost. Here it **fits**:
window set to the model's true 272,000-token input limit, peak prompt 232,748.

| strategy | peak tok | hit% | in | cost | +- | vs none | facts | lost | ignored | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_budget_window_first | 128,262 | 87% | 2,310,063 | $0.3483 | 1% | **-19%** | 26/53 | 27 | 0 | 50% |
| **none** | **232,748** | **94%** | **3,875,490** | **$0.4317** | **0%** | — | **53/53** | **0** | **0** | **100%** |
| truncation | 207,338 | 89% | 3,057,321 | $0.4365 | 3% | +1% | 37/53 | 16 | 0 | 70% |
| token_budget_truncate_first | 129,314 | 80% | 2,310,122 | $0.4498 | 0% | +4% | 29/53 | 24 | 24 | 11% |
| token_budget_tools_first | 129,766 | 80% | 2,322,751 | $0.4568 | 6% | +6% | 29/53 | 24 | 8 | 41% |
| sliding_window | 64,503 | 9% | 858,120 | $0.5322 | 13% | +23% | 0/53 | 45 | 0 | 17% |
| context_window_aggressive | 78,184 | 61% | 1,780,463 | $0.5500 | 0% | +27% | 23/53 | 30 | 2 | 41% |
| tool_result | 183,119 | 84% | 3,372,103 | $0.5632 | 2% | +30% | 53/53 | 0 | 0 | 100% |
| selective_tool_call | 182,732 | 84% | 3,379,748 | $0.5667 | 4% | +31% | 53/53 | 0 | 0 | 100% |
| context_window_lazy | 183,299 | 84% | 3,400,267 | $0.5815 | 41% | +35% | 53/53 | 0 | 0 | 100% |
| context_window | 129,745 | 76% | 2,684,893 | $0.5841 | 2% | +35% | 37/53 | 16 | 8 | 56% |
| token_budget_fallback | 130,746 | 76% | 2,708,528 | $0.5956 | 43% | +38% | 37/53 | 16 | 32 | 11% |
| token_budget_summarize | 129,363 | 67% | 2,679,770 | $0.7354 | 1% | +70% | 37/53 | 16 | 32 | 11% |
| summarization | 206,143 | 46% | 3,286,712 | $1.3286 | 10% | +208% | 53/53 | 0 | 0 | 100% |

**Thirteen of fourteen strategies cost more than not compacting.** The single exception,
`token_budget_window_first` at -19% (+-1%, so the gap is real), loses 27 of 53 facts and scores
50%. There is no setting on this table that is both cheaper and correct.

Only `context_window_lazy` (+-41%) and `token_budget_fallback` (+-43%) are disqualified by
spread; every other row is a real difference.

**This is the most realistic configuration tested and the starkest result.** The reason is
visible in one column: at 232,748 tokens the control reaches a **94% hit rate**, the highest
anywhere in this work. The longer the conversation, the more of it is discounted, and the more
a mutation to its prefix costs. Compaction here cuts 20-40% of tokens and pays for it with 5-18
points of hit rate, which at a 9.4x discount is a losing trade almost every time.

**Do not read Run 9 as a fourth point on the Run 7-8 curve.** Those two held the conversation
at ~165% of budget and varied the window; this one changes the regime. Grouped properly:

| regime | window | `tool_result` vs none | cheapest that keeps 53/53 |
| --- | ---: | ---: | ---: |
| overflowing, 165% of budget | 60,000 | +28% | +22% (`selective_tool_call`) |
| overflowing, 165% of budget | 120,000 | +22% | +22% (`selective_tool_call`) |
| **fitting, 86% of the limit** | **272,000** | **+30%** | **+30% (`tool_result`)** |

The penalty falls with window size while the conversation overflows, and comes back when it
fits — because a conversation that fits has a nearly perfect cache to lose.

**The shipped default is the worst of both.** `context_window` at its 0.5/0.8 thresholds cost
**+35%** and lost 16 of 53 facts, scoring 56%. It is beaten on cost by seven settings and on
accuracy by four. This is now the fourth run in which the shipped default costs more than no
compaction at all.

**`summarization` bought correctness at 3x the price.** It is the only deleting-class strategy
to keep all 53 facts at 100%, and it cost **+208%**. At this size its own summary calls are
almost free ($0.0474 of $1.3286); the bill is the 46% hit rate.

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

## Known gap: every planted fact sits at the head of its tool result

`ToolResultCompactionStrategy` collapses an old tool-call group by **head-truncating its
result at 4,096 characters** — it does not extract, summarise or select. Our markers sit in
the first ~100 characters of each result, so they survive that cut unconditionally.

**So "tool_result kept 17 of 17 facts" is a fact about our data layout, not about the
strategy.** The same strategy on a workload whose salient content trails its filler would
score zero, and every table here would look identical while measuring the opposite thing.

Planned test: return 6,000-8,000 token results from some tool calls with their facts at the
**end**, keeping others head-placed, so one run measures both. At 4,096 characters the
truncation keeps roughly the first 6-8% of such a result, so the tail-placed facts should be
destroyed while the head-placed ones survive — a strategy that currently scores 100% should
land near 50%. If it does not, the mechanism is not what this note claims.

This is also the argument for content-aware extraction over positional truncation: the
information a strategy keeps should not depend on where in the payload a tool happened to put
it.

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
