# Compaction on `gpt-5.4-mini`: report

**Date:** 30 August 2026
**Model:** `gpt-5.4-mini`, Azure Foundry (Responses API), GlobalStandard
**Agent:** the MAF harness — `create_harness_agent`, the wiring a typical caller gets
**Scale:** ~40 matrices over three windows, several hundred conversations, about EUR 150 of
model spend

This is a single-model deep dive and it stands apart from the six-model work in
[`REPORT.md`](REPORT.md), which used a different agent, 17 planted facts instead of 53, and a
control too unstable to rank against. **Nothing here should be averaged with it.**

---

## 1. The headline

**Which compaction strategy is right depends on how large your tool results are, and the
answer inverts between 8,000 and 25,000 tokens.**

| tool result size | best strategy | facts kept | cost vs none |
| ---: | --- | ---: | ---: |
| 8,000 tokens | **record the values, drop the originals** | 53/53 | **-9%** |
| 16,000 tokens | *(crossing over)* | 46/53 vs 53/53 | -8% vs +20% |
| 25,200 tokens | **keep a fixed proportion of each result** | 53/53 | +10% |

Asking the model to extract the values and then dropping what it extracted is the cheapest
approach at every size — and its accuracy falls apart as results grow, because the model
writes a thinner record when there is more to record. Keeping a fixed *proportion* of each
result does the opposite: it is poor when the proportion is small and near-perfect when the
window is large enough to make it generous.

## 2. What was measured

A real agent runs the same 22-turn conversation, changing only the compaction setting. It is
told **53 verifiable facts** — requirements, one mid-conversation correction, and codes
returned by six tool calls — then asked eight closing questions: one per deployment scope,
plus one asking for everything at once.

Scoring is exact substring matching on 6-hex-digit codes. No grader model, no partial credit.
Each closing reply is scored **only against the values its own question asked for**, so a code
listed under the wrong tool does not count: that distinguishes preserving a value from
preserving the labelling that says where it came from.

**Every run is checked against its control before anything else is read.** The uncompacted
row must show 53/53 facts, 0 unfetched, an empty flags column, and a narrow spread. Runs that
failed those checks are recorded as failures rather than quietly dropped.

## 3. The four strategies compared

| | mechanism |
| --- | --- |
| `none` | the uncompacted control |
| `truncation` | oldest-first eviction at 80% of the budget, down to 50% |
| `tool_result` | the framework's tool-result collapse, head-truncating at 4,096 characters |
| `anchored` | keeps a fixed head and tail verbatim, shortens the band between them to a share of the ceiling, position-only decisions (`_anchored.py`) |
| `tool_summary_anchored` | a middleware forces one recall tool call; the strategy then drops every tool group in front of the resulting record (`_toolsummary.py`) |

The last two were written for this work, against what the earlier runs measured.

## 4. Results

Five repeats each, pinned tool calls, neutral narration, values spread through each result.
`c+-` is the gap between the least and most correct repeat; `all` is the single combined
question.

### 60,000-token window (8,000-token tool results)

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| truncation | $0.1417 | -14% | 8% | 84% | 43,108 | 29/53 | 56% | 48pp | 55% |
| **tool_summary_anchored** | $0.1507 | **-9%** | 10% | **90%** | 54,608 | **53/53** | **100%** | 59pp | **100%** |
| **none** | $0.1649 | — | 6% | 94% | 78,003 | 53/53 | 100% | 7pp | 100% |
| anchored | $0.1939 | +18% | 13% | 82% | 54,882 | 27/53 | 22% | 78pp | 21% |
| tool_result | $0.2115 | +28% | 9% | 85% | 63,591 | 46/53 | 85% | 63pp | 85% |

### 120,000-token window (16,000-token tool results)

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_summary_anchored | $0.2715 | -8% | 9% | 90% | 102,652 | 46/53 | 48% | 52pp | 47% |
| truncation | $0.2792 | -5% | 12% | 84% | 92,158 | 27/53 | 52% | 48pp | 51% |
| **none** | $0.2945 | — | 5% | 94% | 149,562 | 53/53 | 93% | 15pp | 92% |
| **anchored** | $0.3541 | +20% | 3% | 84% | 107,562 | **53/53** | **100%** | 48pp | **100%** |
| tool_result | $0.3758 | +28% | 6% | 85% | 118,899 | 39/53 | 67% | 41pp | 66% |

### 272,000-token window, 86% full (25,200-token tool results)

This is the model's **real input limit** — see finding 5.

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_summary_anchored | $0.4049 | **-10%** | 7% | 89% | 194,041 | 18/53 | 31% | 72pp | 30% |
| truncation | $0.4484 | -1% | 3% | 89% | 207,591 | 30/53 | 56% | 22pp | 55% |
| **none** | $0.4517 | — | 36% | 94% | 233,332 | 53/53 | 76% | 17pp | 83% |
| **anchored** | $0.4975 | +10% | 3% | 89% | 203,654 | **53/53** | **100%** | 31pp | **100%** |
| tool_result | $0.5740 | +27% | 3% | 86% | 184,147 | 39/53 | 22% | 78pp | 21% |

## 5. Findings

### Finding 1 — A model-written record degrades with the bulk it must summarise

| tool result size | facts the record preserved |
| ---: | ---: |
| 8,000 | 53 of 53 |
| 16,000 | 46 of 53 |
| 25,200 | **18 of 53** |

Monotonic, and far larger than the spread. The mechanism is not at fault — the middleware
forced the call and the forced call produced a record in every run (`REC:1, FORCED:2,
RECFORCED:1`). What degrades is the *content*: asked to extract every value from more
material, the model writes a shorter list.

This is the fundamental limit of extraction as a compaction primitive. **What survives is the
model's judgement, not a policy**, and that judgement gets worse exactly when compaction
matters most.

### Finding 2 — Proportional retention improves with scale, for the same reason reversed

`anchored` keeps a share of the ceiling for each collapsed result, so at a larger window each
result keeps more of itself: 30% at 60,000 and 44% at 272,000. Its accuracy climbs from 22%
to 100% as a result.

The arithmetic behind it is worth stating, because it also explains why the framework's
`ToolResultCompactionStrategy` loses values: head-and-tail retention keeps the first and last
f/2 of a result, so *n* values spread evenly through it — sitting 1/n apart — survive only
when f exceeds 2/n. With eight values per result that needs **more than 25% of it retained**.
A fixed retention, like the framework's 4,096 characters, becomes a rounding error as results
grow.

### Finding 3 — Not compacting still wins on the axis that matters

`none` was the verdict at 60,000 and at 272,000. The one strategy that beat it, at 60,000,
did so by 9% against a 10% spread — which the tool itself refused to certify.

The reason is unchanged from the earlier work: cached reads are 9.4x cheaper here, the
discount is strict-prefix, and compaction forfeits it. `none` holds a 94% hit rate at every
size. The best any compacting strategy managed was 90%.

**But at 272,000 the picture is more interesting than "don't compact".** The control scored
**76%** — it fits the window, but the model answers less well from a 233,000-token prompt —
while `anchored` scored **100%** from a 203,000-token one. Compaction did not just cost less
information than expected; at that size it produced a *better answer* than not compacting,
because a shorter prompt is easier to answer from.

### Finding 4 — Consistency, not accuracy, is what compaction really costs

Median accuracy for the better strategies is excellent. The spread is not:

| | control | best compacting strategy |
| --- | ---: | ---: |
| accuracy spread across repeats | 7-17pp | 31-78pp |
| cost spread | 5-6% | 3-13% |

The same configuration produced 13, 41, 48, 52, 59, 72 and 78-point spreads on different
runs. That is not sampling noise to be averaged away; it is the property. A compacted agent
answers well *on average* and unpredictably *in particular*, and an application that needs a
dependable answer should read that as the real cost.

### Finding 5 — The advertised context window is not the input limit

`gpt-5.4-mini` is documented at 400,000 tokens. Measured directly: 270,294 accepted, 275,292
refused with HTTP 400 `context_length_exceeded`, `param: "input"`. The 400,000 is
**272,000 input + 128,000 output**, and the two are independent — a large output cap does not
consume input allowance, and a small one does not buy any.

This is not a curiosity. MAF computes every compaction threshold as a fraction of
`max_context_window_tokens - max_output_tokens`. Configure the advertised 400,000 with a
2,048-token reply reservation and the budget comes out at 397,952 — 46% above what the service
accepts — so `TruncationStrategy` triggers at 318,361 and never fires. **The agent dies of a
provider error before its own compaction runs.** Set `max_context_window_tokens` to the input
limit, not the advertised window.

## 6. What to do

**Size your tool results before choosing a strategy.** Below roughly 10,000 tokens per result,
having the model record the values and dropping the originals is both cheapest and lossless.
Above that, keep a fixed proportion of each result and do not trust a summary.

**If the conversation fits your window comfortably, do not compact** — unless the prompt is
large enough that the model answers worse from it, which happened here above 230,000 tokens.

**Never let retention be a fixed size.** Make it a share of the window, or it silently becomes
a rounding error at exactly the scale where it matters.

**Configure the input limit, not the model card.** And set the request's `max_tokens`
explicitly: the compaction reservation is arithmetic, and nothing constrains the model to it
unless the request says so.

**Read the spread, not the median.** A strategy with a 78-point accuracy spread is not
"usually fine".

## 7. Limits

**One model, one route, one conversation shape.** The mechanisms generalise; the crossover
point almost certainly does not.

**Accuracy spreads are wide and did not resolve at five repeats.** Every accuracy figure above
is a median with a spread beside it, and the spread is often larger than the differences
between strategies. Cost and prompt-size figures are much tighter.

**The scoring guidance is inert for the control.** Every row is told how to read a compaction
record; `none` has no record, so the clause helps only the compacting strategies. It cannot
create facts, but the asymmetry is real.

**`tool_summary_anchored` costs an extra agent turn**, visible as 32 model calls against 29.
That is included in every cost figure above.

---

*Source data — verbatim output and exact invocations — is in [`runs/`](runs/). Full tables and
the earlier cross-model work are in [`RESULTS.md`](RESULTS.md).*
