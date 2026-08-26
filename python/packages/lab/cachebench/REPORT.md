# Does context compaction save money? — final report

**Date:** 27 August 2026
**Models:** `gpt-5.6-luna`, `gpt-5.4-mini`, `z-ai/glm-5.2`, `z-ai/glm-5.3-flash`, (`gemini-3.7-flash` excluded)
**Routes:** Azure Foundry (Responses API), OpenRouter (Chat Completions)
**Scale:** 6 model/route combinations x up to 14 settings x 3 repeats, ~290 conversations

---

## 1. The question

When an agent holds a long conversation, the whole history is sent to the model on every
turn, and that gets expensive. **Compaction** is the standard fix: throw away or shrink older
parts of the conversation so each request is smaller.

Two things needed measuring, and normally only the first gets asked:

1. Does it actually save money?
2. What does the agent forget when you do it?

## 2. What was done

A real agent was driven through the same 16-turn conversation about 290 times, changing only
the compaction setting.

During the conversation the agent is told **17 specific facts** it must repeat at the end.
Some are stated by the user, one is a mid-conversation correction ("we are *not* using the
batch pipeline, switch to streaming"), and 12 come from a tool the agent must call. Each fact
is a unique code, so scoring is exact string matching rather than a judgement call.

The final question can only be answered correctly by using all 17. Then the bill is added up.

Two things make the comparison fair, and both were added only after early runs proved
misleading without them:

- **Tool calls are pinned** where the route allows it. Each tool turn forces its own
  dedicated tool; other turns are closed to tools. Without this, models gathered different
  numbers of facts from run to run, moving both axes for reasons unrelated to compaction.
  Both glm models reject forcing on every backend, so their runs are unpinned throughout —
  comparable with each other, not with the pinned runs.
- **Every setting is run 3 times** and the median is reported, with the spread between
  repeats shown. A difference smaller than the spread is not a result.

## 3. Context settings and usage

| Setting | Value |
| --- | --- |
| Context window (the target compaction aims at) | 32,000 tokens |
| Reserved for the reply | 2,048 |
| **Working budget for history** | **29,952** |
| Conversation material | ~36,000 tokens (tool output ~49% of it) |
| Temperature | 0 |

**32,000 is a target we set, not the model's limit.** The real models accept far more — one
uncompacted prompt reached 63,933 tokens without complaint. A smaller target was chosen on
purpose: at the models' real limits nothing would ever trigger and there would be nothing to
measure.

How much context each setting actually used (from the `gpt-5.6-luna` Foundry run):

| Setting | Largest prompt | Final prompt | Fits the 29,952 budget? |
| --- | ---: | ---: | --- |
| `none` | 40,886 | 40,886 | **No — 1.4x over** |
| `tool_result` | 33,159 | 33,159 | **No — 1.1x over** |
| `selective_tool_call` | 33,083 | 33,083 | **No — 1.1x over** |
| `truncation` | 19,960 | 15,074 | Yes |
| `context_window` | 11,009 | 8,609 | Yes |
| `token_budget_*` | ~11,000 | ~8,600 | Yes |
| `context_window_aggressive` | 6,879 | 6,565 | Yes |

This split matters and is explained in finding 4.

## 4. Results

Cost is per 1,000 conversations, in each model's own currency. "Kept" is facts surviving out
of 17. Full tables are in `RESULTS.md`.

### Summary across models

| Model | Route | Cache discount | Uncompacted hit rate | Cheapest setting | Its accuracy | Best accurate setting | Its cost |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| `gpt-5.6-luna` | OpenRouter | 10x | 90% | `token_budget_truncate_first` −11% | 28% | `tool_result` | **+47%** |
| `gpt-5.6-luna` | Foundry | 10x | 91% | `token_budget_tools_first` −22% | 28% | `tool_result` | **+34%** |
| `gpt-5.4-mini` | Foundry | 9.4x | 89% | `context_window_aggressive` −13% | 28% | `tool_result` | **+63%** |
| `glm-5.2` | OpenRouter/BaseTen | 10x | 90% | `token_budget_tools_first` −22% | 6% | `tool_result` | **+24%** |
| `glm-5.2` | OpenRouter/Crusoe | **5x** | 89% | `token_budget_tools_first` −41% | 6% | `tool_result` | **+3%** |
| `glm-5.3-flash` | OpenRouter | **5x** | 90% | `truncation` **−44%** | *n/a* | `tool_result` | +23% |

The last two rows are **the same model at two cached-read prices** and are the controlled test
of finding 3.

### The representative table (`gpt-5.6-luna` on Foundry — the tightest measurement)

The control varied by only 1% across three repeats here, so these differences are the most
trustworthy in the set.

| Setting | Cost /1k | vs. none | Facts kept | Score | Fits window? |
| --- | ---: | ---: | ---: | ---: | --- |
| `token_budget_tools_first` | $16.70 | −22% | 4 of 17 | 28% | Yes |
| `token_budget_truncate_first` | $18.20 | −15% | 4 of 17 | 28% | Yes |
| **`none` (no compaction)** | **$21.50** | — | **17 of 17** | **100%** | **No** |
| `context_window_aggressive` | $22.90 | +6% | 2 of 17 | 17% | Yes |
| `truncation` | $23.30 | +8% | 7 of 17 | 33% | Yes |
| `context_window` *(shipped default)* | $26.50 | +23% | 4 of 17 | 28% | Yes |
| `tool_result` | $28.80 | +34% | 17 of 17 | 100% | No |
| `selective_tool_call` | $30.60 | +42% | 17 of 17 | 100% | No |
| `summarization` | $44.00 | +104% | 10 of 17 | 50% | Yes |

---

## 5. Findings

### Finding 1 — Forgetting is all-or-nothing

There is no gentle trade-off. Settings fall into two groups with nothing in between, and this
held in **every** run on every model:

- Settings that **delete messages** lost **10 to 15 of 17 facts**, scoring 17–39%.
- Settings that **only shrink tool output**, without deleting messages, lost **nothing**.

You cannot dial in "slightly smaller, slightly worse". The moment a setting starts deleting
messages, roughly two-thirds of what the agent was told is gone.

For an agent doing real work, 28% is not a slightly worse answer. It means the agent forgot
the correction and is confidently acting on the plan the user cancelled.

### Finding 2 — Settings that keep everything usually cost *more*

`tool_result` and `selective_tool_call` keep all 17 facts. On luna/Foundry they send **7%
fewer tokens** and cost **34–42% more**.

Sending less and paying more sounds impossible until you price the discount:

> Providers charge far less for text they have already seen — typically **10x less**. The
> discount only holds while the *start* of the conversation stays byte-identical. Change
> anything early, and everything after it is charged at full price again.

Compaction works by changing the earlier part. That is the whole idea of it. So it forfeits
the discount.

| | Tokens sent | Charged cheap | Effective cost |
| --- | ---: | ---: | ---: |
| `none` | 453,004 | 91% | baseline |
| `tool_result` | −7% | 78% | **+53%** |
| `selective_tool_call` | −7% | 78% | **+53%** |

### Finding 3 — Whether compaction pays is predictable from the price list

Compaction pays only when the volume cut exceeds the discount forfeited:

> **`T₂/T₁ < (1 − h₁(1−d)) / (1 − h₂(1−d))`**
> where `h` is the cache hit rate and `d` is the cached price as a fraction of the input price.

In plain terms — how big a cut is needed just to break even:

| Cache discount | Hit rate falls to | Must cut by |
| ---: | ---: | ---: |
| 10x | 78% | 39% |
| 10x | 65% | 56% |
| 10x | 38% | 72% |
| **5x** | **70%** | **27%** |

**This was then tested under control, not just observed across models.** OpenRouter serves
`glm-5.2` on two backends at identical input and output prices, differing only in the
cached-read rate — Crusoe at 5x, BaseTen at 10x. Same model, same weights. Halving the
discount moved every strategy against compaction and none the other way:

| | 5x | 10x |
| --- | ---: | ---: |
| `tool_result` | **+3%** | **+24%** |
| `selective_tool_call` | +2% | +22% |
| `truncation` | −35% | −19% |
| `summarization` | −11% | +22% |

**The prediction was recorded before the second run.** From the 5x run's own tokens and hit
rates, the formula projected `tool_result` at +17% and `truncation` at −24%. Measured: **+24%**
and **−19%** — both inside this model's run-to-run spread.

**The cost axis flips with the discount. The correctness axis does not.** Across both glm-5.2
runs, `none` remained the only setting keeping all 17 facts, at either discount, and
`truncation` lost 11-13 either way. A cheaper cache makes compaction affordable, not safe.

### Finding 4 — The settings that stay accurate do not fit the window

Compare the last two columns of the representative table:

- **Every setting that fits the 32,000 target scored 39% or below.**
- **Every setting scoring above 39% overflowed it**, including `none`.

So "don't compact" is only available if your real context window has room to spare. Ours did.
If you genuinely must fit 32,000, the honest choice is between settings that lose 10–15 of 17
facts, and the question stops being "which is cheapest" and becomes "which damage can I live
with".

### Finding 5 — Summarising does not rescue it

Summarising old turns instead of deleting them is the intuitive fix. It was the most
expensive option tested (**+104%** on luna/Foundry) and still lost 7 of 17 facts.

It pays twice: it destroys caching almost completely (1% hit rate — on `gpt-5.4-mini` it
reached **0%**), and it runs extra model calls to write the summaries, billed separately.

---

## 6. What to do

**If your context window has room: don't compact.** It costs more and forgets more.

**If it doesn't: treat this as damage control, not optimisation.** Pick by what you can afford
to lose, not by price. `truncation` consistently kept the most facts of the settings that fit.

**Check your model's cache discount first.** It is the single number that decides whether
compaction can pay at all. A 5x discount makes it viable; 10x usually does not.

**Consider attacking the size problem instead.** Tool output was ~49% of context here.
Returning smaller tool results, or storing them outside the conversation and fetching on
demand, reduces size without touching the history the discount depends on.

**Do not use the shipped `context_window` default expecting savings.** It cost more than not
compacting in three of four runs (+23%, +41%, and −16% in the fourth) while losing 13 of 17
facts every time.

---

## 7. Limits

**Six model/route combinations, one workload, one conversation shape.**

**All runs sat at 130–170% of the configured window**, which is the regime most favourable to
compaction. Earlier replay-mode measurements at 85–100% of a 400K window found the penalty
*worse* there (+18% to +49% for `context_window`). A usage sweep remains the main open
question.

**Costs are noisy**, though far less after tool pinning. Control spreads across three repeats
ranged from 1% (luna/Foundry) to 21% (luna/OpenRouter, unpinned). Differences smaller than the
spread are not claimed.

**One model could not be measured at all.** `gemini-3.7-flash` fails mid-conversation with
`Invalid thought signature` — its reasoning models require thought signatures to survive into
later requests, and compaction rewrites exactly that. This is a real constraint, not a harness
defect: **client-side compaction and thought-signing reasoning models are not straightforwardly
compatible.**

**One run is contaminated and reported as such.** `glm-5.3-flash` on OpenRouter had tool
forcing accepted on 9 rows and refused on 3 within a single run, because OpenRouter routes one
model across backends that disagree about tool support. Only its three consistently-unpinned
rows are used.

**On confidence.** Ten full measurement rounds were run. Six found a fault in the instrument
rather than an answer to the question — settings scoring perfectly because they had silently
done nothing, one setting's costs added to unrelated rows, tool output 8x smaller than
intended, facts the agent never gathered counted as compaction damage, and a provider whose
history is stored server-side making compaction a no-op. **None of these caused a crash or a
failing test.** Each was caught by noticing that the uncompacted control was behaving in a way
that was physically impossible. All are now covered by automated tests, and the tables above
come from rounds where every column was verified to mean what it says.
