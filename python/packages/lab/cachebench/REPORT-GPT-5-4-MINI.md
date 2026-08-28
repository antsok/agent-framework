# Compaction on `gpt-5.4-mini`: an intermediate report

**Date:** 28 August 2026
**Model:** `gpt-5.4-mini`, Azure Foundry (Responses API), GlobalStandard deployment
**Agent:** the MAF harness — `create_harness_agent`, the wiring a typical caller gets
**Scale:** 3 completed matrices x 14 settings x 3 repeats, plus one aborted matrix and two
probes. About 180 conversations, roughly EUR 65 of model spend.

This is a single-model deep dive and it stands apart from the six-model work in
[`REPORT.md`](REPORT.md). That work ran a different agent, a 32,000-token window, 17 planted
facts and a control whose accuracy was not stable enough to rank against. **Nothing here
should be averaged with it.** What carries over is the mechanism; what is new is that the
accuracy axis can now be trusted.

---

## 1. What was measured

An agent is driven through the same 22-turn conversation, changing only the compaction
setting. During the conversation it is told **53 verifiable facts** — requirements, a
mid-conversation correction, and codes returned by six tool calls. It is then asked seven
targeted questions whose answers together need all 53. Each fact is a unique code, so scoring
is exact string matching.

Then the bill is added up, and the two axes are reported side by side: **what it cost** and
**what the agent forgot**.

Three things make the comparison fair, each added only after an earlier run proved misleading
without it:

- **Tool calls are pinned.** Each tool turn forces its own dedicated tool; other turns are
  closed to tools. Without this, models gather different facts run to run, moving both axes
  for reasons unrelated to compaction.
- **Every setting is run three times**, the median is reported, and the spread between repeats
  is printed next to it. A difference smaller than the spread is not claimed as a result.
- **Facts the agent never fetched are separated from facts compaction removed.** Without that
  split, a model that simply omits something looks identical to a strategy that deleted it.

**The control is the instrument.** In all three runs it scored **53 of 53 facts, 100%
correct, 0 unfetched**, with cost varying 0-7% between repeats. Every claim below is relative
to that.

## 2. The model's real limits, which are not the advertised ones

The model card says 400,000 tokens. Measured directly on the deployment, one prompt per size:

| prompt | result |
| ---: | --- |
| 270,294 | accepted |
| 275,292 | **HTTP 400 `context_length_exceeded`, `param: "input"`** |

**The 400,000 is a total: 272,000 input + 128,000 output.** The two are independent, not a
pool you can allocate between — 270,293 tokens of input are accepted with the output cap set
to 16, to 100,000 and to 128,000 alike, while 280,000 tokens are refused even at 16. A small
output cap buys no input headroom, and a large one costs none.

This matters because MAF computes every compaction threshold from
`max_context_window_tokens - max_output_tokens`, which models a shared pool this model does
not have. **Configure `max_context_window_tokens` as the input limit**, 272,000, whenever a
provider states the two ceilings separately.

## 3. The three configurations

Material is scaled with the window on purpose, so that "how full the window is" stays a
controlled variable rather than drifting. Holding the conversation fixed while widening the
window would leave every strategy inert, and a strategy that evicts nothing scores a perfect
result for having done nothing.

| | Run 7 | Run 8 | Run 9 |
| --- | ---: | ---: | ---: |
| Window configured | 60,000 | 120,000 | 272,000 |
| Working budget | 57,952 | 117,952 | 269,952 |
| Uncompacted peak prompt | 78,307 | 150,112 | 232,748 |
| **Fullness** | **135% — overflows** | **127% — overflows** | **86% — fits** |
| Uncompacted cache hit rate | 93% | 93% | **94%** |
| Cost per conversation, uncompacted | $0.1621 | $0.3009 | $0.4317 |
| Verdict | `none` | `none` | `none` |

Runs 7 and 8 are the regime where compaction is *forced* to act. Run 9 is the regime most
real deployments are actually in: the conversation fits, and compaction is a choice.

## 4. Results

### The headline

**No setting, in any of the three runs, was both cheaper and as accurate as not compacting.**
The verdict was `none` every time.

Run 9 is the sharpest case, because it is the most realistic. Thirteen of fourteen settings
cost *more* than not compacting at all:

| strategy | cost | vs none | facts kept | correct |
| --- | ---: | ---: | ---: | ---: |
| token_budget_window_first | $0.3483 | **-19%** | 26/53 | 50% |
| **none** | **$0.4317** | — | **53/53** | **100%** |
| truncation | $0.4365 | +1% | 37/53 | 70% |
| token_budget_truncate_first | $0.4498 | +4% | 29/53 | 11% |
| token_budget_tools_first | $0.4568 | +6% | 29/53 | 41% |
| sliding_window | $0.5322 | +23% | 0/53 | 17% |
| context_window_aggressive | $0.5500 | +27% | 23/53 | 41% |
| tool_result | $0.5632 | +30% | 53/53 | 100% |
| selective_tool_call | $0.5667 | +31% | 53/53 | 100% |
| context_window_lazy | $0.5815 | +35% | 53/53 | 100% |
| context_window *(shipped default)* | $0.5841 | +35% | 37/53 | 56% |
| token_budget_fallback | $0.5956 | +38% | 37/53 | 11% |
| token_budget_summarize | $0.7354 | +70% | 37/53 | 11% |
| summarization | $1.3286 | +208% | 53/53 | 100% |

The single setting that saved money lost 27 of 53 facts.

### Why sending less can cost more

Providers charge far less for text they have already seen — **9.4x less** on this model. The
discount is strict-prefix: it holds only while the beginning of the conversation stays
byte-identical. Compaction changes the earlier part; that is what it *is*. So it trades a
large discount for a smaller prompt.

Whether that trade pays is arithmetic:

> **`T₂/T₁ < (1 − h₁(1−d)) / (1 − h₂(1−d))`**
> `T` = tokens sent, `h` = cache hit rate, `d` = cached price ÷ input price.

**The formula predicted the cost column of both runs before it was read.** At Run 8, using
each row's own tokens and hit rate:

| strategy | predicted | measured |
| --- | ---: | ---: |
| `context_window_aggressive` | -31% | **-33%** |
| `truncation` | -7% | -11% |
| `tool_result` | +28% | +22% |
| `selective_tool_call` | +24% | **+22%** |
| `context_window_lazy` | +53% | +46% |
| `summarization` | +172% | **+165%** |

Every prediction within 7 points, and all six err the same way — the formula counts input
only, while every compacting strategy also emits fewer output tokens than `none` at 6x the
input price. The residual is the term it omits.

At a 93% -> 84% hit rate you must cut **32%** of your tokens just to break even.
`tool_result` cut 12%.

### Fullness matters more than window size

| | 60,000 (overflows) | 120,000 (overflows) | 272,000 (fits) |
| --- | ---: | ---: | ---: |
| `tool_result` vs none | +28% | +22% | +30% |
| cheapest setting | -15% | -33% | -19% |

While the conversation overflows, a bigger window makes accuracy-preserving compaction
*cheaper* — the preserved prefix is a larger share of a larger prompt. But when the
conversation **fits**, the penalty comes back, because a conversation that fits has a nearly
perfect cache to lose. At 232,748 tokens the control reached a 94% hit rate, the highest
measured anywhere in this work.

**The more of your conversation is cached, the more compacting it costs.**

### Forgetting is a cliff, and surviving is not the same as being usable

Settings that only rewrite tool output, without deleting messages, lost nothing. Settings that
delete messages lost 16 to 53 of 53 facts. There is no gentle middle.

But the count of surviving facts is a **ceiling, not a prediction**:

| Run 7 | facts surviving | present but unused | correct |
| --- | ---: | ---: | ---: |
| `truncation` | 29/53 | **24** | 11% |
| `token_budget_tools_first` | 16/53 | **16** | 2% |
| `token_budget_summarize` | 34/53 | 0 | 65% |

`truncation` left 29 codes in front of the model and the model used none of them. The codes
survive as strings while the turns that say *which deployment each belongs to* are deleted, so
a question about the staging deployment cannot be answered from a bare list of identifiers.
Summarising preserves the labelling, which is why it scores better on fewer facts.

At Run 9's scale this reached a strategy that deletes nothing at all: `selective_tool_call`
kept 53/53 at 120,000 but scored **70% with 16 unused**, because collapsing a tool-call group
whose result is 16,000 tokens removes enough surrounding context to strip the codes of their
meaning.

### The shipped default is the worst of both

`ContextWindowCompactionStrategy` at its shipped 0.5/0.8 thresholds, which is what
`create_harness_agent` installs:

| run | vs none | facts kept | correct |
| --- | ---: | ---: | ---: |
| 60,000 | +21% | 21/53 | 41% |
| 120,000 | +14% | 27/53 | 37% |
| 272,000 | +35% | 37/53 | 56% |

It costs more than not compacting in all three, while losing a third to a half of the facts.
In Run 9 it is beaten on cost by seven settings and on accuracy by four.

## 5. Configuration hazards found along the way

Each of these produces a wrong result or a dead agent, and **none of them reports anything**.

**1. The window must be the input limit.** Passing the advertised 400,000 with a 2,048-token
output reservation gives a 397,952-token budget — 46% above what the service accepts. Four of
fourteen strategies then died with HTTP 400, and `truncation` failed on the *same turn as the
uncompacted control*, which is the proof it never compacted once: its trigger sat at 318,361,
which the service will never accept. Strategies triggering at or below half the budget
survived, so the shipped default escaped by luck rather than design.

**2. The output reservation is arithmetic; the cap is a request option.**
`max_output_tokens` on a strategy only subtracts from the window. What holds the model to it
is `max_tokens` on the request, and the only thing linking them is a `setdefault` in
`create_harness_agent` — which a caller supplying their own value silently defeats. A
hand-built `Agent` carrying a `ContextWindowCompactionStrategy` gets no cap at all, and then
the reservation is fiction: the strategy leaves room for 2,048 tokens of reply and the model
is free to emit 40,000. Measured: with no cap, a prompt asking for length produced 1,444
tokens; there is no modest default.

**3. Half the strategies cannot see tokens.** `SlidingWindowStrategy`,
`ToolResultCompactionStrategy` and `SelectiveToolCallCompactionStrategy` count message groups.
Give them a window and they will happily leave a prompt over it, because they never look at
the number.

**4. A reply cap converts silently into apparent accuracy loss.** This one is about measuring,
not running, and it cost the most to find. Asking the control for all 53 codes at once:

| retrieval guidance | reply cap | correct | present but unlisted |
| --- | ---: | ---: | ---: |
| on | 900 | 100% | 0 |
| **off** | **900** | **33%** | **36** |
| off | 4,000 | **100%** | 0 |

Enumerating 53 labelled codes costs ~640 tokens before any prose, so the answer was truncated
and the scorer counted the missing tail as facts the model ignored. **Truncation is
indistinguishable from compaction damage** — facts in context, absent from the answer — unless
the control is checked at the same cap. Two earlier "fixes" turned out to be compensating for
this rather than curing anything.

## 6. What to do

**If the conversation fits your window, do not compact.** In the fitting regime, 13 of 14
settings cost more and every one of them forgot something.

**If it does not fit, treat compaction as damage control rather than optimisation.** Choose by
what you can afford to lose, not by price. `truncation` kept the most facts of the settings
that reduce size meaningfully.

**Set the window to the model's input limit**, not its advertised context size, and set the
reply cap explicitly on the request rather than trusting the reservation to do it.

**Attack the size at the source instead.** Tool output was about half of the context here.
Returning less, or storing results outside the conversation and fetching on demand, reduces
size without touching the prefix the discount depends on.

**Check the cached-read price first.** It is the single number deciding whether compaction can
pay at all. At this model's 9.4x it rarely does.

## 7. What this does not cover

**One model, one route, one workload.** The mechanism generalises; the magnitudes do not. The
six-model work found the same direction everywhere and magnitudes varying by a factor of two
across routes.

**Two rows in Run 7 cannot be ranked on cost.** `tool_result` (+28% at +-28%) and
`selective_tool_call` (+22% at +-58%) had spreads as large as their effect. The same two rows
were tight in Run 9 (+-2% and +-4%), so the instability is a property of this model at that
size rather than a permanent limitation.

**Facts sit at the head of each tool result.** `ToolResultCompactionStrategy` head-truncates
at 4,096 characters, so our markers survive that cut unconditionally. A workload whose values
sit at the *end* of a long tool result would score it worse. That variant is designed but not
yet run, and it is the most important gap.

**Nothing here measures multi-agent or long-horizon runs**, where compaction decisions
compound across many more turns than 22.

---

*Source data for every run — verbatim output and exact invocation — is in [`runs/`](runs/).
Full tables and the cross-model context are in [`RESULTS.md`](RESULTS.md).*
