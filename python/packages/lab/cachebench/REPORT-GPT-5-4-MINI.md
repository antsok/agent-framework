# Compaction on `gpt-5.4-mini`: report

> **SUPERSEDED, 7 September 2026.** The current report is
> [`REPORT-2026-09-07.md`](REPORT-2026-09-07.md). Two things happened to this document after it was
> written, and both are structural rather than a matter of emphasis. The adversarial review of
> 2 September found two instrument defects sitting under **every** `vs none` figure the project had
> then produced — the control ran a different conversation from every strategy row, and `cost` summed
> the workload with twelve probe re-reads of the snapshot — so *everything measured before run 32 is
> withdrawn on cost* ([`REVIEW-2026-09-02.md`](REVIEW-2026-09-02.md), [`STATE.md`](STATE.md) §3e), and
> every cell here predates run 32. The output reservation was then repaired on 6 September, so no cell
> here sits on the same instrument as runs 41 and 42 either. Read it for the mechanism, the window
> series and the design reasoning; do not read it for cost, and do not put its cells in a table with
> the current ones. Kept unedited so the corrections can be checked against what it said.

**Date:** 1 September 2026
**Model:** `gpt-5.4-mini`, Azure Foundry (Responses API). The current sweep ran on a second
deployment of the same model, `gpt-5.4-mini-2`; everything before it on the first.
**Agent:** the MAF harness — `create_harness_agent`, the wiring a typical caller gets
**Framework:** upstream, unmodified, on both sides of one change. Sections 4.1 and 5 measure it
at our merge-base `3dbaaea3e`, before
[#7912](https://github.com/microsoft/agent-framework/pull/7912); section 4.2 measures it
rebased onto upstream `main` at `e2f7db207`, after
**Scale:** six cells of 30 seed records under the current instrument — four before #7912 and
two after it — plus roughly forty matrices under earlier ones; about EUR 175 of model spend
through the before arm, EUR 31 of that on its four cells, and EUR 22 on the two after it

This is a single-model deep dive and it stands apart from the six-model work in
[`REPORT.md`](REPORT.md), which used a different agent, a replay harness, 17 planted facts
instead of 53, and a control too unstable to rank against. **Nothing here should be averaged
with it.**

> **Which measurements are before #7912, and which are after.** Upstream #7912 rewrote the
> tool-eviction phase of `ContextWindowCompactionStrategy` — the strategy `create_harness_agent`
> installs — on 2026-08-31. Sections 4.1 and 5 were measured before it and are kept as the
> before arm; section 4.2 is the after arm. **One cell has been measured on both sides:
> 120,000 tokens at 0.86 fill.** Everything else in sections 4.1 and 5 — findings 2 through 9
> included — is a before-arm measurement that has not been re-taken since the rebase, so read
> it as a statement about the framework as it was. Where the after arm says something about
> those findings, they say so themselves.

> **Which sections are historical.** Sections 4.1, 4.2 and 5 report the current instrument: the
> conversation is seeded to a share of the window, snapshotted, and every closing question is
> asked from that snapshot. Section 4.3 keeps the earlier window series, which is the only
> measurement this project has at 272,000 tokens and at large tool results. Those tables were
> taken on the previous instrument and carry three defects: their accuracy column is the
> *median-cost* repeat rather than a mean, so each accuracy figure is one draw from a
> two-valued distribution; compaction kept running while the model was being scored; and their
> `all` column is inflated, because the combined question was asked last, after seven per-scope
> answers had re-listed most of the codes into the context it read. Read them for direction,
> not for magnitude.

---

## 1. The headline

**Compaction here buys the ability to continue past the context window. It does not buy a
smaller bill. The setting the framework installs by default used to buy neither — upstream
#7912 removed its cost penalty and left most of its recall penalty standing.**

Four results carry that.

- **The shipped default no longer costs what it did, and still loses half the facts.** Measured
  after #7912 at 120,000 tokens and 0.86 fill, `context_window` reads **+14%** against not
  compacting at a **91%** cache hit rate, keeping **28 of 53** facts for **54%** `acc1` — 57%
  of the control's. On cost that row is no longer distinguishable from the control: +14% sits
  inside the cell's noise, where the control's own seed spread is 15% and `context_window`'s is
  26% (finding 8). On recall it plainly is distinguishable, and the gap is most of what it was.
- **The before arm is the evidence for what the fix was worth, not a description of the
  framework today.** Before #7912 the same cell read **+141%** against not compacting at a 57%
  hit rate, with 21 of 53 facts and 38% `acc1`; across the three 120,000-token fills the cost
  ran **+27%, +113%, +141%** while the hit rate fell **88% → 66% → 57%** and accuracy fell with
  it, 72% → 43% → 38% against a control holding 97-98%. The mechanism behind that series was
  that the strategy compacted to a fixed share of the window whatever the fill. **Only the 0.86
  cell has been re-measured, and it moved**; nothing here says what the other two fills do now.
- **Nothing was measurably cheaper than not compacting with its answers intact**, on either
  arm. Every row reading below the control does so by 1 or 2 points against a control whose own
  cost spread within a cell is 9 to 21%, and the rows that are genuinely cheaper are cheaper
  because they threw the conversation away. Both after-arm cells return the same verdict as the
  four before them: `none`.
- **A strategy's dial has to be sized against the payload, not the window.** `anchored`
  allows each collapsed tool result `max_input_tokens × band_share ÷ tool groups in band`.
  With 3,500-token results that allowance is about 2,900 tokens at a 60,000-token window and
  about 5,900 at 120,000 — larger than the results — so at 120,000 it plans nothing at all.
  It is not performing badly there; it is not performing.

The earlier window series adds the shape result, which the current sweep does not test because
it holds the payload fixed: **a model-written record degrades with the bulk it must
summarise**, and both the length of a result and the number of values buried in it matter
independently.

| bearing results | codes each | per result | facts the record preserved |
| ---: | ---: | ---: | ---: |
| 6 | 8 | 25,200 | 18 of 53 |
| 6 | 8 | **8,000** | **36 of 53** |
| 16 | **3** | 8,000 | **53 of 53** |

*Earlier work — runs 18, 19 and 20, one sample per configuration. Shrinking each result from
25,200 to 8,000 tokens with the code count held at eight lifts recall from 18 to 36; dropping
the values per result from eight to three lifts it from 36 to 53. Neither variable alone
accounts for the collapse, and total tool output is not the driver — the last two rows carry a
comparable total to the first and score very differently. The third row moved three variables
at once and is kept only as the first point of the series.*

The before arm reached the same failure at a fixed payload by filling the window instead: at
120,000 tokens and 0.86 fill the record fell to 47 of 53, at about 96,000 tokens of context.
**That failure did not reproduce after the rebase** — the same cell reads 53 of 53 with no seed
spread in run 29 (section 4.2). The payload series above is untouched by the rebase; the
fixed-payload point is not, and finding 2 says what is and is not known about why.

## 2. What was measured

### The instrument: seed, snapshot, probe

A real agent is driven through a scripted conversation — requirements, one mid-conversation
correction, six tool lookups returning eight 6-hex-digit codes each, and filler turns sized to
land the conversation on a target share of the window. **53 verifiable facts** in all. The
conversation is then **snapshotted**, and every closing question is asked from the restored
snapshot rather than appended to the conversation: seven scoped questions, one on the
requirements and one per tool lookup (`acc1`), and one question asking for all 53 values at
once (`acc2`).

The snapshot matters because of what the old design measured instead. Closing questions used to
be ordinary turns. Each answer listed codes, which became assistant text in the history, so the
next question was asked from a prompt into which earlier answers had already written the codes
back — and survival was scored against the *last* of those prompts. The same strategy read
53/53 on a run that emitted 10,941 output tokens and 18/53 on one that emitted 4,873. Three
more defects followed from the same cause: the first scope was answered from a fuller context
than the last, the combined question from the most compacted context of the run, and compaction
kept firing during scoring, so a fact could be evicted while it was being scored. Probing from
a snapshot removes all four, and survival is scored against that snapshot and nothing else.

One residual is measured rather than assumed. Restoring stops compaction *accumulating* across
probes; it does not stop a strategy acting once more on the restored state. `DRIFT:<n>` counts
probes whose prompt was not the snapshot verbatim, and a row carrying it overstates what
reached the model. It is zero on every row of both arms except `context_window` at the two
lower 120,000 fills of the before arm, which carry `DRIFT:5`.

### The freeze question, settled in the negative

Before the rebuild, the obvious suspect for the wide accuracy spreads was compaction running
through the closing turns. A `--freeze-during-answers` flag was built and validated (run 21),
and both arms were run at 272,000 tokens with three repeats each (run 22). **Freezing narrowed
nothing.**

| strategy | correctness spread, unfrozen | frozen |
| --- | ---: | ---: |
| `none` | 26pp | 19pp |
| `truncation` | 59pp | 59pp |
| `anchored` | 13pp | **37pp** |
| `tool_summary_anchored` | 0pp | **13pp** |

Two rows widened, one held, and the control moved less than its own repeat range. The confound
was real — sixteen compaction calls per strategy were suppressed, so compaction had certainly
been firing during scoring — but it was not what made the accuracy figure jump. What run 22
found instead was that a single sample of a configuration says very little about its accuracy,
and run 23 then measured the within- and across-invocation spreads at 60,000 and found them the
same size.

The instrument was rebuilt around that. Probing from a restored snapshot makes mid-scoring
compaction structurally impossible, so the flag had nothing left to switch and was deleted. Its
evidence survives in `runs/run-21-*` and `runs/run-22-*`, which no longer run because they pass
a flag that no longer exists.

### Scoring

Exact substring matching on 6-hex-digit codes. No grader model, no partial credit. Each reply
is scored **only against the values its own question asked for**, so a code listed under the
wrong tool does not count: that distinguishes preserving a value from preserving the labelling
that says where it came from.

**Every figure is a mean over every sample**, not a representative run. Accuracy arrives as a
distribution, and the spreads are separate columns: `seed+-` between seeds, which is
compaction's own reliability — whether it cleared a retention boundary this time and not last
time — and `rep+-` within one seed across probe repeats, which is the model's willingness to
enumerate and nothing else. At `--probe-repeats 1`, which every current cell uses, `rep+-` is
0pp by construction; `rep2+-` is the same within-seed spread for `acc2`, and is the only
within-seed variance these cells measure.

**Every run is checked against its control before anything else is read.** The uncompacted row
must show 53/53 facts, 0 unfetched, an empty flags column and no disqualification. A
disqualified seed — one that sent a prompt larger than the tried window, which is simulated and
so has to be enforced in our own code — excludes its cell from the ranking rather than being
starred. There are none in either arm of the current sweep, and no errors and no throttling
either.

## 3. The strategies compared

| | mechanism |
| --- | --- |
| `none` | the uncompacted control |
| `context_window` | the framework's `ContextWindowCompactionStrategy` at its shipped 0.5/0.8 thresholds, keeping the framework's default of four retained tool-call groups. **This is what `create_harness_agent` installs** |
| `truncation` | oldest-first eviction triggering at 80% of the input budget, compacting to 50% |
| `anchored` | keeps a fixed head and tail verbatim, shortens the tool results in the band between them to a share of the ceiling, decisions taken from position alone so they never change on a later turn (`compaction/_anchored.py`) |
| `anchored_min_gain` | `anchored` with a floor: it prices the collapse before mutating anything and declines any whose projected saving is under 23% of the included prompt (`compaction/_anchored.py`) |
| `tool_summary_anchored` | a middleware forces one recall tool call; the strategy then drops every tool group in front of the resulting record (`compaction/_toolsummary.py`) |

The last three were written for this work, against what the earlier runs measured;
`compaction/STRATEGIES.md` documents them in full. `tool_result` — the framework's tool-result
collapse, head-truncating at 4,096 characters — appears only in the earlier tables of
section 4.3.

## 4. Results

### 4.1 The before arm — four cells, five seeds each

Runs 26 and 27, on the rebuilt instrument and on the framework at our merge-base, before
upstream #7912. Five seeds per cell, one probe repeat, and the combined question asked three
times per seed at 60,000 and five times per seed at each 120,000 cell. Payload fixed at 24,497
tokens, of which 21,967 is tool results: six results of 3,500 tokens carrying eight codes each. Pinned tool calls, neutral narration, values spread on
labelled lines. 30 seed records per cell, 120 in all; no errors, no throttling, no
disqualifications, and every row gathered every fact the control did.

Columns: `cost` is the whole run, seeding plus every probe. `in$` is the prompt side of that
same money, output and summarizer excluded — worth reading beside it, because output is priced
57 times a cache read here and a reply the model ran long on moves the total further than
compaction does. `+-` is the spread between the cheapest and dearest seed, so a gap smaller
than it is not a result. `snap%` is the snapshot every question was asked from, as a share of
the window: the control sits at the fill the cell was sized to, and a strategy below it removed
that difference. Prices were passed as EUR per million (0.66 input, 0.07 cached, 3.96 output);
the tool prints a `$` regardless, so every money figure below is EUR.

#### 60,000-token window, 86% full — fill landed at +0.0%

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **none** | $0.1642 | $0.1368 | 17% | — | **94%** | 86% | **53/53** | **100%** | 0pp | 95% |
| tool_summary_anchored | $0.1812 | $0.1422 | 25% | +10% | 91% | 68% | **53/53** | **100%** | 0pp | **100%** |
| anchored_min_gain | $0.1834 | $0.1498 | 58% | +12% | 93% | 90% | 51/53 | 96% | 20pp | 72% |
| anchored | $0.1929 | $0.1576 | 38% | +17% | 92% | 89% | 51/53 | 94% | 15pp | 87% |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| truncation | $0.1680 | $0.1414 | 21% | +2% | 89% | 60% | 30/53 | 54% | 11pp | 39% |
| context_window | $0.2598 | $0.2368 | 6% | **+58%** | **65%** | 45% | 21/53 | 41% | 0pp | 35% |

Verdict `none`. `anchored_min_gain` declined 28 and 50 collapses on the two seeds that reported
them (`NOGAIN`); this is the only cell in the sweep where the floor was ever consulted.

#### 120,000-token window, 50% full — fill landed at +1.9%

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| truncation | $0.2055 | $0.1709 | 12% | -1% | 95% | 52% | 53/53 | 98% | 6pp | 52% |
| anchored | $0.2065 | $0.1724 | 5% | -1% | 95% | 52% | 53/53 | 96% | 7pp | 56% |
| **none** | $0.2076 | $0.1682 | 21% | — | 95% | 51% | 53/53 | 97% | 13pp | 87% |
| anchored_min_gain | $0.2141 | $0.1790 | 22% | +3% | 94% | 52% | 53/53 | 99% | 6pp | 81% |
| tool_summary_anchored | $0.2613 | $0.2004 | 24% | +26% | 94% | 48% | 53/53 | **100%** | 0pp | **100%** |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| context_window | $0.2643 | $0.2342 | 21% | +27% | 88% | 46% | 41/53 | 72% | 15pp | 50% |

The tool named `truncation` and then withdrew it — `NOT SUPPORTED`, because one strategy's
seeds varied by 21% against the 1% gap the recommendation rests on. `context_window` carries
`DRIFT:5`.

#### 120,000-token window, 70% full — fill landed at -1.6%

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchored_min_gain | $0.3110 | $0.2713 | 10% | -2% | 96% | 72% | 53/53 | 96% | 13pp | 52% |
| **none** | $0.3171 | $0.2758 | 19% | — | 95% | 69% | 53/53 | 97% | 7pp | 46% |
| truncation | $0.3185 | $0.2767 | 8% | +0% | 96% | 72% | 53/53 | 98% | 6pp | 68% |
| anchored | $0.3248 | $0.2856 | 9% | +2% | 95% | 71% | 53/53 | 98% | 9pp | 68% |
| tool_summary_anchored | $0.3809 | $0.3111 | 23% | +20% | 94% | 61% | 53/53 | **100%** | 0pp | **100%** |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| context_window | $0.6749 | $0.6375 | 26% | **+113%** | **66%** | 47% | 23/53 | 43% | 6pp | 26% |

Withdrawn the same way: a 2% gap against a 19% spread. `context_window` carries `DRIFT:5`.

#### 120,000-token window, 86% full — fill landed at -2.3%

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **none** | $0.4069 | $0.3543 | 9% | — | **97%** | 84% | 53/53 | 98% | 7pp | 62% |
| anchored_min_gain | $0.4207 | $0.3701 | 7% | +3% | 97% | 89% | 53/53 | 94% | 24pp | 68% |
| anchored | $0.4333 | $0.3758 | 16% | +6% | 96% | 89% | **53/53** | **99%** | 6pp | **100%** |
| tool_summary_anchored | $0.4953 | $0.4233 | 22% | +22% | 94% | 80% | 47/53 | 90% | **52pp** | 89% |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| truncation | $0.4032 | $0.3530 | 5% | -1% | 94% | 60% | 23/53 | 42% | 7pp | 34% |
| context_window | $0.9821 | $0.9331 | 24% | **+141%** | **57%** | 47% | 21/53 | 38% | 4pp | 31% |

Verdict `none`. This is the cell where `anchored` is the best faithful row in the sweep —
everything preserved, both accuracy measures at or above the control, six percent dearer — and
the cell where the model-written record first frays. **It is also the one cell re-run after
#7912**; section 4.2 pairs the two.

### 4.2 The after arm — upstream #7912, runs 28 and 29

The branch was rebased onto upstream `main`, which carries
[#7912](https://github.com/microsoft/agent-framework/pull/7912). That PR rewrote the
tool-eviction phase of `ContextWindowCompactionStrategy`: it was a `TokenBudgetComposedStrategy`
called unconditionally, whose built-in fallback evicts whole groups oldest-first, and it is now
a `ToolResultCompactionStrategy` called only when the prompt exceeds the eviction threshold. It
also protects the first user group, and it makes compaction results persist across the
chat-middleware boundary instead of being dropped there.

Same lab code, same flags, same five seeds per cell. Ninety upstream commits separate the arms,
of which one touches compaction, so this is "upstream main before and after #7912" rather than
an isolated bisect of that PR.

#### 120,000-token window, 86% full, after #7912 — run 29, fill landed at -2.8%

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **none** | $0.4029 | $0.3574 | 15% | — | 96% | 84% | 53/53 | 94% | 20pp | 65% |
| anchored | $0.4195 | $0.3730 | 14% | +4% | 96% | 88% | 53/53 | 95% | 13pp | 59% |
| anchored_min_gain | $0.4280 | $0.3726 | 16% | +6% | 97% | 89% | 53/53 | 98% | 7pp | 84% |
| tool_summary_anchored | $0.4681 | $0.3990 | 15% | +16% | 95% | 79% | **53/53** | **100%** | **0pp** | **100%** |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| truncation | $0.3961 | $0.3541 | 5% | -2% | 94% | 58% | 22/53 | 41% | 0pp | 33% |
| context_window | $0.4579 | $0.4099 | 26% | **+14%** | **91%** | 57% | 28/53 | 54% | **59pp** | 48% |

Verdict `none`.

#### The same cell, before and after

| 120,000 / 0.86, `context_window` | before (run 27) | after (run 29) |
| --- | ---: | ---: |
| `snap%` | 47% | 57% |
| cache hit rate | **57%** | **91%** |
| cost | $0.9821 | $0.4579 |
| vs none | **+141%** | **+14%** |
| facts | 21/53 | 28/53 |
| `acc1` | 38% | 54% |
| *the control beside it* | *$0.4069* | *$0.4029* |

**The control moved 1% on cost**, which is what makes the rest of the column readable: the
deployment, the prices and the instrument did not shift underneath the comparison. On accuracy
the control moved too — `acc1` 98% to 94%, against its own 20-point seed spread in the after
arm — so the accuracy rows of this table are being read against a baseline that is itself
noisier than the difference it is measuring in the faithful strategies.

**`snap%` is the whole mechanism.** The eviction phase now fires only above its threshold and
no longer sheds whole groups, so the strategy discards less — 57% of the window left standing
where it left 47%. Less discarded is less prefix rewritten, which is the cache recovery, and it
is also why more facts survive. **This is not a strategy that got smarter; it is one that got
less aggressive**, and both columns follow from that.

Read the accuracy half as direction only. `context_window`'s `acc1` seed spread in the after
arm is **59 points** — the widest in the cell — so 54% is a mean over four seeds near 41% and
one at 100%, not a level the strategy holds.

#### 100,000-token window, 86% full, after #7912 — run 28, fill landed at -2.1%

There is no before-arm cell at this window, so this table ranks strategies against each other
and against its own control; it cannot say what #7912 changed.

| strategy | cost | in$ | +- | vs none | hit% | snap% | facts | acc1 | seed+- | acc2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **none** | $0.3199 | $0.2824 | 20% | — | 95% | 84% | 53/53 | 98% | 6pp | 52% |
| anchored | $0.3620 | $0.3070 | 48% | +13% | 95% | 90% | 53/53 | 97% | 9pp | 68% |
| anchored_min_gain | $0.3802 | $0.3178 | 56% | +19% | 95% | 91% | 53/53 | 96% | 15pp | 84% |
| tool_summary_anchored | $0.4771 | $0.3731 | 117% | +49% | 91% | 82% | **53/53** | **100%** | **0pp** | 99% |
| *— below 90% of the control's `acc1` —* | | | | | | | | | | |
| truncation | $0.3218 | $0.2742 | 25% | +1% | 93% | 61% | 21/53 | 41% | 11pp | 30% |
| context_window | $0.3632 | $0.3193 | 23% | **+14%** | **90%** | 57% | 21/53 | 40% | 6pp | 29% |

Verdict `none`. On cost and cache this agrees with run 29 exactly — `context_window` at +14%
with a 90% hit rate and the same 57% `snap%`. **On recall it does not**: 21 of 53 facts and 40%
`acc1`, which is where the before arm sat at 120,000. The plain reading is that the same
`snap%` is a smaller snapshot at a smaller window — 57,327 tokens left standing here against
68,256 in run 29, against a payload fixed at 24,497 — so the fixed share that is now generous
enough at 120,000 is not at 100,000. That is a reading of one cell with no counterpart, not a
measurement of the fill-versus-window boundary, and it is worth stating that the two after-arm
cells disagree on whether recall improved at all.

This cell also has the sweep's widest cost spreads: 48%, 56% and 117% on the three faithful
compacting rows, against the control's 20%. Nothing in its cost ordering is resolvable
(finding 8), including `tool_summary_anchored`'s +49%.

#### What else the rebase moved

`tool_summary_anchored` at 120,000/0.86 went from 47/53 facts, `acc1` 90% and a 52-point seed
spread to **53/53, 100%, and no spread**, at +16% rather than +22%. Its snapshot barely moved,
96,134 tokens before and 95,159 after, so it is reading about as much as it was: whatever
changed, it is not that the material got smaller.

**This is recorded as unexplained rather than attributed.** #7912 also made compaction results
persist across the chat-middleware boundary, which reaches every compacting row and is a
plausible cause, and nothing here isolates it: five seeds, ninety commits, one arm each side.
It is one cell, and finding 2's payload evidence is untouched by it.

`truncation`, `anchored` and `anchored_min_gain` are unchanged between the arms within their
seed spreads.

### 4.3 Earlier work — the window series

**These three tables predate the rebuild.** They are the only measurements this project has at
272,000 tokens and with tool results larger than 3,500 tokens, which is why they are kept. Read
them with three caveats. Each cell was five repeats, but the `correct` column reports the
median-*cost* repeat, so it is a single draw from a distribution whose range is the `c+-`
column beside it. Compaction ran while the model was being scored. And the `all` column is
inflated, because that question was asked last, from a context seven previous answers had
already written most of the codes into. Tool results scale with the window in this series,
which is the variable it exists to move.

#### 60,000-token window (8,000-token tool results) — run 16

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| truncation | $0.1417 | -14% | 8% | 84% | 43,108 | 29/53 | 56% | 48pp | 55% |
| **tool_summary_anchored** | $0.1507 | **-9%** | 10% | **90%** | 54,608 | **53/53** | **100%** | 59pp | **100%** |
| **none** | $0.1649 | — | 6% | 94% | 78,003 | 53/53 | 100% | 7pp | 100% |
| anchored | $0.1939 | +18% | 13% | 82% | 54,882 | 27/53 | 22% | 78pp | 21% |
| tool_result | $0.2115 | +28% | 9% | 85% | 63,591 | 46/53 | 85% | 63pp | 85% |

#### 120,000-token window (16,000-token tool results) — run 17

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_summary_anchored | $0.2715 | -8% | 9% | 90% | 102,652 | 46/53 | 48% | 52pp | 47% |
| truncation | $0.2792 | -5% | 12% | 84% | 92,158 | 27/53 | 52% | 48pp | 51% |
| **none** | $0.2945 | — | 5% | 94% | 149,562 | 53/53 | 93% | 15pp | 92% |
| **anchored** | $0.3541 | +20% | 3% | 84% | 107,562 | **53/53** | **100%** | 48pp | **100%** |
| tool_result | $0.3758 | +28% | 6% | 85% | 118,899 | 39/53 | 67% | 41pp | 66% |

#### 272,000-token window, 86% full (25,200-token tool results) — run 18

This is the model's **real input limit** — see finding 9.

| strategy | cost | vs none | +- | hit% | peak | facts | correct | c+- | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tool_summary_anchored | $0.4049 | **-10%** | 7% | 89% | 194,041 | 18/53 | 31% | 72pp | 30% |
| truncation | $0.4484 | -1% | 3% | 89% | 207,591 | 30/53 | 56% | 22pp | 55% |
| **none** | $0.4517 | — | 36% | 94% | 233,332 | 53/53 | 76% | 17pp | 83% |
| **anchored** | $0.4975 | +10% | 3% | 89% | 203,654 | **53/53** | **100%** | 31pp | **100%** |
| tool_result | $0.5740 | +27% | 3% | 86% | 184,147 | 39/53 | 22% | 78pp | 21% |

Two things in this series did not survive replication and should not be quoted as results. Run
22 re-ran run 18's command and every row moved, in both directions, by up to 69 points —
including the control, which read 76% here and 100% there. And the control's 76% at 233,332
tokens was read at the time as evidence that a shorter prompt is easier to answer from; it is a
single median-cost repeat of a configuration that scored 100% on replication, so it does not
support that. What the series does support is the direction finding 2 states: the record thins
as the material grows.

## 5. Findings

**Finding 1 is the only one measured on both sides of #7912.** Findings 2 to 9 were taken on
the before arm and have not been re-measured since the rebase. Where the after arm happens to
bear on one — findings 2, 5, 6 and 8 — it is said in place, and in finding 2's case it
withdraws a number. The others are about the instrument, the prices, the model or strategies
written here, and nothing in the change is known to bear on them. That is not the same as
having checked.

### Finding 1 — The shipped harness default cost two and a half times not compacting, until #7912

`context_window` is what `create_harness_agent` installs. **The result below is historical**:
it describes the framework at our merge-base, and upstream #7912 has since rewritten the phase
that produced it. It is kept because it is the measurement the fix is against, and because
only one of its three rows has been re-taken.

| fill | vs none | in$ against the control's | hit% | snap% | facts | acc1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120K / 0.50 | +27% | $0.2342 / $0.1682 | 88% | 46% | 41/53 | 72% |
| 120K / 0.70 | +113% | $0.6375 / $0.2758 | 66% | 47% | 23/53 | 43% |
| 120K / 0.86 | **+141%** | $0.9331 / $0.3543 | **57%** | 47% | 21/53 | 38% |
| **120K / 0.86, after #7912** | **+14%** | **$0.4099 / $0.3574** | **91%** | **57%** | **28/53** | **54%** |

*The first three rows are runs 26 and 27, before. The last is run 29, after, and it is the only
row of the four with a counterpart on both sides. Its control is $0.3574 on `in$` against the
before arm's $0.3543 — the same cell, the same instrument, a control that moved 1% on cost.*

The mechanism was visible in `snap%`, and so is the fix. Before, the strategy held its snapshot
at 45-47% of the window at every fill, because its target was a fraction of the input budget
rather than a function of how much conversation there was: a fuller conversation meant more
discarded, and more of the surviving prefix rewritten, which is what the falling hit rate
measures. At 0.86 that cost two and a half times not compacting for 39% of the control's `acc1`,
and the input-only column earns its place there — the effect was on the prompt side, $0.9331
against $0.3543, not in the replies.

After #7912 the eviction phase runs only above its threshold and no longer sheds whole groups.
`snap%` rises to 57%, the hit rate recovers to 91%, and the cost premium falls to +14% — which
is inside the noise this sweep can resolve, the control's own seed spread in that cell being
15%. **The strategy did not get smarter, it got less aggressive**, and every other column
follows from the one that changed.

**What the fix did not do is make it safe to leave on.** It still keeps 28 of 53 facts against
the control's 53, for 57% of the control's `acc1`, and it is still the row the ranking puts
below the line. Read the accuracy figures as direction only: its seed spread there is 59 points.

**Three qualifications.** The two lower-fill before rows carry `DRIFT:5`, so five probes each
were asked from a prompt the strategy had edited again after restore, and their `facts` figures
overstate what the model saw. The after arm is one cell — 0.50 and 0.70 have not been re-run,
and nothing here says what the fixed-share behaviour does at those fills now. And run 28, at
100,000 tokens with no before-arm counterpart, matches run 29 on cost and cache while keeping
21 of 53 facts for 40% `acc1`; the recall improvement is one cell's result, not two.

### Finding 2 — A model-written record degrades with the bulk it must summarise

`tool_summary_anchored` is the strongest recall result this package has produced. In three of
the four before-arm cells it preserved every planted fact with 100% on both accuracy measures
and zero spread on either. In each of those three it also scored full marks on the combined
question, above the control every time. Both after-arm cells read 53/53 and `acc1` 100% with no
spread as well.

At 120,000 tokens and 0.86 fill, on the before arm, it breaks: 47 of 53 facts, `acc1` 90%, and
a seed spread of 52 points — four seeds at 100% and one at 48%. Its snapshot there is 96,134
tokens, which is where this failure arrived at a fixed payload.

**That break did not reproduce after the rebase.** The same cell re-run as run 29 reads 53/53,
`acc1` 100%, no spread, from a 95,159-token snapshot — the same material, the opposite result.
So the fixed-payload crossover this paragraph reports is a single before-arm cell that did not
survive re-measurement, and it should not be quoted as the point where the record fails. The
payload evidence below did not change: those runs are older still, and the rebase does not
touch them. What the after arm removes is one data point, not the series.

The earlier series found the same thing by growing the payload instead — 53/53 at 8,000-token
results, 46/53 at 16,000, 18/53 at 25,200 — and runs 19 and 20 separated the two causes: both
the length of a result and the number of values buried in it degrade the record, independently,
and total tool output is not the driver.

The mechanism is not at fault. The middleware forced the call and the forced call produced a
record in every run, `FORCED:2, REC:1, RECFORCED:1` on every row of both arms. What degrades is the
*content*: asked to extract every value from more material, the model writes a shorter list.
This is the fundamental limit of extraction as a compaction primitive — **what survives is the
model's judgement, not a policy**, and that judgement gets worse exactly when compaction
matters most.

It also costs, and the premium is its own output rather than its prompt: 17,628 output tokens
against the control's 10,417 at 120K/0.70, and 10 to 26% more in total across the four
before-arm cells. It takes extra model calls to do it — 44 against 40 at 60,000, and 50, 62 and
71 against 45, 57 and 66 at the three 120,000 fills — and those are included in every figure
above. The after arm reads +16% at 120,000/0.86 and +49% at 100,000/0.86, the second against a
117% seed spread of its own, so the premium is still there and its size at 100,000 is not
resolvable.

### Finding 3 — Proportional retention scales with the window, not with the payload

`anchored` allows each collapsed tool result

```text
keep_tokens = max_input_tokens × band_share ÷ tool groups in the band
```

with `band_share` defaulting to 0.25. That is a share of the *ceiling*, so it grows with the
window while the payload does not. With the 3,500-token results the current sweep holds fixed,
the allowance is about 2,900 tokens at 60,000 and about 5,900 at 120,000 — bigger than the
results themselves. **At 120,000 it therefore plans nothing.** Its snapshot is no smaller than
the control's at any of the three fills, and `anchored_min_gain` never reports a decline there,
because its parent proposes nothing to decline.

This is the qualification the earlier finding needs. The window series showed `anchored`
climbing from 22% to 100% correct as the window grew, and read that as proportional retention
improving with scale. But that series scaled tool results with the window, so the results were
always larger than the allowance and the mechanism was always operating. **The mechanism only
operates while results exceed the allowance**, and whether they do is a fact about the payload,
not about the window. The same configuration can be aggressive at 60,000 and idle at 120,000.

The arithmetic underneath is worth stating separately, because it is general and it also
explains why the framework's `ToolResultCompactionStrategy` loses values: head-and-tail
retention keeps the first and last f/2 of a result, so *n* values spread evenly through it —
sitting 1/n apart — survive only when f exceeds 2/n. With eight values per result that needs
**more than 25% of it retained**. A fixed retention, like the framework's 4,096 characters,
becomes a rounding error as results grow.

### Finding 4 — A small edit cannot repay the cache it invalidates

Prompt caching is strict-prefix, so an edit at position K makes the provider re-read everything
behind K once at the uncached price; the edit then saves the tokens it removed on every turn
that follows, at the cached price. With `R` the tokens removed, `B` the tokens behind the edit,
`T` the turns still to come, and `p` and `c` the uncached and cached rates, the edit repays
itself when `T·R·c > (B − R)(p − c)`, that is:

```text
R > B(p − c) / (p + T·c)
```

At these prices, about 40,000 tokens behind the edit and about twenty turns left, that is
**11,456 tokens against a 52,322-token snapshot — 21.9% of the prompt**, which is what
`anchored_min_gain`'s 0.23 default is rounded up from.

An earlier form of this wrote the first term as `(B − R)(p − c)` rather than `(B − R)p − Bc`
and gave 22.7%. It drops an `R·c`: the removed tokens are not re-sent, so they are not re-read
at the cached price either. The floor is conservative under both, so the default is unchanged. `T` is the term nobody knows at decision time,
and it divides: ten remaining turns need 35%, forty need 13%.

The measurement that made this concrete: at the 60,000/0.86 cell plain `anchored` removed
**263 tokens**, 0.5% of the snapshot, dropped its cache hit rate from 92% to 88% — 46,471
tokens re-read at full price — and cost 11% more than not compacting. That is 177 to 1 against.
Nothing was wrong with *what* it shortened; the edit was simply too small to be worth making,
and the strategy had no way to notice, because it never asked.

`anchored_min_gain` asks. In the current 60,000 cell it declined 28 and 50 collapses on the two
seeds that reported them, and came in at +12% against plain `anchored`'s +17%, with the same 51
of 53 facts. **Read that gap carefully**: those two rows have seed spreads of 58% and 38%, so
the five points between them sit far inside the noise, and this cell does not resolve the
ordering. What it does show
is that the declines happened, and that they cost nothing in facts. Where the parent is inert —
every 120,000 cell — the floor is never consulted, and the two rows are indistinguishable from
each other and from the control.

**The floor is worth having and is not worth much.** It prevents a specific waste. Nothing in
it makes a compacted run cheaper than an uncompacted one.

### Finding 5 — Not compacting still wins on cost

**No strategy in either arm is cheaper than the control with its answers intact.** The rows
reading below it — `truncation` at -1%, -1% and +0%, `anchored` at -1%, `anchored_min_gain` at
-2% — sit inside a control spread that runs 9 to 21% within a cell, and the tool refused to
certify both cells where it named one. The rows that are genuinely cheaper paid for it.
`truncation` reads 1% under the control at 120,000/0.86 while keeping 23 of 53 facts; at
60,000/0.86 it costs 2% more and keeps 30 of 53. The after arm repeats it exactly: `truncation`
is the only row under the control there too, at -2% for 22 of 53 facts, and both cells return
the verdict `none`.

The reason is unchanged, and it is not about any particular strategy. Cached reads are 9.4x
cheaper here, the discount is strict-prefix, and compaction forfeits it. The control holds
94-97% cache across all six cells. Compacting rows match that only where they are inert —
`anchored_min_gain` reads 97% at 120,000/0.86, having proposed nothing to do.

Compaction here buys the ability to continue past the window. It does not buy a smaller bill,
and at the cache discounts this class of model offers it is unlikely to.

### Finding 6 — What compaction costs is reliability across conversations

The mean accuracy of the better strategies is excellent. What moves is which conversation you
happen to be in. Per-seed `acc1`, one figure per seed:

| | seeds |
| --- | --- |
| `none` at 60K/0.86 | 100, 100, 100, 100, 100 |
| `anchored` at 60K/0.86 | 100, 85, 98, 89, 100 |
| `tool_summary_anchored` at 120K/0.86, before | 48, 100, 100, 100, 100 |
| `context_window` at 60K/0.86 | 41, 41, 41, 41, 41 |
| `context_window` at 120K/0.86, after | 48, 41, 41, 100, 41 |

*Before-arm rows except the last. The after-arm row is the same shape read the other way: four
seeds where the strategy discarded what the questions needed, one where it did not.*

The split between the two variance sources is the point. Earlier work measured `rep+-` — the
same snapshot re-asked — at 0 to 2 points, while `seed+-` ran to 30-78 points. Re-asking is
nearly perfectly repeatable; the variance is almost all in *which seed*, meaning where the
planted facts fall against a retention boundary. A strategy either clears that boundary on a
given conversation or it does not, which is why survival counts take a handful of discrete
values rather than scattering: across six invocations of one earlier configuration,
`tool_summary_anchored` preserved 53 facts or 39 and nothing between, and `anchored` preserved
27, 52 or 53.

`context_window`'s flat 60K row above is the same effect with the sign reversed: it discards the
same fraction every time, so it fails identically every time. Stability is not accuracy. After
#7912 that row stops being flat — 59 points at 120K/0.86, the widest spread in the cell —
because a strategy that keeps more sometimes keeps the part the question wanted and sometimes
does not. Its mean improved and its reliability got worse, and both are the same change.

The practical form of this: **read the seed spread beside the mean.** A strategy with a
52-point seed spread is not "usually fine"; it is fine on four conversations in five.

### Finding 7 — `acc2` is close to binary, and it is the least trustworthy column

`acc2` asks for all 53 values in one answer. It is the same run scored a second way, not a
second run, and it behaves nothing like `acc1`.

| cell | control `acc1` | control `acc2` |
| --- | ---: | ---: |
| 60K / 0.86 | 100% | 95% |
| 120K / 0.50 | 97% | 87% |
| 120K / 0.70 | 97% | **46%** |
| 120K / 0.86 | 98% | **62%** |

The per-attempt values cluster at two levels rather than spreading: on the rows that preserved
everything, almost every attempt reads about 100% or about 21%, with little in between, so a
cell mean is mostly a count of which side the attempts fell on rather than a graded score.

And it moves on a byte-identical prompt: on the uncompacted control at 60,000,
**one attempt in fifteen collapsed from 100% to 21%** from the same restored snapshot that
produced 100% on the other fourteen. It is asked three times per seed at 60,000 and five times
per seed at each 120,000 cell, and it still moves that much; the control's `rep2+-` reaches 48
points at 120K/0.86.

What it largely measures is the model's willingness to enumerate 53 labelled values in one
reply, which is a different question from whether the values survived compaction. **`acc1` is
the column to read.** `acc2` is worth keeping for the one thing `acc1` cannot show —
`tool_summary_anchored` scoring 100% on it in three cells, above the control every time, which
says a compact grouped record is easier to search than the conversation it replaced — and worth
distrusting everywhere else.

### Finding 8 — Cost differences under about 20% are not resolvable

The control's own cost spread within a cell is 17%, 21%, 19% and 9% on the before arm, and 15%
and 20% on the after arm. That is the floor on what
any cost claim in this sweep can mean, and it is driven by reply length rather than by anything
compaction does: on one clean five-seed control cell — the sequential 60,000/0.86 records in
`runs/raw/oldprompt-conc1-*` — the output ran from 4,134 to 11,704 tokens across the five
seeds, and output is priced 57 times a cache read at these rates.

That is why the tables above carry `in$`, the prompt side of the same money. The instrument's
own note on the column: on a clean five-seed control the total moved 38% while the input side
moved 13%. Rank a *mechanism* on `in$`; `cost` is still the number that gets billed, and it is
what the ranking uses.

The consequence for reading sections 4.1 and 4.2 is blunt. `context_window`'s +113% and +141%
on the before arm are results, and so is the fall to +14% after #7912, because the distance
between those two is far outside any spread here. **The +14% itself is not a result**: it sits
inside the after-arm control's own 15%, so what run 29 licenses is "no longer measurably dearer
than the control", not "14% dearer". `tool_summary_anchored`'s +10% to +26% premium is a
result, and its cause is visible in the output column; its +49% at 100,000/0.86 is not, against
a 117% spread. Everything else in those tables — every row within 20% of the control — is
unresolved on cost, and the honest statement is that those strategies cost about what not
compacting costs.

### Finding 9 — The advertised context window is not the input limit

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

**Take upstream #7912, and still do not run the shipped default where the answers matter.**
Before that change `context_window` was the worst row on both axes in every cell here and got
worse with fill — at 0.86, two and a half times the cost of not compacting for 39% of the
control's accuracy. After it, at the one cell measured on both sides, the cost premium is gone
into the noise and the cache hit rate is back to 91%. What is left is the recall: 28 of 53
facts, 57% of the control's `acc1`, on a row whose seed spread is 59 points. It is now an
affordable way to lose half the conversation rather than an expensive one, and the two lower
fills have not been re-measured at all.

**Size a strategy's dial against the payload you have, not against the window.** A retention
allowance expressed as a share of the ceiling goes inert once it exceeds your tool results, and
turns generous exactly where it should be careful. Compare a strategy's `snap%` against the
control's before believing it is doing anything at all.

**Never let retention be a fixed size either.** A fixed 4,096 characters becomes a rounding
error at exactly the scale where retention matters, and with *n* values spread through a result
you need more than 2/n of it to keep them all.

**Do not compact for cost. Compact to continue past the window.** Nothing measured here is both
cheaper than not compacting and as correct, on any model, provider or API this project has
tested.

**If you must compact and the answers matter, have the model record the values and drop the
originals — and check the size of what it is being asked to read.** That was the only mechanism
here to score above the uncompacted control on the hardest question, and it degrades once one
call has to extract eight values from each of several large results. It costs extra model calls
and materially more output. The context-size half of that warning is now weaker than it was:
the one cell where it frayed at about 96,000 tokens read 53/53 when the same cell was re-run
after the rebase, so treat the payload as the term to watch and the context size as unsettled.

**Configure the input limit, not the model card.** And set the request's `max_tokens`
explicitly: the compaction reservation is arithmetic, and nothing constrains the model to it
unless the request says so.

**Read `acc1`, and read the seed spread beside it.** A single accuracy figure for a compacting
strategy is a statement about one conversation, and the combined-question column is close to
binary — treat it as an indicator, not a measure.

## 7. What to try next

**Summarise each tool result on its own, not all of them in one call.** The record thins as
there is more to record, and the current design asks for one record covering every result in
the band. Bounding it per result bounds both quantities it degrades with — the text one call
must read, and the values it must extract — at the price of more calls. Every measurement in
finding 2 points at it.

Design notes for whoever builds it:

- **Summarise when a result ages into the band, not when it arrives.** Compressing on ingestion
  means the full result never reaches the model on any later turn, which is a different product
  decision from compaction. Ageing keeps the result intact while it is recent and the model is
  still working with it.
- **Cost is the obvious trade.** One forced call per result instead of one per conversation, so
  a six-tool run pays six extra agent turns. Against that, each call is small and reads a
  cached prefix, and the current single-call design already costs extra calls — 44 against 40
  at 60,000, 71 against 66 at 120,000/0.86.
- **Cache behaviour should be no worse and may be better.** Each record is created once,
  frozen, and covers one group, so mutations still march forward and never revisit. Records
  accumulate rather than being rewritten, which is the property that gave this family its
  90-94% hit rates.
- **It gives the fallback something better to do.** With per-result records, a thin or missing
  record loses that result rather than everything the band held.

**Sweep the payload, not just the fill.** `--tool-result-tokens` is fixed at 3,500 in every
current cell, which is why `anchored` has nothing to do at 120,000 and why this sweep says
nothing about the shape result the earlier series found. Varying it across runs — at a fixed
window and fill, against the same strategies — is the one parameter that would make `anchored`
testable at 120,000 and above, and it would put the record's degradation on a controlled axis
rather than one confounded with the window. It is also the cheapest missing measurement: the
payload is the term the fill solver holds fixed, so a payload sweep moves one number per run.

**Finish the after arm of upstream #7912.** One cell of four has been re-run — 120,000 at 0.86
— plus a 100,000 cell with no counterpart. The two lower 120,000 fills and the 60,000 cell are
still before-arm only, and they are where the pre-#7912 series had its shape: the fixed-share
behaviour was worst at the highest fill, and it is the highest fill that has been fixed. The
0.50 and 0.70 cells are the cheap ones and they would say whether the improvement is uniform
or whether the strategy simply stopped acting where it used to act hardest.

**Isolate what moved `tool_summary_anchored`.** It went from 47/53 to 53/53 across the rebase,
which is a strategy of ours changing under a framework change we did not aim at it. #7912's
middleware-boundary persistence is the obvious candidate, and one cell either side of ninety
commits does not test it. The cheap version is to re-run that one cell with the boundary
behaviour forced back, offline if the shape can be reproduced on the stub.

## 8. Limits

**One model, one route, one conversation shape.** The mechanisms generalise; the numbers do
not. The current cells also ran on a second deployment of the model, on a different
subscription and SKU from every earlier run. Nothing suggests the deployment matters, and
nothing here rules it out either.

**The payload is fixed at 3,500-token results in every current cell.** Result size is the
variable the earlier series moved and this one does not, so the two answer different questions,
and sections 4.1 and 4.2 do not supersede section 4.3 on shape.

**Only one window has more than one fill.** The before arm is 60,000 at 0.86 and 120,000 at
three fills; the after arm is 100,000 and 120,000, both at 0.86. There is no measurement at
272,000 under the current instrument, and the fill axis is untested at 60,000 and at 100,000.

**The after arm is one cell pair.** Only 120,000/0.86 exists on both sides of #7912. Everything
else in this report is a before-arm measurement, including all of section 4.1 and findings 2
to 9, and the two arms are separated by ninety upstream commits rather than by that one PR. A
change of this size in one cell is a strong result; it is not a bisect, and where the after arm
disagrees with itself — run 28 keeping 21 of 53 facts where run 29 keeps 28 — the report says so
rather than choosing.

**Cost differences under about 20% are not resolvable** — see finding 8 — and neither are
accuracy differences of a few points at five seeds.

**`acc2` is close to binary and unstable** — see finding 7.

**`rep+-` is 0pp by construction** in these cells, because `--probe-repeats 1`. The within-seed
variance they measure is `rep2+-`, on the combined question only. That trade was deliberate:
re-asking one snapshot was measured at 0-2 points of spread while seeds ran to 30-78, so seeds
buy more than repeats do at the same price.

**The output reservation is not the output cap.** Strategies size their input budget as
`--context-window` minus `--max-output-tokens` (2,048 in these runs), while the value actually
sent on the request is `--answer-max-tokens` (12,000). So every threshold is a fraction of a
budget overstated by roughly 10,000 tokens. It does not distort the results here — this model's
input and output ceilings are independent, and disqualification is checked on the prompt alone
— but the thresholds are not exactly where the flag values suggest. Fixing it needs a clean
re-baseline of every cell, so it waits for a model boundary.

**`context_window` at the two lower 120,000 fills carries `DRIFT:5`**, so five of its probes
per cell were asked from a prompt the strategy had edited again after restore. Those rows
overstate what reached the model. Both are before-arm cells; neither after-arm cell carries the
flag on any row.

**The scoring guidance is inert for the control.** Every row is told how to read a compaction
record; `none` has no record, so the clause helps only the compacting strategies. It cannot
create facts, but the asymmetry is real.

---

*Source data — verbatim output, per-seed records and exact invocations — is in
[`runs/`](runs/); `cachebench_live --from-jsonl <file>` rebuilds any current table from the
records beside it. Full tables and the earlier cross-model work are in
[`RESULTS.md`](RESULTS.md); the strategies written here are documented in
[`agent_framework_lab_cachebench/compaction/STRATEGIES.md`](agent_framework_lab_cachebench/compaction/STRATEGIES.md).*
