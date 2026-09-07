# The eighteen strategies this benchmark selects between

Every name `--strategies` accepts, what it removes, when it fires, what it protects, and what
it retained the one time all eighteen were measured in a single cell.

The registry is [`_strategies.py`](agent_framework_lab_cachebench/_strategies.py). It stays in
the lab rather than in the strategy subpackage because it is benchmark configuration: its job
is to put the framework's strategies and this package's own behind one `--strategies` name
each, built from one `StrategyOptions`, so that a row is a row on equal terms.

Related documents:

- [`agent_framework_lab_cachebench/compaction/STRATEGIES.md`](agent_framework_lab_cachebench/compaction/STRATEGIES.md)
  — the three strategies written here, in depth. **Read that for `anchored`,
  `anchored_min_gain` and `tool_summary_anchored`;** this file gives only their place in the
  set. It is separate because the subpackage is meant to be lifted out whole into a repository
  of its own, and the design notes travel with the code rather than with the instrument that
  measured it. `tests/compaction/test_boundary.py` enforces the same boundary on the imports.
- [`REPORT-2026-09-07.md`](REPORT-2026-09-07.md) — what the measurements mean.
- [`README.md`](README.md) — how to select and tune these from the CLI.
- [`RESULTS.md`](RESULTS.md) — every run, with its caveats.

---

## The shared machinery

**The input budget is the window minus the output reservation.** Every threshold below is a
fraction of `--context-window` − `--max-output-tokens`, which is `StrategyOptions.input_budget_tokens`.
The reservation is real: a seeding reply is appended to the history and re-sent on every later
turn, so the budget the thresholds divide has to hold room for one. This was wrong until
6 September 2026 — the arithmetic subtracted `--max-output-tokens` while the request carried
`--answer-max-tokens`, so at a 60,000-token window with a 12,000-token answer cap the
strategies believed 57,952 tokens were available when 48,000 were. **Runs 26 to 40 sit on that
arithmetic; runs 41 and 42 do not, and the two sets are different instruments.**

**Compaction excludes, it does not delete.** The framework marks a message
`additional_properties["_excluded"] = True` and projects the included ones at send time. Two
strategies also *insert*: `ToolResultCompactionStrategy` and `SummarizationStrategy` replace
what they exclude with a new assistant message carrying trace links back to the originals. The
anchored family is the third case — it rewrites tool-result *text* in place, which removes
tokens without removing messages, and is why `msgs` alone cannot see what it did.

**A group is the unit.** `group_messages()` spans a system message, a user message, an
assistant text message, or a whole tool-call group — the assistant call, any reasoning
messages around it, and every tool result that follows. Nothing here removes half a tool-call
group.

**Preservation is a veto that outranks removal.** A message marked with `PRESERVED_KEY` is one
some other strategy has already made the sole surviving copy of something it deleted. Every
removal path in the anchored family consults `is_preserved` first. A preserved message is not
excluded, so it still counts against the ceiling in full; it may simply not be made smaller.
This exists because the fallback behind `tool_summary_anchored` is an anchored strategy, and
before the fix it shortened the very record the strategy had just paid a model call to produce.

**`tool_summary_anchored` has two lines, and both are one decision.** `--trigger-fraction`
(0.6) is where it asks the model for a record, on both halves at once: the middleware reads the
strategy's own value, so the ask and the wait cannot be configured apart. `--fallback-fraction`
(0.9) is where it stops waiting and compacts without one. The gap between them is what the
record has to arrive in, and it is a whole turn wide by construction — a middleware can only
read the history on the way *out* of a call and can only pin the *next* one.

**And when a record arrives but does not free enough, it falls back to `anchored`.** The lab
builds that fallback explicitly rather than letting the strategy take its own default, because
the default would take the *anchored* class's defaults for `band_share` and `keep_tokens`, so a
sweep moving either would move every anchored row except the one hiding inside this one. A row
that fell back is partly measuring `anchored`; `RECFALLBACK:<n>` in the flags column says so,
and it counts effects rather than attempts.

**Exclusion order versus size.** Every strategy outside the `token_budget_*` family decides
*when* to compact from its own trigger, so different strategies leave prompts of different
sizes and a comparison between them confounds "trimmed harder" with "trimmed smarter". The
`token_budget_*` variants all compact to one shared ceiling (`--budget-fraction`, default half
the input budget) and differ only in the order they delete, which holds size fixed.

---

## What they retained, all eighteen in one cell

Run 42: `gpt-5.6-luna` through the MAF harness, 60,000-token window, 0.86 fill, 3,500-token
tool results, 5 seeds, 90 records, 53 planted facts. `facts` is the mean over seeds and `acc1`
the mean share of scoped questions answered; `snap%` is the snapshot every question was asked
from, as a share of the window, and the uncompacted control sits at 81%.

**The cost column of this run is not usable** — 18 of 90 records were rate-limited, including
the control in all five seeds, which is the baseline every `vs none$` is taken against. It is
shown because the ordering is still informative, not because any figure in it is claimed. See
[`REPORT-2026-09-07.md`](REPORT-2026-09-07.md).

| strategy | `snap%` | facts | `acc1` | `hit%` | `vs none$` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 81% | 53/53 | 100% | 96% | — |
| `tool_result` | 71% | 49/53 | 93% | 89% | +33% |
| `selective_tool_call` | 66% | 50/53 | 94% | 89% | +21% |
| `context_window_lazy` | 60% | 36/53 | 69% | 88% | +29% |
| `anchored_min_gain` | 60% | 46/53 | 88% | 74% | +53% |
| `anchored` | 60% | 50/53 | 92% | 78% | +56% |
| `anchored_no_assistant` | 59% | 51/53 | 94% | 79% | +58% |
| `tool_summary_anchored` | 56% | **53/53** | **100%** | 90% | -6% |
| `context_window` | 40% | 24/53 | 46% | 81% | +1% |
| `truncation` | 39% | 20/53 | 39% | 82% | +20% |
| `token_budget_fallback` | 34% | 18/53 | 35% | 33% | +98% |
| `token_budget_truncate_first` | 32% | 19/53 | 36% | 67% | +26% |
| `token_budget_tools_first` | 32% | 17/53 | 34% | 71% | +26% |
| `token_budget_summarize` | 30% | 40/53 | 76% | 73% | +25% |
| `context_window_aggressive` | 30% | 17/53 | 34% | 71% | +3% |
| `token_budget_window_first` | 26% | 16/53 | 31% | 86% | -19% |
| `sliding_window` | 12% | 8/53 | 17% | 12% | +7% |
| `summarization` | 11% | 18/53 | 35% | 12% | +104% |

Read the table top to bottom and the two axes run in opposite directions: the strategies that
remove least keep most, and the ones that remove most are also, with two exceptions inside
their own noise, the dearest. `tool_summary_anchored` is the only compacting row that retained
everything, and it removed less than anything acting below `context_window_lazy`.

---

## Window-threshold family — `context_window`, `context_window_aggressive`, `context_window_lazy`

`ContextWindowCompactionStrategy`, the strategy `create_harness_agent` installs when it is
given `max_context_window_tokens`. The three rows are the same class at three trigger pairs,
and they exist to answer whether compacting early and often costs more in lost cache reads than
it saves in prompt tokens.

**What it removes, in two phases.** Above `tool_eviction_threshold` of the input budget it runs
`ToolResultCompactionStrategy`, which excludes whole old tool-call groups and inserts a
`[Tool results: …]` assistant summary in their place, keeping the newest four groups verbatim.
It then re-counts, and above `truncation_threshold` it runs `TruncationStrategy` down to the
*tool-eviction* line — so at the shipped thresholds it trims from above 80% of the budget to at
most 50% of it, oldest group first.

**What it preserves.** System groups, the newest four tool-call groups, and whatever the
truncation pass does not reach before it meets its target. If both phases leave the prompt over
the input budget it logs a warning and stops; it does not enforce its own ceiling.

| row | eviction / truncation | run 42 |
| --- | --- | --- |
| `context_window` | 0.5 / 0.8 — the shipped default | 24/53 facts, `acc1` 46% |
| `context_window_aggressive` | 0.3 / 0.5 | 17/53, 34% |
| `context_window_lazy` | 0.7 / 0.95 | 36/53, 69% |

The gradient is monotone and it is the whole finding about this family: the later it fires, the
more it keeps, and none of the three keeps enough. **`context_window` is the framework's own
default and it lost more than half the planted facts at this cell.** The
`keep_last_tool_call_groups` of the default row is deliberately the framework's 4 rather than a
lab value — a local override would make the row harsher than the configuration it claims to
stand for.

## Age-ordered — `truncation` and `sliding_window`

**`truncation`** is `TruncationStrategy(max_n=0.8×budget, compact_to=0.5×budget)` with a
tokenizer, so both numbers are tokens rather than message counts. Above `max_n` it excludes
whole groups oldest-first until the included count is at or under `compact_to`, skipping system
groups and always leaving at least one non-system group. Run 42: 20/53 facts, `acc1` 39%.

It is the reference row in most runs because it is the cheapest possible removal — no model
call, no rewriting, no bookkeeping — so anything more elaborate has to beat it. In run 41 it
read 21 to 45 facts across 25 rows on the same conversations where the record row read 53 in
all 25.

**`sliding_window`** is `SlidingWindowStrategy(keep_last_groups=6)` and is the only row with no
trigger at all: it runs on every invocation and keeps the last six included non-system groups
plus every system group. That is why it is here rather than with the threshold strategies — it
drops the oldest group *every turn*, so it changes the **start** of the prompt every turn, and a
strict-prefix cache is worthless to it. Run 42: 8/53 facts, `acc1` 17%, and a 12% cache hit rate
against the control's 96% — the worst of anything measured, on both axes.

`sliding_window` is the clearest single demonstration of the mechanism this project exists to
measure: it seeded 156,551 input tokens against the control's 665,487, less than a quarter, and
its prompt side still cost $0.0306 against the control's $0.0225. **Four times fewer tokens, a
third more money.**

## Tool-oriented — `tool_result` and `selective_tool_call`

Both act only on tool-call groups, both keep the newest `--keep-last-tool-groups` (default 4)
verbatim, both do nothing when the conversation holds no more than that many, and both need
`--tool-turns` above 4 or they never fire at all.

**`tool_result`** — `ToolResultCompactionStrategy`. Excludes each old tool-call group and
inserts one assistant message in its place, `[Tool results: <tool>: <result>; …]`, capped at
4,096 characters for the whole assembled summary and marked `... [truncated]` when it overruns.
So a value survives only if it falls inside that cap. Run 42: 49/53 facts, `acc1` 93%, and it
removed the least of anything that acted — `snap%` 71% against the control's 81%.

**`selective_tool_call`** — `SelectiveToolCallCompactionStrategy`. The same selection with no
replacement: the assistant call, its reasoning messages and every tool result in the group are
excluded outright. Run 42: 50/53 facts, `acc1` 94%, `snap%` 66%.

That the *lossier* mechanism scored marginally higher is inside the seed spread — both rows
carry a 30pp and 22pp `seed+-` — and is not a finding. What is a finding is that neither
removes much: with four groups of six kept verbatim there are only two to act on.

## The anchored family — `anchored`, `anchored_no_assistant`, `anchored_min_gain`

Written here, and documented in depth in
[`compaction/STRATEGIES.md`](agent_framework_lab_cachebench/compaction/STRATEGIES.md). The
short version:

**What it removes.** A fixed head (`--keep-head-groups`, 3) and a fixed tail
(`--keep-tail-groups`, 4) are never touched. The band between them has its tool results
*shortened in place* — head and tail of each result kept, middle replaced with
`... removed by compaction` — and only if shortening leaves the prompt over the ceiling does it
shed whole tool groups, and only after that assistant narration.

**When it fires.** Immediately, and it has no threshold: its ceiling is the whole input budget
rather than a fraction of it, because it does not need headroom to trip. It collapses the band
from the first turn there is one.

**What is decided how.** From *position alone*, counted from the head, so a decision made on
turn 5 is the same decision on turn 20 and the prefix is invalidated once rather than rewritten
every turn. The `n`-th result in the band keeps an `n`-th of `--band-share` (0.25) of the
ceiling; `--keep-tokens` overrides that with a flat number, which is the older behaviour and is
exposed to make the comparison runnable rather than because it is good.

| row | what separates it | run 42 |
| --- | --- | --- |
| `anchored` | the base | 50/53 facts, `acc1` 92% |
| `anchored_no_assistant` | forbidden to shed assistant narration | 51/53, 94% |
| `anchored_min_gain` | declines any collapse saving less than `--min-gain-fraction` (0.29) of the tokens *behind* it | 46/53, 88% |

The floor in `anchored_min_gain` is derived, not chosen: a strict-prefix cache makes an edit
re-bill everything behind it once at the uncached price and save the removed tokens at the
cached price on every later turn, which repays when `R > B(p−c)/(p+T·c)`. 0.29 is that at the
measured prices with twenty turns remaining. `T` is the term nobody knows at decision time and
it divides — ten remaining turns need 43% of `B` and forty need 17% — so a caller expecting
shorter conversations should raise it. Declines are counted and surface as `NOGAIN:<n>`, so
"never fired" and "fired to no effect" are distinguishable; run 42 shows `NOGAIN:14`, `15` and
`16` on three of its five seeds.

The three-way ordering of facts here — 51, 50, 46 — sits well inside the seeds' own spreads
(30pp, 26pp and 48pp on `acc1`). **The pair `anchored`/`anchored_min_gain` has still not been
compared at more than one value of `--min-gain-fraction`,** which is the only variable that
separates them and was unreachable from the CLI until recently.

## The record strategy — `tool_summary_anchored`

The only design here that carries information forward instead of discarding it, and the only
one that spends an agent turn to do so. Full account in
[`compaction/STRATEGIES.md`](agent_framework_lab_cachebench/compaction/STRATEGIES.md).

**Phase 1.** `ToolResultRecallMiddleware` watches the conversation. Past `--trigger-fraction`
of the input budget — or once `--max-groups-before-record` tool groups have accumulated since
the last record, whichever fires first — it pins the next request to a single tool,
`recall_earlier_tool_results`, whose description asks the model to write down everything from
earlier tool results that later work could depend on. The tool echoes that text straight back,
so the record becomes an ordinary tool result inside the conversation. The middleware sends no
message of its own: an appended message carries no history provider's source tag and would be
persisted into the user's conversation, so the tool's description is the entire prompt.

**Phase 2.** The strategy waits for that record, then drops the tool groups *in front of it
that the record demonstrably carries*. "Demonstrably" is `--coverage-share` (0.8) of the
group's distinctive values quoted in the record, compared as token sets. Groups that fall short
are kept and counted as `UNCOVERED:<n>`.

**What it preserves.** The head groups, the tail groups, every group the record did not cover,
and every record ever written — records are marked preserved, so neither this strategy nor the
anchored fallback behind it may shorten or drop one. Nothing merges two records, because an
older record is the sole account of the groups behind *it*.

**Retention is what it buys.** Run 42: 53/53 facts and `acc1` 100%, the only compacting row
that matched the control on both. Run 41, on the same conversations as `anchored` and
`truncation` inside single invocations: **53/53 in all 25 rows** — two models, two payloads,
repeats on and off, coverage succeeding and failing, fallback firing and not — where `anchored`
read 35 to 53 and `truncation` 21 to 45.

**What it does not buy is money.** Two run-41 arms named it at -1% and -3% against not
compacting and the instrument printed `NOT SUPPORTED` on both; three arms named `none`, and on
`gpt-5.4-mini` it cost 10% to 25% more. Its coverage check also sits on a cliff rather than a
calibrated threshold: `unc` and shrink move one-for-one and flip seed to seed within one arm.

## Summarization — `summarization`

`SummarizationStrategy`, the ladder the framework leads with, and the only registry entry
besides `token_budget_summarize` that needs `--summarizer-provider`. It fires when the
conversation holds more included non-system messages than `target_count` + `threshold`
(6 + 2 = 8 as the lab builds it), summarizes everything older than the last `--keep-last-groups`
(6) groups into one assistant message, and excludes the originals with trace links back.

Run 42: **18/53 facts and `acc1` 35% — level with blind truncation** — at 11% of the window,
the hardest compaction in the set, a 12% cache hit rate, and **+104% against not compacting**,
of which $0.0311 of a $0.0699 seed cost is the summarizer's own calls. It is simultaneously the
row that removed the most and the row that cost the most, which is the project's central result
stated by a single strategy.

It also catches its own errors, logs a warning and returns `False`, so a broken summarizer
produces a run that never compacted and therefore scores *perfect* recall. Failures are counted
and flagged `S<n>`; a row carrying that flag is not evidence that summarization preserves
anything.

## The token-budget family — five variants at one ceiling

`TokenBudgetComposedStrategy` at `--budget-fraction` (0.5) of the input budget. It returns
immediately if already within budget; otherwise it runs its parts in order, re-counting after
each and stopping as soon as the ceiling is met, then removes whatever the parts left over by
evicting oldest groups. A second, strict pass will exclude *system* groups if even that is not
enough. The shared ceiling is what makes the five comparable: any difference between them is
what each discarded, not how much.

| variant | parts, in order | run 42 |
| --- | --- | --- |
| `token_budget_fallback` | none — pure oldest-first eviction | 18/53, `acc1` 35% |
| `token_budget_tools_first` | tool results, tool-call groups, then age | 17/53, 34% |
| `token_budget_truncate_first` | age, then tool results | 19/53, 36% |
| `token_budget_window_first` | a hard six-group recency window, then age | 16/53, 31% |
| `token_budget_summarize` | tool results, then summarize the rest | 40/53, 76% |

`token_budget_fallback` is the floor the other four have to beat, and **three of them do not**:
17, 19 and 16 against 18, a four-point band on rows whose own seed spreads run 0 to 19 points.
Ordering the deletions differently, at a fixed size, changed nothing measurable about what
survived. Only the variant that *summarizes* rather than deletes separates from the floor, at
40/53 — and it pays a summarizer for it and still lands 13 facts below the control.

`token_budget_fallback` also carries the family's other oddity: a 33% cache hit rate and +98%
cost, the second-dearest row in the cell, from pure oldest-first eviction. `token_budget_window_first`
reads -19% at 16/53 facts and 31% accuracy, which is what a cheap row bought by destroying the
conversation looks like, and is why the table is ranked on correctness first.

---

## Caveats that apply to every number here

- **Run 42's cost axis is throttle-biased.** Eighteen strategies across five concurrent
  invocations ran about 2.7M tokens a minute against a 2M ceiling; the control was rate-limited
  in all five seeds and `tool_summary_anchored` in one. Throttling does not change what the
  model is sent or writes, so `facts`, `acc1` and `snap%` stand; `vs none$` does not.
- **One cell, one model.** Everything in the run-42 table is `gpt-5.6-luna` at 60,000/0.86 with
  3,500-token results, and that cell seeded 5.3% under its fill target. `gpt-5.4-mini` behaves
  differently on the record row and has never been measured across all eighteen.
- **Accuracy is a distribution.** Several rows above have `seed+-` of 20 to 60 points. A gap
  smaller than a row's own spread is not a ranking, and this file quotes means.
- **`compaction/STRATEGIES.md` is behind on two numbers.** It documents `trigger_fraction` 0.8
  and `fallback_fraction` 0.95; both were reverted to 0.6 and 0.9 on 6 September 2026, which is
  what the code, the CLI and every archived run use. Its mechanism and design reasoning are
  current; those two defaults are not.
