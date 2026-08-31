# Working state

Where this work stands, what is unresolved, and what is needed to pick it up again. Written
to survive a break in context. Results live in [`RESULTS.md`](RESULTS.md), the analysis in
[`REPORT-GPT-5-4-MINI.md`](REPORT-GPT-5-4-MINI.md), raw output in [`runs/`](runs/).

## 1. How to run anything

```sh
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
```

Authentication is `DefaultAzureCredential` against an `az login` session. There is **no `.env`
file anywhere in the tree**; the deployment was found by listing the subscription
(`az cognitiveservices account deployment list`). The model is deployed on an AI Services
account in resource group `foundry-rg`, project `proj-maf-extensions`.

Pricing must be passed explicitly — only OpenRouter is auto-discovered:
`--price-input 0.66 --price-cached 0.07 --price-output 3.96` (EUR per million).

The scripts that produced every recorded run are in `runs/*.sh` with the endpoint genericised.
Working copies were under `/tmp/cachebench/`, which is volatile.

## 2. Hard facts about the model, measured not assumed

| | |
| --- | --- |
| Real input limit | **272,000 tokens**. 270,294 accepted, 275,292 refused (HTTP 400, `param: "input"`) |
| Output ceiling | 128,000. The advertised 400,000 is the sum, and the two are independent |
| Cached-read discount | 9.4x (0.66 vs 0.07 per million) |
| Uncompacted cache hit rate | 93-97% at every size tested |
| `tiktoken` `o200k_base` accuracy | within 6 tokens on a 270,294-token prompt |

## 3. The measurement was rebuilt: seed, snapshot, probe

**The old closing turns are gone.** They were ordinary conversation turns appended to the
seeded conversation, and four defects followed, all measured:

1. `survived` was circular. It was scored against `final_prompt`, built from the *last*
   turn's prompts, by which time earlier closing answers had re-listed codes into history as
   assistant text. The same strategy read 53/53 on a run that emitted 10,941 output tokens
   and 18/53 on one that emitted 4,873.
2. Ordering bias: the first scope was answered from a fuller context than the last, and the
   combined question from the most compacted context of the run.
3. Compaction kept firing during scoring, so a fact could be evicted while it was scored.
4. Two variance sources arrived as one number. `anchored` scored 52%, 52%, 52% and 22% on
   runs that preserved exactly 27 facts each time.

**What replaced it.** `run_live` now has three phases:

- **Seed** — every turn except the closing questions, driven exactly as before. Only the
  user-side turn list is shared between strategies; replies and therefore transcripts diverge
  from turn one, and that divergence is part of what is measured.
- **Snapshot** — `snapshot_state(session)` deep-copies `session.state`. Deep is load-bearing:
  `apply_compaction` marks exclusions by mutating `additional_properties` in place. The state
  is nested per provider (`state["in_memory"]["messages"]`), so `serialize_history` finds it
  through `agent.context_providers` rather than a fixed key -- the harness installs a
  different provider instance from the plain agent.
- **Probe** — every closing question, asked from the restored snapshot, `--probe-repeats`
  times (default 3). `restore_state` deep-copies again per probe *and* calls
  `ToolResultRecallMiddleware.forget_pending()`: a pending decision to force a recall call is
  taken on one call and applied to the next, so it is conversation state, and left in place it
  would fire on the first probe and no other.

Survival is scored against `LiveOutcome.snapshot_prompt` and nothing else.

**The one residual, measured rather than assumed.** Restoring stops compaction *accumulating*
across the probes; it does not stop the strategy running once more on the restored state. A
strategy that then evicts or rewrites something has answered from slightly less than the
snapshot it is scored against. `LiveOutcome.context_drift` counts the probes whose prompt was
not the snapshot verbatim and the table flags it as `DRIFT:<n>`. It is zero whenever the
strategy has settled by the end of seeding, which is the usual case; a row carrying the flag
overstates what reached the model.

**`--freeze-during-answers` is deleted**, with `CompactionSwitch`, `_FreezableStrategy` and
the `paused` parameter. The new design makes mid-scoring compaction structurally impossible,
so the flag was dead. Its evidence survives in `runs/run-21-*`, `runs/run-22-*` and in git
history; runs 21 to 23 in `RESULTS.md` still describe it. Those two shell scripts no longer
run -- they pass a flag that does not exist -- and are kept as the record of what was done,
not as something to re-run.

**Disqualification.** `--context-window` is the *tried limit*, and it is simulated -- the real
ceiling is 272,000 -- so `LiveOutcome.disqualified(limit)` enforces it in our own code. Any
call whose billed prompt exceeds it disqualifies that seed; `dq` reports the share of a cell's
seeds that did; a cell with `dq > 0` is excluded from the ranking rather than starred. If the
control itself is disqualified the run exits, because there is no admissible baseline at that
size -- which is the finding for the 60,000 cell, where the control ran at 78,003 tokens.

**Fill targeting.** `--fill` (default 0.70) sizes the seeded conversation to a share of the
tried limit, solved analytically in `_fill.py`: the filler count is the coarse dial and its
size the fine one, the payload is fixed, and a payload that will not fit the smallest cell is
refused with an explanatory error. The one term that cannot be computed is the model's own
replies (`ASSUMED_REPLY_TOKENS = 150`), which is why the achieved fill is measured on the
uncompacted run and flagged beyond +-5%. `--fill 0` restores manual sizing from
`--filler-turns` and `--filler-tokens`.

**Accuracy is a distribution.** `_representative()` is gone. Correctness is the mean over
every probe repeat of every seed, the per-sample values are printed below the table, and the
two spreads are separate columns: `seed+-` between seeds (compaction's reliability) and
`rep+-` within one seed across probe repeats (the model's enumeration variance). `cost` stays
one column: seeding plus every probe, summed. `out` has its own column so a total driven by
verbosity is visible.

**Verified live in run 24** (60K/0.86, five strategies): fill landed at +0.0%, probes hit 94-96%
cache, no disqualifications and no drift. The numbers in `RESULTS.md` and
`REPORT-GPT-5-4-MINI.md` all predate the rebuild, so none of them are disqualification-checked
and their `all` column is inflated -- the combined question used to be asked last, after seven
answers had re-listed the codes into the context it read.

## 3b. Stage 1 sweep — paused mid-run, how to resume

Running on the **second Foundry account** (`-002`, project `proj-default`), which is where the
EUR 100 budget sits. That deployment is **DataZoneStandard, capacity 200 — a hard 200,000 TPM
and 200 RPM**, and its subscription quota is fully allocated (GlobalStandard limit is 0), so it
cannot be raised without an Azure quota request. Measured throughput is 85,000-121,000 TPM, and
Stage 1's 225M prompt tokens therefore take roughly 40 hours. `-001` runs the same model build
as GlobalStandard 6000 and is about 30x faster, if throughput ever matters more than which
subscription is billed.

Command: `runs/run-25-stage1.sh` — 60K and 120K, fills 0.50/0.70/0.86, five strategies,
3 seeds, 3 probe repeats, payload fixed at 3,500-token tool results.

| cell | state |
| --- | --- |
| 60K / 0.50 | **done**, `runs/run-25-stage1-60k-fill50.txt` |
| 60K / 0.70 | **done**, `runs/run-25-stage1-60k-fill70.txt` |
| 60K / 0.86 | ran all 15 strategy-seeds over 3.5 hours, then **died before printing its table** |
| 120K / 0.50, 0.70, 0.86 | never ran — the first pass hit `APIConnectionError` on every call |

**Fixed, and to be used when resuming.** A cell used to print its table only when the whole
cell finished, so an interruption anywhere in 3.5 hours lost every seed; the 60K/0.86 cell was
lost to exactly that, having already been paid for. `--results-jsonl PATH` now appends one JSON
record per seed, written and closed as that seed is scored, and `--from-jsonl PATH` rebuilds the
table and verdict from the file with no provider and no calls. It is the same aggregation over
the same records, so a recovered cell is the cell that was measured. The file is appended to, so
one path can hold a whole sweep and a resumed run extends it; a cell missing strategies or seeds
renders and is marked `PARTIAL` with what it holds. Each seed also prints a one-line summary
(strategy, seed, cost, facts, accuracy) as it lands. **Add `--results-jsonl` to every command in
`runs/run-25-stage1.sh` before resuming.**

**The result that changes the remaining plan:** `rep+-` came out at 0-2 points on every row
while `seed+-` ran to 30-78 points. Re-asking the same snapshot is almost perfectly repeatable;
the variance is nearly all in *which seed*, meaning where the facts fall against a retention
boundary. So `--probe-repeats 3` is buying very little and `--repeats` is buying everything:
the remaining cells should trade probe repeats for seeds at roughly 3:1 within the same budget.

## 3c. The before/after of upstream PR #7912

Upstream merged [#7912](https://github.com/microsoft/agent-framework/pull/7912) on
2026-08-31, which changes the behaviour this package measures. The plan agreed is to keep the
current sweep as the **before** arm and re-run it after rebasing as the **after** arm, so the
pair says whether the fix improved the shipped default on cost and on recall.

**What it changes.** `ContextWindowCompactionStrategy`'s tool-eviction phase was
`TokenBudgetComposedStrategy(token_budget=tool_eviction_tokens)` called unconditionally; it is
now `ToolResultCompactionStrategy(compact_to=tool_eviction_tokens)` called only when
`included_token_count(messages) > tool_eviction_tokens`. So the phase both gained its threshold
and lost its destructive oldest-first fallback — exactly the behaviour the `context_window` row
has been penalised for in every run here. It also adds `preserve_first_user_group`, which
protects the turn carrying the requirements.

`_middleware.py` additionally reconciles compaction summaries at the pipeline boundary, so
compaction done inside the client now persists back into the caller's message list instead of
being dropped there. **That one reaches every compacting row**, because it changes whether
exclusions accumulate across turns — `anchored` makes position-only decisions precisely so its
choices stay stable across turns, which was a workaround for this.

**Arm identity.**

| arm | framework | lab code |
| --- | --- | --- |
| before | `3dbaaea3e` (our merge-base) | `5d3b998d0` plus the records/seed-offset work |
| after | `6a0773ba2` (upstream main) | the same, rebased |

We modify no framework file, so the before arm measures upstream compaction verbatim. Our 93
changed files and upstream's 244 **do not intersect**, so the rebase should be conflict-free.
#7918 is `[BREAKING]` for middleware but only enforces sequence-only inputs, which is what we
already pass.

**Honest framing:** the after arm advances 40 upstream commits, not one. #7912 is the
compaction-relevant change among them, and the pair should be reported as "upstream main before
and after #7912", not as an isolated bisect of that PR.

**Do not rebase while a sweep is running.** The venv is an editable install from this worktree,
so the framework would change under processes that have already started.

## 4. The finding the report does not yet state correctly

`REPORT-GPT-5-4-MINI.md` section 1 and finding 1 index the crossover on **tool result size
alone**. Three controlled points now show it is **two variables**:

| bearing results | codes each | per result | asides | facts recalled |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 8 | 25,200 | 0 | 18/53 |
| 6 | 8 | **8,000** | 10 | **36/53** |
| 16 | **3** | 8,000 | 0 | **53/53** |

Shrinking each result 25,200 -> 8,000 with codes held at 8 lifts recall 18 -> 36. Dropping
codes per result 8 -> 3 lifts it 36 -> 53. **Both matter; neither alone explains the
collapse.** The report needs its headline and finding 1 rewritten around that.

The third row is confounded on its own (it changed calls, size and codes together) and is kept
only as the first point of the series. The second row is the honest one: `--filler-tool-turns`
adds code-free lookups so call count varies without varying what must be remembered.

## 5. Strategies built here

| row | file | mechanism |
| --- | --- | --- |
| `anchored` | `_anchored.py` | fixed head and tail verbatim, band between shortened to a share of the ceiling, decisions from position alone so they never change on a later turn |
| `anchored_no_assistant` | same | as above, forbidden from shedding assistant prose |
| `tool_summary_anchored` | `_toolsummary.py` | `ToolResultRecallMiddleware` forces one recall tool call; the strategy drops every tool group in front of the resulting record |

**Not built, and the best next idea:** summarise each tool result *individually* rather than
all in one call. Design notes are in `REPORT-GPT-5-4-MINI.md` section 7. The three-point result
above is the strongest evidence for it — bounding both the size and the code count per
extraction call is exactly what it would do.

## 6. Instrument facts that cost money to learn

Each of these produced a plausible wrong number first.

- **Pinning is per call, not per turn.** A turn's `tool_choice` reaches only its first call;
  the follow-up after a tool result is unconstrained. Verified on the stub: call 1 carries the
  pin, call 2 carries `None`.
- **A tool passed through per-call `options` is never executed.** `FunctionInvocationLayer`
  wraps `ChatMiddlewareLayer` and builds its tool map first, so the tool reaches the model and
  nothing answers its call. Tools must be registered with the harness; only *permission* can be
  controlled per call, hence `RecallGate`.
- **A middleware cannot see the history before `call_next()`.** `context.messages` holds only
  the new turn until the history middleware replaces it during the call.
- **`run_live` forces `store=False`** itself. A caller that forgot it measured the service
  rather than compaction, and every narration mode came back falsely stable.
- **Unpinned runs cannot be measured at this scale.** Five repeats of the control varied 102%
  in cost; three repeats had shown 3%.
- **The dry run under-reported size** until the probe scenario was given `filler_tool_turns`;
  it reported 6 tool groups and 48,000 tokens where the scenario had 16 and 129,000.
- **Accuracy needs several samples; cost does not.** `c+-` is an honest error bar (run 23 showed
  within- and across-invocation spread are the same size), but it routinely runs 26 to 78 points
  on a compacting strategy, so any single accuracy figure is one draw from a wide, often
  two-valued distribution. Cost spread is 1 to 11% and its ordering reproduces every time.
- **`--tool-turns` is silently capped** by `--filler-turns`: extra tool groups are placed
  inside filler sections, so 16 requested with the default 6 filler turns yields 9. `plan_fill`
  now sizes the filler above that floor, and a test pins it.
- **The history provider drops byte-identical messages.** `filter_new_messages` hashes them,
  so a stub that answered the same words on every turn built a history a third the size it
  appeared to be, and the offline sizing test read 16,565 tokens where it should have read
  19,881. The stub numbers its replies now. Nothing about the live runs was affected -- a real
  model never repeats itself exactly -- but any future offline fixture must.

## 7. Repository state

- Branch `python-lab-cachebench`, ~40 commits ahead of `origin`, **not pushed**. Pushing needs
  its own go-ahead.
- `dev/` is untracked Git-LFS junk. **Never stage it.**
- 227 tests pass; ruff and pyright are clean. The measurement rebuild above is **uncommitted**,
  as is the per-seed results file (`_records.py`, `--results-jsonl`, `--from-jsonl`).
- `REPORT.md` and `ARTICLE.md` still describe only the six-model cross-provider work and
  predate everything from run 7 onward. They do not mention the 272,000 input limit, the
  crossover, or either new strategy.

## 8. Spend

About EUR 175 total. Roughly EUR 19 remains of the last top-up. A five-repeat matrix of five
strategies costs about EUR 4 at 60K, EUR 8 at 120K, EUR 11 at 272K, and EUR 10 for the
sixteen-call variants -- **all measured under the old design, where each closing question was
asked once**.

**The probe phase costs more than the seeding does.** Every probe carries the whole snapshot,
and there are `questions x --probe-repeats` of them. At the 270,000/0.86 cell with 16 tool
groups that is 18 questions x 3 = 54 probes of ~232,000 tokens each, against ~11.6M for the
seeding: the probes are now the larger half and `--probe-repeats` scales exactly that half.
`--dry-run` prints the arithmetic. Price a matrix before running one; the estimates above no
longer apply.
