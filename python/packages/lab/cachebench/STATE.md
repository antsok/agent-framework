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
- **Probe** — every closing question, asked from the restored snapshot: a scoped question
  `--probe-repeats` times, the combined one `--combined-repeats` times (both default 3).
  `restore_state` deep-copies again per probe *and* calls
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
spreads are separate columns: `seed+-` between seeds (compaction's reliability), `rep+-`
within one seed across probe repeats (the model's enumeration variance) and `rep2+-` the same
within-seed spread for the combined question. `cost` stays one column: seeding plus every
probe, summed. `out` has its own column so a total driven by verbosity is visible.

**The two accuracy columns are `acc1` and `acc2`.** They were `acc` and `all`, which named the
questions rather than the measures and left nothing saying the two are one run scored twice:
`acc1` is the scoped questions (requirements plus one per tool lookup, seven of them at
`--tool-turns 6`), `acc2` is the single combined question asking for all 53 values at once.
The rename reaches the header, the legend, both per-sample blocks, the progress line, the
ranking and threshold lines and the README. `CellStats.correctness` and `CellStats.combined`
keep their names -- they describe the thing, not the old column -- and say which column they
are in their docstrings, so nothing on disk moved for the rename.

**`acc2` now averages three attempts.** `--combined-repeats` (default 3) is the combined
question's own repeat count, independent of `--probe-repeats`; it is asked that many times
from the same restored snapshot, in the same loop, and `acc2` is the mean over every attempt
of every seed. The reason is that one `acc1` reading averages seven answers while one `acc2`
reading is a single answer, so at the `--probe-repeats 1` the sweep uses, `acc2` was one
sample per seed and noisier for that reason alone. Records already on disk hold one combined
sample per seed and still aggregate as the one answer they are: scoring reads the combined
probes themselves rather than counting up to a repeat count, and `CellParams.from_dict` fills
a missing `combined_repeats` from `probe_repeats`, which is exactly what an old record did.
The schema stays at 2 -- what a version 2 record measured is not in doubt, so there is no
"not measured" to confuse with a measurement.

**Verified live in run 24** (60K/0.86, five strategies): fill landed at +0.0%, probes hit 94-96%
cache, no disqualifications and no drift. The numbers in `RESULTS.md` and
`REPORT-GPT-5-4-MINI.md` all predate the rebuild, so none of them are disqualification-checked
and their `all` column -- now `acc2` -- is inflated: the combined question used to be asked
last, after seven answers had re-listed the codes into the context it read.

## 3b. Stage 1 sweep — abandoned, superseded by runs 26 to 29

**Read this for the account facts and the fixes it paid for, not for the plan.** Run 25's
design — 3 seeds, 3 probe repeats — was replaced by 5 seeds and 1 probe repeat once the
variance split at the end of this section was measured, and the sweep was replaced by the four
cells of runs 26 and 27 (60,000/0.86 and 120,000 at 0.50, 0.70, 0.86), then extended after the
rebase by runs 28 and 29. Its 60,000 cells at 0.50 and 0.70 have no successor and were dropped
rather than re-taken. `runs/run-25-*` is kept because it is where the resumable-records work,
the throttling retry and the `DQ`/`EXCL` split came from. Nothing in it needs resuming.

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
(strategy, seed, cost, facts, accuracy) as it lands. **Every script from `run-26` onward passes
`--results-jsonl`**, which is why runs 26 to 29 have `.jsonl` records beside their tables and
run 25 does not.

**Also fixed, and the reason a later sweep died.** A sweep lost 100 seeds and about EUR 4.17
to HTTP 429. `run_live` had a two-attempt loop, but it existed only to drop a request option
the provider had named; a rate limit is not an unsupported option, so it fell through, failed
the turn, and abandoned the whole seed — every row came back `ERR` with `0/32t`. Throttled
calls are now waited out and re-sent, inside the option-drop loop rather than beside it, so
the two compose. The bounds are named constants in `_live.py`: 6 attempts, 2s base doubling
per attempt, 60s per wait (the quota window is one minute), 300s total per turn, jittered 25%
downwards except when the provider names a `Retry-After`, which is taken verbatim and only
capped. Exhausting them still fails the turn — the point is to survive a spike, not to hide a
wall. Throttling is counted: `LiveOutcome.rate_limit_retries` and `throttled_seconds`, the
same two on `SeedRecord`, a `THROTTLED:<n>` flag in the table and a per-row seconds line under
it. **Records are `schema` 2 as a result; a version 1 file will be refused rather than
averaged in, and there are none on disk to lose.**

**The retry then had a bug of its own, and it destroyed the seeds it was built to save.**
`agent.run` is not idempotent. When a 429 landed *inside* the tool-calling loop -- after the
model had returned a function call, which per-service-call history persistence had already
written to the session, but before the matching tool result existed -- the retry re-sent against
a history holding a call with no output, and the provider refused it outright: `400 ... No tool
output found for function call call_<id>`. Measured at 7 occurrences in one cell, every one on a row
that had been throttled and **including the uncompacted `none` control**, which is what rules
compaction out. The turn then failed and the seed was abandoned. `_send` now snapshots the
session before the first attempt and restores it before every re-send, reusing the same
`snapshot_state`/`restore_state` pair the probe phase uses. Both retries restore, because both
re-send: an unsupported option is named on the call that carries it, and after a tool result
that is the turn's second call, so the option-drop path reaches a half-finished turn exactly as
throttling does. **The probe phase had the same exposure** -- it restored before a probe, not
before each of that probe's attempts -- and is covered by the same fix; an offline test drives a
tool call inside a probe and fails without it. Snapshotting every turn costs **6ms at 230,000
tokens**: `deepcopy` returns the immutable strings as themselves, so the copy rebuilds the 232
message objects around the payloads rather than the payloads, against a call that spends tens of
seconds sending them.

**Then a cell was lost to the network, and the retry covers that too now.** A cell of 30 seed
records came back every row `ERR` with no turns completed, lost whole to `APIConnectionError`,
and three cells of an earlier sweep went the same way. The retry covered HTTP 429 only, and a
request that never arrived is not a refusal, so the turn failed and the seed was abandoned.
`_attempt` now also waits out a **transient failure**: a dropped connection with its timeout
subclasses, and the statuses that say the provider failed to serve a request it received --
408, 500, 502, 503, 504. Anything the provider *decided* is re-raised on the first attempt: a
400, an auth failure, a context-length rejection. Re-sending a 230,000-token prompt against a
wall is the failure the `429`-substring bug already caused once, so the transient statuses are
enumerated rather than written as `>= 500`, and every text marker is a phrase -- a bare
"timeout" would match a provider rejecting an option *called* `timeout`, which belongs to the
option-drop path. Detection walks the exception chain the way `is_rate_limited` does: a status
first, then `isinstance` against `ConnectionError`, `TimeoutError` and `socket.gaierror`, then
the phrases, for a wrapper that rendered its cause and kept no object. `is_connection_error`,
in `_runner.py` beside the other two.

**Its own schedule**, because a network drop has no window to refill: 5 attempts, 1s base
doubling, 20s per wait and 60s per turn, against throttling's 6/2s/60s/300s. Both are
`_RetryBudget` objects and there is one of each per turn, so a turn that reconnects twice can
still wait out a quota window afterwards and starts that wait at the rate limit's own base.
`Retry-After` is honoured on both paths -- a 503 may name one -- and capped by the path's own
ceiling. The same snapshot and restore run before every re-send, so a connection lost between a
function call and its result cannot re-send a dangling call. Exhausting the budget still fails
the turn: the point is surviving a blip, not hiding an outage.

**Counted apart from throttling**, because the two readings differ: a throttled row waited out
a minute-wide quota window and may have lost its cached prefix, a reconnected one waited seconds
and re-sent the same prefix, so one flag for both would put that caveat on every row that merely
survived a blip. `LiveOutcome.connection_retries` and `connection_seconds`, the same two on
`SeedRecord` and `CellStats`, a `RECONNECTED:<n>` flag and a `Reconnected:` block under the
table. **Records are `schema` 3 as a result, and version 2 is still read**: a version 2 record
was written by code that could not re-send, so its zeroes are a measurement and not a gap, and
the six recorded cells -- 180 seeds -- read unchanged. Version 1 is still refused, its accuracy
columns being answers to a question the rebuild replaced.

**The flags column said `DQ` for two different things.** It read "excluded from the ranking",
which covers both a row that oversent and a row that never finished, while the `dq` column
means only the first — which is why the last table showed rows flagged `DQ` beside a `dq` of
`0%`: they had all died on the rate limit. `DQ` now means exactly what `dq` measures and the
other exclusion prints `EXCL`. `dq` itself is unchanged.

**The result that changes the remaining plan:** `rep+-` came out at 0-2 points on every row
while `seed+-` ran to 30-78 points. Re-asking the same snapshot is almost perfectly repeatable;
the variance is nearly all in *which seed*, meaning where the facts fall against a retention
boundary. So `--probe-repeats 3` is buying very little and `--repeats` is buying everything:
the remaining cells should trade probe repeats for seeds at roughly 3:1 within the same budget.
That trade is what made `acc2` a single sample per seed, which is why the combined question got
`--combined-repeats` (default 3) of its own -- three attempts of one question, not of seven.

## 3c. The before/after of upstream PR #7912 — rebased, one cell pair done

**Status: the rebase happened and one of the four cells has an after arm.** Runs 26 and 27 are
the before arm; run 29 re-ran 120,000/0.86 after the rebase and run 28 added a 100,000/0.86
cell with no before-arm counterpart. Written up in `RESULTS.md` §"Runs 28 and 29" and in
`REPORT-GPT-5-4-MINI.md` §4.2. **Still before-arm only: 60,000/0.86, 120,000/0.50,
120,000/0.70** — the two lower fills are the cheap ones and are the obvious next spend.

**What came out.** `context_window` at 120,000/0.86 went from +141% against not compacting to
+14%, hit rate 57% to 91%, `snap%` 47% to 57%, facts 21/53 to 28/53, `acc1` 38% to 54%, on a
control that moved 1% on cost. It discards less; every other column follows from that. The
cost half is a large, clean move; the accuracy half is direction only, its seed spread in the
after arm being 59 points. Run 28 agrees on cost and cache and **not on recall** — 21/53 there
— so whether retention improved is one cell's answer, not two. `tool_summary_anchored` also
moved, 47/53 to 53/53 at +16% rather than +22%, and that is recorded as unexplained: the
middleware-boundary change below is a plausible cause and nothing here isolates it.

Upstream merged [#7912](https://github.com/microsoft/agent-framework/pull/7912) on
2026-08-31, which changes the behaviour this package measures.

**What it changes.** `ContextWindowCompactionStrategy`'s tool-eviction phase was
`TokenBudgetComposedStrategy(token_budget=tool_eviction_tokens)` called unconditionally; it is
now `ToolResultCompactionStrategy(compact_to=tool_eviction_tokens)` called only when
`included_token_count(messages) > tool_eviction_tokens`. So the phase both gained its threshold
and lost its destructive oldest-first fallback — exactly the behaviour the `context_window` row
was penalised for in every run up to 27, and the change the after arm measures. It also adds
`preserve_first_user_group`, which protects the turn carrying the requirements.

`_middleware.py` additionally reconciles compaction summaries at the pipeline boundary, so
compaction done inside the client now persists back into the caller's message list instead of
being dropped there. **That one reaches every compacting row**, because it changes whether
exclusions accumulate across turns — `anchored` makes position-only decisions precisely so its
choices stay stable across turns, which was a workaround for this.

**Arm identity, as executed.**

| arm | framework | lab code | records |
| --- | --- | --- | --- |
| before | `3dbaaea3e`, our old merge-base | `5d3b998d0` plus the records/seed-offset work | `runs/run-26-*.jsonl`, `runs/run-27-*.jsonl` |
| after | `e2f7db207`, upstream `main` | the same code, rebased | `runs/run-28-100k-fill86-post7912.jsonl`, `runs/run-29-120k-fill86-post7912.jsonl` |

**The rebase rewrote our hashes**, so `5d3b998d0` is a pre-rebase name that is on no branch any
more; its rebased twin is `5ff29c1b6`, and the lab code the before arm ran is the tree at
`475de5225` ("runs 26 and 27, the first results from the rebuilt instrument"). **No executable
lab code changed between the arms**: `git diff 475de5225..HEAD` over the package reaches only
documentation — `compaction/STRATEGIES.md`, and a corrected break-even derivation in
`_anchored.py`'s comments — and `DEFAULT_MIN_GAIN_FRACTION` is 0.23 on both sides.

`d2a934d53` is #7912 itself, in case the phase has to be looked at again. **We still modify no
framework file** — every commit on top of `e2f7db207` touches only `python/packages/lab/`, so
both arms measure upstream compaction verbatim.

**The private imports survived the rewrite.** `compaction/_anchored.py` and
`compaction/_toolsummary.py` import nine names from `agent_framework._compaction` —
`EXCLUDED_KEY`, `GROUP_ANNOTATION_KEY`, `SUMMARY_OF_GROUP_IDS_KEY`,
`SUMMARY_OF_MESSAGE_IDS_KEY`, `annotate_message_groups`, `annotate_token_counts`,
`group_messages`, `included_token_count`, `set_excluded` — and all of them are still defined
there after #7912. That is luck, not a guarantee: it is private API and the next rewrite of
that module can take any of them away without notice.

**Honest framing:** the after arm advances ninety upstream commits, not one. #7912 is the
compaction-relevant change among them, and the pair is reported as "upstream main before and
after #7912", not as an isolated bisect of that PR.

**Do not rebase while a sweep is running.** The venv is an editable install from this worktree,
so the framework would change under processes that have already started. This is why runs 28
and 29 were taken after the rebase completed rather than around it.

## 3d. Known bug: the output reservation is not the output cap

**Fix this after the `gpt-5.4-mini` series is finished and before any other model is run.**
Agreed 2026-08-31. Fixing it mid-series would move every strategy's trigger threshold and make
the cells already measured incomparable with the ones after; leaving it past the series would
put the same flaw into a second model's baseline.

**What is wrong.** Strategies size their input budget as `--context-window` minus
`--max-output-tokens`, which the runs set to 2,048. But the value actually sent on the request
is `--answer-max-tokens`, which the runs set to **12,000** (`_providers.py` builds
`{"max_tokens": response_max_tokens}` from it). So at a 60,000 window the strategies believe
57,952 tokens of input are available when a reply may consume 12,000, leaving 48,000. The
budget every threshold is a fraction of is overstated by about 10,000 tokens.

**Why it does not bite yet.** `gpt-5.4-mini` has independent ceilings -- 272,000 input and
128,000 output -- so a long reply cannot push a legal prompt over the limit, and
`LiveOutcome.disqualified` checks the prompt alone. On a model whose window is shared between
input and output the instrument would report an overflowing run as clean.

**The fix** is to make the number used for the arithmetic the number actually sent, rather than
having two. Whichever way round, it needs a clean re-baseline of every cell, which is why it
waits for a model boundary.

## 4. The two-variable crossover — stated in the report now, and one point of it withdrawn

**Done:** `REPORT-GPT-5-4-MINI.md` used to index the crossover on tool result size alone, and
its section 1 and finding 2 now carry both variables. The table below is what they carry.

**One caveat added since:** the *fixed-payload* version of the same failure — the record
falling to 47/53 at 120,000/0.86 — did not reproduce after the rebase. The same cell reads
53/53 in run 29. The three-point series below is older than both arms and is unaffected; the
96,000-token context figure that used to be quoted beside it should not be quoted any more.

| bearing results | codes each | per result | asides | facts recalled |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 8 | 25,200 | 0 | 18/53 |
| 6 | 8 | **8,000** | 10 | **36/53** |
| 16 | **3** | 8,000 | 0 | **53/53** |

Shrinking each result 25,200 -> 8,000 with codes held at 8 lifts recall 18 -> 36. Dropping
codes per result 8 -> 3 lifts it 36 -> 53. **Both matter; neither alone explains the
collapse.**

The third row is confounded on its own (it changed calls, size and codes together) and is kept
only as the first point of the series. The second row is the honest one: `--filler-tool-turns`
adds code-free lookups so call count varies without varying what must be remembered.

## 5. Strategies built here

**They live in a subpackage now, so they can be lifted out whole.**
`agent_framework_lab_cachebench/compaction/` holds all four, plus `make_recall_tool` and
`RecallGate`, which moved out of `_live.py`: the tool the model calls and the gate keeping it
inert when unasked are part of the design, not benchmark scaffolding. Their tests moved with
them to `tests/compaction/`. Nothing in the subpackage imports from the lab, and
`tests/compaction/test_boundary.py` walks every module's imports with `ast` and fails if that
changes -- including relative forms, deferred imports and `TYPE_CHECKING` ones, none of which a
text search would find. `_strategies.py` stays in the lab: it is the `--strategies` registry,
which is benchmark configuration. `compaction/__init__.py` carries what an extractor needs --
the dependency on the private `agent_framework._compaction`, which survived PR #7912's rewrite
intact (§3c) but is still private, and the rename out of the `agent_framework` namespace that
has to happen before publication.

| row | file | mechanism |
| --- | --- | --- |
| `anchored` | `compaction/_anchored.py` | fixed head and tail verbatim, band between shortened to a share of the ceiling, decisions from position alone so they never change on a later turn |
| `anchored_no_assistant` | same | as above, forbidden from shedding assistant prose |
| `anchored_min_gain` | same | as `anchored`, but projects the reduction before mutating anything and declines any collapse worth less than 23% of the included prompt. **Run live in every cell from 26 onward; it only ever fired at 60,000/0.86, where its parent is not inert.** The floor is the break-even `R > B / (1 + T*c/(p-c))` at the 60K/0.86 cell, where `anchored` removed 263 tokens and cost 11% more than the control. Declines are counted and surface as `NOGAIN:<n>` in the flags column, so "never fired" and "fired to no effect" are distinguishable |
| `tool_summary_anchored` | `compaction/_toolsummary.py` | `ToolResultRecallMiddleware` forces one recall tool call; the strategy drops every tool group in front of the resulting record |

### 5a. The recall prompt was rewritten, and the record now has two bounds

**Live from run 26 onward.** Runs 26 to 29 all pass `--record-max-tokens 4000
--record-target-tokens 2000` and carry the rewritten prompt, so those six cells are comparable
with each other. **Run 25 and everything before it used the old prompt and the inherited cap**
and is not comparable with them on anything `tool_summary_anchored` did.

**The prompt was overfitted.** The tool asked only for "identifiers and values seen in earlier
tool results", which is this scenario's hex codes and nothing else -- prose, findings and
conclusions from a real tool result all fell outside it and would be dropped silently. The
description and the `values` text are the *entire* prompt, deliberately: the middleware sends
no message, because an appended one carries no history provider's source tag and would be
persisted into the user's own conversation. The replacement partitions the content in four, so
nothing falls outside all of them -- quote what cannot be reconstructed, keep findings and
conclusions as stated, carry over a summary the tool already wrote, summarise the rest -- and
resolves ties towards exactness. A parametrised test pins each clause, because a lost one is a
class of content dropped with nothing else in the design to notice.

**Asking for everything makes the record longer, so it needed bounding twice.** A `max_tokens`
cap does not make a model plan to fit; it writes until it is cut, and on a *tool call* the cut
lands inside the arguments JSON, so a cap set where the record should end produces no record
rather than a shorter one.

| | flag | default | what it does |
| --- | --- | --- | --- |
| cap | `--record-max-tokens` | 4,000 | `max_tokens` on the forced call **only**. It used to inherit `--answer-max-tokens` (12,000 in the runs), so the one call asked to summarise everything was the one call with no bound of its own. `0` restores that |
| target | `--record-target-tokens` | 2,000 | stated in the tool description at construction (`make_recall_tool(target_tokens=...)`), which is the only channel that reaches the model before it writes. `0` states none |

Roughly 2:1 on purpose, so overshooting the target is not the same event as being cut.

**Truncation is detected and reported.** A record that stopped early while still parsing was
the one silent failure left: the strategy anchors on it and drops every tool group behind
something that covers only part of them, and the loss is then scored as compaction damage.
`ToolResultRecallMiddleware.records_truncated` counts forced calls the provider ended with
`finish_reason == "length"` -- its own statement, rather than an inference from the record's
size -- and it surfaces as `TRUNCATED:<n>` in the flags column through the same duck-typed
`_strategy_notes` path as `REC`, `FORCED` and `FALLBACK`.

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
- **The history provider drops byte-identical messages, and this corrupts the control.**
  `filter_new_messages` falls back to `(role, serialized contents)` for identity when a message
  carries no `message_id`. A row running a strategy gets annotated and always has ids; the
  **control** does not, so its short acknowledgements to filler turns collide and are dropped.
  Measured at 120,000/0.86: the control peaks at 82 messages where every strategy row peaks at
  109, and `anchored` -- which plans nothing at that cell -- ends with a snapshot **5.4% larger**
  than the control's. Two rows that both did nothing are not the same conversation.

  **This entry previously read "Nothing about the live runs was affected -- a real model never
  repeats itself exactly." That was wrong**, and it is why the defect survived four sweeps: the
  `msgs` column showed it in every rendered table. Every `vs none` figure taken before the fix is
  biased in the control's favour. See [`REVIEW-2026-09-02.md`](REVIEW-2026-09-02.md) §1a.

  It also broke the offline sizing test, which read 16,565 tokens where it should have read
  19,881; the stub numbers its replies now, and any future fixture must.

## 7. Repository state

- Branch `python-lab-cachebench`, **rebased onto upstream `main` at `e2f7db207`** and carrying
  81 commits of its own on top of it. Every one of them touches only `python/packages/lab/`.
- The rebase rewrote history, so the branch and `origin/python-lab-cachebench` have diverged:
  the remote holds 21 commits that no longer exist locally and the local branch is 131 ahead of
  it. **A push would not be a fast-forward.** Pushing needs its own go-ahead, and it needs a
  decision about the rewrite first.
- `dev/` is untracked Git-LFS junk. **Never stage it.**
- 361 tests pass against the rebased framework; ruff, pyright and bandit are clean. Everything
  described in §3, §5 and §5a is committed — the measurement rebuild, the per-seed results
  file, the rate-limit retry and its snapshot-and-restore fix, the `acc1`/`acc2` split and the
  `compaction/` subpackage with the rewritten recall prompt and its two bounds.
- The working tree is clean apart from `dev/`, **the runs 28/29 write-ups** and **the
  connection retry**: `RESULTS.md` landed with the run, and the updates to this file,
  `REPORT-GPT-5-4-MINI.md` and `runs/README.md` are uncommitted, as is the transient-failure
  retry of §3b — `_runner.py`, `_live.py`, `_records.py`, `_live_cli.py` and their tests.
- `REPORT.md` and `ARTICLE.md` still describe only the six-model cross-provider work and
  predate everything from run 7 onward. They do not mention the 272,000 input limit, the
  crossover, or either new strategy.

## 8. Spend

About EUR 175 through run 27, plus about EUR 22 on runs 28 and 29 — roughly EUR 11 for one cell
of six strategies and five seeds at 100,000-120,000 and 0.86 fill. **The balance
needs checking before the next cell**: EUR 19 was recorded as remaining before those two runs,
which is less than they cost, so a top-up happened that is not written down here.

A five-repeat matrix of five strategies costs about EUR 4 at 60K, EUR 8 at 120K, EUR 11 at
272K, and EUR 10 for the sixteen-call variants -- **all measured under the old design, where
each closing question was asked once**.

**The probe phase costs more than the seeding does.** Every probe carries the whole snapshot,
and there are `scoped questions x --probe-repeats` of them plus `--combined-repeats`. At the
270,000/0.86 cell with 16 tool groups that is 17 scoped x 3 plus 3 combined = 54 probes of
~232,000 tokens each, against ~11.6M for the seeding: the probes are now the larger half and
the two repeat counts scale exactly that half.
`--dry-run` prints the arithmetic, now as a sum rather than a product. Price a matrix before
running one; the estimates above no longer apply.

**What `--combined-repeats 3` costs, measured on the 25 recorded seeds of the 60K/0.86 cell.**
A combined probe is the snapshot plus one question and is nearly all cache reads: at each row's
own recorded hit rate the two extra attempts add EUR 0.011-0.017 per seed, 6.6-9.3% of that
seed's cost, and EUR 0.36 on a cell that cost EUR 4.58. Priced at the 95% hit a repeat of an
identical prefix actually gets, EUR 0.27 on the cell, 2-8% per seed. The dominant term is the
prompt, not the answer: the combined answer measured 112-526 output tokens.
