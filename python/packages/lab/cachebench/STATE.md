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

## 3d. FIXED 2026-09-06: the output reservation is now the output cap — RE-BASELINE

**Runs 26 to 40 sit on the old arithmetic and are not comparable with anything measured after
this.** Every strategy triggers on a fraction of the input budget, and the input budget has
moved, so a cell taken before this and a cell taken after it are two different instruments.
Do not put them in one table, and do not read a difference between them as an effect.

**What was wrong.** Strategies size their input budget as `--context-window` minus
`--max-output-tokens`, which the runs set to 2,048. But the value actually sent on the request
was `--answer-max-tokens`, which the runs set to **12,000** (`_providers.py` builds
`{"max_tokens": response_max_tokens}` from it). So at a 60,000 window the strategies believed
57,952 tokens of input were available when a reply could consume 12,000, leaving 48,000. The
budget every threshold was a fraction of was overstated by about 10,000 tokens.

**Why it never bit.** `gpt-5.4-mini` has independent ceilings -- 272,000 input and 128,000
output -- so a long reply could not push a legal prompt over the limit, and
`LiveOutcome.disqualified` checks the prompt alone. On a model whose window is shared between
input and output the instrument would have reported an overflowing run as clean.

**Which fix was taken: send what is reserved, option (b).** Ordinary calls now carry
`--max-output-tokens` as `max_tokens`; `--answer-max-tokens` is sent on the closing questions
and on nothing else. The alternative was to reserve what is sent -- budget becomes
`window - answer_max_tokens` -- and it was rejected on three grounds:

- It reserves 12,000 tokens on every seeding call for replies that measure ~150 tokens on
  `gpt-5.4-mini` and ~602 on `gpt-5.6-luna`. That is a 20% haircut on a 60,000 window paid on
  every turn to insure against a reply no model here writes.
- It makes `--max-output-tokens` vestigial in the live runner. Two flags would name one
  quantity, which is the shape of the defect rather than a fix for it.
- The distinction the two flags draw is real and is the one this makes honest. A seeding reply
  is appended to the history and re-sent on every turn after it, so the budget the thresholds
  are fractions of must hold room for one. Nothing follows a closing answer -- the snapshot is
  restored before the next probe -- so its length is never re-sent, while the answer itself has
  to enumerate everything planted or be scored as lost facts.

**One number per call path, reserved and sent.**

| call path | number | reserved by | sent by |
| --- | --- | --- | --- |
| seeding turns | `--max-output-tokens` | `StrategyOptions.input_budget_tokens` = window − it | `runtime.options["max_tokens"]` |
| closing questions | `--answer-max-tokens` | window − it, checked before the run spends | `run_live(answer_max_tokens=...)`, per call |
| forced recall record | `--record-max-tokens` | the seeding reservation, checked before the run spends | the recall middleware, per call |
| summarizer | 1,024, fixed | not against this window — its own client, its own bill | its own `build_provider` |

**A third path had the same defect, found while fixing this.** A recall record is a tool call
the model writes *into* the conversation, and every record is preserved there for the rest of
the run, so that reply is re-sent on every later turn exactly as a seeding reply is. Its
reservation is therefore `--max-output-tokens` too, which makes `--record-max-tokens` a
tightening of the run's cap for one call rather than a cap of its own — and the shipped
defaults have it the other way round, 4,000 against a 2,048 reservation. Not clamped, because a
configuration silently narrowed is a configuration nobody ran; the run warns instead, and only
when `tool_summary_anchored` is selected, since no other row forces the call. Runs 26 to 40 are
on the loose side of this as well.

The closing reservation is checked rather than compacted to. Compacting the snapshot on the way
into a probe would move the material the answers are scored against, which the drift counter
exists to catch rather than to cause. So the run warns, before it spends anything and under
`--dry-run` too, when the fill target does not leave `--answer-max-tokens` of headroom. It is a
warning and not a refusal because every archived cell fails it: at 60,000 and 0.86 fill the
seeded prompt aims at 51,600 tokens and a 12,000-token answer does not fit beside it. That is
worth knowing before the next model is chosen -- on a shared-window model those cells are not
runnable as written.

**What to do with the archive.** Runs 26 to 40 stay as they are and stay quotable for what they
measured; `RESULTS.md` and `REPORT-GPT-5-4-MINI.md` are readings of that instrument. The next
run is the first point of the new baseline, and any cell to be compared against an archived one
has to be re-taken. The schema does not separate them -- `max_output_tokens` and
`answer_max_tokens` are both on the settings block and neither changed value, only what the run
did with them -- so this section is the record of where the line falls.

## 3e. Everything before run 32 is withdrawn on cost — read this first

An adversarial review on 2026-09-02 found two instrument defects sitting under every `vs none`
figure the project had produced. Both are fixed; the full account is in
[`REVIEW-2026-09-02.md`](REVIEW-2026-09-02.md). In short:

- **The control ran a different conversation from every strategy row.** Message identity falls
  back to a content hash when there is no `message_id`; compaction annotates and so always has
  ids, the control did not, so its repeated acknowledgements were dropped. At 120,000/0.86 the
  control peaked at 82 messages against every strategy's 109. `IdentifiedHistoryProvider` fixes
  it, a `MSGS` flag catches any recurrence, and a diverged control now refuses to be ranked.
- **`cost` summed the workload with twelve probe re-reads of the snapshot**, so a strategy that
  compacted hard was discounted twelve times over by the measurement. Cost is now `seed$`
  (workload) plus `probe$` (instrument), and **the ranking is on `seed$`**.

Two strategy defects went with them: the min-gain floor compared its saving against the whole
prompt where the derivation defines `B` as the tokens behind the edit (default moved 0.23 to
0.29), and anchored retention divided by the band's width *at the moment of trimming*, so what
a result kept depended on which turn the trim fired. The freeze test could not have caught
either — at its 3,000-token ceiling every allowance clamped to the same floor.

And the combined question was measuring its own wording: 226 of 240 control samples were either
every value or exactly eleven, because "every code returned by every deployment lookup" reads as
*the* return code of each. It states its counts now, and `acc2` moved from 58% to 100% on the
control.

## 3f. Runs 32 and 33 — the matrix on the repaired instrument

Nine cells, five seeds each, 300 records, $84.69 plus $4.12 to repair one cell. Cost against
not compacting, on the workload axis:

| cell | tool_summary | anchored | min_gain | truncation | context_window | control spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 60K / 0.86 | +47% | +10% | +1% | +15% | +29% | 17% |
| fixed 120K / 0.50 | +25% | -3% | +5% | +5% | +40% | 14% |
| fixed 120K / 0.70 | +27% | -1% | +3% | +7% | +34% | 21% |
| fixed 100K / 0.86 | +13% | -7% | +7% | +17% | +17% | 11% |
| fixed 120K / 0.86 | +9% | +3% | +4% | -2% | +11% | 21% |
| share 0.60, 120K / 0.86 | +8% | -1% | -1% | +10% | +30% | 16% |
| share 0.80, 120K / 0.50 | +12% | -2% | -2% | -2% | +17% | 31% |
| share 0.80, 120K / 0.70 | +11% | +3% | +6% | -2% | +32% | 3% |
| share 0.80, 120K / 0.86 | **-3%** | -2% | +1% | +7% | +25% | 7% |

**No resolvable saving anywhere** — every negative sits inside its control's spread.
`context_window` is dearer in all nine, +11% to +40%, after #7912, and worst or near-worst on
recall throughout. `tool_summary_anchored` reaches parity only at the highest fill and highest
tool share, where it answers from 45% of the window against the control's 88% with all 53 facts:
**headroom, not savings.** The anchored family is free and mostly idle.

Withdrawn by this run: "compaction pays when tool output dominates" (that -14% was the combined
column inflated by cheap probes). Surviving: "compaction never buys a smaller bill", now on
sound evidence, and the #7912 result.

## 3g. `anchored`'s idleness is the regime, not the configuration

`band_share` was unreachable — a constructor argument the registry never passed. It is now
`--band-share`. Tuning it makes the strategy act and never makes it pay: at a 117,952-token
ceiling with 3,500-token results, 0.05 removes 8,952 tokens and 0.01 sheds 94% of every result
to remove 18,114, against a break-even near 29,900. The whole tool payload is 21,000 tokens in a
103,200-token prompt. Idleness is correct behaviour there; the dial only buys a loss. Pinned by
a test that says so.

Note the retention gradient runs **oldest-keeps-most**, which is forced rather than chosen:
position is counted from the head so it never moves as the conversation grows, and counting from
the tail would re-decide every group on every turn — the path dependence just removed.

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
- **`hit%` mixes two phases, and the probe phase's hit rate is a two-valued draw.** 33.3% or about
  99.5% a record on luna, decided by whether the strategy was still acting on the store when
  seeding ended, and it is what most of the `hit%` spread between compacting rows has been since
  run 34. Quote the seeding half for anything about compaction. §3t has the arithmetic and the reach.
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

## 3h. The luna `tool_summary_anchored` row, and a wrong diagnosis corrected

**First diagnosis, now withdrawn:** that `--record-max-tokens 4000` cut the record and caused the
collapse. `TRUNCATED` did appear on 36 of 45 luna records and on none of mini's, and the nine
untruncated ones all kept 53/53, so the correlation was real. The causation was not.

**Run 36 settled it.** Fixed 60K/0.86, three seeds an arm, `truncation` present as a reference
row. Cap 4,000 -> 24,000 and target 2,000 -> 8,000: facts stayed at **21/53 in all eleven records**
across both arms and the archived cell. Raising the target moved cost from +57% to +95% and
accuracy not at all. One seed still truncated at 24,000 -- luna wrote past 24,000 tokens -- and
returned the same 21/53.

**Actual mechanism.** Exactly one record is written (`REC:1`, all 45 records); fact counts are
always `5 + 8k`; loss is uncorrelated with shrink (**r = 0.05** across 45 records). Luna expounds
on two lookups of six and never reaches the rest. Not a setting.

**Consequence.** On luna's fixed-payload cells the record is dominated by `truncation` -- same
facts, half the shrink, plus a model call. It earns its keep only in the share-0.80 cells.

**Instrument note.** Running `none,tool_summary_anchored` alone falsely trips `CONTROL DIVERGED`:
the MSGS check compares the control against the leanest strategy row, and with only that strategy
the leanest row carries the record's own forced calls (+13 messages). Any future single-strategy
run of it must carry a message-neutral row such as `truncation`.

**Process note.** `TaskStop` kills the wrapper, not the loop underneath; and overwriting a running
bash script makes bash re-read it and spawn duplicate loops. Both happened here and both cost
money. New variants get a new filename, and stale trees get killed by command-line match.

## 3i. Run 37: instruction does not move the record either

`RECALL_VALUES_DESCRIPTION` rewritten for breadth-first coverage, a per-result prose cap, and a
split tie-break. Three seeds a model at fixed 60K/0.86, default bounds. Luna 21/53 x3 (unchanged),
mini 53/53 x3 (unchanged), shrink and output both inside the existing seed range. **Reverted.**
The suspected clause -- "keep exactness over brevity" -- is cleared rather than convicted.

Four levers have now failed on 21/53 across fourteen records at this cell: the cap, the target,
the prompt, and both bounds together.

**What is left is structural.** One record is asked to cover every result (`REC:1`, all records),
and `tool_choice` pins one call, with a turn's pin applying to its first call only. Guaranteeing
coverage means forcing per group, or validating that the record names each group about to be
dropped and re-forcing for the remainder. Unbuilt, unmeasured, and it trades one call for several.
Do not retry a wording change without a reason the null result does not already cover.

## 3j. Runs 38-39 -- LARGELY WITHDRAWN, see REVIEW-2026-09-06.md

**Read the review first.** Verified corrections to what this section claimed:

- The post-record fallback **could not have fired in runs 26-35** -- every archived peak is 30-45%
  under its ceiling. "Runs 26-35 cannot be trusted on this axis" is **withdrawn**; they are the
  cleanest measurement of the design. Section 3h's explanation of luna's 21/53 stands and 3j
  wrongly overwrote it.
- The coverage check is broken twice over: it tokenises `code_1=VALUE` whole, so a perfect record
  scores zero, and it matches by substring, so it **can delete groups nothing recorded**. Every
  UNCOVERED number is an artefact.
- Run 39's -2% is not like-for-like (77% vs 85% fill, control moved +27%, NOT SUPPORTED printed).
  Against run 34's control it is +25%.
- Every trigger writes **two** records, repeats reach 4-5, and `repeat_records=True` shipped as the
  default on unit tests alone. RECFALLBACK counts attempts, not effects.
- Corrections to numbers stated: 270 records not 390; $36.08 not $48.77; 7 of 45 mini records had
  already lost facts; a 46 exists so "every count is 5 + 8k" is false; three arms not four levers.

**What still stands:** `_anchored.py` had no concept of a record, the run-38 trim was real, and
protecting the record is a correct fix. It is not the explanation of the archive.

**Next, in order:** fix the tokeniser and the substring match; fix the double record; re-examine
the 0.8 trigger, which the project's own record-quality data argue against; then re-measure.
Nothing should be swept until the coverage check is repaired.

## 3j-original. Runs 38-39: the strategy destroyed its own record

**Root cause.** The post-record fallback (`AnchoredCompactionStrategy`) shortens tool results in
place, and the record is a tool result. `_anchored.py` had no concept of it. A dumped luna record
held 4 lookups and 32 identifiers; the prompt held 2 lookups' worth.

**Two defects it hid behind.** The post-record fallback incremented no counter, so a row silently
became a different strategy. And coverage was keyed on function names no model writes -- luna
writes "extra0 deployment lookup returned codes: ...". That check cost mini 14 points of shrink
and made it lose 9 facts (run 38, the only seed where `RECFALLBACK` fired).

**Fixed.** `_preserve.py` marks the record unshrinkable, honoured by all three anchored removal
paths; coverage re-based on values; `RECFALLBACK` counter; `--max-groups-before-record` wired.

**Result (60K/0.86, 3 seeds a model).** luna 21/53 -> 53/53, +56% -> -2%, ranking above the
control. mini holds 53/53 at +32%.

**Still open, and this is the live question.** Coverage succeeded on 1 of 3 mini seeds and 0 of 3
luna seeds. Mini's other two seeds compacted nothing while paying for a record. All luna's
compaction now comes from the fallback, not from record-and-drop. `DEFAULT_COVERAGE_SHARE = 0.8`
is chosen, not derived, and is **not plumbed to the CLI** -- `_build_tool_summary_anchored` takes
defaults and `StrategyOptions` has no field. Plumbing it and sweeping the share is the next step,
before any conclusion about whether record-and-drop earns its extra call.

**Not re-measured.** Runs 26-35 predate all of this. Every `tool_summary_anchored` row in them
ran the unflagged fallback whenever a record did not free enough, so an unknown share of each was
measuring the anchored strategy. `RECFALLBACK` did not exist to say so.

## 3k. Run 40: the repaired check measured

Repairs confirmed live: `unc=0` on every no-repeat row, `recs=1` per trigger, `recfb=0` on all
three mini no-repeat seeds. Mini's shrink restored to 17-22% from 5-6%.

Cost, from the instrument: luna +9% (no repeats) / **-7%** (repeats); mini **+44%** / **+98%**.
Mini removes 17-22% of the snapshot and costs 44% more because the hit rate falls 81% -> 68% (the
seeding half falls 73% -> 55%, further; §3t) and
output rises 9,278 -> 13,261. **I published a mid-run reading off `snap%` that the cost axis
refutes.** Do not read these arms on shrink.

The record holds 53/53 in eleven of twelve seeds; `anchored` reads 39/44/52/52, `truncation`
20/20/29/29. Reliability is what it buys, at +9% to +98%.

Repeats: better on luna on every axis (40-42% shrink, recfb=0, the only sub-zero cost), worse on
mini on every axis (negative shrink on all three seeds). Conditional on whether one record can
cover everything, which the framework cannot know. **`repeat_records=True` as the default is not
supported by this;** off is the safer default and the knob wants documenting against `UNCOVERED`.
**Changed: the default is now off, and `--record-repeats` turns it on.**

Neither sub-zero figure is resolvable (NOT SUPPORTED at 35% and 46% spreads).

## 3l. Run 41: the clean baseline

First measurement on repaired code (coverage by values, record protected, one record per trigger,
repeats off, reservation fixed). 60K/0.86, 5 seeds, 5 arms, 4 rows per invocation. 100 records,
no throttling or errors. **Not comparable with runs 26-40** -- the reservation fix moved every
threshold.

`vs none$`: luna -1% (fixed), -3% (share 0.80), +7% (share 0.80 with repeats); mini +25% (fixed),
+10% (share 0.80). Two arms name the record, both NOT SUPPORTED (1% gap on a 36% spread, 3% on
19%); three name `none`.

**The record holds 53/53 in all 25 rows.** `anchored` 35-53, `truncation` 21-45. Reliable
retention is the property nothing else has. **Qualified 13 September, §3u:** 25 draws at
60,000 tokens of a row that is bimodal across runs 41-48 -- four of 58 records lost 16 to 32
facts, every one at `UNCOVERED:4` beside `RECFALLBACK` -- and 60,000 is not a floor: the
fallback can shorten a 3,500-token result behind the record there, and did offline.

**Repeats: 63-66% shrink, 10% dearer** than repeats-off on the same workload. Confirms the default
should stay off, and confirms that `snap%` does not predict cost -- I misread it twice during the
run.

**Open: the coverage cliff.** `unc` and shrink move one-for-one and flip within an arm (mini share
0.80: unc 4,3,0,4,3 -> shrink -2%,9%,41%,-1%,8%). Clears in 4/5 fixed rows, 2/6 share rows.
`--coverage-share` is the next sweep and is now reportable.

**Caveats:** the two share-0.80 cells differ in tool share (84% luna, 94% mini) so cross-model
reading there is not like-for-like; **`luna-fixed` at -7.8% is the only arm off fill** (mini-fixed
+4.2%, mini-share80 +2.7%, both luna share arms -2.3%). The earlier "+5.3% / -6.5%" here was run
42's fill misattributed with a flipped sign. Within-arm comparisons unaffected.

## 3m. Run 43: 170K at 0.9 fill -- the sizing breaks and retention stops being reliable

All 18 strategies, luna, 170,000/0.9, 5 seeds, 90 records, no throttling (one or two invocations
at a time; one alone runs near 1.3M TPM, and the quota rise to 4M allowed two).

**Sizing failure to fix before measuring here again.** Seeds seeded 124,636-155,530 tokens against
a 153,000 target -- 73% to 91% of the window. 80 turns compounds reply variance and the solver
cannot predict replies. The seeds are not one operating point.

**No strategy retains reliably.** All four retainers drop at least one seed, on different seeds,
and the failures are bimodal (53, or 13-21) rather than gradual. Fill does not explain it
(r = -0.22).

**Bounds an earlier claim.** `tool_summary_anchored` was 53/53 in all 25 rows at 60K; here it is
4/5 at +113% with a 92% spread. Reliable retention belonged to the small cells, not to the
strategy. The lost seed reads `UNCOVERED:4 RECFALLBACK:43` at 21/53 -- four whole eight-code
groups -- which is the pair every archived loss on this row carries (§3u).

**Cost.** VERDICT: none. `tool_result` +8% at 52/53 is the closest to parity; `anchored` +163% at
26/53; `token_budget_fallback` +241%. Cheap rows are cheap by deleting.

**Next:** re-solve the filler against a measured reply size at 80 turns before any further
large-window work; `--coverage-share` sweep still outstanding.

## 3n. Run 44: 100K at 0.9 fill, and the sizing bug diagnosed

All 18 strategies, luna, 100,000/0.9, 5 seeds, 90 records, no throttling.

**The sizing error is constant, not compounding.** 47 turns gives -8.3%, 80 turns gives -7.6%.
**Section 3m's explanation is withdrawn.** Measured seeding replies: 103, 201, 348, 448, 581
tokens against an assumed 602 -- above every seed, hence the constant shortfall, and a 5.6x spread
between seeds, hence the 22-27% variance. Fix the mean by re-measuring `--assumed-reply-tokens`
per model; the spread needs the solver to iterate or the analysis to use achieved fill.

**Results.** VERDICT: none. `tool_summary_anchored` +21% at 53/53 and the only full retainer
(5/5); next best 48/53. `anchored` +92% at 32/53; `token_budget_fallback` +162%.

**The window series, now three points.** 60K: -1% to -6%, 53/53, 25/25 rows. 100K: +21%, 53/53,
5/5. 170K: +113%, 47/53, 4/5. Retention survives to 100K and breaks after; cost degrades
monotonically and steeply. The strategy's case is a small-window case.

**Qualified 13 September, §3u.** "Survives to 100K and breaks after" reads a threshold into
five draws: all five 100K records carry `UNCOVERED:0`, so the pair that marks every archived
loss could not occur there, and at the archive's per-record rate five clean draws happen about
two times in three. The retention column of this series counts draws; the cost column stands.

## 3o. Runs 45 and 46: scaling the payload, and the first resolvable saving

300,000/0.9, luna, `none` vs `tool_summary_anchored`. Run 45 fixed payload (1 seed), run 46 scaled
(5 seeds). No throttling in either.

**Fixed arm completes the series:** removed 28.8 -> 22.6 -> 17.0 -> 11.7%, hit 88 -> 74 -> 66 ->
62%, cost -6% -> +21% -> +113% -> **+212%** at 60K/100K/170K/300K. The actable payload never grows;
the conversation does. (The hit series is whole-run `hit%`; the seeding half reads 88.0 -> 87.5 ->
76.2 -> 66.8, and the 100K step is entirely the probe-phase draw of §3t.)

**Scaled arm:** three of five seeds at **-21%, -23%, -33% with 53/53**, hit 96% against the
control's 98%. First resolvable saving with full retention in this project. Two seeds failed the
coverage gate (`UNCOVERED:4`) and cost +29% and +19% at 44/53. Those two read `UNCOVERED:4`
beside `RECFALLBACK` 8 and 2, the pair under which this row has lost facts in every archived
case (§3u): a group the gate keeps is not preserved, the kept groups are why the prompt stayed
over the ceiling, and the fallback that then ran shortened the band's results to its fixed
budgets -- the same nine facts on both seeds, which is what fixed budgets cutting the middle out
of results whose codes sit at fixed positions produce.

**The gate is the whole difference.** `--coverage-share 0.8` demands the record quote 80% of every
digit-bearing token in a group; a 26,814-token result holds the same 8 codes among 7.7x more
filler. No notion of result size, calibrated at 3,500-token results.

**Two caveats, both mine.** The cell is unranked -- `CONTROL DIVERGED MSGS:-2` -- because I ran only
`none,tool_summary_anchored` despite section 3h recording that exact trap and its fix (include
`truncation`). Verifiably a false positive (133 vs 135, record adds 2) but the spread guard is lost.
And fixed-vs-scaled moves turn count too (120 vs 51 filler turns), which is intrinsic and not
separable by this design.

**Next:** `--coverage-share` sweep at this cell **with `truncation` included**, which answers the
gate question and restores the verdict machinery in one run.

## 3p. Turn count drives cost, and bimodal rows defeat the aggregate

**Turn count, not context size.** Runs 45 and 46 seed the same size and differ 72% in cost on the
uncompacted control alone: 120 filler turns gives 147 calls, 20,220,087 input tokens and $0.5826;
51 filler turns gives 78 calls, 12,840,933 and $0.3396. Input is size x calls, and this project has
varied size while treating calls as incidental. Compaction acts only after the growth that was
already paid for, so a cell labelled by window says little about the bill. Within-arm figures stand;
cross-cell comparisons at different turn counts compare two things at once.

**Bimodal rows.** Run 46's cell reads 49/53 at acc1 85% with a 55% spread, and **no seed scored 49**
-- three at 53, two at 44. Control spread 16% against the strategy's 55%, so the variance is the
coverage gate, not conditions. The instrument cannot detect bimodality; per-seed tables are the
only honest summary for a row whose behaviour switches.

**Consequence for the next design:** vary turns and window independently, or report cost per turn
alongside cost per cell. Neither is done today.

## 3q. What exclusion does, and what a stateful provider actually breaks

Asked while designing `user_summary_anchored`: should it preserve the message count, since some
providers track the sequence statefully? Audited, and the answer reversed the design back to the
committed one.

**Exclusion removes messages from the wire, not just marks them.** `set_excluded` writes
`EXCLUDED_KEY`; `included_messages` filters on it; `apply_compaction` returns
`project_included_messages(messages)` rather than the mutated list;
`BaseChatClient._prepare_messages_for_model_call` returns exactly that, and it becomes the request
payload. `CompactionProvider.before_run` additionally rewrites the context by object identity. The
one place exclusion is only a mark is **storage** -- `after_run` keeps excluded messages in session
state so annotations survive, and `ChatMessageStore.skip_excluded` (default `False`) decides
whether they reload.

**Ten strategies drop messages.** Framework: `TruncationStrategy`, `SlidingWindowStrategy`,
`SelectiveToolCallCompactionStrategy`, `ToolResultCompactionStrategy`, `SummarizationStrategy`,
`TokenBudgetComposedStrategy` (whose strict pass excludes **system** groups). Lab:
`AnchoredCompactionStrategy._shed` and its min-gain subclass, `_drop_before` in the record
strategy, `_replace` in the user-turn strategy.

**Correction to something I asserted:** "every other strategy drops messages" is wrong for
`_anchored`. Its *primary* phase, `_collapse_tool_results`, rewrites `content.result` in place and
drops nothing -- count and order unchanged -- and says why: "the tool-call structure stays intact".
It only excludes in the secondary `_shed` phase. **An in-place mechanism already exists in this
package**, confined to tool-result text, where a result can be shortened without breaking the
call/result pairing. A user turn has no such pairing to protect, which is why the same trick was
not already applied there.

**The exposure is inertness, not desynchronisation.** On a stateless route (chat completions,
Responses with `store=False`) the client sends the whole projected list and there is no server
sequence to diverge from -- this is every cachebench run. On a stateful route the framework skips
`HistoryProvider.before_run` entirely because the service owns the conversation, so the agent sends
only the new turn and **compaction has nothing to compact**. Preserving the message count would not
have helped: on that route the client's list is not what the server reads. Where inline items *are*
re-sent beside a continuation marker, the risk is duplication, which the OpenAI client already
strips identities to avoid (microsoft/agent-framework#3295).

This package already forces the issue: `_live.py::wants_client_side_history` sets `store=False` for
any client that stores by default, with a measured reason -- a 16-turn conversation reported a
one-message prompt on every row while the service billed 82,708 input tokens for history the client
never sent.

**Incidental:** `exclude_group_ids` is exported from the framework and has zero callers. And a
collapse note, had we gone that way, costs 14 BPE tokens -- about 830 tokens for a 60-turn band at
the 170K cell, ~0.5% of the window.

## 3r. The composed row was structurally tool-only, and the fix is one reading of the prompt

**The defect.** `tool_and_user_summary_anchored` ran the record half at `--trigger-fraction`
0.6 and the user half at `--user-trigger-fraction` 0.8, record half first. On a workload whose
bulk is tool payload the record half removes that payload while the prompt is still in the 60s
of the ceiling and holds it there for the rest of the run, so the prompt never reaches 0.8 and
the user half is never consulted. At a 170,000-token window the arithmetic is flat:

    ceiling            167,952
    user trigger 0.8   134,362
    record trigger 0.6 100,771   <- below the user line

Measured, run 47a seed 2: `snap 63%`, `REC:1 RECORDS:1 FORCED:1 RECFORCED:1`, 94.6% hit rate,
`USERCOMPACT:0`. The row was `tool_summary_anchored` under a longer name, and its own starvation
counter -- then a within-pass transition test -- reported nothing, because the transition it
looked for never happened: the prompt was under the user line before the run began rather than
taken under it during a pass.

**The fix that does not work, written down because it is the obvious one.** Giving both halves
the same fraction changes nothing while the phases are still judged one after the other: the
record half acts first, takes the prompt below the shared line, and the user half declines on
its own re-test. Same inertness, different number.

**The fix.** `__call__` reads `included_token_count` once, before either phase runs, and hands
that one number to both through a new `compact_against(messages, *, prompt_tokens,
trigger_tokens)` seam on each part. Both halves are judged against the size the prompt had when
the pass began, so a half that would have fired on the entry size fires whatever the other
removed first. The shared line is the record half's own (`tool_results.trigger_fraction`), so a
sweep of `--trigger-fraction` moves both halves together; aligning the other way is the unsafe
direction, because the record is written by a model that degrades with the bulk it reads and the
middleware that asks reads the same prompt. `--user-trigger-fraction` therefore moves the single
row only, which is now stated in the flag's help, in the record schema, and in three documents.

Judging a half against a slightly stale size cannot let it claim the other's material -- the two
selection rules name disjoint group kinds. What it costs is that the second half may act when
the prompt is already under the line. That is the deliberate trade.

**`USERSTARVED` had to be redefined, not retargeted.** With pass-entry judging, within-pass
starvation is impossible in any configuration, and the old cumulative counterfactual then fires
on every quiet pass after a successful compaction -- the prompt is small *because the row
worked*. It now accumulates only what the record half removed on passes whose entry size was at
or below the user line, exposed as `tokens_removed_out_of_user_reach` so it can be checked, and
the test runs before the record phase so this pass's removal cannot explain this pass's reading.
**On a default row it is zero by construction**: the record half acts only above the shared line,
and above the line the user half is consulted. Verified from the operators rather than from the
tests -- both parts return early on `prompt_tokens <= trigger_tokens`, and the accumulator is
guarded by `entry_tokens <= user_line`, which cannot both hold, including at exact equality where
neither phase acts. A non-zero value there is a defect report.

**Smoke, 170,000/0.9 scaled payload, one seed, probes cut to 1** (mechanism only -- `seed+- 0%`
means unknown, not stable). **Superseded by run 47, below**, which measured the same rows on five
seeds; the table stays because it is what cleared the relaunch:

| row | msgs | tok left | snap% | hit% | seed$ | vs none$ | facts | acc1 | flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none` | 85/85 | 137,225 | 81% | 97% | $0.0955 | — | 53/53 | 100% | — |
| `tool_summary_anchored` | 79/87 | 91,811 | 54% | 94% | $0.1019 | +7% | 53/53 | 100% | REC:1 RECORDS:1 |
| `user_summary_anchored` | 62/86 | 125,380 | 73% | 92% | $0.1561 | +63% | 53/53 | 100% | USERCOMPACT:2 USERHELD:30 USERREPLACED:24 USERUNDER:63 |
| `tool_and_user_summary_anchored` | 51/90 | 73,830 | 43% | 86% | $0.1576 | +65% | 53/53 | 89% | REC:1 RECORDS:1 UNCOVERED:1 USERCOMPACT:5 USERHELD:22 USERREPLACED:12 USERUNDER:69 |
| `truncation` | 60/85 | 83,659 | 49% | 92% | $0.1268 | +33% | 29/53 | 56% | — |

The mechanism is confirmed: both halves fire, no `USERSTARVED`, and the composition reaches
further than either half alone -- 73,830 tokens left against 91,811 and 125,380, `snap%` 43
against 54 and 73. **Nothing here is a cost result.** One seed, and both user-half rows sit at
+63% and +65% on a single sample.

**The user half's hysteresis bites hard at this sizing.** `USERHELD:30` against `USERCOMPACT:2`
on the single row, and `USERHELD:22` against `USERCOMPACT:5` composed -- both `USERCOMPACT` figures
double-counted, see §3s. That is
`--user-min-band-share` 0.1 doing what it was written to do on a workload where the band is the
minority of the prompt -- which is the other finding here.

**The payload scaling inverted the half each strategy may touch, and two documents still said
otherwise.** Run 43 held the tool payload at a fixed 3,500 tokens per result: 57% user text
against 14% tool results, so the user band was most of the prompt. With scaling now the default
(`--tool-share` 0.6) the same 170,000/0.9 cell seeds 91,791 tokens of tool results against about
38,664 of user-side filler. The record strategy now works on the bulk and the user strategy on
the minority -- the opposite of the regime both strategies' rationales were written against.
Corrected in `STRATEGIES.md` and `compaction/STRATEGIES.md`.

**Run 47 relaunched** with all twenty strategies, five seeds, two streams. Run 47a -- the aborted
attempt -- is archived under `runs/` as evidence, because it is the only record behind figures
the source quotes.

**Run 47 completed and archived** -- `runs/run-47-luna-170k-fill90-all20.txt` and the five
`-s{0..4}.jsonl` beside it: 100 records, $18.22, no throttling, retries or errors. What five seeds
say, against the smoke above:

| row | `snap%` | `hit%` | `seed$` | `seed$+-` | `vs none$` | facts | `acc1` | user-half flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none` | 86% | 97% | $0.1113 | 19% | -- | 53/53 | 100% | -- |
| `tool_summary_anchored` | 50% | 95% | $0.0908 | 22% | -18% | 53/53 | 100% | -- |
| `user_summary_anchored` | 71% | 93% | $0.1471 | 24% | +32% | 53/53 | 100% | USERCOMPACT:2 every seed, USERHELD 26-39, USERREPLACED:24, USERUNDER 62-75 |
| `tool_and_user_summary_anchored` | 30% | 76% | $0.1357 | 30% | +22% | 53/53 | 100% | USERCOMPACT 4-7, USERHELD 2-4, USERREPLACED 8/14/16, USERUNDER 94-98, no USERSTARVED |

- **The mechanism holds at five seeds.** Both halves of the composed row fired on every seed,
  `USERSTARVED` is absent from all five, and the composition is the deepest-compacting row in the
  cell that loses nothing -- 30% against the record half's 50 and the user half's 71, where every
  row below it in `snap%` lost 33 to 45 facts. The smoke's `UNCOVERED:1` and 89% `acc1` on the
  composed row did not recur: 53/53 and 100% on all five, `UNCOVERED` 0 throughout.
  **Qualified 13 September, §3u:** the record row's five 53/53 are five draws of a bimodal row
  -- run 48's recompact arm, the same cell, drew 37/53 on one seed of five -- and the composed
  row's 53/53 stands on 21 archived records, one of them at `UNCOVERED:4 RECFALLBACK:1`.
- **The hysteresis finding holds**: `USERCOMPACT:2` against `USERHELD` 26 to 39 on the single row,
  every seed -- and `USERCOMPACT:2` is one compaction, the count being doubled by the pass that
  §3s describes. The band is the minority of the prompt under the scaled payload and the share
  refuses it; this is the sizing, not the strategy.
- **The cache penalty is the result.** 76% against 95% and 97%, and per seed the composed row's
  hit rate tracked its passes -- 89% at `USERCOMPACT` 4, 73% and 75% at 5, 70% at 6 and at 7. That
  is what rewriting the user summary in place costs, and it is the measurement the `boundary` and
  `fold` modes now in the tree were written against. Neither mode ran here.
  **Withdrawn 13 September, see §3t.** The per-seed figures are the probe-phase draw -- one seed of
  five drew the high value and reads 89%, the four that drew the low value read 70-75% -- and the
  seeding half sits at 81% to 86% on all five with no relation to the pass count, which was itself
  double-counted. What stands is the seeding-phase gap: 84.1% against the record row's 92.3% and
  the control's 95.5%, about eight points against the record half. The modes' strict-prefix
  argument does not rest on the withdrawn figure; run 48 (§3s) carried them and could not rank them.
- **Nothing here is a cost result, and the smoke's +63%/+65% were one seed.** `seed$+-` runs 19% to
  30% across the four rows. The instrument printed `NOT SUPPORTED` on its own verdict
  (`tool_summary_anchored` -18% against a 22% spread), and the composition's +22% is inside its
  30%. The user row's +32% is the one figure that clears the rule -- 24% on the row, 19% on the
  control, every seed dearer at +23% to +45% -- so its direction is supported and its size is not.
  Paired by seed the record row is cheaper on all five (-8% to -30%) and the composition dearer on
  all five (+3% to +35%); those are signs, not sizes. Quote no saving and no penalty from this cell
  for either of those two rows -- `REVIEW-2026-09-06.md` §3 is the precedent.
- **Seed 3 is off target and in the aggregate.** `-s3.jsonl` seeded 137,396 tokens, -10.2%, and
  prints `FILL OFF TARGET` rendered alone; the other four sit at -2.1% to -3.9% and the merged cell
  at -4.6%. Its control is the cheapest of the five ($0.0962, 15% under the next), so it is the
  record row's worst seed at -8%. Re-rendered over the other four: retention unchanged on every
  row, the composition still 76% and 31%, the control's spread down from 19% to 4% and the user
  row's from 24% to 15%, the record row -20% against 21% (still `NOT SUPPORTED`), the composition
  +20% against 26%, the user row +32% against 15%. Dropping it changes nothing above.


## 3s. Run 48: the three modes measured alike, and the double pass behind it

Run 48 (`gpt-5.6-luna`, 170,000 at 0.9 fill, scaled payload, five seeds per mode; archived 13
September as `runs/run-48-luna-170k-fill90-usermodes-*`, one report per arm, $11.42, two throttled
calls on the boundary arm's control and nothing else) put `recompact`, `boundary` and `fold` side
by side. The standalone `user_summary_anchored` row read `USERCOMPACT:2, USERREPLACED:24,
USERSUMMARIES:1` and 93% hit on fourteen of the fifteen records (recompact `-s3` read
`USERCOMPACT:3`, `USERREPLACED:11`, 91%). Diagnosed offline by driving
`build_live_agent(kind="harness")` against a stub client; the reduced form is
`tests/test_live.py::test_the_user_band_strategy_compacts_the_stored_conversation_once_per_crossing`.

- **The boundary mode engages live.** Across three crossings offline it leaves three preserved
  summaries in the store, each found again by its id on the next pass, each band the turns newer
  than the newest; recompact leaves one. The store round trip loses nothing the boundary needs.
- **Every crossing ran the strategy twice on one band.** The harness runs the strategy inside the
  model call on the copies `SessionContext.extend_messages` hands out -- each with its own
  `additional_properties`, so nothing written there reaches the store -- and again after the turn
  on the stored list through `CompactionProvider.after_run`. Each pass called the summarizer; only
  the second persisted. `USERCOMPACT:2` was one persistent compaction, `summ$` was double, and the
  model was sent two different summaries at one position on consecutive calls: a second
  whole-suffix break per crossing, in every mode. A tool turn's second call adds a third copy pass,
  and a crossing the reply pushes over the line lands on the store pass alone, which is why the
  composed row's `USERCOMPACT` reads 4 to 8 against `USERSUMMARIES:3` in the boundary and fold
  arms, and 4 to 9 in the recompact arm, whose counter reads 1 by construction.
- **So the standalone row made one persistent pass per seed in all three modes**, and the modes
  differ only from the second pass on. At trigger 0.8 and fill 0.9 the band cannot regrow to a
  tenth of the prompt before the run ends -- `USERHELD` 29 to 40 is that refusal -- so this cell
  cannot separate the arms on the standalone row at all. The composed row fires at 0.6 and reached
  three standing summaries (`USERSUMMARIES:3` in boundary and fold against 1 in recompact).
- **The composed row's hit rates do not rank the modes either.** Split by phase, seeding sits at
  85.1% / 84.1% / 84.8% (recompact / boundary / fold), and the probe phase is bimodal per seed --
  0.333 or about 0.99 -- with boundary drawing 0.333 on five of five seeds, fold on three and
  recompact on two: not separable by mode at five seeds an arm, and §3t says why the draw is a
  property of the instrument rather than of a mode. The headline 83% against 72% (the recompact
  and boundary arms' `hit%`; this bullet used to say 89%, which is run 47's one high-drawing seed)
  is that draw. `hit%` mixes the two phases; read the seeding half when comparing modes.
- **Fix, committed as `8c463f0e7`:** the strategy keeps its last summarizer request and replays
  the answer, under the same id, when the identical request comes round -- no call, no counted
  compaction, and the store carries the bytes the model already saw. `USERREPLAY:<n>` counts those
  passes (`user_summaries_replayed`, proxied on the composed row). `USERCOMPACT` now counts
  persistent compactions only, so archived `USERCOMPACT` from runs 47 and 48 is inflated by up to
  2x against new rows, and summary ids in the store are consecutive again (`user_summary_0`, `_1`
  rather than `_1`, `_3`, `_5`). Seven tests added; 595 pass.
- **`USERFOLD:0` is right**: standing summaries were 1.3% to 1.7% of the composed prompt against a
  10% threshold, and the standalone row never had the two summaries a fold needs. The three
  schema-11 counters are wired end to end; the records carry non-zero `USERSUMMARIES` and
  `USERSUMMTOKENS`.
- **What a run would need to measure the arms**: at least two persistent passes on the standalone
  row, so a trigger low enough, or a fill high enough, that the user band regrows a tenth of the
  prompt after the first pass -- the composed row's 0.6 does it on this workload. Until then the
  arms cost the same and measure the same, and more seeds at this sizing buy nothing.
- **The recompact arm's `tool_summary_anchored` row lost 16 facts on one seed** (`-s2`, seed 3:
  37/53 at `UNCOVERED:4 RECFALLBACK:3`, the other four at 53/53, the column at 50/53 with `acc1`
  94% and `seed+-` 30pp). Run 47 had read 53/53 on all five at this cell. It is the fourth
  archived loss on that row and carries the same flag pair as the other three; §3u.

## 3t. The probe phase's hit rate is a two-valued draw, and every `hit%` mixes it in

Found 13 September while checking run 48 against its records. `hit%` is `cached_tokens /
input_tokens` over the whole run, seeding and probes together. Split at the record's own `probe_*`
fields -- seeding hit `(cached - probe_cached) / (input - probe_input)`, probe hit `probe_cached /
probe_input` -- the probe half on `gpt-5.6-luna` is **bimodal: 32.8-33.3% or 99.0-99.9%, nothing
between**, on 100 of 100 run 47 records and 64 of 65 run 48 records (one `truncation` record at
97.4%). The seeding half is continuous, and it is the number that says what compaction did to the
cache.

**What 33.3% is.** Twelve probes a record: seven scoped questions once each and the combined
question five times. On every 33.3% record checked, across prompts of 17,000 to 64,000 tokens, the
probe phase's cached total is 3.98 to 4.0 times one probe's prompt. That is four probes served
entirely from cache and eight served cold, and the only four probes with a byte-identical
predecessor are repeats two to five of the combined question. On those records the provider served
the snapshot prefix to an identical prompt and not to a sibling that shared it; on a 99.5% record it
served it to all twelve. The records carry no per-call usage, so this is arithmetic rather than
observation. Per-probe cached tokens in the record is the instrument change that would make it
observation, and the change this section asks for.

**Which records draw it is not a coin flip, and it is not a property of the mode.** Across run
47's twenty rows: the control, `context_window` and its two variants, `truncation`, `anchored`,
`tool_summary_anchored`, `tool_result`, `selective_tool_call`, `token_budget_window_first` and
`user_summary_anchored` drew the high value on all five seeds; `sliding_window`, `summarization`,
`token_budget_tools_first` and `token_budget_truncate_first` drew 33.3% on all five;
`token_budget_fallback`, `token_budget_summarize` and the composed row on four of five;
`anchored_no_assistant` and `anchored_min_gain` on one of five. The rows at rest by the end of
seeding hit; the rows still acting on the store when seeding ended -- a summarizer that fires
every turn, a ceiling the last turns crossed, a user summary rewritten on the last crossing --
miss; on the rows between, which value a seed draws is decided by whether the strategy acted on
the final seed turns, and that is the seed's draw, not the mode's. `context_drift` is 0 on nearly
every 33.3% record, so the probe prompts did begin with the snapshot verbatim (`prompt_text` is
the projected post-compaction prompt, so the counter can see a re-fire); what they did not begin
with is any prompt the provider had already served, because the store changed after the last seed
call -- `prompt_tokens_final` under `seed_prompt_tokens` on `token_budget_truncate_first` and
`anchored_no_assistant -s2`, 53 summarizer calls in 54 on `summarization`. That is the common
property, and it is consistent with every record here; it is not the mechanism, because a prefix
cache is documented to serve a shared prefix to a sibling, and the direct test -- send A+B, then
A+C on this route, read `cached_tokens` on the second -- has not been run. Run 47a's unbounded user
row, which re-fired on every probe (`DRIFT:12`), drew 0% and 17%: the same shape with no
identical repeats left to hit.

**Where it reaches.** Every archived luna record from run 34 on carries it. Runs 32 and 33
(`gpt-5.4-mini`, 270 records) drew high throughout, and mini's probe half in runs 40 and 41 sits
at 74-93% rather than at either pole. The rows whose archived `hit%` is materially the draw and not
the strategy: `anchored` and `anchored_min_gain` at 120K, 170K and 200K (runs 34, 43, 44);
`tool_summary_anchored` at 100K, 170K and 300K fixed (runs 44, 43, 45); `truncation` at 60K (run
34); the `token_budget` family and the composed row at 60K to 170K (runs 42 to 47). The claims this
changes:

- The composed row's per-seed "hit rate tracks `USERREPLACED`" in run 47 -- withdrawn, §3r and
  `RESULTS.md`. Seeding hit 81-86% on all five seeds, the 89% one seed's high draw.
- The run 48 mode ranking -- there is none, §3s. Seeding hit 84.1 / 84.8 / 85.1.
- `tool_summary_anchored`'s hit rate "falling 88 -> 74 -> 66 -> 62" across 60K/100K/170K/300K
  (§3o, `RESULTS.md`, `README.md`, and the `--tool-share` rationale in `_live_cli.py`): the seeding
  half reads **88.0 -> 87.5 -> 76.2 -> 66.8**. The 100K step is entirely the draw (four of five
  seeds at 33.3%); the fall from 170K on is real and about two thirds the size the mixed column
  shows. The removed-share series the payload change rests on is `snap%` and untouched.
- Run 40's "mini costs 44% more because the hit rate falls 81 -> 68" (§3k): the seeding half falls
  73 -> 55, further, so the conclusion stands and the mixed column understated it.

What it does not reach: every cost ranking, verdict and `vs none$` since the money split is on
`seed$`, which excludes the probes by construction; retention and accuracy never read the cache;
and the uncompacted control drew high on every luna record in the archive, so every "control at
95-98%" statement stands. Runs before 32 carry no `probe_*` fields and cannot be split.

**Rule from here.** Quote the seeding-phase hit when the claim is about compaction, and put the
probe half beside it whenever a row is compared on cache at all. The table should print the split
and the record should carry per-probe cached tokens; until they do, the arithmetic above is three
fields from any record.

**Built 13 September, uncommitted.** The table prints `seed hit%`, `probe hit%` and `run hit%` in
place of `hit%` -- the money split's names, in that order, so the eye lands on the seeding half --
with a per-seed `[seeding/probe]` block under it and, on records from schema 12 on, a per-probe
block; the cross-cell section carries the first two. Schema 12 adds `probe_input_samples` and
`probe_cached_samples`, one entry per probe in the order asked, `None` on every older record. No
flag: a low probe half withdraws nothing and the column already carries it. Both cell figures are
pooled over the seeds' tokens as `hit%` always was, so `run hit%` is exactly their token-weighted
mix; the mean of the per-seed rates sits up to half a point off the pooled figure, and the block
shows the per-seed values. Rerendered from the archive, the composed row reads 85.7/85.0/82.2/
86.4/81.0 seeding against 33.3/99.6/33.3/33.3/33.3 probe in run 47, and run 48's arms 83.6/84.7/
85.1 pooled (84.1/84.8/85.1 as per-seed means, the figures §3s quotes). Two things the rerender
says that this section does not: at 60,000 tokens (runs 41 and 42) the probe half is *not*
two-valued -- about 72% appears on five of fifteen `tool_summary_anchored` seeds -- so the two
values are a property of the cells from 100,000 up; and the 88.0 -> 87.5 -> 76.2 series above is
run 41's 60K figure and per-seed means, where the column prints run 42's 87.7 and, pooled, 87.4 ->
75.4 for 100K and 170K.

## 3u. The record row's retention is bimodal, and one pair of flags marks every archived loss

Found 13 September, re-reading the archive after run 48's recompact arm printed the record row
at 50/53 on the cell where run 47 had printed 53/53. The 58 `tool_summary_anchored` records of
runs 41 to 48 (47a included) hold four that lost facts:

| run | file | facts | flags |
| --- | --- | ---: | --- |
| 43 | `run-43-luna-170k-fill90-s2.jsonl` (seed 3) | 21/53 | `UNCOVERED:4 RECFALLBACK:43` |
| 46 | `run-46-luna-300k-fill90-scaled-s0.jsonl` (seed 1) | 44/53 | `UNCOVERED:4 RECFALLBACK:8` |
| 46 | `run-46-luna-300k-fill90-scaled-s4.jsonl` (seed 5) | 44/53 | `UNCOVERED:4 RECFALLBACK:2` |
| 48 | `run-48-luna-170k-fill90-usermodes-recompact-s2.jsonl` (seed 3) | 37/53 | `UNCOVERED:4 RECFALLBACK:3` |

Split by the two counters: neither flag, 30 records, no loss; `RECFALLBACK` alone, 16, no loss;
`UNCOVERED` alone, 5, no loss; both, 7, of which four lost and three held -- the three at 60,000
tokens (run 41 luna share 0.80 seeds 2 and 5 at `UNCOVERED:3`, mini fixed seed 3 at
`UNCOVERED:4`). **Every loss carries `UNCOVERED:4` beside `RECFALLBACK`, no record with either
alone has lost a fact, and the pair is where a loss can happen rather than a loss in itself.** The
composed row's 21 archived records (runs 47, 47a, 48) all held, one at `UNCOVERED:4 RECFALLBACK:1`.

**Why the pair, from the source.** `_drop_before` keeps an uncovered group but does not preserve
it -- only records are preserved (`_preserve_records`). A kept group is why the prompt stays over
the ceiling, and over the ceiling the fallback runs (`RECFALLBACK`) on its band, where every tool
group that is not a record is fair game: `_collapse_tool_results` shortens each in place to a
harmonic budget of `0.25 x ceiling / (position + 1)`, and `_shed` then drops whole groups oldest
first if that was not enough. An excluded group keeps its band position, so the groups behind the
record sit behind the ones the record dropped or kept and get the smallest budgets. Run 43's
21/53 is 32 facts, four whole eight-code groups: at 170,000 with 3,500-token results the budget is
at least 6,998 at every position, nothing can be shortened, and the shed phase takes the oldest
tool groups in the band, which are the four uncovered ones. Run 46's 44/53 is nine on both seeds,
which is what fixed budgets cutting the middle out of results whose codes sit at fixed positions
produce. The archive carries no prompt text, so which phase did what on a given record is
arithmetic, not observation.

**60,000 is not a floor.** The budget there is 14,488, 7,244, 4,829, 3,622, 2,897 and 2,414 at
band positions 0 to 5, and the four groups a record accounts for still occupy positions 0 to 3
after they are excluded, so a 3,500-token result behind the record sits at 2,897 or 2,414 and can
be shortened. Driven offline through the harness with a stub model whose record covered every
group (13 September, `ceiling_probe60.py` in the session scratchpad, a variant of the
`ceiling_probe.py` beside it), the fallback fired once the prompt crossed 57,952 and shortened one
result behind the record: 50/53 at `UNCOVERED:0`. The archive's 30 records at 60,000 lost
nothing; that is 30 draws, not a property of the window.

**Seeds are not paired across runs.** The scenario salt is
`f"{time.strftime('%Y%m%d-%H%M%S')}-{name}-{repeat}"` (`_live_cli.py`), so two invocations of one
cell at different times build different markers, and "seed 3" in run 47 and "seed 3" in run 48
share none of their 52 codes. Runs of one cell are independent draws of the row, not paired
samples: the 170,000/0.9 scaled cell has ten draws of the record row across runs 47 and 48, which
read 5 of 5 and 4 of 5 by run.

**What a five-seed cell can say.** Four losses in 58 records is 7% a record; four in the 28
records from 100,000 tokens up is 14%. At 7% five clean draws happen 0.93^5 = 70% of the time, at
14% 46%; two five-seed runs of one cell then disagree on "held every fact" 42% and 50% of the
time, which is what runs 47 and 48 did. A five-seed 53/53 on this row is a draw whose complement
is a whole group's worth of facts, and the table's `facts` column, a mean, cannot show it: 50/53
reads as mild loss and is four seeds at 53 and one at 37.

**Claims qualified by this, each marked in place:** §3l's "reliable retention is the property
nothing else has" (25 draws at 60,000); §3n's "retention survives to 100K and breaks after" (all
five 100K records carry `UNCOVERED:0`, so the pair could not occur there); §3o's two failed seeds
(the pair); §3r's run 47 table (five draws); §3s, which did not mention the recompact arm's
37/53; `RESULTS.md` runs 41, 42, 43, 44, 46, 47 and 48; `STRATEGIES.md`'s "53/53 in all 25 rows"
and its two "keeps every fact" passages; `REPORT-2026-09-07.md` §1, whose "holds every planted
fact everywhere it has been measured" is now bounded to the report's own date; `README.md`'s
worked example, whose row carries the pair and held, and its flags legend; and `runs/README.md`'s
run 43, 46, 47 and 48 entries.

**Built 13 September, uncommitted.** The table prints a `per-seed facts` block for any row whose
seeds disagree -- `[53 53 37 53 53] of 53` for run 48's record row -- and nothing for rows whose
seeds agree, so the block does not grow for nothing; the `facts` legend entry says why. The
`UNCOVERED` legend entry no longer calls the flag "a cost rather than a loss" unconditionally:
alone it is a cost, beside `RECFALLBACK` it is where every archived loss sits, and the
`RECFALLBACK` entry says the same from its side. Two tests; 609 pass.

**Open.** The archived `RECFALLBACK` counts were produced on records whose billed peak prompt sits
well under the ceiling the fallback is gated on -- run 41 luna fixed seed 1 reads `RECFALLBACK:13`
at a 37,106-token peak against 57,952, and every fixed-payload cell up to 300,000 reads the same
way -- while the source that ran them (`6a9470e3a`) gates it exactly as today's does, and the
stub-driven run of the same cells fires it only over that ceiling. On luna, every record with
`RECFALLBACK` above zero has a final prompt above the 0.6 trigger line but one 2.6% under it
(run 42 seed 4, `RECFALLBACK:14` at 33,881 against 34,771 -- inside the tokenizer's margin), and
every record at zero sits below the line; mini's clean records sit above it at zero, which an
uncounted no-op pass would also give. That fits a gate at the trigger rather than at the
ceiling, but no such code has been found. What those counts measured is not established; the
mechanism above is read off the source and the loss arithmetic, not off an observed pass.

## 3v. The reasoning stamp survives the harness store, proven by a test

Resolved 13 September, the same day the counter fix landed. Commit 434a77da5's last paragraph
left one thing unverified: the stamp is proven on the plain `Agent` path, and every live run
drives `build_live_agent(kind="harness")`, where the recorder stamps `context.result.messages`
*after* the history provider has already stored that response. The stamp only reaches the
replayed message if the store held the same objects rather than copies, and nothing in the
harness's assembly said so -- `ToolApprovalMiddleware`, `MessageInjectionMiddleware` and the
per-service-call persisting middleware all sit between the store and the recorder, any one of
which could have been cloning.

**Settled by `test_the_harness_store_keeps_the_reasoning_stamp_the_replay_is_charged_for` in
`tests/test_live.py`.** A stub client answering as a reasoning model -- a `text_reasoning`
content carrying a 2,728-character base64 `protected_data` blob, plus `reasoning_output_token_count
= 300` in its usage -- drives two turns through `build_live_agent(kind="harness")`, its real
session, and the wrapped tokenizer a live run selects. A probe strategy reads
`included_token_count` on exactly the history the second model call was about to send, which is
the number every threshold label reads. On this fixture the second call charges 417 tokens: the
stamped count. The two numbers it must not be are 117, the stripped count without the stamp (the
~19% under-count the commit warned about -- and note the counter's cached per-message
annotations make that residual permanent once written, not merely a first-pass miss), and 814,
the unwrapped count of the base64 as prompt text (the ~45% over-count the fix replaced). The
after-turn pass charges 388 for `[user, assistant]`, so the stamp is on the stored content
before the second turn starts, and the test asserts that directly as well.

**The test bites.** With the stamp removed from `_live.py` the test fails on the stored content
first (no `REASONING_TOKENS_KEY` on it); restored, it passes. The charge assertions are what the
stored-content check exists to localize -- on this fixture the stamped, stripped and raw counts
are 417, 117 and 814, so no two of them can mask as each other.

**What this resolves.** The stamp is on the message the harness replays, so post-fix luna rows
count the replayed reasoning at what the provider bills for it, and the threshold labels on a
reasoning model mean what they say within the framing residual. **What it does not resolve:**
whether a reasoning model's thresholds now *fire* where labelled -- that is step 3 of the next
live run, and no archived run can answer it because per-call local counts were never recorded.
Luna rows before the fix remain comparable with each other and not with post-fix rows, exactly
as the commit said; nothing in this changes the archive.

**The next run's shape, decided 13 September and executed the same evening as run 49**
(approved at ~$13.9, spent $11.91): the 170K/0.9 cell of runs 47/48, five seeds, strategies
`none,truncation,tool_summary_anchored,user_summary_anchored,
tool_and_user_summary_anchored`, one invocation per `--user-summary-mode` (the flag is
run-level) -- and `--user-trigger-fraction 0.6`, not run 48's 0.8, because at 0.8 with 0.9 fill
the standalone user row crossed only once per seed and the three modes were identical by
construction. Read `USERCOMPACT` off each record before drawing any conclusion about the modes:
only `USERCOMPACT > 1` means the mode had two boundaries to differ on. Three open questions the
run was for: the record strategy's re-force/preserve layers (`REFORCED`/`PRESERVED`, never
measured live), the three user-summary modes at a fraction that actually crosses twice, and
whether the fixed counter puts a reasoning model's thresholds where they are labelled. All three
are answered in §3w; nothing in this section needs re-reading for them.

## 3w. Run 49: the modes measured where they can differ, and the layers measured live

Executed 13 September, `runs/run-49-luna-170k-fill90-usermodes60-*`, written up in `RESULTS.md`
("Run 49") and `runs/README.md`. The three questions §3v sent it for, answered:

**1. The record strategy's re-force/preserve layers, live for the first time.** The standalone
record row read `UNCOVERED:0` and `REFORCED:0` on all fifteen records -- at the corrected counter
the record was asked before the prompt degraded, and it covered every group on the first ask, so
the new layers had nothing to do. They acted only on the composed row, four records of fifteen:
three where the second record covered none of the uncovered groups and layer 2 preserved them
(`UNCOVERED` 4/5/5 = `PRESERVED`), one still pending at the run's end. `dq` 0 everywhere, 53/53 on
every seed of every arm but one (fold `-s3`'s record row, 52/53, no flag -- the record's own text,
not a group shed). **The layers work; the mechanism's frequency on the standalone row is now
"never fired here", which retires §3u's arithmetic as a live risk at this cell while leaving its
offline 60K result standing.** The archived losses (21/53, 44/53 twice, 37/53) did not recur in
fifteen draws.

**2. The three modes, measured where they can differ.** `USERCOMPACT:2` on fourteen of fifteen
standalone records (one 3), counted once each by the post-`8c463f0e7` counter: two genuine
boundaries per seed, where run 48 had one and its arms were identical by construction. The
structural difference is visible -- recompact ends with one standing summary, boundary and fold
with two -- and the cost difference is not: mode gaps 1-5% on `seed$` against seed spreads
15-36%, the cross-cell section refusing every ranking, the fold arm's verdict printing NOT
SUPPORTED. **The fold made zero folds on all fifteen fold records**: at two crossings a standing
summary was never worth its prefix break, so fold ran as boundary mechanically and its apparent
+21% against boundary's +30% is noise. What §3s owed its reader is now paid: the modes are
measured, not paired by construction, and at this cell five seeds cannot rank them.

**3. Thresholds fire where they are labelled.** The record trigger is 0.6 of 167,952 -- 100,771
-- and the record row's billed peaks read 98,490-102,024 on thirteen of fifteen records (two at
109,938 and 113,524, the growth between crossing and the forced call). `RECFALLBACK` zero on all
75 records; the 0.9 gate (151,157) never approached. Pre-fix, luna tripped the same label at
0.61-0.64 of the billed ceiling. **The label is back on the number**, which is the measurement
§3v could not make offline.

Also carried in the run: no probe drew the low value anywhere (every seed 99%+ -- the settled-row
case of §3t), no throttling or reconnects or errors, $11.91 total, and two recompact seeds off
target on fill (-7.0%, -7.9%, the low side as always on luna -- `--assumed-reply-tokens 602`
over-estimates its replies). **Nothing here is a cost result**: the record row's -1%/-6%/-7%
against the control sit inside its own spreads (32%/6%/17%), so no saving is claimed; and the
user row remains dearer than the control on every arm (+21% to +30%).

**Open after this run.** Whether the fold mode can ever repay a break -- a cell with more
crossings (a lower trigger, a fill the band regrows under) or longer summaries is where to look,
and none has been measured. Whether the record row is genuinely cheaper than the control: three
arms now read -1%/-6%/-7% and the boundary arm's -6% is the closest to resolvable (6% spread, and
the fold arm printed NOT SUPPORTED at -7%); more seeds at the boundary mode's operating point is
the cheapest way to answer it. And `--assumed-reply-tokens` still wants re-measuring per model:
every luna seed here seeded low.

## 3x. Run 50: all twenty strategies at 120K/0.8, and the verdict is the control

Executed 13 September on request, `runs/run-50-luna-120k-fill80-all20-*`, written up in
`RESULTS.md` ("Run 50") and `runs/README.md`. Run 47's all-strategies measurement at a smaller
window, on the corrected counter, 100 records, $10.45, four invocations concurrent against the
4M TPM the account allows (mild throttling, 1-21 retries, all re-sent).

**The finding is the control's.** At 0.8 fill of 120K nothing overflows -- the uncompacted
conversation peaks at 91,117 billed, 76% of the window -- and compaction has nothing to buy: every
strategy that keeps the answer intact costs 2-48% more than `none` ($0.0562 a seed), the only
cheaper row deletes to 21/53, and the instrument names **`none`**. First cell in the archive
whose verdict is the control. The natural next cell is 120K at 0.95-1.0 fill, where the control
disqualifies and the comparison inverts; that is where a payoff could first appear on this
window and it has not been measured.

Carried by the run: the record row held 53/53 on all five seeds (`UNCOVERED:0`, `RECFALLBACK`
zero) but its peaks ran +6-10% over the 70,771 trigger label -- a wider gap than run 49's 0-2%
at the same code, unexplained; the composed row fired the re-force chain once and the re-forced
record was `TRUNCATED:1` -- it hit the 2,048-token cap and covered none of the five groups, so
layer 2 preserved them at 53/53, which points the flags at the cap rather than the design;
`user_summary_anchored` crossed once per seed at the default 0.8 trigger (the §3s single-crossing
caveat applies to that row here); and the probe-phase cold draw showed a 0% value on this cell's
`token_budget` rows where 100K-up cells drew 33.3% -- §3t's instrument, a new value at a new
size, nothing more.

`--assumed-reply-tokens` 602 over-estimates luna at every cell tried (fill -5.5% here, -0.2 to
-7.9% in run 49): re-measuring it per model is the cheapest remaining calibration, and until
then every luna cell sits below its labelled fill.
