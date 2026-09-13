# Source data

Verbatim stdout from the live-agent runs analysed in [`../RESULTS.md`](../RESULTS.md), plus
the exact script that produced each one. Nothing here is edited except the Foundry endpoint,
which is replaced with `https://<resource>.services.ai.azure.com/api/projects/<project>`.
The captured output carries a `.txt` extension rather than `.log` only because the
repository's `.gitignore` excludes `*.log`; the contents are unmodified stdout.

The tables in `RESULTS.md` are a selection of these columns. Anything quoted there can be
checked against the file of the same run number here, and the numbers should agree exactly:
they are copied, not recomputed.

| file | run | window | agent | facts | wall clock |
| --- | --- | ---: | --- | ---: | ---: |
| `run-07-60k.*` | 7 | 60,000 | harness | 53 | 54 min |
| `run-08-120k.*` | 8 | 120,000 | harness | 53 | 1 h 34 min |
| `run-09a-400k-aborted.*` | 9a | 400,000 | harness | 53 | 57 min, **failed** |
| `run-09-272k.*` | 9 | 272,000 | harness | 53 | 1 h 45 min |
| `probe-reply-cap.*` | — | 60,000 | harness | 53 | 14 min |
| `run-10-60k-buried.*` | 10 | 60,000 | harness | 53 | part of one sweep |
| `run-11-120k-buried.*` | 11 | 120,000 | harness | 53 | part of one sweep |
| `run-12-272k-buried.*` | 12 | 272,000 | harness | 53 | part of one sweep |
| `run-13-60k-spread.*` | 13 | 60,000 | harness | 53 | 1 h |
| `run-14-272k-spread.*` | 14 | 272,000 | harness | 53 | 1 h 30 min |
| `run-15-272k-anchored-scaled.*` | 15 | 272,000 | harness | 53 | 25 min |
| `run-probe-calibration.*` | — | 60,000 | harness | 53 | 25 min |
| `run-16-60k-final.*` | 16 | 60,000 | harness | 53 | 47 min |
| `run-17-120k-final.*` | 17 | 120,000 | harness | 53 | 1 h 10 min |
| `run-18-272k-final.*` | 18 | 272,000 | harness | 53 | 2 h |
| `run-x-unpinned-void.*` | — | 60,000 | harness | 53 | **void** |
| `run-19-16calls-confounded.*` | 19 | 272,000 | harness | 53 | 2 h |
| `run-20-16calls-controlled.*` | 20 | 272,000 | harness | 53 | 1 h 45 min |
| `run-21-freeze-validation.*` | 21 | 120,000 | harness | 53 | 8 min, **mechanism only** |
| `run-22-freeze-paired-272k-*.txt` | 22 | 272,000 | harness | 53 | 2 h 20 min, two arms |
| `run-23-bimodality-60k.*` | 23 | 60,000 | harness | 53 | 40 min, six invocations |
| `run-24-new-instrument-60k.*` | 24 | 60,000 | harness | 53 | 77 min, instrument validation |
| `run-25-stage1-60k-fill*` | 25 | 60,000 | harness | 53 | old prompt, superseded by 26 |
| `run-26-60k-fill86.*` | 26 | 60,000 | harness | 53 | 60 min, 5 seeds |
| `run-27-120k-fill*` | 27 | 120,000 | harness | 53 | 6 h, 3 fills x 5 seeds |
| `run-28-100k-fill86-post7912.*` | 28 | 100,000 | harness | 53 | 5 seeds, after #7912 |
| `run-29-120k-fill86-post7912.*` | 29 | 120,000 | harness | 53 | 5 seeds, after #7912 |

Run 19 raised the call count to sixteen by dropping the codes per result from eight to three,
so it changed call count, result size and code count together. Run 20 is the honest form: six
code-bearing results unchanged, plus ten code-free asides supplying the extra calls. Read
together with run 18 they separate the two things that break a model-written record.

Runs 16 to 18 are the final configuration under the *old* instrument: five repeats, pinned,
values spread on labelled lines, a 12,000-token reply cap, per-scope scoring and a combined
closing question. They are the three points of the window series, and they are section 4.3 of
`REPORT-GPT-5-4-MINI.md`; that report's current tables are runs 26 to 29.

`run-x-unpinned-void` is kept as evidence, not as data. It is the unpinned matrix whose
uncompacted control varied 102% in cost across five repeats while the strategy row gathered
eight fewer facts than the control, and it is why every later run is pinned.

Runs 10 to 12 are the **buried** arm: codes inline in prose, so the score covers retrieval as
well as preservation. Their controls read 11 of 53, which is the five non-tool facts plus the
one code per result sitting at the start, so their accuracy columns rank nothing. They are
kept because they hold the one result no other arm shows: compaction scoring *above* the
uncompacted control by deleting the noise the codes were hiding in.

`probe-reply-cap` is a 2x2 on the control alone, crossing the retrieval-guidance clause with
the reply cap. It is the evidence that the earlier accuracy instability was truncation rather
than retrieval, and it is why `--answer-max-tokens` now defaults to 4,000.

**The aborted run is kept on purpose.** It exits 1 with no table, because four strategies
died of `context_length_exceeded` and one of them was the control. It is the evidence for the
finding that `gpt-5.4-mini` has a 272,000-token *input* limit rather than the advertised
400,000 window, and that a strategy configured against the advertised number can never fire.
A run that produces no table is not the same as a run that produces nothing.

## Columns the analysis omits

`RESULTS.md` reports `peak tok`, `hit%`, `in`, `cost`, `+-`, `vs none`, `facts`, `lost`,
`ignored` and `correct`. The logs carry four more, and they are worth reading when a row looks
strange:

- **`msgs`** — messages in the final prompt, over the most any call in the run carried.
  Deceptive on its own: a strategy that rewrites content in place removes tokens without
  removing messages, so `msgs` can sit at 53/57 while a third of the tokens are gone.
- **`calls`** — model calls, which exceed turns whenever the agent used a tool. Watch for a
  row with materially more calls than its neighbours: `tool_result` and `selective_tool_call`
  each took an extra call in Run 2, and at that conversation size the extra call cost more
  than the trimming saved.
- **`out`** — output tokens across the run. At 6x the input price these are not a rounding
  error, and they are why the break-even formula, which models input only, consistently
  predicts a slightly larger penalty than the measurement shows.
- **`nofetch`** — facts the agent never gathered, because it never called that tool. Not
  compaction damage. Every row in Runs 7 and 8 shows 0, which is what makes those runs
  comparable; a non-zero value means the row ran a different conversation from the control
  and its accuracy cannot be read against it.
- **`flags`** — `ERR` a failed turn, `S<n>` summarizer failures, `<n>/<n>t` turns completed,
  `NO:<opt>` an option the provider rejected and the harness dropped, `FETCH` a row that
  gathered a different fact set from the control, `MSGS:<±n>` the control ran a conversation
  n messages away from the strategy rows', `NOSPLIT` the row cannot say what its probing cost.
  **An empty flags column is a precondition for reading the row at all.**

## The money columns moved

Every `.txt` in this directory was rendered before the cost axis was split, when `in$`, `cost`,
`+-` and `vs none` all described the whole run — seeding plus twelve probes, each of which
re-sends the entire snapshot. Re-rendering the same `.jsonl` today produces `seed in$`, `seed$`,
`probe$`, `run$`, `seed$+-` and `vs none$`, of which only `run$` is comparable with what the
`.txt` beside it shows: `run$` is the old `cost` unchanged, and the four other money columns
read `?` because these records counted their calls in one total and the halves cannot be
recovered from them. **Every one of these cells also carries `MSGS` on its control**, so its
cost comparisons are withdrawn and it prints no verdict. `REVIEW-2026-09-02-REREAD.md` is the
cell-by-cell reading.

## Noise at the end of each log

Every log ends with one or more `Unclosed client session` lines from `aiohttp`. That is a
teardown warning from the Azure SDK's transport, not a failed run. `EXIT=0` on the last line
is the run's real status.

## Reproducing

The `.sh` files assume `cachebench_live` is on `PATH` (`pip install -e python/packages/lab`)
and that `az login` has been run, since the Foundry provider authenticates with
`DefaultAzureCredential`. Set `FOUNDRY_PROJECT_ENDPOINT` to your own project endpoint first.

Costs are not small. Run 7 was about EUR 8 and Run 8 about EUR 14, both at the rates passed
on the command line; a run that omits `--price-input` will fail rather than guess, because
only OpenRouter pricing is auto-discoverable.

Runs 26 to 29 were taken under the rebuilt instrument: the conversation is seeded to a
share of the tried window, snapshotted, and every closing question asked from that snapshot
rather than appended to it. Their `.jsonl` files are the records themselves -- `cachebench_live
--from-jsonl <file>` rebuilds the table from them, which is how the `.txt` beside each was
produced. Anything earlier used the old design and its `all` column, today's `acc2`, is
inflated: the combined question was asked last, after seven answers had already re-listed the
codes into the context it read.

**Runs 26 and 27 are before upstream #7912, runs 28 and 29 after it.** The branch was rebased
onto upstream `main` between them, so the framework under the lab is not the same in the two
groups; `RESULTS.md` §"Runs 28 and 29" and `REPORT-GPT-5-4-MINI.md` §4.2 report the pair.
**Run 29 is the only cell measured on both sides** — 120,000 tokens at 0.86 fill, against run
27's `run-27-120k-fill086.*`. Run 28 is a 100,000-token cell with no before-arm counterpart,
so it ranks strategies against its own control and says nothing about what the change did.
Run 29 has no `.sh` of its own. `run-28-100k.sh` takes the window and fill as arguments, and
`run-28-100k.sh 120000 0.86` reproduces run 29's cell — same strategies, seeds, payload and
bounds as the header of `run-29-120k-fill86-post7912.txt` records — writing to a different
output directory than the one that run used.

## Raw records

[`raw/`](raw/) holds per-seed records from runs made under the rebuilt instrument, including
the failed ones, with a README of its own saying what each group is evidence for. Runs 26, 28
and 29 are not split out there: their merged `.jsonl` beside the table in this directory is the
whole cell, 30 records, one per strategy-seed.

Either way the records are the measurement: `cachebench_live --from-jsonl` rebuilds any table
from them through the same aggregation the live run uses, so nothing here has to be taken on
trust.

Runs before 24 predate that format and exist only as captured stdout.

Run 47 is all twenty strategies at a 170,000-token window and 0.9 fill with the scaled payload
(`15,125x6 at share 60%`), `gpt-5.6-luna`, five seeds, 100 records, one file per seed, two
invocations at a time -- `run-47-all20.sh`, one seed index per argument. **No throttling, retries
or errors**, $18.22 across the run, and `--from-jsonl` over the five files rebuilds
`run-47-luna-170k-fill90-all20.txt` exactly. It is the first archived cell to carry
`user_summary_anchored` and `tool_and_user_summary_anchored` as they now stand -- the band share
on, the composed row on its shared line -- which is what makes run 47a below evidence rather than
data. Four rows keep 53/53 at `acc1` 100%: the control at 86% of the window, the record row at 50%,
the user row at 71% and the composition at 30%, with the composition's cache hit rate at 76%
against the record row's 95% and the control's 97%. Those three `hit%` figures are the whole run,
seeding and probes together; split by phase (`STATE.md` §3t) the seeding half reads 84% / 92% /
95.5%, and the composition's 76% carries a probe-phase draw of 33.3% on four seeds of five. The
user-half counters in these records are inflated by the double pass fixed in `8c463f0e7`:
`USERCOMPACT:2` on the standalone row is one compaction. **Its cost axis is inside the noise**:
`seed$+-` of 19% to 30% on those four rows, `NOT SUPPORTED` on the verdict, and only the user
row's +32% clears its own spread. Seed 3 (`-s3.jsonl`) seeded -10.2% against the target and prints
`FILL OFF TARGET` rendered alone; the other four sit at -2.1% to -3.9% and the merged cell at
-4.6%. Re-rendered without it the retention and hit rates are unchanged, the snapshots move by a
point, and the cost picture is the same. `RESULTS.md` has the per-seed tables.

Run 47a is the **aborted** first attempt at run 47, kept as evidence rather than as data. All
twenty strategies at a 170,000-token window and 0.9 fill with the scaled payload, stopped after
two of five seeds because both new rows were misbehaving: seed 0 holds 19 rows, seed 1 all 20,
seed 3 only the control. It is the only place two figures quoted in the source have a record
behind them, which is why it is here at all.

What it shows, and what it is void for:

- `user_summary_anchored` ran **unbounded**, at `USERCOMPACT:31` (seed 1, 53.4% hit rate) and
  `USERCOMPACT:30` (seed 2, 61.3%), `USERREPLACED:2` on both, 53 of 53 facts on both. Both counts
  are inflated by the double pass fixed in `8c463f0e7` -- about fifteen passes each -- and both hit
  rates are the whole run: the seeding half alone reads 77.0% and 80.6% against the control's
  95.6%, and the probe half 0% and 17%, because the unbounded strategy fired again on every probe
  (`DRIFT:12`), which is the instrument's phase and not a cost a conversation pays. That is
  the pass-per-turn behaviour `--user-min-band-share` was written to stop, and these records
  predate it: their `user_min_band_share` is absent and reads as `0.0`, which is what those runs
  did rather than what this version defaults to. The row as it now stands has no archived run.
- `tool_and_user_summary_anchored` (seed 2) reads `USERCOMPACT:0`, `RECORDS:1`, a 94.6% hit rate
  and `snap 63%` — the composed row acting as `tool_summary_anchored` under a longer name, with
  its own starvation counter reporting nothing. **This is the defect record**, and the reason the
  two halves now share one trigger judged at the size the pass began with. Every number on that
  row is void for the row as it stands.
- The eighteen older rows ran unchanged code and are ordinary records of that cell, on two seeds
  rather than five.

No `.sh` is kept: the script that produced it is the same one run 47 uses, and re-running it
against this tree produces the repaired rows rather than these.

Run 48 is the three `--user-summary-mode` arms -- `recompact`, `boundary` and `fold` -- on the same
cell as run 47 (170,000 tokens, 0.9 fill, scaled payload, `gpt-5.6-luna`), five seeds an arm, one
invocation per seed and mode, 65 records: `none`, `truncation`, `user_summary_anchored` and
`tool_and_user_summary_anchored` in every arm, plus `tool_summary_anchored` in the recompact arm only,
since it does not read the mode. One file per seed and mode --
`run-48-luna-170k-fill90-usermodes-<mode>-s{0..4}.jsonl` -- and one report per arm, rendered with
`--from-jsonl` over that arm's five files; over all fifteen the report prints the three cells
unlabelled, in the order boundary, fold, recompact. `run-48-usermodes.sh`, one seed index per
argument. $11.42 across the run; two throttled calls, both on the boundary arm's control (`-s0` and
`-s2`, 3 s each), no other retries and no errors; every arm seeded -2.8% to -3.5% against target.

**It is kept as evidence, not as a ranking: run 48 cannot rank the three modes**, for three reasons
the records show directly.

- On the standalone `user_summary_anchored` row every arm made **one persistent crossing per seed**
  -- `USERSUMMARIES:1`, `USERREPLACED:24` and `USERHELD` 26 to 40 on fourteen of the fifteen records
  (the fifteenth, recompact `-s3`, read `USERCOMPACT:3`, `USERREPLACED:11`). At trigger 0.8 and fill
  0.9 the band never regrows to a tenth of the prompt after the first pass, and the three modes only
  differ from the second pass on, so the arms were identical there by construction: seeding-phase
  hit 89.2% / 89.8% / 89.3% (boundary / fold / recompact) and `seed$` within 2%.
- The composed row does fire at 0.6 and reached three standing summaries in the boundary and fold
  arms (`USERSUMMARIES:3`), but its `hit%` -- 72% / 81% / 83% -- is the probe-phase draw of
  `STATE.md` §3t and not the modes: the probe half of every record reads either 33.3% or about
  99.5%, and the composed row drew 33.3% on five, three and two seeds of the three arms. The seeding
  half alone reads 84.1% / 84.8% / 85.1%, one point apart.
- `USERCOMPACT` in every record here is inflated by up to 2x. Until `8c463f0e7` the strategy ran
  twice per crossing -- inside the model call on copies the store never saw, and again on the store
  -- and both passes counted. `USERCOMPACT:2` is one compaction; the composed row's 4 to 8 in the
  boundary and fold arms stands against `USERSUMMARIES` 3 (one fold record 2), and the recompact
  arm's 4 to 9 against a counter that reads 1 in that mode by construction. The two passes also
  sent two different summaries at one position on consecutive calls, in every arm alike -- a cache
  break no mode was designed around, and a second reason the arms could not separate.

The cost axis is inside the noise on every arm: the composed row's `seed$+-` is 83% / 49% / 17%,
the recompact arm's verdict prints `NOT SUPPORTED` (a 59% spread against a 7% gap), and no money
figure here is to be quoted. What a run that could rank the arms would need is in `STATE.md` §3s.

Runs 45 and 46 are one cell in two arms: 300,000 tokens at 0.9 fill, `none` against
`tool_summary_anchored`, with run 45 on the **fixed** payload every earlier cell used and run 46 on
the payload **scaled to the window**, which is now the default. Run 45 is one seed, run 46 five, no
throttling in either. Run 46 holds the project's first resolvable saving with full retention --
three seeds at -21%, -23% and -33% keeping 53/53 -- and two seeds where the coverage gate shut and
the row cost more while losing nine facts.

**Neither cell is ranked.** Both carried only `none` and `tool_summary_anchored`, so the leanest
strategy row is the one adding the record's own messages and the instrument withdraws the cost
comparison with `CONTROL DIVERGED`. It is a false positive -- control peak 133 against strategy 135,
and the record contributes exactly 2 -- but the spread guard is gone, so read the per-seed figures
in `RESULTS.md` and not a cell verdict.

Run 44 is the same shape at a 100,000-token window, five seeds, 90 records, one file per seed,
no throttling. It is the cell that diagnosed the sizing bug the 170K run only exposed: at 47 turns
the shortfall is -8.3% against 80 turns' -7.6%, so the error is constant rather than compounding.
The seeding replies measure 103, 201, 348, 448 and 581 tokens against an assumed 602 -- above
every seed, and varying 5.6x between them. `--assumed-reply-tokens` wants re-measuring per model
before any cell is trusted on its label.

Run 43 is all 18 strategies at a 170,000-token window and 0.9 fill -- the largest and fullest
cell measured -- five seeds, 90 records, one file per seed. Run one or two invocations at a time
rather than five, because a single one runs near 1.3M tokens a minute; **no throttling, retries or
errors**. Its cost axis is sound and its **sizing is not**: the seeds seeded 124,636 to 155,530
tokens against a 153,000 target, 73% to 91% of the window, because 80 turns compounds reply-length
variance the solver cannot predict. Read the extremes of the ordering, not neighbouring rows, and
re-solve the filler before measuring at this size again.

Run 42 is every registered strategy in one cell, `gpt-5.6-luna` at 60,000/0.86, five seeds, 90
records, split one file per seed because the merged file exceeds the repository's 500 KB limit --
`--from-jsonl` takes all five paths and reads them as one body. **Its cost axis is void and its
accuracy axis is sound**: five concurrent invocations of eighteen strategies overran the endpoint
quota, so 18 records throttled including the control in every seed, and the cell also seeded -5.3%
off target. Throttling delays requests without changing what the model is sent, so facts and
`acc1` are unaffected. Read it for the retention ordering across the whole field, not for money.

Run 40 is the repaired coverage check measured at 60K/0.86, three seeds, four rows in one
invocation so `anchored` -- the strategy the record strategy falls back to -- is compared on the
same conversations rather than across runs. Four arms: each model with and without record
repeats. The repairs hold (`UNCOVERED:0`, one record per trigger, no fallback on mini's
no-repeat seeds), and the cost axis still refuses: +9%/-7% on luna, +44%/+98% on mini. Read
`RESULTS.md` for why the snapshot axis and the cost axis disagree there.

**These records are schema 6 and carry neither the strategy settings nor the workload flags**, so
the two arms of each model are indistinguishable inside the files; the cross-cell report correctly
says the settings are not recorded, and the workload heading says `flags not recorded`, rather than
either being inferred from the filenames. Everything in this directory is schema 2 to 6, and all of
it predates both blocks.

Runs 38 and 39 are the strategy repair, three seeds a model at fixed 60K/0.86. Run 38 is
coverage keyed on tool names, which is a **regression** kept as evidence: it held back complete
records on `gpt-5.4-mini`, collapsed its shrink from 20% to 5-6%, and made it lose nine facts on
the one seed where `RECFALLBACK` fired. Run 39 is the repair -- the record protected from the
fallback, coverage re-based on values -- and takes luna from 21/53 at +56% to 53/53 at -2%.

`run-38-luna-record-dumped.txt` is the artefact that broke the case open: luna's actual record,
written by `--dump-record`, holding four lookups and all 32 identifiers at a point where the row
scored 21 of 53. It is the evidence that the record was good and the strategy was eating it.

Run 37 is the prompt A/B, three seeds a model at the same cell with default bounds: a
breadth-first rewrite of `RECALL_VALUES_DESCRIPTION` against the shipped wording. Null on both
models, so the rewrite was reverted and only the evidence is kept. The `.sh` reproduces it against
whatever wording is in the tree, which is now the original -- checking out `af1ac1e59`'s successor
and re-running would produce the shipped arm, not the tested one.

Run 36 is the calibration that corrected run 34/35's `tool_summary_anchored` reading. Fixed
60K/0.86, three seeds an arm, `--record-max-tokens` 4,000 -> 24,000 and `--record-target-tokens`
2,000 -> 8,000. Facts held at 21/53 in every record of both arms, so the cap was never what
limited the record. It carries `truncation` as a third row for two reasons: as the reference the
`vs none$` comparison needs, and because a run of `none,tool_summary_anchored` alone falsely trips
`CONTROL DIVERGED` -- the MSGS check compares the control against the leanest strategy row, and
with only that strategy the leanest row carries the record's own forced calls.

Runs 34 and 35 are the same nine-cell matrix on `gpt-5.6-luna`, five seeds a cell, 390 records.
**Seven of the nine print `FILL OFF TARGET`** (-3.1% to -12.0%), because the filler sizing was
solved against 5.4-mini's reply length and luna writes shorter replies. The cells are internally
sound -- one turn list and one actual fill per cell -- but they sit 4 to 9 points below the fill
their filenames claim, so a cell here is not exactly the same operating point as the run-32/33
cell of the same name. `RESULTS.md` reports the deficit per cell.
