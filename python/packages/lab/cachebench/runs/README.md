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
  gathered a different fact set from the control. **An empty flags column is a precondition
  for reading the row at all.**

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
