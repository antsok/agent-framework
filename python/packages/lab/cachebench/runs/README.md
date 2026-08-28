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
