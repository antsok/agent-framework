# Raw records

Per-seed output from every live run made under the rebuilt instrument, including the ones that
failed. The `.jsonl` files are the records themselves, one JSON object per strategy-seed,
written and flushed as each seed was scored. The `.log` files are the stdout beside them.

**Rebuild any table from these**: concatenate the seeds of one cell and read them back.

```sh
cat raw/oldprompt-conc1-*.jsonl > /tmp/cell.jsonl
cachebench_live --from-jsonl /tmp/cell.jsonl
```

Aggregation is the same code path the live run uses, so a rebuilt table is not a
reconstruction — it is the same arithmetic over the same records.

| group | what it is | records | verdict |
| --- | --- | ---: | --- |
| `void-throttled-*` | five seeds per cell run concurrently against a 1M TPM deployment | 100 | **void**, 94 errored |
| `oldprompt-conc2-*` | 60K/0.86, two seeds concurrent | 25 | usable, but throttled |
| `oldprompt-conc1-*` | 60K/0.86, sequential | 25 | clean |
| `run27-*` | the 120,000 cells, stdout only | — | records are in `../run-27-*.jsonl` |

## What each group is evidence for

**`void-throttled`** is why the sweeps run sequentially. Five concurrent invocations of
roughly 100,000-token prompts exceed the deployment's burst window — a 1,000,000 TPM quota is
enforced in sub-windows of about 166,000 tokens — so every call drew HTTP 429. The harness had
no retry for it at the time, so each 429 failed a turn and abandoned its seed. Kept because the
error text is the record of what a rate limit looks like here.

**`oldprompt-conc2` against `oldprompt-conc1`** is the pair that settled concurrency. Same cell,
same code, two seeds in flight against one: 96 rate-limit retries versus **zero**, and a cost
spread of 24-45% versus 5-21%. Backoff makes throttling survivable but not free — waiting lets
cached prefixes go cold, which lands in the column the whole cost axis depends on. Two workers
also bought no wall clock, because the waits gave back what the parallelism won.

Both are the **old recall prompt**, the one fitted to this benchmark's hex codes. Read against
`../run-26-*` for what the generic prompt changed: nothing measurable, which is the result that
lets the earlier numbers stand.

## What is not here

The run that lost seven of twenty-five rows to a dangling function call — a rate-limit retry
re-invoking a partly advanced turn, so the retried request carried a tool call with no result —
was overwritten when the same cell was re-run after the fix. Its evidence survives only in the
commit that fixed it (`2d531bf90`) and in the regression tests that reproduce the provider's
own 400.

Per-seed records for the 120,000 cells are not duplicated here; `../run-27-*.jsonl` holds them
merged per cell, which is the form the tables were built from.
