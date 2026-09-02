# Re-read of the ten recorded cells, 2 September 2026

Companion to [`REVIEW-2026-09-02.md`](REVIEW-2026-09-02.md). Its §1a and §1b are now fixed in
the instrument and guarded against. This file re-reads every committed cell through the
repaired instrument and says which conclusions survive.

Every figure below comes from `cachebench_live --from-jsonl runs/<file>` over the records
exactly as committed. **Nothing was re-run.** No provider was called and nothing was spent.

---

## 1. What moved in the instrument

**1a — the control now runs the strategies' conversation.** `IdentifiedHistoryProvider` issues
every stored message an id at the point the history receives it, so `filter_new_messages` no
longer falls back to `(role, serialized contents)` and no longer drops the control's
byte-identical acknowledgements of filler turns. It is wired into both agent kinds, including
the `harness` kind every recorded cell used. `serialize_message` excludes `message_id`, so no
prompt, token count or cache prefix moves; a test asserts that no id reaches any recorded
prompt. Offline, on a stub that answers every call with the same words, the control peaked at
14 messages against a strategy row's 25 before the fix and at 25 after it.

**1a — and a cell that has the defect now says so.** The leanest strategy row is the turn list
at its own size: compaction excludes and rewrites in place, never deletes from stored history,
and what it adds is its own. A control below that row therefore ran a different conversation.
When it does, the control carries `MSGS:<±n>` in the flags column, is excluded from the
ranking, the cost comparison in `vs none$` is withdrawn from every row, and the cell prints
`CONTROL DIVERGED` and no verdict.

**1b — the probe phase is priced apart.** `ProbeOutcome` carries each probe's calls, so the
phases are now counted exactly rather than modelled: `seed$` is the conversation (seeding plus
the strategy's own summarizer calls), `probe$` is the instrument, `run$` is what the run was
billed, and `seed$ + probe$ = run$` by construction. The ranking, the verdict and `vs none$`
are all on `seed$`. Schema moved to **4**; versions 2 and 3 stay readable and their probe
counts read `None`, not zero.

---

## 2. Neither defect can be corrected on the records already taken

### 2a. The control's conversation is the one on the record

**All ten cells carry the defect.** The control is 3 to 28 messages short of the leanest
strategy row on the same turn list:

| cell | control peak | leanest row | gap | control snapshot | `anchored` snapshot | difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60K / 0.86 | 54 | 61 | **−7** | 51,623 | 53,241 † | +3.1% |
| 120K / 0.50 | 60 | 67 | **−7** | 61,124 | 62,364 * | +2.0% |
| 120K / 0.70 | 72 | 91 | **−19** | 82,677 | 85,463 * | +3.4% |
| 120K / 0.86 | 81 | 109 | **−28** | 100,862 | 107,230 * | +6.3% |
| 100K / 0.86, after #7912 | 71 | 91 | **−20** | 84,182 | 89,831 † | +6.7% |
| 120K / 0.86, after #7912 | 82 | 109 | **−27** | 100,278 | 105,672 * | +5.4% |
| 120K / 0.86, share 0.60 | 57 | 67 | **−10** | 103,204 | — | — |
| 120K / 0.86, share 0.80 | 59 | 67 | **−8** | 103,908 | — | — |
| 120K / 0.50, share 0.80 | 52 | 55 | **−3** | 61,938 | — | — |
| 120K / 0.70, share 0.80 | 56 | 61 | **−5** | 85,103 | — | — |

*\* `anchored` planned nothing at all in these four cells — every fact kept, no message
excluded, final prompt equal to the peak — and still ended 2.0% to 6.3% larger than the row it
should have matched exactly. † in these two it removed content and **still** ended larger. On
the four tool-share cells every strategy compacted, so no row sits at the turn list's own size
and the control's deficit there can only be read off the message counts.*

There is no correction. The control's messages are not "missing from the record" — they were
never sent, so no cost, no cache read and no prompt exists for them anywhere. **Every cell
needs re-running to be read on cost.**

**How large the bias is, relative to what was being measured.** The control under-seeded by
2.0–6.7% of its prompt. That is the same order as, and in several cells larger than, the cost
differences the tables report: of the fifty `vs none` figures in §3, **twenty-nine are inside
±7%**. The direction is fixed — the control was too cheap — so every one of them overstates
what compaction costs.

**The fill check was measured on the same deficient row.** `_fill_note` reads the control's
seeded size. Taken instead from an inert strategy row, the achieved fill of the four clean
cells is +3.9%, +1.7%, +3.9% and +2.4% against target where the control reported +1.9%, −1.6%,
−2.3% and −2.8%. Both readings are inside the ±5% tolerance, so **the cells do sit on the axis
they are labelled with** — but the published fill figures are understated, and the tool shares
on the four share cells, whose denominator is the same number, are correspondingly overstated
by roughly the same 2–7%.

### 2b. The probe phase was never counted apart

The committed records store `input_tokens`, `cached_tokens` and `output_tokens` for the whole
seed and nothing per probe. Exactly one probe's prompt is recoverable — `prompt_tokens_final`
is the last call, and on a run that finished the last call is a probe — and the rest are not.
Nine of the ten cells asked **12 probes per seed**; run 26 asked 10.

Twelve probes priced at `prompt_tokens_final` at the row's own hit rate is a **model of the
run, not the run**, and it is the model the review used to produce its −3.5% / +14.3% / +3.2%
column. The repaired instrument declines to publish it: those cells now read `?` in `seed
in$`, `seed$`, `probe$`, `seed$+-` and `vs none$`, carry `NOSPLIT`, print `NO PHASE SPLIT`, and
fall back to ranking on `run$` — with the ranking line saying so.

So **1b is also not correctable retrospectively.** The review's estimate that stripping the
probes turns −14.1% into −3.5% remains an estimate. It is a plausible one and its direction is
not in doubt, but it is not a measurement and this instrument will not print it as one.

### Which cells can be honestly re-read, and which need re-running

**None of the ten can be honestly re-read on cost, on either defect.** 1a voids every
comparison against the control and cannot be undone; 1b cannot be applied because the split was
never recorded. The answer the review anticipated — "some cells are 1b-only" — does not exist:
the two defects have the same remedy, which is to re-run.

What is still readable from these records, unaffected by either defect:

- every **absolute** column: `msgs`, `tok left/peak`, `snap%`, `calls`, `in`, `hit%`, `out`,
  `run$`, `facts`, `lost`, `nofetch`, `ignored`;
- every **strategy row's** `acc1`, `acc2` and both spreads;
- the **strategy-versus-strategy** comparisons, which never involved the control.

Readable with a caveat: `vs none` on `acc1` and `acc2`. What the control lost was its own
replies, not planted values — `facts` reads 53/53 on every control row in all ten cells — so
the denominators are sound. But the control answered from a context 3 to 28 messages shorter
than the rows measured against it, and some of those messages may have restated codes the
model would otherwise have read back. Treat the ratios as approximate rather than
like-for-like.

---

## 3. The ten cells as they now read

`run$` is the old `cost` column unchanged, so these are the same numbers the `.txt` logs and
`RESULTS.md` carry; the ranking order of every cell is also unchanged, because with no phase
split the fallback basis is the same total the old order used. What changed is that the control
is out of the ranking, the verdict is gone, and the percentages below are marked as withdrawn.

**Cost against the control (withdrawn — the control ran a different conversation):**

| cell | `none` | truncation | anchored | anchored_min_gain | tool_summary_anchored | context_window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60K / 0.86 | $0.1642 | +2% | +17% | +12% | +10% | +58% |
| 120K / 0.50 | $0.2076 | −1% | −1% | +3% | +26% | +27% |
| 120K / 0.70 | $0.3171 | +0% | +2% | −2% | +20% | +113% |
| 120K / 0.86 | $0.4069 | −1% | +6% | +3% | +22% | +141% |
| 100K / 0.86, after #7912 | $0.3199 | +1% | +13% | +19% | +49% | +14% |
| 120K / 0.86, after #7912 | $0.4029 | −2% | +4% | +6% | +16% | +14% |
| 120K / 0.86, share 0.60 | $0.3187 | +3% | +7% | +5% | −4% | +26% |
| 120K / 0.86, share 0.80 | $0.3259 | −5% | −3% | +3% | **−14%** | +14% |
| 120K / 0.50, share 0.80 | $0.1990 | −4% | +1% | −4% | +7% | +0% |
| 120K / 0.70, share 0.80 | $0.2607 | +2% | −0% | −1% | **−8%** | +16% |

**`acc1`, which is not withdrawn:**

| cell | `none` | truncation | anchored | anchored_min_gain | tool_summary_anchored | context_window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60K / 0.86 | 100% | 54% | 94% | 96% | 100% | 41% |
| 120K / 0.50 | 97% | 98% | 96% | 99% | 100% | 72% |
| 120K / 0.70 | 97% | 98% | 98% | 96% | 100% | 43% |
| 120K / 0.86 | 98% | 42% | 99% | 94% | 90% | 38% |
| 100K / 0.86, after #7912 | 98% | 41% | 97% | 96% | 100% | 40% |
| 120K / 0.86, after #7912 | 94% | 41% | 95% | 98% | 100% | 54% |
| 120K / 0.86, share 0.60 | 93% | 58% | 96% | 71% | 83% | 87% |
| 120K / 0.86, share 0.80 | 95% | 53% | 83% | 80% | 80% | 72% |
| 120K / 0.50, share 0.80 | 89% | 96% | 84% | 96% | 100% | 90% |
| 120K / 0.70, share 0.80 | 87% | 98% | 81% | 89% | 84% | 76% |

**Verdicts withdrawn.** Each cell used to close with a recommendation. Under the guard none of
them does:

| cell | verdict, now withdrawn | its stated saving |
| --- | --- | ---: |
| 60K / 0.86 | `none` | — |
| 120K / 0.50 | `truncation` | 1.0% |
| 120K / 0.70 | `anchored_min_gain` | 1.9% |
| 120K / 0.86 | `none` | — |
| 100K / 0.86, after #7912 | `none` | — |
| 120K / 0.86, after #7912 | `none` | — |
| 120K / 0.86, share 0.60 | `none` | — |
| 120K / 0.86, share 0.80 | `none` | — |
| 120K / 0.50, share 0.80 | `truncation` | 4.3% |
| 120K / 0.70, share 0.80 | `tool_summary_anchored` | 8.3% |

The three non-`none` recommendations rested on savings of 1.0%, 1.9%, 4.3% and 8.3% against a
control that under-seeded by 2.0–6.7%. Three of the four are inside the bias.

---

## 4. Which conclusions survive

### Withdrawn

- **"Compaction never buys a smaller bill."** This is the report's third headline result and it
  is the one 1a hits hardest. Every row reading at or below the control does so by 0–5 points
  against a control that was 2.0–6.7% too cheap by construction. The claim may well be right —
  the direction of the bias makes compaction look *worse*, so a corrected control can only
  narrow the gap, and rows at +14% to +141% are far outside it — but **the rows that carry the
  claim, the ones within a few points of the control, are exactly the rows the bias covers.**
  Re-run required.
- **"Compaction pays when tool output dominates."** The −14.1% at share 0.80 / fill 0.86 and
  the −8.3% at share 0.80 / fill 0.70 are `run$` figures: seeding plus twelve re-reads of a
  snapshot that `tool_summary_anchored` had cut to 43% and 26% of the window against the
  control's 87% and 71%. The discount is collected twelve times on a phase no deployed agent
  has. The split that would settle it was never recorded. **Withdrawn on both defects.**
  Separately worth noting: at share 0.80 / fill 0.86 that row scored 80% `acc1` against the
  control's 95% — 84%, below the 90% bar — so **the cell's own verdict was already `none`, and
  the −14.1% was quoted from a row that had not qualified.**
- **Every cell's recommendation**, per the table above.

### Standing

- **#7912 halved the shipped default's cost.** `context_window` at 120K / 0.86 reads $0.9821
  before and $0.4579 after, **−53%**, while the control moved $0.4069 → $0.4029, −1%. Both arms
  carry both defects identically and the comparison is between two strategy rows across two
  runs, not against a control. Its recall half stands the same way: 21 of 53 facts and 38%
  `acc1` before, 28 of 53 and 54% after. 1b touches this only through the probe phase, whose
  snapshot grew from 56,378 to 68,256 tokens while the hit rate went 57% → 91%; the direction
  is not obviously either way and the size of the move is far larger than the phase. **The
  strongest surviving result in the project.**
- **The recall findings, in full.** `lost`, `facts`, `nofetch` and `ignored` are absolute
  counts, and `acc1` and `acc2` are properties of each row's own snapshot. `truncation` losing
  23–32 of 53 facts in the fixed-payload cells, `context_window` losing 12–32 of them,
  `anchored` losing 0–2 there and 2–7 in the share cells — none of that goes through the control.
- **`anchored` is inert at 120,000 tokens with 3,500-token results.** Directly visible in the
  records and now *better* evidenced than before: it planned nothing in four cells and its
  snapshot is the honest size of the turn list, which is how the control's deficit was measured
  in the first place.
- **The payload-shape series** (runs 18–20): earlier instrument, no control comparison in the
  claim, unaffected by either defect. Its own caveats in `REPORT-GPT-5-4-MINI.md` §4.3 still
  apply.
- **The 272,000-token input-limit finding**, the break-even algebra, `rep+-` of 0, the data
  hygiene of the ten cells, and the engineering of the recall mechanism — all as the review
  left them.
- **The cells are on the axis they claim.** Corrected for the deficient control, every achieved
  fill stays inside the ±5% tolerance.

### Changed but not withdrawn

- **`vs none` on `acc1` and `acc2`** — read as approximate, per §2. The control kept all 53
  facts in all ten cells, so the ratios are not far wrong, but they are not like-for-like.
- **"Cost variance is cache-miss episodes, not reply length"** (review §6) stands as a
  statement about the seeds, but the paired-difference argument it supports pairs a strategy
  seed against a control seed and inherits 1a. The pairing method is sound; the specific
  resolutions of "1–7 points" are not re-readable.

---

## 5. What re-running costs

The ten cells cost **$97.61** as recorded — $5.75 at 60K up to $15.71 at 120K / 0.86, thirty
seeds each — and re-taking the set is that again at the same rates, before the probe repeats
are reconsidered.

Two things are worth deciding before spending it. `probe$` is now measured, so the probe
repeat counts can be chosen against a number instead of an estimate. And the guard means a cell
whose control diverges will say so at the end of its own run rather than months later, which is
the failure this whole file is about.
