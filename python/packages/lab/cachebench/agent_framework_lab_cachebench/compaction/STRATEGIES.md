# The strategies in this package

Three compaction strategies and one supporting middleware, written against what the
`cachebench` benchmark measured rather than against intuition. This file says what each one
does, why it is shaped that way, and the short version of when it earns its keep. Full numbers
live in the benchmark's `RESULTS.md`; only enough appears here to make each claim checkable.

## The problem all of them are shaped by

Providers cache prompt prefixes and charge a fraction of the input rate to re-read them — on
the model these were measured against, **9.4× less**. Compaction rewrites the prefix, so every
edit invalidates the cache from the edit point onward and the tokens behind it are re-read at
full price.

That gives a break-even a strategy has to clear. With `p` the input rate, `c` the cached rate,
`R` the tokens removed, `B` the tokens behind the edit and `T` the turns remaining:

```
compaction pays when   R > B(p − c) / (p + T·c)
```

The removed tokens are not re-sent, so they are not re-read at the cached price either.
Writing the first term as `(B − R)(p − c)` rather than `(B − R)p − Bc` drops an `R·c` and
overstates the floor by about 3%.

The right-hand side is a share of `B` and of nothing else. With twenty turns left it is **29%
of the tokens behind the edit** — not of the prompt, which it only resembles when the edit
starts near the front of the conversation. Removing less than that costs more than it saves,
however sensible the removal looks. Measured: at a 60,000-token window the anchored row held a
lower prompt-cache hit rate than the uncompacted control on every one of five seeds, 89–93%
against 91–95%. That gap is the price of editing at all, paid whether or not the edit removed
anything worth having.

Two consequences run through everything here. **Decide from position, not content**, so a
decision made on turn 5 is still the same decision on turn 20 and the prefix stabilises after
one invalidation instead of being rewritten every turn. And **prefer shortening to removing**,
so the conversation stays legible and the edit stays local.

---

## `AnchoredCompactionStrategy`

**Mechanism.** Keeps a fixed number of head groups and tail groups verbatim, and shortens the
tool results in the band between them. Each banded result is trimmed to a share of the ceiling
set by where in the band it sits, counting from the head:

```
keep_tokens = max_input_tokens × band_share ÷ (band position + 1)
```

`band_share` defaults to 0.25. Only if shortening leaves the prompt over the ceiling does it
remove whole groups, oldest first. Every decision is a function of a message's position, so it
does not change as the conversation grows.

The divisor used to be the number of tool groups in the band *at that moment*, and a result is
trimmed once — `_shorten` leaves a result that already carries the marker alone — so retention
froze at whatever the band happened to be wide on the turn the trim fired. At a 120,000-token
window that is 29,488 tokens for a result caught alone in the band against 4,914 for one caught
sixth: a decision that moved with the conversation's current size, which is what this design
exists to avoid.

Reading the divisor off the result's own position instead costs the band its constant bound: it
is now `band_share` of the ceiling times the harmonic number of the band's tool groups — 2.4×
at six groups, 3.6× at twenty — with each further group adding less than the one before. That
is forced rather than chosen. A budget that never trims what the old rule left alone has to be
at least `band_share × ceiling ÷ (position + 1)` at every position, and those terms sum without
limit; the alternative is a bounded band that starts spending cache at windows where everything
already fits.

The head anchor is load-bearing: it holds the system prompt and the opening requirements, which
are what give later values their meaning. The tail is what the model is currently working on.

**When it works.** When tool results are larger than the per-group allowance, which is the
common case in a real agent trace. In the window series it preserved all 53 planted facts at a
272,000-token window, where the allowance is generous enough to keep a useful slice of each
result.

**When it does not.** When results are *smaller* than the allowance, it plans nothing and
returns without acting. With 3,500-token results at a 120,000-token window the smallest
allowance in a six-group band is 4,914 tokens, so it is inert. It is not performing badly
there; it is not performing.

The allowance scales with the window and not with the payload, so the same configuration can be
aggressive at 60,000 tokens and idle at 120,000. `band_share` is now a constructor argument and
a `--band-share` flag, which it was not when that sentence was first written.

**But tuning it does not rescue a small payload, and this was checked rather than assumed.** At
a 117,952-token ceiling with six 3,500-token results in a 103,200-token prompt, lowering the
share does make the strategy act -- and every setting still falls short of the break-even the
edit has to clear:

| `band_share` | oldest keeps | newest keeps | removed | clears ~29,900? |
| ---: | ---: | ---: | ---: | --- |
| 0.25 (default) | 3,500 | 3,500 | 0 | never fires |
| 0.05 | 3,500 | 982 | 8,952 | no |
| 0.01 | 1,179 | 196 | 18,114 | no |

At 0.01 it sheds 94% of every result and still cannot pay. The whole tool payload is 21,000
tokens against a break-even near 30,000, so **this is the regime, not the setting**: idleness is
the correct behaviour on a payload this small, and the dial only lets you buy a loss.

**Variant: `collapse_assistant_text=False`.** Forbids shedding assistant prose in the
last-resort step. Exists because when the model narrates tool values into its replies, that
prose can be the only surviving copy of a result already shortened — this pairs with the
default to measure what dropping it costs.

---

## `MinimumGainAnchoredCompactionStrategy`

**Mechanism.** `AnchoredCompactionStrategy`, plus a floor. Before mutating anything it prices
the collapse it is about to perform, and declines if the projected saving is below
`min_gain_fraction` of the tokens the collapse puts back on the meter — the included prompt
from its earliest rewrite to the end, the `B` above. The default is **0.29**, the break-even
above rounded up so the floor is never under its own threshold.

The base is `B` and not the prompt because everything in front of the earliest rewrite stays
cached and is never paid for again. Charging the collapse for the prompt overstates its cost by
`prompt ÷ B`: invisible for a collapse that starts at the head of the band, and enough to
refuse every incremental collapse of a group that has just aged out of the tail, where the edit
is near the end and `B` is a fraction of the prompt.

The projection is the plan the collapse then executes, not a mutate-and-roll-back: compaction
marks exclusions by mutating `additional_properties` in place, and a rollback that misses one
field is a silent wrong measurement.

**The floor does not apply when the prompt is already over the ceiling.** There, shortening is
not an optimisation whose saving must beat a cache cost — it is what keeps the conversation
admissible, and declining it would hand the work to whole-group removal.

**When it works.** Where the parent wastes money, though the one cell that measured it does not
say so cleanly. At a 60,000-token window with 3,500-token results it declined 28 to 50
collapses a seed. Its cost against the parent, seed by seed, was 0.67, **1.40**, 0.86, 0.96,
0.96: cheaper on four seeds, 40% dearer on the fifth, and the "about 5% cheaper" that the
totals give is one seed carrying the other four. The control's own cost varies by 18% across
those same five seeds, so the difference does not clear the run's noise. Recall moved in both
directions too — on seed 4 the floor kept all 53 planted facts where the parent lost five, and
on seed 2 it kept 42 where the parent kept 48.

Those rows were produced by a floor measured against the whole prompt rather than against `B`
(`REVIEW-2026-09-02.md` §2), so they describe a floor that refused a great deal more than the
break-even asks it to.

**When it does not.** It prevents a specific waste; it does not make compaction pay. Where the
parent is inert the floor is never consulted at all, and both rows behave identically. Nothing
in it makes a compacted run cheaper than an uncompacted one.

`declined_collapses` counts the refusals, so a run that never fired and a run that fired to no
effect can be told apart.

---

## `ToolResultAnchoredSummarizationCompactionStrategy`

The only design here that carries information forward instead of discarding it, and the only
one that spends model calls to do so.

**Mechanism, in two phases.**

1. `ToolResultRecallMiddleware` watches the conversation size. Past `trigger_fraction` of the
   ceiling it pins the next request to a single tool, `recall_earlier_tool_results`, whose
   description asks the model to write down everything from earlier tool results that later
   work could depend on. The tool echoes that text straight back.
2. The strategy waits for the resulting tool result to appear, then drops every tool group in
   front of it. The record is what those groups were reduced to.

If the record never arrives — the model may simply not call it — a `fallback_fraction`
threshold hands over to `AnchoredCompactionStrategy` rather than letting the conversation
overflow while waiting.

**Four design constraints, each of which cost a wrong measurement to learn.**

- **The record must be a tool result the provider issued.** Synthesising a call/result pair
  client-side breaks on routes that track tool calls server-side.
- **The middleware sends no message.** An appended instruction carries no history-provider
  source tag, so under per-service-call persistence it would be stored and would appear in the
  conversation the application replays to its user. The tool's description is the entire
  prompt.
- **The tool cannot be hidden, so it is gated.** A registered tool is advertised on every
  request, and the model called this one uninvited on unpinned follow-ups in every early run.
  `RecallGate` makes it inert unless the middleware armed it; an uninvited call still answers
  honestly, it just records nothing.
- **A cap does not make a model fit a budget — it truncates.** For a tool call the cut lands
  inside the arguments JSON, so the record is lost rather than shortened. Hence two separate
  bounds: `record_max_tokens` on the forced call as a safety limit, and `record_target_tokens`
  stated in the description, which is the only channel that makes the model aim for a size.
  Truncation is detected from the provider's `finish_reason` and counted.

**When it works.** It is the strongest recall result this package has produced. In three cells
of four it preserved every planted fact with full marks on both accuracy measures and no spread
across seeds.

It also read above the uncompacted control on the combined question — the one asking for every
value at once — and that comparison does not survive scrutiny. The combined question never had
its counts spelled out the way the scoped ones did, so 226 of 240 control samples land on
exactly 1.0 or 11/53: the five non-tool facts plus one code per lookup, the model reading *every
code returned by every deployment lookup* as *the* return code of each. Where the expected
answer is a grouped list, a grouped record is being copied straight into it, and the measure is
rewarding the record's format (`REVIEW-2026-09-02.md` §5).

**When it does not.** The record degrades with the bulk it must read and the number of values
it must pull out. Both matter independently: at 25,200-token results it preserved 18 of 53
facts; shrinking them to 8,000 while holding eight codes each recovered only 36; reaching 53
also needed the codes per result to drop to three. At around 96,000 tokens of context it began
to fray — 47 of 53, with a 52-point spread across seeds.

It also costs more than the alternatives, and the premium is its own output: roughly 10 to 26%
above not compacting, with output tokens running about 1.7× the control's. When the context is
not under pressure that is pure overhead.

**The obvious next design, not built.** Summarise each tool result *individually* rather than
all of them in one call. That bounds both quantities the record degrades with — the text one
call must read, and the values it must extract — at the price of more calls. Every measurement
above points at it.

---

## What none of them do

**None of them is cheaper than not compacting while keeping the answers intact.** Across every
cell measured, each row reading below the uncompacted control sits inside the control's own
cost spread, and the rows that are genuinely cheaper are cheaper because they discarded the
conversation.

Read that as a direction and not as a size. Two instrument defects sit under every cost number
here (`REVIEW-2026-09-02.md` §1): the control drops byte-identical acknowledgements that the
annotated rows keep, which makes the control artificially cheap, and `cost` includes twelve
probe re-reads of a frozen snapshot, which flatters whatever compacted hardest. They push
opposite ways on different cells, so no single correction rescues the axis, and the headline
cost comparisons are pending a repaired instrument.

Compaction here buys the ability to continue past the context window. It does not buy a
smaller bill, and at the cache discounts these models offer it is unlikely to.
