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

At a 40,000-token suffix with twenty turns left, that is about **22% of the prompt**. Removing
less than that costs more than it saves, however sensible the removal looks. Measured: one
strategy removed 263 tokens, dropped its cache hit rate from 92% to 88% — 46,471 extra
full-price tokens — and cost 11% more than not compacting at all.

Two consequences run through everything here. **Decide from position, not content**, so a
decision made on turn 5 is still the same decision on turn 20 and the prefix stabilises after
one invalidation instead of being rewritten every turn. And **prefer shortening to removing**,
so the conversation stays legible and the edit stays local.

---

## `AnchoredCompactionStrategy`

**Mechanism.** Keeps a fixed number of head groups and tail groups verbatim, and shortens the
tool results in the band between them. Each banded result is trimmed to a share of the
ceiling:

```
keep_tokens = max_input_tokens × band_share ÷ tool groups in the band
```

`band_share` defaults to 0.25. Only if shortening leaves the prompt over the ceiling does it
remove whole groups, oldest first. Every decision is a function of a message's position, so it
does not change as the conversation grows.

The head anchor is load-bearing: it holds the system prompt and the opening requirements, which
are what give later values their meaning. The tail is what the model is currently working on.

**When it works.** When tool results are larger than the per-group allowance, which is the
common case in a real agent trace. In the window series it preserved all 53 planted facts at a
272,000-token window, where the allowance is generous enough to keep a useful slice of each
result.

**When it does not.** When results are *smaller* than the allowance, it plans nothing and
returns without acting. With 3,500-token results at a 120,000-token window the allowance is
about 5,900 tokens, so it is inert — measured across ten seeds, its snapshots were no smaller
than the uncompacted control's. It is not performing badly there; it is not performing.

The allowance scales with the window and not with the payload, so the same configuration can be
aggressive at 60,000 tokens and idle at 120,000. Size `band_share` against the results you
actually have.

**Variant: `collapse_assistant_text=False`.** Forbids shedding assistant prose in the
last-resort step. Exists because when the model narrates tool values into its replies, that
prose can be the only surviving copy of a result already shortened — this pairs with the
default to measure what dropping it costs.

---

## `MinimumGainAnchoredCompactionStrategy`

**Mechanism.** `AnchoredCompactionStrategy`, plus a floor. Before mutating anything it prices
the collapse it is about to perform, and declines if the projected saving is below
`min_gain_fraction` of the current prompt. The default is **0.23**, the break-even above
rounded up so the floor is never under its own threshold.

The projection is the plan the collapse then executes, not a mutate-and-roll-back: compaction
marks exclusions by mutating `additional_properties` in place, and a rollback that misses one
field is a silent wrong measurement.

**The floor does not apply when the prompt is already over the ceiling.** There, shortening is
not an optimisation whose saving must beat a cache cost — it is what keeps the conversation
admissible, and declining it would hand the work to whole-group removal.

**When it works.** Exactly where the parent wastes money: at a 60,000-token window with
3,500-token results it declined 28 to 50 collapses per seed and came out about 5% cheaper than
the parent, with the same facts preserved.

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
of four it preserved every planted fact with full marks on both accuracy measures and no
spread across seeds, and it is the only strategy that has ever scored *above* the uncompacted
control on the hardest question — the one asking for every value at once. A compact grouped
record is easier to search than the conversation it replaced.

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

Compaction here buys the ability to continue past the context window. It does not buy a
smaller bill, and at the cache discounts these models offer it is unlikely to.
