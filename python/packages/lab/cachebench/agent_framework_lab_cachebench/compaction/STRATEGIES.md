# The strategies in this package

Five compaction strategies and one supporting middleware, the fifth of them a composition of two
of the others, written against what the `cachebench` benchmark measured rather than against
intuition. This file says what each one does, why it is shaped that way, and the short version of
when it earns its keep. Full numbers live in the benchmark's `RESULTS.md`; only enough appears
here to make each claim checkable.

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

**It also honours one annotation it cannot produce itself.** A message another strategy has
marked *preserved* (`compaction/_preserve.py`) is skipped by all three of this strategy's
removal paths: the in-place shortening, the tool-group shed and the narration shed. Preserved
is not the same as excluded — the message is still sent and still counted in full — so a
conversation can end a pass over its ceiling with everything removable already gone. That is
the intended outcome: the shed loop stops on "nothing moved" rather than "the ceiling is met",
so it neither loops against a band it may not touch nor reports success on a prompt the
provider will reject. Only the record-then-drop strategy below sets the mark today, and it had
to, because this strategy was measured shredding that record.

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

`min_gain_fraction` is the whole of what separates this row from its parent, and until the
`--min-gain-fraction` flag existed the lab's builder took the constructor default — so every
comparison between the pair had been made at one value of the variable under test. `T`, the
turns still to come, is the term nobody knows at decision time and it divides: ten remaining
turns need 43% of `B` and forty need 17%, so a workload with shorter conversations than the
twenty this default assumes should raise it rather than inherit it.

---

## `ToolResultAnchoredSummarizationCompactionStrategy`

The only design here that carries information forward instead of discarding it, and the only
one that spends model calls to do so.

**Mechanism, in two phases.**

1. `ToolResultRecallMiddleware` watches the conversation size. Past `trigger_fraction` of the
   ceiling — or once `max_groups_before_record` tool groups have piled up since the last
   record, whichever fires first — it pins the next request to a single tool,
   `recall_earlier_tool_results`, whose description asks the model to write down everything
   from earlier tool results that later work could depend on. The tool echoes that text
   straight back.
2. The strategy waits for the resulting tool result to appear, then drops the tool groups in
   front of it **that the record demonstrably carries**. That last clause is the coverage
   check below, and `groups_kept_uncovered` counts what it refused to delete.

**The two thresholds are one decision.** `trigger_fraction` defaults to **0.6** and
`fallback_fraction` to **0.9**, and the constructor refuses a fallback at or below the trigger.
Both were briefly raised to 0.8 and 0.95 on 6 September and reverted the same day: no run had
ever used them, this project's own data say the record degrades with the bulk it must read -- so
a later trigger means a bigger ask and a worse record -- and the argument for firing late was
inverted, since an edit repays over the turns that follow it and firing later leaves fewer.
0.6 was measured firing at 58% of a 60,000-token window: an agent turn and a broken cached prefix
spent early in a conversation that may never have needed compacting, when the break-even only
favours compaction with a long remaining horizon. Raising the trigger forced the give-up line up
with it, because the record arrives one call late by construction — `process` can only read the
history on the way *out* of a call and can only pin the *next* one — so a 0.9 line left a single
turn's growth of room, and one turn carrying a large tool result crossed it. The strategy would
then compact without a record while the record it asked for was still in flight, which is the one
outcome this design exists to avoid. It cannot go to 1.0 either: past that line the fallback still
has to fit the conversation under the ceiling.

**Several records, and what that costs.** The size trigger asks more than once, and it cannot do
that on size alone: the size that fired it does not go away when a record arrives, because the
record is *added* to the conversation and then preserved. A trigger reading size would therefore
pin every remaining call in the run — which is why it used to be gated on there being no record
at all, and it is the regression to watch for. What re-arms it is new material: at least one
non-recall tool-call group after the newest record. `_record_due` is the one place that rule is
written down.

Repeats are **off by default**. They are needed because a record covers what existed when it was
written and nothing after it.
Without them, every group gathered later is uncoverable for the rest of the run, so it sits in the
prompt to the end and the row reports `UNCOVERED` for work no record was ever asked to account
for. The price is accumulation: every record is preserved — unshrinkable, undroppable, counted
against the ceiling in full — and nothing merges them, because an older record is the sole account
of the groups behind *it* and a merge rewrites the evidence rather than the bulk. So each record
raises a floor under the prompt that no later pass can lower, and `records_in_conversation` is what
says so. It is a different question from `records_found`, which saturates at 1 and answers only
whether the model ever complied. `repeat_records=True` turns repeats on; the default is the
single-record behaviour every run up to and including 39 had, and it governs the size trigger
alone, since setting `max_groups_before_record` is asking for repeats outright.

Off is the default because on is conditional and the condition is not knowable from inside the
strategy. `groups_kept_uncovered` — the `UNCOVERED:<n>` flag — is the signal: non-zero means one
record is not covering everything, which is when repeats pay. Where a record is already complete
they can only cost, since every record is preserved and a second one is duplication added as
unshrinkable prompt. Measured in run 40: negative shrink on all three `gpt-5.4-mini` seeds
(-1%, -4%, -2%), and better on every axis on `gpt-5.6-luna`, whose record named two of six groups.

**Coverage is measured in values, not in tool names.** The check that went in first asked
whether the record contained the group's function name, reading `RECALL_VALUES_DESCRIPTION`'s
"grouped by the tool that produced it" as the contract. Models do not write function names.
Luna's record says *"extra0 deployment lookup returned codes: AB-123456, …"* — every identifier
present, `lookup_extra0` nowhere — and gpt-5.4-mini, whose records carry every value from every
group, was scored `UNCOVERED:4` by the same rule while its compaction fell from a 20% reduction
to 5–6% for nothing. A check that penalises the model which complied is a net negative, and
that one was.

The rule is now the clause the description actually leads with, *"Quote verbatim any value that
cannot be reconstructed or guessed"*. For each candidate group the strategy extracts the
distinctive tokens of its tool **results** — whitespace-delimited, punctuation stripped from
both ends, at least four characters, at least one digit — and calls the group covered when the
record quotes at least `coverage_share` of them, case-insensitively. The default is **0.8**:
1.0 is as brittle as the name rule, since one value the model reformatted keeps a whole group,
and much below 0.5 licenses deleting six of eight values because two were quoted.

Two deliberate gaps in that rule, both stated rather than discovered later. It cannot see a
value with no digit in it, so a group whose results yield **no** distinctive tokens falls back
to the old tool-name test — not to "covered", which would let an empty record delete a tool's
prose findings, and not to "uncovered", which would make every prose-only tool permanently
undroppable. And it over-collects ordinary numbers, which makes coverage harder to claim and
keeps more; that is the direction everything here errs in.

**Two fallback paths, and both are counted.** Either way the row is partly measuring
`AnchoredCompactionStrategy` rather than this design, which is why neither is silent.

- **No record ever arrived.** The model may simply not call the tool. Past `fallback_fraction`
  the strategy stops waiting and hands the whole conversation over rather than letting it
  overflow. `fallbacks_used`.
- **A record arrived and did not free enough.** The groups after the record are untouched by
  design and can exceed the ceiling alone, and since the coverage check went in so can the
  groups the record failed to carry. The fallback then runs behind the record and shortens
  whatever is left. `fallbacks_after_record` — which for a while was counted nowhere, so a seed
  reporting `UNCOVERED:4` lost the same facts as the control, three messages shorter and 16,617
  tokens lighter, and carried no flag at all.

**The record is protected from the fallback, and had to be.** The fallback shortens and sheds
tool results; the record *is* a tool result, and nothing in `_anchored.py` had ever heard of
one. It trimmed the record like any other bulk — the 16,617 tokens above are that trim, not a
deletion — which destroys the sole surviving copy of everything phase 2 had just deleted on the
record's authority. The strategy now marks every record it observes, older records included,
through `compaction/_preserve.py`, and the anchored strategy skips preserved messages in each
of its three removal paths. The mark is re-applied on every pass, because compaction runs
against a freshly loaded conversation and the previous pass's annotations are not in it.

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

## `UserTurnAnchoredSummarizationCompactionStrategy`

The mirror of the strategy above on the other half of the conversation, and the only one here
that rewrites the same position twice on purpose.

**Why there is a second half at all.** Everything above acts on tool output, and both of them
say in as many words that user turns are what give surviving values their meaning. That is true
of the first user turn and of the last. It is not true of the seventy in between, and on this
benchmark's own sizing those seventy are most of the prompt. Run 43 — a 170,000-token cell
filled to 90% — divides as **57% user-turn text, 28% assistant replies, 14% tool results**:
87,551 tokens of user turns against 21,978 of tool payload over 72 seeded turns. A strategy
confined to the tool half is working on a seventh of the conversation, which is why the anchored
rows in that run sit at 72% of the window where the uncompacted control sits at 85%.

**Mechanism.** Past `trigger_fraction` of the ceiling it takes the user turns between a fixed
head and a fixed tail, sends them to a summarizer client, and puts the result back in their
place as a single **user** message — the framework's own replace-and-link mechanism, so the
summary carries `_summary_of_message_ids` and `_summary_of_group_ids`, each superseded turn
carries `_summarized_by_summary_id` back, and the originals are excluded with a reason. It reads
nothing but user groups: tool calls, tool results and assistant narration are never annotated and
never excluded, so this row and the tool-side rows measure independent things and can be put in
one table.

The defaults are **1 and 1**, and one at each end is what is load-bearing rather than a round
number. The first turn carries the task and its requirements, which is the labelling truncation
was measured losing — 29 of 53 facts left in a prompt the model could not use. The last is the
live request, and a model answering a summary of the question it was just asked answers a
different question. The turn before that has already been answered and has no such claim.

**It recompacts its own output, and `_anchored.py::_shorten` deliberately does not.** That
method refuses to touch a result already carrying `REMOVAL_MARKER`, because its trigger is the
band's geometry: a result sits in the band from the turn it ages out of the tail until the end
of the run, so "trim what is in the band" fires every single pass and re-trimming would be a
fresh mutation at one position on every turn. Here the trigger is a threshold that the
compaction moves away from — a pass removes tens of thousands of tokens, and the next one cannot
happen until the conversation has grown all of that back — so between two compactions the prefix
is byte-identical on every turn. Measured on the replay below, the default 0.8 fires **once** on
a 72-turn conversation. The two decisions are opposite and neither was taken in ignorance of the
other.

Refusing to recompact is not the neutral option either: the summary is a user message, so a
strategy that would not re-read its own output would have to keep every earlier summary beside
every new one. That is `records_in_conversation`'s accumulation problem exactly — a floor under
the prompt that no later pass can lower — and recompaction is what stops it rising.

A pass whose band holds **only** the previous summary does nothing, because rewriting one
message at one position for no reduction is the thrash `_shorten` refuses. The band must contain
at least one turn that is not a summary, which is the same rule `_record_due` applies to the
recall middleware and for the same reason: the size that fired the trigger does not go away when
the compaction that answered it is already in the prompt.

**The trigger is 0.8 where the record strategy's is 0.6,** and they are separate settings
(`--user-trigger-fraction` against `--trigger-fraction`). The record has to be asked for before
the bulk it must read degrades it — 53 of 53 facts at 8,000-token results, 18 of 53 at 25,200 —
so it wants to fire early. This one asks for prose whose quality is not what the row measures and
pays only in a broken cached prefix, so it wants to fire as late as it still can.

**Expected effect, from an offline replay of run 43's conversation** — the real turns and tool
results, the run's own assumed 602-token replies, and a stub summarizer, so no model was called:

| row | snapshot | `snap%` | passes |
| --- | ---: | ---: | ---: |
| uncompacted control | 158,301 | 93% | — |
| `anchored` (default band share) | 158,301 | 93% | inert at 3,500-token results |
| `user_summary_anchored`, trigger 0.8 | 83,761 | 49% | 1, replacing 60 turns |
| `user_summary_anchored`, trigger 0.6 | 69,635 | 41% | 2 |
| `user_summary_anchored`, trigger 0.4 | 69,635 | 41% | 9 |

The size of the summary barely matters: 500, 1,000 and 2,000 tokens move the result by 0.6
points, because what it replaces is around 75,000. And the floor at 41% is not this strategy
failing — it is the two halves it may not touch, the assistant's replies (~43,000 tokens) plus
the tool payload (~22,000). Reaching below it needs this row *composed with* a tool-side one.
That composition now exists — `ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy`,
below — and has not been run, so this floor is still the last thing measured on the question.

**When it does not work.** On a conversation whose bulk is tool output rather than user text,
which is the shape `tool_summary_anchored` was built for; on short conversations, where the band
between the anchors is a turn or two and the summarizer call costs more than it saves; and
wherever the exact wording of an earlier turn matters, since what replaces it is a paraphrase
written by a different model. Fact retention across the band is deliberately not measured — the
planted facts live in tool results, which this never touches — so a row's `snap%` is what it
answers and its accuracy columns say only that it did not disturb the other half.

**The summarizer is a trust boundary, and a sharper one than the framework's.**
`SummarizationStrategy` carries an indirect-prompt-injection caveat because its output becomes
permanent conversation history; here that output stands in for *the user's own turns*, which is
the half a model treats as instructions. Point it only at a service trusted as much as the
primary model.

---

## `ToolResultAndUserTurnAnchoredSummarizationCompactionStrategy`

The two strategies above, over one conversation. It is the only entry here that composes rather
than compacts: it owns no selection rule and removes nothing itself.

**Why it exists.** Each of its parts is capped by the share of the conversation it is allowed to
touch, and both say so. The record strategy works on the tool half — a seventh of run 43's
prompt — and the user-turn strategy's offline replay bottomed out at 41% of the window, a floor
that is exactly "the assistant replies and the tool payload it may not touch". The section above
names composing the two as the obvious next measurement and does not make it. **That measurement
has still not been made.** Nothing below is a result; it is what the mechanism does, and what a
run of it would be answering.

**Mechanism.** One pass runs the record strategy, re-reads the conversation, and runs the
user-turn strategy. It returns True when either did. The two select disjoint messages —
`group_messages` gives a user message a group of kind `user` and a call-and-result pair a group of
kind `tool_call`, and each part's rule names exactly one kind — so there is no message both could
claim and nothing either could supersede twice.

**Order: the record strategy first.** Three reasons, the first of which is the expensive one.

- *The record has to be asked for early.* Its trigger is 0.6 against the user band's 0.8, and the
  two constants document why they differ: the record degrades with the bulk it is given to read,
  so it fires early, while the user band pays only in a broken cached prefix and fires as late as
  it can. `ToolResultRecallMiddleware` reads the same size the strategy does. A user phase running
  first would shrink the prompt below the line that asks for a record at all, so the row would not
  have a late record — it would have none, and would report a model that never complied.
- *The phase that can remove less goes first.* The record phase is capped at the tool share and
  the user phase at the user share, and on this benchmark's sizing those are a seventh and most of
  the prompt. The other order takes the prompt from above the user line to below the record line
  in one step, every pass.
- *The record strategy's fallback counts group positions.* Running it behind the user phase would
  have it count a band containing this pass's summary message rather than the turns it replaced,
  so the fallback's geometry — and what the `tool_summary_anchored` row means — would differ
  between the composed row and the row it is meant to be read against.

**The two triggers stay separate, which is the opposite of what `TokenBudgetComposedStrategy`
does.** That family gives every variant one ceiling precisely so that size is held fixed and only
the ordering of deletion varies. Here the sizes the two phases reach *are* the measurement, and a
shared trigger would fire both halves at a line neither single row was ever measured at.

**The interference is the token count, and it is made visible rather than removed.** Both parts
begin by reading the included token count and comparing it with their own trigger, so the second
sees a number the first moved. The conversation is re-annotated between the phases — forced, when
the first changed anything, because the record strategy's fallback rewrites tool results in place
and the counts are cached per message. The order above is chosen so the number moves as little as
it can. And `user_passes_starved` — the `USERSTARVED:<n>` flag — counts the passes where the
record phase's own removals took the prompt from above the user phase's line to at or below it.
That is the one reading of `USERCOMPACT:0` that is not about the user half at all, and without it
it is indistinguishable from a band that held nothing.

It is reported rather than prevented. Preventing it means overriding a part's trigger, and then
the composed row's user half fires where no `user_summary_anchored` row fires, which takes the two
rows the composition exists to be compared with and makes them incomparable.

**It has no ceiling and no fallback of its own,** and returning False does not mean the prompt
fits — the same as for both its parts. A third shed step here would put a removal in the composed
row that neither single row can make.

**Counters.** Both parts' counters are readable off the composed object, so one row's flags say
which half did what: `REC`, `RECORDS`, `FALLBACK`, `RECFALLBACK` and `UNCOVERED` are the record
half, `USERCOMPACT`, `USERREPLACED` and `USERSUMMFAIL` the user half, and `USERSTARVED` is the
composition's own.

**The wiring the row needs, and the one that was missing.** The record half is a strategy *and* a
middleware, and the benchmark installs that middleware for the strategy it finds inside whatever
it built rather than for an object of that class. A composition is not an instance of its parts,
so an `isinstance` there leaves a composed row with no middleware and no recall tool: no call is
pinned, no record is written, the strategy waits and falls back, and the row prints `FALLBACK` —
which is what a model that refused to comply looks like. Anything embedding these strategies owes
the same check.

**When it will not help.** Wherever either part is already inert: a conversation with no tool work
for a record to carry, or one short enough that the band between the user anchors is a turn or
two. It also spends both parts' costs — an agent turn for the record and a summarizer call for the
summary — so a cell running it beside the uncompacted control is comparing a row with two extra
call types against a row with none.

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
