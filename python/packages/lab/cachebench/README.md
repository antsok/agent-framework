# cachebench — compaction vs. prompt caching

Measures what Agent Framework's compaction strategies cost you in provider prompt-cache
hits, across providers, at mid and large context sizes.

## Why this exists

Provider prompt caches match on **exact prefixes**. Every compaction strategy in
`agent_framework._compaction` works by excluding or rewriting messages *inside* an
existing history. So compaction breaks the cached prefix by construction — the only
questions are how badly, how often, and whether the prompt tokens it saves are worth more
than the cache reads it destroys.

That trade-off is not obvious in either direction:

- Compacting **more** shrinks every prompt but re-breaks the prefix, forcing a full
  re-prefill at full price.
- Compacting **less** keeps the cache warm but sends more tokens, most of them discounted.

There is a cadence that minimises real cost, and it differs per provider because cache
discounts, minimum cacheable sizes, and TTLs differ.

## How it measures

Conversations are **scripted, not live**. Each turn appends fixed request messages,
compaction runs over the history exactly as `CompactionProvider.before_run` would, the
projection goes to the provider, and then a *scripted* reply is appended — the model's real
answer is discarded. That is what lets every provider and every strategy replay a
byte-identical conversation, which is the only way cross-provider numbers mean anything.

Two independent measurement channels:

| Channel | Source | Available on |
|---|---|---|
| **Reported** | `cache_read_input_token_count` in `UsageDetails`, which Agent Framework already normalises across providers | Providers that report it |
| **Local prefix oracle** | This package recomputes how much of each prompt stayed byte-identical to the previous one | Always |

The oracle is the theoretical ceiling: a provider can never serve more cache than the
prefix that survived. Comparing the two gives `real%` — how much of the reusable prefix the
provider actually delivered. On providers that report nothing, the oracle plus latency is
all you get, and the tool says so rather than printing a misleading 0%.

Matching is at **message granularity**: a message that changed at all contributes zero
reusable tokens. The oracle therefore never overstates reuse.

## Provider support

| Provider | Cache reporting | Engages | Notes |
|---|---|---|---|
| `azure` | yes | automatic | 1,024-token minimum, 128-token increments before GPT-5.6. TTL 5–10 min idle, 1 hour absolute. Cache reads discounted ~50%. |
| `openrouter` | yes | automatic | Returns `cached_tokens` and `cache_discount`, via the standard `agent_framework_openai` path. **Pin `OPENROUTER_PROVIDER_ORDER`** — otherwise routing changes upstream between turns and you are measuring the router, not the cache. |
| `mistral` | yes | automatic, **intermittent** | Caches with no `prompt_cache_key` at all. But engagement is erratic — see below — so a single repeat is noise. Cache reads billed at **10%** of input; pass `--cache-read-ratio 0.1`. |
| `foundry` | unknown | — | Depends on the deployed model. |
| `ollama` | **no** | automatic, but invisible | Caching demonstrably happens and is never reported. Judge Ollama by `reuse%` and latency only. |

Ollama measured directly against `ollama.com` on 2026-08-25 with a 6,000-token shared
prefix, across `glm-5.2`, `minimax-m3`, `gpt-oss:120b` and `mistral-large-3:675b`, on both
`/api/chat` and `/v1/chat/completions`: **no cache field on any of them**, and
`prompt_eval_count` stayed pinned at the full prompt size on every call. Yet `glm-5.2` went
4,078 ms cold → 1,157 ms → 1,056 ms warm on that identical prefix. Prefix KV reuse is real
there; the usage payload just never mentions it. Two independent filters would hide it even
if the server did send one: the local daemon reshapes cloud responses (it drops
`prompt_eval_duration`, `load_duration` and `eval_duration` on `:cloud` models), and the
`ollama` SDK's `ChatResponse` is a closed pydantic model that discards unknown fields.
Re-check with `samples/probe_ollama_usage.py`, which bypasses the SDK.

> **Mistral engages caching intermittently.** Measured 2026-08-25 on
> `mistral-large-latest`, three identical 6k-token-prefix calls reported
> `cached_tokens` of 0, 0, 6000 without a key — and 0, 6000, 0 with one. Across two full
> `mid` sweeps the same cell swung from 42.1% to 29.0% hit rate on byte-identical input.
> Treat any single-repeat Mistral number as noise: use `--repeats` and read the spread,
> not the value. `prompt_cache_key` is *not* the switch — it made no measurable
> difference — so no provider sends one unless you pass `--prompt-cache-key`.

Environment variables per provider:

```bash
# azure — direct Azure OpenAI deployment (distinct from the foundry project route)
AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_COMPLETION_MODEL
# foundry — also needs a working DefaultAzureCredential (`az login`, or a managed
# identity when deployed); a project endpoint alone is rejected by the client
FOUNDRY_PROJECT_ENDPOINT, FOUNDRY_MODEL
# openrouter — model must be a real slug, e.g. z-ai/glm-5.2; check /api/v1/models
OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_PROVIDER_ORDER  # e.g. "openai"
# mistral — MISTRAL_MODEL is accepted as a fallback
MISTRAL_API_KEY, MISTRAL_CHAT_MODEL
# ollama (cloud) — model drops the ":cloud" suffix on the direct API
OLLAMA_MODEL, OLLAMA_HOST=https://ollama.com, OLLAMA_API_KEY
```

A provider that fails to construct is skipped with a warning rather than aborting the
sweep, so one missing credential does not cost you every other provider's cells.

## Usage

Validate the matrix and see prompt sizes without spending anything:

```bash
cachebench --dry-run --providers azure --sizes mid,large --strategies none,context_window,truncation
```

A cheap live sweep (the defaults: one provider, `mid`, four strategies, one repeat):

```bash
cachebench --providers azure
```

Compare models on the same provider with `provider:model` — cache behaviour varies by model
family at least as much as it varies by provider:

```bash
cachebench --providers "openrouter:openai/gpt-5.4-mini,openrouter:z-ai/glm-5.2,foundry:gpt-5.6-luna"
```

The full cross-provider comparison:

```bash
cachebench \
  --providers azure,mistral,openrouter,ollama \
  --sizes mid,large \
  --strategies none,context_window,context_window_aggressive,context_window_lazy,truncation,sliding_window,tool_result \
  --repeats 3
```

Results are printed as a table and written to `--out` as per-turn JSONL plus a summary CSV.

### Replayed or live

The four commands above replay a scripted transcript: the assistant's replies are canned and
the model's output is discarded, so every provider receives a byte-identical prompt. That is
what makes their numbers comparable across providers, and it is the right mode for asking how
a *prompt* caches.

`cachebench-live` gives that up deliberately. It drives the same scenario through a real
`Agent` that writes its own replies and calls a real tool, so the history compaction acts on
is the history an agent would actually accumulate:

```bash
cachebench-live openrouter:openai/gpt-5.6-luna
```

Two things only exist in this mode. Replies become history, so a strategy that compacts badly
produces a worse reply, which becomes worse history, which it compacts again — replay cannot
show that compounding. And a turn is no longer one model call: a turn that uses a tool bills
several prompts, each a different size.

The cost is comparability. Two models write different replies, so their histories diverge from
the first turn. **Live numbers compare strategies within one model, never models with each
other.** Use the replay commands for cross-provider work.

#### Seed, snapshot, probe

A live run has three phases, and the split is the measurement design rather than plumbing.

The **seed** phase drives every turn except the closing questions, exactly as an agent in use
would. Only the user-side turn list is shared between strategies; the replies, and so the
histories, diverge from the first turn.

The **snapshot** is a deep copy of the session state taken once seeding ends. Deep because
compaction records its decisions by mutating the messages themselves.

Every closing question is then asked as a **probe**: the snapshot is restored, the question is
put, and the answer is thrown back into nothing. Each per-scope question is asked
`--probe-repeats` times (3 by default) and the one combined question `--combined-repeats`
times (also 3), independently — see [the two accuracy columns](#the-two-accuracy-columns) for
why the counts are separate. No probe's answer can reach another probe's context, no question
is asked from a context an earlier question compacted further, and survival is scored against
the snapshot, which is by construction exactly the context every probe was answered from.

None of the three held when the closing questions were ordinary turns appended to the
conversation. Each answer re-listed codes into the history as assistant text, so `survived`
was scored against a prompt the previous answers had written: the same strategy read 53/53 on
a run that emitted 10,941 output tokens and 18/53 on one that emitted 4,873.

#### Results are durable per seed

A cell is every strategy times `--repeats` seeds, and at realistic sizes it runs for hours.
Its table only exists once all of it has finished, so anything that stops the process in
between used to discard every seed that had already completed — and already been paid for.
The 60,000/0.86 cell ran all fifteen strategy-seeds over three and a half hours, died before
printing, and left nothing.

`--results-jsonl PATH` appends one JSON object per seed, written and closed the moment that
seed is scored. Each line carries the cell parameters, the cost and token components, what
survived and what was lost, every per-probe correctness sample, and any error — everything the
table reads, so nothing has to be re-scored later against a scenario that is salted per seed
and gone with the process.

```bash
cachebench-live foundry --results-jsonl runs/stage1.jsonl ...
cachebench-live --from-jsonl runs/stage1.jsonl
```

`--from-jsonl` rebuilds the table and the verdict from that file and runs nothing, so it needs
no provider. It is the same aggregation the live path uses, over the same records, which is
what makes a recovered cell the cell that was measured rather than a second reading of it.

The file is appended to, never truncated: a sweep can point every cell at one path, and a run
resumed after a crash extends what is already there. Records are grouped back into cells by
what they measured, so one file holds a whole sweep and each cell gets its own table. A cell
that is missing strategies or seeds still renders, and says which of each it holds against
what the run set out to take — every mean in the table is over what is present, and nothing in
the table itself would otherwise distinguish four strategy-seeds from fifteen.

Each seed also prints a one-line summary as it lands: strategy, seed, cost, facts, and both
accuracies.
A row that has stopped preserving anything shows up there hours before the table would.

#### Reading what the model recorded

`tool_summary_anchored` has the model write the facts into a tool result and then drops the
tool groups that result replaced, but only the groups the record demonstrably carries — the
rest are kept, and counted as `UNCOVERED:<n>` in the flags column and as `groups_kept_uncovered`
on each seed's record. "Demonstrably" means the record quotes at least `coverage_share` (0.8 by
default) of the distinctive values the group's results contained: tokens of four characters or
more with a digit in them, split on whitespace and on the punctuation that separates values, and
read from after the last `=` so that `code_1=TL-BA44A9` yields the value rather than the label.
The record is tokenised the same way and a value counts only when it is one of the record's own
tokens. Both halves of that had to be fixed: reading the compound meant a group was covered only
by a record that had copied the harness's label format, and testing `value in record` meant a
group holding `2026` was "covered" by any record mentioning a 2026 date, and then deleted. It
used to mean the record contained the group's
function name, which measured badly on both models — luna writes `extra0 deployment lookup
returned codes: …` and never the function name, and mini's complete records scored `UNCOVERED:4`
while its compaction fell from 20% to 5–6%. A group whose results hold no such values — prose
findings, say — still falls back to the name test, because the value rule has nothing to read
there.

That count says how many groups fell short of the ask, not how far short any one of them fell.
A record carrying seven of a group's eight values passes at the default share and the eighth is
gone with nothing saying so. It describes where the conversation stands at the last pass that
read a record, not every shortfall the run ever had: a group a later record covered and the
strategy then deleted is not still reported as kept.

`--dump-record DIR` writes the full text of each record to `DIR/<strategy>-seed<n>.txt`, one
file per strategy and seed, for reading by eye. It is read back out of the finished
conversation after the run is over, so it adds nothing to any prompt and makes no call: the
tokens, the cache hits and the cost are identical whether or not it is set. Seeds that took no
record produce no file rather than an empty one.

A record that arrives and still leaves the prompt over the ceiling makes the strategy fall back
to shortening tool results, which is counted as `RECFALLBACK:<n>` and read the same way as
`FALLBACK`: part of that row measures the fallback strategy rather than the one it is named
for. It is the quieter of the two, because shortening in place leaves the message count nearly
untouched while the values inside those messages go — and beside `UNCOVERED` it is shortening
exactly the groups the coverage check had just declined to delete. It went uncounted until a
seed flagged `UNCOVERED:4` was measured losing the control's facts three messages shorter and
16,617 tokens lighter. It counts passes where the fallback actually changed something: counting
the attempt made `RECFALLBACK:5` mean anything between five losses and five no-ops, and archived
rows carry exactly that number.

`--max-groups-before-record N` forces a fresh record every N tool-call groups rather than asking
one record to cover everything. Coverage does not scale with how much there is to cover:
gpt-5.6-luna named two groups of six, and raising `--record-max-tokens`, raising
`--record-target-tokens` and rewriting the tool's own guidance each left that unchanged. What
was left was to ask each record for less. Since the strategy keeps whatever a record does not
name, an unbounded ask degrades into compacting almost nothing, and this is what buys the
compaction back. Each record costs an agent turn, so a small bound is not free; off by default,
which leaves the size trigger as the only thing that asks.

#### One record or several

The size trigger now asks more than once. It cannot do that on size alone — the size that fired
it does not go away when a record arrives, because the record is *added* to the conversation and
then preserved, so a trigger reading size would pin every remaining call in the run. What
re-arms it is new material: at least one non-recall tool-call group after the newest record.
A conversation sitting above the trigger with nothing recorded since its last record is settled,
and is left alone.

That matters because a record covers what existed when it was written and nothing after it.
Without repeats, every group gathered later is uncoverable for the rest of the run: the strategy
will not delete what no record carries, so those groups sit in the prompt to the end and the row
reports `UNCOVERED` for work no record was ever asked to account for.

The price is that records accumulate and nothing merges them — an older record is the sole
account of the groups behind *it*, so a merge would rewrite the evidence rather than the bulk.
Every record is preserved: unshrinkable, undroppable, counted against the ceiling in full. So
each one raises a floor under the prompt that no later pass can lower, and `RECORDS:<n>` in the
flags column, stored as `records_in_conversation` on each seed, is what says so. Read it
alongside `REC:<n>`, which saturates at 1 and answers only whether the model ever complied, and
against `FORCED:<n>`, which is how many times a record was asked for: one record per ask is the
mechanism working, and more records than asks is a defect. It was one — every trigger event
wrote two records, because the middleware re-decided on the exit of the call it had just pinned,
where the record it asked for is not yet in the loaded history. Archived rows show it as
`FORCED:2, RECFORCED:1`.

`--no-record-repeats` asks once and no more. Every run up to and including 39 was single-record,
so a cell meant to sit on the same axis as those has to set it. It governs the size trigger
alone: `--max-groups-before-record` is asking for repeats outright and keeps forcing them.

#### Tuning the strategies

Every constructor parameter a sweep would want to vary is a flag, because a knob that is only a
default cannot be measured — the pair `anchored`/`anchored_min_gain` differ in exactly one
setting, `min_gain_fraction`, and it was unreachable, so the pair had only ever been compared at
one value of the thing being tested.

- Anchored family: `--keep-head-groups`, `--keep-tail-groups`, `--band-share`, `--keep-tokens`
  (0 derives the retention from `--band-share` instead of fixing it), `--min-gain-fraction`.
- Record strategy: `--trigger-fraction`, `--fallback-fraction`, `--coverage-share`, plus the
  record flags above. The trigger reaches both halves from one flag, so the ask and the wait
  cannot be set apart.
- Everything else: `--keep-last-groups` (sliding window, summarization target),
  `--keep-last-tool-groups`, `--budget-fraction`.

Ranges are checked by the strategies themselves, and every selected strategy is built before the
run spends anything, so a value out of range fails at the command line rather than on the first
paid call. `--dry-run` builds from the flags it is printing a plan for.

The two record thresholds default to **0.6** and **0.9**, which is what every archived run used.
They were briefly moved to 0.8 and 0.95 on the argument that 0.6 fires at 58% of a 60,000-token
window — an agent turn and a broken cached prefix spent early in a conversation that may never
have needed compacting. That is arithmetic about where the line falls rather than a measured
cost, and no run had used the replacement. The measurements point the other way: the record
degrades with the bulk it must read — 53/53 facts at 8,000-token tool results, 46/53 at 16,000,
18/53 at 25,200 — so a later trigger is a bigger ask and a worse record. The break-even argument
also inverts: an edit repays over the turns that follow it, and firing later leaves fewer of
them. The give-up line moves with the trigger because the record arrives one call late by
construction — the middleware can only read the history on the way out of a call and can only
pin the next one — and at 0.6/0.9 the gap it has to land in is three tenths of the budget.
`fallback_fraction` must exceed `trigger_fraction`; the strategy raises `ValueError` when it
does not.

#### Fill and the tried limit

`--context-window` is the limit the run stands in for. It is simulated — the model itself
accepts far more — so it is enforced here: any call whose prompt exceeds it disqualifies that
row, and a cell that disqualifies at all leaves the ranking rather than being starred. Before
this, the 60,000 control ran at 78,003 tokens and was ranked anyway, which made every
"cheaper than not compacting" at that size a comparison with a baseline no model that size
could have produced.

`--fill` is the share of that limit the seeded conversation is sized to reach, solved
analytically from the payload and filler sizes rather than by running one strategy and
adjusting. The filler is the dial; the payload — how many tool results, how large, how many
codes each — is a run-level parameter, varied between runs and compared across them, never
inside one matrix. So a fill fraction means "how much irrelevant context surrounds a fixed set
of facts", and a payload that will not fit inside the smallest cell is refused with an error
rather than quietly overshooting. The achieved fill is measured on the uncompacted run and
flagged if it lands more than 5% from the target.

`--tool-share` states the payload as a share of that target instead of an absolute size, and
derives the size of each tool result from it. Absolute is right within one window and wrong
across two: 3,500-token results are 6% of a 60,000-token context and 3% of a 120,000-token
one, so a sweep over window sizes is a sweep over two variables. It disabled a strategy that
way — `AnchoredCompactionStrategy` shortens each banded result to a share of the *ceiling*, so
its allowance grew from about 2,900 tokens to about 5,900 while the results stayed at 3,500,
and at 120,000 it planned nothing at all while its rows were read as measurements. A share
holds the proportions, so two window sizes are one cell at two scales.

It wins when `--tool-result-tokens` is also given, needs `--fill` since it is a share of that
target, and `0` restores the stated size the same way `--fill 0` restores manual sizing. The
share covers *every* tool result including the code-free ones `--filler-tool-turns` adds, so
turning those on divides one budget over more results rather than adding to it — which is
deliberate: what the strategies act on is tool groups, bearing or not. A share the rest of the
conversation cannot fit inside is refused with an error naming what to move, and the achieved
share is reported and flagged beside the achieved fill, to the same 5% tolerance.

#### The two accuracy columns

One run is scored twice, and the columns say so: **`acc1`** is the scoped questions —
requirements plus one per tool lookup, seven of them in the cells recorded so far — each reply
scored only against the values its own question asked for; **`acc2`** is the one combined
question, which asks for all 53 values at once from a context they are scattered through.
They were `acc` and `all`, which named the questions rather than the measures and left nothing
in the table saying the two were the same run read two ways.

The two need different numbers of attempts to be equally settled. One `acc1` reading averages
seven answers; one `acc2` reading is a single answer. So the combined question has its own
`--combined-repeats` (3 by default), independent of `--probe-repeats` — which the runs that
matter set to 1, the per-scope repeat spread having measured 0 to 2 points while the
between-seed spread ran to 78. Probes are nearly all cache reads, so the two extra attempts
cost 6-9% of a seed measured against the recorded 60,000/0.86 cell: EUR 0.27-0.36 on a cell
that cost EUR 4.58. `acc2` is the mean over every combined attempt of every seed, so a file
merged from runs that asked it once and runs that asked it three times weights each answer
once rather than each seed once.

#### Accuracy is a distribution

Three variance sources used to arrive as one number. They are now reported apart:

- `seed+-` is the spread between seeds — different conversations, so this is compaction's own
  reliability: whether it cleared a retention boundary this time and not last time.
- `rep+-` is the spread between `acc1` repeats *within* one seed — identical facts in
  identical positions, so this is the model's willingness to enumerate and nothing else.
- `rep2+-` is the same within-seed spread for `acc2`, over its own attempts. At
  `--probe-repeats 1` it is the only within-seed variance the cell measures, since `rep+-` is
  then 0 by construction.

Every sample is printed below the table, `acc1` and `acc2` in their own blocks. A strategy
that scored 52, 52, 52 and 22 while preserving exactly the same 27 facts every time used to
read the same as one that lost different facts each time.

`--agent harness` swaps the plain agent for `create_harness_agent`, which is what production
code actually calls. Its optional providers are switched off, because each one adds tools and
system-prompt text to every measured prompt and would shift the trigger points without saying
anything about compaction.

#### The table is ranked on both axes, and priced on both

Rows used to be ordered by cost ascending, which puts the strategy that threw the conversation
away above the one that kept it: the cheapest row of a cell is reliably the one that destroyed
the most. Rows that retain at least `--min-correctness` of the control's accuracy — the same
relative test the verdict applies, default 0.9 — now come first, cheapest **total** cost first,
and the rest follow below a line naming the threshold, in that same order. The count and the
threshold are printed above the table, so a cell where every row clears reads differently from
one where none does even though neither draws a line. The control is ordered by the same rule
as everything else and is marked with a star in the `acc1` column wherever it lands. `acc1` is
what the ranking, the threshold and the verdict are judged on; `acc2` is reported beside it.

`in$` prices the prompt side alone — uncached plus cached, with output and the summarizer left
out — beside the total. On a clean five-seed cell the control's total cost varied 38% while its
input tokens varied 13% and its hit rate 4 points: the whole gap was output, priced at $3.96/M
against $0.07/M for a cache read. That variance is the model's verbosity rather than anything
compaction did, and it swamps the axis compaction acts on — three of five rows differed from
the control by less than the control's own spread. So `in$` is the low-variance view of what a
strategy changed, and `cost` remains the number that is actually billed and the one the rows
are ranked on. It is derived from tokens and rates the records already carry, so every results
file already on disk gains the column.

### The token_budget family

Every other strategy decides *when* to compact from its own trigger, so different strategies
leave prompts of different sizes and comparing them confounds "trimmed harder" with "trimmed
smarter". The `token_budget_*` variants all compact to one shared ceiling
(`--budget-fraction`, default half the input budget) and differ only in the order they delete
things, which holds size fixed and isolates the choice of what to discard:

| variant | order |
| --- | --- |
| `token_budget_fallback` | nothing — pure oldest-first eviction, the floor the others must beat |
| `token_budget_tools_first` | tool results, then tool-call groups, then age |
| `token_budget_truncate_first` | age, then tool results — the mirror, to isolate ordering |
| `token_budget_window_first` | a hard recency window, then age |
| `token_budget_summarize` | tool results, then summarize the rest instead of dropping it |

### Summarization is priced, and its failures are counted

`SummarizationStrategy` calls a model of its own, so it needs `--summarizer-provider`. Those
calls never reach the agent's middleware, and charging them at zero would score the one
strategy that spends money to preserve information as though preserving it were free. They are
metered and added to its cost.

Prefer the same model as the one under test: summarizer tokens are priced at the tested model's
rates, so a cheaper summarizer would be billed at the wrong price.

The strategy also catches its own errors, logs a warning and returns `False`. A broken
summarizer therefore produces a run that never compacted — and so scores *perfect* recall. Its
failures are counted and flagged in the table (`S<n>`); a row carrying that flag is not
evidence that summarization preserves anything.

### Controlling spend

Output is capped at `--response-max-tokens 16` because answers are discarded — you are only
paying for prompts. Cost scales with `sizes` × `strategies` × `providers` × `repeats`, and
`--dry-run` reports exactly how many prompt tokens a live run would send. Start there.

`mid` is ~20 turns and ~50 messages; `large` is ~100 turns and ~270 messages, and costs
roughly 20× more per cell.

## Reading the output

A real `mid` sweep on `mistral-large-latest`, two repeats, cache reads priced at 10%:

```text
provider  size  strategy        turns  sent_tok  in_tok  cached  hit%  reuse%  real%  breaks  eff_in@0.1
mistral   mid   none            20     93,786    48,723  20,512  42.1  91.3    46.1   0       30,262
mistral   mid   none            20     93,786    48,723  14,112  29.0  91.3    31.7   0       36,022
mistral   mid   truncation      20     42,106    22,721   7,392  32.5  79.2    41.1   5       16,068
mistral   mid   truncation      20     42,106    22,721   6,560  28.9  79.2    36.5   5       16,817
mistral   mid   context_window  20     31,668    17,419   6,720  38.6  78.0    49.5  12       11,371
mistral   mid   context_window  20     31,668    17,419   5,488  31.5  78.0    40.4  12       12,480
```

Two things to read off it. First, `in_tok` is **identical across repeats** for each
strategy — that is the byte-identical replay working, and it is what makes the varying
`cached` column attributable to the provider rather than to the harness. Second, on this
provider compaction wins decisively on cost: `context_window` lands at roughly a third of
the baseline's effective input despite breaking the prefix 12 times, because Mistral only
realises 30–50% of the reusable prefix anyway. The lost discount is smaller than the saved
tokens. On a provider that reliably realises ~100%, that arithmetic can invert — which is
the whole reason to measure per provider rather than reason about it.

- `sent_tok` — total prompt tokens the strategy actually sent across the session.
- `reuse%` — share left byte-identical to the previous prompt. **The cache ceiling.**
- `hit%` — what the provider actually served from cache.
- `real%` — `hit%` ÷ `reuse%`, a quotient of two fractions. Below 100% means misses
  compaction does *not* explain: eviction, TTL expiry, minimum-size floors, intermittent
  engagement, or (on OpenRouter) upstream re-routing. It is deliberately *not*
  `cached ÷ reusable_tokens`: those totals use different tokenizers (provider vs. local
  estimator, which runs ~2× higher), and dividing them directly halves the answer.
- `breaks` — turns where the prompt was not a pure extension of the previous one. Each one
  is a forced re-prefill.
- `no_in` — turns that reported cached tokens but no input count. Some providers drop
  `input_token_count` on a cache hit; when this is non-zero, `hit%` is suppressed rather
  than divided by a denominator the provider never sent.
- `eff_in@0.25` — fresh tokens plus cached tokens priced at `--cache-read-ratio`. Set this
  to your provider's actual cache-read discount to compare strategies on real cost.

The baseline to compare against is always `none`: it sends the most tokens but breaks the
prefix zero times.

## Experimental controls

These matter, and the tool enforces them:

- **Cache namespace isolation.** Each cell gets a unique salt at the very front of the
  system message, so cells cannot serve each other cache hits.
- **Turn 1 is always a cache write, never a read.** It is included in totals because a real
  session pays for it too.
- **Sequential execution.** Overlapping cells would contend for the same cache and rate
  limits.
- **The system anchor is sized above 1,024 tokens** so that prompts clear the provider
  minimum from turn 1. Otherwise early turns report zero cached tokens for reasons that
  have nothing to do with compaction.
- **Opt-in caching is opted into.** Mistral gets a per-cell `prompt_cache_key` derived from
  the cell salt — stable across the cell's turns, distinct between cells. Without it the
  provider simply never caches and the whole row is a false negative.
- **Cached tokens are clamped to the input count** they are a subset of, and turns that
  report cache reads without an input count are excluded from `hit%`. Both guard against
  ratios above 100% that read as a broken benchmark rather than as upstream inconsistency.
- **Simulated context window.** Budgets default to 60% of a transcript's fully-replayed
  size rather than the model's real window — a 20-turn transcript never approaches 128k, so
  a real window would mean no strategy ever fires. Override with `--context-window`.

## Compaction can shrink prompts out of cacheable range

The most surprising measured result, on `foundry` / `gpt-5.4-mini`:

```
strategy         per-turn input tokens        cached
none             668 → 4,096 (growing)        0 until turn 7, then 1280/1792/2304/…
truncation       667–1,400 (oscillating)      0 on every turn   (9/20 turns below 1,024)
context_window   666–850  (pinned)            0 on every turn  (20/20 turns below 1,024)
```

Compaction did not break the cache here — it shrank prompts **below the provider's minimum
cacheable size**, so caching never engaged at all. The uncompacted control proves the
mechanism is size and not compaction: with no compaction whatsoever, the same model still
reported 0 cached at 1,216 / 1,325 / 1,434 tokens, and only began caching at 1,769. Every
compacted prompt sat below that.

The practical consequence: on a provider with a 1,024-token floor, a strategy aggressive
enough to hold prompts near that floor forfeits caching entirely. Compare `eff_in` rather
than `hit%` before concluding it was worth it, and consider raising the compaction budget
so prompts stay comfortably above the floor.

### Scope: stateless routes only

Every provider here is driven statelessly — the full projected message list goes up on each
turn. `FoundryChatClient` carries no `conversation_id` / `store` / `previous_response_id`,
so the Foundry rows measure the stateless path.

Routes where the **service** owns the conversation (Responses-style APIs, hosted agent
threads) are a different regime and are out of scope. There, the client uploads only a delta
and the service maintains a stable prefix of its own — so local compaction works against it,
rewriting history the service was already caching. Applications that let the service own
context should expect the opposite conclusion from the one this benchmark reaches, and the
useful comparison there is compaction-on versus compaction-off, not strategy versus strategy.

## Caveats

- Token counts used locally come from `CharacterEstimatorTokenizer` (4 chars/token), so
  `sent_tok` and `reuse%` are estimates. Provider-reported `in_tok` and `cached` are exact.
  Ratios between strategies are reliable; absolute local token counts are not.
- Cache TTLs are minutes. A long sweep may see later cells behave differently from earlier
  ones purely through cache pressure. Use `--repeats` and compare variance.
- `real%` above 1.0 is possible and means the provider served cache beyond what the
  message-granularity oracle predicted — usually partial-message token-level matching.

## Development

```bash
cd python/packages/lab
poe test-cachebench
```

Tests are offline; the provider call is stubbed.

### The strategies are a subpackage, not part of the lab

`agent_framework_lab_cachebench/compaction/` holds the strategies written here — `anchored`,
`anchored_no_assistant`, `anchored_min_gain` and `tool_summary_anchored`, together with the
recall tool and the gate that last one cannot work without. They are the thing this benchmark
measures rather than a part of it, and they are meant to leave for a repository of their own,
so the subpackage is kept liftable: nothing in it imports from the benchmark, its tests sit
beside it in `tests/compaction/`, and `tests/compaction/test_boundary.py` walks every module's
imports and fails if either stops being true.

`_strategies.py` stays in the lab. It is the registry that puts ours and the framework's behind
one `--strategies` name each and builds them all from one `StrategyOptions`, which is benchmark
configuration rather than a strategy.

The subpackage depends on `agent_framework._compaction`, which is **private API** and which
upstream PR [#7912](https://github.com/microsoft/agent-framework/pull/7912) has just rewritten.
`compaction/__init__.py` says what that means for whoever extracts it, and what has to be
renamed before anything is published from there.

## The strategies written here

`agent_framework_lab_cachebench/compaction/STRATEGIES.md` explains each one: the
mechanism, the design constraints behind it, and the short evidence for when it works
and when it does not. It lives inside the subpackage because it travels with it.
