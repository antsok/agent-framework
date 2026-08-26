# We measured context compaction against prompt caching. It usually costs more.

*Lessons from building a benchmark for Microsoft Agent Framework, and from getting it wrong
six times first.*

---

## The short version

Compaction — dropping or shrinking old turns so each request is smaller — is treated as an
obvious cost saving. We measured it across six model/route combinations, about 290 real
agent conversations, scoring both what it costs and what the agent forgets.

**On four of six, compaction cost more than not compacting at all.** The settings that kept
the agent's information intact were 34–63% *more* expensive despite sending *fewer* tokens.
The settings that did save money lost 10–15 of 17 planted facts.

The mechanism is prompt caching, and once you see it the result stops being surprising.

## Why shrinking the prompt can raise the bill

Providers charge much less for text they have already seen — commonly **10x less**. That
discount is strict-prefix: it holds only while the beginning of the conversation stays
byte-identical. Change anything early, and everything after it is billed at full price again.

Compaction changes the earlier part. That is what it *is*. So it trades a large discount for a
smaller prompt, and the trade is often bad.

You can write down exactly when it pays:

> **`T₂/T₁ < (1 − h₁(1−d)) / (1 − h₂(1−d))`**
>
> `T` = tokens sent, `h` = cache hit rate, `d` = cached price ÷ input price.

At a 10x discount, dropping from a 91% hit rate to 78% means you must cut **39%** of your
tokens just to break even. Our best information-preserving strategy cut 7%.

**The formula is predictive, and we tested it under control.** OpenRouter happens to serve
`glm-5.2` on two backends at identical input and output prices that differ only in the
cached-read rate: 5x on one, 10x on the other. Same model, same weights. That isolates the
one variable the formula claims is decisive.

| | 5x discount | 10x discount |
| --- | ---: | ---: |
| `tool_result` (keeps every fact) | **+3%** | **+24%** |
| `selective_tool_call` (keeps every fact) | +2% | +22% |
| `truncation` | −35% | −19% |
| `summarization` | −11% | +22% |

Halving the discount moved every strategy against compaction and none the other way. We
recorded the prediction before the second run — from the first run's own tokens and hit
rates the formula projected `tool_result` at +17% and `truncation` at −24%; measured **+24%**
and **−19%**, both inside the run-to-run spread.

So the first thing to check is not your context window, and not which strategy to pick. It is
your provider's cached-read price. On the same model, the same strategy is nearly free at 5x
and clearly expensive at 10x.

## The forgetting is a cliff, not a slope

We planted 17 verifiable facts across a 16-turn conversation — requirements, a
mid-conversation correction, and tool results — then asked a final question that needs all of
them. Exact string matching, no judgement calls.

Every run, on every model, split into two groups with nothing in between:

- Strategies that **delete messages** lost **10–15 of 17 facts**.
- Strategies that **only rewrite tool output**, without deleting messages, lost **none**.

And the discount does not touch this axis at all. Across both glm-5.2 runs, `none` was the
only setting keeping all 17 facts at *either* discount. A cheaper cache makes compaction
affordable, not safe.

There is no "slightly smaller, slightly lossy" setting. And 28% correct is not a marginally
worse answer — it means the agent forgot the correction and is confidently executing the plan
the user cancelled.

Worth noting for anyone tuning thresholds: `ContextWindowCompactionStrategy` at its shipped
0.5/0.8 defaults cost **more** than not compacting in three of four runs, while losing 13 of
17 facts in each.

## Five framework things worth knowing

These cost us real money to discover. None produced an error.

**1. `CompactionProvider(before_strategy=...)` is a no-op under per-service-call history
persistence.** The agent skips `HistoryProvider.before_run`, so the provider only ever sees an
empty context. The before-phase strategy has to travel as the agent's `compaction_strategy`
instead. `create_harness_agent` does this correctly; hand-rolled wiring easily does not, and
it fails silently.

**2. Responses-API clients store history server-side, which makes compaction do nothing.**
`FoundryChatClient` and `OpenAIChatClient` set `STORES_BY_DEFAULT = True`. When they do, MAF
skips history loading entirely — the code comment is explicit that "the service owns loading;
the providers are write-only sinks" — and the agent sends only the new turn. Your compaction
strategy receives one message and compacts nothing.

Our first Foundry run reported a 16-turn conversation as a **one-message prompt on every row**,
with zero facts surviving, a 100% correct answer, and 82,708 billed input tokens. Pass
`store=False` if you want client-side compaction to apply.

**3. The harness has no default context window, and does nothing without one.**
`create_harness_agent(disable_compaction=False)` reads as "compaction is on". But a strategy is
only built if **both** `max_context_window_tokens` and `max_output_tokens` are supplied, and
both default to `None`. Pass neither and you get no compaction, no warning, and no per-model
inference. The `128_000` and `200_000` figures in the codebase are docstring examples.

**4. Half the strategies cannot see tokens at all.** `SlidingWindowStrategy`,
`ToolResultCompactionStrategy` and `SelectiveToolCallCompactionStrategy` count *message
groups*. Hand them a 32,000-token window and they will happily leave a 34,000-token prompt,
because they never look at the number. Only `TruncationStrategy`,
`TokenBudgetComposedStrategy` and `ContextWindowCompactionStrategy` enforce a token target —
and even those are best-effort, since a single oversized tool result cannot be split.

**5. `UsageDetails` is a `TypedDict`, not an object.** `getattr(usage, "input_token_count")`
returns `None` silently, and every call gets scored as free. Use `dict(...)` and `.get()`.

And one that is not MAF's fault but bites anyway: **Gemini's reasoning models reject histories
whose thought signatures have been altered** (`Invalid thought signature`, HTTP 400). That is
precisely what compaction does. We could not measure `gemini-3.7-flash` at all. Client-side
compaction and thought-signing reasoning models are not straightforwardly compatible.

## The part that generalises: how easy it is to measure nothing

Ten measurement rounds. **Six found a fault in the instrument rather than an answer.** Every
one made compaction look better or worse than it was. Not one produced an error or a failing
test.

- Tool-oriented strategies scored **100% correct** — because the scenario had 3 tool groups
  and they retain the last 4, so they evicted nothing. A perfect score for doing nothing.
- One strategy's summarizer cost was added to **every subsequent row**, a flat offset invisible
  in a total, which inverted the ranking of a whole family.
- Tool results were **8x smaller than intended**: the sizing helper took *characters* and we
  passed a token count.
- Facts the agent **never fetched** were counted as destroyed by compaction, so the
  uncompacted control appeared to lose 6 of 17 facts to a strategy that did not exist.
- One model **ignored explicit instructions to call tools**, gathering different facts each
  run and swinging costs 121%.
- And the server-side history issue above, which made every row measure the same thing.

What caught all six was the same habit: **look at the control row first and ask whether it is
physically possible.** An uncompacted run cannot lose information to compaction. A strategy
that changed no tokens cannot have compacted. A 16-turn conversation is not one message. The
test suite was green throughout; arithmetic that refused to reconcile was the only signal.

Three practices we would keep for any benchmark of this kind:

- **Report the spread between repeats, and refuse to rank when the gap between options is
  narrower than the gap between repeats of one option.** Our tool prints `NOT SUPPORTED` and
  withholds the verdict. It fired, correctly, on a run where repeats varied 121%.
- **Separate "the model didn't use it" from "compaction removed it".** Without that split, a
  model that simply omits facts looks identical to a strategy that deleted them.
- **Hold agent behaviour constant.** Pinning tool calls — forcing a specific function per turn
  and closing other turns to tools — halved measurement noise on both models tested.

## What we would tell a team today

**Check your cached-read price before anything else.** At 10x, compaction rarely pays. At 5x it
can pay handsomely.

**Compact to fit a window, not to save money.** In our runs, every setting that fit the target
window scored ≤39%, and every setting scoring above that overflowed it. That is the real
trade: not cost against accuracy, but *fitting* against accuracy.

**If tool output dominates your context — it was ~49% of ours — shrink it at the source.**
Returning less, or storing results outside the conversation and fetching on demand, reduces
size without touching the prefix your discount depends on.

**Measure on your own workload.** The one model that broke the pattern broke it because of a
line in a price list.

---

*The benchmark is `agent_framework_lab_cachebench` in the `python/packages/lab` tree. Full
tables in `RESULTS.md`, method and caveats in `REPORT.md`. It runs against Azure Foundry,
Azure OpenAI, OpenRouter, Mistral and Ollama, in a deterministic replay mode for
cross-provider comparison and a live-agent mode for within-model work.*
