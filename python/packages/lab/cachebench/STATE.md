# Working state

Where this work stands, what is unresolved, and what is needed to pick it up again. Written
to survive a break in context. Results live in [`RESULTS.md`](RESULTS.md), the analysis in
[`REPORT-GPT-5-4-MINI.md`](REPORT-GPT-5-4-MINI.md), raw output in [`runs/`](runs/).

## 1. How to run anything

```sh
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
```

Authentication is `DefaultAzureCredential` against an `az login` session. There is **no `.env`
file anywhere in the tree**; the deployment was found by listing the subscription
(`az cognitiveservices account deployment list`). The model is deployed on an AI Services
account in resource group `foundry-rg`, project `proj-maf-extensions`.

Pricing must be passed explicitly — only OpenRouter is auto-discovered:
`--price-input 0.66 --price-cached 0.07 --price-output 3.96` (EUR per million).

The scripts that produced every recorded run are in `runs/*.sh` with the endpoint genericised.
Working copies were under `/tmp/cachebench/`, which is volatile.

## 2. Hard facts about the model, measured not assumed

| | |
| --- | --- |
| Real input limit | **272,000 tokens**. 270,294 accepted, 275,292 refused (HTTP 400, `param: "input"`) |
| Output ceiling | 128,000. The advertised 400,000 is the sum, and the two are independent |
| Cached-read discount | 9.4x (0.66 vs 0.07 per million) |
| Uncompacted cache hit rate | 93-97% at every size tested |
| `tiktoken` `o200k_base` accuracy | within 6 tokens on a 270,294-token prompt |

## 3. The current question, unanswered

**Compaction keeps running during the closing questions.** The eight closing turns go through
the same `agent.run()` loop as every other turn and the strategy is the agent's
`compaction_strategy`, so it fires before every model call including those. Consequences, none
of them controlled:

- The `early` scope is asked first from a fuller context; the **combined question is asked
  last, from the most compacted context of all**.
- Each closing answer lists codes, putting them back into history as assistant text, which
  partly offsets the above in an uncontrolled direction.
- `survived` is computed from the closing prompts, so a fact can be evicted *during* scoring.

Eviction is demonstrably still active at the end: `tool_summary_anchored` ends at 95 messages
against a peak of 123, `truncation` at 82 against 121.

**`--freeze-during-answers` now exists** (`_live.py`: `CompactionSwitch`, `_FreezableStrategy`;
`run_live(freeze_during_answers=...)`). It freezes on the way *into* the first closing turn,
and it pauses `ToolResultRecallMiddleware` as well — a record forced mid-answer would move the
history the answers are scored against, which is the thing being held still. The wrapper is
installed only when the flag is passed, so an ordinary run is byte-identical to every run
already recorded. Suppressed calls surface as a `FROZEN:<n>` note in the flags column.

Validated live in run 21 (120,000, one repeat, EUR 0.93): 16 compaction calls suppressed per
strategy, so the closing turns really were still compacting, and `tool_summary_anchored` still
reaches `REC:1 / RECFORCED:1` before the freeze. That run measures the mechanism only.

**Still to do:** both arms at 272K, which is what actually answers the question. This is a
live candidate for the accuracy spread (39-78 points) that nothing else has explained — if
eviction fires mid-sequence in some repeats and not others, that is exactly the bimodality
observed. Estimated EUR 10.

## 4. The finding the report does not yet state correctly

`REPORT-GPT-5-4-MINI.md` section 1 and finding 1 index the crossover on **tool result size
alone**. Three controlled points now show it is **two variables**:

| bearing results | codes each | per result | asides | facts recalled |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 8 | 25,200 | 0 | 18/53 |
| 6 | 8 | **8,000** | 10 | **36/53** |
| 16 | **3** | 8,000 | 0 | **53/53** |

Shrinking each result 25,200 -> 8,000 with codes held at 8 lifts recall 18 -> 36. Dropping
codes per result 8 -> 3 lifts it 36 -> 53. **Both matter; neither alone explains the
collapse.** The report needs its headline and finding 1 rewritten around that.

The third row is confounded on its own (it changed calls, size and codes together) and is kept
only as the first point of the series. The second row is the honest one: `--filler-tool-turns`
adds code-free lookups so call count varies without varying what must be remembered.

## 5. Strategies built here

| row | file | mechanism |
| --- | --- | --- |
| `anchored` | `_anchored.py` | fixed head and tail verbatim, band between shortened to a share of the ceiling, decisions from position alone so they never change on a later turn |
| `anchored_no_assistant` | same | as above, forbidden from shedding assistant prose |
| `tool_summary_anchored` | `_toolsummary.py` | `ToolResultRecallMiddleware` forces one recall tool call; the strategy drops every tool group in front of the resulting record |

**Not built, and the best next idea:** summarise each tool result *individually* rather than
all in one call. Design notes are in `REPORT-GPT-5-4-MINI.md` section 7. The three-point result
above is the strongest evidence for it — bounding both the size and the code count per
extraction call is exactly what it would do.

## 6. Instrument facts that cost money to learn

Each of these produced a plausible wrong number first.

- **Pinning is per call, not per turn.** A turn's `tool_choice` reaches only its first call;
  the follow-up after a tool result is unconstrained. Verified on the stub: call 1 carries the
  pin, call 2 carries `None`.
- **A tool passed through per-call `options` is never executed.** `FunctionInvocationLayer`
  wraps `ChatMiddlewareLayer` and builds its tool map first, so the tool reaches the model and
  nothing answers its call. Tools must be registered with the harness; only *permission* can be
  controlled per call, hence `RecallGate`.
- **A middleware cannot see the history before `call_next()`.** `context.messages` holds only
  the new turn until the history middleware replaces it during the call.
- **`run_live` forces `store=False`** itself. A caller that forgot it measured the service
  rather than compaction, and every narration mode came back falsely stable.
- **Unpinned runs cannot be measured at this scale.** Five repeats of the control varied 102%
  in cost; three repeats had shown 3%.
- **The dry run under-reported size** until the probe scenario was given `filler_tool_turns`;
  it reported 6 tool groups and 48,000 tokens where the scenario had 16 and 129,000.
- **`--tool-turns` is silently capped** by `--filler-turns`: extra tool groups are placed
  inside filler sections, so 16 requested with the default 6 filler turns yields 9.

## 7. Repository state

- Branch `python-lab-cachebench`, ~40 commits ahead of `origin`, **not pushed**. Pushing needs
  its own go-ahead.
- `dev/` is untracked Git-LFS junk. **Never stage it.**
- 194 tests pass; ruff and the pre-commit hooks are clean.
- `REPORT.md` and `ARTICLE.md` still describe only the six-model cross-provider work and
  predate everything from run 7 onward. They do not mention the 272,000 input limit, the
  crossover, or either new strategy.

## 8. Spend

About EUR 175 total. Roughly EUR 19 remains of the last top-up. A five-repeat matrix of five
strategies costs about EUR 4 at 60K, EUR 8 at 120K, EUR 11 at 272K, and EUR 10 for the
sixteen-call variants.
