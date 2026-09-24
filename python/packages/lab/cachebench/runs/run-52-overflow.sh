#!/usr/bin/env bash
# Run 52: confirmation of 00061004b (the post-record fallback may no longer shorten tool
# groups no record covers) at run 51's exact cell, on the two record rows plus the control
# and truncation that every cell carries. Everything else identical to run 51.
# Run 51: all 20 strategies at 120,000 tokens with the UNCOMPACTED CONVERSATION SIZED PAST THE
# WINDOW. No control in the archive has ever disqualified, so every earlier verdict was taken
# in a cell where not compacting simply worked. Here the control is expected to overflow and
# be disqualified, and the question inverts: which compacting row keeps the run under the
# limit, with the facts, at the least cost.
#
# Same shape as run 50 except: --fill 1.15 (was 0.8; needs the cap lifted to allow
# >1.0), and --assumed-reply-tokens 384 (was 602), luna's measured median seeding reply across
# the control rows of runs 45-50 -- 602 undershot every luna cell by 2-9%.
# Three streams against the endpoint's 7M TPM limit (raised from 4M): ~1.16M TPM each at
# this cell, so ~50% average load and a worst-case burst of ~420K against a ~1.17M
# ten-second window. Run 50 throttled at 80% load, and a throttle wait long enough to
# expire a cached prefix is billed to the strategy as a miss -- so headroom here protects
# the cost measurement, not only the wall clock. Seeds split {0,3} {1,4} {2}.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
OUT="/tmp/run52"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,truncation,tool_summary_anchored,tool_and_user_summary_anchored"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna     --agent harness --strategies "$ALL"     --summarizer-provider foundry:gpt-5.6-luna     --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5     --fill 1.15 --context-window 120000     --assumed-reply-tokens 384 --max-output-tokens 2048 --answer-max-tokens 12000     --record-max-tokens 2048     --markers-per-tool 8 --tool-turns 6     --narration neutral --fact-placement spread     --results-jsonl "$OUT/s${s}.jsonl"     --price-input 0.20 --price-cached 0.02 --price-output 1.20     > "$OUT/s${s}.log" 2>&1
  echo "$(date -Is) DONE s${s} exit=$?" >> "$OUT/progress.txt"
done
