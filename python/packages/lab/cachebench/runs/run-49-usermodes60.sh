#!/usr/bin/env bash
# Run 49: the three user-summary modes at a trigger that crosses twice
# (170,000 tokens, 0.9 fill, scaled payload), same cell as runs 47 and 48 so the
# arms are comparable with them.
#
# --user-trigger-fraction 0.6, not run 48's 0.8: at 0.8 with 0.9 fill the standalone
# user row crossed only once per seed, so the three modes were identical by construction
# (runs/README.md, run 48). 0.6 crossed twice on fourteen of fifteen records and three
# times on the fifteenth, so the modes had two boundaries to differ on.
#
# The full five-strategy shape was approved and run on 13 September: tool_summary_anchored
# rides in every arm even though it does not read the mode, so each arm carries its own
# reference record row. Every arm has its own `none` (vs none$ is within an invocation) and
# its own `truncation` (none + one strategy falsely trips CONTROL DIVERGED, STATE.md 3h).
#
# Executed as a canary arm (s0 recompact) plus two workers, at most two invocations live at
# once; this file is the sequential equivalent and takes the seed indices as arguments.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
STRATEGIES="none,truncation,tool_summary_anchored,user_summary_anchored,tool_and_user_summary_anchored"
run_arm() {
  local seed="$1" mode="$2"
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness --strategies "$STRATEGIES" \
    --user-summary-mode "$mode" --user-trigger-fraction 0.6 \
    --summarizer-provider foundry:gpt-5.6-luna \
    --repeats 1 --seed-offset "$seed" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.9 --context-window 170000 \
    --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 2048 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run49/s${seed}-${mode}.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    > "/tmp/run49/s${seed}-${mode}.log" 2>&1
  echo "$(date -Is) DONE s${seed} ${mode} exit=$?" >> /tmp/run49/progress.txt
}
for s in "$@"; do
  run_arm "$s" recompact
  run_arm "$s" boundary
  run_arm "$s" fold
done