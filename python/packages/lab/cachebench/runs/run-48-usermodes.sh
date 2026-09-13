#!/usr/bin/env bash
# Run 48: the three user-summary modes head to head, same cell as run 47
# (170,000 tokens, 0.9 fill, scaled payload) so the arms are comparable with it.
#
# --user-summary-mode is a run-level flag, not a per-strategy one, so each mode is its own
# invocation. Each arm carries its own `none` because vs none$ is taken within an invocation,
# and its own `truncation` because a cell of none + one strategy falsely trips CONTROL DIVERGED
# (STATE.md 3h). tool_summary_anchored rides in the recompact arm only: it does not read the
# user mode, so three copies would be three identical rows, and one copy keeps the reference
# row that wins on cost inside the run rather than across runs -- which run 47 showed is not
# safe, the control having moved between two runs of the same seed.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
BASE="none,truncation,user_summary_anchored,tool_and_user_summary_anchored"
run_arm() {
  local seed="$1" mode="$2" strategies="$3"
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness --strategies "$strategies" \
    --user-summary-mode "$mode" \
    --summarizer-provider foundry:gpt-5.6-luna \
    --repeats 1 --seed-offset "$seed" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.9 --context-window 170000 \
    --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 2048 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run48/s${seed}-${mode}.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    > "/tmp/run48/s${seed}-${mode}.log" 2>&1
  echo "$(date -Is) DONE s${seed} ${mode} exit=$?" >> /tmp/run48/progress.txt
}
for s in "$@"; do
  run_arm "$s" recompact "$BASE,tool_summary_anchored"
  run_arm "$s" boundary  "$BASE"
  run_arm "$s" fold      "$BASE"
done
