#!/usr/bin/env bash
# Run 57: run 56 repeated on 32f1a9ae8, where the composed row's user half waits for the
# record half. Run 56 measured the user half alone because the record was never asked for.
# Same cell: 200,000 tokens, 0.9 fill, --trigger-fraction 0.8.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
OUT="/tmp/run57"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,truncation,tool_and_user_summary_anchored"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna     --agent harness --strategies "$ALL"     --summarizer-provider foundry:gpt-5.6-luna     --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5     --fill 0.9 --context-window 200000 --trigger-fraction 0.8     --assumed-reply-tokens 384 --max-output-tokens 2048 --answer-max-tokens 12000     --record-max-tokens 2048     --markers-per-tool 8 --tool-turns 6     --narration neutral --fact-placement spread     --results-jsonl "$OUT/s${s}.jsonl"     --price-input 0.20 --price-cached 0.02 --price-output 1.20     > "$OUT/s${s}.log" 2>&1
  echo "$(date -Is) DONE s${s} exit=$?" >> "$OUT/progress.txt"
done
