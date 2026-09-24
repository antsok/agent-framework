#!/usr/bin/env bash
# Run 58: the composed row (32f1a9ae8) at 200,000 tokens, 1.0 fill, --trigger-fraction 0.9,
# compared with none. --fallback-fraction raised to 1.0: at its default of 0.9 the record
# half's give-up line coincides with a 0.9 trigger, so it would give up on the same pass it
# asks for a record and the row would measure anchored.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
OUT="/tmp/run58"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,truncation,tool_and_user_summary_anchored"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna     --agent harness --strategies "$ALL"     --summarizer-provider foundry:gpt-5.6-luna     --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5     --fill 1.0 --context-window 200000 --trigger-fraction 0.9 --fallback-fraction 1.0     --assumed-reply-tokens 384 --max-output-tokens 2048 --answer-max-tokens 12000     --record-max-tokens 2048     --markers-per-tool 8 --tool-turns 6     --narration neutral --fact-placement spread     --results-jsonl "$OUT/s${s}.jsonl"     --price-input 0.20 --price-cached 0.02 --price-output 1.20     > "$OUT/s${s}.log" 2>&1
  echo "$(date -Is) DONE s${s} exit=$?" >> "$OUT/progress.txt"
done
