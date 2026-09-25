#!/usr/bin/env bash
# Run 60: the long cell again (run 59a's settings) on d60c44e51, where merged and rewritten
# records are assistant messages rather than synthetic tool calls -- 59a crashed the composed
# row at its first merge with Foundry's 400 invalid_payload. 30K window, fill 6.5, 24 tool
# turns, tool share 0.4, trigger 0.7, output and answer reservation 6000.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
OUT="/tmp/run60"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,truncation,tool_and_user_summary_anchored"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna     --agent harness --strategies "$ALL"     --summarizer-provider foundry:gpt-5.6-luna     --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5     --fill 6.5 --context-window 30000 --trigger-fraction 0.7 --tool-share 0.4     --assumed-reply-tokens 384 --max-output-tokens 6000 --answer-max-tokens 6000     --record-max-tokens 2048     --markers-per-tool 8 --tool-turns 24     --narration neutral --fact-placement spread     --results-jsonl "$OUT/s${s}.jsonl"     --price-input 0.20 --price-cached 0.02 --price-output 1.20     > "$OUT/s${s}.log" 2>&1
  echo "$(date -Is) DONE s${s} exit=$?" >> "$OUT/progress.txt"
done
