#!/usr/bin/env bash
# Run 64: tool_summary_anchored with record repeats on (the new default, d3aca3450), against the control.
# on gpt-5.6-luna and gpt-6-luna. Shape as runs 53-55 (6 tool turns, reply 384, output 2048,
# answer 12000). Usage: run63-stream.sh MODEL "PRICE ARGS" fill:seed [fill:seed ...]
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
MODEL="$1"; PRICES="$2"; shift 2
export FOUNDRY_MODEL="$MODEL"
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
OUT="/tmp/run64/$MODEL"
mkdir -p "$OUT"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,tool_summary_anchored"
for job in "$@"; do
  f="${job%%:*}"; s="${job##*:}"
  "$BIN" "${PROVIDER:-foundry}:$MODEL"     --agent harness --strategies "$ALL"     --summarizer-provider "${PROVIDER:-foundry}:$MODEL"     --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5     --fill "$f" --context-window 120000     --assumed-reply-tokens 384 --max-output-tokens 2048 --answer-max-tokens 12000     --record-max-tokens 2048     --markers-per-tool 8 --tool-turns 6     --narration neutral --fact-placement spread     --results-jsonl "$OUT/f${f}-s${s}.jsonl"     $PRICES ${DRY:-}     > "$OUT/f${f}-s${s}.log" 2>&1
  echo "$(date -Is) DONE $MODEL f$f s$s exit=$?" >> "/tmp/run64/progress.txt"
done
