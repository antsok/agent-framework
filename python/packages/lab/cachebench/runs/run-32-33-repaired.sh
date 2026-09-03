#!/usr/bin/env bash
# One cell, five seeds, sequential. Repaired instrument (identified history, seed/probe cost
# split) and repaired strategies (positional retention, floor measured against B).
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini-2"
BIN="cachebench_live.exe"
TAG="$1"; W="$2"; F="$3"; shift 3   # remaining args: payload spec
OUT=/tmp/rerun2
for s in 0 1 2 3 4; do
  "$BIN" foundry:gpt-5.4-mini-2 \
    --agent harness \
    --strategies none,context_window,truncation,anchored,anchored_min_gain,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill "$F" --context-window "$W" \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "$OUT/$TAG-s$s.jsonl" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 \
    "$@" > "$OUT/$TAG-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-s$s exit=$?" >> "$OUT/progress.txt"
done
echo "$(date -Is) CELL DONE $TAG" >> "$OUT/progress.txt"
