#!/usr/bin/env bash
# Run 41: clean baseline on the repaired code. Thresholds 0.6/0.9, repeats off by
# default, output reservation fixed (ordinary calls send --max-output-tokens).
# 'anchored' is a row because tool_summary_anchored falls back to it.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
BIN="cachebench_live"
MODEL="$1"; TAG="$2"; PIN="$3"; PCA="$4"; POU="$5"; shift 5
export FOUNDRY_MODEL="$MODEL"
for s in 0 1 2 3 4; do
  "$BIN" "foundry:$MODEL" \
    --agent harness \
    --strategies none,truncation,anchored,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.86 --context-window 60000 \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run41/$TAG-s$s.jsonl" \
    --price-input "$PIN" --price-cached "$PCA" --price-output "$POU" \
    "$@" > "/tmp/run41/$TAG-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-s$s exit=$?" >> /tmp/run41/progress.txt
done
