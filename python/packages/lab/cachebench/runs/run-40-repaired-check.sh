#!/usr/bin/env bash
# Run 40: the repaired coverage check at 60K/0.86, thresholds back to 0.6/0.9.
# 'anchored' is in the row list because tool_summary_anchored falls back to it --
# comparing them across runs is what made run 39 unreadable.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
BIN="cachebench_live"
MODEL="$1"; TAG="$2"; PIN="$3"; PCA="$4"; POU="$5"; ART="$6"; REPEAT="$7"; S0="$8"; S1="$9"; shift 9
export FOUNDRY_MODEL="$MODEL"
for s in $(seq "$S0" "$S1"); do
  "$BIN" "foundry:$MODEL" \
    --agent harness \
    --strategies none,truncation,anchored,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.86 --context-window 60000 $ART $REPEAT \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread --tool-result-tokens 3500 \
    --results-jsonl "/tmp/run40/$TAG-s$s.jsonl" \
    --price-input "$PIN" --price-cached "$PCA" --price-output "$POU" \
    "$@" > "/tmp/run40/$TAG-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-s$s exit=$?" >> /tmp/run40/progress.txt
done
