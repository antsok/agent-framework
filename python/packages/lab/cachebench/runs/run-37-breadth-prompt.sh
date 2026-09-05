#!/usr/bin/env bash
# Run 37: does the breadth-first record prompt change what the model covers?
# Same cell and bounds as run-34 fixed-60k-f86 and run-32-60k-fill86, default cap/target,
# so the only difference from those baselines is RECALL_VALUES_DESCRIPTION.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
BIN="cachebench_live"
MODEL="$1"; TAG="$2"; PIN="$3"; PCA="$4"; POU="$5"; ART="$6"; S0="$7"; S1="$8"; shift 8
export FOUNDRY_MODEL="$MODEL"
for s in $(seq "$S0" "$S1"); do
  "$BIN" "foundry:$MODEL" \
    --agent harness \
    --strategies none,truncation,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.86 --context-window 60000 $ART \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --tool-result-tokens 3500 \
    --results-jsonl "/tmp/lunaprompt/$TAG-s$s.jsonl" \
    --price-input "$PIN" --price-cached "$PCA" --price-output "$POU" \
    "$@" > "/tmp/lunaprompt/$TAG-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-s$s exit=$?" >> /tmp/lunaprompt/progress.txt
done
