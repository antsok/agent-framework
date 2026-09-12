#!/usr/bin/env bash
# Run 45: tool_summary_anchored alone against the control at a 300,000-token window, 0.9 fill.
# Tests the prediction from runs 41/43/44: the tool payload is fixed at ~22K, so the
# removable share falls as the window grows while the cache gap widens.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="cachebench_live"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness --strategies none,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.9 --context-window 300000 --tool-result-tokens 3500 \
    --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 2048 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run45/s$s.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    > "/tmp/run45/s$s.log" 2>&1
  echo "$(date -Is) DONE s$s exit=$?" >> /tmp/run45/progress.txt
done
