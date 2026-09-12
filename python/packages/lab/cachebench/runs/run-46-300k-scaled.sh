#!/usr/bin/env bash
# Run 46: the 300,000/0.9 cell again, with the payload SCALED to the window
# (--tool-share now defaults to 0.6). Same window, same fill, same strategies as
# run 45's fixed-payload probe -- one variable changed.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="cachebench_live"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness --strategies none,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.9 --context-window 300000 \
    --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 2048 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run46/s$s.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    > "/tmp/run46/s$s.log" 2>&1
  echo "$(date -Is) DONE s$s exit=$?" >> /tmp/run46/progress.txt
done
