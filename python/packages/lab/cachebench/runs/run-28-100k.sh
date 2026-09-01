#!/usr/bin/env bash
# One cell, five seeds, sequential. Concurrency 1 because two in flight throttles at these
# prompt sizes and the backoff waits gave back exactly what the parallelism won.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini-2"
BIN="cachebench_live.exe"
W="$1"; F="$2"; OUT=/tmp/stage7
for s in 0 1 2 3 4; do
  tag="$(printf 'w%s-f%s-s%s' "$W" "${F/./}" "$s")"
  "$BIN" foundry:gpt-5.4-mini-2 \
    --agent harness \
    --strategies none,context_window,truncation,anchored,anchored_min_gain,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill "$F" --context-window "$W" \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 --tool-result-tokens 3500 \
    --narration neutral --fact-placement spread \
    --results-jsonl "$OUT/$tag.jsonl" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 \
    > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) DONE $tag exit=$?" >> "$OUT/progress.txt"
done
echo "$(date -Is) CELL DONE w=$W f=$F" >> "$OUT/progress.txt"
