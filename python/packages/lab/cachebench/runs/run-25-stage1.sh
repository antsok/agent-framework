#!/usr/bin/env bash
# Stage 1: 60K and 120K, fills 0.50/0.70/0.86, 5 strategies, 3 seeds, 3 probe repeats.
# Cheapest cell first so partial results are usable if this is stopped part-way.
# One log per cell, and a failing cell must not take the rest of the run with it.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
BIN="cachebench_live.exe"
cell () {  # window fill
  local w="$1" f="$2" tag
  tag="$(printf 'w%s-f%s' "$w" "${f/./}")"
  echo "$(date -Is) START $tag" >> /tmp/stage1-progress.txt
  "$BIN" foundry:gpt-5.4-mini \
    --agent harness \
    --strategies none,context_window,truncation,anchored,tool_summary_anchored \
    --repeats 3 --probe-repeats 3 \
    --fill "$f" --context-window "$w" \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --markers-per-tool 8 --tool-turns 6 --tool-result-tokens 3500 \
    --narration neutral --fact-placement spread \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 \
    > "/tmp/stage1-$tag.log" 2>&1
  echo "$(date -Is) DONE  $tag exit=$?" >> /tmp/stage1-progress.txt
}
: > /tmp/stage1-progress.txt
cell 60000  0.50
cell 60000  0.70
cell 60000  0.86
cell 120000 0.50
cell 120000 0.70
cell 120000 0.86
echo "$(date -Is) ALL CELLS FINISHED" >> /tmp/stage1-progress.txt
