#!/usr/bin/env bash
# Tool share held at 0.80, fill varied. Answers whether "compaction pays when tool output
# dominates" is about the tool share or merely about sitting near the ceiling.
# Filler turn count is not held: share, fill and turn count together over-determine the
# sizing, so the solver picks the count and it drifts 12 / 15 / 18 across the three fills.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini-2"
BIN="cachebench_live.exe"
FILL="$1"; OUT=/tmp/stage10
for s in 0 1 2 3 4; do
  tag="$(printf 'f%s-s%s' "${FILL/./}" "$s")"
  "$BIN" foundry:gpt-5.4-mini-2 \
    --agent harness \
    --strategies none,context_window,truncation,anchored,anchored_min_gain,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill "$FILL" --context-window 120000 \
    --tool-share 0.80 --filler-tokens 877 \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "$OUT/$tag.jsonl" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 \
    > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) DONE $tag exit=$?" >> "$OUT/progress.txt"
done
echo "$(date -Is) CELL DONE fill=$FILL" >> "$OUT/progress.txt"
