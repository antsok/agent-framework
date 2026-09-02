#!/usr/bin/env bash
# Tool-share series: tool output as a share of the conversation rather than an absolute size.
# Separate from the fixed-payload series -- the workload differs, so the numbers are not
# comparable with runs 26 to 29. Filler turns held at 18 across both shares so the pair
# differs only in how the context is split between tool output and filler.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini-2"
BIN="cachebench_live.exe"
SHARE="$1"; FILLER="$2"; OUT=/tmp/stage9
for s in 0 1 2 3 4; do
  tag="$(printf 'share%s-s%s' "${SHARE/./}" "$s")"
  "$BIN" foundry:gpt-5.4-mini-2 \
    --agent harness \
    --strategies none,context_window,truncation,anchored,anchored_min_gain,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.86 --context-window 120000 \
    --tool-share "$SHARE" --filler-tokens "$FILLER" \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "$OUT/$tag.jsonl" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 \
    > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) DONE $tag exit=$?" >> "$OUT/progress.txt"
done
echo "$(date -Is) CELL DONE share=$SHARE" >> "$OUT/progress.txt"
