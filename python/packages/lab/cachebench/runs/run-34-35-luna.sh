#!/usr/bin/env bash
# gpt-5.6-luna, same nine-cell shape as runs 32/33, windows 60K/120K/200K.
# --assumed-reply-tokens 602 measured from a probe: luna writes four times what
# gpt-5.4-mini does, and the default overshot a 200K cell by 24%.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.6-luna"
BIN="cachebench_live.exe"
TAG="$1"; W="$2"; F="$3"; shift 3
for s in 0 1 2 3 4; do
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness \
    --strategies none,context_window,truncation,anchored,anchored_min_gain,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill "$F" --context-window "$W" --assumed-reply-tokens 602 \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 4000 --record-target-tokens 2000 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/luna/$TAG-s$s.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    "$@" > "/tmp/luna/$TAG-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-s$s exit=$?" >> /tmp/luna/progress.txt
done
echo "$(date -Is) CELL DONE $TAG" >> /tmp/luna/progress.txt
