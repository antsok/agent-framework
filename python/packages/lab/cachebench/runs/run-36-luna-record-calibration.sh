#!/usr/bin/env bash
# Calibration: what does luna's recall record want to be, and does a complete one
# still save anything? --record-max-tokens and --record-target-tokens vary; all else
# matches runs/run-34-35-luna.sh. 'truncation' is present only to give the MSGS check
# a strategy row that adds no messages of its own -- with tool_summary_anchored alone
# the leanest row carries the record and the control reads 13 messages short.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.6-luna"
BIN="cachebench_live"
TAG="$1"; W="$2"; F="$3"; CAP="$4"; TGT="$5"; S0="$6"; S1="$7"; shift 7
for s in $(seq "$S0" "$S1"); do
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness \
    --strategies none,truncation,tool_summary_anchored \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill "$F" --context-window "$W" --assumed-reply-tokens 602 \
    --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens "$CAP" --record-target-tokens "$TGT" \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/lunacap/$TAG-c$CAP-t$TGT-s$s.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    "$@" > "/tmp/lunacap/$TAG-c$CAP-t$TGT-s$s.log" 2>&1
  echo "$(date -Is) DONE $TAG-c$CAP-t$TGT-s$s exit=$?" >> /tmp/lunacap/progress.txt
done
