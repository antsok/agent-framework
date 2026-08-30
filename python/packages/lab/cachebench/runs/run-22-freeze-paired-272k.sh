#!/usr/bin/env bash
# Paired frozen/unfrozen at 272,000, 3 repeats. Profile identical to run 18 apart from the
# flag and the dropped tool_result row, so the unfrozen arm is also a replication of run 18.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
arm () {  # label extra-flag...
  local label="$1"; shift
  echo "########## ARM ${label} ##########"
  cachebench_live.exe foundry:gpt-5.4-mini \
    --agent harness \
    --strategies none,truncation,anchored,tool_summary_anchored \
    --repeats 3 \
    --narration neutral --fact-placement spread \
    --context-window 272000 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --markers-per-tool 8 --tool-turns 6 \
    --filler-tokens 12600 --tool-result-tokens 25200 \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96 "$@"
  echo "ARM ${label} EXIT=$?"
}
arm unfrozen                       > run-22-freeze-paired-272k-unfrozen.txt 2>&1
arm frozen --freeze-during-answers > run-22-freeze-paired-272k-frozen.txt   2>&1
