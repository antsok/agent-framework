#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
BIN=cachebench_live.exe
run () {  # window filler tool label
  echo "########## RUN $4 : window=$1 filler=$2 tool=$3 ##########"
  "$BIN" foundry:gpt-5.4-mini \
    --agent harness \
    --strategies none,truncation,tool_result,anchored,tool_summary_anchored \
    --repeats 5 \
    --narration neutral --fact-placement spread \
    --context-window "$1" --max-output-tokens 2048 --answer-max-tokens 12000 \
    --markers-per-tool 8 --tool-turns 6 \
    --filler-tokens "$2" --tool-result-tokens "$3" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96
  echo
}
run 120000 8000  16000 "120k"
run 272000 12600 25200 "272k"
