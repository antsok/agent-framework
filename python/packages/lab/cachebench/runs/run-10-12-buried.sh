#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
BIN=cachebench_live.exe
STRATS="none,truncation,sliding_window,tool_result,selective_tool_call,context_window,context_window_aggressive,context_window_lazy,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize"
run () {  # window filler tool label
  echo "########## RUN $4 : window=$1 filler=$2 tool=$3 ##########"
  "$BIN" foundry:gpt-5.4-mini \
    --agent harness --strategies "$STRATS" \
    --summarizer-provider foundry:gpt-5.4-mini \
    --repeats 3 \
    --context-window "$1" --max-output-tokens 2048 \
    --answer-max-tokens 4000 \
    --fact-placement spread \
    --markers-per-tool 8 --tool-turns 6 \
    --filler-tokens "$2" --tool-result-tokens "$3" \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96
  echo
}
run 60000  4000  8000  "10-60k"
run 120000 8000  16000 "11-120k"
run 272000 12600 25200 "12-272k"
