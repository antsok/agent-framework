#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
cachebench_live.exe foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,truncation,tool_result,anchored,tool_summary_anchored \
  --repeats 5 \
  --narration neutral --fact-placement spread \
  --context-window 60000 --max-output-tokens 2048 --answer-max-tokens 12000 \
  --markers-per-tool 8 --tool-turns 6 \
  --filler-tokens 4000 --tool-result-tokens 8000 \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
