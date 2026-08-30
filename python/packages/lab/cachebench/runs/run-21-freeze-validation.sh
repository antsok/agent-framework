#!/usr/bin/env bash
# Mechanism validation for --freeze-during-answers, not a measurement: one repeat, so the
# accuracy columns carry no error bars and nothing here isolates the freeze's effect.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
cachebench_live.exe foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,anchored,tool_summary_anchored \
  --repeats 1 \
  --no-force-tool-calls \
  --narration neutral --fact-placement spread \
  --context-window 120000 --max-output-tokens 2048 --answer-max-tokens 4000 \
  --markers-per-tool 8 --tool-turns 6 \
  --filler-tokens 8000 --tool-result-tokens 16000 \
  --freeze-during-answers \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
