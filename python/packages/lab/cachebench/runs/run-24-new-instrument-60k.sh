#!/usr/bin/env bash
# Validation of the rebuilt instrument: seed -> snapshot -> independent probes.
# Two passes at the same cell. The first ran none,anchored,tool_summary_anchored and looked
# inert; those are the two least threshold-sensitive strategies, so the cell appeared to
# measure nothing. The second adds the threshold-driven rows and the cell discriminates.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
cachebench_live.exe foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,context_window,truncation,anchored,tool_summary_anchored \
  --repeats 1 --probe-repeats 3 \
  --fill 0.86 --context-window 60000 \
  --max-output-tokens 2048 --answer-max-tokens 12000 \
  --markers-per-tool 8 --tool-turns 6 --tool-result-tokens 3500 \
  --narration neutral --fact-placement spread \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
