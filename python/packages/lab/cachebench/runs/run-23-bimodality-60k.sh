#!/usr/bin/env bash
# Invocation-level repetition at 60,000: six independent invocations, one repeat each, so the
# spread ACROSS invocations can be compared with run 16's spread WITHIN one. Profile is run
# 16's exactly. Run on a second Foundry account with the same model version (2026-03-17); the
# variance question is self-contained within these six, but absolute levels are not comparable
# with run 16 because the endpoint differs.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
for i in 1 2 3 4 5 6; do
  echo "########## INVOCATION $i ##########"
  cachebench_live.exe foundry:gpt-5.4-mini \
    --agent harness \
    --strategies none,anchored,tool_summary_anchored \
    --repeats 1 \
    --narration neutral --fact-placement spread \
    --context-window 60000 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --markers-per-tool 8 --tool-turns 6 \
    --filler-tokens 4000 --tool-result-tokens 8000 \
    --price-input 0.66 --price-cached 0.07 --price-output 3.96
  echo "INVOCATION $i EXIT=$?"
done
