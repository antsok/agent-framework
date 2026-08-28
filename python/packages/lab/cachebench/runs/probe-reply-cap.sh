#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
BIN=cachebench_live
for cap in 900 4000; do
  for guidance in on off; do
    flag=""
    [ "$guidance" = "off" ] && flag="--no-retrieval-guidance"
    echo "=== guidance=$guidance  answer_cap=$cap ==="
    cachebench_live foundry:gpt-5.4-mini \
      --agent harness --strategies none --repeats 3 \
      --sweeping-question $flag \
      --answer-max-tokens "$cap" \
      --context-window 60000 --max-output-tokens 2048 \
      --markers-per-tool 8 --tool-turns 6 \
      --filler-tokens 4000 --tool-result-tokens 8000 \
      --price-input 0.66 --price-cached 0.07 --price-output 3.96 2>&1 \
      | grep -E "^none|^-> |ERROR"
  done
done
