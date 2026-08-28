#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
BIN=cachebench_live
cachebench_live foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,truncation,sliding_window,tool_result,selective_tool_call,context_window,context_window_aggressive,context_window_lazy,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize \
  --summarizer-provider foundry:gpt-5.4-mini \
  --repeats 3 \
  --context-window 272000 \
  --max-output-tokens 2048 \
  --markers-per-tool 8 \
  --tool-turns 6 \
  --filler-tokens 12600 \
  --tool-result-tokens 25200 \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
