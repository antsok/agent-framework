#!/usr/bin/env bash
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL="gpt-5.4-mini"
cachebench_live.exe foundry:gpt-5.4-mini \
  --agent harness \
  --strategies none,truncation,sliding_window,tool_result,selective_tool_call,context_window,context_window_aggressive,context_window_lazy,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize,anchored,anchored_no_assistant \
  --summarizer-provider foundry:gpt-5.4-mini \
  --repeats 3 \
  --narration neutral --fact-placement spread \
  --context-window 60000 --max-output-tokens 2048 --answer-max-tokens 4000 \
  --markers-per-tool 8 --tool-turns 6 \
  --filler-tokens 4000 --tool-result-tokens 8000 \
  --price-input 0.66 --price-cached 0.07 --price-output 3.96
