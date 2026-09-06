#!/usr/bin/env bash
# Run 42: every strategy in one cell at high fill, on the repaired code.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="cachebench_live"
ALL="none,context_window,context_window_aggressive,context_window_lazy,truncation,anchored,tool_summary_anchored,anchored_no_assistant,anchored_min_gain,sliding_window,tool_result,selective_tool_call,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize"
s="$1"
"$BIN" foundry:gpt-5.6-luna \
  --agent harness --strategies "$ALL" \
  --summarizer-provider foundry:gpt-5.6-luna \
  --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
  --fill 0.86 --context-window 60000 --tool-result-tokens 3500 \
  --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
  --markers-per-tool 8 --tool-turns 6 \
  --narration neutral --fact-placement spread \
  --results-jsonl "/tmp/run42/all-s$s.jsonl" \
  --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
  > "/tmp/run42/all-s$s.log" 2>&1
echo "$(date -Is) DONE s$s exit=$?" >> /tmp/run42/progress.txt
