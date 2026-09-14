#!/usr/bin/env bash
# Run 50 (volatile working copy): all 20 strategies at 120,000 tokens and 0.8 fill, the
# first all-strategies cell measured on the corrected reasoning counter (434a77da5) --
# run 47's archive at 170,000/0.9 was taken pre-fix, so its luna rows are comparable with
# each other and not with this one. Defaults everywhere else, on run 47's basis.
# Four seeds live at once: one invocation runs near 0.8M TPM at this cell against 4M.
set -u
export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
export FOUNDRY_MODEL=gpt-5.6-luna
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
ALL="none,context_window,context_window_aggressive,context_window_lazy,truncation,anchored,tool_summary_anchored,anchored_no_assistant,anchored_min_gain,sliding_window,tool_result,selective_tool_call,summarization,token_budget_fallback,token_budget_tools_first,token_budget_truncate_first,token_budget_window_first,token_budget_summarize,user_summary_anchored,tool_and_user_summary_anchored"
for s in "$@"; do
  "$BIN" foundry:gpt-5.6-luna \
    --agent harness --strategies "$ALL" \
    --summarizer-provider foundry:gpt-5.6-luna \
    --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 \
    --fill 0.8 --context-window 120000 \
    --assumed-reply-tokens 602 --max-output-tokens 2048 --answer-max-tokens 12000 \
    --record-max-tokens 2048 \
    --markers-per-tool 8 --tool-turns 6 \
    --narration neutral --fact-placement spread \
    --results-jsonl "/tmp/run50/s${s}.jsonl" \
    --price-input 0.20 --price-cached 0.02 --price-output 1.20 \
    > "/tmp/run50/s${s}.log" 2>&1
  echo "$(date -Is) DONE s${s} exit=$?" >> /tmp/run50/progress.txt
done
