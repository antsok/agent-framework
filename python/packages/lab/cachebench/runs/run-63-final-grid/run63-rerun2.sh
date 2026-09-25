#!/usr/bin/env bash
# Re-run, for each fill:seed job, every gpt-6-luna row that failed on a 429 (read from the cell's
# log), plus the control the benchmark requires. Writes -rerun files; the originals are kept.
S="/tmp"
OUT="$S/run63/gpt-6-luna"
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
BIN="D:/github/antsok/agent-framework/.claude/worktrees/bridge-cse_019XUH36XMKe8oyLVdmoWpdp/python/.venv/Scripts/cachebench_live.exe"
for job in "$@"; do
  f="${job%%:*}"; s="${job##*:}"
  until grep -q "DONE gpt-6-luna f$f s$s " "$S/run63/progress.txt"; do sleep 60; done
  failed=$(grep -E 'seed [0-9]+/1 .*(Error code|Exception)' "$OUT/f$f-s$s.log" | awk '{print $1}' | grep -v '^none$' | paste -sd, -)
  strats="none${failed:+,$failed}"
  echo "$(date -Is) RERUN gpt-6-luna f$f s$s: $strats" >> "$S/run63/progress.txt"
  "$BIN" azure-responses:gpt-6-luna --agent harness --strategies "$strats" --summarizer-provider azure-responses:gpt-6-luna --repeats 1 --seed-offset "$s" --probe-repeats 1 --combined-repeats 5 --fill "$f" --context-window 120000 --assumed-reply-tokens 384 --max-output-tokens 2048 --answer-max-tokens 12000 --record-max-tokens 2048 --markers-per-tool 8 --tool-turns 6 --narration neutral --fact-placement spread --results-jsonl "$OUT/f$f-s$s-rerun.jsonl" --price-input 0.10 --price-cached 0.01 --price-output 0.50 --price-cache-write 0.125 > "$OUT/f$f-s$s-rerun.log" 2>&1
  echo "$(date -Is) DONE-RERUN gpt-6-luna f$f s$s exit=$?" >> "$S/run63/progress.txt"
done
