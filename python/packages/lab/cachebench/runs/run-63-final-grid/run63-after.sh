#!/usr/bin/env bash
# Wait for "DONE <model> f0.9 s<seed>" in progress.txt, then run that seed's fill 3.0 cell.
MODEL="$1"; SEED="$2"; PRICES="$3"
S="/tmp"
until grep -q "DONE $MODEL f0.9 s$SEED " "$S/run63/progress.txt" 2>/dev/null; do sleep 60; done
bash "$S/run63-stream-p.sh" "$MODEL" "$PRICES" "3.0:$SEED"
