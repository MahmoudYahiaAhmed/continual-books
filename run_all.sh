#!/usr/bin/env bash
# Full pipeline: data -> M1 (book A) -> three book-B runs on top of M1 -> answers -> judge -> report.
# Each stage is skipped if its output already exists, so a crashed Colab session can resume.
set -euo pipefail
cd "$(dirname "$0")"

BASE="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
M1=models/m1/merged

step() { echo; echo "=== $* ==="; }

if [ ! -f data/B/eval.jsonl ]; then step "QA generation"; python -m cl_books.qa; fi

if [ ! -d "$M1" ]; then step "Step 1: book A -> M1"; python -m cl_books.train --base "$BASE" --book A --out models/m1 --merge; fi

if [ ! -d models/m2_naive/adapter ]; then
  step "Step 2: book B, naive"
  python -m cl_books.train --base "$M1" --book B --method naive --out models/m2_naive
fi
if [ ! -d models/m2_replay/adapter ]; then
  step "Step 2: book B, replay"
  python -m cl_books.train --base "$M1" --book B --method replay --out models/m2_replay
fi
if [ ! -f data/A/rehearsal.jsonl ]; then step "M1 self-distillation targets"; python -m cl_books.teacher --model "$M1" --out data/A/rehearsal.jsonl; fi
if [ ! -d models/m2_distill/adapter ]; then
  step "Step 2: book B, self-distill to M1"
  python -m cl_books.train --base "$M1" --book B --method distill --rehearsal data/A/rehearsal.jsonl --out models/m2_distill
fi

gen() { [ -f "results/answers/$1.jsonl" ] || { step "Answers: $1"; python -m cl_books.generate --name "$@"; }; }
gen base --model "$BASE"
gen m1 --model "$M1"
gen m2_naive --model "$M1" --lora models/m2_naive/adapter
gen m2_replay --model "$M1" --lora models/m2_replay/adapter
gen m2_distill --model "$M1" --lora models/m2_distill/adapter

step "Judge"; python -m cl_books.judge
step "Report"; python -m cl_books.report
