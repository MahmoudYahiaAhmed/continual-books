# continual-books

Teach Qwen2.5-7B-Instruct two close-topic books **one after another** and measure how much of the first book survives the second.

- **Book A:** *The Red Thumb Mark* (R. Austin Freeman, 1907, Project Gutenberg #11128)
- **Book B:** *The Mystery of 31 New Inn* (R. Austin Freeman, 1912, Project Gutenberg #12187)

Both are Dr. Thorndyke novels with the same detective, the same narrator (Jervis) and the same London legal setting. The overlap is deliberate: close topics are where sequential fine-tuning forgets the most.

## Setup

| Stage | What happens |
|---|---|
| Data | Each book is split into ~1,800-character passages. **Qwen2.5-7B-Instruct itself** writes 4 fact QA pairs per passage, each with a reworded version of the question. No text from closed models is used. |
| Train / test split | Every fact is trained with its original question. 150 facts per book are tested with the **reworded** question, so the score measures recall of the fact, not memorised wording. |
| Step 1 | LoRA (r=32, all linear layers) on book A, giving **M1**, which is merged into the weights. |
| Step 2 | LoRA on book B on top of M1, three ways: **naive**, **replay** (25% book-A QA mixed in), and **self-distill** (top-20 KL to M1's own answers on 800 book-A questions, answer tokens only). |
| Scoring | Greedy answers are graded by base Qwen2.5-7B-Instruct as judge (yes/no against the reference). A string-match check is stored next to it. |
| Retention | Of the book-A questions **M1 answered correctly**, the share still correct after step 2. |

Why self-distillation: a KL penalty on raw book-A text protects next-token prediction on prose, but the knowledge is queried through question → answer. Distilling M1's own answers to book-A questions protects exactly that pathway and needs no reference answers.

## Results

Run of 2026-11-04, one A100 80 GB (Colab), seed 1234. 150 reworded test questions per book.

<!-- RESULTS:START -->
| Model | Book A correct | Book B correct | Book A kept (of M1-correct) | Book B gain vs M1 [95% CI] | Train GPU h |
|---|---|---|---|---|---|
| Base Qwen2.5-7B-Instruct | 15/150 | 3/150 | - | - | - |
| M1: + book A (LoRA) | 58/150 | 6/150 | - | - | 0.06 |
| M1 + book B, naive | 22/150 | 61/150 | 14/58 (24%) | +55 [+43, +68] | 0.06 |
| M1 + book B, replay 25% | 34/150 | 59/150 | 27/58 (47%) | +53 [+41, +65] | 0.08 |
| M1 + book B, self-distill to M1 | 45/150 | 42/150 | 34/58 (59%) | +36 [+26, +47] | 0.13 |
<!-- RESULTS:END -->

**What it shows**

- **Forgetting is severe on close topics.** Naive training on book B keeps only 24% of the book-A answers M1 had right.
- **Self-distillation to M1 protects the most:** 59% kept, against 47% for replay of the reference QA, with no reference answers needed.
- **Protection costs plasticity.** Distillation learns less of book B (+36 against +55 for naive); the intervals do not overlap. Balancing the two (KD weight, schedule, more paraphrased new-book data) is the next experiment.
- **Cheap.** Each step trains in under 0.15 A100-hours; the whole pipeline, data generation and judging included, ran in about 40 minutes.

Single seed: differences of a few questions are noise. The book-B gain column carries a paired bootstrap 95% interval.

Per-question answers and verdicts for every model are in `results/judged/*.jsonl`. Training loss curves are in W&B when `WANDB_PROJECT` is set.

## Run it

Needs one GPU with at least 40 GB (A100 40/80 GB, A6000, L40S). The whole pipeline takes under 1 hour on an A100.

```bash
pip install -r requirements.txt
export WANDB_PROJECT=continual-books   # optional
bash run_all.sh
```

Every stage skips itself if its output already exists, so an interrupted run can be resumed by running the same command again.

### Google Colab (A100)

```python
!git clone https://github.com/MahmoudYahiaAhmed/continual-books /content/continual-books
%cd /content/continual-books
!pip install -q -r requirements.txt
!pip uninstall -y -q torchao   # Colab ships torchao 0.10, which recent peft rejects
import os; os.environ["WANDB_PROJECT"] = "continual-books"   # optional; run `!wandb login` first
!bash run_all.sh 2>&1 | tee run.log
```

Models (~15 GB for merged M1) stay on Colab's local disk. When the run ends, save the evidence:

```python
from google.colab import drive; drive.mount('/content/drive')
!cp -r data results README.md run.log models/*/train_meta.json /content/drive/MyDrive/ 2>/dev/null; ls results
```

## Layout

```
cl_books/
  books.py     download + clean + split Gutenberg text
  qa.py        QA generation with Qwen as teacher
  train.py     LoRA training: naive / replay / distill
  teacher.py   M1 top-k targets for self-distillation
  losses.py    top-k distillation loss
  generate.py  greedy answers on the eval questions
  judge.py     LLM judge + string match
  report.py    results table (README + results/results.md)
run_all.sh     the full pipeline
tests/         CPU unit tests (pytest)
```

## Limits

- One seed, 150 test questions per book: treat differences under ~10 questions as noise (the gain column shows a paired bootstrap 95% interval).
- The judge is the same base model family as the student; spot-check `results/judged/` before trusting a small difference.
- General skills (IFEval, GSM8K, MMLU) are not measured here.

## License

Code: Apache-2.0. Books: public domain (Project Gutenberg).
