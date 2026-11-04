| Model | Book A correct | Book B correct | Book A kept (of M1-correct) | Book B gain vs M1 [95% CI] | Train GPU h |
|---|---|---|---|---|---|
| Base Qwen2.5-7B-Instruct | 15/150 | 3/150 | - | - | - |
| M1: + book A (LoRA) | 58/150 | 6/150 | - | - | 0.06 |
| M1 + book B, naive | 22/150 | 61/150 | 14/58 (24%) | +55 [+43, +68] | 0.06 |
| M1 + book B, replay 25% | 34/150 | 59/150 | 27/58 (47%) | +53 [+41, +65] | 0.08 |
| M1 + book B, self-distill to M1 | 45/150 | 42/150 | 34/58 (59%) | +36 [+26, +47] | 0.13 |
