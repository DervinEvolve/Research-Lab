# Run 1 (archived)

Run 1 was the Observable Lab's first research run on this bench: a panel of
models from different labs took turns as Planner, Executor, Challenger,
Reviewer and Commentator, one experiment per epoch, editing `train.py` to
lower `val_bpb` under a 300-second, one-GPU training budget.

- **Dates:** first epoch 7 Jul 2026 23:05 UTC, last epoch 13 Nov 2026 22:08 UTC.
- **Epochs:** 49. Verdicts: 4 keep, 28 discard, 17 crash.
- **Best result:** **0.7067 val_bpb at epoch 14** (8 Jul 2026, commit `35b8a23`:
  cosine schedule horizon set to 4,100 steps; 4,335 steps in 300.1 s, 3,191 MB
  peak VRAM, 10.72 M parameters). This score is self-reported and was never
  reproduced (see below).
- **Archived:** 13 Nov 2026, before run 2.

## What is in this folder

| File | What it is |
|---|---|
| `results.tsv` | The ledger exactly as it stood on `main` after epoch 49 (commit `f71ab4d`): 783,556 bytes, 2,375 data rows. Not edited. |
| `results-dedup.tsv` | A convenience copy with one row per epoch (49 rows): the last row written for each epoch, sorted by epoch. Rows from before the provenance columns existed are padded with `-` to the 13-column header. |
| `findings/` | The 49 finding pages the lab wrote, `epoch-001.md` to `epoch-049.md`, unchanged. |
| `train.py` | `train.py` as it stood at the end of run 1: the epoch 14 champion, unchanged since 8 Jul. |

## Every score is unverified

Each `val_bpb` here was printed by the generated `train.py` itself, in its
`RESULT` line. The runner parsed that line; nothing evaluated the model
independently or repeated a run. From epoch 25 on, rows say so in the
`metrics_source` column (`experiment_stdout_unverified`); earlier rows predate
that column, and the same applies to them. Read every number as a claim made by
the code that produced it.

## Three phases that are not comparable with each other

| Epochs | When (UTC) | Data and runner | What happened |
|---|---|---|---|
| 1 to 4 | 7 to 8 Jul | enwik8, bytes | Scores of 6.57 to 7.94 bpb. On 8 Jul at 04:09 UTC (commit `0fd7555`) the bench switched to TinyStories because enwik8's XML made a 300 s run mostly noise. These four epochs are not comparable with anything after them, including epoch 4's "keep" at 6.5712. |
| 5 to 20 | 8 to 9 Jul | TinyStories-GPT4-clean, unpinned download, the July runner image, where `torch.compile` worked | Three keeps: epoch 7 at 0.7223 (RMSNorm), epoch 9 at 0.7156 (SwiGLU MLP), epoch 14 at 0.7067 (schedule horizon). Runs took about 4,100 to 4,400 steps. |
| 21 to 49 | 4 to 13 Nov | TinyStories pinned to dataset revision `0397e271` and checked against its SHA-256 (harness SHA-256 `28b03c9f`), each run in a sealed sandbox on an A10G, image `python:3.11.18-slim-bookworm` | The image had no C compiler, and `torch.compile` needs one the first time it runs a kernel. 14 of these 29 epochs crashed: 21, 22, 24, 27, 32, 35, 36, 39, 41, 44 and 46 inside `torch.compile`/Triton, and 23, 29 and 31 with syntax errors. The 15 that ran fell back to eager mode, reached 1,335 to 3,678 steps and scored 0.7352 to 0.8371. Epoch 46 tried to reproduce the epoch 14 champion exactly and crashed, so 0.7067 was never reproduced on this runner. |

## Why the ledger had 2,326 duplicate rows

`results.tsv` has 2,375 data rows for 49 epochs. 2,325 of the extra rows are
epoch 20 and one is epoch 44.

Epoch 20's `train.py` had a syntax error (an unterminated string literal), so
the run crashed within seconds. The lab recorded the crash, a ledger row and a
finding page, but the epoch never got marked complete in the lab's database:
a later step of the same epoch failed afterwards (which step is not recorded
in this repository). From then on, every hourly tick resumed epoch 20, reused
its saved outputs and wrote the same row and finding page again: 2,326 rows for
epoch 20 between 9 Jul 04:02 UTC and 4 Nov 13:00 UTC. Epoch 44 was written twice
the same way on 5 Nov. That loop is also why the ledger grew to within about
200 KB of the 1 MB limit for reading a file through GitHub's contents API.

What changed since:

- the runner compiles `train.py` before preparing data or starting a GPU, and
  returns a syntax error as a crash at stage `syntax`;
- the lab no longer appends a row identical to one already in the ledger,
  reads large ledgers through the git blobs API, and refuses to rewrite a
  ledger whose header it does not recognise;
- a finding page from another experiment is never overwritten.

## What run 2 starts from, and why

Run 2's `train.py` is this folder's `train.py`, the epoch 14 champion, with one
change: if `torch.compile` fails on its first use, the run falls back to eager
mode with a warning instead of crashing, and the `RESULT` line reports
`"compiled": true` or `false`. The runner image now also includes a C and C++
compiler (`gcc`, `g++`, `libc6-dev`), which is what `torch.compile` was missing.

Why not go back to the original scaffold: the scaffold calls `torch.compile`
the same way, so on the November image it would have crashed exactly like the
champion. The crash was the environment, not the model code, so the fix belongs
in the runner image. Starting from the champion keeps what run 1 learned
(RMSNorm, SwiGLU, the schedule horizon) without claiming its score: the image
changed, so run 2's baseline has to be measured again before its first epoch,
and nothing in run 2 is compared with the numbers in this folder. The
`compiled` flag keeps an eager-mode fallback from being mistaken for a real
result, the confound that sank epochs 45 and 48.
