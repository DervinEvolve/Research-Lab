# Research-Lab

This is the Observable Lab's research bench: an experiment where several
frontier LLM agents autonomously edit a single training script to drive
down validation loss on a real language-modeling benchmark, under a fixed
compute budget, with a fixed measurement harness they cannot tamper with.

Concretely: `train.py` implements a small nanoGPT-style byte-level
transformer trained on [TinyStories-GPT4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
Each epoch, a researcher agent may rewrite anything in `train.py` -- the
architecture, the optimizer, the schedule, the batching strategy -- and the
harness measures `val_bpb` (validation bits-per-byte, lower is better)
after training for a fixed wall-clock budget (`TIME_BUDGET`, default 300s)
on one GPU. `prepare.py` (the data pipeline and the eval metric) is frozen
infrastructure the agents never touch. See `program.md` for the full
multi-agent contract, and `wiki/` for the accumulated findings.

## Runs

| Run | Dates | Status | Records |
|---|---|---|---|
| 1 | 7 Jul to 13 Nov 2026 | Archived. 49 epochs; best self-reported score 0.7067 val_bpb at epoch 14, never reproduced. | [`archive/run-1/`](./archive/run-1/) |
| 2 | not started | Starts fresh: an empty `results.tsv` (header only) and an empty `wiki/findings/`. | `results.tsv`, `wiki/` |

Scores from different runs are not comparable. Run 1 spanned three setups
(enwik8, then TinyStories on the July runner, then a pinned dataset on a new
runner image where `torch.compile` could not run); its
[README](./archive/run-1/README.md) explains each, the epoch 20 crash loop that
wrote 2,326 duplicate ledger rows, and why run 2 starts from run 1's champion
`train.py` with a fresh baseline measurement.

## Running locally

```bash
# one-time: download TinyStories parquet (~673MB) and materialize train/val/test .bin files
python prepare.py

# train the current baseline for TIME_BUDGET_S seconds (default 300) and
# print a final RESULT line with val_bpb
python train.py

# fast CPU smoke test (~20s of training, tiny model) -- useful for CI and
# for sanity-checking that a new train.py variant at least runs end to end
python train.py --smoke
```

Environment variables:

- `RESEARCH_CACHE` -- where `prepare.py` stores the dataset parquet and the
  `train.bin` / `val.bin` / `test.bin` memmap files. Defaults to
  `~/.cache/research-lab`.
- `TIME_BUDGET_S` -- wall-clock training budget in seconds, read by
  `prepare.py` and used by `train.py`. Defaults to `300`.

## The contract

| # | Rule |
|---|------|
| 1 | Only `train.py` may be modified by researcher agents. `prepare.py` is immutable. |
| 2 | The sole metric is `val_bpb` from `prepare.evaluate_bpb()` -- deterministic, seeded crops of `val.bin`. |
| 3 | `train.py` must end by printing exactly one `RESULT {...}` JSON line (see below) and nothing after it. |
| 4 | No new dependencies beyond torch / numpy (+ requests, used only by `prepare.py`). |
| 5 | Must run on CPU (slowly) so `--smoke` works without a GPU. |
| 6 | Dramatic VRAM increases and budget-gaming (touching `val.bin`/`test.bin`, manipulating the eval call, extending the clock) are grounds for immediate revert. |

Full detail, including the multi-agent org (Planner / Executor / Challenger
/ Reviewer / Commentator) and the budget-gaming clause, is in
[`program.md`](./program.md).

### The RESULT line

```
RESULT {"val_bpb": <float>, "train_steps": <int>, "train_seconds": <float>, "peak_vram_mb": <float>, "params_m": <float>, "tokens_seen": <int>, "compiled": <bool>}
```

This is the only machine-readable output the harness (and CI) parses. It
must be valid JSON on a single line, prefixed with `RESULT `, and nothing
may be printed to stdout after it. `compiled` says whether training actually
ran through `torch.compile` (from run 2 on): an eager-mode run takes far fewer
steps in the same budget, so the two are not comparable.

Every score is self-reported: the runner parses the line that the generated
`train.py` prints and does not evaluate the model independently.

## Layout

```
prepare.py            fixed harness: data pipeline + evaluate_bpb (never edited by agents)
train.py              agent-owned baseline: nanoGPT-style byte-level GPT + training loop
program.md            the multi-agent research contract
results.tsv           append-only log of every epoch's result in the current run
wiki/                 accumulated findings, open questions, failed approaches
archive/run-N/        a finished run: its ledger, finding pages, final train.py and a README
.github/workflows/ci.yml   smoke-tests prepare.py + train.py on every PR
```

## Starting a later run

Archive the finished run first, so its records are not mixed with the next:
copy `results.tsv`, `wiki/findings/` and `train.py` into `archive/run-N/`,
write its README, then reset `results.tsv` to the header line and
`wiki/findings/` to an empty folder. The lab also refuses to overwrite a
finding page written by a different experiment.

## Credit

Training code derived from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
(MIT license) patterns. Experiment contract inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).
