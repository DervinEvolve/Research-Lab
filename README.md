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
RESULT {"val_bpb": <float>, "train_steps": <int>, "train_seconds": <float>, "peak_vram_mb": <float>, "params_m": <float>, "tokens_seen": <int>}
```

This is the only machine-readable output the harness (and CI) parses. It
must be valid JSON on a single line, prefixed with `RESULT `, and nothing
may be printed to stdout after it.

## Layout

```
prepare.py            fixed harness: data pipeline + evaluate_bpb (never edited by agents)
train.py              agent-owned baseline: nanoGPT-style byte-level GPT + training loop
program.md            the multi-agent research contract
results.tsv           append-only log of every epoch's measured result
wiki/                 accumulated findings, open questions, failed approaches
.github/workflows/ci.yml   smoke-tests prepare.py + train.py on every PR
```

## Credit

Training code derived from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
(MIT license) patterns. Experiment contract inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).
