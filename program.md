# Research Program: TinyStories val_bpb under a fixed budget

This is the operating contract for the autonomous multi-agent research org
that iterates on `train.py`. It is adapted from the shape of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s
single-agent contract, restructured for a multi-role org. No code from that
repo is used here (it carries no license); `train.py`'s implementation is
original, derived from [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT)
patterns.

## Mission

Drive `val_bpb` (validation bits-per-byte on TinyStories, byte-level) as low as possible,
under a **fixed wall-clock training budget** (`TIME_BUDGET`, default 300s)
on **one GPU**. This is a constrained-compute research loop, not a
leaderboard chase against unlimited resources: the only lever is making
better use of the same budget every epoch.

## The metric

`val_bpb`, computed exclusively by `prepare.evaluate_bpb()`:

- Model must expose `forward(idx, targets=None) -> (logits, loss)` with
  `loss` = mean cross-entropy in nats/token.
- Since the vocabulary is raw bytes (256 tokens, no BPE/tokenizer), bits-
  per-byte falls out exactly as `loss / ln(2)` -- no vocab-size correction
  needed, no cross-tokenizer ambiguity. This makes bpb directly comparable
  across every architecture change an agent might try.
- Evaluation crops are selected with a **fixed seeded generator**
  (`prepare.EVAL_SEED`) over `val.bin`, so every epoch is scored against the
  identical set of byte windows. Lower is strictly better.
- **Only `val_bpb` from `evaluate_bpb` counts.** `test.bin` is held out for
  end-of-program reporting only and must never be touched by `train.py` or
  by any agent during the loop.

## Rules

1. **Only `train.py` may be modified.** `prepare.py` (data pipeline, batch
   sampler, `evaluate_bpb`) is immutable infrastructure. If an agent
   believes `prepare.py` has a bug, it raises this as a finding in
   `wiki/open-questions.md` -- it does not patch around it in `train.py`.
2. **No new dependencies.** The stack is stdlib + numpy + torch (+ requests,
   used only by `prepare.py`). Do not import anything else, vendor
   anything else, or shell out to install packages mid-run.
3. **Simpler is better on ties.** If two variants land within noise of each
   other on `val_bpb`, prefer the simpler / lower-risk diff as the new
   baseline. Complexity must be paid for with a measured win.
4. **Dramatic VRAM increases are unacceptable.** A hypothesis that trades a
   large peak_vram_mb increase for a marginal bpb win does not clear the
   bar -- the harness targets a single fixed GPU, not an elastic one.
5. **The budget is the budget.** `TIME_BUDGET` is wall-clock and excludes
   only the one untimed warmup/compile step already built into `train.py`.
   Agents may not extend it, checkpoint across runs to accumulate more
   effective compute, or read the clock and stop early to game variance.

## The org

Five roles collaborate each epoch. Names are functional, not literal
processes -- an implementation may fuse or split them across models.

- **Planner** proposes exactly ONE falsifiable hypothesis for the epoch
  (e.g. "RMSNorm instead of LayerNorm reduces val_bpb by lowering
  normalization overhead per step, at ~equal step time") with a stated
  expected effect and direction. One hypothesis per epoch keeps attribution
  clean.
- **Executor** implements the hypothesis as a **minimal diff** to
  `train.py`. Minimal means: touch only what the hypothesis requires. No
  opportunistic drive-by changes bundled into the same diff.
- **Harness (fixed, non-agentic)** runs the resulting `train.py` on fixed
  hardware for `TIME_BUDGET` seconds and records the `RESULT` line.
- **Challenger** attacks the epoch's methodology before it's accepted:
  confounds (did step time change enough to alter effective tokens seen?),
  budget gaming (is the diff spending the clock differently rather than
  learning faster?), eval overfitting (did the diff special-case anything
  that smells like the eval crop distribution?). A clean challenge blocks
  acceptance until resolved or explicitly overridden with reasoning logged.
- **Reviewer** compares measured `val_bpb` against the current best,
  decides keep-or-revert, and logs the row to `results.tsv` regardless of
  outcome (reverted epochs are logged too: `commit=reverted`, with
  `status=discard` for a run that finished and `status=crash` for one that
  did not).
- **Commentator** narrates the epoch in plain language and ensures
  `wiki/` is updated to reflect the verdict -- this is how the org
  accumulates institutional memory instead of re-discovering the same dead
  end every epoch.

## Budget-gaming clause

- **Allowed:** changes that spend the fixed time budget differently
  (larger batch + fewer steps, different sequence length, different
  optimizer stepping cadence, gradient accumulation, etc.) as long as they
  are honestly trying to lower loss faster, not to manipulate the
  measurement.
- **Grounds for immediate revert, no Challenger debate required:**
  - Any code path that inspects, reads from, or samples `val.bin` or
    `test.bin` outside of the unmodified `evaluate_bpb` call.
  - Any change to how `evaluate_bpb` is invoked that alters its seed,
    batch count, or crop selection to make the number look better without
    a corresponding real improvement in the model.
  - Caching or memoizing eval results across epochs instead of re-running
    the fixed eval harness on the current model weights.
  - Silently lengthening effective training time beyond `TIME_BUDGET`
    (e.g. excluding slow steps from the clock, background threads that
    keep training after the loop reports it stopped).

Any of the above is an automatic revert to the last known-good `train.py`,
logged in `results.tsv` as reverted with the reason in the epoch's finding
page, and written up in `wiki/failed-approaches.md` so it is never quietly
retried.

## Runs

Records are kept per run. A finished run moves to `archive/run-N/` (ledger,
finding pages, final `train.py`, README) and the next run starts with an
empty ledger and an empty `wiki/findings/`. Scores are never compared across
runs: each run measures its own baseline on its own runner image.

**Run 1** (7 Jul to 13 Nov 2026) is archived in
[`archive/run-1/`](./archive/run-1/README.md).

**Run 2 starts from run 1's champion `train.py`** (epoch 14, self-reported
0.7067 val_bpb on the July runner), with one change: if `torch.compile` fails
on its first use, training falls back to eager mode with a warning instead of
crashing, and the `RESULT` line reports `"compiled": true|false`. Reasons:

- Run 1's November crashes came from the runner image, which had no C
  compiler for `torch.compile`, not from the model code. The image now has
  one. The original scaffold calls `torch.compile` the same way, so restoring
  it would not have avoided the crash; the champion keeps what run 1 learned.
- The image changed, so run 1's numbers do not carry over. Run 2's baseline is
  the starting `train.py` measured once on the new image before epoch 1.
- `compiled` makes an eager fallback visible. Eager runs take far fewer steps
  in the same budget (about 2,400 to 3,700 against about 4,300 compiled in
  run 1), so a `compiled: false` result is never compared with a compiled one
  without saying so.

## Syntax check

Before any data is prepared or a GPU is started, the runner compiles
`train.py` with the same Python that will run it. A file that does not
compile is recorded as a crash at stage `syntax` and costs no GPU time. Four
of run 1's epochs (20, 23, 29 and 31) were lost to syntax errors.
