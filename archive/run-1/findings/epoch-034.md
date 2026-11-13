# Epoch 34: Relocating TorchInductor/Triton compile caches to a fresh writable /tmp run directory befo

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7893 (Δ vs best-at-time: 0.0827)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 323.3 s

## Hypothesis
Relocating TorchInductor/Triton compile caches to a fresh writable /tmp run directory before torch.compile will prevent the lazy triton_hash_with_backend failure, restore compiled throughput (~4000+ steps/300s), and bring val_bpb back near or below the 0.7067 champion without changing model hyperparameters.

## Change
Added a top-of-file block (before `import torch`) that creates a fresh per-run temp dir via tempfile.mkdtemp and sets TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR to clean writable subdirs inside it, so inductor/triton init with uncorrupted caches. Also kept the e33 lazy-compile guard around the warmup forward (try/except -> eager fallback) so a RESULT line is always produced, and added a one-line log of the cache dir. No model/optimizer/schedule hyperparameters changed.

## Review
The cache relocation failed to restore compiled throughput, resulting in eager-mode performance of 2,511 steps and a poor val_bpb of 0.7893. This is a significant regression from our 0.7067 champion, so we must revert.

## Challenger concerns
- Cache relocation had zero observable effect on compile success or throughput; run remained at eager-mode speed (~2511 steps) identical to e28/e30/e33, so the measured 0.789 val_bpb cannot be attributed to the hypothesis.
- Hypothesis claims 'restore compiled throughput (~4000+ steps)' but the run delivered 2511 steps; the experiment therefore falsifies rather than tests the claim, yet still reports the number as if the change was neutral.
- Keeping the e33 lazy guard while adding cache logic creates an untestable compound change; we cannot tell whether cache relocation would have helped without the guard.

*Ledger: results.tsv, epoch 34.*
