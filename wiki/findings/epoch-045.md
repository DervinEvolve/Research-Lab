# Epoch 45: Adding a learned 256-dimensional output logit bias to the tied lm_head, while otherwise ke

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7838 (Δ vs best-at-time: 0.0771)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 324.5 s

## Hypothesis
Adding a learned 256-dimensional output logit bias to the tied lm_head, while otherwise keeping the epoch-14 champion architecture/LR/schedule unchanged, will lower compiled-regime val_bpb from 0.7067 to 0.703-0.706 by giving the byte model a direct unigram-frequency prior.

## Change
Added a single learned nn.Parameter `self.logit_bias` (shape vocab_size=256, zero-init) to GPT, applied as `logits = self.lm_head(x) + self.logit_bias` in forward — the only architectural change vs the epoch-14 champion. Critically, I also fixed the recurring compile trap: the warmup forward+backward (which is what actually triggers lazy Triton compilation) is now wrapped in try/except that falls back to eager `run_model = model`, so a deferred compile failure no longer crashes the run (as it did in e44). LR/schedule/batch/seq unchanged (1.2e-3, 4100 horizon, 64x512).

## Review
Run delivered 0.7838 val_bpb at 2490 steps—clear eager-mode result, worse than 0.7067 champion. Compile-fallback wrapper switched regime mid-experiment, so bias hypothesis was never tested under the claimed compiled ~4100-step conditions; confound invalidates attribution.

## Challenger concerns
- The logit bias change is entangled with a compile-fallback fix that silently switched the training regime from compiled to eager. The hypothesis was for the compiled regime (0.7067 baseline, ~4100 steps), but the run completed only 2490 steps in 300s — a hallmark of eager mode. The measured val_bpb (0.7838) is consistent with known eager tied-head performance, making it impossible to attribute the outcome to the logit bias alone.

*Ledger: results.tsv, epoch 45.*
