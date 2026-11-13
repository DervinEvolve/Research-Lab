# Epoch 42: Explicitly disabling torch.compile (run_model = model, no try/except) will eliminate the 4

- **Author:** Claude
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7619 (Δ vs best-at-time: 0.0552)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 317.0 s

## Hypothesis
Explicitly disabling torch.compile (run_model = model, no try/except) will eliminate the 4th+ Triton crash and let an untied-head eager run reach the full ~2500 steps, dropping val_bpb below the 0.7583 floor because compile overhead and failures are the only remaining blocker to stable eager training.

## Change
Removed the torch.compile try/except block entirely; run_model is now unconditionally set to the eager model (no Triton/compile path). Untied the output head by deleting the `self.transformer.wte.weight = self.lm_head.weight` tying line (keeping the e40 untied-head config that reached 0.7583). Aligned the cosine horizon total_steps_estimate from 4100 to 2500 to match the real eager step budget. No other logic changed.

## Review
The run completed but val_bpb is 0.7619, worse than both the champion (0.7067) and the recent eager baseline (0.7583 from epoch 40). The hypothesis that removing torch.compile would drop val_bpb below the 0.7583 eager floor is falsified; instead we see a slight regression and fewer steps (2390 vs 2508). The change does not move the needle toward the champion and introduces an unattributable multi-factor confound.

## Challenger concerns
- The stated hypothesis is not cleanly attributable: the run combines explicit compile disable, untied head, and cosine horizon retuning. Untying and horizon alignment are not just safeguards if the comparison point is the current train.py.
- The result does not support the claimed direction. It reached 0.7619, worse than e40's 0.7583 untied-head eager result, and completed fewer steps: 2390 vs 2508.
- The comparison to e40 is potentially noisy because a ~0.0036 bpb delta may be within seed/system variance, while the train-step delta is nontrivial.
- No metric-gaming smell: evaluation still appears to use prepare.evaluate_bpb with normal seq_len/eval batch coverage, and budget overrun is small and consistent with checking wall time only between steps.

*Ledger: results.tsv, epoch 42.*
