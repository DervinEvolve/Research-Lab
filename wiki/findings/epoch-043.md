# Epoch 43: Reverting the eager experiments entirely back to the champion train.py (tied head, 4100-st

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8246 (Δ vs best-at-time: 0.1179)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 318.9 s

## Hypothesis
Reverting the eager experiments entirely back to the champion train.py (tied head, 4100-step horizon, LR 1.2e-3) BUT keeping the torch.compile try/except AND wrapping the compile call so that on ANY Triton/compile failure it falls back to eager mid-run is unnecessary — instead, the single highest-value move is to re-run the UNMODIFIED champion configuration to confirm whether compile is truly dead; if compile succeeds we land ~0.7067, if it crashes we learn the floor is now eager-only. More concretely: raising max_lr to 2.0e-3 with total_steps_estimate=2400 in forced-eager mode will cut eager val_bpb below 0.74 because the eager regime only reaches ~2400 steps and the 1.2e-3 peak (tuned for 4100 compiled steps) leaves the cosine under-decayed and the model under-trained at this shorter horizon.

## Change
I disabled torch.compile by replacing the compile try/except with `run_model = model`, preserving a clean forced-eager measurement. I changed only the LR schedule knobs for the hypothesis: `max_lr` is now 2.0e-3 and `total_steps_estimate` is now 2400. Architecture, batch/sequence shape, tied embeddings, evaluation, and RESULT contract are unchanged.

## Review
The experiment resulted in a significant regression to 0.8246 val_bpb, far worse than the current best of 0.7067. Raising the learning rate to 2.0e-3 with a tied head in eager mode was highly detrimental, likely causing instability or under-training.

## Challenger concerns
- The hypothesis implicitly compares against eager runs that used untied heads (e.g., 0.758–0.786), but this experiment reverted to tied embeddings. The drop to 0.8246 could be partially due to the tied head being more sensitive to high LR, rather than the LR increase alone being harmful.

*Ledger: results.tsv, epoch 43.*
