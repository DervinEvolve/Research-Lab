# Epoch 33: Reducing AdamW weight_decay from 0.1 to 0.01 will lower val_bpb below the 0.7067 champion 

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7923 (Δ vs best-at-time: 0.0856)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 326.4 s

## Hypothesis
Reducing AdamW weight_decay from 0.1 to 0.01 will lower val_bpb below the 0.7067 champion because the model is trained in a strictly underfitting regime (single pass over <10% of TinyStories), where weight decay acts as an unnecessary regularizer that slows convergence under the 300-second wall-clock budget.

## Change
Changed the AdamW optimizer construction in train() from weight_decay=0.1 to weight_decay=0.01. Added a minimal guard around the excluded warmup/compile forward-backward: if torch.compile fails lazily on first use, it falls back to eager mode and retries the same warmup batch before starting the timed loop. No architecture, batch, schedule, evaluation, or timing logic was changed.

## Review
Measured val_bpb=0.7923 at 2389 eager steps, worse than champion 0.7067 (and prior eager 0.7775/0.7810). Compile fallback triggered, so result is confounded with the ~2400-step eager regime and does not test the hypothesis on the actual champion config. No improvement; revert.

## Challenger concerns
- CONFOUND: this run was NOT compared against the champion config. It ran at 2389 steps in eager mode (tok/s=260k, no torch.compile acceleration), yielding 0.7922 — which is in the SAME regime as the eager compile-off runs (e28=0.7775, e30=0.7810), NOT the champion 0.7067. The compile fallback silently triggered, so weight_decay's effect is confounded with the ~2400-step eager regime. We are comparing apples (eager 2389 steps) to oranges (champion was compiled, presumably far more steps/tokens).
- The real blocker is unaddressed: the champion 0.7067 (e14) required a working torch.compile reaching far more than 2389 steps. Every recent run (28-32) is stuck in the ~2400-step eager regime around 0.78. No optimizer tweak will close a 0.07 bpb gap that is actually a throughput/compile gap.

*Ledger: results.tsv, epoch 33.*
