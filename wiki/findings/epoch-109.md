# Epoch 109: Reducing only Muon's Newton–Schulz iteration count from 5 to 3 will lower val_bpb below 0.

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6901 (Δ vs best-at-time: 0.0131)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 344.5 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e109-15ef16b7 (43699663 bytes, SHA-256 `9c1b9dd23cff8b91e41171b49f89e542eabea479a814380153cd7745cd58ecb4`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Reducing only Muon's Newton–Schulz iteration count from 5 to 3 will lower val_bpb below 0.6770 under the fixed 300-second budget: cheaper optimizer updates will permit more training steps while retaining sufficient update orthogonalization. Change ns_steps=5 to ns_steps=3 only in GPT.configure_optimizers; preserve all other settings.

## Change
One token changed: in GPT.configure_optimizers the Muon constructor now gets ns_steps=3 instead of ns_steps=5. Everything else (model, LRs, 4100-step cosine schedule, batch/seq shape, eval call, RESULT line) is byte-for-byte unchanged. The Muon class default and zeropower_ns default remain 5 but are unused, since the value is passed explicitly.

## Review
Measured val_bpb regressed to 0.6901 against the 0.6770 baseline, falsifying the hypothesis. Cutting Muon ns_steps from 5 to 3 did raise throughput to 3,752 steps (122.9M tokens in 300.0s at 3,488 MB VRAM), but truncating the 5-step Newton-Schulz polynomial at 3 iterations degraded update quality more than the extra steps compensated.

## Challenger concerns
- Improvement hypothesis is falsified, but the harm magnitude is not a clean ns_steps effect. Steps rose to 3752 (122.9M tokens, steady ~410k tok/s) from the ~111M-token / ~3400-step reference, so the cut did run and the 'more steps' mechanism fired — and val_bpb still landed at 0.6901 vs 0.6770. Do not log a precise +0.013 harm: one seed, noise is 0.0068, and 0.0131 sits inside the 0.0137 margin.
- Newton-Schulz coeffs (3.4445, -4.7750, 2.0315) are the five-iteration polynomial. Three steps is an incomplete polar factor, not a retuned 3-step Muon. A negative here kills this one-token change only.
- Cosine is still step-indexed at 4100. This run finished at progress ~(3752-100)/4000≈0.91 (terminal LR ~0.12 of peak) versus ~0.82 (~0.17 of peak) if the reference is ~3400 steps. Mild versus the e108 batch-128 confound, but not zero.

*Ledger: results.tsv, epoch 109.*
