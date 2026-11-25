# Epoch 126: Changing the learning-rate schedule from immediate cosine decay after warmup to a warmup-s

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6709 (Δ vs best-at-time: -0.0061)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 335.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e126-15ef16b7 (43699663 bytes, SHA-256 `b1a7c46f3b3924c2b76c9bde265d9abc259ba804b8a65ba7ee036d5829305f63`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Changing the learning-rate schedule from immediate cosine decay after warmup to a warmup-stable-decay (WSD) schedule that holds max_lr=1.2e-3 until step 2400 and then cosine-decays to 0.1*max_lr by step 3400 will lower val_bpb below 0.6652. The current schedule decays too early in the 300s budget, spending most of the ~3400 actual steps at a reduced LR and underfitting the byte distribution.

## Change
lr_at_step now takes a decay_start_step argument: linear warmup for 100 steps, hold at max_lr until step 2400, then cosine decay to 0.1*max_lr at step 3400 (floor afterwards). In train(), added decay_start_step=2400, changed total_steps_estimate from 4100 to 3400, and passed the new argument at the single call site. The schedule applies to both AdamW and Muon through the existing lr_mult, as before; model, optimizer, batch shape and eval are untouched.

## Review
Run finished cleanly at 0.6709 val_bpb after 3722 steps / 122.0M tokens (3488 MB, 10.72M params), 0.00608 below the 0.6770 best but inside the 0.0118 confirmation margin and well short of the 0.6652 claim. The schedule also failed to isolate the hypothesis: decay end was moved from 4100 to 3400, and the last 322 steps sat at the LR floor. No cheating, but this is a within-noise negative, not a new baseline.

## Challenger concerns
- The observed improvement is within the confirmation margin, not evidence of a reproducible gain. The score improves on 0.6770 by only 0.00608 and misses the stated 0.6652 target.
- This is a valid schedule-level intervention, but it does not isolate the claim that early decay causes underfitting: both the plateau and the decay endpoint changed.
- The presumed 3400-step runtime underestimated the actual 3722 steps. Training therefore spent its final 322 updates, approximately 26 seconds, at the learning-rate floor.
- No evaluation tampering, validation leakage, or material budget violation is evident from the supplied change description and logs; the exact modified source was not supplied for inspection.

*Ledger: results.tsv, epoch 126.*
