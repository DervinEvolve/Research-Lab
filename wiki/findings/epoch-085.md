# Epoch 85: Raising weight decay from 0.1 to 0.0 on the matrix parameters (only this change) will lowe

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7106 (Δ vs best-at-time: 0.0039)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 331.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e85-15ef16b7 (43699663 bytes, SHA-256 `c4f8dec34384aefc259453744d36df464ca315209a19d83c5fc0a5d7254d598c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Raising weight decay from 0.1 to 0.0 on the matrix parameters (only this change) will lower val_bpb below 0.7067, because a 4,300-step, 300s run is far from overfitting and decay on 10.7M params only slows fitting.

## Change
Changed the weight_decay argument passed to model.configure_optimizer from 0.1 to 0.0 in train(). This is the only edit; the two-group optimizer still applies the given weight decay to all p.dim()>=2 matrix parameters and 0.0 to scalar/bias parameters, so the change affects exactly the matrix parameter group as hypothesized.

## Review
Setting weight_decay from 0.1 to 0.0 on matrix parameters produced val_bpb=0.71058 across 4,271 steps in 300.06s, which is 0.00388 worse than the 0.7067 best and also trails the epoch 79 compile control (0.7078 at 4,346 steps). Because the run regressed on val_bpb with no improvement over baseline, we revert to weight_decay=0.1.

## Challenger concerns
- The target was missed: 0.71058 is 0.00388 worse than the current best. This supports reverting the change, not concluding that weight decay intrinsically improves this model.
- The hypothesis describes 0.1 to 0.0 as 'raising' weight decay, but the implementation correctly removes it. Its rationale also assumes that absence of overfitting makes decay harmful; decay can affect optimization and parameter scale even before overfitting.
- The supplied change and logs show no apparent evaluation tampering, leakage, or material budget violation. Attribution is limited by run variability, not an evident compound edit.

*Ledger: results.tsv, epoch 85.*
