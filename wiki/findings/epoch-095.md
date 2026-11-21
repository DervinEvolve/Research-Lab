# Epoch 95: Reducing only the fused c_attn Muon aspect multiplier from sqrt(3) to 1/sqrt(3), leaving a

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6802 (Δ vs best-at-time: -0.0140)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e95-15ef16b7 (43699663 bytes, SHA-256 `00f30fa07385a358523fbf77b7af54ea98086565fefa3ab4e140cf0897c9121e`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Reducing only the fused c_attn Muon aspect multiplier from sqrt(3) to 1/sqrt(3), leaving all other updates unchanged, will produce a two-run mean val_bpb below 0.6795. The mechanism is that jointly orthogonalized QKV projections need substantially smaller updates than the default rectangular-matrix scaling supplies.

## Change
In configure_optimizers, the six fused c_attn weights now go into their own Muon param group with aspect_mult = 1/sqrt(3); all other block matrices stay in the default group. Muon.step reads an optional per-group aspect_mult and otherwise uses the unchanged sqrt(max(1, rows/cols)) rule, so every other matrix gets exactly the update it had before. Muon LR 0.02, momentum, AdamW, init, fused QKV, the 4100-step horizon and the training loop are untouched.

## Review
0.68015 beats 0.6942 by only 0.01405, short of the 0.0147 margin, so a matched repeat still averages 0.68015 and misses the 0.6795 bar the hypothesis required. The run also saw 129.3M tokens versus 117.9M on e94, and a scalar aspect change cannot explain that throughput, so the gain is confounded. 0.68015 sits 0.001 from e94's unconfirmed aspect=1 mean of 0.6812, so 1/sqrt(3) is not isolated.

## Challenger concerns
- Token-count confound: this run trained 129.3M tokens vs e94's 117.9M and e89 baseline ~119M; a 9.7% increase in training tokens alone could explain the 0.014 val_bpb improvement. The wall-clock budget does not fix token count.
- Single-run variance: 0.68015 is only 0.01405 below the 0.6942 best, below the 0.0147 confirmation margin. e92/e94 showed run-to-run swings up to 0.0146, so this could be luck.
- Possible lr_mult omission in the new c_attn Muon group: if lr_mult was not set to muon_lr/lr, the train loop's g.get('lr_mult', 1.0) would give c_attn an LR of 1.2e-3 instead of 0.02, confounding update scale with learning rate.
- Redundancy with e94: e94's aspect_mult=1 gave a two-run mean of 0.6812; e95's 1/sqrt3 single run 0.68015 is only 0.001 better, within noise. This does not distinguish 1 from 1/sqrt3.

*Ledger: results.tsv, epoch 95.*
