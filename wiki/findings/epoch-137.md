# Epoch 137: Batch Muon's Newton–Schulz calculations across parameters of identical shape, preserving e

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6736 (Δ vs best-at-time: -0.0034)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e137-15ef16b7 (43699663 bytes, SHA-256 `6302ea835470230d684df637c3cd82c36bc3e16636c9203eb5d77a7e2ef3832e`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Batch Muon's Newton–Schulz calculations across parameters of identical shape, preserving each matrix's independent normalization and existing update rule. Reducing small GPU launches will increase training steps by at least 20% within 300 seconds and lower val_bpb below 0.6615 without changing the model or learning-rate schedule.

## Change
Updated zeropower_ns to operate on batched 3D tensors using last-two-axis transposes (transpose(-2, -1)) and per-matrix Frobenius norms (norm(dim=(-2, -1), keepdim=True)). Updated Muon.step to group parameters by (shape, dtype), maintain stacked Nesterov momentum buffers, run batched Newton-Schulz and aspect-ratio scaling per shape group, and apply updates via torch._foreach_add_. All hyperparameters, model architecture, LR schedule, and evaluation are unchanged.

## Review
The run finished at 0.6736 val_bpb after 3806 steps, 0.0034 under the 0.6770 best, but that sits inside the 0.0077 noise and far short of the 0.0155 confirmation margin and the predicted 0.6615. Steps rose only about 1-2% versus 3746-3770 on the same config, not the hypothesized >=20%, so Newton-Schulz launch overhead was not the bottleneck. With throughput matched, an equivalent Muon refactor has no mechanism for a real bpb drop, and the extra grouping plus torch._foreach_add_ loses on Rule 6.

## Challenger concerns
- The stated mechanism failed. The hypothesis predicted at least 20% more steps; the run got 3806, against 3746-3770 in recent same-config runs (e133-e135). That is roughly +1-2%, which is inside run-to-run throughput variation. Newton-Schulz launch overhead is evidently not the bottleneck at this model size.
- The val_bpb gain is within noise. 0.6736 is 0.0034 below the 0.6770 best, against measured noise of 0.0077 and a margin of 0.0155. It also missed the predicted 0.6615 by a wide margin. With steps essentially matched, there is no mechanism by which a mathematically equivalent refactor would lower bpb.
- Numerical equivalence of the refactor was not verified. Batched bfloat16 Newton-Schulz with per-matrix norms should match the per-parameter version, but the stacked momentum buffers and the use of the private torch._foreach_add_ are new places for a silent bug. I could not see the edited code, only its description.
- The change adds complexity for no measured benefit. Rule 6 says simpler wins on ties, and this adds shape-grouping logic and a private torch API to get a tie.

*Ledger: results.tsv, epoch 137.*
