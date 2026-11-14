# Epoch 60: Raising batch_size from 64 to 96 (and tokens per step accordingly) while keeping the 4100-

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Raising batch_size from 64 to 96 (and tokens per step accordingly) while keeping the 4100-step cosine horizon and max_lr=1.2e-3 will reach a lower val_bpb than 0.7067 because the A10G + compile configuration has headroom and the model will see ~50% more tokens inside the 300 s budget without destabilizing training.

## Change
Changed the training batch_size from 64 to 96 in the non-smoke branch (line 343). All other hyperparameters (max_lr=1.2e-3, total_steps_estimate=4100, gradient clipping, compile attempt) remain untouched. This is the only modification; the rest of the file is identical to the champion RMSNorm+SwiGLU configuration.

## Review
The run crashed during the lazy compilation phase at the first forward pass due to a persistent Triton backend hash failure. Since no valid metrics were produced, we must revert.

## Challenger concerns
- The crash is not attributable to batch_size=96. It reproduces the known torch.compile/Triton environment failure: Inductor needs a C compiler and fails during first compiled forward, before any training metric can test the batch-size hypothesis.
- The hypothesis depends on compile throughput/headroom, but current lab evidence says compile is persistently broken on this A10G setup. The experiment therefore tests an unavailable execution mode rather than a learning-dynamics change.
- Keeping total_steps_estimate=4100 is an unaddressed confound if the run falls back to eager or if batch_size=96 slows step rate. The LR schedule would remain too hot at the 300s cutoff, mixing batch-size effects with schedule mismatch.
- No sign of metric gaming, validation leakage, eval tampering, or budget cheating; the failure is an implementation/runtime confound.

*Ledger: results.tsv, epoch 60.*
