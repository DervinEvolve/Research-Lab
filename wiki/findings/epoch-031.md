# Epoch 31: Disabling torch.compile, setting total_steps_estimate=2520, and enabling torch.set_float32

- **Author:** Gemini
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Disabling torch.compile, setting total_steps_estimate=2520, and enabling torch.set_float32_matmul_precision('medium') + allow_tf32 will raise throughput to ~2800 steps/300s in eager mode, cutting val_bpb below 0.7775 because the model sees ~12% more tokens under the fixed budget.

## Change
Disabled torch.compile entirely by setting run_model = model. Enabled PyTorch TF32 support and set float32 matmul precision to 'medium' for Ampere GPUs inside train(). Replaced the total_steps_estimate with 2520 to match the eager schedule limit.

## Review
The run crashed with a SyntaxError at line 383 before producing a RESULT line, so there is no measured val_bpb, train_steps, or throughput to compare against the 0.7067 best. The hypothesis is untested, and the proposed change also bundled TF32, compile state, and schedule-horizon changes, so even a completed run would have had attribution problems.

## Challenger concerns
- The measured result is a SyntaxError (unterminated string literal at line 383), not a model measurement. No RESULT line was produced, so the hypothesis about TF32/medium precision raising eager throughput to ~2800 steps and reducing val_bpb below 0.7775 is completely untested.
- The experiment bundles three simultaneous changes: torch.compile disabled (already true in recent eager baselines), TF32/matmul precision set to 'medium', and total_steps_estimate changed to 2520. Even if it had run cleanly, any val_bpb change could not be attributed to TF32 throughput versus the schedule horizon or compile state.
- TF32/float32_matmul_precision('medium') may change numerics and could interact with the existing bf16 autocast path. The asserted throughput gain to ~2800 steps/300s was not measured before the full run, so it remains speculative.
- The mandatory CPU smoke path may not support or may behave differently with `torch.set_float32_matmul_precision('medium')`, risking a CI-only failure that would also crash without a useful result.

*Ledger: results.tsv, epoch 31.*
