# Epoch 50: Enabling the native `fused=True` flag in the AdamW optimizer will reduce CUDA kernel launc

- **Author:** DeepSeek-V4
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Enabling the native `fused=True` flag in the AdamW optimizer will reduce CUDA kernel launch and memory bandwidth overhead in eager mode, increasing step throughput by at least 10% and improving final val_bpb under the 300-second budget.

## Change
Added a `fused` keyword to `GPT.configure_optimizer` and pass `fused=(device == 'cuda')` when constructing AdamW in `train()`, so CUDA runs use PyTorch's fused AdamW kernel while CPU smoke tests still use `fused=False`. No other architecture, schedule, or data changes were made.

## Review
The experiment crashed before producing a RESULT line or any val_bpb, so it cannot be kept under the rules. The stack trace matches the known lazy Triton/torch.compile failure, meaning the fused AdamW hypothesis was not actually measured. Current best remains 0.7067 val_bpb; this run provides no completed comparison.

## Challenger concerns
- The run CRASHED before measuring anything — the error is the SAME lazy-Triton 'Failed to find C compiler' failure that has killed epochs 44-46. This has NOTHING to do with fused=True; torch.compile is still active and dies at warmup. The fused-AdamW hypothesis was never actually tested.
- Attribution is impossible: the current train.py has compile ON, but e49's measured baseline (3678 steps, 0.7352) ran with compile explicitly OFF. Even if this run hadn't crashed, comparing a compile-attempted run to an eager baseline would confound compile-vs-eager with fused-vs-unfused.

*Ledger: results.tsv, epoch 50.*
