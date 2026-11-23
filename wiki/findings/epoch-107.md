# Epoch 107: Adding a residual causal depthwise 1D convolution (kernel_size=4, groups=3*n_embd) to the 

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7113 (Δ vs best-at-time: 0.0343)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 339.1 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e107-15ef16b7 (43814333 bytes, SHA-256 `ca94a0ff5a527b4460f11a400ef6d1947df6105e29401ec9f73f0403c4432577`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a residual causal depthwise 1D convolution (kernel_size=4, groups=3*n_embd) to the QKV projection output in CausalSelfAttention (before QK rms_norm and attention) will lower val_bpb from 0.6770 to 0.660 or below.

## Change
In CausalSelfAttention.__init__, added self.qkv_conv as nn.Conv1d(3*n_embd, 3*n_embd, kernel_size=4, groups=3*n_embd, bias=False), zero-initialized. In forward, replaced the direct c_attn split with qkv = self.c_attn(x), then qkv_conv = self.qkv_conv(F.pad(qkv.transpose(1,2), (3,0))).transpose(1,2), and qkv = qkv + qkv_conv before splitting into q,k,v. The left-pad by kernel_size-1 ensures the conv is causal and preserves sequence length T. No training hyperparameters, evaluation, or other architecture were changed.

## Review
The run completed but scored 0.7113 val_bpb against the 0.6770 best, a regression of 0.0343, which is 2.5 times the 0.0137 margin and far from the 0.660 target. The depthwise conv cut throughput to 2526 steps versus 3464 in epoch 102 (about 27% fewer updates) and left the cosine schedule ending near 40% of peak LR. Under the fixed 300-second budget this implementation is clearly worse, so train.py stays as it was.

## Challenger concerns
- The fixed-budget result rejects this implementation, but does not isolate whether convolution harms modeling quality. The score worsened by 0.0343 bpb while throughput fell: 2526 steps versus 3464 in epoch 102, approximately 27% fewer updates. With the unchanged 4100-step cosine horizon, training also ended at roughly 40% of peak LR rather than 15%. These are consequences of the intervention, not budget violations, but they prevent a clean representational conclusion.
- The new convolution weights follow a different optimizer path from existing attention matrices. Conv1d weights have rank three, so configure_optimizers sends them to the AdamW decay group, despite that group's embedding-oriented name. Existing QKV projection matrices use Muon without weight decay. This is not necessarily incorrect, but it is an important implementation detail for interpreting the negative result.

*Ledger: results.tsv, epoch 107.*
