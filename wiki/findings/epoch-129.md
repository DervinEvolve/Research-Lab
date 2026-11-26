# Epoch 129: Adding a learnable per-channel residual-stream scaling (x = x + a_l * attn(ln(x)), x = x +

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6825 (Δ vs best-at-time: 0.0055)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 342.7 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e129-15ef16b7 (43721963 bytes, SHA-256 `77b4a63459203602c66a498e3e985e2ef503136c606d78be4b716f0ea7f8f556`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a learnable per-channel residual-stream scaling (x = x + a_l * attn(ln(x)), x = x + m_l * mlp(ln(x)), with scalars initialized to 1.0 and trained under AdamW with no weight decay) will lower val_bpb from 0.6770 to below 0.6652 in 300s, because it lets each layer tune its contribution cheaply. Zero-parameter-count overhead and no extra kernels beyond fused elementwise multiplies keep throughput near 3,600 steps.

## Change
Added learnable per-channel residual-stream scaling parameters (`resid_scale_attn` and `resid_scale_mlp`, shape `(cfg.n_embd,)`, initialized to 1.0) to `Block.__init__` and applied them in `Block.forward` as `x = x + self.resid_scale_attn * self.attn(self.ln_1(x))` and `x = x + self.resid_scale_mlp * self.mlp(self.ln_2(x))`. Because these are 1D parameters, `GPT.configure_optimizers` automatically routes them to the AdamW `no_decay` group (weight_decay=0.0). No other files or parts of `train.py` were modified.

## Review
The run measured 0.6825 val_bpb, which is +0.0055 worse than the 0.6770 best, so it fails the basic keep condition regardless of noise. It also completed only 3,362 steps (110.2M tokens) against roughly 3,600-3,700 for the baseline, falsifying the hypothesis's claim of throughput neutrality. The regression sits inside the 0.0118 noise margin and is confounded with the token deficit, so we cannot say the scales are harmful, only that they did not help.

## Challenger concerns
- The claimed throughput neutrality is falsified: 3,362 steps vs ~3,600 baseline is a ~6.6% token deficit. Fewer training tokens alone can explain the +0.0055 val_bpb, so the scaling mechanism is confounded with training budget.
- Single run, no repeat; val_bpb 0.6825 vs 0.6770 best is +0.0055, inside the 0.0118 noise margin. Cannot distinguish mechanism from run-to-run variance.
- Per-channel scales initialized at 1.0 are optimized by AdamW with full max_lr=1.2e-3 and no weight decay; without constraints they may drift and alter effective residual gain, interacting with Muon's orthogonalized weight updates in an uncontrolled way.

*Ledger: results.tsv, epoch 129.*
