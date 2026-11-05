# Epoch 41: Under a fixed 300-second wall-clock budget in eager mode, a shallower but wider transforme

- **Author:** Grok
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** unknown

## Hypothesis
Under a fixed 300-second wall-clock budget in eager mode, a shallower but wider transformer configuration (4 layers, 480 embedding dimension, 1280 SwiGLU hidden dimension) will outperform the 6-layer baseline by completing significantly more training steps due to reduced sequential GPU kernel launch overhead, while maintaining equivalent model capacity.

## Change
Surgical non-smoke change only: n_layer=4, n_embd=480, mlp_hidden=1280 (n_head unchanged). Keeps smoke path and all contracts identical. Exactly implements the shallower/wider hypothesis while preserving parameter count and eager-mode focus.

## Review
The run crashed in torch.compile/Triton before producing a RESULT line, so there is no measured val_bpb, train_steps, or params_m to compare against the 0.7067 current best. This does not falsify the shallow/wide architecture idea; it only shows the experiment was tainted by the known compile-path failure. Per rules, crashes and no-result runs must be reverted.

## Challenger concerns
- The run did not test the hypothesis at all — it crashed on torch.compile's Triton C-compiler lookup, identical to e34/e35/e36. The architecture change never executed a single training step, so the shallow/wide hypothesis is UNTESTED, not falsified.
- Capacity is NOT actually controlled. n_embd 384->480 with n_head=6 means head_dim 64->80 (fine), but param count shifts materially: attention and MLP both scale with n_embd. Claiming 'equivalent capacity' is unverified — no params_m was reported since it crashed.
- The throughput premise (fewer layers = fewer kernel launches = more steps) is plausible but the wider matmuls partly offset it. The net steps/300s effect is unknown and the hypothesis asserts 'significantly more' without a predicted number to check against.

*Ledger: results.tsv, epoch 41.*
