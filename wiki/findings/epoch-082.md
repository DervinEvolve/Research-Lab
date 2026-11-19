# Epoch 82: Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (Ro

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7208 (Δ vs best-at-time: 0.0142)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 341.7 s

## Hypothesis
Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (RoPE, base=10000) applied to Q and K in CausalSelfAttention will lower val_bpb below 0.703 (beating both the 0.7078 compile control and 0.7067 best) because byte-level modeling requires translation-invariant relative position encoding across arbitrary byte offsets.

## Change
Removed the learned absolute positional embedding (wpe) from GPT and replaced it with Rotary Position Embeddings (RoPE, base=10000) applied to query and key tensors inside CausalSelfAttention. Added precomputed cos/sin buffers for efficiency and updated num_params() to reflect the absence of wpe. All other architecture, training loop, LR schedule, and evaluation paths remain unchanged.

## Review
RoPE measured 0.7208 val_bpb at 4062 steps, worse than the 0.7067 best and the 0.7078 same-mode compiled control (e79, 4346 steps), and far from the predicted <0.703. The 0.0130 regression is well beyond the ~0.004 run-to-run swing seen in e79-e81, so this is a real loss under the fixed budget. The run also lost 6.5% of its steps (4346 -> 4062), which removed the min-LR tail, so the result rejects this implementation at this budget but does not isolate RoPE's representational value.

## Challenger concerns
- The fixed-budget result rejects this implementation, but does not isolate RoPE's representational benefit. Validation worsened by about 0.0130 bpb versus e79 while completed steps fell 6.5%, from 4346 to 4062. The unchanged schedule also loses the control's approximately 246-step minimum-LR tail.
- The edited source is not supplied, so rotation correctness and dtype behavior cannot be verified. Precomputed float32 cos/sin buffers can promote autocast Q/K, increase overhead, or change attention kernel dispatch.
- Removing wpe can change random-number consumption during initialization, so retaining seed 1337 does not guarantee identical initial weights for shared layers. A single historical comparison cannot cleanly estimate the positional-encoding mechanism.
- The unchanged reported parameter count is an accounting artifact: the original num_params() already excluded wpe. Actual trainable parameters decrease by 196608 when the 512-by-384 table is removed.

*Ledger: results.tsv, epoch 82.*
