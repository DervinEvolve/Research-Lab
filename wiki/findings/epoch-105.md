# Epoch 105: Replacing the learned absolute position embedding (wpe) with rotary position embeddings (R

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6655 (Δ vs best-at-time: -0.0115)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.5 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e105-15ef16b7 (44489326 bytes, SHA-256 `afd3e4722aa6b405918ec02178d4ab839561181bb633a02d3b105b45d1e69656`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing the learned absolute position embedding (wpe) with rotary position embeddings (RoPE, base 10000, head_dim 64) applied to q and k after the existing QK rms_norm will lower val_bpb to 0.663 or below, against the 0.6770 best. Mechanism: byte-level prediction is dominated by relative offsets (the previous few bytes, word and sentence boundaries). An absolute wpe forces the model to relearn each local pattern at all 512 positions, while RoPE gives every head translation-invariant relative addressing from step 0. Nothing else changes: remove wpe and its add, drop the wpe subtraction in num_params, and keep the cos/sin tables as non-persistent buffers or compute them in forward.

## Change
Removed learned absolute wpe (ModuleDict, the tok+pos add, and the num_params subtraction) so reported params stay the same. After the existing QK rms_norm, q and k get Llama-style RoPE (base 10000, duplicated freqs, half-split rotate) from non-persistent fp32 cos/sin buffers sliced to T and cast to q.dtype. head_dim comes from config (64 in the real run, 32 under --smoke). No LR, schedule, width, or optimizer changes.

## Review
The run completed with val_bpb 0.6655, improving on the 0.6770 best by only 0.0115, which is below the 0.0137 confirmation margin. The Challenger's margin check is decisive: one run at about 1.7x noise cannot establish a new best, and the wpe-to-RoPE swap plus the +194-step throughput swing leaves the gain unattributable. Reverting preserves a clean baseline until RoPE is isolated and fixed-step retested.

## Challenger concerns
- Margin shortfall against the 0.0137 confirmation bar: 0.6655 beats the 0.6770 best by 0.0115 val_bpb, which is ~1.7x the 0.0068 noise floor but falls short of the 0.0137 margin (and missed the 0.663 target). Unless the repeat run scores <= 0.6611, the two-run mean will fail confirmation.
- Paired swap (removing learned wpe + adding RoPE) plus +194 steps vs e102 (3658 vs 3464). Step count is inside the known 3452-3709 throughput band, and e103 at 3709 steps only scored 0.6769, so extra steps alone do not explain 0.6655; however, removing wpe vs keeping wpe+RoPE is not isolated.

*Ledger: results.tsv, epoch 105.*
