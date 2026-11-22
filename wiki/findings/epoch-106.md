# Epoch 106: Re-applying RoPE (half-split, base 10000, after QK rms_norm, wpe removed) plus a single ch

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6757 (Δ vs best-at-time: -0.0013)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 344.0 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e106-15ef16b7 (44489326 bytes, SHA-256 `82a80446bbdd0be872c571b4e3cf72bf45fec4cd5f164da16187631ebbf4066e`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Re-applying RoPE (half-split, base 10000, after QK rms_norm, wpe removed) plus a single change, raising the MLP/attention-free throughput by removing the now-unused wpe from the optimizer, is not needed; instead the one change is: RoPE with base 1000 instead of 10000 will lower val_bpb to 0.660 or below versus 0.6770 best, because byte-level TinyStories at seq_len 512 needs sharper local-offset resolution and faster-rotating frequencies give more discriminable short-range positions.

## Change
Removed the learned absolute position embedding (wpe) from GPT, its addition in GPT.forward, and its subtraction in GPT.num_params. Added half-split rotary position embeddings (rotate_half and non-persistent fp32 cos/sin buffers with base=1000.0) in CausalSelfAttention, applied to q and k after QK rms_norm and cast to q.dtype. All other architecture, optimizer, schedule, and training loop settings are unchanged.

## Review
RoPE base 1000 scored 0.6757 after 3417 steps in 300.07 seconds, missing the 0.660 target. Its 0.0013 improvement over 0.6770 is well inside measured noise (0.0068) and does not establish the required 0.0137 confirmation margin. It also trails epoch 105's base-10000 score of 0.6655, but differing step counts and unconfirmed single runs prevent attributing that difference confidently to the rotary base.

## Challenger concerns
- The measured 0.6757 is only 0.0013 below the 0.6770 best, well inside the 0.0068 run-to-run noise. It is also 0.0102 worse than e105's base 10000 result of 0.6655, so the data actually suggest base 1000 is harmful, not helpful.
- Relative to the current best, the change is compound: wpe removal plus RoPE base 1000. Attributing the effect to base 1000 relies on comparing to e105's reverted single-run result, which was itself sub-margin and unconfirmed. Step count also differs (3417 vs e105's 3658), which shifts the cosine schedule position and confounds the comparison.

*Ledger: results.tsv, epoch 106.*
