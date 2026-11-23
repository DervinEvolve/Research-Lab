# Epoch 113: Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (Ro

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6678 (Δ vs best-at-time: -0.0092)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 339.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e113-15ef16b7 (43702894 bytes, SHA-256 `4941df283f313d952cf9bc2c3d5239096cba15d8833ce77a272ab4cbf5767186`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (RoPE, base=10000) on Q and K in CausalSelfAttention will lower val_bpb from 0.6770 to below 0.6675 within the 300s budget, because byte-level modeling depends heavily on relative token offsets that RoPE encodes directly without spending capacity learning translation invariance.

## Change
Removed the learned absolute position embedding (transformer.wpe) and its addition in GPT.forward. Added RoPE (base 10000, half-split pairing) in CausalSelfAttention: non-persistent cos/sin buffers of shape (1,1,seq_len,head_dim/2) are applied to Q and K after their rms_norm and before scaled_dot_product_attention; V is untouched. num_params no longer subtracts wpe, so params_m should still read 10.720; optimizer, schedule, batch shape and everything else are unchanged.

## Review
The measured 0.66783 is only 0.00917 below the 0.6770 best — under the 0.0095 keep margin — so a single run cannot establish a new best. The change also bundles RoPE with deletion of learned wpe, making the gain unattributable, and params_m stays 10.720 only because the wpe subtraction was removed while true capacity fell. This is a package confound, not a clean RoPE mechanism test.

## Challenger concerns
- 0.6770→0.66783 is a 0.00917 drop, short of both the stated target (0.6675) and the lab keep margin (0.0095). At noise 0.0048 this is ~1.9σ — the same single-run class as e110 (0.6754) and e112 (0.6721), which repeats then killed. Ledgering a RoPE win off this score would be a false finding.
- The edit is a package, not a RoPE mechanism test: learned wpe is deleted (~196608 AdamW params and the tok+pos sum) and RoPE is added. A gain cannot be credited to relative offsets rather than 'absolute PE was a liability' or a smaller AdamW set. Reported params_m stays 10.720 only because the wpe subtraction was also removed — true capacity fell.
- base=10000, head_dim=64, T=512 leaves most rotary frequencies nearly constant inside the window, so the run barely tests the claimed relative-offset mechanism even if the rotate is correct.
- RoPE cost a few steps (3564, 116.8M tokens, 3608MB) versus recent ~3639–3724-step runs, so the fixed 4100-step cosine ends at a slightly higher LR. Too small to explain 0.009 bpb (e110's full anneal was ~0.002) but it is a residual confound. No eval-gaming smell: tokens_seen matches 3564*64*512 and the budget was honored.

*Ledger: results.tsv, epoch 113.*
