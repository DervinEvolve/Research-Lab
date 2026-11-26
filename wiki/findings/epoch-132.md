# Epoch 132: Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (Ro

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6767 (Δ vs best-at-time: -0.0004)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 348.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e132-15ef16b7 (42912530 bytes, SHA-256 `34500fc0c115e489f1b4e5357ff9cf7f24c70adbec23352924ceb0a7ce3f2cb2`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing learned absolute positional embeddings (wpe) with Rotary Position Embeddings (RoPE, base=10000) applied to Q and K in CausalSelfAttention will lower val_bpb from 0.6770 to <=0.6615 in 300s, because byte-level modeling relies on translation-invariant relative offsets (1-8 byte spans to form subwords) that absolute wpe must relearn independently at every position 0..511.

## Change
Removed the learned absolute positional embedding wpe from GPT (both the ModuleDict entry and the forward pass). Added rotary position embeddings (RoPE, base=10000) inside CausalSelfAttention: after RMSNorm on q and k, a helper _apply_rope rotates the two halves of the head dimension using cos/sin derived from absolute positions. All other model components, optimizer routing, and training loop remain unchanged.

## Review
RoPE in place of wpe measured 0.6767 against the 0.6770 best. That is a 0.0003 delta, far inside the 0.0155 margin and the 0.0077 run-to-run noise, and it misses the <=0.6615 target by about 0.015. The run also managed only 3234 steps and 106M tokens, roughly 10% below the best run, so the cosine ended at progress of about 0.77 and the comparison is confounded. This is a null result: RoPE is not worse per token, but nothing shows it is better.

## Challenger concerns
- The result is null: 0.6767 vs 0.6770 is a 0.0003 delta, far inside the 0.0155 margin and the 0.0077 measured noise. The target of <=0.6615 was missed by about 0.015. Reading this as 'RoPE equals wpe' or 'RoPE fails' is unsupported either way.
- Throughput confound: 3234 steps and 106M tokens, versus about 3600 steps and about 120M tokens for the best run. That is roughly 10% fewer steps. The RoPE ops on q and k are un-fused and add overhead, and the log shows throughput dips (320k tok/s at step 2350). Equal bpb at 10% fewer tokens could mean RoPE is slightly better per token, but this was not tested.
- The schedule is step-indexed with total_steps_estimate=4100. With 3234 steps the cosine stops at progress of about 0.77, so the LR never decays fully. Fewer steps therefore also means a higher final LR, which handicaps this run in a way unrelated to RoPE.
- Implementation is unverified. wpe was removed and params_m is unchanged at 10.720 because num_params already excluded wpe. That is consistent, but there is no evidence of the rotation convention (half-split versus interleaved), the dtype of cos/sin under autocast, or causal correctness. Applying RoPE after qk-norm is also an unusual ordering. A subtle bug would look just like a null.
- There is no sign of metric gaming. The eval harness is untouched and the budget is respected at 300.05s. The hypothesis premise that wpe must be relearned at every position is also weak, since at seq 512 with batch 64 every position gets plenty of gradient.

*Ledger: results.tsv, epoch 132.*
