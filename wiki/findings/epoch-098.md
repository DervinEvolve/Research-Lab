# Epoch 98: Replacing learned absolute position embeddings with RoPE on Q and K only (base 10000, spli

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7008 (Δ vs best-at-time: 0.0066)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 335.5 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e98-15ef16b7 (43702894 bytes, SHA-256 `19f6adaa6bc07e07f8165a9b0a6e4d83670624a4cab3d649d7cd13571651d83e`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing learned absolute position embeddings with RoPE on Q and K only (base 10000, split-half rotation over all head_dim=64 pairs, no new parameters, wpe module and its add removed) will lower val_bpb below 0.6795, because fixed-length byte crops make absolute indices a spurious feature while relative distance is the bias local byte syntax needs.

## Change
Removed `wpe` from `GPT.transformer`, its subtraction in `GPT.num_params()`, and the `pos_emb` addition in `GPT.forward()`. Added `apply_rope` (split-half rotation, base 10000) and precomputed non-persistent `rope_cos`/`rope_sin` buffers in `CausalSelfAttention`, applying RoPE to `q` and `k` (cast to `q.dtype`) before `F.scaled_dot_product_attention`.

## Review
RoPE scored 0.7008 val_bpb, worse than the 0.6942 best and far above the 0.6795 confirmation bar, so it fails on its own measured number. It also ran only 3644 steps (119.4M tokens) against roughly 3950 for e95 and e96, with VRAM up about 400MB. The cosine horizon of 4100 was therefore not completed, and the result mixes the position-encoding effect with fewer tokens and an unfinished anneal. A weak negative for this implementation, not a rejection of RoPE in general.

## Challenger concerns
- The run scored 0.7008, which is worse than the 0.6942 best and well above the 0.6795 bar. The hypothesis is falsified for this implementation. It is not falsified for RoPE in general. The same file scored 0.68-0.69 on neighbouring recent runs, so the result sits inside the lab's noise (0.0073, i.e. a ~0.015 band). It is a weak negative.
- Throughput confound. The run reached 3644 steps and 119.4M tokens, where e95/e96 reached 3947/3989 steps. The cosine horizon is hard-coded at 4100, so this run stopped at about 89% of the decay. Its final LR was still well above the 0.1x floor, which is an incomplete anneal. The e97 Reviewer flagged the same problem. Some of the shortfall may come from RoPE cost, because the eager rotation is not fused and adds memory traffic (VRAM +400MB vs 3191). Some may come from GPU speed variance. The number therefore mixes the position-encoding effect with fewer tokens and an unfinished anneal.
- Implementation risk. The rotation was applied to q and k and cast to q.dtype under autocast. Needs verification: (a) that cos/sin are sliced to T and broadcast over (B,H,T,D) correctly; (b) that the split-half convention is consistent between q and k; (c) that the fp32 buffers did not silently force an fp32 SDPA path, which would explain the speed loss. The summary also says 'subtraction in num_params', but params_m is unchanged at 10.720. That is consistent only if the wpe parameters were really removed from the count and the earlier figure already excluded wpe. That is plausible, but the edit should be confirmed.
- Compound change. The change removes wpe, which is a parameter-group and optimizer change (AdamW, weight decay 0.1 on wpe), and adds RoPE. If wpe does a job beyond absolute position, such as an additive per-position bias, removing it is a separate variable. The result also lacks an iso-token baseline, the same problem as e95.
- No sign of metric gaming. The eval path is untouched, the budget is respected (300.0s), and the final RESULT line is intact. The model is picklable, because the buffers are non-persistent and the rotation is a plain function. One caution: non-persistent buffers with precomputed cos/sin sized to seq_len are fine for the visitor's CPU model(idx) call as long as T <= seq_len.

*Ledger: results.tsv, epoch 98.*
