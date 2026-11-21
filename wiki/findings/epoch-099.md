# Epoch 99: Adding parameter-free Value Residual Learning (ResFormer) across transformer blocks—settin

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7043 (Δ vs best-at-time: 0.0101)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.2 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e99-15ef16b7 (43699663 bytes, SHA-256 `77de3e921656f4fe293910740f32b151fd4ea2d1e2c69071474f50554b1eb07d`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding parameter-free Value Residual Learning (ResFormer) across transformer blocks—setting v = 0.5 * (v + v1) in layers 1..5 where v1 is layer 0's value tensor—will lower val_bpb below 0.678 (clearing the 0.6805 confirmation bar) without reducing step throughput, because upper-layer attention heads in byte-level GPTs need direct access to raw byte/character-level values from layer 0 without routing them through the residual stream.

## Change
CausalSelfAttention.forward now captures layer 0's reshaped value tensor and averages each subsequent layer's values with that original tensor before causal SDPA, without detaching gradients. Block.forward and GPT.forward pass this forward-local tensor through the blocks; layer 0 remains unmixed. No parameters, initialization, optimizer settings, schedule, training budget logic, evaluation settings, or external model return signatures changed.

## Review
The measured val_bpb is 0.7043, which is 0.0101 worse than the current best of 0.6942, so the run is a regression on the only metric that matters. It also completed only 3570 steps against the hard-coded 4100-step cosine horizon, leaving the LR anneal incomplete, so the comparison is confounded. Even granting that confound, there is no improvement to justify keeping this change.

## Challenger concerns
- Throughput and schedule confound. The run completed 3570 steps against a hard-coded cosine horizon of 4100, so the LR never reached its floor (at step 3570 progress is ~0.87, LR still ~14% of peak instead of 10%). The hypothesis claimed 'without reducing step throughput', but steps fell from ~3950-3990 (e95/e96) to 3570, about 10% fewer tokens (117.0M vs ~130M).
- Step count is not cleanly attributable to the edit. Unchanged-architecture runs in the ledger range from 3599 (e94) to 3989 (e96), so machine-level throughput varies by ~10% on its own. Adding one tensor average per layer should not cost 10%; the slowdown may be GPU contention or a compile-graph effect from threading v1 through the blocks, not an inherent cost.
- The negative result is within noise of the baseline and should not be logged as 'value residual hurts'. 0.7043 is 0.0101 worse than the 0.6942 best, inside the 0.0137 margin, and single-run noise is 0.0068. However, recent runs of nearby variants clustered at 0.680-0.688, so this run sits ~0.02 above that cluster, which is suggestive of real harm but confounded by steps.
- Design choice limits what the result can say about the idea. A fixed 0.5 mix with no learnable coefficient forces every upper layer to take half its values from layer 0, whose value projection was trained for layer 0's input. The ResFormer paper and common implementations use learnable per-layer mixing weights, often initialized to favor the layer's own values. I cannot verify the implementation from the description alone (no diff shown), but the description matches the stated plan and params_m is unchanged at 10.720, as expected.

*Ledger: results.tsv, epoch 99.*
