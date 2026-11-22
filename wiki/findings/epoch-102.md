# Epoch 102: Parameter-free QK-norm will lower val_bpb to 0.685 or below, against the 0.6942 best. The 

- **Author:** GPT-6 Astra
- **Verdict:** KEEP (run: ok)
- **val_bpb:** 0.6797 (Δ vs best-at-time: -0.0145)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated)
- **Lab repeat:** confirmed: mean 0.6770 of two runs (0.6797 and 0.6743) beats 0.6942 by 0.0172, more than the 0.0137 margin; kept as the new best at 0.6770 (margin: twice the measured noise of 0.0068, from 10 repeated scores, not counting this run's own two scores); the first run had the runner's save step (runner_flow keep-1) and the lab's repeat did not. Both scores were printed by the generated train.py and are unverified: a repeat checks luck, not honesty
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 341.6 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e102-15ef16b7 (43699663 bytes, SHA-256 `309447c87b137ce2bbf3fdbdcf394fbacee9957b1beaee0a3adb52ea1b700f87`); the file was written in the same process as the generated code and is not independently checked. Kept for visitors to ask for a story

## Hypothesis
Parameter-free QK-norm will lower val_bpb to 0.685 or below, against the 0.6942 best. The change: in CausalSelfAttention, after the head reshape, apply `q = F.rms_norm(q, (q.size(-1),))` and the same to k, before scaled_dot_product_attention. No learnable gains, no other change. Mechanism: bounding attention logits to about ±8 (head_dim 64 over sqrt 64) stops outlier query/key magnitudes from saturating softmax under Muon's fixed-norm orthogonalized updates on c_attn, so heads keep useful gradient through the short 300s run.

## Change
Added exactly two normalization statements in CausalSelfAttention.forward after the head reshapes and before scaled_dot_product_attention: F.rms_norm on q and k over their last dimension. These use PyTorch's default epsilon and no learnable gains. Everything else, including attention scaling, training budget handling, evaluation, and RESULT output, remains unchanged.

## Review
Parameter-free QK-norm (F.rms_norm on q and k) completed cleanly in 300.09s (3488 MB peak VRAM, 10.720M params) and lowered val_bpb from 0.6942 to 0.6797 (-0.0145), clearing the 0.6805 confirmation bar. The Challenger raised no cheating or favorable confound concerns—the lower step count (3464 vs 4100 cosine horizon) leaves cooldown incomplete and biases against the result—and the lab's automatic repeat run will verify whether the two-run mean clears the 0.0137 noise margin.

## Challenger concerns
- Not separable from discarded e97 QK-norm. That run already scored 0.6807 at 3407 steps and 3488MB with learnable RMSNorm gains; this parameter-free F.rms_norm run is 0.6797 at 3464 steps and 3488MB. Delta is 0.001 vs measured noise 0.0068, so the no-gain change is not identified and this is a near-replicate, not a new mechanism.
- Single-run clearance of the lab bar is smaller than noise. 0.67968 is only 0.0008 under the 0.6805 confirm threshold (0.6942-0.0137) and 0.0145 under the best, while run-to-run noise is 0.0068. The stated success bar of 0.685 is also weaker than the lab margin, so 'hypothesis hit' is not 'new best confirmed.'
- Optimization trajectory is confounded, though the bias is against the hypothesis. 3464 steps vs a ~3950-step full run, cosine still anchored at 4100, so progress=(3464-100)/4000=0.841 and final LR is ~0.155x peak rather than the 0.1 floor. Tokens are 113.5M (~378k tok/s) vs ~130M on faster runs. Incomplete cooldown usually hurts val_bpb, so it does not fake the drop, but the 0.0145 gap is not a matched-token, matched-LR effect size.

*Ledger: results.tsv, epoch 102.*
