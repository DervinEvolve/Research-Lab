# Epoch 88: Replacing AdamW with Muon (5-step Newton-Schulz orthogonalized Nesterov momentum, max_lr=0

- **Author:** Claude Fable 5.2
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6895 (Δ vs best-at-time: -0.0171)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 330.7 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e88-15ef16b7 (43699663 bytes, SHA-256 `becebb0d0ee196e80acac81d02a6487ccd96024875808e13bd11d3cebfe6b8d2`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Replacing AdamW with Muon (5-step Newton-Schulz orthogonalized Nesterov momentum, max_lr=0.02, momentum=0.95) for all 2D hidden-layer weight matrices while keeping AdamW (max_lr=1.2e-3) for embeddings and 1D norm parameters will lower val_bpb below 0.7020 in 300s because orthogonalized updates dramatically improve per-step conditioning on small transformer matrices with negligible wall-clock overhead.

## Change
Added a Muon optimizer class (Nesterov momentum 0.95, 5-step quintic Newton-Schulz, bfloat16 on CUDA and float32 on CPU, same-shaped matrices batched) and changed configure_optimizer to return (adamw, muon). Muon takes every 2D weight under transformer.h.* (c_attn, both c_proj, c_fc, c_gate) at max_lr 0.02; AdamW keeps the tied wte/lm_head and wpe (wd 0.1) plus the RMSNorm gains at 1.2e-3. Both follow the unchanged warmup/cosine schedule (horizon 4100), with Muon's LR set to the AdamW LR times 0.02/1.2e-3. Not specified by the Planner and chosen by me: Muon has no weight decay and uses the standard sqrt(max(1, rows/cols)) update scale; the global grad clip at 1.0 is left in place.

## Review
Measured 0.6895 beats 0.7067, but the run is confounded: hidden matrices changed from AdamW lr 1.2e-3/wd 0.1 to Muon lr 0.02/wd 0 plus a sqrt(rows/cols) update scale, and it completed only 3972 steps vs ~4346 control with the same 4100-step horizon, so the schedule is not matched. No matched-LR AdamW control exists, so the 0.0172 gain cannot be attributed to Newton-Schulz orthogonalization. Revert; isolate Muon with matched LR/decay/schedule next.

## Challenger concerns
- The 0.017 bpb drop is not attributable to Newton-Schulz orthogonalization alone. Hidden matrices also moved from AdamW lr 1.2e-3 and weight decay 0.1 to Muon lr 0.02 (16.7x), momentum 0.95, update scale sqrt(max(1,rows/cols)), and weight decay 0. Executor added the last two; they were not in the Planner spec. e85 already showed wd=0 under AdamW does not help, so decay is an unlikely sole cause, but no matched-LR AdamW control exists.
- Negligible-overhead claim is false, and the schedule is not matched to the control. Throughput is ~434k tok/s vs ~475k on the compiled AdamW control: 3972 vs ~4346 steps (~9% less data). Horizon stayed 4100, so this run stopped just before the LR floor while the control sits on the floor for ~250 steps. The bpb win despite less compute supports better per-step updates, but not a free optimizer.
- Single-run variance cannot invent this gap, but 0.6895 must not be locked as the point estimate. e87 same-recipe scatter was ~0.008 (0.7155 vs 0.7078). This gap vs 0.7067 is 0.0172, well above the 0.002 margin. Tokens, params (10.72M), VRAM (3191MB), and late loss ~0.47 nats vs 0.690 bpb are internally consistent; no eval-coverage or budget cheat is visible.

*Ledger: results.tsv, epoch 88.*
