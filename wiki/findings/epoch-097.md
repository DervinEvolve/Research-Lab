# Epoch 97: Adding RMSNorm to the query and key projections inside each attention head (QK-Norm) befor

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6807 (Δ vs best-at-time: -0.0135)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 341.7 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e97-15ef16b7 (43708971 bytes, SHA-256 `f11952f6124ba1d52830579db9e9ca835155dcab7788602472bcd158ab02677c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding RMSNorm to the query and key projections inside each attention head (QK-Norm) before the dot product will lower val_bpb below 0.6795 (the confirmation bar), because byte-level sequences have highly variable token frequencies that create outlier Q/K magnitudes, and normalizing them stabilizes attention logits and improves gradient flow.

## Change
Only CausalSelfAttention changed. Each block now has q_norm and k_norm (RMSNorm over head_dim=64, scale shared across heads), applied to Q and K after the per-head reshape and before scaled_dot_product_attention. V, SDPA's 1/sqrt(d) scale, Muon, tied embeddings, init, and the schedule are untouched. The new 1D scales are 768 params total, init to ones via the existing RMSNorm path, and fall into AdamW with no decay.

## Review
QK-Norm scored 0.6807 against the 0.6942 best, an improvement of 0.0135. That is short of the 0.0147 confirmation margin, since the run needed to reach 0.6795 or lower. The change cut throughput to 3407 steps (111.6M tokens) from the 3599-3989 steps of recent runs. The cosine schedule is tied to a fixed 4100-step horizon, so a run that finishes fewer steps ends before the LR has fully decayed, and that confounds the comparison. The score also sits in the same 0.680-0.685 band as e92/e94/e95. The result is unconfirmed rather than disproven.

## Challenger concerns
- Single-run score (0.6807) missed the 0.6795 confirmation bar (-0.0135 vs 0.6942, < 0.0147 margin) and sits in the same 0.680-0.685 noise band as e92/e94/e95.
- Throughput and LR-schedule confound under fixed 300s budget: QK-Norm reduced steps to 3407 (111.6M tokens, +297 MB VRAM) vs 3599-3989 steps in e94-e96.
- Clean single-variable attribution and rule compliance: +768 params (10.7209M), 300.06s wall-clock respected, no eval or metric gaming.

*Ledger: results.tsv, epoch 97.*
