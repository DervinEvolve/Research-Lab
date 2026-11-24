# Epoch 119: Adding a first-layer value residual to attention will reduce val_bpb below 0.6652 within 3

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6764 (Δ vs best-at-time: -0.0006)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.2 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e119-15ef16b7 (43699663 bytes, SHA-256 `ccfa6aec1ef723cf64d24b0c8b230c3c1728302ca3602e358709122620ba1436`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding a first-layer value residual to attention will reduce val_bpb below 0.6652 within 300 seconds by preserving byte-level information across depth. Cache the first block's projected V tensor as v0, leave that block unchanged, and use v_used = 0.5 * v + 0.5 * v0 in attention for blocks 2–6. Keep gradients through v0; change no other architecture, optimizer, or schedule settings.

## Change
Modified CausalSelfAttention.forward to accept an optional v0 tensor: when v0 is None (first block) it stores the projected V as v0 and uses it unchanged; otherwise it mixes the current projected V with v0 as 0.5*v + 0.5*v0. Modified Block.forward to accept and return v0, and modified GPT.forward to initialize v0=None and thread it through the block loop. No optimizer, schedule, data, or other architecture settings were changed.

## Review
The run finished cleanly at 0.6764 val_bpb after 3703 steps in 300.0s (3608 MB, 10.72M params), only 0.0006 below the 0.6770 best. That delta sits well inside the 0.0059 noise and cannot clear the 0.0118 confirmation margin, and it misses the hypothesized 0.6652 target. The ungated 0.5*v+0.5*v0 residual is a null; ~120 MB extra VRAM bought no attributable gain.

## Challenger concerns
- Effect size (-0.0006 val_bpb) is well within run-to-run noise (sigma=0.0059) and cannot clear the 0.0118 confirmation margin.
- Hardcoded 0.5 * v + 0.5 * v0 mixing without per-layer/per-head gating or V normalization may conflate mechanism failure with poor scaling.
- Retaining v0 with gradients across all blocks added ~120 MB peak VRAM (3608 MB vs ~3488 MB) for zero measured gain.

*Ledger: results.tsv, epoch 119.*
