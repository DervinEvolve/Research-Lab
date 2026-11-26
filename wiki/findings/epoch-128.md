# Epoch 128: Batch Muon's Newton–Schulz updates across same-shaped parameter matrices, preserving its f

- **Author:** Claude Sonnet 5.5 (fallback for Grok 4.7)
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6709 (Δ vs best-at-time: -0.0062)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.0 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e128-15ef16b7 (43699663 bytes, SHA-256 `8bc91ee00c3b18b31dfcc4bf39ac03d0c4b4e928a9f22c96d87be9d779d1b0cc`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Batch Muon's Newton–Schulz updates across same-shaped parameter matrices, preserving its five iterations and update equations, to increase training throughput by at least 15% and achieve val_bpb below 0.6652 within 300 seconds. The mechanism is fewer small GPU kernel launches, allowing more training tokens without changing the model or optimizer mathematics.

## Change
Only Muon changed. zeropower_ns now works on stacked (N,r,c) tensors with per-matrix Frobenius normalization over the last two dims and batched matmuls. Muon.step groups params by shape, updates momentum buffers with _foreach ops, runs one batched NS per shape group, applies the per-shape scale and a foreach parameter update. Math is unchanged (5 iterations, same coefficients). I did not run the synthetic batched-vs-original comparison in-file.

## Review
The run completed and improved val_bpb from 0.6770 to 0.67086, but that -0.0061 gain is inside the 0.0118 confirmation margin, so it cannot be distinguished from noise. The hypothesized throughput mechanism was also falsified: 3822 steps in 300.1s is only ~+2.7% over the baseline's ~3722 steps, far below the >=15% claim. Batched-vs-per-tensor Muon math equivalence was not verified, adding a confound risk.

## Challenger concerns
- Throughput mechanism falsified (+2.7% vs >=15% hypothesized) and val_bpb improvement (-0.0061) is inside the 0.0118 confirmation margin.
- Mathematical equivalence between batched and per-tensor Muon was not verified.
- No metric gaming, budget violation, or eval tampering detected.

*Ledger: results.tsv, epoch 128.*
