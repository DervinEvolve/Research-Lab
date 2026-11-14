# Epoch 66: Enabling TensorFloat-32 (TF32) matmuls on the A10G GPU in pure eager mode (disabling torch

- **Author:** Grok
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7733 (Δ vs best-at-time: 0.0667)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 312.9 s

## Hypothesis
Enabling TensorFloat-32 (TF32) matmuls on the A10G GPU in pure eager mode (disabling torch.compile) will increase training throughput to over 3300 steps within the 300s budget, allowing proper learning rate annealing and lowering val_bpb below 0.74.

## Change
Minimal edit: added TF32 enable (allow_tf32=True for matmul/cudnn) right after device selection; removed torch.compile try/except entirely so run_model=model (pure eager); set total_steps_estimate=3300 to match expected TF32 throughput. No other changes.

## Review
The run completed, but it regressed badly versus the current best: 0.7733 val_bpb vs 0.7067. The throughput premise was also falsified: 2521 steps in 300.05s is essentially identical to recent eager runs (~2520 steps), nowhere near the hypothesized >3300 steps. Because bf16 autocast likely bypasses TF32 and the schedule was set to 3300 despite only reaching 2521 steps, this is not a keeper and not a useful new baseline.

## Challenger concerns
- The hypothesis's core premise is falsified by its own log: TF32 did NOT increase throughput. 273k tok/s = 2521 steps is identical to the eager baseline from e64 (2524 steps). TF32 gave zero speedup, so the experiment did not actually test what it claimed — it just re-ran eager with a mis-set total_steps_estimate=3300.
- Result is worse than champion (0.7733 vs 0.7067) but for the wrong reason the Planner thinks. It's not that TF32 is bad — it's that eager throughput is capped at ~2520 steps AND the cosine schedule was tuned for 3300. This is a compound of a null intervention + a schedule bug, making the 0.7733 number unattributable.
- TF32 likely had no measurable effect because bf16 autocast is already active for all matmuls, so TF32 (an fp32 matmul mode) is mostly bypassed. The hypothesis didn't account for the existing autocast path.

*Ledger: results.tsv, epoch 66.*
