# Epoch 65: Wrapping the model in torch.jit.script (with eager fallback) will eliminate Python overhea

- **Author:** Gemini
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8027 (Δ vs best-at-time: 0.0961)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.8 s

## Hypothesis
Wrapping the model in torch.jit.script (with eager fallback) will eliminate Python overhead, boosting training throughput to >3500 steps under the 300s budget and lowering val_bpb below 0.75, because past compile-based runs achieved ~4100 steps and 0.7067 val_bpb but compile no longer works on this A10G; scripting is a lighter-weight acceleration that may succeed where compile fails.

## Change
Replaced the torch.compile configuration with a robust torch.jit.script compilation block. It attempts to compile the entire model first; if JIT scripting fails, it falls back to scripting individual transformer blocks, and finally to scripting MLP and RMSNorm layers individually, ensuring maximum possible compilation speedups without crash risks.

## Review
val_bpb=0.8027 is far worse than the 0.7067 champion, and 2366 steps is even below the prior eager ~2520 — jit.script added overhead rather than accelerating. The hypothesis required >3500 steps to exercise its mechanism; it never happened, so scripting failed just like compile. Clear regression, revert.

## Challenger concerns
- Scripting produced 2366 steps—below both prior eager (~2520) and claimed >3500—indicating net overhead or failed scripting rather than acceleration.
- Hypothesis required compile-like step count (>3500) to be falsifiable; the result lands in the eager regime, so the claimed mechanism (JIT eliminating Python overhead) was never exercised.

*Ledger: results.tsv, epoch 65.*
