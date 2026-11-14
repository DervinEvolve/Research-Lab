# Epoch 54: Dynamically recomputing total_steps_estimate each 50 steps from observed step rate (so the

- **Author:** Gemini
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.7387 (Δ vs best-at-time: 0.0320)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 316.3 s

## Hypothesis
Dynamically recomputing total_steps_estimate each 50 steps from observed step rate (so the cosine horizon always ends near the 300 s mark) will let the eager LayerNorm+SwiGLU run finish annealing at ~0.1*max_lr instead of stalling at a high LR, dropping val_bpb below 0.7293.

## Change
Replaced RMSNorm with PyTorch's native nn.LayerNorm in Block and GPT classes, and updated _init_weights. Explicitly disabled compile by setting run_model = model to guarantee eager stability. Added dynamic total_steps_estimate calculation inside the logging step loop to align the cosine schedule decay perfectly with the actual step limit.

## Review
The run completed (3304 steps, 300s, 3.6GB) but landed at val_bpb 0.7387 — worse than the eager LN baseline's 0.7293 (e51) and nowhere near the 0.7067 compiled champion. The dynamic-horizon idea did not improve annealing; if anything the piecewise-adaptive schedule the Challenger flags left the LR trajectory noisier than a clean fixed cosine. No improvement on best means revert.

## Challenger concerns
- The dynamic recalculation of total_steps_estimate produces a schedule that is not a pure cosine with a fixed horizon; the effective LR trajectory is piecewise-adaptive and can differ substantially from a fixed-horizon cosine, confounding attribution of the result to 'annealing completion at 0.1×max_lr'.
- The observed val_bpb difference (0.7387 vs 0.7293) is only 1.2% and well within plausible evaluation noise for a single run; we cannot be confident that the change caused a real degradation rather than just natural variance.

*Ledger: results.tsv, epoch 54.*
