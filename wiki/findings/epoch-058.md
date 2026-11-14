# Epoch 58: Setting total_steps_estimate=3300 (to match eager step count) and max_lr=2e-3 will yield v

- **Author:** GPT-5
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.8028 (Δ vs best-at-time: 0.0962)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 318.3 s

## Hypothesis
Setting total_steps_estimate=3300 (to match eager step count) and max_lr=2e-3 will yield val_bpb < 0.70, because the higher LR accelerates learning in the limited steps, and the proper cosine annealing to 2e-4 prevents overfitting; the champion's 1.2e-3 LR was too conservative for 3300 steps.

## Change
Disabled the torch.compile attempt and set run_model=model so this is the clean eager RMSNorm+SwiGLU run specified by the planner and recent notebook warnings. Changed max_lr from 1.2e-3 to 2e-3 and total_steps_estimate from 4100 to 3300, leaving AdamW standard/unfused and all architecture, batch, eval, and RESULT handling unchanged.

## Review
Run completed but regressed sharply to 0.8028 val_bpb (vs best 0.7067) after only 2520 steps. Challenger correctly flags the core confound: total_steps_estimate=3300 was based on an unverified throughput claim; actual eager rate caps at ~2520, leaving the 2e-3 LR schedule badly truncated and unannealed. No metric gaming occurred, but the hypothesis cannot be evaluated cleanly.

## Challenger concerns
- CRITICAL CONFOUND: the hypothesis assumed ~3300 eager steps, but the run only hit 2520 steps — identical to e52/e55/e57 (~2509-2533). The 273k tok/s rate caps us near 2520, NOT 3300. So total_steps_estimate=3300 means the cosine schedule only reached ~76% progress, leaving LR annealing stuck well above min_lr. The higher max_lr=2e-3 therefore never properly decayed. We cannot attribute the 0.8028 regression to 'high LR is bad' — it's high LR + truncated anneal, two tangled effects.
- The notebook (e56 Executor gemini) claimed eager RMSNorm+SwiGLU runs ~3300 steps/300s. This run — same config, eager — got 2520 steps. That 3300 figure is UNVERIFIED and has now been contradicted by e52/55/57/58 all landing ~2510-2530. The planner built the entire schedule on a wrong throughput number.
- No metric gaming: eval harness, batch sizes, eval coverage, and RESULT line are all untouched. VRAM 4207MB and seconds 300.05 are consistent with prior eager runs. The bad number is real, not a budget/leakage artifact. Good.

*Ledger: results.tsv, epoch 58.*
