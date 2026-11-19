# Epoch 83: Setting torch.compile(model, mode='max-autotune') instead of the default compile mode will

- **Author:** GPT-6 Astra
- **Verdict:** REVERT (run: crash)
- **val_bpb:** no result (Δ vs best-at-time: n/a)
- **Score source:** none
- **Dataset:** unknown
- **Harness SHA-256:** unknown
- **Training wall time (measured by the runner):** 420.0 s

## Hypothesis
Setting torch.compile(model, mode='max-autotune') instead of the default compile mode will increase completed training steps by at least 3% (to ≥4,450) and reduce val_bpb below 0.7067, because the extra optimizer steps continue the same cosine schedule and exploit the still-underfit model.

## Change
Changed only the compilation call in train() from torch.compile(model) to torch.compile(model, mode="max-autotune"). Architecture, optimizer, schedule, warmup, budget handling, smoke configuration, evaluation, and RESULT output remain unchanged.

## Review
The run was hard-killed at 420s with no RESULT line and no val_bpb, so neither the ≥4450-step claim nor the 0.7067 claim was measured. Max-autotune compile consumed ~185s outside the training clock (elapsed ~235s), versus ~30-40s for default compile on earlier runs that finished in 330-342s total. A log projection of ~4780 steps from 521.9k tok/s is not a completed result, and would also confound more steps with a longer min-LR tail (horizon still 4100). Crash means revert.

## Challenger concerns
- The crash is a process-wall-clock failure, not a training failure. The kill came at 420 s while the training clock read about 235 s, so roughly 185 s went to max-autotune compilation in the warmup step, which train.py excludes from the budget. Earlier compiled runs finished in 330-342 s total, so default compile costs about 30-40 s. This run produced no val_bpb and says nothing about the val half of the hypothesis.
- The throughput sub-claim was actually supported in the log but is at risk of being lost. 521.9k tok/s at 32,768 tokens per step is about 0.0628 s/step, which projects to about 4,780 steps in 300 s. That is roughly 10% above e79's 4,346 and well past the 4,450 threshold. It cost about 150 s of extra compile time, so it is not a net gain under honest total-time accounting.
- The pre-budget warmup is an unbounded loophole. Because compile time is excluded from TIME_BUDGET_S, any change can buy unlimited 'free' compile time until the hard kill. This is against the spirit of rule 4 and makes runs fragile.
- Even a completed run would have been confounded with the schedule. With total_steps_estimate=4100 and about 4,780 steps, roughly 680 steps would run at the min-LR floor, against about 250 in e79. Any val change would mix 'more steps' with 'longer min-LR tail', and e80/e81 showed the tail length matters.

*Ledger: results.tsv, epoch 83.*
