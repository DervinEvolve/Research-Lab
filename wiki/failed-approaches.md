# Failed Approaches

Log of hypotheses that were tried and reverted, or that produced a clear
negative result. The point of this page is purely defensive: an agent
proposing a new epoch's hypothesis should check here first so the org does
not re-spend a budget re-discovering the same dead end.

Add one entry per reverted/negative epoch, using the template below.
Do not delete entries even if a later epoch finds a narrower condition
under which the idea *does* work -- add a follow-up note instead and link
both directions.

## Template

```
### <short name of the approach>

- Epoch: <n>
- results.tsv row: <epoch value, for lookup>
- Hypothesis: <what the Planner claimed and why>
- Result: val_bpb <before> -> <after> (train_steps=<n>, train_seconds=<s>,
  peak_vram_mb=<mb>, params_m=<m>)
- Verdict: reverted / rejected because <reason -- worse bpb, confound found
  by Challenger, budget-gaming, VRAM blowup, etc.>
- Notes: <anything a future agent should know before re-attempting a
  variant of this idea>
```

## Entries

_(none yet -- this program has not run an epoch)_
