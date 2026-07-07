# Open Questions

Unresolved questions worth a future epoch's hypothesis. Append new ones as
they surface during review; strike through (don't delete) ones that get
answered, with a link to the findings page that answered them.

## Seeded starter questions

1. **Is the baseline compute-bound or data-loading-bound at batch_size=64,
   seq_len=512?** If `get_batch`'s numpy random-crop + host-to-device copy
   is a meaningful fraction of step time, architecture changes to the model
   itself will under-deliver relative to their theoretical FLOPs savings.
   Worth profiling step time split (data vs. forward vs. backward vs.
   optimizer) before trusting tok/s deltas between epochs as attributable
   to the model change alone.

2. **What is the marginal value of context length under a fixed time
   budget?** The baseline uses `seq_len=512` out of a `MAX_SEQ_LEN=1024`
   ceiling. Longer sequences give the model more context per token but cost
   more per step (attention is quadratic in seq_len); shorter sequences buy
   more gradient steps in the same wall-clock budget. Is there a seq_len
   that dominates 512 for bpb-per-second-of-compute specifically at the
   300s budget, or does the optimum shift with the budget size?

3. **Does the 100-step linear warmup + cosine-to-`total_steps_estimate`
   schedule match the *actual* number of steps the fixed time budget buys?**
   `train.py`'s `total_steps_estimate` is a rough guess (5000) used only to
   shape the cosine decay; if a hypothesis changes step time materially
   (e.g. bigger model, different attention), the schedule may decay too
   fast or too slow relative to the real number of steps taken, which would
   confound any conclusion about whether the *architecture* change helped
   versus the *schedule* mismatch happening to help or hurt.
