# Epoch 121: Training at seq_len=1024 with batch_size=32 (same 32,768 tokens per step, everything else 

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6558 (Δ vs best-at-time: -0.0212)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 362.4 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e121-15ef16b7 (44486095 bytes, SHA-256 `c008777075276771e38dddde5710e9dac875d7b7b80b711a3b26c6665a5206af`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Training at seq_len=1024 with batch_size=32 (same 32,768 tokens per step, everything else unchanged) will lower val_bpb from 0.6770 to below 0.6652 within the 300s budget. The mechanism: at T=512 a large share of predicted bytes sit near the start of a window cut mid-story, with too little context to resolve names, plot state and sentence position. Doubling the window halves that cold-start fraction and lets attention use story context beyond 512 bytes.

## Change
In the non-smoke branch of train(), changed seq_len from 512 to 1024 and batch_size from 64 to 32, keeping tokens per step constant at 32,768. No other code was modified; wpe automatically resizes through cfg.seq_len, and eval uses the same 1024-byte window as specified by the hypothesis.

## Review
Although val_bpb dropped to 0.6558 (3297 steps, 300.1s, 3491 MB VRAM) vs 0.6770 best, the Challenger's critical confound holds: evaluate_bpb was called with seq_len=1024 instead of the 512-byte window used for all prior runs. Including positions 512-1023 in the eval average mechanically lowers bpb by diluting cold-start positions, making 0.6558 incomparable to the 0.6770 scoreboard despite ~10% fewer tokens seen (108.0M).

## Challenger concerns
- The 0.0212 drop is not on the same scale as the 0.6770 scoreboard. evaluate_bpb was called with seq_len=1024, so this score averages 200*32*1024 = 6.55M val bytes; every prior ledger row, the 0.0059 noise, and the 0.0118 margin are seq_len=512 (3.28M bytes). The hypothesis's own mechanism — halving cold-start positions and conditioning on >512 bytes — applies to the average being scored, not only to training. Unchanged loss on positions 0-511 still lowers bpb once positions 512-1023, which no T=512 run ever scores, enter the mean. A confirmation repeat reruns this file and will not separate the confound; 0.021 already clears 0.0118, so a stable repeat would lock an incomparable number in as the new best and make further context stretching a metric hack.
- Even the training effect is not isolated. Tokens per step stayed 32768, but steps fell to 3297 (108.0M tokens) versus ~3500-3700 steps (~115-121M tokens) on recent T=512 runs, and update noise changed from 64 sequences to 32. At step 3297 the 4100-step cosine is less decayed than a ~3600-step run, so final LR is slightly higher. A 0.021 gain despite ~10% less data is what an easier eval looks like, not a clean context-length win. Token accounting itself matches (3297*32*1024) and the budget was respected — this is confounding, not a logging bug.
- params_m stays 10.720128 only because num_params() subtracts wpe. The positional table doubled: +512*384 = 196,608 real parameters. VRAM stayed ~3491 MB, so this is not the driver, but the run is not capacity-matched and the reported count hides it.

*Ledger: results.tsv, epoch 121.*
