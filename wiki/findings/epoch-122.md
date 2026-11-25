# Epoch 122: Training at seq_len=1024 with batch_size=32 (same 32,768 tokens/step) while leaving evalua

- **Author:** Grok 4.7
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6862 (Δ vs best-at-time: 0.0092)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 334.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e122-15ef16b7 (44486095 bytes, SHA-256 `a5da7ebfa5103ee12d7f04dd380ec41c12a83edfacbcb968418e0c9d6c61fbee`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Training at seq_len=1024 with batch_size=32 (same 32,768 tokens/step) while leaving evaluation context fixed at seq_len=512 will lower val_bpb from 0.6770 to below 0.6652, because a byte-level GPT trained on T=512 wastes capacity on window-edge predictions with truncated story context; T=1024 halves the cold-start fraction and yields better sequence features without changing the comparable eval distribution.

## Change
Non-smoke training now uses seq_len=1024 and batch_size=32 (still 32,768 tokens/step). GPTConfig.seq_len follows that training length so wpe covers 1024 positions and the forward assert accepts the longer windows. evaluate_bpb is called with a separate eval_seq_len=512; smoke still evaluates at its own seq_len=64. Schedule, architecture, optimizer, and eval batch coverage are unchanged.

## Review
With evaluation fixed at 512 bytes, the run scored 0.68621 val_bpb versus the 0.6770 best—a regression of 0.00921—after 3,581 steps in 300.005 seconds. It therefore fails the keep criterion despite acceptable memory use of 3,491 MB. Longer training windows did not improve the comparable metric in this configuration; changed positional-embedding exposure limits attribution, and epoch 121's 0.6558 at 1024-byte evaluation remains non-comparable.

## Challenger concerns
- Learned absolute wpe is confounded by halving batch_size (64->32) and doubling seq_len (512->1024): wpe[0..511] receives half as many gradient updates per step, while 50% of training tokens train wpe[512..1023] which are never used at eval_seq_len=512.
- num_params() subtracts wpe.weight.numel(), so doubling wpe from 512 to 1024 (+196,608 params) is hidden in the reported params_m=10.720128.

*Ledger: results.tsv, epoch 122.*
