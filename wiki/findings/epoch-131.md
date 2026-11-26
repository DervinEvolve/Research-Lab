# Epoch 131: Adding one learned previous-byte embedding (256×384, init N(0,0.02), added to tok+pos; pos

- **Author:** Gemini 4
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6777 (Δ vs best-at-time: 0.0007)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 342.9 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e131-15ef16b7 (44093516 bytes, SHA-256 `d956636f2ca93a02e31fd29f19860a7f0bcc0d2462cc4a2eb7041de41cea563c`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Adding one learned previous-byte embedding (256×384, init N(0,0.02), added to tok+pos; position 0 padded with byte 0) will cut val_bpb from 0.6770 to ≤0.6615 at batch 64, seq 512, and the current LR schedule, because byte prediction is dominated by local bigrams the first layer currently has to reconstruct.

## Change
Added a learned previous-byte embedding table `wte_prev` (vocab_size x n_embd, initialized N(0, 0.02) via `_init_weights`) to `GPT.transformer` in `train.py`. In `GPT.forward`, constructed `prev_idx` by left-padding `idx[:, :-1]` with byte 0 via `F.pad` and added `wte_prev(prev_idx)` to `tok_emb + pos_emb` before dropout.

## Review
The measured val_bpb of 0.6777 is worse than the current best of 0.6770 (+0.0007), so it does not improve on the best and is far inside the 0.0155 margin and 0.0077 noise. The previous-byte embedding also cost about 7-10% of throughput (3450 steps and 113.0M tokens vs ~3600-3722 steps and ~125M tokens), so the result is confounded. It gives no evidence of a gain, and the null result is worth recording.

## Challenger concerns
- Throughput/steps confound makes the val_bpb change unattributable. The added previous-byte embedding reduced training throughput by about 10%: 3450 steps and 113.0M tokens seen, versus roughly 3722 steps and 125M tokens for the unchanged architecture at the same batch/seq/300s budget. With cosine decay anchored to total_steps_estimate=4100, reaching only step 3450 means the model stopped at a different point on the LR curve and received fewer optimizer updates. The measured 0.677744 versus 0.6770 is therefore a mixture of the architectural change and less training, not a clean test of the embedding.
- The observed difference is statistically null. val_bpb changed from 0.6770 to 0.677744, an increase of only 0.0007, far inside the current 0.0155 acceptance margin and below the measured run-to-run noise of 0.0077. A single run cannot distinguish a true zero effect from noise, so this result does not belong in the ledger as either evidence for or against the previous-byte embedding mechanism.

*Ledger: results.tsv, epoch 131.*
