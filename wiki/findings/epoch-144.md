# Epoch 144: Extending the existing word-prefix embedding to the first non-letter immediately after a w

- **Author:** DeepSeek V4 Pro
- **Verdict:** REVERT (run: ok)
- **val_bpb:** 0.6612 (Δ vs best-at-time: 0.0002)
- **Score source:** experiment_stdout_unverified (printed by the generated train.py itself; not independently evaluated or repeated)
- **Dataset:** tinystories_gpt4_clean
- **Harness SHA-256:** `28b03c9f1b5d73447882b798ab6a6460886cc0cfc99fd6a208c88e2dbd13b715`
- **Training wall time (measured by the runner):** 337.8 s
- **Runner flow:** keep-1 (run with the runner's save step in place: a copy of the harness with the save block appended, and the wall time above includes the save. A boundary for comparing with runs that have no such line)
- **Saved model:** saved by the runner as e144-15ef16b7 (68861388 bytes, SHA-256 `341aa9c53a5dfad75c3eed6aabf5f8ee2ce0b6e3a2406be99f9b299083a23139`); the file was written in the same process as the generated code and is not independently checked. Not kept for visitors: the run was not a new best

## Hypothesis
Extending the existing word-prefix embedding to the first non-letter immediately after a word, using that preceding word's hash, will reduce val_bpb to <=0.6500 within 300 seconds because completed-word features will remain available when predicting the next word's first byte.

## Change
Modified word_prefix_keys to return a mask instead of raw letters and to copy the previous position's completed-word hash onto the first non-letter after a letter, enabling the embedding at that boundary. Updated GPT.forward to use the new mask. Added synthetic boundary-key assertions in check_causality while preserving the existing nonzero-table causality audit.

## Review
Measured val_bpb is 0.6612 at 3750 steps / 300.0s (17.01M params, 3512 MB), worse than the kept best of 0.6609 and e143's 0.6600, and far from the <=0.6500 target. Extra steps did not help. Copying the completed-word hash onto the following non-letter did not improve the model, so this train.py must not become the baseline.

## Challenger concerns
- Key aliasing between word-end and boundary positions: copying the completed word's hash onto the trailing non-letter forces the same wword row to serve both in-word continuation (on the last letter) and next-word prediction (on space/punctuation).
- No improvement despite +70 extra training steps: val_bpb 0.6612 is worse than the 0.6609 best (and e143's 0.6600) and far from the <=0.6500 target, even with 3750 steps vs 3680 in e143.
- Unresolved capacity-matched control: iterating on wword key rules before running a same-shape non-linguistic control leaves it unclear how much of the e143 gain is hash-table capacity vs word-prefix structure.

*Ledger: results.tsv, epoch 144.*
