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

### Training at `seq_len=1024` (eval confound + true regression) and `seq_len=256`

- Epoch: 121, 122, 101
- results.tsv row: 121, 122, 101
- Hypothesis: Changing training `seq_len` (1024 with `batch_size=32` in e121/e122 for longer context, or 256 with `batch_size=128` in e101 for lower attention cost) at fixed tokens/step will reduce `val_bpb`.
- Result:
  - e121: `val_bpb` 0.6770 -> 0.6558 (`train_steps=3297`, `300.1s`, `3491 MB`, `10.720M`) — **eval confound** (`evaluate_bpb` called with `seq_len=1024`).
  - e122: `val_bpb` 0.6770 -> 0.6862 (+0.0092, `train_steps=3581`, `300.0s`, `3491 MB`) — true test with eval fixed at `seq_len=512`.
  - e101: `val_bpb` 0.6942 -> 0.7856 (+0.0914, `train_steps=3637`, `300.0s`, `3191 MB`) — `seq_len=256, bs=128`.
- Verdict: Reverted. e121 was an invalid eval-context confound; e122 proved training at 1024 hurts 512-byte evaluation, and e101 proved halving context to 256 severely degrades byte modeling.
- Notes: Never pass a non-512 `seq_len` to `evaluate_bpb` when comparing against the 512-byte baseline (`findings/epoch-121.md`, `findings/epoch-122.md`, `findings/epoch-101.md`).

### Batch size extremes (`batch_size=32`, `batch_size=128` unscaled, `batch_size=256`)

- Epoch: 124, 108, 86 (contrast with e130)
- results.tsv row: 124, 108, 86
- Hypothesis: Shrinking `batch_size` to 32 buys ~70% more optimizer steps (e124), or raising `batch_size` to 128/256 improves A10G utilization and gradient smoothness (e108, e86).
- Result:
  - e124 (`bs=32`): `val_bpb` 0.6770 -> 0.6956 (+0.0186, `train_steps=6453`, `1913 MB`).
  - e108 (`bs=128`, horizon unchanged at 4100): `val_bpb` 0.6770 -> 0.6943 (+0.0173, `train_steps=2069`, `6641 MB`).
  - e86 (`bs=256`, pre-Muon): `val_bpb` 0.7067 -> 0.7889 (+0.0822, `train_steps=1161`, `11755 MB`).
- Verdict: Reverted for clear regressions.
- Notes: Both e124 and e108 also suffered schedule-horizon mismatch (`total_steps_estimate=4100` vs 6453 or 2069 actual steps). When e130 paired `bs=128` with a rescaled horizon (`total_steps_estimate=2050`), it reached `0.6707` (-0.0063, within noise, `2262 steps`), so `bs=128` is only viable if the LR horizon is rescaled with step count (`findings/epoch-124.md`, `findings/epoch-108.md`, `findings/epoch-130.md`).

### Increasing depth to `n_layer=8`

- Epoch: 136 (and earlier e15)
- results.tsv row: 136, 15
- Hypothesis: Increasing `n_layer` from 6 to 8 (10.72M -> 14.26M params) adds needed capacity for byte-level story structure within 300s.
- Result: e136: `val_bpb` 0.6770 -> 0.7017 (+0.0247, `train_steps=2519`, `300.0s`, `4474 MB`, `14.261M`). Earlier e15 (pre-Muon): 0.7067 -> 0.7213 (+0.0146, `3298 steps`).
- Verdict: Reverted — real regression exceeding the noise margin in both eras.
- Notes: At 300s on one A10G, the ~27% drop in optimizer steps outweighs extra depth. Do not scale `n_layer` up without a large throughput win first (`findings/epoch-136.md`).

### Raising Muon / AdamW learning rates (`muon_lr=0.04`, 2x LR)

- Epoch: 134, 104
- results.tsv row: 134, 104
- Hypothesis: Doubling `muon_lr` from `0.02` to `0.04` (e134) or doubling both `max_lr` (`1.2e-3 -> 2.4e-3`) and `muon_lr` (`0.02 -> 0.04`) (e104) after QK-norm will accelerate convergence in 300s.
- Result:
  - e134 (`muon_lr=0.04`): `val_bpb` 0.6770 -> 0.6807 (+0.0037, `train_steps=3764`, `3488 MB`).
  - e104 (`2x` both LRs): `val_bpb` 0.6770 -> 0.6825 (+0.0055, `train_steps=3452`, `3488 MB`).
- Verdict: Reverted — both regressed cleanly against `0.6770`.
- Notes: `muon_lr=0.02` and `max_lr=1.2e-3` are already near-optimal for this architecture and 300s budget (`findings/epoch-134.md`, `findings/epoch-104.md`).

### Causal depthwise Conv1d local mixers

- Epoch: 127, 107, 116
- results.tsv row: 127, 107, 116
- Hypothesis: Adding causal depthwise `Conv1d(kernel_size=4)` to pre-attention states in every block (e127), to QKV outputs (e107), or to input embeddings (e116) will build local byte n-grams cheaply.
- Result:
  - e127 (6x pre-attn conv): `val_bpb` 0.6770 -> 0.6904 (+0.0134, `train_steps=2997`, `3634 MB`, `10.729M`).
  - e107 (6x QKV conv): `val_bpb` 0.6770 -> 0.7113 (+0.0343, `train_steps=2526`, `3923 MB`, `10.748M`).
  - e116 (1x input-emb conv): `val_bpb` 0.6770 -> 0.6798 (+0.0028, `train_steps=3397`, `3512 MB`, `10.722M`).
- Verdict: Reverted — all three regressed, primarily due to heavy kernel/throughput overhead (depthwise convs cut step count by 13-27% under `torch.compile`).
- Notes: For local byte n-grams, embedding-table lookups (`e138` hashed bigram+trigram tables -> `0.6639` at 3636 steps) avoid the Conv1d throughput penalty (`findings/epoch-127.md`, `findings/epoch-107.md`, `findings/epoch-116.md`).

### Cutting Muon Newton-Schulz steps (`ns_steps=5 -> 3`)

- Epoch: 109
- results.tsv row: 109
- Hypothesis: Lowering `ns_steps` from 5 to 3 trades slight orthogonalization accuracy for higher step throughput.
- Result: `val_bpb` 0.6770 -> 0.6901 (+0.0131, `train_steps=3752`, `300.1s`, `3488 MB`).
- Verdict: Reverted — throughput rose only ~8% while loss regressed by +0.0131.
- Notes: Keep `ns_steps=5`; if speeding up Muon, batch same-shape NS matmuls (`e128`/`e137`) instead of cutting iterations (`findings/epoch-109.md`).

### Doubling attention heads (`n_head=6 -> 12`, `head_dim=32`)

- Epoch: 111
- results.tsv row: 111
- Hypothesis: Finer-grained heads (`12 x 32`) better capture short-range byte dependencies at matched parameter count.
- Result: `val_bpb` 0.6770 -> 0.6815 (+0.0045, `train_steps=3581`, `300.1s`, `3503 MB`, `10.720M`).
- Verdict: Reverted (`findings/epoch-111.md`).

### Learnable per-channel residual-stream scalars

- Epoch: 129
- results.tsv row: 129
- Hypothesis: Learnable vectors `a_l`, `m_l` (init 1.0) scaling each block's attention and MLP residual branches let layers tune contributions cheaply.
- Result: `val_bpb` 0.6770 -> 0.6825 (+0.0055, `train_steps=3362`, `300.0s`, `3752 MB`, `10.725M`).
- Verdict: Reverted — added memory/step overhead and regressed (`findings/epoch-129.md`).

### Schedule horizon & floor misfits (`total_steps_estimate=8000/3500`, `min_lr_ratio=0.0`)

- Epoch: 90, 84, 100, 117
- results.tsv row: 90, 84, 100, 117
- Hypothesis: Extending horizon to 8000 (e90), compressing to 3500 (e84), linear WSD decay to 0 (e100), or setting `min_lr_ratio=0.0` at horizon 3600 (e117) improves final bpb.
- Result:
  - e90 (`horizon=8000` under Muon): `val_bpb` 0.6942 -> 0.7215 (+0.0273, `3930 steps`).
  - e84 (`horizon=3500` pre-Muon): `val_bpb` 0.7067 -> 0.7239 (+0.0172, `4170 steps`).
  - e100 (WSD linear decay to 0, pre-QK-norm): `val_bpb` 0.6942 -> 0.6984 (+0.0042, `3942 steps`).
  - e117 (`min_lr_ratio=0.0`, horizon 3600): `val_bpb` 0.6770 -> 0.6796 (+0.0026, `3736 steps`).
- Verdict: Reverted — under-annealing (8000) and overly aggressive decay both hurt; keep a modest floor (`min_lr_ratio=0.1`). Note e126 (WSD with cosine decay to `0.1*max_lr`) scored `0.6709` (-0.0061, within noise).

### `torch.compile(mode='max-autotune')` and CUDA-fused AdamW

- Epoch: 83, 78, 87
- results.tsv row: 83, 78, 87
- Hypothesis: `max-autotune` compile (e83) or `fused=True` AdamW (e78, e87) will raise step throughput in 300s.
- Result:
  - e83: hard-killed at 420s crash (~185s compile overhead ate the runner grace window).
  - e78: crashed in Triton/Inductor backend-hash during warmup.
  - e87 (`fused=True` AdamW): `val_bpb` 0.7067 -> 0.7155 (+0.0089, `4344 steps`, zero throughput gain vs `4346` control).
- Verdict: Reverted / crashed (`findings/epoch-083.md`, `findings/epoch-087.md`).

### Pre-QK-norm RoPE (`e82`, `e91`, `e98`) and Value Residual (`e99`)

- Epoch: 82, 91, 98, 99
- results.tsv row: 82, 91, 98, 99
- Hypothesis: Replacing `wpe` with RoPE or adding layer-0 value residual (`ResFormer`) before QK-norm existed.
- Result: e82 `0.7208` (vs `0.7067`), e91 `0.7029` and e98 `0.7008` (vs `0.6942`), e99 `0.7043` (vs `0.6942`).
- Verdict: Reverted.
- Notes: **Follow-up condition:** After parameter-free QK-norm landed in e102, RoPE reversed sign and won 7/7 single runs (`0.6655-0.6767`, see `wiki/index.md` Watchlist), whereas post-QK-norm value residual (`e119` `0.6764`) remained indistinguishable from noise.
