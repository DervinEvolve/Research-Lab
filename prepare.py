"""
prepare.py -- FIXED HARNESS. Researcher agents must NEVER modify this file.

Byte-level data pipeline for TinyStories-GPT4-clean
(https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean):
2.73M short stories, ~2.19B chars of pure ASCII (74 unique bytes).
Canonical pre-shuffled splits by row index (per the dataset card):
    test  : rows      0 -  9,999
    val   : rows 10,000 - 19,999
    train : rows 20,000+  (capped at TRAIN_BYTE_CAP bytes)
Stories are joined with "\n\n". Vocab is raw bytes (256 tokens).

This module also owns:
  - TIME_BUDGET / MAX_SEQ_LEN constants agents can read but not redefine here
  - get_batch(): random-crop batch sampler over the memmapped .bin files
  - evaluate_bpb(): THE metric. Deterministic, seeded, model-agnostic.
    Any train.py variant must expose model(idx, targets) -> (logits, loss)
    where loss is mean cross-entropy in nats/token over bytes; bpb follows
    from the change of base (loss / ln(2)) since tokens are bytes.

Run directly to materialize the .bin files:
    python prepare.py
    python prepare.py --smoke   # synth ~2MB of random-ish bytes, no download
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

DATASET_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet"
# Versioned data dir: switching datasets must never mix with stale bins.
CACHE_DIR = Path(os.environ.get("RESEARCH_CACHE", "~/.cache/research-lab")).expanduser() / "tinystories_v1"

VOCAB_SIZE = 256
TEST_ROWS = 10_000       # rows 0..9,999
VAL_ROWS = 10_000        # rows 10,000..19,999
TRAIN_BYTE_CAP = 500_000_000  # cap joined train text (plenty for 300s runs)

TIME_BUDGET = float(os.environ.get("TIME_BUDGET_S", 300))
MAX_SEQ_LEN = 1024

EVAL_SEED = 1337  # fixed seed for deterministic val crops -- do not change


def _bin_paths(cache_dir: Path):
    return {
        "train": cache_dir / "train.bin",
        "val": cache_dir / "val.bin",
        "test": cache_dir / "test.bin",
    }


def _download_parquet(dest: Path):
    import requests

    print(f"[prepare] downloading {DATASET_URL} -> {dest}", file=sys.stderr)
    tmp = dest.with_suffix(".part")
    with requests.get(DATASET_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    tmp.rename(dest)


def _synth_smoke_bytes(n_bytes: int, seed: int = 0) -> bytes:
    """Deterministic pseudo-text bytes for CI smoke tests (no network)."""
    rng = np.random.default_rng(seed)
    # Bias toward printable ASCII so it vaguely resembles text; still random.
    data = rng.integers(low=32, high=127, size=n_bytes, dtype=np.uint8)
    return data.tobytes()


def prepare(cache_dir: Path = CACHE_DIR, smoke: bool = False, force: bool = False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _bin_paths(cache_dir)

    if not force and all(p.exists() for p in paths.values()):
        print("[prepare] bin files already present, skipping", file=sys.stderr)
        return paths

    if smoke:
        # ~2MB synthetic corpus, split with the same proportions as prod.
        total = 2_000_000
        train_n = int(total * 0.90)
        val_n = int(total * 0.05)
        test_n = total - train_n - val_n
        raw = _synth_smoke_bytes(total, seed=0)
    else:
        import pyarrow.parquet as pq

        pq_path = cache_dir / "tinystories_gpt4_clean.parquet"
        if not pq_path.exists():
            _download_parquet(pq_path)
        print("[prepare] reading parquet + building byte splits", file=sys.stderr)
        pf = pq.ParquetFile(pq_path)
        sep = b"\n\n"
        test_parts, val_parts, train_parts = [], [], []
        train_bytes = 0
        row = 0
        done = False
        for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
            for t in batch.column("text").to_pylist():
                b = t.encode("utf-8", errors="ignore") + sep
                if row < TEST_ROWS:
                    test_parts.append(b)
                elif row < TEST_ROWS + VAL_ROWS:
                    val_parts.append(b)
                elif train_bytes < TRAIN_BYTE_CAP:
                    train_parts.append(b)
                    train_bytes += len(b)
                else:
                    done = True
                    break
                row += 1
            if done:
                break
        splits_raw = {
            "train": b"".join(train_parts),
            "val": b"".join(val_parts),
            "test": b"".join(test_parts),
        }
        splits = {k: np.frombuffer(v, dtype=np.uint8) for k, v in splits_raw.items()}

    if smoke:
        arr = np.frombuffer(raw, dtype=np.uint8)
        splits = {
            "train": arr[:train_n],
            "val": arr[train_n : train_n + val_n],
            "test": arr[train_n + val_n : train_n + val_n + test_n],
        }

    for name, data in splits.items():
        p = paths[name]
        mm = np.memmap(p, dtype=np.uint8, mode="w+", shape=data.shape)
        mm[:] = data[:]
        mm.flush()
        print(f"[prepare] wrote {p} ({data.nbytes} bytes)", file=sys.stderr)

    return paths


def _load_memmap(split: str, cache_dir: Path = CACHE_DIR) -> np.memmap:
    p = _bin_paths(cache_dir)[split]
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run prepare() first")
    return np.memmap(p, dtype=np.uint8, mode="r")


def get_batch(split: str, batch_size: int, seq_len: int, device, cache_dir: Path = CACHE_DIR):
    """Random-crop batch sampler. Returns (x, y) int64 tensors of shape
    (batch_size, seq_len), where y is x shifted by one byte (next-byte
    prediction)."""
    data = _load_memmap(split, cache_dir)
    n = len(data)
    ix = np.random.randint(0, n - seq_len - 1, size=batch_size)
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def evaluate_bpb(
    model,
    batch_size: int = 32,
    max_batches: int = 200,
    seq_len: int = MAX_SEQ_LEN,
    device=None,
    cache_dir: Path = CACHE_DIR,
) -> float:
    """THE metric. Deterministic, seeded crop selection over val.bin so that
    every train.py variant is scored on the exact same evaluation set.

    Requires model(idx, targets) -> (logits, loss) with loss = mean
    cross-entropy in nats/token. Since tokens are raw bytes, bits-per-byte
    is simply loss / ln(2).

    Restores the model's original train/eval mode on exit.
    """
    if device is None:
        device = next(model.parameters()).device

    data = _load_memmap("val", cache_dir)
    n = len(data)
    if n <= seq_len + 1:
        raise ValueError("val split too small for requested seq_len")

    was_training = model.training
    model.eval()

    gen = torch.Generator()
    gen.manual_seed(EVAL_SEED)

    total_loss = 0.0
    total_batches = 0
    try:
        for _ in range(max_batches):
            ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=gen).numpy()
            x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
            y = np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)

            _, loss = model(x, targets=y)
            total_loss += float(loss.item())
            total_batches += 1
    finally:
        model.train(was_training)

    mean_nats = total_loss / max(total_batches, 1)
    bpb = mean_nats / math.log(2)
    return bpb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="synth tiny corpus, no download")
    ap.add_argument("--force", action="store_true", help="rebuild bins even if present")
    ap.add_argument("--cache-dir", type=str, default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else CACHE_DIR
    paths = prepare(cache_dir=cache_dir, smoke=args.smoke, force=args.force)
    for name, p in paths.items():
        size = p.stat().st_size if p.exists() else 0
        print(f"{name}: {p} ({size} bytes)")


if __name__ == "__main__":
    main()
