#!/usr/bin/env python3
"""Load movie_encoded.pt, validate it, and print its statistics.

Usage:
    python movie_validate_encoded.py

Reads movie_encoded.pt from ../../data/stage_4_data/text_classification
(relative to this script).
"""
from pathlib import Path

import numpy as np
import torch

DATA_DIR = (Path(__file__).resolve().parent
            / "../../data/stage_4_data/text_classification").resolve()
ENC_PATH = DATA_DIR / "movie_encoded.pt"

SPLITS = ("train", "val", "test")
EXPECTED_SIZES = {"train": 22500, "val": 2500, "test": 25000}
EXPECTED_KEYS = {"vocab", "word_counts", "train", "val", "test", "config"}
PERCENTILES = (50, 90, 95, 99)
N_SAMPLES = 2
SAMPLE_WORDS = 60


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return bool(passed)


def decode(row, itos, pad_idx):
    return " ".join(itos[j] for j in row.tolist() if j != pad_idx)


def validate_vocab(data):
    print(f"\n{'=' * 60}\nVOCABULARY\n{'=' * 60}")
    failures = 0
    cfg = data["config"]
    stoi, itos = data["vocab"]["stoi"], data["vocab"]["itos"]
    counts = data["word_counts"]
    pad_idx, unk_idx = cfg["pad_idx"], cfg["unk_idx"]

    print(f"  size (incl. special tokens): {len(itos)}")
    print(f"  special tokens: {cfg['pad_token']}={pad_idx}, {cfg['unk_token']}={unk_idx}")

    failures += not check(f"vocabulary size is exactly {cfg['max_vocab_size']} (incl. special tokens)",
                          len(itos) == cfg["max_vocab_size"], f"found {len(itos)}")
    failures += not check("stoi and itos have the same length", len(stoi) == len(itos),
                          f"{len(stoi)} vs {len(itos)}")
    failures += not check("itos[0] is <PAD> and itos[1] is <UNK>",
                          itos[pad_idx] == cfg["pad_token"] and itos[unk_idx] == cfg["unk_token"])
    failures += not check("stoi / itos are consistent for every word",
                          all(stoi.get(w) == i for i, w in enumerate(itos)))
    real_words = itos[2:]
    failures += not check("word_counts covers exactly the non-special words",
                          set(counts) == set(real_words),
                          f"{len(counts)} counted vs {len(real_words)} words")
    min_count = min(counts.values()) if counts else 0
    failures += not check(f"every kept word has count >= {cfg['min_freq']}",
                          min_count >= cfg["min_freq"], f"min count = {min_count}")
    in_order = all(counts[a] >= counts[b] for a, b in zip(real_words, real_words[1:]))
    failures += not check("words are ordered by descending frequency", in_order)

    print(f"\n  Most frequent words : {[(w, counts[w]) for w in real_words[:12]]}")
    print(f"  Least frequent words: {[(w, counts[w]) for w in real_words[-8:]]}")
    vals = np.array(list(counts.values()))
    print(f"  Count distribution  : median={int(np.median(vals))}, "
          f"count==min_freq: {int((vals == cfg['min_freq']).sum())} words, "
          f"count>=100: {int((vals >= 100).sum())} words")
    return failures


def validate_split(name, data):
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
    failures = 0
    cfg = data["config"]
    max_len, pad_idx, unk_idx = cfg["max_len"], cfg["pad_idx"], cfg["unk_idx"]
    vocab_size = len(data["vocab"]["itos"])
    itos = data["vocab"]["itos"]
    d = data[name]
    x, lengths, y = d["x"], d["lengths"], d["y"]
    n = len(y)

    failures += not check("x is int32, lengths and y are int64",
                          x.dtype == torch.int32 and lengths.dtype == torch.int64
                          and y.dtype == torch.int64,
                          f"x {x.dtype}, lengths {lengths.dtype}, y {y.dtype}")
    print(f"  x       : shape {tuple(x.shape)}, dtype {x.dtype}")
    print(f"  lengths : shape {tuple(lengths.shape)}, dtype {lengths.dtype}")
    print(f"  y       : shape {tuple(y.shape)}, dtype {y.dtype}")

    failures += not check(f"expected size {EXPECTED_SIZES[name]}",
                          n == EXPECTED_SIZES[name], f"found {n}")
    failures += not check(f"x shape is [N, {max_len}]",
                          tuple(x.shape) == (n, max_len))
    failures += not check("x, lengths and y have the same N",
                          len(x) == len(lengths) == len(y))
    failures += not check("labels are only 0 / 1",
                          set(y.unique().tolist()) <= {0, 1})
    n_pos = int(y.sum())
    failures += not check("classes are balanced", n_pos == n - n_pos,
                          f"pos {n_pos} / neg {n - n_pos}")
    failures += not check("indices are within [0, vocab_size)",
                          int(x.min()) >= 0 and int(x.max()) < vocab_size,
                          f"min {int(x.min())}, max {int(x.max())}, vocab {vocab_size}")
    failures += not check(f"lengths are within [1, {max_len}]",
                          int(lengths.min()) >= 1 and int(lengths.max()) <= max_len,
                          f"min {int(lengths.min())}, max {int(lengths.max())}")

    # padding: right-padded, and non-pad count equals the stored length
    non_pad = (x != pad_idx)
    failures += not check("non-<PAD> count per row equals stored length",
                          bool((non_pad.sum(dim=1) == lengths).all()))
    pos = torch.arange(max_len).unsqueeze(0)
    expected_mask = pos < lengths.unsqueeze(1)
    failures += not check("padding is on the right (real tokens fill the start of each row)",
                          bool((non_pad == expected_mask).all()))

    # statistics
    L = lengths.numpy()
    print(f"\n  Length: mean={L.mean():.1f}, min={L.min()}, max={L.max()}, "
          + ", ".join(f"p{p}={int(np.ceil(np.percentile(L, p)))}" for p in PERCENTILES))
    n_full = int((lengths == max_len).sum())
    print(f"  Reviews at the max length ({max_len}, i.e. truncated or exactly {max_len}): "
          f"{n_full} ({100 * n_full / n:.1f}%)")

    real = x[non_pad]
    unk_rate = 100 * float((real == unk_idx).sum()) / real.numel()
    print(f"  <UNK> share of real tokens: {unk_rate:.2f}%")
    print(f"  Reviews with >= 20% <UNK>: "
          f"{int((((x == unk_idx).sum(dim=1).float() / lengths.float()) >= 0.2).sum())}")

    print(f"\n  Decoded samples (first {SAMPLE_WORDS} words):")
    for i in range(min(N_SAMPLES, n)):
        words = decode(x[i], itos, pad_idx).split()[:SAMPLE_WORDS]
        print(f"    [{i}] label={'pos' if int(y[i]) else 'neg'}, length={int(lengths[i])}: "
              f"{' '.join(words)}")
    return failures


def main():
    print(f"File: {ENC_PATH}")
    if not ENC_PATH.exists():
        print("  [FAIL] file not found. Run movie_build_vocab.py first.")
        return
    size_mb = ENC_PATH.stat().st_size / 1e6
    data = torch.load(ENC_PATH, weights_only=True)
    print(f"Loaded OK ({size_mb:.1f} MB)")

    failures = 0
    print(f"\n{'=' * 60}\nSTRUCTURE\n{'=' * 60}")
    keys = set(data.keys())
    failures += not check("top-level keys are as expected", keys == EXPECTED_KEYS,
                          f"found {sorted(keys)}")
    if keys != EXPECTED_KEYS:
        print("\nCannot continue: unexpected structure.")
        return

    print("  config:")
    for k, v in data["config"].items():
        print(f"    {k}: {v}")

    failures += validate_vocab(data)
    for name in SPLITS:
        failures += validate_split(name, data)

    print(f"\n{'=' * 60}")
    if failures == 0:
        print("All checks passed.")
    else:
        print(f"{failures} check(s) FAILED. See [FAIL] lines above.")


if __name__ == "__main__":
    main()
