#!/usr/bin/env python3
"""Build the vocabulary and encode the cleaned IMDb reviews for the RNN.

Usage:
    python movie_build_vocab.py

Reads train_clean.csv and test_clean.csv from
../../data/stage_4_data/text_classification (relative to this script) and
writes a single file to the same directory:
    movie_encoded.pt

Steps:
    1. Split train 90/10 (stratified by label, fixed seed) into train / val.
    2. Build the vocabulary from the 90% train portion only
       (min frequency 3, total size capped at 25,000 including the two
       special tokens, <PAD>=0, <UNK>=1).
    3. Encode every review: keep the FIRST 300 tokens, map words to indices,
       right-pad with <PAD> to length 300 (real tokens at the start of each row).
    4. Save everything in one dict (see SAVED FILE LAYOUT below).

SAVED FILE LAYOUT (torch.load-able, also with weights_only=True):
    {
      "vocab": {"stoi": {word: index}, "itos": [word, ...]},
      "word_counts": {word: count in the 90% train portion, kept words only},
      "train" / "val" / "test": {
          "x":       IntTensor  [N, 300]  right-padded token indices (int32)
          "lengths": LongTensor [N]       true length, capped at 300 (int64)
          "y":       LongTensor [N]       0 = neg, 1 = pos (int64)
      },
      "config": {max_len, min_freq, max_vocab_size, val_fraction, seed, pad_token, unk_token,
                 pad_idx, unk_idx, label_map}
    }
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Hardcoded settings
# ---------------------------------------------------------------------------
DATA_DIR = (Path(__file__).resolve().parent
            / "../../data/stage_4_data/text_classification").resolve()
OUT_PATH = DATA_DIR / "movie_encoded.pt"

MAX_LEN = 300
MIN_FREQ = 3
MAX_VOCAB_SIZE = 25000  # total, including <PAD> and <UNK>
VAL_FRACTION = 0.10
SEED = 42

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1
LABEL_MAP = {"neg": 0, "pos": 1}


def load_csv(split):
    path = DATA_DIR / f"{split}_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["tokens"] = df["text"].str.split()
    df["y"] = df["label"].map(LABEL_MAP)
    if df["y"].isna().any():
        raise ValueError(f"{split}: found labels other than pos/neg")
    return df


def stratified_split(df, val_fraction, seed):
    """Return (train_df, val_df); val has val_fraction of each label."""
    rng = np.random.RandomState(seed)
    val_idx = []
    for label in LABEL_MAP:
        idx = np.flatnonzero((df["label"] == label).to_numpy())
        idx = rng.permutation(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
    is_val = np.zeros(len(df), dtype=bool)
    is_val[val_idx] = True
    return (df[~is_val].reset_index(drop=True),
            df[is_val].reset_index(drop=True))


def build_vocab(train_df):
    """Vocabulary from the training portion only."""
    counts = Counter(t for tokens in train_df["tokens"] for t in tokens)
    frequent = [(w, c) for w, c in counts.items() if c >= MIN_FREQ]
    frequent.sort(key=lambda wc: (-wc[1], wc[0]))  # by frequency, then alphabetical
    n_words = MAX_VOCAB_SIZE - 2  # room left after <PAD> and <UNK>
    kept = frequent[:n_words]
    cut = frequent[n_words:]
    itos = [PAD_TOKEN, UNK_TOKEN] + [w for w, _ in kept]
    stoi = {w: i for i, w in enumerate(itos)}
    word_counts = {w: c for w, c in kept}
    info = {
        "n_distinct": len(counts),
        "n_frequent": len(frequent),
        "n_cut_by_cap": len(cut),
        "min_count_kept": kept[-1][1] if kept else 0,
    }
    return stoi, itos, word_counts, info


def encode_split(df, stoi):
    """Encode a dataframe into padded tensors. Also returns UNK / truncation stats."""
    n = len(df)
    x = np.full((n, MAX_LEN), PAD_IDX, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int64)
    n_tokens = n_unk = n_truncated = 0

    for i, tokens in enumerate(df["tokens"]):
        if len(tokens) > MAX_LEN:
            n_truncated += 1
        tokens = tokens[:MAX_LEN]  # keep the FIRST MAX_LEN tokens
        ids = [stoi.get(t, UNK_IDX) for t in tokens]
        n_tokens += len(ids)
        n_unk += sum(1 for j in ids if j == UNK_IDX)
        lengths[i] = len(ids)
        if ids:
            x[i, :len(ids)] = ids  # right padding: real tokens first

    return {
        "x": torch.from_numpy(x),
        "lengths": torch.from_numpy(lengths),
        "y": torch.tensor(df["y"].to_numpy(dtype=np.int64)),
    }, {"n_tokens": n_tokens, "n_unk": n_unk, "n_truncated": n_truncated}


def decode(row, itos):
    return " ".join(itos[j] for j in row.tolist() if j != PAD_IDX)


def main():
    print(f"Data directory: {DATA_DIR}")

    # 1. load + split
    full_train = load_csv("train")
    test_df = load_csv("test")
    train_df, val_df = stratified_split(full_train, VAL_FRACTION, SEED)
    print(f"\nSplit train {len(full_train)} -> train {len(train_df)} / val {len(val_df)} "
          f"(seed={SEED}); test {len(test_df)}")

    # 2. vocabulary from the 90% train portion only
    stoi, itos, word_counts, vinfo = build_vocab(train_df)
    print(f"\nVocabulary (built from {len(train_df)} train reviews only)")
    print(f"  distinct words in train portion : {vinfo['n_distinct']}")
    print(f"  with count >= {MIN_FREQ}                : {vinfo['n_frequent']}")
    print(f"  cut by the {MAX_VOCAB_SIZE} size cap        : {vinfo['n_cut_by_cap']}")
    print(f"  lowest count among kept words   : {vinfo['min_count_kept']}")
    print(f"  final size incl. <PAD>, <UNK>   : {len(itos)}")

    # 3. encode
    splits = {}
    stats = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        splits[name], stats[name] = encode_split(df, stoi)

    # 4. save one file
    payload = {
        "vocab": {"stoi": stoi, "itos": itos},
        "word_counts": word_counts,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "config": {
            "max_len": MAX_LEN,
            "min_freq": MIN_FREQ,
            "max_vocab_size": MAX_VOCAB_SIZE,
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "pad_idx": PAD_IDX,
            "unk_idx": UNK_IDX,
            "label_map": LABEL_MAP,
        },
    }
    torch.save(payload, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6

    # 5. summary
    print("\nSplit summary")
    for name in ("train", "val", "test"):
        d, s = splits[name], stats[name]
        n = len(d["y"])
        n_pos = int(d["y"].sum())
        print(f"  {name:<5}: {n} reviews (pos {n_pos} / neg {n - n_pos}) | "
              f"<UNK> tokens: {100 * s['n_unk'] / s['n_tokens']:.2f}% | "
              f"truncated to {MAX_LEN}: {s['n_truncated']} "
              f"({100 * s['n_truncated'] / n:.1f}%)")

    print("\nSample (first val review, decoded from indices)")
    print(f"  label: {'pos' if int(splits['val']['y'][0]) else 'neg'}")
    print(f"  text : {decode(splits['val']['x'][0], itos)[:400]}")

    print(f"\nSaved {OUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
