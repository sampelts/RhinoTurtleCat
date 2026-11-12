#!/usr/bin/env python3
"""Build the vocabulary and encode the cleaned jokes for the RNN.

Usage:
    python joke_build_vocab.py

Reads jokes_clean.csv (columns: id, text) from
../../data/stage_4_data/text_generation (relative to this script) and writes
a single file to the same directory:
    joke_encoded.pt
Needs joke_tokenizer.py in the same folder.

Steps:
    1. Tokenize every joke (words and punctuation as separate tokens), remove
       duplicate jokes (same token sequence), and DROP jokes longer than
       MAX_LEN tokens (no truncation, so every stored joke is complete).
    2. Shuffle and split 80/10/10 into train / val / test (fixed seed).
    3. Build the vocabulary from the train portion only
       (min frequency MIN_FREQ; <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3).
    4. Encode every joke as   <BOS> t1 t2 ... tn <EOS>   and right-pad with
       <PAD> to SEQ_LEN = MAX_LEN + 2.
    5. Save everything in one dict (see SAVED FILE LAYOUT below).

To try a different minimum frequency, change MIN_FREQ below and re-run.

SAVED FILE LAYOUT (torch.load-able, also with weights_only=True):
    {
      "vocab": {"stoi": {token: index}, "itos": [token, ...]},
      "word_counts": {token: count in the train portion, kept tokens only},
      "train" / "val" / "test": {
          "x":       IntTensor  [N, SEQ_LEN]  <BOS> ... <EOS> then <PAD> (int32)
          "lengths": LongTensor [N]           true length INCLUDING <BOS> and <EOS> (int64)
      },
      "config": {max_len, seq_len, min_freq, val_fraction, test_fraction, seed,
                 pad_token, unk_token, bos_token, eos_token,
                 pad_idx, unk_idx, bos_idx, eos_idx}
    }
    There is no "y": for next-token prediction the model input is x[:, :-1]
    and the target is x[:, 1:] (the loader's collate function does the shift).
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from joke_tokenizer import tokenize

# ---------------------------------------------------------------------------
# Hardcoded settings
# ---------------------------------------------------------------------------
DATA_DIR = (Path(__file__).resolve().parent
            / "../../data/stage_4_data/text_generation").resolve()
CSV_PATH = DATA_DIR / "jokes_clean.csv"
OUT_PATH = DATA_DIR / "joke_encoded.pt"

MAX_LEN = 50            # max tokens per joke, NOT counting <BOS> and <EOS>
SEQ_LEN = MAX_LEN + 2   # stored row length: <BOS> + up to MAX_LEN tokens + <EOS>
MIN_FREQ = 1            # a token must appear at least this often in train to be kept
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
SEED = 42

PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def load_jokes():
    """Tokenize, deduplicate and length-filter. Returns (token_lists, info)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")

    info = {"rows": len(df), "empty": 0, "duplicates": 0, "too_long": 0}
    seen, kept = set(), []
    for text in df["text"].tolist():
        tokens = tokenize(text.strip())
        if not tokens:
            info["empty"] += 1
            continue
        key = tuple(tokens)
        if key in seen:
            info["duplicates"] += 1
            continue
        seen.add(key)
        if len(tokens) > MAX_LEN:
            info["too_long"] += 1  # dropped, not truncated
            continue
        kept.append(tokens)
    info["kept"] = len(kept)
    return kept, info


def split_indices(n):
    """Shuffled split indices: (train, val, test)."""
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = int(round(n * VAL_FRACTION))
    n_test = int(round(n * TEST_FRACTION))
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def build_vocab(train_jokes):
    """Vocabulary from the training portion only."""
    counts = Counter(t for tokens in train_jokes for t in tokens)
    kept = [(t, c) for t, c in counts.items() if c >= MIN_FREQ]
    kept.sort(key=lambda tc: (-tc[1], tc[0]))  # by frequency, then alphabetical
    itos = SPECIAL_TOKENS + [t for t, _ in kept]
    stoi = {t: i for i, t in enumerate(itos)}
    word_counts = {t: c for t, c in kept}
    info = {
        "n_distinct": len(counts),
        "n_kept": len(kept),
        "n_cut": len(counts) - len(kept),
        "min_count_kept": kept[-1][1] if kept else 0,
    }
    return stoi, itos, word_counts, info


def encode_split(jokes, stoi):
    """Encode jokes into padded tensors. Also returns <UNK> statistics."""
    n = len(jokes)
    x = np.full((n, SEQ_LEN), PAD_IDX, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int64)
    n_tokens = n_unk = 0

    for i, tokens in enumerate(jokes):
        ids = [stoi.get(t, UNK_IDX) for t in tokens]
        n_tokens += len(ids)
        n_unk += sum(1 for j in ids if j == UNK_IDX)
        row = [BOS_IDX] + ids + [EOS_IDX]
        x[i, :len(row)] = row  # right padding: real tokens first
        lengths[i] = len(row)

    return {
        "x": torch.from_numpy(x),
        "lengths": torch.from_numpy(lengths),
    }, {"n_tokens": n_tokens, "n_unk": n_unk}


def decode(row, itos):
    return " ".join(itos[j] for j in row.tolist() if j != PAD_IDX)


def main():
    print(f"Data file: {CSV_PATH}")

    # 1. load, tokenize, deduplicate, drop over-length
    jokes, linfo = load_jokes()
    print(f"\nJokes")
    print(f"  rows in CSV                  : {linfo['rows']}")
    print(f"  empty                        : {linfo['empty']}")
    print(f"  duplicates removed           : {linfo['duplicates']}")
    print(f"  dropped (> {MAX_LEN} tokens)        : {linfo['too_long']}")
    print(f"  jokes kept                   : {linfo['kept']}")

    # 2. split
    idx_train, idx_val, idx_test = split_indices(len(jokes))
    parts = {
        "train": [jokes[i] for i in idx_train],
        "val": [jokes[i] for i in idx_val],
        "test": [jokes[i] for i in idx_test],
    }
    print(f"\nSplit {len(jokes)} jokes (seed={SEED}): train {len(parts['train'])} / "
          f"val {len(parts['val'])} / test {len(parts['test'])}")

    # 3. vocabulary from the train portion only
    stoi, itos, word_counts, vinfo = build_vocab(parts["train"])
    print(f"\nVocabulary (built from {len(parts['train'])} train jokes only)")
    print(f"  distinct tokens in train        : {vinfo['n_distinct']}")
    print(f"  with count >= {MIN_FREQ} (kept)          : {vinfo['n_kept']}")
    print(f"  cut by min frequency            : {vinfo['n_cut']}")
    print(f"  lowest count among kept tokens  : {vinfo['min_count_kept']}")
    print(f"  final size incl. {len(SPECIAL_TOKENS)} special tokens : {len(itos)}")

    # 4. encode
    splits, stats = {}, {}
    for name in ("train", "val", "test"):
        splits[name], stats[name] = encode_split(parts[name], stoi)

    # 5. save one file
    payload = {
        "vocab": {"stoi": stoi, "itos": itos},
        "word_counts": word_counts,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "config": {
            "max_len": MAX_LEN,
            "seq_len": SEQ_LEN,
            "min_freq": MIN_FREQ,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "seed": SEED,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_idx": PAD_IDX,
            "unk_idx": UNK_IDX,
            "bos_idx": BOS_IDX,
            "eos_idx": EOS_IDX,
        },
    }
    torch.save(payload, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6

    # 6. summary
    print("\nSplit summary")
    for name in ("train", "val", "test"):
        d, s = splits[name], stats[name]
        n = len(d["lengths"])
        print(f"  {name:<5}: {n} jokes | mean length {d['lengths'].float().mean():.1f} "
              f"(with <BOS>/<EOS>) | <UNK> tokens: "
              f"{100 * s['n_unk'] / s['n_tokens']:.2f}%")

    print("\nSample (first train joke, decoded from indices)")
    print(f"  {decode(splits['train']['x'][0], itos)[:300]}")

    print(f"\nSaved {OUT_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
