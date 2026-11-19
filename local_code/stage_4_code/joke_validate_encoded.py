#!/usr/bin/env python3
"""Validate joke_encoded.pt and print statistics.

Usage:
    python joke_validate_encoded.py

Loads joke_encoded.pt (made by joke_build_vocab.py, same folder) and checks it
with [PASS] / [FAIL] lines. Besides checking the file on its own, it rebuilds
the expected result from jokes_clean.csv (using the functions and settings in
joke_build_vocab.py) and compares, so it also catches split, vocabulary and
encoding mistakes.

Needs joke_build_vocab.py and joke_tokenizer.py in the same folder.
Exits with code 1 if any check fails.
"""
import sys
from collections import Counter

import numpy as np
import torch

import joke_build_vocab as B

n_pass = n_fail = 0


def check(label, ok, detail=""):
    global n_pass, n_fail
    ok = bool(ok)
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    print(f"File: {B.OUT_PATH}")
    if not B.OUT_PATH.exists():
        check("joke_encoded.pt exists (run joke_build_vocab.py first)", False)
        sys.exit(1)
    try:
        data = torch.load(B.OUT_PATH, weights_only=True)
    except Exception as e:  # noqa: BLE001
        check("file loads with torch.load(weights_only=True)", False, str(e))
        sys.exit(1)
    check("file loads with torch.load(weights_only=True)", True)

    # ------------------------------------------------------------------
    print("\n-- Top-level structure and config")
    expected_keys = {"vocab", "word_counts", "train", "val", "test", "config"}
    check("top-level keys are exactly vocab, word_counts, train, val, test, config",
          set(data.keys()) == expected_keys, f"found {sorted(data.keys())}")
    cfg = data["config"]
    expected_cfg = {
        "max_len": B.MAX_LEN, "seq_len": B.SEQ_LEN, "min_freq": B.MIN_FREQ,
        "val_fraction": B.VAL_FRACTION, "test_fraction": B.TEST_FRACTION,
        "seed": B.SEED, "pad_token": B.PAD_TOKEN, "unk_token": B.UNK_TOKEN,
        "bos_token": B.BOS_TOKEN, "eos_token": B.EOS_TOKEN,
        "pad_idx": B.PAD_IDX, "unk_idx": B.UNK_IDX,
        "bos_idx": B.BOS_IDX, "eos_idx": B.EOS_IDX,
    }
    bad = {k: (cfg.get(k), v) for k, v in expected_cfg.items() if cfg.get(k) != v}
    check("config matches the settings in joke_build_vocab.py", not bad, f"mismatches: {bad}" if bad else "")

    # ------------------------------------------------------------------
    print("\n-- Vocabulary")
    itos = data["vocab"]["itos"]
    stoi = data["vocab"]["stoi"]
    word_counts = data["word_counts"]
    check("itos has no duplicate tokens", len(set(itos)) == len(itos))
    check("first four tokens are <PAD>, <UNK>, <BOS>, <EOS> at indices 0-3",
          itos[:4] == B.SPECIAL_TOKENS, f"found {itos[:4]}")
    check("stoi and itos are exact inverses",
          len(stoi) == len(itos) and all(stoi[t] == i for i, t in enumerate(itos)))
    check("word_counts covers exactly the non-special tokens",
          set(word_counts) == set(itos[len(B.SPECIAL_TOKENS):]))
    check(f"every kept token has count >= {B.MIN_FREQ}",
          all(c >= B.MIN_FREQ for c in word_counts.values()),
          f"lowest count {min(word_counts.values())}")
    check("tokens are ordered by frequency (then alphabetically)",
          itos[len(B.SPECIAL_TOKENS):] ==
          sorted(word_counts, key=lambda t: (-word_counts[t], t)))

    # ------------------------------------------------------------------
    print("\n-- Rebuild from jokes_clean.csv and compare")
    jokes, info = B.load_jokes()
    idx_train, idx_val, idx_test = B.split_indices(len(jokes))
    parts = {
        "train": [jokes[i] for i in idx_train],
        "val": [jokes[i] for i in idx_val],
        "test": [jokes[i] for i in idx_test],
    }
    expected_counts = {t: c for t, c in
                       Counter(t for toks in parts["train"] for t in toks).items()
                       if c >= B.MIN_FREQ}
    check("vocabulary equals the one rebuilt from the TRAIN jokes only (no val/test leakage)",
          word_counts == expected_counts)
    sets = {k: {tuple(t) for t in v} for k, v in parts.items()}
    check("no joke appears in more than one split",
          not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
          and not (sets["val"] & sets["test"]))
    total_n = sum(len(data[s]["lengths"]) for s in ("train", "val", "test"))
    check("total jokes in the file equals the jokes kept from the CSV",
          total_n == len(jokes), f"file {total_n}, CSV after cleaning {len(jokes)}")

    # ------------------------------------------------------------------
    vocab_size = len(itos)
    seq_len = B.SEQ_LEN
    for name in ("train", "val", "test"):
        print(f"\n-- Split: {name}")
        d = data[name]
        check(f"{name}: keys are exactly x and lengths (no labels)",
              set(d.keys()) == {"x", "lengths"}, f"found {sorted(d.keys())}")
        x, lengths = d["x"], d["lengths"]
        n = len(parts[name])
        check(f"{name}: x is int32 with shape [{n}, {seq_len}]",
              x.dtype == torch.int32 and tuple(x.shape) == (n, seq_len),
              f"dtype {x.dtype}, shape {tuple(x.shape)}")
        check(f"{name}: lengths is int64 with shape [{n}]",
              lengths.dtype == torch.int64 and tuple(lengths.shape) == (n,),
              f"dtype {lengths.dtype}, shape {tuple(lengths.shape)}")
        xn, ln = x.numpy(), lengths.numpy()
        rows = np.arange(n)

        check(f"{name}: all indices are within the vocabulary [0, {vocab_size})",
              xn.min() >= 0 and xn.max() < vocab_size, f"min {xn.min()}, max {xn.max()}")
        check(f"{name}: lengths are between 3 and {seq_len}",
              ln.min() >= 3 and ln.max() <= seq_len, f"min {ln.min()}, max {ln.max()}")
        check(f"{name}: every joke starts with <BOS>", bool((xn[:, 0] == B.BOS_IDX).all()))
        check(f"{name}: every joke ends with <EOS> at position lengths-1",
              bool((xn[rows, ln - 1] == B.EOS_IDX).all()))
        check(f"{name}: <BOS> and <EOS> each appear exactly once per joke",
              bool(((xn == B.BOS_IDX).sum(1) == 1).all() and ((xn == B.EOS_IDX).sum(1) == 1).all()))
        real = np.arange(seq_len)[None, :] < ln[:, None]
        check(f"{name}: right-padded (real tokens first, <PAD> only after the end)",
              bool(((xn != B.PAD_IDX) == real).all()))

        mismatches = 0
        for i, tokens in enumerate(parts[name]):
            ids = xn[i, 1:ln[i] - 1]
            if len(ids) != len(tokens):
                mismatches += 1
                continue
            for j, t in zip(ids, tokens):
                if not (itos[j] == t or (j == B.UNK_IDX and t not in stoi)):
                    mismatches += 1
                    break
        check(f"{name}: every encoded joke matches its source tokens "
              f"(unknown tokens -> <UNK>, nothing else changed)",
              mismatches == 0, f"{mismatches} mismatching jokes" if mismatches else "")

        n_unk = int((xn == B.UNK_IDX).sum())
        n_real = int((ln - 2).sum())
        print(f"   stats: {n} jokes | length with <BOS>/<EOS>: mean {ln.mean():.1f}, "
              f"median {np.median(ln):.0f}, p95 {np.percentile(ln, 95):.0f}, "
              f"min {ln.min()}, max {ln.max()} | <UNK> {100 * n_unk / n_real:.2f}% of tokens")

    # ------------------------------------------------------------------
    print("\n-- Samples")
    print("   Top 15 tokens by count: "
          + ", ".join(f"{t!r}({word_counts[t]})" for t in itos[4:19]))
    rng = np.random.RandomState(B.SEED)
    for name in ("train", "val"):
        x = data[name]["x"]
        for i in rng.choice(len(x), size=3, replace=False):
            print(f"   {name}: {B.decode(x[i], itos)[:200]}")

    print(f"\nSummary: {n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
