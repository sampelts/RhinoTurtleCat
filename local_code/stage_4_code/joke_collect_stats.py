#!/usr/bin/env python3
"""Collect statistics on the cleaned jokes dataset (read-only, writes no files).

Usage:
    python joke_collect_stats.py

Reads jokes_clean.csv (columns: id, text) from
../../data/stage_4_data/text_generation (relative to this script).
Needs joke_tokenizer.py in the same folder.

The statistics are meant to help us choose:
    - the maximum joke length (jokes longer than it will be DROPPED)
    - the minimum word frequency for the vocabulary
and to show whether the cleaned text still has problems.

Before anything is measured, jokes are deduplicated on their token sequence
(this is what the vocabulary script will do before splitting).

Note: the build script will make the vocabulary from the 80% train portion
only. Section 6 previews this with the same kind of 80/10/10 split (seed 42),
so its numbers are a close preview, not the final ones.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from joke_tokenizer import tokenize, is_word

# ---------------------------------------------------------------------------
# Hardcoded settings
# ---------------------------------------------------------------------------
DATA_DIR = (Path(__file__).resolve().parent
            / "../../data/stage_4_data/text_generation").resolve()
CSV_PATH = DATA_DIR / "jokes_clean.csv"

SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
N_SPECIAL = 4  # <PAD>, <UNK>, <BOS>, <EOS>

MIN_FREQ_CANDIDATES = [1, 2, 3, 4, 5, 10]
LENGTH_CANDIDATES = [20, 25, 30, 40, 50, 60, 75, 100, 150, 200]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe(values, label):
    a = np.asarray(values)
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    print(f"  {label}: min {a.min()} | mean {a.mean():.1f} | median {p50:.0f} | "
          f"p90 {p90:.0f} | p95 {p95:.0f} | p99 {p99:.0f} | max {a.max()}")


def show_tokens(label, tokens, counter, limit=15):
    """Print how many tokens are in the list and the most frequent few."""
    tokens = sorted(tokens, key=lambda t: (-counter[t], t))
    shown = ", ".join(f"{t!r}({counter[t]})" for t in tokens[:limit])
    print(f"  {label}: {len(tokens)} distinct" + (f" | e.g. {shown}" if tokens else ""))


def split_indices(n, seed):
    """Shuffled 80/10/10 split indices: (train, val, test)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * VAL_FRACTION))
    n_test = int(round(n * TEST_FRACTION))
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def main():
    print(f"Reading {CSV_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    missing = {"id", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")

    # ------------------------------------------------------------------
    section("1. Rows, empty jokes and duplicates")
    texts = [t.strip() for t in df["text"].tolist()]
    nonempty = [t for t in texts if t]
    n_exact_extra = len(nonempty) - len(set(nonempty))

    seen, jokes, toks = set(), [], []
    for t in nonempty:
        tk = tokenize(t)
        key = tuple(tk)
        if not tk or key in seen:
            continue
        seen.add(key)
        jokes.append(t)
        toks.append(tk)
    n_dedupe_removed = len(nonempty) - len(jokes)

    print(f"  rows in CSV                          : {len(df)}")
    print(f"  duplicate ids                        : {int(df['id'].duplicated().sum())}")
    print(f"  empty / whitespace-only texts        : {len(texts) - len(nonempty)}")
    print(f"  exact duplicate texts (extra copies) : {n_exact_extra}")
    print(f"  removed by token-sequence dedupe     : {n_dedupe_removed} "
          f"(includes the exact duplicates)")
    print(f"  unique jokes used below              : {len(jokes)}")

    # ------------------------------------------------------------------
    section("2. Characters")
    chars = Counter(ch for t in jokes for ch in t)
    non_ascii = {c: n for c, n in chars.items() if ord(c) > 127}
    symbols = {c: n for c, n in chars.items() if not c.isalnum() and not c.isspace()}
    print(f"  distinct characters   : {len(chars)}")
    print(f"  uppercase letters     : {sum(n for c, n in chars.items() if c.isupper())}")
    print(f"  digits                : {sum(n for c, n in chars.items() if c.isdigit())}")
    print(f"  non-ASCII characters  : {len(non_ascii)} distinct"
          + (" | " + ", ".join(f"{c!r}({n})" for c, n in
                               sorted(non_ascii.items(), key=lambda kv: -kv[1])[:20])
             if non_ascii else ""))
    print("  symbols (not letter/digit/space), most common:")
    print("    " + ", ".join(f"{c!r}({n})" for c, n in
                             sorted(symbols.items(), key=lambda kv: -kv[1])[:30]))
    describe([len(t) for t in jokes], "characters per joke")

    # ------------------------------------------------------------------
    section("3. Length in tokens (words + punctuation, without <BOS>/<EOS>)")
    lengths = np.array([len(tk) for tk in toks])
    describe(lengths, "tokens per joke")
    print("  The stored sequence will be 2 longer (<BOS> and <EOS>).")
    print("\n  If we DROP jokes longer than a limit (limit counts tokens, no <BOS>/<EOS>):")
    print(f"  {'limit':>6} | {'dropped':>8} | {'dropped %':>9} | {'kept':>6}")
    for limit in LENGTH_CANDIDATES:
        dropped = int((lengths > limit).sum())
        print(f"  {limit:>6} | {dropped:>8} | {100 * dropped / len(lengths):>8.1f}% | "
              f"{len(lengths) - dropped:>6}")
    order = np.argsort(lengths, kind="stable")
    print("\n  Shortest 3 jokes:")
    for i in order[:3]:
        print(f"    [{lengths[i]:>3} tokens] {jokes[i][:200]}")
    print("  Longest 3 jokes:")
    for i in order[::-1][:3]:
        print(f"    [{lengths[i]:>3} tokens] {jokes[i][:200]}")

    # ------------------------------------------------------------------
    section("4. Joke structure (question / punchline)")
    q_counts = Counter(min(sum(1 for t in tk if t == "?"), 3) for tk in toks)
    print("  question marks per joke: "
          + " | ".join(f"{'3+' if k == 3 else k}: {q_counts.get(k, 0)}" for k in range(4)))
    last = Counter(tk[-1] if not is_word(tk[-1]) else "(a word)" for tk in toks)
    print("  last token of a joke   : "
          + ", ".join(f"{k!r}({v})" for k, v in last.most_common(8)))

    # ------------------------------------------------------------------
    section("5. Vocabulary (all unique jokes)")
    counter = Counter(t for tk in toks for t in tk)
    total = sum(counter.values())
    words = [t for t in counter if is_word(t)]
    puncts = [t for t in counter if not is_word(t)]
    print(f"  total tokens          : {total}")
    print(f"  distinct tokens       : {len(counter)} ({len(words)} words, {len(puncts)} punctuation/symbols)")
    print("\n  How many distinct tokens appear exactly N times:")
    for c in (1, 2, 3, 4, 5):
        n = sum(1 for v in counter.values() if v == c)
        print(f"    {c} time(s): {n} ({100 * n / len(counter):.1f}% of distinct tokens)")
    for c in (10, 50, 100):
        n = sum(1 for v in counter.values() if v >= c)
        print(f"    >= {c} times: {n}")
    print("\n  Top 30 tokens:")
    print("    " + ", ".join(f"{t!r}({n})" for t, n in counter.most_common(30)))
    print("  Punctuation / symbol tokens:")
    print("    " + ", ".join(f"{t!r}({counter[t]})" for t in
                             sorted(puncts, key=lambda t: -counter[t])[:20]))

    print("\n  Possible leftovers to inspect:")
    show_tokens("tokens containing digits",
                [t for t in counter if any(ch.isdigit() for ch in t)], counter)
    show_tokens("tokens with an apostrophe",
                [t for t in counter if "'" in t], counter)
    show_tokens("one-letter words other than a / i",
                [t for t in words if len(t) == 1 and t not in ("a", "i")], counter)
    show_tokens("tokens with non-ASCII characters",
                [t for t in counter if any(ord(ch) > 127 for ch in t)], counter)
    show_tokens("tokens with an underscore",
                [t for t in counter if "_" in t], counter)

    # ------------------------------------------------------------------
    section("6. Preview: minimum frequency vs vocabulary size and <UNK> rate")
    idx_train, idx_val, idx_test = split_indices(len(toks), SEED)
    print(f"  Preview split (seed {SEED}): train {len(idx_train)} / "
          f"val {len(idx_val)} / test {len(idx_test)} jokes")
    print(f"  Vocabulary is built from the train portion only; size includes the "
          f"{N_SPECIAL} special tokens.")
    train_counter = Counter(t for i in idx_train for t in toks[i])

    def unk_rate(indices, vocab):
        n = unk = 0
        for i in indices:
            for t in toks[i]:
                n += 1
                if t not in vocab:
                    unk += 1
        return 100 * unk / n if n else 0.0

    print(f"\n  {'min_freq':>8} | {'vocab size':>10} | {'<UNK> train':>11} | "
          f"{'<UNK> val':>9} | {'<UNK> test':>10}")
    for mf in MIN_FREQ_CANDIDATES:
        vocab = {t for t, c in train_counter.items() if c >= mf}
        print(f"  {mf:>8} | {len(vocab) + N_SPECIAL:>10} | "
              f"{unk_rate(idx_train, vocab):>10.2f}% | "
              f"{unk_rate(idx_val, vocab):>8.2f}% | "
              f"{unk_rate(idx_test, vocab):>9.2f}%")

    # ------------------------------------------------------------------
    section("7. Tokenizer check: 5 random jokes")
    rng = np.random.RandomState(SEED)
    for i in rng.choice(len(jokes), size=min(5, len(jokes)), replace=False):
        print(f"  text  : {jokes[i][:250]}")
        print(f"  tokens: {' | '.join(toks[i])[:300]}\n")

if __name__ == "__main__":
    main()
