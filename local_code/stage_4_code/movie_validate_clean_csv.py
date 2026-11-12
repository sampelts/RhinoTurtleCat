"""Validate the cleaned IMDb CSVs and report review-length percentiles.

Run with:  python validate_clean_csv.py   (no arguments)

Reads train_clean.csv and test_clean.csv from the data directory, runs
sanity checks, and prints token-length statistics to help pick a
truncation length for the RNN.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded paths / constants
# ---------------------------------------------------------------------------
DATA_DIR = (
    Path(__file__).resolve().parent
    / ".." / ".." / "data" / "stage_4_data" / "text_classification"
).resolve()

SPLITS = ["train", "test"]
EXPECTED_COLUMNS = ["id", "rating", "label", "text"]
EXPECTED_ROWS = 25000
EXPECTED_PER_CLASS = 12500
TOKEN_PATTERN = re.compile(r"^[a-z]{2,}$")
PERCENTILES = [50, 75, 90, 95, 99]
CANDIDATE_LENGTHS = [100, 128, 150, 200, 256, 300, 400, 500]
RARE_THRESHOLD = 3  # for the vocabulary preview

# URL leftovers: these must never appear (FAIL if any review contains them).
URL_FAIL_TOKENS = ["http", "https", "www"]
# Likely URL fragments but also real words, so informational only.
URL_WATCH_TOKENS = ["com", "html", "htm", "org", "net", "videoplay", "docid"]
N_CONTEXT_EXAMPLES = 3  # how many example snippets to print per leftover token


def check(name, passed, detail=""):
    """Print a PASS/FAIL line and return whether the check passed."""
    status = "PASS" if passed else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return passed


def print_contexts(df, token_lists, token, window=6):
    """Print a few example snippets (with review id) around a leftover token."""
    shown = 0
    for review_id, label, tokens in zip(df["id"], df["label"], token_lists):
        if token in tokens:
            i = tokens.index(token)
            snippet = " ".join(tokens[max(0, i - window): i + window + 1])
            print(f"      e.g. id={review_id} ({label}): ...{snippet}...")
            shown += 1
            if shown >= N_CONTEXT_EXAMPLES:
                break


def validate_split(split):
    """Run all validation checks on one split. Returns (dataframe, n_failures)."""
    path = DATA_DIR / f"{split}_clean.csv"
    print(f"\n{'=' * 60}\n{split.upper()}  ({path.name})\n{'=' * 60}")

    if not path.exists():
        print(f"  [FAIL] file not found: {path}")
        return None, 1

    # Read everything as strings, and don't let pandas turn text like "nan" or
    # "null" into missing values.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    failures = 0

    # --- structure ---------------------------------------------------------
    failures += not check("columns are id, rating, label, text",
                          list(df.columns) == EXPECTED_COLUMNS,
                          f"found {list(df.columns)}")
    if list(df.columns) != EXPECTED_COLUMNS:
        return df, failures  # cannot continue safely

    failures += not check(f"row count is {EXPECTED_ROWS}",
                          len(df) == EXPECTED_ROWS, f"found {len(df)}")

    # --- labels and ratings -----------------------------------------------
    counts = df["label"].value_counts().to_dict()
    failures += not check("labels are only pos / neg",
                          set(counts) <= {"pos", "neg"}, f"found {counts}")
    failures += not check(f"{EXPECTED_PER_CLASS} reviews per class",
                          counts.get("pos") == EXPECTED_PER_CLASS
                          and counts.get("neg") == EXPECTED_PER_CLASS,
                          f"found {counts}")

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    failures += not check("ratings are integers",
                          ratings.notna().all(),
                          f"{int(ratings.isna().sum())} unparseable")
    failures += not check("ratings are in 1-10",
                          bool(ratings.between(1, 10).all()))
    mismatch = (
        ((df["label"] == "neg") & (ratings > 4))
        | ((df["label"] == "pos") & (ratings < 7))
    )
    failures += not check("labels match ratings (neg <= 4, pos >= 7)",
                          not mismatch.any(), f"{int(mismatch.sum())} mismatches")

    # --- ids ----------------------------------------------------------------
    dup_ids = int(df.duplicated(subset=["id", "label"]).sum())
    failures += not check("no duplicate (id, label) pairs", dup_ids == 0,
                          f"{dup_ids} duplicates")

    # --- text content -------------------------------------------------------
    empty = int((df["text"].str.strip() == "").sum())
    failures += not check("no empty texts", empty == 0, f"{empty} empty")

    messy_ws = int((df["text"] != df["text"].str.split().str.join(" ")).sum())
    failures += not check("single spaces only, no leading/trailing whitespace",
                          messy_ws == 0, f"{messy_ws} rows affected")

    token_lists = df["text"].str.split()
    bad_rows = 0
    bad_examples = Counter()
    for tokens in token_lists:
        bad = [t for t in tokens if not TOKEN_PATTERN.match(t)]
        if bad:
            bad_rows += 1
            bad_examples.update(bad)
    failures += not check("every token matches ^[a-z]{2,}$", bad_rows == 0,
                          f"{bad_rows} rows with bad tokens; "
                          f"examples: {list(bad_examples)[:10]}")

    all_tokens = Counter(t for tokens in token_lists for t in tokens)

    # --- URL leftovers ------------------------------------------------------
    token_sets = token_lists.apply(set)
    for tok in URL_FAIL_TOKENS:
        n_rows = int(token_sets.apply(lambda ts, t=tok: t in ts).sum())
        failures += not check(f"no '{tok}' tokens", n_rows == 0,
                              f"{n_rows} reviews, {all_tokens.get(tok, 0)} occurrences")
        if n_rows:
            print_contexts(df, token_lists, tok)

    print("  URL-fragment watch list (informational; some are real words):")
    for tok in URL_WATCH_TOKENS:
        n_rows = int(token_sets.apply(lambda ts, t=tok: t in ts).sum())
        print(f"    '{tok}': {n_rows} reviews, {all_tokens.get(tok, 0)} occurrences")
        if n_rows:
            print_contexts(df, token_lists, tok)

    # Informational (not pass/fail): things worth eyeballing.
    print(f"  [INFO] leftover 'br' tokens (HTML line breaks): {all_tokens.get('br', 0)}")
    dup_texts = int(df.duplicated(subset=["text"]).sum())
    print(f"  [INFO] duplicate review texts within split: {dup_texts}")

    # --- length statistics -------------------------------------------------
    lengths = token_lists.str.len().to_numpy()
    print("\n  Token length per review")
    print(f"    mean={lengths.mean():.1f}  min={lengths.min()}  max={lengths.max()}")
    for p in PERCENTILES:
        print(f"    p{p:<2} = {int(np.ceil(np.percentile(lengths, p)))}")

    print("\n  Percentiles by label")
    for label in ["neg", "pos"]:
        sub = lengths[(df["label"] == label).to_numpy()]
        vals = ", ".join(
            f"p{p}={int(np.ceil(np.percentile(sub, p)))}" for p in PERCENTILES
        )
        print(f"    {label}: mean={sub.mean():.1f}, {vals}")

    print("\n  Candidate truncation lengths")
    print("    max_len | reviews fully kept | tokens kept overall")
    total_tokens = lengths.sum()
    for L in CANDIDATE_LENGTHS:
        kept_reviews = (lengths <= L).mean() * 100
        kept_tokens = np.minimum(lengths, L).sum() / total_tokens * 100
        print(f"    {L:>7} | {kept_reviews:>17.1f}% | {kept_tokens:>18.1f}%")

    # --- vocabulary preview (helps pick min-frequency / size cap) ---------
    n_rare = sum(1 for c in all_tokens.values() if c < RARE_THRESHOLD)
    print(f"\n  Vocabulary preview (this split only)")
    print(f"    distinct tokens: {len(all_tokens)}")
    print(f"    tokens appearing < {RARE_THRESHOLD} times: {n_rare}")
    print(f"    top 15: {all_tokens.most_common(15)}")

    return df, int(failures)


def main():
    print(f"Data directory: {DATA_DIR}")
    total_failures = 0
    for split in SPLITS:
        _, failures = validate_split(split)
        total_failures += failures

    print(f"\n{'=' * 60}")
    if total_failures == 0:
        print("All checks passed.")
    else:
        print(f"{total_failures} check(s) FAILED. See [FAIL] lines above.")


if __name__ == "__main__":
    main()
