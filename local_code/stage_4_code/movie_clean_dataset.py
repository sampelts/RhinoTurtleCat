#!/usr/bin/env python3
"""Clean the IMDb movie review dataset and write one CSV per split.

Usage:
    python clean_dataset.py

Reads train/ and test/ from ../../data/stage_4_data/text_classification
(relative to this script) and writes the cleaned CSVs to the same directory:
    train_clean.csv, test_clean.csv
    columns: id, rating, label, text   (text is last and every string is quoted)
"""
import csv
import random
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Standard NLTK stop words are removed, except these negations / intensifiers,
# which carry sentiment signal.
KEEP_WORDS = {
    "no", "nor", "not", "very", "too", "but", "against",
    "only", "more", "most", "few",
}

# Order matters: irregular forms first, then the generic "n't" rule.
CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "is not"),
    (r"n't\b", " not"),
]

HTML_RE = re.compile(r"<[^>]+>")
# URLs in reviews are often broken by stray spaces, e.g.
# "http: //video.google.com/ videoplay?docid=-30&q =innerviews+ jamie&hl=en ."
# After the "http://" or "www." start, keep consuming following chunks as long
# as they contain a URL-ish character, and stop at the first chunk that doesn't.
URL_CHARS = r"[/=&?+%#]"
URL_RE = re.compile(
    r"(?:\bhttps?\s*:\s*//|\bwww\s*\.)\S*"
    r"(?:\s+(?=\S*" + URL_CHARS + r")\S+)*",
    re.IGNORECASE,
)
HYPHEN_RE = re.compile(r"(?<=[a-z])-(?=[a-z])")
TOKEN_RE = re.compile(r"^[a-z]{2,}$")  # alphabetic only, length >= 2

SPLITS = ("train", "test")
LABELS = ("pos", "neg")

# Fixed project layout, resolved relative to this script's location.
DATA_DIR = (Path(__file__).resolve().parent
            / "../../data/stage_4_data/text_classification").resolve()
OUT_DIR = DATA_DIR


def download_nltk_data():
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)


# Safety net: leftover URL fragments that the URL regex missed.
JUNK_TOKENS = {"http", "https", "www"}


def build_stop_words():
    return (set(stopwords.words("english")) - KEEP_WORDS) | JUNK_TOKENS


def clean_text(raw, stop_words):
    """Return the cleaned review as a list of tokens."""
    text = HTML_RE.sub(" ", raw)
    text = URL_RE.sub(" ", text)
    text = text.lower().replace("\u2019", "'")  # curly apostrophe
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text)
    text = HYPHEN_RE.sub(" ", text)  # "well-made" -> "well made"
    tokens = word_tokenize(text)
    return [t for t in tokens if TOKEN_RE.match(t) and t not in stop_words]


def load_split(data_dir, split, stop_words):
    rows = []
    for label in LABELS:
        folder = data_dir / split / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for path in sorted(folder.glob("*.txt")):
            review_id, rating = path.stem.split("_")
            raw = path.read_text(encoding="utf-8")
            tokens = clean_text(raw, stop_words)
            rows.append({
                "id": int(review_id),
                "rating": int(rating),
                "label": label,
                "text": " ".join(tokens),
                "_raw": raw,
            })
    return pd.DataFrame(rows)


def report(df, split, n_samples=2, seed=0):
    n_tokens = df["text"].str.split().str.len()
    print(f"\n=== {split} ===")
    print(df["label"].value_counts().to_string())
    print(f"avg tokens: {n_tokens.mean():.1f} | min: {n_tokens.min()} | "
          f"max: {n_tokens.max()} | empty reviews: {(n_tokens == 0).sum()}")
    rng = random.Random(seed)
    for i in rng.sample(range(len(df)), n_samples):
        print(f"\n--- sample (id={df.iloc[i]['id']}, label={df.iloc[i]['label']}) ---")
        print("RAW  :", df.iloc[i]["_raw"][:300].replace("\n", " "))
        print("CLEAN:", df.iloc[i]["text"][:300])


def main():
    print(f"Data directory: {DATA_DIR}")
    download_nltk_data()
    stop_words = build_stop_words()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"Cleaning {split} ...")
        df = load_split(DATA_DIR, split, stop_words)
        report(df, split)
        out_path = OUT_DIR / f"{split}_clean.csv"
        df[["id", "rating", "label", "text"]].to_csv(
            out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"\nSaved {len(df)} reviews to {out_path}")


if __name__ == "__main__":
    main()
