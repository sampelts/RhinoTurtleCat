#!/usr/bin/env python3
"""Tokenizer shared by the joke scripts (statistics, vocabulary building, ...).

Keeping it in one module guarantees that the statistics script and the
vocabulary script split text in exactly the same way.

Rules:
    - a word is a run of letters/digits/underscore, optionally with apostrophe
      parts inside it:   don't, they're, rock'n'roll
    - every other non-space character is its own token:   ?  !  .  ,  "  -  ...
    - curly apostrophes are turned into straight ones first
    - no lowercasing here (the cleaned data is already lowercase; the
      statistics script reports whether any uppercase is left)

Example:
    "they're the wurst! " -> ["they're", "the", "wurst", "!"]
"""
import re

_TOKEN_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]")
_WORD_START_RE = re.compile(r"\w")


def tokenize(text):
    """Split a joke into a list of word and punctuation tokens."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return _TOKEN_RE.findall(text)


def is_word(token):
    """True for word tokens, False for punctuation / symbol tokens."""
    return _WORD_START_RE.match(token) is not None
