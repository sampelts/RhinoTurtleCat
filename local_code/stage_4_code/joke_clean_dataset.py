import html
import re
from pathlib import Path

import pandas as pd
from nltk.tokenize import RegexpTokenizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "stage_4_data" / "text_generation"

IN_FILE = DATA_DIR / "data"
OUT_FILE = DATA_DIR / "jokes_clean.csv"
MIN_WORDS = 3            # need at least 3 word tokens to form a prompt

# Entries that are not jokes (judged from a 160-character preview).
DROP_IDS = {
    391,   # link to an AskReddit thread
    522,   # "found this video" + link
    607,   # subreddit milestone announcement
    773,   # punchline is only a video link
    795,   # reddit game post
    828,   # photo link
    936,   # link to an AskReddit thread
    1293,  # video link
    88,    # SMS / meme text ("omg = oh my girl"), not a joke
    110,   # emergency help request ("i need underwear jokes"), not a joke
    166,   # subreddit meta ("just found this sub"), not a joke
    667,   # subreddit meta ("expecting jokes about soap"), not a joke
    1164,  # subreddit meta ("true story so my friend saw me browsing this subreddit")
    1345,  # joke request ("tell me your best joke that includes..."), not a joke
    237,
    599,
    106,
    1453,
    1465,
    1310,  # strikethrough list gag, relies on formatting
}

# ---------------------------------------------------------------------------
# Per-ID attribution fixes for rows that have noise at a known position.
# Each entry: ID -> ("prefix", marker)  strip everything BEFORE marker
#              ID -> ("suffix", marker)  strip marker AND everything AFTER it
# The marker is a plain substring (case-sensitive, as it appears in the raw CSV).
# ---------------------------------------------------------------------------
ID_FIXES = {
    # Each marker is a plain substring of the RAW joke text (as it appears in the
    # input CSV before tokenization).  For "suffix" cuts the marker begins at the
    # first character of the noise — i.e. AFTER the joke's own terminal punct —
    # so that punctuation is preserved in the output.
    # For "prefix" the marker is the first character of the actual joke content.
    16:   ("prefix", "What did the bee"),             # "bee jokes , courtesy of my niece ( age 8 ) . what did the bee..."
    19:   ("suffix", " (via friend who got"),                # "sir cumference . ( via friend who got this from a street..."
    53:   ("suffix", " EDIT: I"),                      # "no eye - deer . edit : i totally messed this joke up ..."
    61:   ("suffix", " - From Big Nate"),             # "ein stein . - from big nate , as told by my kid ."
    67:   ("prefix", "Why do sorority"),              # "clean joke about sorority girls why do sorority girls ..."
    76:   ("suffix", "Thanks"),
    96:   ("suffix", " : D Source:"),
    101:  ("suffix", " Thought of you"),              # "meat patty ! thought of you guys !"
    123:  ("suffix", " From my 9 year old"),          # "japan ! from my 9 year old ."
    124:  ("suffix", " I'll show myself"),        # "gluten tag . i'll show myself out ."
    126:  ("suffix", "From: http"),
    132:  ("suffix", " (p.s. I have"),                  # "a monkey . ( p . s . i have a wonderful , terrible love ..."
    146:  ("suffix", "- Tim Vine"),                  # "...i didn't see that coming - tim vine"
    152:  ("suffix", " reddit is fun"),
    157:  ("suffix", "Oh my-oh-my!"),
    170:  ("suffix", "(Also: can anyone"),
    180:  ("prefix", '"My sink was'),
    191:  ("prefix", "What did one"),
    193:  ("suffix", " ~Skip"),
    232:  ("suffix", "From /r/PeterL"),
    247:  ("prefix", "Mommy! I found a $10 bill"),
    317:  ("suffix", "&gt; (This exchange"),
    318:  ("suffix", " (Credit goes"),
    342:  ("suffix", " ~Skip"),
    362:  ("suffix", "Just"),
    388:  ("suffix", " Aonther one"),         # "bambooboo . aonther one from my 9 year old ."
    423:  ("suffix", " *From my 9"),          # "u - turns ! from my 9 year old son yesterday . fixed typo ."
    453:  ("suffix", " Sorry that this"),                  # "my crap . sorry that this isn't a clean joke . heh"
    473:  ("suffix", " ~Skip"),
    491:  ("suffix", " [Credit to Milton"),
    492:  ("suffix", " My favorite jok"),
    506:  ("prefix", "Why did the"),
    523:  ("suffix", " (as told by one"),
    531:  ("prefix","\"Hey! The dog you"),
    568:  ("suffix", "VCN"),
    575:  ("suffix", " (As told by my 8yo"),          # "pea soup . ( as told by..."
    586:  ("suffix", " (Seriously"),
    587:  ("suffix", " I'm on a roll"),
    590:  ("suffix", " I woke up with"),
    593:  ("prefix", "Two iPhones walk"),
    642:  ("prefix", "How much does"),
    647:  ("suffix", "(As in board)"),
    654:  ("suffix", "welcome to"),
    744:  ("suffix", " c:"),
    753:  ("suffix", "*Come on guys,"),
    778:  ("suffix", " Another one from"),         # "launch ! another one from my 9 year old ."
    785:  ("suffix", "**Tim Vine**"),                    # "...i'm not stopping you . \" tim vine"
    805:  ("suffix", "!!!!! this is my original"),
    810:  ("suffix", " (not an original"),
    819:  ("suffix", " (I will show my self out)"),
    845:  ("suffix", "Edit:"),                      # "arch enemies . edit : spelling"
    853:  ("suffix", ":D (Puns for the"),
    866:  ("suffix", " [Thanks, Wagon"),
    867:  ("suffix", " (Not sure of the"),
    870:  ("suffix", " [X Post from r/"),
    871:  ("suffix", " ^imagine ^this"),
    892:  ("prefix", "why is it impossible"),
    893:  ("suffix", " #foreveralone"),
    896:  ("suffix", " ~Skip"),
    991:  ("suffix", " Edit: a wor"),                      # "cash or czech edit : a word"
    929:  ("suffix", " I woke up with"),
    982:  ("suffix", " *knee slap*"),
    1075: ("prefix", "Why do galaxies"),
    1084: ("suffix", "&lt;p&gt; My"),
    1183: ("suffix", " ~Skip"),
    1228: ("prefix", "My son took"),
    1235: ("suffix", ". Joke by Tim"),            # "velcro what a rip off . joke by tim vine ."
    1255: ("suffix", " ~Skip"),
    1347: ("prefix", "What holds up a"),
    1353: ("suffix", " ~Skip"),
    1362: ("suffix"," ~ Stewart Francis"),
    1369: ("suffix", " Another one from"),         # "starburst ! another one from my 9 year old ."
    1391: ("suffix", " note: This one"),
    1398: ("suffix", " [Works best IRL"), 
    1403: ("suffix"," (my dad wanted"),
    1452: ("suffix"," --My amazing"),
    1541: ("prefix","How do you find"),
    1569: ("suffix", " P.S. kids"),                  # "bless you . . p . s . kids love it"
    1584: ("suffix", " My little sister"),
    1585: ("prefix", "Where do"),
    1587: ("suffix", " (I made thi"),
    1618: ("suffix"," (I was told this"),
}


def strip_attribution(joke_id: int, raw: str) -> str:
    """Apply a known per-ID positional fix, or return the text unchanged."""
    if joke_id not in ID_FIXES:
        return raw
    direction, marker = ID_FIXES[joke_id]
    idx = raw.find(marker)
    if idx == -1:          # marker not found (e.g. already cleaned) — leave as-is
        print(raw)
        return raw
    if direction == "suffix":
        return raw[:idx].strip()
    else:  # "prefix" — keep from marker onward
        return raw[idx:].strip()


# ellipsis | word (contractions kept whole) | any other single non-space char
TOKEN_PATTERN = r"\.{3}|[a-z0-9]+(?:'[a-z]+)*|[^\sa-z0-9]"
tokenizer = RegexpTokenizer(TOKEN_PATTERN)


def clean(joke_id: int, text: str) -> list:
    text = strip_attribution(joke_id, text)                             # per-ID attribution fix
    text = html.unescape(text)                                          # &amp; &gt; ...
    text = re.sub(r"\[\s*oc\s*\]", " ", text, flags=re.I)               # [OC] tag
    text = re.sub(r"\[([^\]]*)\]\(https?://[^)\s]*\)", r"\1", text)     # [text](url) -> text
    text = re.sub(r"\b(?:source|from|via|credit)\s*:\s*https?://\S+", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+", " ", text)                           # bare URLs
    text = re.sub(r"\(x-?post[^)]*\)", " ", text, flags=re.I)           # (x-post from /r/...)
    text = re.sub(r"\[x-?post[^\]]*\]", " ", text, flags=re.I)          # [x-post from /r/...]
    text = text.replace("*", "").replace('"', "").replace("~", "")
    # collapse repeated punctuation marks (e.g. '!!', '! !', '??', '... ...', ', ,')
    text = re.sub(r"(\.\s*){3,}", "... ", text)
    text = re.sub(r"(\.{3}\s*)+", "... ", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    text = re.sub(r"(!\s*){2,}", "! ", text)
    text = re.sub(r"(\?\s*){2,}", "? ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"(-\s*){2,}", "- ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r":-?\)", " ", text)
    return tokenizer.tokenize(text)


df = pd.read_csv(IN_FILE, encoding="utf-8").dropna(subset=["Joke"])
print("raw jokes:", len(df))
print("with URL:", df["Joke"].str.contains(r"https?://").sum(),
      "| with '*':", df["Joke"].str.contains(r"\*", regex=True).sum(),
      "| with HTML entity:", df["Joke"].str.contains(r"&(?:#?\w+);").sum())

n = len(df)
df = df[~df["ID"].isin(DROP_IDS)]
print("dropped by DROP_IDS:", n - len(df))

df["tokens"] = df.apply(lambda row: clean(row["ID"], row["Joke"]), axis=1)
n_words = df["tokens"].map(lambda ts: sum(t[0].isalnum() for t in ts))
print("dropped (fewer than MIN_WORDS words):", (n_words < MIN_WORDS).sum())
df = df[n_words >= MIN_WORDS].copy()

df["text"] = df["tokens"].map(" ".join)
n_before = len(df)
df = df.drop_duplicates(subset="text")
print("dropped duplicates:", n_before - len(df))

out = df[["ID", "text"]].rename(columns={"ID": "id"})
out.to_csv(OUT_FILE, index=False)
print("saved", len(out), "jokes to", OUT_FILE)