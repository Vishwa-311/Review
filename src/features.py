#!/usr/bin/env python3
from __future__ import annotations
import re
import pandas as pd
from typing import Iterable
from textblob import TextBlob
COMMON_FAKE_PHRASES = ["best product ever","highly recommend","sponsored","discount for review","free sample","life changing","five stars"]
def extract_numeric_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in texts:
        s = str(s or "")
        blob = TextBlob(s)
        rows.append({
            "sentiment": float(blob.sentiment.polarity),
            "exclamation_count": s.count("!"),
            "all_caps_tokens": sum(1 for w in s.split() if len(w)>3 and w.isupper()),
            "repeated_phrases": sum(1 for p in COMMON_FAKE_PHRASES if p in s.lower()),
            "char_length": len(s),
            "unique_word_ratio": (len(set(s.split())) / max(1, len(s.split()))),
        })
    return pd.DataFrame(rows)


def highlight_text_flags(text: str) -> str:
    """Highlight suspicious triggers (cliches, ALL-CAPS, exclamation marks) in HTML."""
    if not text:
        return ""
    
    html = text
    # Highlight cliché phrases
    for phrase in COMMON_FAKE_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        html = pattern.sub(f'<mark style="background-color: #ff4b4b; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">\\g<0></mark>', html)
    
    # Highlight ALL-CAPS words (>3 letters)
    words = html.split()
    new_words = []
    for w in words:
        clean_w = re.sub(r"[^a-zA-Z]", "", w)
        if len(clean_w) > 3 and clean_w.isupper() and not w.startswith("<mark"):
            new_words.append(f'<mark style="background-color: #ff9800; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{w}</mark>')
        else:
            new_words.append(w)
    
    return " ".join(new_words)

