"""Normalise text and fix proper nouns that Whisper mangles.

The vocabulary file lists canonical spellings, one per line, `#` for comments.
The corrections file holds `wrong => right` pairs, applied after transcription.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

FILLERS = ("euh", "heu", "hum", "bah", "ben", "voilà", "tu vois", "en fait", "du coup")
FILLER_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(f) for f in FILLERS) + r")(?![\w-])[ ,]*",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"[ \t]{2,}")


def read_vocabulary(path: str | Path) -> list[str]:
    terms = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms


def read_corrections(path: str | Path) -> dict[str, str]:
    rules: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=>" not in line:
            continue
        wrong, right = (part.strip() for part in line.split("=>", 1))
        if wrong and right:
            rules[wrong] = right
    return rules


def initial_prompt(terms: list[str], limit: int = 200) -> str:
    """Seed Whisper with the proper nouns, so it spells them instead of guessing."""
    if not terms:
        return ""
    return "Vocabulaire technique attendu : " + ", ".join(terms[:limit]) + "."


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def apply_corrections(text: str, rules: dict[str, str]) -> str:
    for wrong, right in sorted(rules.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![\w-]){re.escape(wrong)}(?![\w-])", right, text, flags=re.IGNORECASE)
    return text


def drop_fillers(text: str) -> str:
    text = FILLER_RE.sub("", text)
    return SPACE_RE.sub(" ", text)


def clean(text: str, rules: dict[str, str] | None = None) -> str:
    text = drop_fillers(text)
    text = apply_corrections(text, rules or {})
    return text.strip()


def sentences(segments: list[dict]) -> list[str]:
    """Split whisper segments into sentences, keeping their start time."""
    out: list[str] = []
    for seg in segments:
        chunks = re.split(r"(?<=[.!?…])\s+", seg["text"].strip())
        for i, chunk in enumerate(chunks):
            if chunk:
                out.append({"t": seg["start"] if i == 0 else seg["mid"], "text": chunk})
    return out


def paragraphs(sent_texts: list[str], per_paragraph: int = 4) -> str:
    """Group sentences into paragraphs. Punctuation comes from the LLM pass, not from here."""
    blocks = []
    for i in range(0, len(sent_texts), per_paragraph):
        blocks.append(" ".join(sent_texts[i : i + per_paragraph]).strip())
    return "\n\n".join(b for b in blocks if b)


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
