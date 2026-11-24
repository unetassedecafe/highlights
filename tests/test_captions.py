"""Caption wrapping and clip endings — the two defects a viewer sees first.

Both were shipped broken: a 70-character sentence became "…pourrait pas / faire"
(two thirds of the text silently dropped) and clips ended on "…un gros boulot de
tour". The existing tests checked the ASS header, never the words.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from highlights import shorts


def test_wrap_never_drops_words():
    """A 70-char sentence must survive whole, not shrink to its first 40 chars."""
    text = "Attend, est-ce qu'on pourrait pas faire tourner des VMs dans Kubernetes?"
    wrapped = shorts._wrap(text, width=34)
    flat = wrapped.replace("\\N", " ")
    assert flat == text
    for w in text.split():
        assert w in flat


def test_wrap_respects_the_width():
    text = "en plus, ça ne change pas le monde, mais c'est ça de gagner et c'est ça de simplicité."
    for line in shorts._wrap(text, width=34).split("\\N"):
        assert len(line) <= 40, line


def test_wrap_does_not_leave_a_lonely_word():
    wrapped = shorts._wrap("Attend est-ce qu'on pourrait pas faire tourner des VMs ici", width=34)
    last = wrapped.split("\\N")[-1]
    assert len(last.split()) > 1, f"last line is a single word: {wrapped!r}"


def test_a_very_long_sentence_spans_several_chunks_without_loss():
    """120 words cannot fit in two lines; they must not be cut either."""
    text = " ".join(["mot"] * 120)
    chunks = shorts._split_for_display(text, width=34, max_lines=2)
    assert len(chunks) > 1
    for c in chunks:
        assert c.count("\\N") <= 1, "no chunk may exceed two lines"
    flat = " ".join(c.replace("\\N", " ") for c in chunks)
    assert flat == text, "every word must survive"


def test_wrap_keeps_a_short_line_whole():
    assert shorts._wrap("Bonjour.", width=34) == "Bonjour."


def test_captions_contain_the_full_sentence_text():
    """Split across time is fine; losing the end of the sentence is not."""
    seg = {"start": 0.0, "end": 4.0,
           "text": "Attend, est-ce qu'on pourrait pas faire tourner des VMs dans Kubernetes?"}
    ass = shorts.render_captions([seg], 0.0, 45.0)
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    joined = " ".join(e.split(",,", 1)[1] for e in events).replace("\\N", " ")
    assert "Kubernetes?" in joined, f"the end of the sentence is missing: {joined!r}"
    for w in seg["text"].split():
        assert w in joined, f"lost word: {w}"


def test_a_clip_ending_mid_sentence_is_trimmed_back():
    clip = {
        "start": 0.0, "end": 40.0,
        "segments": [
            {"start": 0.0, "end": 25.0, "text": "Voici une idée complète et finie."},
            {"start": 25.0, "end": 40.0, "text": "et c'est pas un gros boulot de tour"},
        ],
    }
    out = shorts.trim_to_sentence(clip)
    assert not out["truncated"]
    assert out["end"] == 25.0
    assert out["text"].endswith("finie.")


def test_a_clip_that_cannot_close_is_flagged_not_published():
    clip = {
        "start": 0.0, "end": 16.0,
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "ça commence comme ça"},
            {"start": 10.0, "end": 16.0, "text": "et ça continue sans fin"},
        ],
    }
    out = shorts.trim_to_sentence(clip, min_s=15.0)
    assert out["truncated"] is True, "a dangling clip must be marked for dropping"


def test_a_clip_already_closing_is_untouched():
    clip = {
        "start": 0.0, "end": 30.0,
        "segments": [
            {"start": 0.0, "end": 15.0, "text": "Première phrase."},
            {"start": 15.0, "end": 30.0, "text": "Deuxième phrase complète."},
        ],
    }
    out = shorts.trim_to_sentence(clip)
    assert out["truncated"] is False
    assert out["trimmed_segments"] == 0
    assert len(out["segments"]) == 2


def test_trimming_stops_at_the_minimum_duration():
    clip = {
        "start": 0.0, "end": 20.0,
        "segments": [
            {"start": 0.0, "end": 12.0, "text": "Une phrase qui finit bien."},
            {"start": 12.0, "end": 20.0, "text": "et ça part en vrille"},
        ],
    }
    out = shorts.trim_to_sentence(clip, min_s=15.0)
    assert len(out["segments"]) == 2, "12 s is under the floor, so nothing is trimmed"
    assert out["truncated"] is True
