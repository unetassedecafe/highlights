"""Cut vertical shorts out of a replay, choosing the passages from the transcript.

Two selection lanes, deliberately separate:

  - audio (NeonClipping's `smart.find`): loudness spikes, raised voice, motion.
    Good for reaction content, blind on a technical talk.
  - transcript: an LLM reads what is said and picks self-contained passages.
    This module is that lane.

A candidate is always a run of whole segments: a short that starts mid-sentence
is dead on arrival, so boundaries snap to segment edges before anything renders.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

# 9:16 from a 1920x1080 frame without letterboxing: 608 wide keeps the face and
# drops the side panels, and 608/1080 == 9/16 exactly.
SHORT_W, SHORT_H = 608, 1080


def load_segments(vod_dir: str | Path) -> list[dict]:
    """Every segment of a processed replay, in order, with absolute timings.

    Reads the per-chunk files `clip-transcript-publish` leaves behind. Their
    timings are already shifted to absolute, so no offset maths here.
    """
    vod_dir = Path(vod_dir)
    chunks = sorted(vod_dir.glob("chunk[0-9][0-9][0-9].json"))
    if not chunks:
        raise FileNotFoundError(
            f"no chunk*.json in {vod_dir}: run `highlights vod <id>` first"
        )
    segments: list[dict] = []
    for path in chunks:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments.extend(data["segments"])
    segments.sort(key=lambda s: s["start"])
    return segments


def sentences(segments: list[dict], max_len: float = 20.0, gap: float = 0.6) -> list[dict]:
    """Merge the 2-second Whisper segments into sentence-sized units.

    Whisper emits a segment every ~2 s (measured median: 2.0 s, 27 chars), which
    cuts mid-sentence and makes every window start on a fragment — a model asked
    to judge "does this stand alone?" then says yes to everything, because it only
    ever sees fragments. Merging on punctuation and pauses gives units a human can
    read a boundary on.
    """
    out: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "text": text}
            continue
        closes = cur["text"].rstrip().endswith((".", "!", "?", "…"))
        too_long = seg["end"] - cur["start"] > max_len
        long_pause = seg["start"] - cur["end"] > gap
        if closes or too_long or long_pause:
            out.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "text": text}
        else:
            cur["text"] = f"{cur['text']} {text}"
            cur["end"] = seg["end"]
    if cur and cur["text"]:
        out.append(cur)
    return out


def windows(segments: list[dict], length: float = 45.0, hop: float | None = None) -> list[dict]:
    """Candidate passages: consecutive units, snapped to their edges.

    Windows end on a unit boundary rather than at `length` exactly, so the last
    word is never cut. `hop` defaults to half the length, which gives the LLM
    overlapping candidates and costs nothing.

    Callers should pass sentence-sized units (`sentences()`), not raw Whisper
    segments: a window built from 2-second fragments starts mid-thought.
    """
    if not segments:
        return []
    hop = hop or length / 2
    out = []
    starts = []
    t = segments[0]["start"]
    end_of_all = segments[-1]["end"]
    while t < end_of_all:
        starts.append(t)
        t += hop

    for start in starts:
        picked = [s for s in segments if s["start"] >= start]
        if not picked:
            break
        chosen = []
        for seg in picked:
            if seg["end"] - start > length:
                break
            chosen.append(seg)
        if not chosen:
            continue
        text = " ".join(s["text"].strip() for s in chosen).strip()
        if len(text) < 120:          # too little said to be a short
            continue
        out.append(
            {
                "start": round(chosen[0]["start"], 2),
                "end": round(chosen[-1]["end"], 2),
                "text": text,
                "segments": chosen,
            }
        )
    return out


def candidates_for_llm(windows_: list[dict], segments: list[dict] | None = None,
                       context: int = 2, limit: int = 60) -> str:
    """Render candidates as a numbered list the model can answer about by index.

    When `segments` is given, each candidate is preceded by the last few segments
    *before* it, marked as context. Without that, a passage opening on "Non, ça n'a
    rien à ça" looks self-contained to the model, which then marks everything
    `compréhensible_seul: true` — measured on a real replay.
    """
    by_start = {round(s["start"], 2): i for i, s in enumerate(segments or [])}
    lines = []
    for i, w in enumerate(windows_[:limit]):
        text = " ".join(w["text"].split())
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0] + "…"
        if segments:
            pos = by_start.get(round(w["start"], 2))
            if pos:
                before = segments[max(0, pos - context):pos]
                if before:
                    lead = " ".join(s["text"].strip() for s in before)
                    lines.append(f"    (contexte avant, NE PAS inclure : …{lead[-220:]})")
        stamp = _mmss(w["start"])
        lines.append(f"[{i}] {stamp} ({w['end'] - w['start']:.0f}s) {text}")
    return "\n".join(lines)


def _mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


PROMPT = """Tu choisis des extraits courts pour les réseaux sociaux dans la transcription
d'une émission technique francophone.

Sujet de l'épisode : {topic}

Extraits candidats :
{candidates}

Choisis les {count} meilleurs extraits. Un bon extrait :
- se comprend seul, sans avoir suivi l'émission avant ;
- contient une idée complète : un problème posé et sa réponse, une conclusion, un constat net ;
- ne commence pas au milieu d'une phrase et ne se termine pas en suspens ;
- évite les salutations, les digressions, les « euh » et les blagues sans contexte ;
- se termine sur une pensée finie : pas de « je ne t'ai pas dit que… » qui coupe net.

Le titre doit décrire ce qui est RÉELLEMENT dit dans l'extrait, pas le sujet dont
on parle à côté. Si l'extrait est une private joke ou une digression sur la
semaine dernière, ne le choisis pas : préfère un extrait qui explique quelque
chose, même moins spectaculaire.

Avant de répondre, relis chaque extrait et demande-toi : un spectateur qui tombe
dessus sans contexte comprend-il de quoi il s'agit ? Si non, écarte-le.

Réponds uniquement en JSON :
{{"choix": [{{"i": <index>, "titre": "<60 caractères max>", "pourquoi": "<une phrase>", "compréhensible_seul": true|false}}]}}
Ne choisis que des index présents dans la liste, sans doublon."""


def pick(windows_: list[dict], count: int = 5, topic: str = "", llm=None,
         segments: list[dict] | None = None) -> list[dict]:
    """Ask a model which candidates are worth publishing.

    `llm` is a callable taking a prompt and returning JSON text, so this stays
    testable and provider-agnostic: wire it to an OpenAI-compatible endpoint.
    `segments` (the full transcript) lets each candidate be shown with its lead-in,
    which is what makes the self-contained judgement possible at all.
    """
    if not windows_:
        return []
    if llm is None:
        raise ValueError("pick() needs an llm callable: no passage can be chosen without one")
    pool = windows_[:60]
    prompt = PROMPT.format(
        topic=topic or "(non précisé)",
        candidates=candidates_for_llm(pool, segments=segments),
        count=count,
    )
    raw = llm(prompt)
    data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    chosen = []
    seen = set()
    for item in data.get("choix", [])[:count]:
        i = int(item["i"])
        if i in seen or not (0 <= i < len(pool)):
            continue
        # The model is asked to flag passages that need prior context. Respect the
        # flag: a clip it admits is not self-contained is a clip viewers scroll past.
        if item.get("compréhensible_seul") is False:
            continue
        seen.add(i)
        clip = dict(pool[i])
        clip["titre"] = (item.get("titre") or "").strip()[:60]
        clip["pourquoi"] = (item.get("pourquoi") or "").strip()
        clip["standalone"] = item.get("compréhensible_seul")
        chosen.append(clip)
    return chosen


def trim_to_sentence(clip: dict, min_s: float = 15.0) -> dict:
    """End the clip on a sentence that actually closes.

    A clip ending on "…c'est pas un gros boulot de tour" is worse than a shorter
    clip: the viewer waits for a word that never comes. The rule is now strict —
    the last kept segment must end on `.`, `!`, `?` or `…`. Segments are dropped
    until one does, stopping if that would take the clip under `min_s`. If nothing
    closes at all, the clip is marked `truncated` so the caller can drop it rather
    than publish a dangling sentence.
    """
    segs = list(clip["segments"])
    if not segs:
        return clip

    def closes(seg: dict) -> bool:
        return seg["text"].strip().rstrip('"»').endswith((".", "!", "?", "…"))

    dropped = 0
    while len(segs) > 1 and not closes(segs[-1]):
        remaining = segs[-2]["end"] - segs[0]["start"]
        if remaining < min_s:
            break
        segs.pop()
        dropped += 1

    out = dict(clip)
    out["segments"] = segs
    out["end"] = round(segs[-1]["end"], 2)
    out["text"] = " ".join(s["text"].strip() for s in segs).strip()
    out["truncated"] = not closes(segs[-1])
    out["trimmed_segments"] = dropped
    return out


def render_captions(segments: list[dict], clip_start: float, clip_end: float) -> str:
    """ASS subtitle file, timings shifted to the clip and wrapped to two lines."""
    def stamp(t: float) -> str:
        t = max(0.0, t - clip_start)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"

    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\n"
        "PlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        "Style: Short,DejaVu Sans,58,&H00FFFFFF,&H00000000,&H80000000,"
        "1,3,1,2,60,60,180\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        text = seg["text"].strip()
        if not text:
            continue
        start_f = max(seg["start"], clip_start)
        end_f = min(seg["end"], clip_end)

        # Two lines is the limit before a caption covers the speaker's mouth, so a
        # long sentence is split ACROSS TIME, not squeezed onto a third line and
        # never dropped. Reading pauses are placed on word boundaries.
        chunks = _split_for_display(text, width=34, max_lines=2)
        dur = max(0.01, end_f - start_f)
        for j, chunk in enumerate(chunks):
            a = start_f + dur * j / len(chunks)
            b = start_f + dur * (j + 1) / len(chunks)
            events.append(
                f"Dialogue: 0,{stamp(a)},{stamp(b)},Short,,0,0,0,,{chunk}"
            )
    return header + "\n".join(events) + "\n"


def _greedy_lines(text: str, width: int) -> list[str]:
    """Hard wrap on word boundaries, every line at most `width` characters."""
    out, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}".strip()
        if len(cand) <= width:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def _wrap(text: str, width: int = 34, max_lines: int = 3) -> str:
    """Wrap into balanced lines of at most `width`, never dropping a word.

    A greedy fill produces a long first line and a stub second one ("C'est les gens
    qui utilisent le LLM dès le tout premier / début"). Lines are split at balanced
    character shares instead, and the result is verified against the hard width:
    if any line overflows, this falls back to the greedy wrap rather than ever
    cutting text.
    """
    words = text.split()
    if not words:
        return ""
    if len(text) <= width:
        return text

    n = min(max_lines, -(-len(text) // width))  # ceil, capped
    target = len(text) / n
    lines: list[list[str]] = [[]]
    size = 0
    for w in words:
        over_share = size + len(w) + 1 > target * len(lines)
        if lines[-1] and len(lines) < n and over_share:
            lines.append([])
            size = 0
        lines[-1].append(w)
        size += len(w) + 1

    out = [" ".join(l) for l in lines if l]
    if any(len(l) > width for l in out):
        return "\\N".join(_greedy_lines(text, width))
    return "\\N".join(out)


def _split_for_display(text: str, width: int = 34, max_lines: int = 2) -> list[str]:
    """Cut a caption into display chunks of at most `max_lines` lines each.

    The two constraints conflict on a long sentence: it will not fit in two lines,
    and a third line covers the speaker's mouth. Splitting across time satisfies
    both — the viewer reads the first half, then the second — and loses no word.

    The split is recursive and verified, not estimated: if a chunk still needs more
    than `max_lines` lines after wrapping, it is halved and each half retried. That
    terminates (a single word always fits) and guarantees the invariant.
    """
    if not text.split():
        return []
    if len(_greedy_lines(text, width)) <= max_lines:
        return [_wrap(text, width=width, max_lines=max_lines)]

    # Halve on a word boundary near the middle, preferring a punctuation break.
    words = text.split()
    mid = len(text) // 2
    best = len(words) // 2
    best_score = abs(mid)
    pos = 0
    for i, w in enumerate(words[:-1]):
        pos += len(w) + 1
        if abs(pos - mid) < best_score:
            best, best_score = i, abs(pos - mid)
        if w.endswith((",", ".", "?", "!", "…", ":", ";")) and abs(pos - mid) < len(text) * 0.25:
            best = i
            break
    left = " ".join(words[: best + 1])
    right = " ".join(words[best + 1:])
    if not right:
        return [_wrap(text, width=width, max_lines=max_lines)]
    return _split_for_display(left, width, max_lines) + _split_for_display(right, width, max_lines)


def probe_size(video: str | Path) -> tuple[int, int]:
    """Width and height of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def crop_parts(src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Geometry for a 9:16 centre crop, whatever the source size.

    The short side is fixed by the frame height: a 608x1080 window is exactly
    9:16, and a 1280x720 source needs `crop=405:720`, not a hardcoded 608x1080 —
    which fails outright when the source is shorter than the crop.
    """
    h = src_h if src_h % 2 == 0 else src_h - 1
    w = int(round(h * 9 / 16))
    w -= w % 2                       # H.264 needs even dimensions
    if w > src_w:                    # portrait source: fit the other way
        w = src_w - (src_w % 2)
        h = int(round(w * 16 / 9))
        h -= h % 2
    x = (src_w - w) // 2
    y = (src_h - h) // 2
    return w, h, x, y


def render(
    video: str | Path,
    clip: dict,
    out: str | Path,
    crop: bool = True,
    captions: bool = True,
    ass_path: str | Path | None = None,
    size: tuple[int, int] | None = None,
) -> Path:
    """Cut one short: vertical crop, burned captions, loudness normalised.

    The crop is a centre cut, not a pad: on a 16:9 stream the speaker sits in the
    middle, so the vertical window keeps the face and loses the chat panel.
    `crop=False` keeps the full frame for a demo where the terminal matters.

    The source resolution is read rather than assumed: a 720p replay is the norm
    on Twitch, and a hardcoded 1080 crop fails with "nothing was written".
    """
    video, out = Path(video), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h = size or probe_size(video)

    # Two clocks disagree here. Transcript timings are absolute in the replay
    # (3673 s for a passage an hour in); the mp4 handed to ffmpeg starts at 0.
    # Without the offset ffmpeg finds no frame at -ss and writes an empty file
    # ("No filtered frames for output stream"). `base` says where the file starts
    # on the replay timeline; a clip carrying `file_start` is authoritative.
    start = clip["start"] - float(clip.get("file_start", 0.0))
    duration = clip["end"] - clip["start"]
    start = max(0.0, start)

    vf = []
    if crop:
        w, h, x, y = crop_parts(src_w, src_h)
        vf.append(f"crop={w}:{h}:{x}:{y}")
        vf.append("scale=1080:1920:flags=lanczos")
    else:
        # Keep the full 16:9 frame inside a 1920-tall canvas: width follows the
        # aspect ratio, and `-2` alone would *upscale* a 1280x720 source to 3414px
        # wide (measured), so cap it at 1080 by fitting, not stretching.
        vf.append("scale=1080:-2:flags=lanczos")
        vf.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black")
    if captions and ass_path:
        # Quotes and colons are the two characters that break a filter argument.
        safe = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        vf.append(f"ass='{safe}'")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(video),
        "-vf", ",".join(vf),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    if out.stat().st_size < 50_000:
        raise RuntimeError(f"render produced {out.stat().st_size} B for {out.name}")
    return out
