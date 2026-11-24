"""Process a whole replay: chunk by chunk, resumable, then one long-form metadata.

The chunk loop is the important part. Each chunk writes its transcript to disk
before moving on, so a crash at 80% costs one chunk instead of 34 minutes, and
re-running picks up where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import metadata, transcribe, vod


def chunk_paths(work: Path, index: int) -> tuple[Path, Path]:
    return work / f"chunk{index:03d}.wav", work / f"chunk{index:03d}.json"


def process_vod(
    vod_id: str,
    workdir: str | Path,
    model: str = "small",
    chunk_s: float = 600.0,
    vocab: str | Path | None = None,
    corrections: str | Path | None = None,
    max_chunks: int | None = None,
    title_hint: str | None = None,
    progress=print,
) -> dict:
    """Transcribe a replay in windows and build long-form metadata.

    Returns the payload plus the per-chunk artefacts. Never uploads: the caller
    decides, because a 2h30 video is a different decision from a 30 s clip.
    """
    work = Path(workdir) / vod_id
    work.mkdir(parents=True, exist_ok=True)

    master = vod.playback_url(vod_id)
    info = vod.twitch_vod_info(vod_id)
    duration = float(info["lengthSeconds"])
    progress(f"vod {vod_id}: {info['title']!r}, {duration / 60:.0f} min")

    windows = vod.chunks(duration, chunk_s)
    if max_chunks:
        windows = windows[:max_chunks]
    progress(f"{len(windows)} chunks of {chunk_s / 60:.0f} min")

    all_segments: list[dict] = []
    for i, (start, length) in enumerate(windows):
        wav, raw_json = chunk_paths(work, i)
        if raw_json.exists():
            data = json.loads(raw_json.read_text(encoding="utf-8"))
            progress(f"  chunk {i} cached ({len(data['segments'])} segments)")
        else:
            progress(f"  chunk {i}: {start / 60:.0f}–{(start + length) / 60:.0f} min")
            vod.extract_audio(master, start, length, wav)
            data = transcribe.transcribe(
                wav, model_size=model, vocab=vocab, corrections=corrections
            )
            data["offset"] = start
            raw_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_segments.extend(vod.shift_segments(data["segments"], float(data.get("offset", start))))

    transcript = " ".join(s["text"] for s in all_segments).strip()
    progress(f"transcript: {len(transcript)} chars over {len(all_segments)} segments")

    chapters = vod.build_long_chapters(all_segments, duration)
    if not chapters:
        # A partial run (--max-chunks) transcribes a fraction of the replay, so
        # the spacing rule cannot reach three chapters. Say so: an empty
        # chapters.txt next to a 9000 s duration looks like a bug.
        covered = all_segments[-1]["end"] if all_segments else 0
        progress(
            f"no chapters: only {covered / 60:.0f} min of {duration / 60:.0f} min transcribed"
            if covered < duration * 0.9
            else "no chapters: not enough distinct speech to place three"
        )
    else:
        progress(f"chapters: {len(chapters)}")

    meta = metadata.draft_templates(title_hint or info["title"])
    meta.description = metadata.render_description(
        meta,
        {"vod_url": f"https://www.twitch.tv/videos/{vod_id}"},
    )
    meta.chapters = chapters

    payload = meta.as_dict()
    payload["vod_id"] = vod_id
    payload["duration_s"] = duration
    payload["transcript"] = transcript
    payload["segments"] = len(all_segments)
    payload["chapters_text"] = vod.render_chapters(chapters)
    (work / "vod.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (work / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    (work / "chapters.txt").write_text(payload["chapters_text"] + "\n", encoding="utf-8")

    payload["problems"] = metadata.validate(meta)
    payload["workdir"] = str(work)
    return payload
