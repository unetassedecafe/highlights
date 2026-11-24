"""One clip in, YouTube-ready JSON out. The command the skill calls."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import metadata, transcribe, twitch


@dataclass
class Settings:
    workdir: Path
    db: Path
    vocab: Path | None = None
    corrections: Path | None = None
    model: str = "small"
    use_llm: bool = True
    client_id: str | None = None
    privacy: str = "private"
    youtube_url: str | None = None
    since_hours: int = 24
    redo: bool = False
    owner: str = ""


def process_file(
    path: Path,
    out_dir: Path | None = None,
    model: str = "small",
    use_llm: bool = True,
    vocab: Path | None = None,
    corrections: Path | None = None,
    twitch_title: str | None = None,
) -> dict:
    """ffmpeg → whisper → metadata for a local file. No upload, no index."""
    out = Path(out_dir or path.parent)
    out.mkdir(parents=True, exist_ok=True)

    media = path
    if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a"}:
        media = transcribe.to_wav(path, out / "audio.wav")

    data = transcribe.transcribe(
        media, model_size=model, vocab=vocab, corrections=corrections
    )
    paths = transcribe.write_result(data, out)

    context = {"source": str(path)}
    if twitch_title:
        context["twitch_title"] = twitch_title
    try:
        meta = metadata.draft_with_llm(data["text"], context=context) if use_llm else None
    except Exception as exc:  # noqa: BLE001
        print(f"llm metadata unavailable ({exc}); using template", file=sys.stderr)
        meta = None
    if meta is None:
        meta = metadata.draft_templates(twitch_title or path.stem)

    meta.chapters = metadata.build_chapters(data["segments"], data["duration"])
    problems = metadata.validate(meta)

    result = meta.as_dict()
    result["transcript"] = data["text"]
    result["duration_s"] = data["duration"]
    result["problems"] = problems
    out_json = out / "clip.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "json": str(out_json),
        "transcript_txt": str(paths["txt"]),
        "transcript_json": str(paths["json"]),
        "duration_s": data["duration"],
        "problems": problems,
        "title": meta.title,
    }


def process_clip(slug: str, out_dir: Path, model: str = "small", use_llm: bool = True,
                 vocab: Path | None = None, corrections: Path | None = None) -> dict:
    """Same thing, starting from a Twitch clip slug."""
    sources = twitch.clip_sources(slug)
    raw = twitch.download(sources["landscape"], Path(out_dir) / "raw.mp4")
    video = twitch.normalize(raw, Path(out_dir) / "clip.mp4")
    result = process_file(video, Path(out_dir), model, use_llm, vocab, corrections,
                          twitch_title=sources["title"])
    result["clip_id"] = sources["clip_id"]
    result["slug"] = slug
    result["twitch_title"] = sources["title"]
    result["video"] = str(video)
    return result
