"""Twitch VOD (replay) support: listing, chunked audio extraction, long-form metadata.

A full replay is 3-4 GB and transcribes at ~14 s per audio-minute on 4 vCPU, so
it is never downloaded whole. `ffmpeg -ss -t -c copy` on the HLS playlist pulls
only the requested window (measured: 10 min in 10.8 s), and the work is done in
chunks so a crash costs one chunk, not the whole episode.
"""
from __future__ import annotations

import json
import subprocess
import urllib.parse
from pathlib import Path

from . import twitch

USHER = "https://usher.ttvnw.net/vod/{id}.m3u8"


def list_vods(channel: str, first: int = 20) -> list[dict]:
    """Newest archived broadcasts, with no auth. Zero-length entries are dropped."""
    query = (
        'query{user(login:"%s"){id login videos(first:%d,type:ARCHIVE){edges{node{'
        "id title lengthSeconds createdAt viewCount broadcastType"
        "}}}}}" % (channel, first)
    )
    data = twitch._post_gql(query)
    user = data.get("user")
    if not user:
        raise RuntimeError(f"twitch channel not found: {channel}")
    vods = []
    for edge in user["videos"]["edges"]:
        node = edge["node"]
        if not node.get("lengthSeconds"):
            # A 0-second archive is a broken entry, not an episode.
            continue
        vods.append(
            {
                "vod_id": node["id"],
                "title": node["title"],
                "duration_s": node["lengthSeconds"],
                "created_at": node["createdAt"],
                "view_count": node["viewCount"],
            }
        )
    return vods


def playback_url(vod_id: str) -> str:
    """Signed HLS master playlist for a VOD, no OAuth needed."""
    return _signed_playlist(vod_id)


def twitch_vod_info(vod_id: str) -> dict:
    """Title and length of one VOD, so the caller does not have to pass them."""
    data = twitch._post_gql(
        'query{video(id:"%s"){id title lengthSeconds createdAt}}' % vod_id
    )
    video = data.get("video")
    if not video:
        raise RuntimeError(f"vod not found: {vod_id}")
    return video


def _signed_playlist(vod_id: str) -> str:
    data = twitch._post_gql(
        'query{video(id:"%s"){id title lengthSeconds '
        'playbackAccessToken(params:{platform:"web",playerBackend:"mediaplayer",playerType:"site"})'
        "{signature value}}}" % vod_id
    )
    video = data.get("video")
    if not video:
        raise RuntimeError(f"vod not found: {vod_id}")
    token = video.get("playbackAccessToken") or {}
    if not token.get("signature"):
        raise RuntimeError(f"no playback token for vod {vod_id} (sub-only or expired?)")
    return (
        USHER.format(id=vod_id)
        + "?sig=" + urllib.parse.quote(token["signature"])
        + "&token=" + urllib.parse.quote(token["value"])
    )


def master_playlist(url: str) -> dict[str, str]:
    """Quality label -> variant URL, from the master m3u8."""
    out = subprocess.run(
        ["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", url],
        capture_output=True, text=True, check=True,
    ).stdout
    variants: dict[str, str] = {}
    pending: str | None = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-STREAM-INF") and "RESOLUTION=" in line:
            res = line.split("RESOLUTION=")[1].split(",")[0]
            pending = res.split("x")[1]
        elif line.startswith("https://") and pending:
            variants[pending] = line
            pending = None
    if not variants:
        raise RuntimeError("no variants in the master playlist (expired token?)")
    return variants


def extract_audio(
    master_url: str,
    start_s: float,
    duration_s: float,
    dst: str | Path,
    quality: str | None = None,
) -> Path:
    """Cut one window straight to 16 kHz mono WAV, never storing the video.

    `-c copy` on the video side would be faster, but going straight to audio
    means the 250 MB-per-10-min intermediate never touches the disk at all.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    variants = master_playlist(master_url)
    labels = sorted(variants, key=int)
    pick = quality if quality in variants else labels[0]
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_s:.1f}", "-t", f"{duration_s:.1f}",
            "-i", variants[pick],
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
    )
    if dst.stat().st_size < 10_000:
        raise RuntimeError(f"segment {start_s}-{start_s + duration_s}s came back empty")
    return dst


def chunks(duration_s: float, size_s: float = 600.0) -> list[tuple[float, float]]:
    """(start, length) windows covering the VOD."""
    out = []
    pos = 0.0
    while pos < duration_s:
        length = min(size_s, duration_s - pos)
        # A trailing sliver under 20 s is not worth a whole Whisper pass.
        if length < 20 and out:
            out[-1] = (out[-1][0], out[-1][1] + length)
            break
        out.append((pos, length))
        pos += length
    return out


def shift_segments(segments: list[dict], offset_s: float) -> list[dict]:
    """Make chunk-relative timings absolute again.

    Without this, every chapter after the first chunk points at 00:00-10:00.
    """
    out = []
    for seg in segments:
        moved = dict(seg)
        for key in ("start", "end", "mid"):
            if key in moved:
                moved[key] = round(moved[key] + offset_s, 2)
        out.append(moved)
    return out


def render_chapters(chapters: list[dict]) -> str:
    """YouTube chapter block: first at 00:00, MM:SS or HH:MM:SS."""
    lines = []
    for c in chapters:
        lines.append(f"{c['t']} {c['titre']}")
    return "\n".join(lines)


def download_video(vod_id: str, dst: str | Path, quality: str | None = None) -> Path:
    """Whole replay as an MP4, remuxed not re-encoded.

    Re-encoding 149 min at 1080p on 4 vCPU takes hours and loses quality for
    nothing: the HLS source is already H.264/AAC, so `-c copy` into an MP4 is
    I/O-bound and bit-identical to what Twitch serves. `+faststart` lets YouTube
    start processing before the upload finishes.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    variants = master_playlist(playback_url(vod_id))
    labels = sorted(variants, key=int)
    pick = quality if quality in variants else labels[-1]
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", variants[pick],
            "-c", "copy", "-bsf:a", "aac_adtstoasc",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
    )
    if dst.stat().st_size < 1_000_000:
        raise RuntimeError(f"download produced {dst.stat().st_size} B, that is not a replay")
    return dst


def long_timestamp(seconds: float) -> str:
    """HH:MM:SS past an hour, because MM:SS repeats itself on a 2h30 replay."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_long_chapters(
    segments: list[dict],
    duration_s: float,
    target_s: float = 600.0,
    min_len_s: float = 120.0,
    max_chapters: int = 25,
) -> list[dict]:
    """Chapters for a long replay: one per ~10 min, titled from what is said there.

    Mechanical spacing is unavoidable at this length, but a chapter titled with
    the first sentence spoken in its window is still readable, and infinitely
    better than "Partie 3". The first chapter is always 00:00 — YouTube ignores
    the whole list otherwise.
    """
    if duration_s < 120 or not segments:
        return []
    chapters = [{"t": "00:00", "titre": first_title(segments, 0)}]
    nxt = target_s
    while nxt < duration_s - min_len_s and len(chapters) < max_chapters:
        seg = next((s for s in segments if s["start"] >= nxt), None)
        if not seg:
            break
        if seg["start"] - _to_seconds(chapters[-1]["t"]) < min_len_s:
            nxt += target_s
            continue
        chapters.append({"t": long_timestamp(seg["start"]), "titre": first_title(segments, seg["start"])})
        nxt = seg["start"] + target_s
    return chapters if len(chapters) >= 3 else []


def _to_seconds(stamp: str) -> float:
    parts = [int(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def first_title(segments: list[dict], at_s: float, max_len: int = 60) -> str:
    """A short, human title from the first sizeable sentence at or after `at_s`."""
    for seg in segments:
        if seg["start"] < at_s:
            continue
        text = " ".join(seg["text"].split())
        if len(text) < 15:
            continue
        if len(text) <= max_len:
            return text.rstrip(" ,;:-—")
        cut = text[:max_len].rsplit(" ", 1)[0]
        return cut.rstrip(" ,;:-—") + "…"
    return "Suite"
