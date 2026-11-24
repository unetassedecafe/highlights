"""Twitch clip listing and download.

Two paths, and the cheap one is not the documented one:

  - Helix `/helix/clips` needs an app token but returns no media URL.
  - Helix `/helix/clips/downloads` returns media URLs but needs a user token
    with `editor:manage:clips` — the only real OAuth friction in the project.
  - Public GraphQL `clip(slug: …)` returns `videoQualities[].sourceURL` with no
    token at all — but those URLs answer 401. They only work once you append
    the `playbackAccessToken`'s `sig` and `token` as query parameters, and that
    token itself needs `platform`, `playerBackend` and `playerType` or the GQL
    call fails on a missing required argument. That combination is what v1
    uses: no OAuth, no user token, and it served a 1080p 41 s clip end to end.

The Helix path stays here for the cases GraphQL cannot serve (a clip whose
sourceURLs are gone, a channel-wide sweep at scale).
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GQL = "https://gql.twitch.tv/gql"
HELIX = "https://api.twitch.tv/helix"
# Public web client id, the one the Twitch site itself uses for the clip player.
WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"


def _post_gql(query: str, attempts: int = 3) -> dict:
    """POST to GQL, retrying the transient `service error` the clips field returns.

    Twitch's `user.clips` answers `{"errors":[{"message":"service error",
    "path":["user","clips"]}]}` intermittently — the same query succeeds a
    second later. Treating it as fatal makes a cron pass skip a whole stream day.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            GQL,
            data=json.dumps({"query": query}).encode(),
            headers={"Client-ID": WEB_CLIENT_ID, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                raise
            time.sleep(1.5 * (attempt + 1))
            continue
        errors = payload.get("errors") or []
        transient = any(
            "service error" in str(e.get("message", "")).lower() for e in errors
        )
        if errors and transient and attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
            continue
        if errors:
            raise RuntimeError(f"twitch gql: {errors}")
        return payload["data"]
    raise RuntimeError(f"twitch gql: giving up after {attempts} attempts ({last})")


def list_clips(channel: str, first: int = 20) -> list[dict]:
    """Newest clips of a channel, with no auth."""
    query = (
        'query{user(login:"%s"){id login clips(first:%d){edges{node{'
        "id slug title durationSeconds createdAt viewCount curator{login}"
        "}}}}}" % (channel, first)
    )
    data = _post_gql(query)
    user = data.get("user")
    if not user:
        raise RuntimeError(f"twitch channel not found: {channel}")
    clips = []
    for edge in user["clips"]["edges"]:
        node = edge["node"]
        clips.append(
            {
                "clip_id": node["id"],
                "slug": node["slug"],
                "title": node["title"],
                "duration_s": node["durationSeconds"],
                "created_at": node["createdAt"],
                "twitch_views": node["viewCount"],
                "creator": (node.get("curator") or {}).get("login"),
            }
        )
    return clips


def clip_sources(slug: str) -> dict:
    """Direct, signed media URLs for one clip, best quality first.

    The sourceURL alone 401s. The playback token goes in two query parameters,
    and the token request needs the full player param triple.
    """
    data = _post_gql(
        'query{clip(slug:"%s"){id title durationSeconds '
        'playbackAccessToken(params:{platform:"web",playerBackend:"mediaplayer",playerType:"site"})'
        "{signature value} videoQualities{quality sourceURL}}}" % slug
    )
    clip = data.get("clip")
    if not clip:
        raise RuntimeError(f"clip not found: {slug}")
    token = clip.get("playbackAccessToken") or {}
    query = ""
    if token.get("signature") and token.get("value"):
        query = "?sig=" + urllib.parse.quote(token["signature"]) + "&token=" + urllib.parse.quote(token["value"])
    qualities = sorted(clip["videoQualities"], key=lambda q: int(q["quality"]), reverse=True)
    return {
        "clip_id": clip["id"],
        "title": clip["title"],
        "duration_s": clip["durationSeconds"],
        "landscape": next((q["sourceURL"] + query for q in qualities if q["sourceURL"]), None),
        "qualities": {q["quality"]: q["sourceURL"] + query for q in qualities if q["sourceURL"]},
        "expires": json.loads(token["value"]).get("expires") if token.get("value") else None,
    }


def download(url: str, dst: str | Path, timeout: int = 120) -> Path:
    """Download now: these URLs are short-lived, queueing them loses them."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cuistops-highlights/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dst.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    if dst.stat().st_size < 10_000:
        raise RuntimeError(f"suspiciously small download ({dst.stat().st_size} B): {url}")
    return dst


def normalize(src: str | Path, dst: str | Path, height: int = 1080) -> Path:
    """H.264 + AAC, faststart, even dimensions. Twitch already gives 16:9 1080p."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-vf", f"scale=-2:{height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(dst),
        ],
        check=True,
    )
    return dst


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    return {
        "duration": float(data["format"].get("duration", 0)),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
        "size": int(data["format"].get("size", 0)),
    }
