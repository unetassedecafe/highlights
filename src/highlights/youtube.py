"""YouTube upload: resumable, unlisted by default, then wait for processing.

Quota since 2026-06-01: `videos.insert` costs 1 unit in its own bucket capped
at 100 calls/day, so 100 uploads/day. `status.privacyStatus` defaults to
PRIVATE when omitted; we always set it explicitly.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.googleapis.com/youtube/v3"
OAUTH = "https://oauth2.googleapis.com/token"
UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
MAX_ATTEMPTS = 8
CHUNK = 8 * 1024 * 1024


def access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    req = urllib.request.Request(
        OAUTH,
        data=urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    if "access_token" not in data:
        raise RuntimeError(f"no access token: {data}")
    return data["access_token"]


class ScopeTooNarrow(RuntimeError):
    """The token lacks a scope a check needs. Never swallow this as "no problem"."""


class Privacy:
    """YouTube's three visibility values, nothing else.

    There is no "invisible" mode. `private` is the closest: only the channel
    owner can watch, and it does not appear in search or on the channel page.
    `unlisted` is link-only — anyone holding the URL can watch.
    """

    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"
    ALL = (PUBLIC, UNLISTED, PRIVATE)


def upload(
    path: str | Path,
    metadata: dict,
    token: str,
    privacy: str = Privacy.PRIVATE,
    category_id: str = "28",  # Science & Technology
) -> str:
    """Resumable upload. `metadata` holds titre/description/tags. Returns the video id.

    Defaults to PRIVATE: an omitted `privacyStatus` is private anyway, so
    defaulting to anything else would publish more than the caller asked for.
    """
    if privacy not in Privacy.ALL:
        raise ValueError(f"privacyStatus must be one of {Privacy.ALL}, got {privacy!r}")
    path = Path(path)
    size = path.stat().st_size
    body = {
        "snippet": {
            "title": metadata["titre"],
            "description": metadata["description"],
            "tags": metadata.get("tags", []),
            "categoryId": category_id,
            "defaultLanguage": "fr",
            "defaultAudioLanguage": "fr",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "license": "youtube",
            "embeddable": True,
        },
    }
    init = urllib.request.Request(
        UPLOAD + "?uploadType=resumable&part=snippet,status",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(size),
            "X-Upload-Content-Type": "video/mp4",
        },
        method="POST",
    )
    with urllib.request.urlopen(init, timeout=60) as resp:
        session = resp.headers["Location"]

    video_id = ""
    sent = 0
    with path.open("rb") as fh:
        while sent < size:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            req = urllib.request.Request(
                session,
                data=chunk,
                headers={
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {sent}-{sent + len(chunk) - 1}/{size}",
                },
                method="PUT",
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    payload = json.load(resp)
                    video_id = payload.get("id", "")
            except urllib.error.HTTPError as exc:
                # 308 is the expected "resume incomplete" answer, not a failure.
                if exc.code == 308:
                    range_hdr = exc.headers.get("Range")
                    if range_hdr:
                        sent = int(range_hdr.split("-")[1]) + 1
                        fh.seek(sent)
                        continue
                raise
            sent += len(chunk)
    if not video_id:
        raise RuntimeError("upload finished without a video id")
    return video_id


def wait_processed(video_id: str, token: str, timeout_s: int = 900) -> str:
    """A brand-new video is listed before it is playable: poll uploadStatus."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        req = urllib.request.Request(
            f"{API}/videos?part=status,processingDetails&id={video_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.load(resp).get("items", [])
        if not items:
            time.sleep(5)
            continue
        status = items[0].get("status", {})
        state = status.get("uploadStatus")
        details = items[0].get("processingDetails", {})
        if status.get("failureReason") or status.get("rejectionReason"):
            raise RuntimeError(
                f"youtube rejected {video_id}: {status.get('failureReason') or status.get('rejectionReason')}"
            )
        if state == "processed" or details.get("processingStatus") == "succeeded":
            return "processed"
        time.sleep(10)
    return "timeout"


def video_exists(video_id: str) -> bool:
    """Is this id a live, watchable video? Needs no credential at all.

    The oEmbed endpoint answers 200 with the title and channel for any public or
    unlisted video and 404 for anything else, which makes it the cheapest way to
    settle "did my half-finished upload actually land?" — `videos.list` cannot be
    used for that here, since `youtube.upload` alone cannot call it.
    """
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "cuistops-highlights/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 401, 403, 404):
            return False
        raise


def find_duplicate(title: str, token: str, channel_id: str | None = None) -> str | None:
    """Return the id of an existing video carrying this exact title, or None.

    `search.list` needs a read scope: with `youtube.upload` alone Google answers
    `403 insufficientPermissions` (measured). Returning None there would claim
    "no duplicate" when nothing was actually checked, so this raises
    `ScopeTooNarrow` instead and the caller decides what to do about it.
    """
    query = urllib.parse.urlencode(
        {"part": "snippet", "forMine": "true", "q": title, "type": "video", "maxResults": "5"}
    )
    req = urllib.request.Request(
        f"{API}/search?{query}", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.load(resp).get("items", [])
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ScopeTooNarrow(
                "cannot search the channel for an existing upload: the token has no "
                "read scope. Add https://www.googleapis.com/auth/youtube.readonly and "
                "re-consent, or rely on the local index alone."
            ) from exc
        raise
    for item in items:
        snippet = item.get("snippet", {})
        if snippet.get("title", "").strip() != title.strip():
            continue
        if channel_id and snippet.get("channelId") not in (None, channel_id):
            continue
        return item["id"]["videoId"]
    return None


def credentials_from_env() -> tuple[str, str, str]:
    """OAuth triple from the environment. Never put these in a file in the repo."""
    try:
        return (
            os.environ["YT_CLIENT_ID"],
            os.environ["YT_CLIENT_SECRET"],
            os.environ["YT_REFRESH_TOKEN"],
        )
    except KeyError as exc:
        raise RuntimeError(f"missing YouTube credential in environment: {exc}") from exc
