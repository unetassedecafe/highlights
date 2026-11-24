"""The pipeline: list clips, filter, download, transcribe, publish, index.

Idempotence lives in the `clips` table (`clip_id` PRIMARY KEY) plus a `status`
that records how far a clip got, so a crash resumes instead of republishing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import cleanup, index, metadata, transcribe, twitch, youtube

MIN_DURATION = 8
MAX_DURATION = 90


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def select(
    clips: list[dict],
    con,
    since_hours: int = 24,
    min_s: int = MIN_DURATION,
    max_s: int = MAX_DURATION,
    max_attempts: int = 3,
) -> list[dict]:
    """New clips in the window that still need work.

    A clip is skipped only when it is already published (`uploaded`) or has
    burned its retries. Fingerprinting on "clip_id is known" alone would make a
    failed upload permanent, which is the one thing the index must not do.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    done = {
        r["clip_id"]
        for r in con.execute(
            "SELECT clip_id FROM clips "
            "WHERE status IN ('uploaded', 'skipped') OR attempts >= ?",
            (max_attempts,),
        )
    }
    picked = []
    for clip in clips:
        if clip["clip_id"] in done:
            continue
        if not (min_s <= clip["duration_s"] <= max_s):
            continue
        created = datetime.fromisoformat(clip["created_at"].replace("Z", "+00:00"))
        if created < cutoff:
            continue
        picked.append(clip)
    picked.sort(key=lambda c: c["created_at"])
    return picked


def build_metadata(
    transcript_data: dict,
    clip: dict,
    vocab: str | Path | None = None,
    use_llm: bool = True,
    context: dict | None = None,
) -> metadata.Metadata:
    """LLM when a key is configured, templates otherwise. Never a silent guess."""
    if use_llm:
        try:
            return metadata.draft_with_llm(transcript_data["text"], context=context or {})
        except Exception as exc:  # noqa: BLE001 - degrade, but say so
            print(f"  llm metadata unavailable ({exc}); falling back to template")
    return metadata.draft_templates(clip["title"])


def publish_once(
    con,
    clip_id: str,
    video,
    payload: dict,
    settings,
    report: dict,
) -> str:
    """Upload exactly once, whatever happens between the request and the record.

    Three guards, in order of reliability:

    1. `status = 'uploaded'` in the local index — the ordinary case.
    2. An unresolved `upload_intents` row: a previous pass started an upload and
       never came back (crash, kill, network death mid-transfer). The video may
       exist on YouTube, so each intent is settled with a credential-free oEmbed
       lookup before anything is re-sent.
    3. A title search on the channel, for the case where the index itself is
       gone (a fresh workdir, a restored-from-nothing DB). It is best-effort:
       `youtube.upload` cannot call `search.list`, so this returns None when the
       scope is too narrow rather than blocking the upload.

    The intent row is written BEFORE the request, so a process killed inside
    `upload()` still leaves the evidence behind. That is the whole point: a
    status written after the fact cannot survive the crash it is meant to guard.
    """
    existing = con.execute(
        "SELECT status, yt_video_id FROM clips WHERE clip_id = ?", (clip_id,)
    ).fetchone()
    if existing and existing["status"] == "uploaded" and existing["yt_video_id"]:
        print(f"  already published as {existing['yt_video_id']}, nothing to do")
        report["skipped"] += 1
        return existing["yt_video_id"]

    for intent in index.unresolved_intents(con, clip_id):
        print(f"  settling a previous upload attempt from {intent['started_at']}")
        if intent["video_id"] and youtube.video_exists(intent["video_id"]):
            index.resolve_intent(con, intent["id"], intent["video_id"], "found on youtube")
            index.set_status(con, clip_id, "uploaded", yt_video_id=intent["video_id"], at=now_iso())
            print(f"  it did land: {intent['video_id']} — not uploading again")
            report["skipped"] += 1
            return intent["video_id"]
        # No id was ever returned, so the request most likely never completed.
        index.resolve_intent(con, intent["id"], None, "no video found, safe to retry")
        print("  no video for that attempt, safe to retry")

    token = youtube.access_token(*youtube.credentials_from_env())

    if settings.client_id:
        try:
            dup = youtube.find_duplicate(payload["titre"], token)
        except youtube.ScopeTooNarrow as exc:
            # Loud, once per clip, and then we proceed on the local index alone.
            # Silence here would be the dangerous option: the operator would
            # believe the channel was checked.
            print(f"  WARNING: {exc}")
            dup = None
        if dup:
            index.set_status(con, clip_id, "uploaded", yt_video_id=dup, at=now_iso())
            print(f"  a video with this title already exists ({dup}); adopting it")
            report["skipped"] += 1
            return dup

    intent_id = index.record_intent(con, clip_id, payload["titre"])
    try:
        vid = youtube.upload(video, payload, token, privacy=settings.privacy)
    except Exception as exc:
        # Deliberately leave the intent UNRESOLVED. A timed-out upload may still
        # have created the video on YouTube, so the next pass must be able to see
        # that the attempt happened and settle it with an oEmbed lookup. Marking
        # it resolved here would erase the only evidence of the attempt and let
        # the next pass publish a duplicate.
        print(f"  upload raised ({exc}); intent left open for the next pass")
        raise
    index.resolve_intent(con, intent_id, vid, "uploaded")
    index.set_status(con, clip_id, "uploaded", yt_video_id=vid, at=now_iso())
    return vid


def run(
    channel: str,
    settings,
    limit: int = 5,
    dry_run: bool = False,
) -> dict:
    """One pass. Returns a report dict; prints progress as it goes."""
    con = index.connect(settings.db)
    report = {"seen": 0, "new": 0, "uploaded": 0, "failed": 0, "skipped": 0, "items": []}

    clips = twitch.list_clips(channel, first=100)
    report["seen"] = len(clips)
    picked = select(clips, con, since_hours=settings.since_hours)[:limit]
    report["new"] = len(picked)
    if not picked:
        print("no new clip in the window")
        return report

    for clip in picked:
        cid = clip["clip_id"]
        work = Path(settings.workdir) / cid
        work.mkdir(parents=True, exist_ok=True)
        index.upsert_clip(
            con,
            {
                **{k: clip.get(k) for k in
                   ("clip_id", "slug", "title", "creator", "twitch_views", "created_at")},
                "twitch_url": f"https://clips.twitch.tv/{clip['slug']}",
                "duration_s": clip["duration_s"],
                "status": "listed",
                "updated_at": now_iso(),
            },
        )
        print(f"[{cid}] {clip['title']} ({clip['duration_s']}s)")

        # Two overlapping passes are the ordinary way a clip lands twice: both
        # list it before either writes a status. Take the lease first.
        if not index.claim(con, cid, settings.owner):
            print("  another pass holds this clip, skipping")
            report["skipped"] += 1
            continue

        try:
            payload_path = work / f"clip.{cid}.json"
            if payload_path.exists() and not settings.redo:
                # Resuming an unpublished clip: the expensive part is already on
                # disk, so do not download and transcribe it a second time.
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                print("  reusing the existing transcript and metadata")
                if not settings.client_id:
                    index.set_status(con, cid, "ready", at=now_iso())
                    print("  no YouTube credentials: metadata ready, stopping before upload")
                    report["skipped"] += 1
                    continue
                vid = publish_once(con, cid, work / "clip.mp4", payload, settings, report)
                token = youtube.access_token(*youtube.credentials_from_env())
                print(f"  published {vid} ({youtube.wait_processed(vid, token)})")
                report["uploaded"] += 1
                report["items"].append({"clip_id": cid, "title": payload["titre"], "problems": []})
                continue

            sources = twitch.clip_sources(clip["slug"])
            if not sources["landscape"]:
                raise RuntimeError("no landscape source available")
            raw = twitch.download(sources["landscape"], work / "raw.mp4")
            index.set_status(con, cid, "downloaded", at=now_iso())

            video = twitch.normalize(raw, work / "clip.mp4")
            info = twitch.probe(video)
            print(f"  normalized {info['width']}x{info['height']} {info['duration']:.1f}s")

            audio = transcribe.to_wav(video, work / "audio.wav")
            transcript_data = transcribe.transcribe(
                audio,
                model_size=settings.model,
                vocab=settings.vocab,
                corrections=settings.corrections,
            )
            paths = transcribe.write_result(transcript_data, work)
            print(f"  transcript {len(transcript_data['text'])} chars -> {paths['txt'].name}")

            context = {
                "twitch_url": f"https://clips.twitch.tv/{clip['slug']}",
                "youtube_url": settings.youtube_url,
            }
            meta = build_metadata(transcript_data, clip, use_llm=settings.use_llm, context=context)
            meta.chapters = metadata.build_chapters(
                transcript_data["segments"], transcript_data["duration"]
            )
            meta.description = metadata.render_description(meta, context)
            problems = metadata.validate(meta)
            if problems:
                print("  metadata problems: " + "; ".join(problems))

            payload = meta.as_dict()
            payload["clip_id"] = cid
            payload["transcript"] = transcript_data["text"]
            (work / f"clip.{cid}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            if dry_run:
                print(f"  dry-run: title={meta.title!r}")
                report["skipped"] += 1
                continue

            index.index_transcript(con, cid, meta.title, transcript_data["text"], meta.summary)
            index.remember_products(con, meta.products, now_iso())

            if settings.client_id:
                vid = publish_once(con, cid, video, payload, settings, report)
                token = youtube.access_token(*youtube.credentials_from_env())
                state = youtube.wait_processed(vid, token)
                print(f"  published {vid} ({state})")
                report["uploaded"] += 1
            else:
                # Transcribed and indexed, just not published: `ready` is the
                # truthful status, and select() will pick it up again.
                index.set_status(con, cid, "ready", at=now_iso())
                print("  no YouTube credentials: metadata ready, stopping before upload")
                report["skipped"] += 1

            report["items"].append({"clip_id": cid, "title": meta.title, "problems": problems})

        except transcribe.SilentAudio as exc:
            # Permanent: the source has no audio, so burning retries on it just
            # re-downloads 20 MB every pass. Mark it and move on.
            index.set_status(con, cid, "skipped", error=str(exc)[:500], at=now_iso(), attempt=True)
            print(f"  SKIPPED (no audio): {exc}")
            report["skipped"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the pass
            index.set_status(con, cid, "failed", error=str(exc)[:500], at=now_iso(), attempt=True)
            print(f"  FAILED: {exc}")
            report["failed"] += 1
        finally:
            index.release(con, cid)

    con.close()
    return report
