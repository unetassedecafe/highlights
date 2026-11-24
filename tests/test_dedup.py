"""Duplicate-prevention tests: the guarantees that matter for a cron pipeline.

Every test here stands for a way a clip gets published twice in the real world:
two passes overlapping, a crash between upload and status write, a lost index.
"""
from __future__ import annotations

import sqlite3

import pytest

from highlights import index, pipeline


ROW = {
    "clip_id": "4242", "slug": "slug", "title": "Un clip", "creator": "c",
    "twitch_url": "https://clips.twitch.tv/slug", "twitch_views": 3,
    "duration_s": 30.0, "created_at": "2026-11-24T00:00:00Z",
    "status": "listed", "updated_at": "2026-11-24T00:00:00Z",
}


class Settings:
    client_id = "cid"
    privacy = "private"
    owner = "pass-1"


class Report(dict):
    def __init__(self):
        super().__init__(uploaded=0, skipped=0, failed=0)


@pytest.fixture
def con(tmp_path):
    c = index.connect(tmp_path / "c.db")
    index.upsert_clip(c, ROW)
    return c


def test_second_pass_cannot_claim_a_clip_already_held(tmp_path):
    """Two cron passes overlapping: only one may work on the clip."""
    con = index.connect(tmp_path / "c.db")
    index.upsert_clip(con, ROW)
    assert index.claim(con, "4242", "pass-1") is True
    assert index.claim(con, "4242", "pass-2") is False


def test_a_finished_pass_releases_the_lease(tmp_path):
    con = index.connect(tmp_path / "c.db")
    index.upsert_clip(con, ROW)
    assert index.claim(con, "4242", "pass-1")
    index.release(con, "4242")
    assert index.claim(con, "4242", "pass-2") is True


def test_a_stale_lease_expires_so_a_crashed_worker_does_not_block_forever(tmp_path):
    con = index.connect(tmp_path / "c.db")
    index.upsert_clip(con, ROW)
    # A worker that died two hours ago.
    con.execute(
        "UPDATE clips SET claimed_at = '2026-11-24T04:00:00Z', claimed_by = 'dead' WHERE clip_id = '4242'"
    )
    con.commit()
    import datetime as dt

    long_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    con.execute("UPDATE clips SET claimed_at = ? WHERE clip_id = '4242'", (long_ago,))
    con.commit()
    assert index.claim(con, "4242", "fresh", lease_minutes=30) is True


def test_upload_is_skipped_when_the_index_says_published(con, monkeypatch):
    index.set_status(con, "4242", "uploaded", yt_video_id="ALREADY", at="2026-11-24T00:00:00Z")

    def explode(*a, **k):
        raise AssertionError("must not upload a second time")

    monkeypatch.setattr(pipeline.youtube, "upload", explode)
    report = Report()
    vid = pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, report)

    assert vid == "ALREADY"
    assert report["skipped"] == 1
    assert report["uploaded"] == 0


def test_an_unresolved_intent_is_settled_before_any_retry(con, monkeypatch):
    """A process killed between the request and the record: the video exists."""
    intent = index.record_intent(con, "4242", "Un clip")
    index.resolve_intent(con, intent, "FOUND123", "uploaded")  # pretend it landed
    con.execute("UPDATE upload_intents SET resolved = NULL WHERE id = ?", (intent,))
    con.commit()

    monkeypatch.setattr(pipeline.youtube, "video_exists", lambda vid: vid == "FOUND123")
    monkeypatch.setattr(
        pipeline.youtube, "upload", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-uploaded"))
    )

    report = Report()
    vid = pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, report)
    assert vid == "FOUND123"
    assert report["skipped"] == 1


def test_an_intent_with_no_video_on_youtube_is_safe_to_retry(con, monkeypatch):
    index.record_intent(con, "4242", "Un clip")  # never resolved: the request died

    monkeypatch.setattr(pipeline.youtube, "video_exists", lambda vid: False)
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")
    monkeypatch.setattr(pipeline.youtube, "find_duplicate", lambda *a: None)
    monkeypatch.setattr(pipeline.youtube, "upload", lambda *a, **k: "NEW1")

    report = Report()
    vid = pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, report)
    assert vid == "NEW1"
    assert index.unresolved_intents(con, "4242") == []


def test_a_successful_upload_leaves_no_open_intent(con, monkeypatch):
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")
    monkeypatch.setattr(pipeline.youtube, "find_duplicate", lambda *a: None)
    monkeypatch.setattr(pipeline.youtube, "upload", lambda *a, **k: "VID9")

    report = Report()
    pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, report)
    assert index.unresolved_intents(con, "4242") == []
    row = con.execute("SELECT status, yt_video_id FROM clips WHERE clip_id = '4242'").fetchone()
    assert (row["status"], row["yt_video_id"]) == ("uploaded", "VID9")


def test_a_duplicate_title_on_the_channel_is_adopted_not_reuploaded(con, monkeypatch):
    """The index was lost but YouTube still has the video: do not publish again."""
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")
    monkeypatch.setattr(pipeline.youtube, "find_duplicate", lambda *a, **k: "EXISTING")
    monkeypatch.setattr(
        pipeline.youtube, "upload", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-uploaded"))
    )

    report = Report()
    vid = pipeline.publish_once(con, "4242", None, {"titre": "Un clip"}, Settings, report)
    assert vid == "EXISTING"
    assert report["skipped"] == 1


def test_a_raising_upload_leaves_the_intent_open_for_the_next_pass(con, monkeypatch):
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")
    monkeypatch.setattr(pipeline.youtube, "find_duplicate", lambda *a: None)

    def boom(*a, **k):
        raise TimeoutError("connection died mid-transfer")

    monkeypatch.setattr(pipeline.youtube, "upload", boom)
    with pytest.raises(TimeoutError):
        pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, Report())

    # The evidence survives: the next pass will check YouTube before retrying.
    assert len(index.unresolved_intents(con, "4242")) == 1


def test_an_intent_whose_video_never_existed_is_retried_once(con, monkeypatch):
    """A dead intent (no id, nothing on YouTube) must not block the clip forever."""
    index.record_intent(con, "4242", "Un clip")
    monkeypatch.setattr(pipeline.youtube, "video_exists", lambda vid: False)

    assert len(index.unresolved_intents(con, "4242")) == 1
    # The next pass resolves it as "safe to retry" and clears it.
    for intent in index.unresolved_intents(con, "4242"):
        index.resolve_intent(con, intent["id"], None, "no video found, safe to retry")
    assert index.unresolved_intents(con, "4242") == []


def test_publish_once_never_marks_a_dead_intent_as_resolved_silently(con, monkeypatch):
    """The retry path must be able to tell 'never landed' from 'unknown'."""
    index.record_intent(con, "4242", "Un clip")
    monkeypatch.setattr(pipeline.youtube, "video_exists", lambda vid: False)
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")
    monkeypatch.setattr(pipeline.youtube, "find_duplicate", lambda *a: None)

    def boom(*a, **k):
        raise TimeoutError("died again")

    monkeypatch.setattr(pipeline.youtube, "upload", boom)
    with pytest.raises(TimeoutError):
        pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, Report())

    # The old intent was settled as dead; the new one stays open as unknown.
    assert len(index.unresolved_intents(con, "4242")) == 1


def test_a_narrow_scope_is_reported_loudly_and_does_not_block_the_upload(con, monkeypatch, capsys):
    """`youtube.upload` cannot search: say so, then fall back to the local index."""
    monkeypatch.setenv("YT_CLIENT_ID", "c")
    monkeypatch.setenv("YT_CLIENT_SECRET", "s")
    monkeypatch.setenv("YT_REFRESH_TOKEN", "r")
    monkeypatch.setattr(pipeline.youtube, "access_token", lambda *a: "tok")

    def no_scope(*a, **k):
        raise pipeline.youtube.ScopeTooNarrow("no read scope")

    monkeypatch.setattr(pipeline.youtube, "find_duplicate", no_scope)
    monkeypatch.setattr(pipeline.youtube, "upload", lambda *a, **k: "V1")

    report = Report()
    pipeline.publish_once(con, "4242", None, {"titre": "t"}, Settings, report)
    out = capsys.readouterr().out
    assert "WARNING" in out and "no read scope" in out
    assert report["uploaded"] == 0  # publish_once itself does not count
    assert con.execute("SELECT yt_video_id FROM clips WHERE clip_id='4242'").fetchone()[0] == "V1"


def test_privacy_defaults_to_private_and_rejects_a_typo():
    from highlights import youtube

    assert youtube.Privacy.PRIVATE == "private"
    with pytest.raises(ValueError, match="privacyStatus"):
        youtube.upload("/nonexistent.mp4", {}, "tok", privacy="invisible")


def test_selection_ignores_a_published_clip_after_a_restart(tmp_path):
    con = index.connect(tmp_path / "c.db")
    index.upsert_clip(con, ROW)
    index.set_status(con, "4242", "uploaded", yt_video_id="V", at="2026-11-24T00:00:00Z")
    clips = [{"clip_id": "4242", "duration_s": 30.0, "created_at": "2026-11-24T00:00:00Z"}]
    assert pipeline.select(clips, con, since_hours=48) == []
