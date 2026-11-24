"""Tests for the parts that do not need a model or the network."""
from __future__ import annotations

import json

from highlights import cleanup, index, metadata, transcribe


def test_vocabulary_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "v.md"
    path.write_text("# commentaire\nTalos\n\n  Kubernetes  # inline\n", encoding="utf-8")
    assert cleanup.read_vocabulary(path) == ["Talos", "Kubernetes"]


def test_initial_prompt_mentions_the_terms(tmp_path):
    prompt = cleanup.initial_prompt(["Talos", "KubeVirt"])
    assert "Talos" in prompt and "KubeVirt" in prompt
    assert cleanup.initial_prompt([]) == ""


def test_corrections_are_whole_words_and_case_insensitive(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("talos => Talos\n# rien\n", encoding="utf-8")
    rules = cleanup.read_corrections(path)
    assert cleanup.apply_corrections("on parle de talos et de talosien", rules) == (
        "on parle de Talos et de talosien"
    )


def test_corrections_prefer_the_longest_match():
    rules = {"argo cd": "ArgoCD", "argo": "Argo"}
    assert cleanup.apply_corrections("argo cd puis argo", rules) == "ArgoCD puis Argo"


def test_fillers_are_dropped():
    cleaned = cleanup.clean("alors euh on va voir, en fait, Cilium")
    assert "euh" not in cleaned
    assert "Cilium" in cleaned


def test_timestamp_formats_minutes():
    assert cleanup.timestamp(0) == "00:00"
    assert cleanup.timestamp(75.9) == "01:15"


def test_fit_tags_respects_the_total_budget():
    tags = metadata.fit_tags([f"tag{i}" for i in range(100)], total_limit=50)
    assert len(",".join(tags)) <= 50
    assert tags


def test_fit_tags_drops_overlong_ones():
    assert metadata.fit_tags(["court", "x" * 40]) == ["court"]


def test_validate_flags_a_title_that_youtube_would_truncate():
    meta = metadata.Metadata(title="a" * 80, description="ok")
    problems = metadata.validate(meta)
    assert any("tronqué" in p for p in problems)


def test_validate_flags_a_missing_chapter_zero():
    meta = metadata.Metadata(
        title="ok",
        description="ok",
        chapters=[{"t": "00:30", "titre": "a"}, {"t": "01:00", "titre": "b"}, {"t": "02:00", "titre": "c"}],
    )
    assert any("00:00" in p for p in metadata.validate(meta))


def test_short_clip_gets_no_chapters():
    segments = [{"start": 0, "text": "bonjour"}, {"start": 10, "text": "voila"}, {"start": 20, "text": "fin"}]
    assert metadata.build_chapters(segments, duration=41) == []


def test_long_vod_gets_chapters_starting_at_zero():
    segments = [{"start": i * 30, "text": f"phrase {i}"} for i in range(20)]
    chapters = metadata.build_chapters(segments, duration=600)
    assert len(chapters) >= 3
    assert chapters[0]["t"] == "00:00"


def test_template_metadata_stays_within_limits():
    meta = metadata.draft_templates("Un titre très long " * 10, episode="7")
    assert len(meta.title) <= metadata.TITLE_UI
    assert metadata.validate(meta) == []


def test_index_dedupes_on_clip_id(tmp_path):
    con = index.connect(tmp_path / "c.db")
    row = {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 3, "duration_s": 30.0, "created_at": "2026-01-01T00:00:00Z",
        "status": "listed", "updated_at": "2026-01-01T00:00:00Z",
    }
    assert index.upsert_clip(con, row) is True
    assert index.upsert_clip(con, row) is False
    assert con.execute("SELECT count(*) FROM clips").fetchone()[0] == 1


def test_full_text_search_finds_the_episode(tmp_path):
    con = index.connect(tmp_path / "c.db")
    row = {
        "clip_id": "42", "slug": "s", "title": "Kamaji", "creator": "c",
        "twitch_url": "u", "twitch_views": 1, "duration_s": 30.0,
        "created_at": "2026-01-01T00:00:00Z", "status": "uploaded",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    index.upsert_clip(con, row)
    index.index_transcript(con, "42", "Kamaji", "on a installé Kamaji pour gérer les control planes", "Kamaji")
    hits = index.search(con, "control planes")
    assert len(hits) == 1 and hits[0]["clip_id"] == "42"
    assert index.search(con, "Cilium") == []


def test_search_survives_punctuation(tmp_path):
    con = index.connect(tmp_path / "c.db")
    assert index.search(con, "kubectl (get)") == []


def test_products_accumulate_mentions(tmp_path):
    con = index.connect(tmp_path / "c.db")
    index.remember_products(con, ["Talos", "Ceph"], "2026-01-01T00:00:00Z")
    index.remember_products(con, ["Talos"], "2026-01-02T00:00:00Z")
    rows = {r["name"]: r["mentions"] for r in con.execute("SELECT name, mentions FROM products")}
    assert rows == {"Talos": 2, "Ceph": 1}


def test_select_retries_a_clip_that_never_published(tmp_path):
    from datetime import datetime, timezone

    from highlights import pipeline

    con = index.connect(tmp_path / "c.db")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    row = {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 1, "duration_s": 30.0, "created_at": now,
        "status": "ready", "updated_at": now,
    }
    index.upsert_clip(con, row)
    clips = [{"clip_id": "1", "duration_s": 30.0, "created_at": now}]
    assert len(pipeline.select(clips, con, since_hours=24)) == 1


def test_select_skips_a_published_clip(tmp_path):
    from datetime import datetime, timezone

    from highlights import pipeline

    con = index.connect(tmp_path / "c.db")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index.upsert_clip(con, {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 1, "duration_s": 30.0, "created_at": now,
        "status": "uploaded", "updated_at": now,
    })
    clips = [{"clip_id": "1", "duration_s": 30.0, "created_at": now}]
    assert pipeline.select(clips, con, since_hours=24) == []


def test_select_gives_up_after_the_attempt_budget(tmp_path):
    from datetime import datetime, timezone

    from highlights import pipeline

    con = index.connect(tmp_path / "c.db")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index.upsert_clip(con, {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 1, "duration_s": 30.0, "created_at": now,
        "status": "failed", "updated_at": now,
    })
    for _ in range(3):
        index.set_status(con, "1", "failed", error="boom", at=now, attempt=True)
    clips = [{"clip_id": "1", "duration_s": 30.0, "created_at": now}]
    assert pipeline.select(clips, con, since_hours=24) == []


def test_select_respects_the_duration_window(tmp_path):
    from datetime import datetime, timezone

    from highlights import pipeline

    con = index.connect(tmp_path / "c.db")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    clips = [
        {"clip_id": "short", "duration_s": 3.0, "created_at": now},
        {"clip_id": "ok", "duration_s": 40.0, "created_at": now},
        {"clip_id": "long", "duration_s": 300.0, "created_at": now},
    ]
    picked = pipeline.select(clips, con, since_hours=24)
    assert [c["clip_id"] for c in picked] == ["ok"]


def test_select_does_not_retry_a_clip_that_has_no_audio(tmp_path):
    from datetime import datetime, timezone

    from highlights import pipeline

    con = index.connect(tmp_path / "c.db")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index.upsert_clip(con, {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 1, "duration_s": 30.0, "created_at": now,
        "status": "skipped", "updated_at": now,
    })
    clips = [{"clip_id": "1", "duration_s": 30.0, "created_at": now}]
    assert pipeline.select(clips, con, since_hours=24) == []


def test_attempts_only_count_real_failures(tmp_path):
    con = index.connect(tmp_path / "c.db")
    now = "2026-01-01T00:00:00Z"
    index.upsert_clip(con, {
        "clip_id": "1", "slug": "s", "title": "t", "creator": "c", "twitch_url": "u",
        "twitch_views": 1, "duration_s": 30.0, "created_at": now,
        "status": "listed", "updated_at": now,
    })
    index.set_status(con, "1", "downloaded", at=now)
    index.set_status(con, "1", "ready", at=now)
    assert con.execute("SELECT attempts FROM clips WHERE clip_id='1'").fetchone()[0] == 0
    index.set_status(con, "1", "failed", error="boom", at=now, attempt=True)
    assert con.execute("SELECT attempts FROM clips WHERE clip_id='1'").fetchone()[0] == 1


def test_metadata_json_schema_has_the_expected_keys():
    meta = metadata.draft_templates("Titre", "3")
    payload = json.loads(json.dumps(meta.as_dict()))
    assert {"titre", "description", "tags", "chapitres", "produits_cites", "resume_une_phrase"} <= set(payload)


def test_wav_roundtrip_rejects_a_non_16k_file(tmp_path):
    import wave

    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(48000)
        fh.writeframes(b"\x00\x00" * 4800)
    try:
        transcribe.read_wav(path)
    except ValueError as exc:
        assert "16 kHz" in str(exc)
    else:
        raise AssertionError("a 48 kHz file must be refused")


def test_a_silent_track_is_caught_before_transcribing(tmp_path):
    import wave

    import numpy as np

    path = tmp_path / "silent.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    audio = transcribe.read_wav(path)
    assert transcribe.peak(audio) == 0.0
    try:
        transcribe.assert_audible(audio, path)
    except transcribe.SilentAudio as exc:
        assert "silent" in str(exc)
    else:
        raise AssertionError("a dead audio track must not reach Whisper")


def test_speech_level_audio_passes_the_silence_check(tmp_path):
    import wave

    import numpy as np

    path = tmp_path / "loud.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes((np.sin(np.linspace(0, 500, 16000)) * 8000).astype(np.int16).tobytes())
    transcribe.assert_audible(transcribe.read_wav(path), path)
