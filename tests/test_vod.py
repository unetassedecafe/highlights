"""VOD path tests: chunking, timestamp shifting, long chapters, resume."""
from __future__ import annotations

import json

from highlights import metadata, vod, vodrun


def test_chunks_cover_the_whole_vod():
    windows = vod.chunks(8940, 600)  # 2h29
    assert windows[0] == (0.0, 600.0)
    covered = sum(length for _, length in windows)
    assert covered == 8940


def test_a_trailing_sliver_is_merged_into_the_previous_chunk():
    # 605 s with 600 s chunks: a 5 s tail is not worth a whole Whisper pass.
    windows = vod.chunks(605, 600)
    assert len(windows) == 1
    assert windows[0] == (0.0, 605)


def test_shift_segments_makes_timings_absolute():
    """A chunk starting at 10 min must not report 00:30 as 00:30."""
    segs = [{"start": 30.0, "end": 35.0, "mid": 32.5, "text": "bonjour"}]
    moved = vod.shift_segments(segs, 600.0)
    assert moved[0]["start"] == 630.0
    assert moved[0]["mid"] == 632.5
    assert moved[0]["text"] == "bonjour"


def test_shift_segments_does_not_mutate_the_input():
    segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
    vod.shift_segments(segs, 100)
    assert segs[0]["start"] == 1.0


def test_long_timestamp_goes_to_hours_only_past_an_hour():
    assert vod.long_timestamp(65) == "01:05"
    assert vod.long_timestamp(3665) == "01:01:05"
    assert vod.long_timestamp(0) == "00:00"


def test_long_chapters_start_at_zero_and_are_spaced():
    segs = [{"start": i * 30.0, "text": f"on parle du sujet numéro {i} assez longuement"} for i in range(200)]
    chapters = vod.build_long_chapters(segs, duration_s=6000)
    assert chapters[0]["t"] == "00:00"
    assert 3 <= len(chapters) <= 25
    # No chapter may be closer than two minutes to the previous one.
    for prev, cur in zip(chapters, chapters[1:]):
        assert vod._to_seconds(cur["t"]) - vod._to_seconds(prev["t"]) >= 120


def test_a_short_vod_gets_no_chapters():
    segs = [{"start": 1.0, "text": "court"}]
    assert vod.build_long_chapters(segs, duration_s=60) == []


def test_chapter_titles_come_from_speech_not_from_an_index():
    segs = [{"start": i * 60.0, "text": f"segment numéro {i} sur Talos et le cluster"} for i in range(30)]
    chapters = vod.build_long_chapters(segs, duration_s=1800)
    assert len(chapters) >= 3
    titles = " ".join(c["titre"] for c in chapters)
    assert "Talos" in titles
    assert "Partie" not in titles
    assert "Chapitre" not in titles


def test_render_chapters_is_one_line_per_chapter():
    text = vod.render_chapters([{"t": "00:00", "titre": "Intro"}, {"t": "01:00", "titre": "Suite"}])
    assert text.splitlines() == ["00:00 Intro", "01:00 Suite"]


def test_metadata_validate_accepts_long_form_chapters():
    meta = metadata.Metadata(
        title="Une rediff", description="d",
        chapters=[{"t": "00:00", "titre": "a"}, {"t": "10:00", "titre": "b"}, {"t": "20:00", "titre": "c"}],
    )
    assert metadata.validate(meta) == []


def test_a_chunk_is_reused_instead_of_retranscribed(tmp_path, monkeypatch):
    """Resume: an existing chunkNNN.json must not trigger ffmpeg or Whisper."""
    work = tmp_path / "123"  # process_vod uses workdir/<vod_id>
    work.mkdir(parents=True)

    monkeypatch.setattr(vodrun.vod, "playback_url", lambda vid: "https://master")
    monkeypatch.setattr(vodrun.vod, "twitch_vod_info", lambda vid: {"title": "T", "lengthSeconds": 1200})
    monkeypatch.setattr(vodrun.vod, "chunks", lambda d, s: [(0.0, 600.0), (600.0, 600.0)])

    # Pretend chunk 0 already ran.
    (work / "chunk000.json").write_text(json.dumps({
        "offset": 0.0,
        "segments": [{"start": 10.0, "end": 12.0, "mid": 11.0, "text": "déjà transcrit"}],
    }), encoding="utf-8")

    calls = {"extract": 0}

    def fake_extract(*a, **k):
        calls["extract"] += 1
        return work / "chunk001.wav"

    monkeypatch.setattr(vodrun.vod, "extract_audio", fake_extract)
    monkeypatch.setattr(vodrun.transcribe, "transcribe",
                        lambda *a, **k: {"segments": [{"start": 5.0, "end": 6.0, "mid": 5.5, "text": "suite"}],
                                         "text": "suite", "duration": 600.0})

    payload = vodrun.process_vod("123", tmp_path, progress=lambda *_: None)
    assert calls["extract"] == 1  # only the missing chunk
    assert "déjà transcrit" in payload["transcript"]
    assert "suite" in payload["transcript"]
