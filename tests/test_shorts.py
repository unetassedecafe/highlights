"""Shorts selection tests: windowing, boundaries, captions, rendering."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from highlights import shorts


def segs(n: int, step: float = 3.0, dur: float = 2.8) -> list[dict]:
    return [
        {"start": i * step, "end": i * step + dur, "text": f"phrase numéro {i} avec assez de contenu pour compter"}
        for i in range(n)
    ]


def test_load_segments_reads_every_chunk_in_order(tmp_path):
    for i in range(3):
        (tmp_path / f"chunk{i:03d}.json").write_text(
            json.dumps({"segments": [{"start": i * 600 + 1, "end": i * 600 + 4, "text": "x"}]})
        )
    loaded = shorts.load_segments(tmp_path)
    assert len(loaded) == 3
    assert [s["start"] for s in loaded] == [1, 601, 1201]


def test_load_segments_explains_a_missing_transcription(tmp_path):
    with pytest.raises(FileNotFoundError, match="highlights vod"):
        shorts.load_segments(tmp_path)


def test_windows_never_exceed_the_length():
    for w in shorts.windows(segs(200), length=45):
        assert w["end"] - w["start"] <= 45.0 + 0.01


def test_windows_end_on_a_segment_boundary():
    """A short that ends mid-sentence is unusable, so the end is a real edge."""
    data = segs(200)
    ends = {round(s["end"], 2) for s in data}
    for w in shorts.windows(data, length=45):
        assert round(w["end"], 2) in ends


def test_windows_drop_passages_with_too_little_speech():
    thin = [{"start": i * 3.0, "end": i * 3.0 + 2.8, "text": "oui"} for i in range(50)]
    assert shorts.windows(thin, length=45) == []


def test_candidates_are_numbered_for_the_model():
    text = shorts.candidates_for_llm(shorts.windows(segs(60), length=45))
    assert text.startswith("[0] 00:00:00")
    assert "[1]" in text


def test_pick_requires_a_model():
    with pytest.raises(ValueError, match="llm"):
        shorts.pick(shorts.windows(segs(60)), count=2)


def test_pick_maps_indices_back_to_passages():
    pool = shorts.windows(segs(200), length=45)
    reply = json.dumps({"choix": [{"i": 0, "titre": "Premier", "pourquoi": "net"},
                                  {"i": 2, "titre": "Deuxième", "pourquoi": "complet"}]})
    chosen = shorts.pick(pool, count=2, llm=lambda p: reply)
    assert [c["titre"] for c in chosen] == ["Premier", "Deuxième"]
    assert chosen[0]["start"] == pool[0]["start"]
    assert chosen[1]["text"] == pool[2]["text"]


def test_pick_ignores_out_of_range_and_duplicate_indices():
    pool = shorts.windows(segs(200), length=45)
    reply = json.dumps({"choix": [{"i": 0, "titre": "a"}, {"i": 0, "titre": "bis"},
                                  {"i": 9999, "titre": "faux"}]})
    chosen = shorts.pick(pool, count=3, llm=lambda p: reply)
    assert len(chosen) == 1


def test_pick_survives_a_model_wrapping_json_in_prose():
    pool = shorts.windows(segs(200), length=45)
    noisy = 'Voici mon choix:\n```json\n{"choix": [{"i": 1, "titre": "t"}]}\n```\nVoilà.'
    chosen = shorts.pick(pool, count=1, llm=lambda p: noisy)
    assert len(chosen) == 1 and chosen[0]["titre"] == "t"


def test_captions_start_at_zero_relative_to_the_clip():
    clip_segments = [{"start": 100.0, "end": 103.0, "text": "bonjour"}]
    ass = shorts.render_captions(clip_segments, 100.0, 145.0)
    assert "0:00:00.00,0:00:03.00" in ass
    assert "bonjour" in ass


def test_captions_drop_segments_outside_the_clip():
    clip_segments = [
        {"start": 10.0, "end": 12.0, "text": "avant"},
        {"start": 100.0, "end": 103.0, "text": "dedans"},
        {"start": 200.0, "end": 202.0, "text": "après"},
    ]
    ass = shorts.render_captions(clip_segments, 100.0, 145.0)
    assert "dedans" in ass
    assert "avant" not in ass and "après" not in ass


def test_captions_wrap_to_two_lines_maximum():
    """Two lines per event is the rule; a longer sentence becomes two events."""
    long = ("ceci est une phrase vraiment tres longue qui ne tiendra jamais "
            "sur une seule ligne dans une video verticale")
    ass = shorts.render_captions([{"start": 0.0, "end": 3.0, "text": long}], 0.0, 45.0)
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue) >= 1
    for line in dialogue:
        assert line.count("\\N") <= 1, "no event may exceed two lines"
    # And no word may be lost when the event is split across time.
    joined = " ".join(l.split(",,", 1)[1] for l in dialogue).replace("\\N", " ")
    for w in long.split():
        assert w in joined, f"lost word: {w}"


def test_ass_header_declares_the_vertical_canvas():
    ass = shorts.render_captions([], 0.0, 45.0)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


def test_sentences_merge_whisper_fragments_into_readable_units():
    """Whisper emits a segment every 2 s; a window built on those starts mid-thought."""
    raw = [
        {"start": 0.0, "end": 2.0, "text": "Alors on a testé"},
        {"start": 2.0, "end": 4.0, "text": "deux approches du"},
        {"start": 4.0, "end": 6.0, "text": "problème hier soir."},
        {"start": 6.0, "end": 8.0, "text": "La seconde gagne"},
        {"start": 8.0, "end": 10.0, "text": "et voici pourquoi."},
    ]
    units = shorts.sentences(raw)
    assert len(units) == 2
    assert units[0]["text"] == "Alors on a testé deux approches du problème hier soir."
    assert units[0]["start"] == 0.0 and units[0]["end"] == 6.0
    assert units[1]["text"].endswith("pourquoi.")


def test_sentences_break_on_a_long_pause():
    raw = [
        {"start": 0.0, "end": 2.0, "text": "il parle encore"},
        {"start": 5.0, "end": 7.0, "text": "puis après un silence"},
    ]
    assert len(shorts.sentences(raw, gap=0.6)) == 2


def test_sentences_cap_a_runaway_unit():
    raw = [{"start": i * 2.0, "end": i * 2.0 + 2.0, "text": "parole continue"} for i in range(30)]
    units = shorts.sentences(raw, max_len=20.0)
    assert all(u["end"] - u["start"] <= 22.0 for u in units)
    assert len(units) >= 2


def test_candidates_include_the_lead_in_so_the_model_can_judge_context():
    """Without a lead-in, "Non, ça n'a rien à ça" looks self-contained."""
    data = segs(200)
    pool = shorts.windows(data, length=45)
    text = shorts.candidates_for_llm(pool, segments=data, context=2)
    assert "contexte avant" in text
    assert "NE PAS inclure" in text


def test_candidates_without_segments_stay_bare():
    pool = shorts.windows(segs(60), length=45)
    text = shorts.candidates_for_llm(pool)
    assert "contexte avant" not in text
    assert text.startswith("[0]")


def test_pick_forwards_the_transcript_for_context():
    data = segs(200)
    pool = shorts.windows(data, length=45)
    prompts = []

    def llm(p):
        prompts.append(p)
        return json.dumps({"choix": [{"i": 1, "titre": "t"}]})

    shorts.pick(pool, count=1, llm=llm, segments=data)
    assert "contexte avant" in prompts[0]


def test_a_clip_ending_on_a_dangling_connector_is_trimmed():
    """A short ending on "je ne t'ai pas dit que…" never resolves."""
    clip = {
        "start": 0.0, "end": 42.0,
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "Voici le problème posé par le cache."},
            {"start": 10.0, "end": 30.0, "text": "La solution est de changer la clé."},
            {"start": 30.0, "end": 42.0, "text": "je ne t'ai pas dit que"},
        ],
    }
    out = shorts.trim_to_sentence(clip)
    assert out["end"] == 30.0
    assert "que" not in out["text"].split()[-1:]


def test_trimming_never_below_the_minimum_duration():
    """If removing the dangling segment would drop under the floor, keep it."""
    clip = {
        "start": 0.0, "end": 16.0,
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "Une idée complète et finie ici."},
            {"start": 10.0, "end": 16.0, "text": "et"},
        ],
    }
    out = shorts.trim_to_sentence(clip, min_s=15.0)
    assert len(out["segments"]) == 2, "10 s is below the 15 s floor, so nothing is trimmed"


def test_trimming_happens_when_the_result_stays_above_the_floor():
    clip = {
        "start": 0.0, "end": 40.0,
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "Une idée complète et finie ici."},
            {"start": 30.0, "end": 40.0, "text": "et"},
        ],
    }
    out = shorts.trim_to_sentence(clip, min_s=15.0)
    assert len(out["segments"]) == 1
    assert out["end"] == 30.0


def test_a_complete_clip_is_left_untouched():
    clip = {
        "start": 0.0, "end": 40.0,
        "segments": [
            {"start": 0.0, "end": 20.0, "text": "On a testé deux approches."},
            {"start": 20.0, "end": 40.0, "text": "La seconde gagne, et voici pourquoi."},
        ],
    }
    out = shorts.trim_to_sentence(clip)
    assert out["end"] == 40.0
    assert len(out["segments"]) == 2


def test_a_passage_flagged_not_standalone_is_dropped():
    pool = shorts.windows(segs(200), length=45)
    reply = json.dumps({"choix": [
        {"i": 0, "titre": "ok", "compréhensible_seul": True},
        {"i": 1, "titre": "privée", "compréhensible_seul": False},
    ]})
    chosen = shorts.pick(pool, count=2, llm=lambda p: reply)
    assert [c["titre"] for c in chosen] == ["ok"]
    assert chosen[0]["standalone"] is True


def test_render_converts_absolute_timings_to_file_offsets(tmp_path, monkeypatch):
    """A clip at 3673 s in a 180 s file must seek to 13 s, not 3673 s."""
    seen = {}

    def fake_run(cmd, check=False, **kw):
        seen["cmd"] = cmd
        out = Path(cmd[-1])
        out.write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(shorts.subprocess, "run", fake_run)
    clip = {"start": 3673.0, "end": 3715.0, "file_start": 3660.0, "text": "x"}
    shorts.render("/tmp/in.mp4", clip, tmp_path / "out.mp4", size=(1280, 720))

    cmd = seen["cmd"]
    assert cmd[cmd.index("-ss") + 1] == "13.000"
    assert cmd[cmd.index("-t") + 1] == "42.000"


def test_a_clip_without_file_start_is_used_verbatim(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, check=False, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(shorts.subprocess, "run", fake_run)
    shorts.render("/tmp/in.mp4", {"start": 10.0, "end": 40.0, "text": "x"},
                  tmp_path / "out.mp4", size=(1920, 1080))
    assert seen["cmd"][seen["cmd"].index("-ss") + 1] == "10.000"


def test_render_refuses_a_silently_empty_output(tmp_path, monkeypatch):
    def fake_run(cmd, check=False, **kw):
        Path(cmd[-1]).write_bytes(b"tiny")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(shorts.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="render produced"):
        shorts.render("/tmp/in.mp4", {"start": 0.0, "end": 30.0, "text": "x"},
                      tmp_path / "out.mp4", size=(1920, 1080))


def test_uncropped_render_fits_the_frame_instead_of_upscaling(tmp_path, monkeypatch):
    """`scale=-2:1920` on a 720p source produced 3414x1920 (measured). Fit it."""
    seen = {}

    def fake_run(cmd, check=False, **kw):
        seen["vf"] = cmd[cmd.index("-vf") + 1]
        Path(cmd[-1]).write_bytes(b"x" * 60_000)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(shorts.subprocess, "run", fake_run)
    shorts.render("/tmp/in.mp4", {"start": 0.0, "end": 30.0, "text": "x"},
                  tmp_path / "out.mp4", crop=False, size=(1280, 720))
    assert "scale=1080:-2" in seen["vf"]
    assert "pad=1080:1920" in seen["vf"]
    assert "1920" not in seen["vf"].split("scale=")[1].split(":")[0]


def test_crop_geometry_is_exactly_9_16_for_a_1080p_source():
    w, h, x, y = shorts.crop_parts(1920, 1080)
    assert (w, h) == (608, 1080)
    assert abs(w / h - 9 / 16) < 0.001
    assert (x, y) == (656, 0)


def test_crop_geometry_scales_down_to_a_720p_source():
    """Twitch replays are 720p: a hardcoded 608x1080 crop fails outright."""
    w, h, x, y = shorts.crop_parts(1280, 720)
    assert (w, h) == (404, 720)
    assert abs(w / h - 9 / 16) < 0.01
    assert h <= 720, "never crop taller than the source"
    assert (x, y) == (438, 0)


def test_crop_geometry_stays_even_for_h264():
    for src in ((1920, 1080), (1280, 720), (854, 480), (1920, 1082)):
        w, h, _, _ = shorts.crop_parts(*src)
        assert w % 2 == 0 and h % 2 == 0, src
