"""Speech to text with faster-whisper (CPU, int8).

Run this in a sandbox: the Hermes pod is small and the model needs ~500 MB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import cleanup


def to_wav(src: str | Path, dst: str | Path) -> Path:
    """16 kHz mono PCM: the input every transcription engine expects."""
    import subprocess

    dst = Path(dst)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
        ],
        check=True,
    )
    return dst


def read_wav(path: str | Path):
    """Load a PCM s16le mono 16 kHz WAV into the float32 array Whisper wants."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as fh:
        if fh.getframerate() != 16000:
            raise ValueError(f"expected 16 kHz audio, got {fh.getframerate()} Hz: {path}")
        if fh.getnchannels() != 1:
            raise ValueError(f"expected mono audio, got {fh.getnchannels()} channels: {path}")
        if fh.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {fh.getsampwidth() * 8}-bit: {path}")
        frames = fh.readframes(fh.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def peak(audio) -> float:
    """Highest absolute sample, 0.0 for a silent track."""
    import numpy as np

    if not len(audio):
        return 0.0
    return float(np.abs(audio).max())


SILENT_PEAK = 0.001  # ~ -60 dBFS


class SilentAudio(RuntimeError):
    """The source has no usable audio. Permanent: retrying cannot help."""


def assert_audible(audio, source: str | Path = "") -> None:
    """Fail before transcribing: Whisper invents subtitles over a dead track.

    Twitch serves some old clips with a silent AAC stream (measured: mean and
    max volume at -91 dB). Whisper then outputs "Sous-titres réalisés par la
    communauté d'Amara.org" with a low no_speech_prob — a plausible-looking
    transcript of nothing. Catch it here, where the cause is unambiguous.
    """
    level = peak(audio)
    if level < SILENT_PEAK:
        raise SilentAudio(
            f"silent audio track (peak {level:.5f}): nothing to transcribe in {source}. "
            "Check the source file with `ffmpeg -i f -af volumedetect -f null -`."
        )


def transcribe(
    media: str | Path,
    model_size: str = "small",
    language: str | None = "fr",
    vocab: str | Path | None = None,
    corrections: str | Path | None = None,
    beam_size: int = 5,
) -> dict:
    """Transcribe a 16 kHz mono WAV.

    The audio is decoded here rather than handed to faster-whisper as a path:
    faster-whisper 1.2.x calls `av.open(metadata_errors=…)`, which PyAV 19
    removed, so its own decode path raises TypeError. numpy is already a
    dependency of ctranslate2, and we control the format (PCM s16le, 16 kHz,
    mono), so reading the frames is a few lines and one less version to pin.
    """
    from faster_whisper import WhisperModel

    terms = cleanup.read_vocabulary(vocab) if vocab else []
    rules = cleanup.read_corrections(corrections) if corrections else {}

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        download_root=os.environ.get("WHISPER_CACHE", str(Path.home() / "models")),
    )
    audio = read_wav(media)
    assert_audible(audio, media)
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=cleanup.initial_prompt(terms) or None,
        condition_on_previous_text=True,
    )

    kept = []
    for seg in segments:
        # Whisper invents text over silence; no_speech_prob is the signal.
        if getattr(seg, "no_speech_prob", 0.0) > 0.6:
            continue
        text = cleanup.clean(seg.text, rules)
        if not text:
            continue
        kept.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "mid": round((seg.start + seg.end) / 2, 2),
                "text": text,
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0)), 3),
            }
        )

    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "model": model_size,
        "segments": kept,
        "text": " ".join(s["text"] for s in kept).strip(),
    }


def write_result(result: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_json = out / "transcript.json"
    clean_txt = out / "transcript.txt"
    raw_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clean_txt.write_text(cleanup.paragraphs([s["text"] for s in result["segments"]]) + "\n", encoding="utf-8")
    return {"json": raw_json, "txt": clean_txt}
