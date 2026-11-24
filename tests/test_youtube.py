"""Test the resumable upload against a fake YouTube API.

The protocol has three traps that a real upload would surface only at the worst
moment: the 308 "resume incomplete" answer is not a failure, Content-Range must
be exact, and a brand-new video is listed before it is processed. A mock server
catches all three without credentials.

Run with: python -m pytest tests/test_youtube.py -q
"""
from __future__ import annotations

import http.server
import json
import threading

import pytest

from highlights import youtube


class FakeYouTube(http.server.BaseHTTPRequestHandler):
    """Minimal resumable-upload server, with an intentional mid-upload 308."""

    protocol_version = "HTTP/1.1"
    received = b""
    session_calls = 0
    init_body: dict = {}
    fail_resume_once = True

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        assert self.path.startswith("/upload/youtube/v3/videos")
        assert self.headers["Authorization"] == "Bearer FAKE_TOKEN"
        assert self.headers["X-Upload-Content-Type"] == "video/mp4"
        length = int(self.headers["Content-Length"])
        FakeYouTube.init_body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/session")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self):
        length = int(self.headers["Content-Length"])
        chunk = self.rfile.read(length)
        total = int(self.headers["Content-Range"].split("/")[1])
        assert self.headers["Content-Length"] == str(len(chunk))

        # First PUT: pretend the connection dropped, answering 308 with a Range.
        if FakeYouTube.fail_resume_once:
            FakeYouTube.fail_resume_once = False
            FakeYouTube.received += chunk[: len(chunk) // 2]
            done = len(FakeYouTube.received) - 1
            self.send_response(308)
            self.send_header("Range", f"bytes=0-{done}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        FakeYouTube.received += chunk
        FakeYouTube.session_calls += 1
        if len(FakeYouTube.received) >= total:
            body = json.dumps({"id": "VID123"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            done = len(FakeYouTube.received) - 1
            self.send_response(308)
            self.send_header("Range", f"bytes=0-{done}")
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        assert self.path.startswith("/youtube/v3/videos")
        body = json.dumps(
            {"items": [{"status": {"uploadStatus": "processed"}, "processingDetails": {}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_api(monkeypatch):
    FakeYouTube.received = b""
    FakeYouTube.session_calls = 0
    FakeYouTube.fail_resume_once = True
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeYouTube)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(youtube, "UPLOAD", f"http://127.0.0.1:{port}/upload/youtube/v3/videos")
    monkeypatch.setattr(youtube, "API", f"http://127.0.0.1:{port}/youtube/v3")
    yield port
    server.shutdown()


def test_upload_survives_a_308_and_sends_every_byte(tmp_path, fake_api):
    video = tmp_path / "clip.mp4"
    payload = b"x" * (youtube.CHUNK + 1234)  # two chunks, so the loop really loops
    video.write_bytes(payload)
    meta = {
        "titre": "Un titre",
        "description": "Une description",
        "tags": ["Kubernetes", "Talos"],
    }

    vid = youtube.upload(video, meta, "FAKE_TOKEN", privacy="unlisted")

    assert vid == "VID123"
    assert FakeYouTube.received == payload


def test_upload_sets_privacy_explicitly_never_private(tmp_path, fake_api):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"y" * 2048)
    youtube.upload(video, {"titre": "t", "description": "d", "tags": []}, "FAKE_TOKEN", privacy="unlisted")

    status = FakeYouTube.init_body["status"]
    # Omitting privacyStatus means PRIVATE on the real API.
    assert status["privacyStatus"] == "unlisted"
    assert status["selfDeclaredMadeForKids"] is False


def test_upload_sends_the_french_language_and_category(tmp_path, fake_api):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"z" * 1024)
    youtube.upload(video, {"titre": "t", "description": "d", "tags": []}, "FAKE_TOKEN")

    snippet = FakeYouTube.init_body["snippet"]
    assert snippet["defaultLanguage"] == "fr"
    assert snippet["defaultAudioLanguage"] == "fr"
    assert snippet["categoryId"] == "28"  # Science & Technology


def test_wait_processed_returns_on_processed(fake_api):
    assert youtube.wait_processed("VID123", "FAKE_TOKEN", timeout_s=20) == "processed"


def test_credentials_from_env_names_the_missing_variable(monkeypatch):
    for name in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as exc:
        youtube.credentials_from_env()
    assert "YT_CLIENT_ID" in str(exc)
