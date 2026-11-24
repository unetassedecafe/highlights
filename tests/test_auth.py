"""Tests for the OAuth helper. No network: the exchange is exercised with a stub."""
from __future__ import annotations

import json
import urllib.parse

import pytest

from highlights import auth


def test_load_client_reads_the_installed_shape(tmp_path):
    path = tmp_path / "client_secret.json"
    path.write_text(
        json.dumps({"installed": {"client_id": "cid.apps.googleusercontent.com", "client_secret": "sekret"}}),
        encoding="utf-8",
    )
    assert auth.load_client(path) == ("cid.apps.googleusercontent.com", "sekret")


def test_load_client_rejects_a_file_without_credentials(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"installed": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="client_id"):
        auth.load_client(path)


def test_consent_url_asks_for_offline_access_and_forces_consent():
    url = auth.consent_url("cid", auth.new_verifier())
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    # access_type=offline is what yields a refresh token at all;
    # prompt=consent is what yields one on a second authorization.
    assert q["access_type"] == ["offline"]
    assert q["prompt"] == ["consent"]
    assert q["scope"] == [auth.SCOPE]
    assert q["code_challenge_method"] == ["S256"]


def test_pkce_challenge_is_the_s256_of_the_verifier():
    verifier = auth.new_verifier()
    expected = auth.challenge(verifier)
    import base64
    import hashlib

    assert expected == base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")


def test_verifiers_are_unique():
    assert len({auth.new_verifier() for _ in range(20)}) == 20


def test_extract_code_accepts_a_full_redirect_url():
    url = "http://localhost:1/?code=4/0AabcDEF&scope=https://www.googleapis.com/auth/youtube.upload"
    assert auth.extract_code(url) == "4/0AabcDEF"


def test_extract_code_accepts_a_bare_code():
    assert auth.extract_code("4/0AabcDEF") == "4/0AabcDEF"


def test_extract_code_rejects_an_empty_input():
    with pytest.raises(ValueError):
        auth.extract_code("http://localhost:1/?error=access_denied")


def test_extract_code_names_access_denied_instead_of_passing_it_as_a_code():
    with pytest.raises(ValueError, match="access_denied"):
        auth.extract_code("http://localhost:1/?error=access_denied&state=")


def test_extract_code_rejects_blank_input():
    with pytest.raises(ValueError):
        auth.extract_code("   ")


def test_save_refresh_token_is_private_and_never_printed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(auth, "PENDING", tmp_path / "oauth_pending.json")
    path = auth.save_refresh_token("1//a-refresh-token")
    assert path.read_text() == "1//a-refresh-token"
    assert oct(path.stat().st_mode)[-3:] == "600"
    # Nothing on stdout: the caller must read the file, not scrape the output.
    assert capsys.readouterr().out == ""


def test_pending_roundtrip_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "PENDING", tmp_path / "pending.json")
    auth.save_pending("ver", "cid", "sekret")
    assert auth.load_pending()["verifier"] == "ver"
    assert oct(auth.PENDING.stat().st_mode)[-3:] == "600"
    auth.clear_pending()
    with pytest.raises(RuntimeError, match="no pending"):
        auth.load_pending()


def test_exchange_reports_a_missing_refresh_token(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps({"access_token": "at"}).encode()

    monkeypatch.setattr(auth.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="no refresh_token"):
        auth.exchange("code", "cid", "sekret", "ver")


def test_exchange_returns_the_refresh_token(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps({"refresh_token": "1//refresh", "scope": auth.SCOPE}).encode()

    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["body"] = urllib.parse.parse_qs(req.data.decode())
        return FakeResp()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)
    data = auth.exchange("the-code", "cid", "sekret", "ver", "http://localhost")
    assert data["refresh_token"] == "1//refresh"
    # The verifier must actually be sent, or PKCE fails at Google.
    assert seen["body"]["code_verifier"] == ["ver"]
    # And the redirect_uri must be the same string as in the consent request.
    assert seen["body"]["redirect_uri"] == ["http://localhost"]


def test_consent_url_uses_the_registered_redirect_uri():
    url = auth.consent_url("cid", auth.new_verifier(), "http://localhost")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["redirect_uri"] == ["http://localhost"]


def test_redirect_uri_for_replaces_a_bare_localhost_with_a_failing_port(tmp_path):
    path = tmp_path / "client_secret.json"
    path.write_text(
        json.dumps({"installed": {"client_id": "c", "client_secret": "s",
                                  "redirect_uris": ["http://localhost"]}}),
        encoding="utf-8",
    )
    # `http://localhost` with no port sends the browser to port 80. If something
    # listens there, the tab loads and hangs with no code to copy — the exact
    # symptom this avoids. A Desktop client accepts any loopback port.
    assert auth.redirect_uri_for(path) == f"http://localhost:{auth.SAFE_PORT}/"


def test_redirect_uri_for_keeps_an_explicit_port(tmp_path):
    path = tmp_path / "client_secret.json"
    path.write_text(
        json.dumps({"installed": {"client_id": "c", "client_secret": "s",
                                  "redirect_uris": ["http://localhost:3000/cb"]}}),
        encoding="utf-8",
    )
    assert auth.redirect_uri_for(path) == "http://localhost:3000/cb"


def test_redirect_uri_for_falls_back_when_nothing_is_declared(tmp_path):
    path = tmp_path / "client_secret.json"
    path.write_text(
        json.dumps({"installed": {"client_id": "c", "client_secret": "s"}}),
        encoding="utf-8",
    )
    assert auth.redirect_uri_for(path) == auth.DEFAULT_REDIRECT_URI
