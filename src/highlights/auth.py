"""One-time OAuth: get a refresh token for `youtube.upload`.

This pod is headless, so the loopback flow cannot listen on localhost and be
reached by your browser. The flow is therefore split in two commands:

    highlights auth --client-secret ~/Downloads/client_secret.json
    highlights auth --code 'http://localhost:1/?code=4/0A...'

The first prints a consent URL. You open it, approve, your browser fails on
`http://localhost:1` (expected), and you paste the whole redirected URL back.

PKCE (S256) is used even though the client is confidential: it costs nothing and
protects the code against a redirect interception in a pasted-URL flow.

A refresh token from a client still in "Testing" expires after 7 days. Publish
the OAuth app to Production in Google Cloud Console before you rely on this in a
cron, or the pipeline stops working silently a week later.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import urllib.parse
import urllib.request
from pathlib import Path

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"
# Loopback redirect. Google validates this AFTER the user signs in, so a mismatch
# only shows up at the last step — which is why the value comes from the client
# file itself (`redirect_uris`) rather than being guessed here. Nothing listens
# on it: the user copies the URL out of the address bar.
DEFAULT_REDIRECT_URI = "http://localhost:8765/"
# A port nothing listens on, so the browser fails immediately instead of hanging.
SAFE_PORT = 8765


def load_client(path: str | Path) -> tuple[str, str]:
    """Read a Google client_secret.json. Returns (client_id, client_secret)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data.get("installed") or data.get("web") or {}
    client_id = block.get("client_id")
    client_secret = block.get("client_secret")
    if not client_id or not client_secret:
        raise ValueError(
            f"{path} has no client_id/client_secret under 'installed' or 'web'. "
            "Download it from Cloud Console → Credentials → OAuth 2.0 Client ID (Desktop app)."
        )
    return client_id, client_secret


def redirect_uri_for(path: str | Path) -> str:
    """Pick a loopback redirect that fails FAST, not one that hangs.

    A client registered with `http://localhost` (no port) sends the browser to
    port 80 after approval. If anything listens on port 80 and does not speak
    OAuth, the tab loads and then hangs forever — no error, no code, nothing to
    copy. Desktop clients accept any loopback port at the token endpoint
    (measured: every candidate answered `invalid_grant`, never
    `redirect_uri_mismatch`), so an explicit high port is both allowed and
    better: nothing listens there, the browser says "connection refused"
    immediately, and the URL bar holds the code.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_REDIRECT_URI
    block = data.get("installed") or data.get("web") or {}
    for uri in block.get("redirect_uris") or []:
        if not (uri.startswith("http://localhost") or uri.startswith("http://127.0.0.1")):
            continue
        parsed = urllib.parse.urlparse(uri)
        if parsed.port:
            return uri              # explicit port: trust it
        return f"{parsed.scheme}://{parsed.hostname}:{SAFE_PORT}/"
    return DEFAULT_REDIRECT_URI


def new_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")


def challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def consent_url(client_id: str, verifier: str, redirect_uri: str = DEFAULT_REDIRECT_URI) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",     # without this there is no refresh token
        "prompt": "consent",          # force it, or a re-auth returns none
        "code_challenge": challenge(verifier),
        "code_challenge_method": "S256",
    }
    return AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params)


def extract_code(pasted: str) -> str:
    """Accept the whole redirected URL or the bare code.

    An `error=access_denied` redirect must be reported as such, not sent to the
    token endpoint as if it were a code.
    """
    pasted = pasted.strip()
    if not pasted:
        raise ValueError("no authorization code in that input")
    if "error=" in pasted or pasted.startswith("http"):
        query = urllib.parse.urlparse(pasted).query or pasted.split("?", 1)[-1]
        params = urllib.parse.parse_qs(query)
        error = params.get("error", [""])[0]
        if error:
            raise ValueError(
                f"Google refused the authorization: {error}. "
                "If it is access_denied, add your account as a test user at "
                "https://console.cloud.google.com/auth/audience"
            )
        code = params.get("code", [""])[0]
    else:
        code = pasted
    if not code:
        raise ValueError("no authorization code in that input")
    return code


def exchange(code: str, client_id: str, client_secret: str, verifier: str,
             redirect_uri: str = DEFAULT_REDIRECT_URI) -> dict:
    """Trade the code for a refresh token. `redirect_uri` must match the request."""
    payload = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    if "refresh_token" not in data:
        raise RuntimeError(
            "Google returned no refresh_token. The usual causes: the consent was "
            "already granted for this client (revoke at "
            "https://myaccount.google.com/permissions and retry), or `access_type=offline` "
            f"was dropped. Response: {data}"
        )
    return data


PENDING = Path(os.environ.get("HIGHLIGHTS_HOME", Path.home() / ".config" / "highlights")) / "oauth_pending.json"


def save_pending(verifier: str, client_id: str, client_secret: str,
                 redirect_uri: str = DEFAULT_REDIRECT_URI) -> None:
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(
        json.dumps(
            {
                "verifier": verifier,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            }
        ),
        encoding="utf-8",
    )
    PENDING.chmod(0o600)


def load_pending() -> dict:
    if not PENDING.exists():
        raise RuntimeError(f"no pending authorization at {PENDING}; run `highlights auth --client-secret …` first")
    return json.loads(PENDING.read_text(encoding="utf-8"))


def clear_pending() -> None:
    if PENDING.exists():
        PENDING.unlink()


def save_refresh_token(token: str) -> Path:
    """Persist the refresh token next to the client secret, mode 600.

    Never printed: a value echoed to a terminal gets truncated or redacted, and
    one code exchange yields exactly one refresh token — losing it means asking
    the user to consent again.
    """
    path = PENDING.parent / "refresh_token.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token.strip(), encoding="utf-8")
    path.chmod(0o600)
    return path
