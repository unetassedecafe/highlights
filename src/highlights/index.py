"""Local index: SQLite + FTS5, plus the vocabulary that improves over time."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
  clip_id      TEXT PRIMARY KEY,
  slug         TEXT,
  title        TEXT NOT NULL,
  creator      TEXT,
  twitch_url   TEXT,
  twitch_views INTEGER,
  duration_s   REAL,
  created_at   TEXT,
  yt_video_id  TEXT,
  status       TEXT NOT NULL DEFAULT 'pending',
  attempts     INTEGER DEFAULT 0,
  last_error   TEXT,
  updated_at   TEXT NOT NULL,
  claimed_at   TEXT,
  claimed_by   TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts USING fts5(
  clip_id UNINDEXED,
  titre,
  transcript,
  resume,
  tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS products (
  name     TEXT PRIMARY KEY,
  first_seen TEXT NOT NULL,
  mentions INTEGER DEFAULT 1
);

-- One row per publication attempt that may have reached YouTube. Written BEFORE
-- the insert request, so a crash mid-upload still leaves the intent behind and
-- the next pass can check YouTube instead of blindly retrying.
CREATE TABLE IF NOT EXISTS upload_intents (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_id    TEXT NOT NULL,
  title      TEXT NOT NULL,
  started_at TEXT NOT NULL,
  video_id   TEXT,
  resolved   TEXT
);
"""

MIGRATIONS = {
    "claimed_at": "ALTER TABLE clips ADD COLUMN claimed_at TEXT",
    "claimed_by": "ALTER TABLE clips ADD COLUMN claimed_by TEXT",
}


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    existing = {r["name"] for r in con.execute("PRAGMA table_info(clips)")}
    for column, ddl in MIGRATIONS.items():
        if column not in existing:
            con.execute(ddl)
    con.commit()
    return con


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def claim(con: sqlite3.Connection, clip_id: str, owner: str, lease_minutes: int = 30) -> bool:
    """Take the exclusive right to process a clip. Returns False if someone holds it.

    Two cron passes overlapping is the ordinary way a clip gets published twice:
    each lists the clip, neither sees `uploaded` yet, both upload. The lease is
    time-bounded so a crashed worker does not block the clip forever.
    """
    stamp = now()
    stale = datetime.now(timezone.utc).timestamp()
    row = con.execute(
        "SELECT claimed_at FROM clips WHERE clip_id = ?", (clip_id,)
    ).fetchone()
    if row and row["claimed_at"]:
        try:
            held = datetime.fromisoformat(row["claimed_at"].replace("Z", "+00:00")).timestamp()
        except ValueError:
            held = 0
        if stale - held < lease_minutes * 60:
            return False
    cur = con.execute(
        "UPDATE clips SET claimed_at = ?, claimed_by = ? "
        "WHERE clip_id = ? AND (claimed_at IS NULL OR claimed_at = ?)",
        (stamp, owner, clip_id, row["claimed_at"] if row else None),
    )
    con.commit()
    return cur.rowcount == 1


def release(con: sqlite3.Connection, clip_id: str) -> None:
    con.execute("UPDATE clips SET claimed_at = NULL, claimed_by = NULL WHERE clip_id = ?", (clip_id,))
    con.commit()


def upsert_clip(con: sqlite3.Connection, row: dict) -> bool:
    """Insert a clip row. Returns False when the clip_id was already known."""
    cur = con.execute(
        """INSERT INTO clips (clip_id, slug, title, creator, twitch_url, twitch_views,
                              duration_s, created_at, status, updated_at)
           VALUES (:clip_id, :slug, :title, :creator, :twitch_url, :twitch_views,
                   :duration_s, :created_at, :status, :updated_at)
           ON CONFLICT(clip_id) DO NOTHING""",
        row,
    )
    con.commit()
    return cur.rowcount == 1


def set_status(con: sqlite3.Connection, clip_id: str, status: str, yt_video_id: str | None = None,
               error: str | None = None, at: str = "", attempt: bool = False) -> None:
    """Record a status change.

    `attempt` counts a real try, so only failures pass it. Incrementing on every
    transition made a clip burn its retry budget just by moving
    listed -> downloaded -> ready, and then get abandoned unpublished.
    """
    con.execute(
        """UPDATE clips SET status = ?, yt_video_id = COALESCE(?, yt_video_id),
                            last_error = ?, attempts = attempts + ?, updated_at = ?
           WHERE clip_id = ?""",
        (status, yt_video_id, error, 1 if attempt else 0, at, clip_id),
    )
    con.commit()


def index_transcript(con: sqlite3.Connection, clip_id: str, title: str, transcript: str,
                     summary: str = "") -> None:
    con.execute("DELETE FROM clips_fts WHERE clip_id = ?", (clip_id,))
    con.execute(
        "INSERT INTO clips_fts (clip_id, titre, transcript, resume) VALUES (?, ?, ?, ?)",
        (clip_id, title, transcript, summary),
    )
    con.commit()


def search(con: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """FTS5 MATCH. Quote each word so punctuation in a query cannot break the syntax."""
    terms = " ".join(f'"{w}"' for w in query.split() if w)
    if not terms:
        return []
    rows = con.execute(
        """SELECT c.clip_id, c.title, c.yt_video_id, c.twitch_url, c.created_at,
                  snippet(clips_fts, 2, '[', ']', '…', 12) AS extrait,
                  bm25(clips_fts) AS score
           FROM clips_fts JOIN clips c ON c.clip_id = clips_fts.clip_id
           WHERE clips_fts MATCH ?
           ORDER BY score LIMIT ?""",
        (terms, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def remember_products(con: sqlite3.Connection, names: list[str], at: str) -> None:
    for name in names:
        con.execute(
            """INSERT INTO products (name, first_seen) VALUES (?, ?)
               ON CONFLICT(name) DO UPDATE SET mentions = mentions + 1""",
            (name, at),
        )
    con.commit()


def record_intent(con: sqlite3.Connection, clip_id: str, title: str) -> int:
    """Declare an imminent upload. Call this BEFORE the insert request."""
    cur = con.execute(
        "INSERT INTO upload_intents (clip_id, title, started_at) VALUES (?, ?, ?)",
        (clip_id, title, now()),
    )
    con.commit()
    if cur.lastrowid is None:
        raise RuntimeError("could not record the upload intent")
    return int(cur.lastrowid)


def resolve_intent(con: sqlite3.Connection, intent_id: int, video_id: str | None,
                   resolved: str) -> None:
    con.execute(
        "UPDATE upload_intents SET video_id = ?, resolved = ? WHERE id = ?",
        (video_id, resolved, intent_id),
    )
    con.commit()


def unresolved_intents(con: sqlite3.Connection, clip_id: str) -> list[dict]:
    """Uploads that were started and never came back. Each may exist on YouTube."""
    return [
        dict(r)
        for r in con.execute(
            "SELECT * FROM upload_intents WHERE clip_id = ? AND resolved IS NULL",
            (clip_id,),
        )
    ]


def known_video_ids(con: sqlite3.Connection) -> set[str]:
    """Every YouTube id this index has ever recorded, intents included."""
    ids = {r["yt_video_id"] for r in con.execute("SELECT yt_video_id FROM clips") if r["yt_video_id"]}
    ids |= {r["video_id"] for r in con.execute("SELECT video_id FROM upload_intents") if r["video_id"]}
    return ids
