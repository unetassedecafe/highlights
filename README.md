# highlights

Twitch clips to YouTube, with a transcript and metadata that knows what was said.

The v1 pipeline transports a clip. This one understands it: Whisper reads the
audio, rules and one LLM call turn that into a title, a description, tags and
chapters, and a local FTS5 index makes the whole corpus searchable.

Published clips stay `unlisted` until someone promotes them by hand.

## What it does

```
list clips ──► filter (8–90 s, unseen) ──► download ──► normalise 1080p
                                                              │
                                    index ◄── upload ◄── metadata ◄── transcript
                                     FTS5      YouTube      LLM ou gabarit    Whisper
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # no runtime dependency
pip install faster-whisper  # only for transcription
```

`ffmpeg` and `ffprobe` must be on `PATH`.

## Use

```bash
git clone https://github.com/unetassedecafe/highlights.git
cd highlights && python3 -m venv .venv && . .venv/bin/activate && pip install -e .
```

```bash
# what is new on the channel
highlights list --channel cuistops

# one clip, end to end, without touching YouTube
highlights from-clip TawdryTacitDogeMoreCowbell-jLtGZIvYbQfAuuM_ --out ./work --template

# a local file
highlights from-file clip.mp4 --out ./work --vocab vocabulary.md

# a full pass; without credentials it stops before the upload
highlights run --channel cuistops --workdir ./work --db ./clips.db --limit 3 --dry-run

# which episode mentioned Kamaji
highlights search kamaji
```

Exit code 1 from `from-file` / `from-clip` means the metadata has a problem
YouTube would reject or truncate. The problems are listed, and the JSON is still
written so you can read it.

## Credentials

Upload only, all from the environment:

| Variable | Scope |
|---|---|
| `YT_CLIENT_ID` | OAuth client |
| `YT_CLIENT_SECRET` | OAuth client |
| `YT_REFRESH_TOKEN` | `youtube.upload` |

Get the refresh token once, in two steps (the pod is headless, so the loopback
redirect cannot be listened on):

```bash
highlights auth --client-secret ~/Downloads/client_secret.json
# open the printed URL, approve, the browser fails on http://localhost:1
highlights auth --code 'http://localhost:1/?code=4/0A...'
```

You need an OAuth client of type **Desktop app** with the **YouTube Data API v3**
enabled. Publish the OAuth app to Production before trusting it in a cron: while
it is in Testing, Google expires the refresh token after 7 days and the pipeline
stops silently.

Listing and downloading clips needs no credential: the public GraphQL endpoint
serves the clip list and the signed playback URLs. The Helix
`/helix/clips/downloads` endpoint would need a user token with
`editor:manage:clips`, which is why it is not the default path.

## Layout

| File | Role |
|---|---|
| `twitch.py` | list clips, resolve signed media URLs, download, normalise, probe |
| `transcribe.py` | ffmpeg → faster-whisper → segments with timings |
| `cleanup.py` | hesitations, proper nouns, paragraphs, `MM:SS` |
| `metadata.py` | LLM or template → JSON, tag budget, chapters, validation |
| `index.py` | SQLite: `clips`, `clips_fts` (FTS5), `products` |
| `youtube.py` | resumable upload, unlisted, then wait for `processed` |
| `pipeline.py` | the pass: select, download, transcribe, publish, index |
| `cli.py` | `list`, `from-clip`, `from-file`, `run`, `search` |

## Limits it respects

- Title 100 chars (70 before YouTube truncates), description 5 000, tags **500
  chars in total**, each tag 30.
- Chapters: 3 minimum, first at `00:00`, and none below 2 minutes.
- `videos.insert` = 1 unit in its own bucket, 100 per day.
- `clip_id` is the primary key, so a clip is published once even if the pass
  runs fifty times.

## Tests

```bash
python -m pytest -q
```

The suite covers cleanup rules, the tag budget, chapter rules, metadata
validation, the dedup key and FTS5 search. Transcription and upload need a real
clip and credentials, so they are not in it.
