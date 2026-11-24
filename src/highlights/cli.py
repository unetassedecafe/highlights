"""CLI. `highlights from-file` is the one-command path the skill documents."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from . import index, pipeline, transcribe, twitch
from .clip import Settings


def cmd_from_file(args) -> int:
    from . import clip

    result = clip.process_file(
        Path(args.file),
        out_dir=Path(args.out) if args.out else None,
        model=args.model,
        use_llm=not args.template,
        vocab=Path(args.vocab) if args.vocab else None,
        corrections=Path(args.corrections) if args.corrections else None,
        twitch_title=args.title,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["problems"] else 0


def cmd_from_clip(args) -> int:
    from . import clip

    result = clip.process_clip(
        args.slug,
        Path(args.out),
        model=args.model,
        use_llm=not args.template,
        vocab=Path(args.vocab) if args.vocab else None,
        corrections=Path(args.corrections) if args.corrections else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["problems"] else 0


def cmd_list(args) -> int:
    clips = twitch.list_clips(args.channel, first=args.first)
    if args.json:
        print(json.dumps(clips, ensure_ascii=False, indent=2))
    else:
        for c in clips:
            print(f"{c['clip_id']:>12}  {c['duration_s']:>3}s  {c['created_at'][:10]}  {c['title'][:70]}")
    return 0


def cmd_run(args) -> int:
    settings = Settings(
        workdir=Path(args.workdir),
        db=Path(args.db),
        vocab=Path(args.vocab) if args.vocab else None,
        corrections=Path(args.corrections) if args.corrections else None,
        model=args.model,
        use_llm=not args.template,
        client_id=args.client_id,
        privacy=args.privacy,
        youtube_url=args.youtube_url,
        since_hours=args.hours,
        redo=args.redo,
        # Identifies this pass in the lease, so a second concurrent run can see
        # who holds a clip instead of guessing.
        owner=f"{socket.gethostname()}:{os.getpid()}",
    )
    report = pipeline.run(args.channel, settings, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 2


def cmd_search(args) -> int:
    con = index.connect(args.db)
    rows = index.search(con, args.query, limit=args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(f"{r['clip_id']:>12}  {r['title'][:60]}\n              …{r['extrait']}")
    return 0 if rows else 1


def cmd_auth(args) -> int:
    from . import auth

    if args.client_secret:
        client_id, client_secret = auth.load_client(args.client_secret)
        redirect_uri = auth.redirect_uri_for(args.client_secret)
        verifier = auth.new_verifier()
        auth.save_pending(verifier, client_id, client_secret, redirect_uri)
        print("Open this URL, approve, then paste the ENTIRE redirected URL back:")
        print()
        print(auth.consent_url(client_id, verifier, redirect_uri))
        print()
        print(f"The browser will fail on {redirect_uri} after approval — that is expected.")
        print("Then run:  highlights auth --code '<the full url from the address bar>'")
        return 0

    if args.code:
        pending = auth.load_pending()
        code = auth.extract_code(args.code)
        data = auth.exchange(
            code,
            pending["client_id"],
            pending["client_secret"],
            pending["verifier"],
            pending.get("redirect_uri", auth.DEFAULT_REDIRECT_URI),
        )
        auth.clear_pending()
        # Write the token to disk: printing it invites two failures — the terminal
        # truncates or redacts it, and a copy-paste into an env var leaves it in
        # shell history. One code exchange yields one refresh token, so losing the
        # printed value costs a whole new consent.
        path = auth.save_refresh_token(data["refresh_token"])
        print(f"Authorized. Refresh token written to {path} (mode 600).")
        print(f"  client_id: {pending['client_id']}")
        print(f"  scopes:    {data.get('scope', '(not reported)')}")
        print()
        print("Store it in Vault, never in the repo:")
        print(f"  mise run vault:kv patch kv/hermes-carolyn "
              f"google_youtube_refresh_token=\"$(cat {path})\"")
        return 0

    print("give --client-secret <file> or --code '<redirect url>'", file=sys.stderr)
    return 2


def cmd_vod(args) -> int:
    from . import vodrun
    from . import vod as vodmod

    if args.list:
        rows = vodmod.list_vods(args.channel, first=args.first)
        for r in rows:
            print(f"{r['vod_id']:>12}  {r['duration_s'] // 60:>3} min  "
                  f"{r['created_at'][:10]}  {r['title'][:60]}")
        return 0

    if not args.vod_id:
        print("give a vod id, or --list", file=sys.stderr)
        return 2

    payload = vodrun.process_vod(
        args.vod_id,
        Path(args.workdir),
        model=args.model,
        chunk_s=args.chunk_minutes * 60,
        vocab=Path(args.vocab) if args.vocab else None,
        corrections=Path(args.corrections) if args.corrections else None,
        max_chunks=args.max_chunks,
        title_hint=args.title,
    )
    print(json.dumps({k: payload[k] for k in
                      ("titre", "duration_s", "segments", "problems", "workdir")},
                     ensure_ascii=False, indent=2))
    print()
    print(payload["chapters_text"])

    if args.publish:
        from . import youtube

        work = Path(payload["workdir"])
        video = work / "vod.mp4"
        if not video.exists():
            print(f"\ndownloading the replay for upload (quality {args.quality})…")
            vodmod.download_video(args.vod_id, video, quality=args.quality)
        token = youtube.access_token(*youtube.credentials_from_env())
        print(f"uploading {video.stat().st_size / 1e9:.2f} GB as {args.privacy}…")
        vid = youtube.upload(video, payload, token, privacy=args.privacy)
        print(f"published: https://www.youtube.com/watch?v={vid} ({args.privacy})")
        payload["yt_video_id"] = vid

    return 1 if payload["problems"] else 0


def cmd_shorts(args) -> int:
    from . import metadata, shorts

    vod_dir = Path(args.vod_dir)
    segs = shorts.load_segments(vod_dir)
    print(f"{len(segs)} segments from {vod_dir}")
    units = shorts.sentences(segs)
    print(f"{len(units)} sentences after merging the 2 s fragments")

    pool = shorts.windows(units, length=args.length)
    print(f"{len(pool)} candidate passages of ≤{args.length:.0f}s")

    def ask(prompt: str) -> str:
        return metadata._llm_call(
            prompt,
            args.model,
            os.environ.get("OPENAI_API_KEY") or os.environ.get("HERMES_LLM_KEY", ""),
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    clips = shorts.pick(pool, count=args.count, topic=args.topic, llm=ask, segments=units)
    if not clips:
        print("the model chose nothing", file=sys.stderr)
        return 1
    # Drop trailing words that leave the thought hanging, and refuse to publish a
    # clip that never closes: that is the defect a viewer notices first.
    trimmed = [shorts.trim_to_sentence(c) for c in clips]
    closed = [c for c in trimmed if not c.get("truncated")]
    for c in trimmed:
        if c.get("truncated"):
            print(f"  dropping a clip that ends mid-sentence: {c['text'][-60:]!r}")
    clips = closed
    if not clips:
        print("no clip ends on a complete sentence; raise --length or lower the bar",
              file=sys.stderr)
        return 1

    video = Path(args.video) if args.video else vod_dir / "vod.mp4"
    if not video.exists():
        print(f"no video at {video}: pass --video, or download it with `highlights vod --publish --dry-run`",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else vod_dir / "shorts"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, clip in enumerate(clips, 1):
        name = f"short{i:02d}"
        ass = out_dir / f"{name}.ass"
        ass.write_text(
            shorts.render_captions(clip["segments"], clip["start"], clip["end"]),
            encoding="utf-8",
        )
        target = out_dir / f"{name}.mp4"
        shorts.render(video, clip, target, crop=not args.no_crop, ass_path=ass)
        size = target.stat().st_size / 1e6
        print(f"  {name}  {clip['start']:.0f}–{clip['end']:.0f}s  {size:.1f} MB  {clip['titre']}")
        manifest.append(
            {
                "file": str(target),
                "start": clip["start"],
                "end": clip["end"],
                "duration_s": round(clip["end"] - clip["start"], 1),
                "titre": clip["titre"],
                "pourquoi": clip["pourquoi"],
                "texte": clip["text"],
            }
        )
    (out_dir / "shorts.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest)} shorts in {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="highlights", description="Clips Twitch -> YouTube, with a transcript.")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("from-file", help="a local video/audio file to metadata")
    f.add_argument("file")
    f.add_argument("--out")
    f.add_argument("--title", help="Twitch title, used as context for the LLM")
    f.add_argument("--model", default="small")
    f.add_argument("--vocab")
    f.add_argument("--corrections")
    f.add_argument("--template", action="store_true", help="skip the LLM, use templates")
    f.set_defaults(func=cmd_from_file)

    c = sub.add_parser("from-clip", help="a Twitch clip slug to metadata")
    c.add_argument("slug")
    c.add_argument("--out", default="./work")
    c.add_argument("--model", default="small")
    c.add_argument("--vocab")
    c.add_argument("--corrections")
    c.add_argument("--template", action="store_true")
    c.set_defaults(func=cmd_from_clip)

    l = sub.add_parser("list", help="newest clips of a channel")
    l.add_argument("--channel", default="cuistops")
    l.add_argument("--first", type=int, default=20)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("run", help="full pass: list, download, transcribe, upload, index")
    r.add_argument("--channel", default="cuistops")
    r.add_argument("--workdir", default="./work")
    r.add_argument("--db", default="./clips.db")
    r.add_argument("--vocab")
    r.add_argument("--corrections")
    r.add_argument("--model", default="small")
    r.add_argument("--limit", type=int, default=5)
    r.add_argument("--hours", type=int, default=24)
    r.add_argument("--privacy", default="private", choices=["public", "unlisted", "private"],
                   help="YouTube visibility; private = owner only (no 'invisible' mode exists)")
    r.add_argument("--youtube-url")
    r.add_argument("--client-id", help="set to enable upload; reads the rest from env")
    r.add_argument("--template", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--redo", action="store_true", help="transcribe again even if a payload exists")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("search", help="full-text search over indexed transcripts")
    s.add_argument("query")
    s.add_argument("--db", default="./clips.db")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    a = sub.add_parser("auth", help="one-time OAuth to get a youtube.upload refresh token")
    a.add_argument("--client-secret", help="path to the Google client_secret.json (Desktop app)")
    a.add_argument("--code", help="the whole redirected URL after approving")
    a.set_defaults(func=cmd_auth)

    v = sub.add_parser("vod", help="process a full replay (2h+) in chunks")
    v.add_argument("vod_id", nargs="?", help="the numeric Twitch video id")
    v.add_argument("--list", action="store_true", help="list recent replays and exit")
    v.add_argument("--channel", default="cuistops")
    v.add_argument("--first", type=int, default=20)
    v.add_argument("--workdir", default="./work")
    v.add_argument("--model", default="small")
    v.add_argument("--chunk-minutes", type=float, default=10)
    v.add_argument("--max-chunks", type=int, help="stop after N chunks (for a smoke test)")
    v.add_argument("--title", help="override the VOD title used for metadata")
    v.add_argument("--vocab")
    v.add_argument("--corrections")
    v.add_argument("--publish", action="store_true",
                   help="download the replay and upload it to YouTube")
    v.add_argument("--privacy", default="private", choices=["public", "unlisted", "private"])
    v.add_argument("--quality", default="1080", help="video variant height to download")
    v.set_defaults(func=cmd_vod)

    sh = sub.add_parser("shorts", help="cut vertical shorts out of a processed replay")
    sh.add_argument("vod_dir", help="the workdir produced by `highlights vod`")
    sh.add_argument("--count", type=int, default=5)
    sh.add_argument("--length", type=float, default=45.0, help="max seconds per short")
    sh.add_argument("--topic", default="", help="episode subject, helps the model choose")
    sh.add_argument("--video", help="the replay mp4 (defaults to <vod_dir>/vod.mp4)")
    sh.add_argument("--out", help="output directory")
    sh.add_argument("--no-crop", action="store_true", help="keep 16:9 (for terminal demos)")
    sh.add_argument("--model", default="gpt-4o-mini")
    sh.set_defaults(func=cmd_shorts)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
