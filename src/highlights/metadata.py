"""Metadata generation: transcript in, YouTube-ready JSON out.

The LLM step is optional and pluggable. With no API key configured, `draft`
produces a template-based result marked `"source": "template"` so the caller
knows nothing was inferred from the audio.

Two rules the LLM prompt enforces, because YouTube punishes both:
  - never dump the transcript into the description
  - never invent a conclusion the speaker did not reach
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

TITLE_MAX = 100          # hard API limit
TITLE_UI = 70            # where YouTube truncates in most surfaces
DESC_MAX = 5000          # hard API limit
TAGS_CHARS = 500         # hard limit on the joined tag string, not per tag
TAG_MAX_LEN = 30         # per-tag limit
CHAPTERS_MIN = 3
CHAPTER_MIN_DURATION = 120  # below this, chapters are noise


@dataclass
class Metadata:
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    summary: str = ""
    source: str = "template"

    def as_dict(self) -> dict:
        return {
            "titre": self.title,
            "description": self.description,
            "tags": self.tags,
            "chapitres": self.chapters,
            "produits_cites": self.products,
            "resume_une_phrase": self.summary,
            "source": self.source,
        }


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def fit_tags(tags: list[str], total_limit: int = TAGS_CHARS, per_tag: int = TAG_MAX_LEN) -> list[str]:
    """Keep tags until the joined string hits YouTube's total budget."""
    out: list[str] = []
    used = 0
    for tag in tags:
        tag = re.sub(r"[\s,]+", " ", tag).strip().strip(",")
        if not tag or len(tag) > per_tag:
            continue
        cost = len(tag) + (1 if out else 0)
        if used + cost > total_limit:
            break
        out.append(tag)
        used += cost
    return out


def build_chapters(segments: list[dict], duration: float, limit: int = 8) -> list[dict]:
    """YouTube needs >= 3 chapters starting at 00:00, else it ignores them all."""
    if duration < CHAPTER_MIN_DURATION or len(segments) < CHAPTERS_MIN:
        return []
    step = duration / min(limit, max(CHAPTERS_MIN, len(segments)))
    chapters = [{"t": "00:00", "titre": "Intro"}]
    elapsed = 0.0
    for seg in segments:
        if seg["start"] - elapsed >= step and seg["start"] < duration - 20:
            chapters.append(
                {
                    "t": f"{int(seg['start']) // 60:02d}:{int(seg['start']) % 60:02d}",
                    "titre": _truncate(seg["text"], 55),
                }
            )
            elapsed = seg["start"]
        if len(chapters) >= limit:
            break
    return chapters if len(chapters) >= CHAPTERS_MIN else []


def render_description(meta: Metadata, context: dict | None = None) -> str:
    """Assemble the description. Context carries the links the pipeline knows."""
    ctx = context or {}
    lines = [meta.description.strip(), ""]
    if meta.chapters:
        lines.append("Chapitres :")
        lines += [f"{c['t']} {c['titre']}" for c in meta.chapters]
        lines.append("")
    if meta.products:
        lines.append("Cité dans l'épisode : " + ", ".join(meta.products))
        lines.append("")
    if ctx.get("twitch_url"):
        lines.append(f"Clip original : {ctx['twitch_url']}")
    if ctx.get("vod_url"):
        lines.append(f"Épisode complet : {ctx['vod_url']}")
    if ctx.get("youtube_url"):
        lines.append(f"Tous nos replays : {ctx['youtube_url']}")
    if ctx.get("issue_url"):
        lines.append(f"Notes de l'épisode : {ctx['issue_url']}")
    body = "\n".join(lines).strip()
    return body[:DESC_MAX]


def draft_templates(title: str, episode: str | None = None, products: list[str] | None = None) -> Metadata:
    """No-LLM fallback: the v1 template of the pipeline. Cannot invent anything."""
    suffix = f" — CuistOps #{episode}" if episode else " — CuistOps"
    return Metadata(
        title=_truncate(f"{title}{suffix}", TITLE_UI),
        description=_truncate(title, 300),
        tags=fit_tags(["Kubernetes", "DevOps", "Talos", "GitOps", "CuistOps"] + (products or [])),
        products=products or [],
        source="template",
    )


PROMPT = """Tu produis les métadonnées YouTube d'un extrait d'émission technique francophone.

Transcription :
\"\"\"
{transcript}
\"\"\"

Contexte fourni :
{context}

Réponds en JSON, sans texte autour, avec exactement ces clés :
- "titre" : 70 caractères maximum, factuel, pas de clickbait, pas de majuscules criardes.
- "description" : 3 à 6 phrases qui résument ce qui est réellement dit. Jamais de transcription brute.
- "tags" : 5 à 12 tags courts, technologies et concepts cités.
- "produits_cites" : uniquement les projets et outils nommés explicitement dans la transcription.
- "resume_une_phrase" : une phrase, pour les annonces réseaux sociaux.

Interdits absolus :
- ne rien conclure que la transcription ne dit pas ;
- ne pas inventer de chiffre, de version ni de nom de produit ;
- ne pas recopier des phrases entières de la transcription.
Si la transcription est trop pauvre pour un champ, mets une chaîne vide plutôt qu'une supposition."""


def _llm_call(prompt: str, model: str, api_key: str, base_url: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def draft_with_llm(
    transcript: str,
    context: dict | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Metadata:
    """Ask a model for the metadata. Falls back to a raised error, not a fake result."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("HERMES_LLM_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.environ.get("HIGHLIGHTS_LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError(
            "no API key: set OPENAI_API_KEY (or HERMES_LLM_KEY), or use draft_templates()"
        )
    prompt = PROMPT.format(transcript=transcript[:12000], context=json.dumps(context or {}, ensure_ascii=False))
    raw = _llm_call(prompt, model, api_key, base_url)
    data = json.loads(raw)
    meta = Metadata(
        title=_truncate(data.get("titre", ""), TITLE_UI),
        description=_truncate(data.get("description", ""), DESC_MAX),
        tags=fit_tags(data.get("tags") or []),
        products=[p for p in (data.get("produits_cites") or []) if isinstance(p, str)][:15],
        summary=_truncate(data.get("resume_une_phrase", ""), 200),
        source="llm",
    )
    if not meta.title:
        raise ValueError("the model returned no title")
    return meta


def validate(meta: Metadata) -> list[str]:
    """Return the reasons the metadata would be rejected or degraded by YouTube."""
    problems = []
    if not meta.title:
        problems.append("titre vide")
    if len(meta.title) > TITLE_MAX:
        problems.append(f"titre {len(meta.title)} > {TITLE_MAX}")
    elif len(meta.title) > TITLE_UI:
        problems.append(f"titre {len(meta.title)} > {TITLE_UI} : tronqué dans l'interface")
    if len(meta.description) > DESC_MAX:
        problems.append(f"description {len(meta.description)} > {DESC_MAX}")
    joined = ",".join(meta.tags)
    if len(joined) > TAGS_CHARS:
        problems.append(f"tags {len(joined)} > {TAGS_CHARS} caractères au total")
    if any(len(t) > TAG_MAX_LEN for t in meta.tags):
        problems.append(f"un tag dépasse {TAG_MAX_LEN} caractères")
    if meta.chapters:
        if len(meta.chapters) < CHAPTERS_MIN:
            problems.append(f"{len(meta.chapters)} chapitres : YouTube en exige {CHAPTERS_MIN} minimum")
        if meta.chapters[0]["t"] != "00:00":
            problems.append("le premier chapitre n'est pas à 00:00")
    elif meta.chapters == []:
        pass  # no chapters is valid; YouTube just shows none
    return problems
