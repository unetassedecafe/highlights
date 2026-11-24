"""Guard the shipped dictionary: no rule may rewrite correct French.

Every rule in data/corrections.txt is applied to sentences where the pattern is
legitimate text, not a misheard product name. Three rules failed this check when
written — "cargo => Kargo", "traffic => Traefik", "et cetera => etcd" — and were
removed. A regression here means the transcript becomes confidently wrong.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from highlights import cleanup

CORRECTIONS = Path(__file__).parent.parent / "src/highlights/data/corrections.txt"

# (input, substring that must survive untouched)
MUST_SURVIVE = [
    ("Cargo c'est le gestionnaire de paquets Rust", "Cargo"),
    ("le traffic network est correct", "traffic"),
    ("et cetera, et cetera", "et cetera"),
    ("minimaliste comme approche", "minimaliste"),
    ("le trafic réseau est élevé", "trafic"),
    ("l'audio du stream est bon", "audio"),
    ("on déploie Talos sur Infomaniak", "Talos"),
    ("on a mis du Ceph dans le cluster", "Ceph"),
]

# (input, expected substring after correction)
SHOULD_FIX = [
    ("bienvenue sur Quistop ce soir", "CuistOps"),
    ("on a du rust fs et ça marche", "RustFS"),
    ("S3 avec radoze gateway", "Rook Gateway"),
    ("on parle de minali", "MinIO"),
    ("il a installé sillium", "Cilium"),
    ("le cluster est sous talos", "Talos"),
]


@pytest.fixture(scope="module")
def rules():
    return cleanup.read_corrections(CORRECTIONS)


def test_the_dictionary_exists_and_has_rules(rules):
    assert len(rules) > 40, "the corrections file looks truncated or unparsed"


@pytest.mark.parametrize("text,keep", MUST_SURVIVE)
def test_no_rule_rewrites_correct_text(rules, text, keep):
    out = cleanup.apply_corrections(text, rules)
    assert keep in out, f"{text!r} -> {out!r} destroyed {keep!r}"


@pytest.mark.parametrize("text,expected", SHOULD_FIX)
def test_the_rules_still_fix_what_they_target(rules, text, expected):
    out = cleanup.apply_corrections(text, rules)
    assert expected in out, f"{text!r} -> {out!r} did not produce {expected!r}"


def test_rules_are_case_insensitive_but_word_bounded(rules):
    assert "CuistOps" in cleanup.apply_corrections("QUISTOP", rules)
    # Inside a longer word, nothing happens.
    assert cleanup.apply_corrections("quistoper", rules) == "quistoper"
