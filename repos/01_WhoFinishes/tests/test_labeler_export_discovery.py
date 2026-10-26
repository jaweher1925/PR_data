"""The shipped labeler HTMLs and score.py must agree on annotator filenames.

Each labeler HTML is generated public output whose ``a.download=...`` value is
its filename contract: it is the name a replicator's exported sheet lands under.
score.py discovers sheets by globbing those names, so a stale prefix makes a
real annotator's labels vanish silently -- no error, just a set scored without
them. These tests take the name straight from the shipped HTML and hand it to
score.py's real discovery functions.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from prfix.config import PROJECT_ROOT

HA = PROJECT_ROOT / "human-annotation"
JUDGE_LABELS = HA / "labels" / "judge"
HUMAN_LABELS = HA / "labels" / "humans"

DOWNLOAD_RE = re.compile(r"\.download\s*=\s*'([^']+)'")


def _load_score():
    spec = importlib.util.spec_from_file_location("_score", HA / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _export_prefix(html: Path) -> str:
    """The filename stem the labeler prepends to the annotator's name."""
    m = DOWNLOAD_RE.search(html.read_text(encoding="utf-8"))
    assert m, f"no export filename contract in {html.name}"
    return m.group(1)


@pytest.mark.parametrize("labeler,stem", [
    ("rq1_judge_sample50_enriched_labeler.html", "rq1_judge_sample50_enriched"),
    ("rq1_judge_sample50_enriched_humanP_labeler.html",
     "rq1_judge_sample50_enriched_humanP"),
])
def test_judge_labeler_export_is_discovered(tmp_path, monkeypatch, labeler, stem):
    """A sheet exported by the shipped judge labeler is picked up by
    score.discover_annotators for the set it belongs to."""
    prefix = _export_prefix(HA / "data" / "judge" / labeler)
    score = _load_score()

    for f in JUDGE_LABELS.glob(f"{stem}_*model_key.csv"):
        shutil.copy(f, tmp_path / f.name)

    key = pd.read_csv(tmp_path / f"{stem}_model_key.csv")
    pd.DataFrame({"pair_id": key.pair_id, "label": "Direct fix"}).to_csv(
        tmp_path / f"{prefix}freshannotator.csv", index=False)

    monkeypatch.setattr(score, "JUDGE", tmp_path)
    found = score.discover_annotators(stem)

    assert "freshannotator" in found, (
        f"{labeler} exports '{prefix}<name>.csv', which score.py never finds "
        f"for stem '{stem}'")
    assert len(found["freshannotator"]) == len(key)


def test_net_diff_labeler_export_is_discovered(tmp_path, monkeypatch):
    """A sheet exported by the shipped follow-up-validation labeler is picked up
    by score.discover_sheets alongside the real annotator sheets."""
    prefix = _export_prefix(HA / "data" / "net_diff" / "labeler.html")
    score = _load_score()

    for f in HUMAN_LABELS.glob("rq1_followup_validation_annotator*.csv"):
        shutil.copy(f, tmp_path / f.name)
    real = sorted(tmp_path.glob("*.csv"))
    assert real, "no shipped annotator sheets to model the export on"

    sheet = pd.read_csv(real[0])
    sheet["annotator"] = "freshannotator"
    sheet.to_csv(tmp_path / f"{prefix}freshannotator.csv", index=False)

    monkeypatch.setattr(score, "HUMANS", tmp_path)
    found = score.discover_sheets()

    assert "freshannotator" in found, (
        f"labeler.html exports '{prefix}<name>.csv', which score.py never finds")
    assert len(found) == len(real) + 1
