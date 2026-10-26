"""Prompt construction for the RQ1 LLM-as-judge.

The system prompt is RUBRIC + a frozen LEARNED INSIGHTS block (distilled from all
annotator-agreed pairs) + a few worked-example anchors. Labels match the human task:
Direct fix | Related touch | Unrelated/mislabeled | Indeterminate. This module owns
both the READ side (rendering, ``build_system_prompt``) and the WRITE side that
distils the agreed pairs into ``insights.md`` (CLI ``insights build``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
INSIGHTS_PATH = _HERE / "insights.md"


def load_insights() -> str:
    """The frozen LEARNED INSIGHTS block distilled from the annotator-agreed pairs
    (produced by ``build_insights`` below). Empty string if not yet generated."""
    try:
        return INSIGHTS_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""

VALID_LABELS = ("Direct fix", "Related touch", "Unrelated/mislabeled", "Indeterminate")
# Map an off-enum model label (forced tool-use doesn't hard-guarantee the enum)
# back to the canonical class. Keys lowercased; only confident, unambiguous mappings.
_LABEL_SYNONYMS = {
    "unrelated": "Unrelated/mislabeled",
    "mislabeled": "Unrelated/mislabeled",
    "mislabelled": "Unrelated/mislabeled",
    "unrelated/mislabelled": "Unrelated/mislabeled",  # uk spelling
    "direct fix": "Direct fix",
    "related touch": "Related touch",
    "indeterminate": "Indeterminate",
}


def coerce_label(label: str | None) -> str | None:
    """Canonical enum value for a model-returned label, or ``None`` if it cannot
    be confidently mapped (callers mark that an ``ERROR`` rather than silently
    inventing a category that scoring would drop)."""
    if label in VALID_LABELS:
        return label
    return _LABEL_SYNONYMS.get((label or "").strip().lower())


JUDGE_TOOL: dict[str, Any] = {
    "name": "submit_judgment",
    "description": "Return the relationship judgment for the (P, Q) pull-request pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["Direct fix", "Related touch",
                         "Unrelated/mislabeled", "Indeterminate"],
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence in the chosen label.",
            },
            "rationale": {
                "type": "string",
                "description": "1-3 sentences citing the descriptions / commits / "
                               "shared-file diffs that drove the decision.",
            },
            "evidence_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Shared filenames whose diffs were decisive (may be empty).",
            },
        },
        "required": ["label", "confidence", "rationale"],
    },
}

RUBRIC = """\
You are an expert software-engineering reviewer. You are given two pull requests \
from the SAME repository:

  - P: an earlier, AI-agent-authored PR (already merged).
  - Q: a later PR that changed at least one of the same files as P.

Decide how Q relates to P. The question is directional: "Does Q fix P?" -- NOT \
"is Q a good PR?". Judge ONLY from the material provided (the two descriptions, \
the commit messages, and the diffs of the shared files). You do NOT have the rest \
of the codebase; do not assume facts not present in this material.

LABELS
- "Direct fix": Q exists primarily to REPAIR a defect, regression, bug, crash, \
failing test, or wrong/incomplete behavior that P introduced or left behind. \
Tell-tale signs: Q's text describes fixing/correcting an error tied to P's area; \
Q's diff modifies, corrects, or reverts the very lines/logic P added in the \
shared file.
- "Related touch": Q does INCREMENTAL or ADDITIVE work in the same area as P -- \
extending, continuing, or refactoring it -- but is NOT repairing a defect. \
Tell-tale signs: Q adds more of the same kind of thing P added (another feature \
flag, another service, more cases), sanitizes/reformats, or carries the same \
effort forward. Human notes for this class read like "add more X in addition to \
the first PR" and "incremental PR".
- "Unrelated/mislabeled": Q touches the shared file for an INDEPENDENT reason; \
the overlap is coincidental (often a large or common file edited by many PRs). \
Q's purpose has nothing to do with P, or Q is not really a fix at all.
- "Indeterminate": the provided material is genuinely insufficient to decide.

DECISION GUIDANCE
1. Anchor on Q's intent: corrective (fix) vs extending (related) vs independent \
(unrelated).
2. Read the shared-file diffs concretely: does Q change/revert the SPECIFIC lines \
P introduced (-> fix), add alongside them (-> related), or edit a different region \
for a different purpose (-> unrelated)?
3. Do NOT reward mere file overlap. Two PRs editing the same big file for \
different reasons is Unrelated.
4. A "fix" conventional-commit type on Q is WEAK evidence -- many fix-typed PRs \
fix something unrelated to P. Judge the actual relationship, not the label.
5. Be conservative with "Direct fix": choose it only when Q clearly targets P's \
change or a defect in it.
6. ATTRIBUTION -- credit Q with repairing P only when Q's OWN added/removed lines \
land on lines P introduced, NOT when the two merely share a file. Each shared file \
carries a machine-computed [P-overlap] note that does this attribution for you. \
When the note says "git blame confirms ...", it is GROUND TRUTH: GitHub's blame \
was run at Q's merge-base on the exact lines Q removes, so "git blame confirms N \
line(s) Q removes were authored by P" is the STRONGEST Direct-fix evidence, and \
"git blame confirms none ... were authored by P -> NOT a fix of P" reliably rules a \
direct fix OUT (Q is deleting someone else's code, not P's). When blame is silent \
(pure-addition file, or the note instead reports content matches), it falls back to \
matching line CONTENT: "Q deletes/replaces N line(s) P introduced" is strong \
evidence, "comment/whitespace-only" means Q's net change is cosmetic (NOT a fix), \
and "do not touch any line P introduced" means Q edits different code (-> Related or \
Unrelated). Why this is needed: P's patch is the diff vs the state BEFORE P and Q's \
patch is the diff vs the state BEFORE Q -- DIFFERENT baselines -- so an unprefixed \
CONTEXT line in Q's patch is NOT necessarily code P wrote; a PR merged BETWEEN P and \
Q may have authored it. Always weight Q's actual +/- overlap with P (especially a \
blame confirmation) over mere context matches -- a context-only match is likely an \
intermediate PR's line, the more so when the metadata says many other PRs touched \
the file between P and Q. If the note says attribution is UNCERTAIN (rebase), lean \
on Q's described intent and the diff content instead.
7. INTENT vs EFFECT -- commit messages state intent and may narrate work that \
actually landed in OTHER PRs (so the net diff dedups it away). The diff is the \
effect. When the messages claim a fix the shown +/- lines do not contain, trust \
the diff, not the messages.
"""


def truncate(text: str | None, cap: int | None) -> str:
    """Cap ``text`` at ``cap`` chars with an explicit marker, so the judge knows it
    is seeing a PARTIAL diff. ``cap=None`` disables truncation (the render-time
    default, since patches are already capped once at build time)."""
    text = text or ""
    if cap is None or len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {len(text) - cap:,} more chars] ..."


def _attribution_blame(s: dict, r: int) -> str | None:
    """Verdict from the authoritative Method-B signal (``git blame`` at Q's
    merge-base), or None when blame is unavailable so the caller falls back to the
    content-overlap heuristic. ``r`` = Method-A removed-line content matches, used
    only to flag a disagreement (a likely rebase-merge that rewrote P's commits)."""
    status, b2p, bn = s.get("blame_status"), s.get("blame_to_p"), s.get("blame_n_removed")
    if status != "ok":
        return None  # no_removed / unavailable / absent -> use content heuristic
    if b2p:
        return (f"git blame confirms {b2p} of the {bn} line(s) Q removes here were "
                f"authored by P -> Q directly deletes/replaces code P wrote (strong "
                f"Direct-fix evidence)")
    if r:  # blame maps none to P, yet the text matches P's added lines
        return (f"git blame attributes none of the {bn} line(s) Q removes to P, though "
                f"{r} match P's added text -- P's commits were likely rebase-rewritten, "
                f"so attribution here is UNCERTAIN (weigh other evidence)")
    return (f"git blame confirms none of the {bn} line(s) Q removes were authored by P "
            f"-> Q edits code P did not write (NOT a fix of P)")


def _attribution_line(s: dict) -> str:
    """The per-file [P-overlap] note: does Q's OWN change land on lines P introduced?

    Prefers Method B (``git blame`` ground truth, attached at build time as
    ``blame_*``); falls back to Method A (content line-matching from the two patches
    in the case) when blame is unavailable (pure-addition file, rename, doc file, or
    a ``--no-blame`` build). Returns '' when P has no recorded additions AND blame is
    silent (nothing to attribute against). See prompt RUBRIC #6."""
    from prfix.rq1_fix_rates.judge.attribution import file_provenance, is_code  # noqa: PLC0415
    p_patch = s.get("p_patch") or ""
    if not p_patch.strip():
        return ""  # P's patch for this file is unavailable -> nothing to attribute
    prov = file_provenance(p_patch, s.get("q_patch"))
    r, cx, eff = prov["n_removed_hits"], prov["n_context_hits"], prov["q_eff"]
    k = s.get("intervening") or 0
    doc = ("" if is_code(s["name"])
           else " (doc/data file -- a change here is not a code-defect repair)")
    verdict = _attribution_blame(s, r)
    if verdict is None:                                    # ---- Method A fallback ----
        if not prov["p_added"]:
            verdict = ("P made no substantive additions to this file, so the file overlap "
                       "is likely coincidental")
        elif eff == 0:
            verdict = "Q's net change here is comment/whitespace-only -> NOT a code fix of P"
        elif r:
            verdict = (f"Q deletes/replaces {r} line(s) P introduced -> Q directly edits "
                       f"a line P wrote")
        elif cx:
            caveat = (f"; {k} other PR(s) touched this file between P and Q, so these "
                      f"context matches may be theirs, not P's") if k else ""
            verdict = (f"Q changes real code but none of P's added lines; {cx} of P's lines "
                       f"appear only as unchanged context{caveat}")
        else:
            verdict = "Q's changed lines do not touch any line P introduced"
    b2p, bn = s.get("blame_to_p"), s.get("blame_n_removed")
    blame_tail = (f", blame: {b2p}/{bn} removed-lines-to-P"
                  if s.get("blame_status") == "ok" else "")
    return (f"  [P-overlap]{doc} {verdict}. "
            f"(Q-removes-on-P={r}, P-lines-in-Q-context={cx}, "
            f"Q-substantive-lines={eff}{blame_tail})\n")


def _patch_block(shared: list[dict], max_files: int | None, patch_cap: int | None) -> str:
    if not shared:
        return "(no shared-file diffs available)"
    ordered = sorted(
        shared,
        key=lambda s: (s.get("boilerplate", False),
                       -(len(s.get("p_patch", "")) + len(s.get("q_patch", "")))),
    )
    if max_files is not None:
        ordered = ordered[:max_files]
    out = []
    for s in ordered:
        tag = " [boilerplate]" if s.get("boilerplate") else ""
        attr = _attribution_line(s)
        p = truncate(s.get("p_patch") or "(none)", patch_cap)
        q = truncate(s.get("q_patch") or "(none)", patch_cap)
        out.append(
            f"File: {s['name']}{tag}\n"
            f"{attr}"
            f"--- P's change to this file ---\n{p}\n"
            f"--- Q's change to this file ---\n{q}"
        )
    extra = len(shared) - len(ordered)
    if extra > 0:
        out.append(f"(+{extra} more shared file(s) omitted)")
    return "\n\n".join(out)


def _commits(msgs: list[str]) -> str:
    return "\n".join(f"- {m}" for m in msgs) if msgs else "(none captured -- e.g. squash-merged)"


def _provenance_banner(shared: list[dict]) -> str:
    """A machine-computed caution shown above the diffs when Q's only change to the
    shared CODE is cosmetic (the #69 over-credit pattern). Empty otherwise."""
    from prfix.rq1_fix_rates.judge.attribution import pair_provenance  # noqa: PLC0415
    warn = pair_provenance(shared).get("warning")
    return f"!! {warn}\n\n" if warn else ""


def _extra_block(files: list[dict], key: str, label: str,
                 max_files: int | None, patch_cap: int | None) -> str:
    """Render one PR's non-shared changed files (single-sided patches)."""
    if not files:
        return ""
    ordered = sorted(
        files, key=lambda s: (s.get("boilerplate", False), -len(s.get(key, ""))),
    )
    if max_files is not None:
        ordered = ordered[:max_files]
    rows = [f"\n\n{label}:"]
    for s in ordered:
        tag = " [boilerplate]" if s.get("boilerplate") else ""
        rows.append(f"File: {s['name']}{tag}\n{truncate(s.get(key) or '(none)', patch_cap)}")
    extra = len(files) - len(ordered)
    if extra > 0:
        rows.append(f"(+{extra} more non-shared file(s) omitted)")
    return "\n".join(rows)


def render_case(case: dict, *, max_files: int | None = None, patch_cap: int | None = None,
                include_extra: bool = True) -> str:
    """Render one pair into the judge's user message. ``max_files``/``patch_cap``
    cap files-per-section and patch length (``None`` = show all / no re-truncation,
    the eval default). With ``include_extra`` on a ``--full-diff`` case, each PR's
    non-shared files are appended; exemplars pass small caps and
    ``include_extra=False`` to stay compact.
    """
    base = (
        f"Earlier PR  P: \"{case['p_title']}\"\n"
        f"P description:\n{(case.get('p_body') or '(empty)').strip()}\n"
        f"P commit messages:\n{_commits(case.get('p_commits', []))}\n\n"
        f"Later PR  Q: \"{case['q_title']}\"\n"
        f"Q description:\n{(case.get('q_body') or '(empty)').strip()}\n"
        f"Q commit messages:\n{_commits(case.get('q_commits', []))}\n\n"
        f"Metadata: Q merged {case.get('day_gap')} day(s) after P; "
        f"Q conventional-commit type = {case.get('q_type') or 'unknown'}.\n"
        f"Between P and Q, {case.get('n_intervening', 0)} other PR(s) merged that "
        f"also touched the shared file(s) (so some unchanged context in Q's patch "
        f"may be their work, not P's); see each file's [P-overlap] note for whether "
        f"Q's own edits actually land on lines P introduced.\n\n"
        f"{_provenance_banner(case.get('shared', []))}"
        f"SHARED FILES CHANGED BY BOTH P AND Q:\n"
        f"{_patch_block(case.get('shared', []), max_files, patch_cap)}"
    )
    if not include_extra:
        return base
    base += _extra_block(
        case.get("q_only", []), "q_patch",
        "OTHER FILES CHANGED BY Q (not shared with P -- a relocated fix may live here)",
        max_files, patch_cap)
    base += _extra_block(
        case.get("p_only", []), "p_patch",
        "OTHER FILES CHANGED BY P (not shared with Q)",
        max_files, patch_cap)
    return base


def pick_exemplars(cases: list[dict], per_class: int = 3) -> tuple[list[int], list[int]]:
    """Deterministically choose ``per_class`` worked-example anchors per label,
    preferring pairs with a human reason note, then more substantive shared files,
    then lowest pair_id. Returns (exemplar_ids, rest_ids) over consensus-labeled
    pairs only. (The insights block already distils ALL agreed pairs; these are a
    few concrete anchors, so ``rest_ids`` is no longer a held-out eval set.)"""
    classes = ["Direct fix", "Related touch", "Unrelated/mislabeled"]
    consensus = [c for c in cases if c.get("consensus_label") in classes]
    chosen: list[int] = []
    for cls in classes:
        pool = [c for c in consensus if c["consensus_label"] == cls]
        pool.sort(key=lambda c: (
            0 if (c.get("reason_note") or "").strip() else 1,
            -int(c.get("n_substantive", 0)),
            int(c["pair_id"]),
        ))
        chosen.extend(int(c["pair_id"]) for c in pool[:per_class])
    chosen_set = set(chosen)
    eval_ids = [int(c["pair_id"]) for c in consensus if int(c["pair_id"]) not in chosen_set]
    return chosen, eval_ids


def build_system_prompt(cases: list[dict], exemplar_ids: list[int],
                        *, patch_cap: int = 1200, max_files: int = 2,
                        insights: str | None = None) -> str:
    """RUBRIC + LEARNED INSIGHTS (distilled from all agreed pairs) + a few worked-
    example anchors. ``insights`` defaults to the frozen ``insights.md`` artifact;
    pass ``""`` to omit the block (e.g. before it has been generated)."""
    by_id = {int(c["pair_id"]): c for c in cases}
    ins = load_insights() if insights is None else insights.strip()
    blocks = [RUBRIC]
    if ins:
        blocks.append(
            "\nLEARNED INSIGHTS (distilled from the annotator-agreed pairs):\n" + ins)
    blocks.append("\nWORKED EXAMPLES (gold-labeled by human reviewers):")
    for i, pid in enumerate(exemplar_ids, 1):
        c = by_id[pid]
        note = (c.get("reason_note") or "").strip()
        why = f"\nWhy: {note}" if note else ""
        blocks.append(
            f"\n### Example {i} -- GOLD LABEL: {c['consensus_label']}{why}\n"
            f"{render_case(c, max_files=max_files, patch_cap=patch_cap, include_extra=False)}"
        )
    blocks.append(
        "\nNow judge the new pair the user provides. Call submit_judgment with your "
        "label, confidence, a short rationale, and the decisive shared files."
    )
    return "\n".join(blocks)


# ============================================================================
# LEARNED INSIGHTS -- distil the annotator-agreed gold pairs into insights.md
# ============================================================================
# Write side (the read side is above). distil() makes ONE LLM call over every agreed
# pair to extract per-class heuristics + failure patterns that COMPLEMENT the rubric,
# frozen to insights.md so the prompt is reproducible. CLI: `prompt insights build`.
# The engine import is function-local everywhere below: engine.py imports THIS module
# (JUDGE_TOOL, render_case), so a top-level import here would be circular.
CONSENSUS_CLASSES = ["Direct fix", "Related touch", "Unrelated/mislabeled"]

DISTILL_MODEL = "claude-opus-4-8"   # default distiller; any Anthropic id via --model
DISTILL_EFFORT = "low"              # the task is narrow; low effort is plenty

DISTILL_SYSTEM = (
    "You are an expert methodologist distilling decision guidance from a set of "
    "human-adjudicated examples. The downstream task: given two pull requests P "
    "(an AI-agent PR) and Q (a later PR) in the same repo that change overlapping "
    "files, decide whether Q is a genuine follow-up FIX of P -- labelled one of "
    "Direct fix / Related touch / Unrelated-mislabeled / Indeterminate. Below is "
    "the rubric the judge already uses, then every pair TWO independent annotators "
    "agreed on, with the agreed label and their reasoning. Produce concise, "
    "actionable INSIGHTS that COMPLEMENT the rubric (do not restate it): concrete "
    "decision heuristics that separate the classes, and the recurring failure / "
    "edge patterns the agreed examples reveal (e.g. cosmetic-only net diffs, "
    "overlap that is coincidental or boilerplate, attribution to code P did not "
    "author). Ground each point in the examples. Output GitHub-flavoured Markdown: "
    "short '### Direct fix vs ...' / '### Common traps' style headers with tight "
    "bullets. No preamble, no restatement of the task, no closing remarks."
)


def _consensus(cases: list[dict]) -> list[dict]:
    """The pairs both annotators agreed on (a judged class, not Indeterminate)."""
    return [c for c in cases if c.get("consensus_label") in CONSENSUS_CLASSES]


def _examples_block(cases: list[dict]) -> str:
    """All agreed pairs, grouped by class, each rendered compactly with its gold
    label + reason note -- the evidence the distiller reasons over."""
    by_class: dict[str, list[dict]] = {c: [] for c in CONSENSUS_CLASSES}
    for c in cases:
        by_class[c["consensus_label"]].append(c)
    parts: list[str] = []
    for cls in CONSENSUS_CLASSES:
        group = sorted(by_class[cls], key=lambda c: int(c["pair_id"]))
        parts.append(f"\n## GOLD CLASS: {cls}  ({len(group)} agreed pairs)")
        for c in group:
            note = (c.get("reason_note") or "").strip()
            why = f"\nAnnotator reasoning: {note}" if note else ""
            parts.append(
                f"\n### pair {c['pair_id']} -- {cls}{why}\n"
                f"{render_case(c, max_files=2, patch_cap=800, include_extra=False)}"
            )
    return "\n".join(parts)


def build_distill_prompt(cases: list[dict]) -> str:
    """The user-message the distiller reads: rubric + all agreed examples."""
    cons = _consensus(cases)
    return (
        "CURRENT RUBRIC (for context -- produce insights that complement it):\n"
        f"{RUBRIC}\n\n"
        f"AGREED EXAMPLES ({len(cons)} pairs both annotators labelled the same):\n"
        f"{_examples_block(cons)}"
    )


def distil(model: str = DISTILL_MODEL, effort: str | None = DISTILL_EFFORT) -> str:
    """One LLM call over all agreed pairs -> the LEARNED INSIGHTS markdown block."""
    from prfix.rq1_fix_rates.judge import (
        engine as engine_mod,  # noqa: PLC0415 -- lazy: engine imports this module
    )
    cases = engine_mod.load_cases()
    user = build_distill_prompt(cases)
    key = engine_mod._resolve_api_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (env or .env).")
    text = engine_mod.complete_text(
        f"anthropic/{model}", DISTILL_SYSTEM, user, api_key=key,
        api_base=engine_mod._resolve_env("ANTHROPIC_BASE_URL"),
        max_tokens=4096,
        temperature=0 if engine_mod._supports_temperature(model) else None,
        extra={"output_config": {"effort": effort}} if effort else None,
    )
    return text.strip()


def build_insights(model: str = DISTILL_MODEL, effort: str | None = DISTILL_EFFORT) -> str:
    """Distil and freeze ``insights.md``. Returns the written text."""
    from prfix.rq1_fix_rates.judge import engine as engine_mod  # noqa: PLC0415
    text = distil(model, effort)
    n = len(_consensus(engine_mod.load_cases()))
    header = (f"<!-- Distilled from {n} annotator-agreed gold pairs by "
              f"{model}{f' (effort={effort})' if effort else ''}. "
              f"Regenerate: python -m prfix.rq1_fix_rates.judge.prompt insights build -->\n\n")
    INSIGHTS_PATH.write_text(header + text + "\n", encoding="utf-8")
    return text


def main() -> None:
    """CLI: ``python -m prfix.rq1_fix_rates.judge.prompt insights {build,show}``."""
    import argparse  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    from prfix.rq1_fix_rates.judge import engine as engine_mod  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        prog="python -m prfix.rq1_fix_rates.judge.prompt",
        description="Prompt assets for the RQ1 LLM-judge (distil / show LEARNED INSIGHTS)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("insights", help="distil / show the LEARNED INSIGHTS block")
    insub = ins.add_subparsers(dest="action", required=True)
    bp = insub.add_parser("build", help="distil the agreed pairs and freeze insights.md")
    bp.add_argument("--model", default=DISTILL_MODEL, help="distiller model (default: %(default)s)")
    bp.add_argument("--effort", default=DISTILL_EFFORT,
                    help="Anthropic output_config effort ('' to omit; default: %(default)s)")
    bp.add_argument("--dry-run", action="store_true",
                    help="print the distillation prompt + counts and exit (no API call)")
    insub.add_parser("show", help="print the current insights.md")
    args = ap.parse_args()

    if args.action == "show":
        print(INSIGHTS_PATH.read_text(encoding="utf-8") if INSIGHTS_PATH.exists()
              else f"(no insights yet at {INSIGHTS_PATH} -- run `prompt insights build`)")
        return

    effort = args.effort or None
    if args.dry_run:
        cases = engine_mod.load_cases()
        cons = _consensus(cases)
        dist = Counter(c["consensus_label"] for c in cons)
        print(f"agreed pairs: {len(cons)}  {dict(dist)}")
        print(f"distiller: anthropic/{args.model}" + (f" effort={effort}" if effort else ""))
        print(f"would write: {INSIGHTS_PATH}")
        print("\n===== DISTILLATION PROMPT (user message) =====\n")
        print(build_distill_prompt(cases))
        return

    text = build_insights(args.model, effort)
    print(f"wrote {INSIGHTS_PATH}  ({len(text)} chars)")


if __name__ == "__main__":
    main()
