"""Step 1+2 of manual analysis: LLM identifies performance-improving hunks and classifies
them into 18 mutation categories; you spot-check the results.

Workflow:
  1. --auto-accept  LLM identifies perf hunks in all patches, auto-records them (no review).
  2. --classify     Runs 18-category classifier on every accepted hunk (also automatic).
  3. (default)      Interactive spot-check of accepted results — confirm, reject, or override.
  4. --export       Write final CSV.

Usage:
  .venv/bin/python3 scripts/perf_improvement_filter.py --auto-accept      # Step 1 (automated)
  .venv/bin/python3 scripts/perf_improvement_filter.py --classify         # Step 2a: Anthropic classifier
  .venv/bin/python3 scripts/perf_improvement_filter.py --classify-openai  # Step 2b: OpenAI classifier
  .venv/bin/python3 scripts/perf_improvement_filter.py --retry-errors        # Step 2c: retry failed classifications
  .venv/bin/python3 scripts/perf_improvement_filter.py --retry-cache-errors  # Step 1b: retry overloaded/errored perf-hunk identifications
  .venv/bin/python3 scripts/perf_improvement_filter.py                    # Step 3 (spot-check)
  .venv/bin/python3 scripts/perf_improvement_filter.py --export           # Step 4 (export CSV)
  .venv/bin/python3 scripts/perf_improvement_filter.py --manual-only      # restrict to manual_review PRs only

Commands during review:
  y / a       — ACCEPT: confirm the LLM's suggested hunks
  r           — REJECT: not a performance improvement (override LLM)
  1 2 3       — Override: accept only those hunk numbers instead
  s           — SKIP: come back later
  b           — BACK: go to previous patch
  j NUM       — JUMP to patch number NUM
  o           — OPEN PR URL in browser
  q           — QUIT and save
  Enter       — Expand truncated diff
"""
import asyncio
import csv
import json
import os
import re
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import openai
import pandas as pd

from analysis.patch_classifier import _build_prompt, _parse_response
from config import (
    ANTHROPIC_MODEL,
    BUSINESS_LOGIC_EXTENSIONS,
    CLASSIFY_CONCURRENCY,
    EXCLUDED_FILE_PATTERNS,
    MSR_DATA_DIR,
    OPENAI_MODEL,
)

PROGRESS_FILE = "perf_improvement_progress.json"
EXPORT_CSV = "perf_improvement_parts.csv"
OPENAI_CLASSIFICATIONS_FILE = "openai_classifications.json"

# ANSI colors
RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

HUNK_PREVIEW_LINES = 50

PERF_HUNK_PROMPT = """\
You are analysing a code diff from a pull request that has been labelled as a \
performance improvement. Your task: identify which numbered diff hunks contain \
code changes that actually cause a runtime performance improvement.

A hunk IS a performance improvement if it:
- Improves algorithmic complexity (e.g. O(n²) → O(n))
- Adds caching, memoisation, or avoids repeated computation
- Replaces a slow data structure or library with a faster one
- Batches operations to reduce I/O, network, or API round-trips
- Enables asynchronous or parallel execution
- Reduces memory allocations or copies
- Avoids unnecessary work (early exit, lazy evaluation, short-circuit)
- Adds connection/thread pooling

A hunk is NOT a performance improvement if it only:
- Reformats, renames, or reorganises code without changing behaviour
- Updates comments or documentation
- Adds/changes type annotations with no runtime effect
- Adds error handling, logging, or test coverage
- Makes style-only changes

File: {filename}

Diff (hunks numbered):
{numbered_patch}

Return ONLY valid JSON, no markdown fences:
{{"performance_hunks": [{{"hunk_index": <int>, "reasoning": "<one sentence>"}}], \
"overall_reasoning": "<one sentence summary>"}}

If no hunk is a performance improvement return {{"performance_hunks": [], \
"overall_reasoning": "<why none qualify>"}}"""


# ── Data loading ──────────────────────────────────────────────────────────────

def _is_source_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in BUSINESS_LOGIC_EXTENSIONS:
        return False
    for pat in EXCLUDED_FILE_PATTERNS:
        if re.search(pat, filename):
            return False
    return True


def load_patches() -> list[dict]:
    """Return sorted list of patch dicts for all qualifying manual_review PRs."""
    agent_prs_path = os.path.join(MSR_DATA_DIR, "Agent_PRs.csv")
    with open(agent_prs_path, newline="", encoding="utf-8") as f:
        all_prs = list(csv.DictReader(f))

    manual_only = "--manual-only" in sys.argv
    label_sources = {"manual_review"} if manual_only else {"manual_review", "llm_agreement"}
    filtered = [
        r for r in all_prs
        if r["Label Source"] in label_sources and r["Final Sub Pattern"].strip()
    ]
    pr_meta = {int(r["PR ID"]): r for r in filtered}
    pr_ids = set(pr_meta.keys())

    parquet_path = os.path.join(MSR_DATA_DIR, "ai_pr_commit_details.parquet")
    commits = pd.read_parquet(parquet_path)
    mc = commits[commits["pr_id"].isin(pr_ids)].copy()
    source = mc[mc["filename"].apply(_is_source_file)].copy()
    source = source[source["patch"].notna() & (source["patch"] != "")]

    patches = []
    for _, row in source.iterrows():
        pr_id = int(row["pr_id"])
        meta = pr_meta[pr_id]
        patches.append({
            "pr_id": pr_id,
            "pr_url": meta["PR URL"],
            "final_pattern": meta["Final Pattern"],
            "final_sub_pattern": meta["Final Sub Pattern"],
            "sha": str(row["sha"]) if pd.notna(row["sha"]) else "",
            "commit_sha": str(row["commit_sha"]) if pd.notna(row["commit_sha"]) else "",
            "filename": row["filename"],
            "additions": int(row["additions"]) if pd.notna(row["additions"]) else 0,
            "deletions": int(row["deletions"]) if pd.notna(row["deletions"]) else 0,
            "patch": row["patch"],
        })

    patches.sort(key=lambda p: (p["final_sub_pattern"], p["pr_id"], p["sha"], p["filename"]))
    return patches


def patch_key(p: dict) -> str:
    return f"{p['pr_id']}:{p['sha']}:{p['filename']}"


# ── Hunk parsing ──────────────────────────────────────────────────────────────

def parse_hunks(patch: str) -> list[str]:
    """Split unified diff into individual @@ hunks."""
    parts = re.split(r"(?=^@@)", patch, flags=re.MULTILINE)
    return [p for p in parts if p.strip().startswith("@@")]


def _numbered_patch(hunks: list[str]) -> str:
    """Format hunks with [N] headers for the LLM prompt."""
    sections = []
    for i, hunk in enumerate(hunks, 1):
        sections.append(f"[{i}]\n{hunk}")
    return "\n".join(sections)


# ── LLM classification ────────────────────────────────────────────────────────

def _parse_llm_response(text: str) -> dict:
    """Parse LLM JSON response. Returns {performance_hunks, overall_reasoning}."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        hunks = [
            {"hunk_index": int(h["hunk_index"]), "reasoning": str(h.get("reasoning", ""))}
            for h in data.get("performance_hunks", [])
        ]
        return {
            "performance_hunks": hunks,
            "overall_reasoning": str(data.get("overall_reasoning", "")),
        }
    except Exception:
        return {"performance_hunks": [], "overall_reasoning": f"parse error: {text[:120]}"}


async def _classify_batch(patches_to_classify: list[dict]) -> dict:
    """Classify patches. Returns {patch_key: llm_result}."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    results = {}

    async def classify_one(p):
        hunks = parse_hunks(p["patch"])
        prompt = PERF_HUNK_PROMPT.format(
            filename=p["filename"],
            numbered_patch=_numbered_patch(hunks),
        )
        key = patch_key(p)
        async with semaphore:
            try:
                while True:
                    try:
                        msg = await client.messages.create(
                            model=ANTHROPIC_MODEL,
                            max_tokens=2048,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        results[key] = _parse_llm_response(msg.content[0].text)
                        return
                    except anthropic.RateLimitError as exc:
                        wait = 60.0
                        if exc.response is not None:
                            h = (exc.response.headers.get("retry-after") or
                                 exc.response.headers.get("x-ratelimit-reset-requests"))
                            if h:
                                wait = float(h)
                        await asyncio.sleep(wait)
            except Exception as exc:
                results[key] = {
                    "performance_hunks": [],
                    "overall_reasoning": f"ERROR: {exc}",
                }

    await asyncio.gather(*[classify_one(p) for p in patches_to_classify])
    await client.close()
    return results


def ensure_llm_classifications(patches: list[dict], progress: dict) -> dict:
    """Run LLM on any uncached patches. Results stored in progress['llm_cache']."""
    cache = progress.setdefault("llm_cache", {})
    missing = [p for p in patches if patch_key(p) not in cache]
    if not missing:
        return progress

    print(f"\n{CYAN}Running LLM on {len(missing)} patches to find performance hunks...{RESET}")
    print("(Runs once — results are cached)\n")

    results = asyncio.run(_classify_batch(missing))
    cache.update(results)
    save_progress(progress)
    print(f"{GREEN}Done. LLM classifications cached.{RESET}\n")
    return progress


# ── Auto-accept (step 1 fully automated) ─────────────────────────────────────

def auto_accept(patches: list[dict], progress: dict):
    """Record all LLM hunk suggestions without interactive review."""
    decisions = progress.setdefault("decisions", {})
    accepted_parts = progress.setdefault("accepted_parts", [])
    llm_cache = progress.get("llm_cache", {})

    newly_decided = 0
    for patch in patches:
        key = patch_key(patch)
        if key in decisions:
            continue  # already reviewed — don't overwrite
        llm = llm_cache.get(key, {})
        hunks = parse_hunks(patch["patch"])
        suggested = sorted({
            h["hunk_index"] for h in llm.get("performance_hunks", [])
            if 1 <= h["hunk_index"] <= len(hunks)
        })
        if suggested:
            _record_accepted(patch, hunks, suggested, decisions, accepted_parts, key)
        else:
            decisions[key] = {"decision": "rejected", "accepted_hunks": None}
        newly_decided += 1

    save_progress(progress)
    accepted_count = sum(1 for d in decisions.values() if d["decision"] in ("accepted", "partial"))
    rejected_count = sum(1 for d in decisions.values() if d["decision"] == "rejected")
    total_parts = len(accepted_parts)
    print(f"{GREEN}Auto-accept complete.{RESET}  "
          f"accepted={accepted_count}  rejected={rejected_count}  "
          f"total hunks stored={total_parts}  (newly decided: {newly_decided})")


# ── Classify accepted hunks into 18 categories (step 2 fully automated) ──────

async def _classify_hunks_batch(parts: list[dict]) -> dict:
    """Classify accepted hunk contents into 18 mutation categories.
    Returns {index_in_list: (categories, reasoning)}."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    results = {}

    async def classify_one(i, part):
        prompt = _build_prompt(part["filename"], part["hunk_content"])
        async with semaphore:
            try:
                while True:
                    try:
                        msg = await client.messages.create(
                            model=ANTHROPIC_MODEL,
                            max_tokens=2048,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        cats, reasoning = _parse_response(msg.content[0].text)
                        results[i] = (cats, reasoning)
                        return
                    except anthropic.RateLimitError as exc:
                        wait = 60.0
                        if exc.response is not None:
                            h = (exc.response.headers.get("retry-after") or
                                 exc.response.headers.get("x-ratelimit-reset-requests"))
                            if h:
                                wait = float(h)
                        await asyncio.sleep(wait)
            except Exception as exc:
                results[i] = ([], f"ERROR: {exc}")

    await asyncio.gather(*[classify_one(i, p) for i, p in enumerate(parts)])
    await client.close()
    return results


def classify_accepted_hunks(progress: dict):
    """Run 18-category classifier on all accepted hunks that lack a classification."""
    parts = progress.get("accepted_parts", [])
    missing_idx = [i for i, p in enumerate(parts) if "categories" not in p]
    if not missing_idx:
        print(f"{GREEN}All accepted hunks already classified.{RESET}")
        return

    print(f"\n{CYAN}Classifying {len(missing_idx)} accepted hunks into 18 categories...{RESET}")
    print("(Results stored per hunk in the progress file)\n")

    missing_parts = [parts[i] for i in missing_idx]
    results = asyncio.run(_classify_hunks_batch(missing_parts))

    for list_pos, part_idx in enumerate(missing_idx):
        cats, reasoning = results.get(list_pos, ([], ""))
        parts[part_idx]["categories"] = cats
        parts[part_idx]["categories_reasoning"] = reasoning

    save_progress(progress)
    print(f"{GREEN}Done. {len(missing_idx)} hunks classified.{RESET}\n")


# ── Part key helper ───────────────────────────────────────────────────────────

def _part_key(part: dict) -> str:
    """Stable string key identifying a specific hunk across both classification files."""
    return f"{part['pr_id']}:{part['commit_sha']}:{part['filename']}:{part['hunk_index']}"


# ── Classify accepted hunks with OpenAI (step 2b, alternative model) ──────────

async def _classify_hunks_batch_openai(parts: list[dict]) -> dict:
    """Classify accepted hunk contents using OpenAI.
    Returns {index_in_list: (categories, reasoning)}."""
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    results = {}

    async def classify_one(i, part):
        prompt = _build_prompt(part["filename"], part["hunk_content"])
        async with semaphore:
            try:
                while True:
                    try:
                        resp = await client.chat.completions.create(
                            model=OPENAI_MODEL,
                            max_completion_tokens=512,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        cats, reasoning = _parse_response(resp.choices[0].message.content)
                        results[i] = (cats, reasoning)
                        return
                    except openai.RateLimitError as exc:
                        wait = 60.0
                        retry_after = getattr(exc, "response", None)
                        if retry_after is not None:
                            h = (retry_after.headers.get("retry-after") or
                                 retry_after.headers.get("x-ratelimit-reset-requests"))
                            if h:
                                wait = float(h)
                        await asyncio.sleep(wait)
            except Exception as exc:
                results[i] = ([], f"ERROR: {exc}")

    await asyncio.gather(*[classify_one(i, p) for i, p in enumerate(parts)])
    await client.close()
    return results


def classify_accepted_hunks_openai(progress: dict):
    """Run 18-category OpenAI classifier on all accepted hunks that lack a classification."""
    parts = progress.get("accepted_parts", [])

    # Load existing OpenAI results (resumable)
    if os.path.exists(OPENAI_CLASSIFICATIONS_FILE):
        with open(OPENAI_CLASSIFICATIONS_FILE, encoding="utf-8") as f:
            openai_data = json.load(f)
    else:
        openai_data = {}

    missing_idx = [i for i, p in enumerate(parts) if _part_key(p) not in openai_data]
    if not missing_idx:
        print(f"{GREEN}All accepted hunks already classified by OpenAI.{RESET}")
        return

    print(f"\n{CYAN}Classifying {len(missing_idx)} accepted hunks with OpenAI ({OPENAI_MODEL})...{RESET}")
    print("(Results stored in openai_classifications.json)\n")

    missing_parts = [parts[i] for i in missing_idx]
    results = asyncio.run(_classify_hunks_batch_openai(missing_parts))

    for list_pos, part_idx in enumerate(missing_idx):
        cats, reasoning = results.get(list_pos, ([], ""))
        key = _part_key(parts[part_idx])
        openai_data[key] = {"categories": cats, "reasoning": reasoning}

    with open(OPENAI_CLASSIFICATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(openai_data, f, indent=2)
    print(f"{GREEN}Done. {len(missing_idx)} hunks classified by OpenAI.{RESET}\n")


# ── Retry errors ─────────────────────────────────────────────────────────────

def _is_error(reasoning: str) -> bool:
    return str(reasoning).startswith("ERROR:")


def retry_classification_errors(progress: dict):
    """Re-classify any hunks that previously errored in Anthropic or OpenAI runs."""
    parts = progress.get("accepted_parts", [])

    # ── Anthropic retry ───────────────────────────────────────────────────────
    error_idx = [
        i for i, p in enumerate(parts)
        if _is_error(p.get("categories_reasoning", ""))
    ]
    if not error_idx:
        print(f"{GREEN}No Anthropic classification errors to retry.{RESET}")
    else:
        print(f"\n{CYAN}Retrying {len(error_idx)} Anthropic classification errors...{RESET}")
        error_parts = [parts[i] for i in error_idx]
        results = asyncio.run(_classify_hunks_batch(error_parts))
        for list_pos, part_idx in enumerate(error_idx):
            cats, reasoning = results.get(list_pos, ([], ""))
            parts[part_idx]["categories"] = cats
            parts[part_idx]["categories_reasoning"] = reasoning
        save_progress(progress)
        print(f"{GREEN}Done. {len(error_idx)} Anthropic hunks re-classified.{RESET}")

    # ── OpenAI retry ──────────────────────────────────────────────────────────
    if os.path.exists(OPENAI_CLASSIFICATIONS_FILE):
        with open(OPENAI_CLASSIFICATIONS_FILE, encoding="utf-8") as f:
            openai_data = json.load(f)
    else:
        openai_data = {}

    error_keys = [k for k, v in openai_data.items() if _is_error(v.get("reasoning", ""))]
    if not error_keys:
        print(f"{GREEN}No OpenAI classification errors to retry.{RESET}")
    else:
        print(f"\n{CYAN}Retrying {len(error_keys)} OpenAI classification errors...{RESET}")
        part_lookup = {_part_key(p): p for p in parts}
        error_parts = [part_lookup[k] for k in error_keys if k in part_lookup]
        results = asyncio.run(_classify_hunks_batch_openai(error_parts))
        for list_pos, key in enumerate(k for k in error_keys if k in part_lookup):
            cats, reasoning = results.get(list_pos, ([], ""))
            openai_data[key] = {"categories": cats, "reasoning": reasoning}
        with open(OPENAI_CLASSIFICATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(openai_data, f, indent=2)
        print(f"{GREEN}Done. {len(error_parts)} OpenAI hunks re-classified.{RESET}")

    # ── Validation ────────────────────────────────────────────────────────────
    ant_remaining = [p for p in parts if _is_error(p.get("categories_reasoning", ""))]
    oai_remaining = [k for k, v in openai_data.items() if _is_error(v.get("reasoning", ""))]
    print(f"\n  Validation:")
    ant_ok = len(ant_remaining) == 0
    oai_ok = len(oai_remaining) == 0
    print(f"    Anthropic errors remaining: {len(ant_remaining)}  {'✓' if ant_ok else '✗'}")
    print(f"    OpenAI errors remaining:    {len(oai_remaining)}  {'✓' if oai_ok else '✗'}")
    if ant_remaining:
        for p in ant_remaining[:5]:
            print(f"      {_part_key(p)}: {p.get('categories_reasoning','')[:80]}")
    if oai_remaining:
        for k in oai_remaining[:5]:
            print(f"      {k}: {openai_data[k].get('reasoning','')[:80]}")


def retry_cache_errors(patches: list[dict], progress: dict):
    """Re-run perf-hunk identification for cache entries that errored (e.g. overloaded_error)."""
    cache = progress.setdefault("llm_cache", {})
    decisions = progress.setdefault("decisions", {})

    error_keys = {k for k, v in cache.items()
                  if str(v.get("overall_reasoning", "")).startswith("ERROR")}
    if not error_keys:
        print(f"{GREEN}No errored cache entries found.{RESET}")
        return

    print(f"\n{CYAN}Clearing {len(error_keys)} errored cache entries and their decisions...{RESET}")
    for k in error_keys:
        del cache[k]
        decisions.pop(k, None)
    save_progress(progress)

    retry_patches = [p for p in patches if patch_key(p) in error_keys]
    print(f"{CYAN}Re-running LLM on {len(retry_patches)} patches...{RESET}\n")
    results = asyncio.run(_classify_batch(retry_patches))
    cache.update(results)
    save_progress(progress)

    new_errors = [k for k in error_keys if str(cache.get(k, {}).get("overall_reasoning", "")).startswith("ERROR")]
    print(f"{GREEN}Done. {len(error_keys) - len(new_errors)}/{len(error_keys)} retried successfully.{RESET}")
    if new_errors:
        print(f"  Still errored: {len(new_errors)}")

    print(f"\n{CYAN}Auto-accepting retried patches...{RESET}")
    auto_accept(retry_patches, progress)


# ── Progress I/O ──────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": 0, "decisions": {}, "accepted_parts": [], "llm_cache": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ── Display helpers ───────────────────────────────────────────────────────────

def _color_line(line: str) -> str:
    if line.startswith("+"):
        return f"{GREEN}{line}{RESET}"
    if line.startswith("-"):
        return f"{RED}{line}{RESET}"
    if line.startswith("@@"):
        return f"{CYAN}{line}{RESET}"
    return line


def show_hunk(hunk: str, hunk_num: int, highlighted: bool = False, expanded: bool = False) -> bool:
    """Display one hunk. highlighted=True marks it as LLM-suggested. Returns True if fully shown."""
    lines = hunk.splitlines()
    total = len(lines)
    to_show = lines if expanded else lines[:HUNK_PREVIEW_LINES]

    marker = f"{GREEN}★{RESET} " if highlighted else "  "
    print(f"\n{marker}{BOLD}{YELLOW}[{hunk_num}]{RESET}")
    for line in to_show:
        print(f"    {_color_line(line)}")

    if not expanded and total > HUNK_PREVIEW_LINES:
        print(f"    {DIM}... {total - HUNK_PREVIEW_LINES} more lines (press Enter to expand){RESET}")
        return False
    return True


def show_patch_header(patch: dict, idx: int, total: int):
    width = 72
    print(f"\n{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD} Patch {idx+1}/{total}  │  PR {patch['pr_id']}  │  {patch['final_sub_pattern']}{RESET}")
    print(f"  {CYAN}{patch['pr_url']}{RESET}")
    print(f"  {DIM}File:{RESET} {patch['filename']}  "
          f"{GREEN}+{patch['additions']}{RESET} {RED}-{patch['deletions']}{RESET}  "
          f"{DIM}sha:{RESET} {patch['sha'][:8]}")
    print(f"{'─' * width}")


def show_llm_suggestion(llm: dict, n_hunks: int):
    suggested = [h["hunk_index"] for h in llm.get("performance_hunks", [])]
    reasoning = llm.get("overall_reasoning", "")
    if suggested:
        idx_str = "  ".join(str(i) for i in suggested)
        print(f"\n  {BOLD}LLM suggests:{RESET} {GREEN}hunks [{idx_str}]{RESET}  "
              f"({GREEN}★{RESET} = suggested)")
    else:
        print(f"\n  {BOLD}LLM suggests:{RESET} {RED}no performance hunks{RESET}")
    if reasoning:
        print(f"  {DIM}└─ {reasoning}{RESET}")


def show_existing_decision(decision: dict):
    d = decision.get("decision", "")
    if not d:
        return
    color = GREEN if d == "accepted" else (YELLOW if d in ("partial", "skipped") else RED)
    hunks = decision.get("accepted_hunks")
    hunk_str = f" → hunks {hunks}" if hunks else ""
    print(f"  {DIM}[already reviewed: {color}{d}{RESET}{DIM}{hunk_str}]{RESET}")


# ── Main review loop ──────────────────────────────────────────────────────────

def run_filter(patches: list[dict], progress: dict):
    total = len(patches)
    decisions = progress.setdefault("decisions", {})
    accepted_parts = progress.setdefault("accepted_parts", [])
    llm_cache = progress.get("llm_cache", {})

    # Resume from first un-reviewed patch
    idx = progress.get("last_index", 0)
    for i in range(idx, total):
        if patch_key(patches[i]) not in decisions:
            idx = i
            break
    else:
        idx = 0

    reviewed_this_session = 0
    expanded = False

    while 0 <= idx < total:
        patch = patches[idx]
        key = patch_key(patch)
        hunks = parse_hunks(patch["patch"])
        n_hunks = len(hunks)
        llm = llm_cache.get(key, {"performance_hunks": [], "overall_reasoning": ""})
        suggested_indices = {
            h["hunk_index"] for h in llm.get("performance_hunks", [])
            if 1 <= h["hunk_index"] <= len(hunks)
        }
        existing = decisions.get(key)

        expanded = False
        show_patch_header(patch, idx, total)

        for i, hunk in enumerate(hunks, 1):
            show_hunk(hunk, i, highlighted=(i in suggested_indices), expanded=False)

        show_llm_suggestion(llm, n_hunks)
        if existing:
            show_existing_decision(existing)

        done_count = len(decisions)
        accepted_count = sum(1 for d in decisions.values() if d["decision"] in ("accepted", "partial"))
        print(f"\n  {DIM}Progress: {done_count}/{total} reviewed, {accepted_count} with accepted hunks{RESET}")
        print(f"  {BOLD}[y/a]ccept LLM  [r]eject all  1..{n_hunks} override  [s]kip  [b]ack  [j N]ump  [o]pen  [q]uit{RESET}")

        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{YELLOW}Saving and exiting...{RESET}")
                progress["last_index"] = idx
                save_progress(progress)
                return

            if not raw:
                if not expanded:
                    for i, hunk in enumerate(hunks, 1):
                        show_hunk(hunk, i, highlighted=(i in suggested_indices), expanded=True)
                    expanded = True
                continue

            cmd = raw.split()[0].lower()
            arg = raw[len(cmd):].strip()

            if cmd == "q":
                progress["last_index"] = idx
                save_progress(progress)
                print(f"\n{GREEN}Saved. Reviewed {reviewed_this_session} patches this session "
                      f"({len(decisions)}/{total} total).{RESET}")
                return

            elif cmd == "o":
                webbrowser.open(patch["pr_url"])
                continue

            elif cmd == "b":
                idx = max(0, idx - 1)
                break

            elif cmd == "j":
                try:
                    target = int(arg) - 1
                    if 0 <= target < total:
                        idx = target
                        break
                    print(f"  {RED}Out of range. Enter 1-{total}.{RESET}")
                except ValueError:
                    print(f"  {RED}Usage: j NUM{RESET}")
                continue

            elif cmd == "s":
                decisions[key] = {"decision": "skipped", "accepted_hunks": None}
                save_progress(progress)
                idx += 1
                break

            elif cmd == "r":
                decisions[key] = {"decision": "rejected", "accepted_hunks": None}
                save_progress(progress)
                reviewed_this_session += 1
                idx += 1
                break

            elif cmd in ("y", "a"):
                if suggested_indices:
                    selected = sorted(suggested_indices)
                    _record_accepted(patch, hunks, selected, decisions, accepted_parts, key)
                else:
                    # LLM found nothing; accept all if user insists
                    decisions[key] = {"decision": "rejected", "accepted_hunks": None}
                    print(f"  {DIM}LLM found no performance hunks — recorded as rejected. "
                          f"Use hunk numbers to override.{RESET}")
                save_progress(progress)
                reviewed_this_session += 1
                idx += 1
                break

            else:
                # Try to parse as hunk number override
                try:
                    nums_raw = raw.replace(",", " ").split()
                    selected = [int(x) for x in nums_raw]
                    invalid = [n for n in selected if not (1 <= n <= n_hunks)]
                    if invalid:
                        print(f"  {RED}Invalid hunk numbers: {invalid}. Valid: 1-{n_hunks}{RESET}")
                        continue
                    _record_accepted(patch, hunks, selected, decisions, accepted_parts, key)
                    save_progress(progress)
                    reviewed_this_session += 1
                    idx += 1
                    break
                except ValueError:
                    print(f"  {RED}Unknown command. Use: y r 1..{n_hunks} s b j o q{RESET}")

    print(f"\n{GREEN}All patches reviewed! Run with --export to generate CSV.{RESET}")


def _record_accepted(patch: dict, hunks: list[str], selected: list[int],
                     decisions: dict, accepted_parts: list, key: str):
    decision = "accepted" if len(selected) == len(hunks) else "partial"
    decisions[key] = {"decision": decision, "accepted_hunks": selected}

    # Remove any prior entries for this patch (re-review)
    accepted_parts[:] = [
        p for p in accepted_parts
        if not (p["pr_id"] == patch["pr_id"]
                and p["commit_sha"] == patch["commit_sha"]
                and p["filename"] == patch["filename"])
    ]

    for hunk_num in selected:
        accepted_parts.append({
            "pr_id": patch["pr_id"],
            "commit_sha": patch["commit_sha"],
            "filename": patch["filename"],
            "pr_url": patch["pr_url"],
            "final_sub_pattern": patch["final_sub_pattern"],
            "hunk_index": hunk_num,
            "hunk_content": hunks[hunk_num - 1],
        })


# ── CSV export ────────────────────────────────────────────────────────────────

def _fmt_categories(cats: list) -> str:
    if not cats:
        return ""
    return "; ".join(
        f"{c['id']}:{c['name']}({c['confidence']:.2f})"
        + (f"@L{c['start_line']}-{c['end_line']}" if c.get("start_line") else "")
        for c in cats
    )


def export_csv(progress: dict):
    parts = progress.get("accepted_parts", [])

    # Load OpenAI classifications and apply intersection at export time.
    # perf_improvement_progress.json is kept with original Anthropic labels so
    # that agreement stats (full / partial / full_disagree) remain accurate.
    openai_data: dict = {}
    if os.path.exists(OPENAI_CLASSIFICATIONS_FILE):
        with open(OPENAI_CLASSIFICATIONS_FILE, encoding="utf-8") as f:
            openai_data = json.load(f)

    fieldnames = [
        "part_id", "pr_id", "pr_url", "final_sub_pattern",
        "commit_sha", "filename", "hunk_index",
        "agreement_type", "categories", "categories_reasoning", "hunk_content",
    ]
    n_full = n_partial = n_disagree = 0
    part_id = 0
    with open(EXPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for part in parts:
            ant_cats = part.get("categories") or []
            ant_ids = {c["id"] for c in ant_cats if isinstance(c.get("id"), int)}

            oai_entry = openai_data.get(_part_key(part), {})
            oai_ids = {c["id"] for c in oai_entry.get("categories", [])
                       if isinstance(c.get("id"), int)}

            inter_ids = ant_ids & oai_ids

            if ant_ids == oai_ids:
                agreement_type = "full_agree"
                final_cats = ant_cats
                n_full += 1
            elif inter_ids:
                # Partial agreement: keep Anthropic objects for the intersected IDs
                # (preserves confidence scores and line ranges from the calibrated model)
                agreement_type = "partial_agree"
                final_cats = [c for c in ant_cats if c.get("id") in inter_ids]
                n_partial += 1
            else:
                # Full disagreement — discard per policy (no shared categories).
                n_disagree += 1
                continue

            part_id += 1
            writer.writerow({
                "part_id": part_id,
                "pr_id": part["pr_id"],
                "pr_url": part["pr_url"],
                "final_sub_pattern": part["final_sub_pattern"],
                "commit_sha": part["commit_sha"],
                "filename": part["filename"],
                "hunk_index": part["hunk_index"],
                "agreement_type": agreement_type,
                "categories": _fmt_categories(final_cats),
                "categories_reasoning": part.get("categories_reasoning", ""),
                "hunk_content": part["hunk_content"],
            })
    print(f"  Agreement breakdown: {n_full} full | {n_partial} partial (intersection applied) | {n_disagree} full-disagree (discarded)")

    decisions = progress.get("decisions", {})
    reviewed = sum(1 for d in decisions.values() if d["decision"] != "skipped")
    print(f"{GREEN}Exported {part_id} agreed hunks ({n_disagree} full-disagree discarded) from {reviewed} reviewed patches → {EXPORT_CSV}{RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])
    export_mode          = "--export" in args
    auto_mode            = "--auto-accept" in args
    classify_mode        = "--classify" in args
    classify_openai_mode = "--classify-openai" in args
    retry_errors_mode      = "--retry-errors" in args
    retry_cache_errors_mode = "--retry-cache-errors" in args

    progress = load_progress()

    if export_mode:
        export_csv(progress)
        return

    if classify_mode:
        classify_accepted_hunks(progress)
        return

    if classify_openai_mode:
        classify_accepted_hunks_openai(progress)
        return

    if retry_errors_mode:
        retry_classification_errors(progress)
        return

    print(f"{BOLD}Loading patches...{RESET}")
    patches = load_patches()
    print(f"Loaded {len(patches)} source patches across {len(set(p['pr_id'] for p in patches))} PRs.")

    progress = ensure_llm_classifications(patches, progress)

    if retry_cache_errors_mode:
        retry_cache_errors(patches, progress)
        return

    if auto_mode:
        auto_accept(patches, progress)
        return

    run_filter(patches, progress)


if __name__ == "__main__":
    main()
