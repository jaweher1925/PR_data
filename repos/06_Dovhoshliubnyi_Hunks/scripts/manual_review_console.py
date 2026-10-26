"""Interactive console tool for manual patch analysis.

Usage:
  .venv/bin/python3 scripts/manual_review_console.py          # Start/resume review
  .venv/bin/python3 scripts/manual_review_console.py --export # Export results to CSV

Commands during review:
  c         — CORRECT: accept LLM categories as-is
  p         — PARTIAL: accept some, then fix (prompted)
  i         — INCORRECT: enter correct category IDs manually
  s         — SKIP: come back later
  b         — BACK: go to previous patch
  j NUM     — JUMP: jump to patch number NUM
  o         — OPEN: open PR URL in browser
  ?         — show category reference
  q         — QUIT and save
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
import pandas as pd

from analysis.patch_classifier import MUTATION_CATEGORIES, _build_prompt, _parse_response
from config import (
    ANTHROPIC_MODEL,
    BUSINESS_LOGIC_EXTENSIONS,
    CLASSIFY_CONCURRENCY,
    EXCLUDED_FILE_PATTERNS,
    MSR_DATA_DIR,
)

PROGRESS_FILE = "manual_review_progress.json"
EXPORT_CSV = "manual_review_results.csv"

# ANSI colors
RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DIFF_PREVIEW_LINES = 60
DIFF_EXPAND_THRESHOLD = 60


# ── Data loading ─────────────────────────────────────────────────────────────

def _is_source_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in BUSINESS_LOGIC_EXTENSIONS:
        return False
    for pat in EXCLUDED_FILE_PATTERNS:
        if re.search(pat, filename):
            return False
    return True


def load_patches() -> list[dict]:
    """Return sorted list of patch dicts for all qualifying PRs."""
    agent_prs_path = os.path.join(MSR_DATA_DIR, "Agent_PRs.csv")
    with open(agent_prs_path, newline="", encoding="utf-8") as f:
        all_prs = list(csv.DictReader(f))

    filtered = [
        r for r in all_prs
        if r["Label Source"] == "manual_review" and r["Final Sub Pattern"].strip()
    ]
    pr_meta = {int(r["PR ID"]): r for r in filtered}
    pr_ids = set(pr_meta.keys())

    parquet_path = os.path.join(MSR_DATA_DIR, "ai_pr_commit_details.parquet")
    commits = pd.read_parquet(parquet_path)
    mc = commits[commits["pr_id"].isin(pr_ids)].copy()
    source = mc[mc["filename"].apply(_is_source_file)].copy()

    source = source.sort_values(["pr_id", "sha", "filename"])

    patches = []
    for _, row in source.iterrows():
        pr_id = int(row["pr_id"])
        meta = pr_meta[pr_id]
        patches.append({
            "pr_id": pr_id,
            "pr_url": meta["PR URL"],
            "final_pattern": meta["Final Pattern"],
            "final_sub_pattern": meta["Final Sub Pattern"],
            "sha": str(row["sha"])[:8] if pd.notna(row["sha"]) else "",
            "filename": row["filename"],
            "additions": int(row["additions"]) if pd.notna(row["additions"]) else 0,
            "deletions": int(row["deletions"]) if pd.notna(row["deletions"]) else 0,
            "patch": (row["patch"] or "") if pd.notna(row["patch"]) else "",
        })

    # Sort by sub_pattern -> pr_id -> sha -> filename
    patches.sort(key=lambda p: (p["final_sub_pattern"], p["pr_id"], p["sha"], p["filename"]))
    return patches


def patch_key(p: dict) -> str:
    return f"{p['pr_id']}:{p['sha']}:{p['filename']}"


# ── LLM classification ────────────────────────────────────────────────────────

async def _classify_batch(patches_to_classify: list[dict]) -> dict:
    """Classify a list of patches. Returns {patch_key: (categories, reasoning)}."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    results = {}

    async def classify_one(p):
        prompt = _build_prompt(p["filename"], p["patch"])
        async with semaphore:
            try:
                while True:
                    try:
                        msg = await client.messages.create(
                            model=ANTHROPIC_MODEL,
                            max_tokens=512,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        cats, reasoning = _parse_response(msg.content[0].text)
                        results[patch_key(p)] = (cats, reasoning)
                        return
                    except anthropic.RateLimitError as exc:
                        wait = 60.0
                        if exc.response is not None:
                            h = exc.response.headers.get("retry-after") or \
                                exc.response.headers.get("x-ratelimit-reset-requests")
                            if h:
                                wait = float(h)
                        await asyncio.sleep(wait)
            except Exception as exc:
                results[patch_key(p)] = ([], f"ERROR: {exc}")

    await asyncio.gather(*[classify_one(p) for p in patches_to_classify])
    await client.close()
    return results


def ensure_llm_classifications(patches: list[dict], progress: dict) -> dict:
    """For any patch without a cached classification, run the LLM. Returns updated progress."""
    missing = [p for p in patches if patch_key(p) not in progress.get("llm_cache", {})]
    if not missing:
        return progress

    print(f"\n{CYAN}Running LLM classifier on {len(missing)} patches...{RESET}")
    print("(This only runs once — results are cached)\n")

    llm_results = asyncio.run(_classify_batch(missing))
    cache = progress.setdefault("llm_cache", {})
    for key, (cats, reasoning) in llm_results.items():
        cache[key] = {"categories": cats, "reasoning": reasoning}

    save_progress(progress)
    print(f"{GREEN}Done. Classifications cached.{RESET}\n")
    return progress


# ── Progress I/O ──────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": 0, "reviews": {}, "llm_cache": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ── Display helpers ───────────────────────────────────────────────────────────

def _color_diff_line(line: str) -> str:
    if line.startswith("+"):
        return f"{GREEN}{line}{RESET}"
    if line.startswith("-"):
        return f"{RED}{line}{RESET}"
    if line.startswith("@@"):
        return f"{CYAN}{line}{RESET}"
    return line


def show_diff(patch: str, expanded: bool = False):
    lines = patch.splitlines()
    total = len(lines)
    if not expanded and total > DIFF_EXPAND_THRESHOLD:
        to_show = lines[:DIFF_PREVIEW_LINES]
    else:
        to_show = lines

    for i, line in enumerate(to_show, 1):
        print(f"  {DIM}{i:3d}{RESET} │ {_color_diff_line(line)}")

    if not expanded and total > DIFF_EXPAND_THRESHOLD:
        hidden = total - DIFF_PREVIEW_LINES
        print(f"\n  {DIM}... {hidden} more lines — press Enter to expand{RESET}")
        return False  # not fully expanded
    return True  # fully shown


def show_patch_header(patch: dict, idx: int, total: int):
    width = 70
    print(f"\n{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD} Patch {idx+1}/{total}  │  PR {patch['pr_id']}  │  {patch['final_sub_pattern']}{RESET}")
    print(f"  {CYAN}{patch['pr_url']}{RESET}")
    print(f"  {DIM}File:{RESET} {patch['filename']}  "
          f"{GREEN}+{patch['additions']}{RESET} {RED}-{patch['deletions']}{RESET}  "
          f"{DIM}sha:{RESET} {patch['sha']}")
    print(f"{'─' * width}")


def show_llm_result(cats: list, reasoning: str):
    if not cats:
        print(f"\n  {DIM}LLM: (no categories above threshold){RESET}")
    else:
        parts = []
        for c in cats:
            loc = ""
            if c.get("start_line") and c.get("end_line"):
                loc = f"@L{c['start_line']}-L{c['end_line']}"
            parts.append(f"{YELLOW}{c['id']}:{c['name']}({c['confidence']:.2f}){loc}{RESET}")
        print(f"\n  {BOLD}LLM:{RESET} {'; '.join(parts)}")
    if reasoning:
        print(f"  {DIM}└─ {reasoning}{RESET}")


def show_category_reference():
    print(f"\n{BOLD}  18 Mutation Categories:{RESET}")
    for cid, (name, desc) in MUTATION_CATEGORIES.items():
        print(f"  {YELLOW}{cid:2d}{RESET}  {name:<30} {DIM}{desc}{RESET}")
    print()


def show_existing_verdict(review: dict):
    v = review.get("verdict", "")
    mc = review.get("manual_categories")
    if v:
        color = GREEN if v == "CORRECT" else (YELLOW if v == "PARTIAL" else RED)
        label = f"{color}{v}{RESET}"
        cats_str = ""
        if mc:
            cats_str = f" → {', '.join(str(c) for c in mc)}"
        print(f"  {DIM}[already reviewed: {label}{cats_str}]{RESET}")


# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt_categories(prompt_text: str, prefill: list = None) -> list[int]:
    """Prompt for a list of category IDs. Returns list of ints."""
    valid = set(MUTATION_CATEGORIES.keys())
    while True:
        pre = ", ".join(str(c) for c in prefill) if prefill else ""
        raw = input(f"  {prompt_text} [{pre}]: ").strip()
        if not raw and prefill is not None:
            return list(prefill)
        try:
            ids = [int(x.strip()) for x in raw.replace(",", " ").split() if x.strip()]
            invalid = [i for i in ids if i not in valid]
            if invalid:
                print(f"  {RED}Invalid IDs: {invalid}. Valid: 0-17{RESET}")
                continue
            return ids
        except ValueError:
            print(f"  {RED}Enter space- or comma-separated numbers, e.g.: 7 14{RESET}")


def prompt_notes() -> str:
    raw = input(f"  Notes (optional, Enter to skip): ").strip()
    return raw


# ── Main review loop ──────────────────────────────────────────────────────────

def run_review(patches: list[dict], progress: dict):
    total = len(patches)
    reviews = progress.setdefault("reviews", {})
    llm_cache = progress.get("llm_cache", {})

    # Start from first un-reviewed patch, or last_index
    start = progress.get("last_index", 0)
    # Find first unreviewed starting from start
    idx = start
    for i in range(start, total):
        if patch_key(patches[i]) not in reviews:
            idx = i
            break
    else:
        # All reviewed — start from 0 (allow correction)
        idx = 0

    reviewed_this_session = 0
    diff_expanded = False

    while 0 <= idx < total:
        patch = patches[idx]
        key = patch_key(patch)
        llm_data = llm_cache.get(key, {})
        cats = llm_data.get("categories", [])
        reasoning = llm_data.get("reasoning", "")
        existing = reviews.get(key)

        diff_expanded = False
        show_patch_header(patch, idx, total)

        # Show diff
        fully_shown = show_diff(patch["patch"], expanded=False)

        show_llm_result(cats, reasoning)
        if existing:
            show_existing_verdict(existing)

        done_count = len(reviews)
        print(f"\n  {DIM}Progress: {done_count}/{total} reviewed{RESET}")
        print(f"  {BOLD}[c]orrect  [p]artial  [i]ncorrect  [s]kip  [b]ack  [j N]ump  [o]pen  [?]ref  [q]uit{RESET}")

        while True:
            try:
                raw = input("  Verdict: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{YELLOW}Saving and exiting...{RESET}")
                progress["last_index"] = idx
                save_progress(progress)
                return

            if not raw:
                # Expand diff if not yet expanded
                if not diff_expanded and len(patch["patch"].splitlines()) > DIFF_EXPAND_THRESHOLD:
                    print()
                    show_diff(patch["patch"], expanded=True)
                    diff_expanded = True
                continue

            cmd = raw.split()[0]
            arg = raw[len(cmd):].strip()

            if cmd == "q":
                progress["last_index"] = idx
                save_progress(progress)
                print(f"\n{GREEN}Saved. Reviewed {reviewed_this_session} patches this session ({len(reviews)}/{total} total).{RESET}")
                return

            elif cmd == "?":
                show_category_reference()
                continue

            elif cmd == "o":
                webbrowser.open(patch["pr_url"])
                continue

            elif cmd == "b":
                idx = max(0, idx - 1)
                break

            elif cmd == "j":
                try:
                    target = int(arg) - 1  # 1-indexed input
                    if 0 <= target < total:
                        idx = target
                        break
                    else:
                        print(f"  {RED}Out of range. Enter 1-{total}.{RESET}")
                except ValueError:
                    print(f"  {RED}Usage: j NUM{RESET}")
                continue

            elif cmd == "s":
                idx += 1
                break

            elif cmd == "c":
                reviews[key] = {
                    "verdict": "CORRECT",
                    "llm_categories": [c["id"] for c in cats],
                    "manual_categories": None,
                    "notes": "",
                }
                save_progress(progress)
                reviewed_this_session += 1
                idx += 1
                break

            elif cmd == "p":
                print(f"\n  {YELLOW}PARTIAL — edit the category list.{RESET}")
                show_category_reference()
                llm_ids = [c["id"] for c in cats]
                manual_ids = prompt_categories("Final categories", prefill=llm_ids)
                notes = prompt_notes()
                reviews[key] = {
                    "verdict": "PARTIAL",
                    "llm_categories": llm_ids,
                    "manual_categories": manual_ids,
                    "notes": notes,
                }
                save_progress(progress)
                reviewed_this_session += 1
                idx += 1
                break

            elif cmd == "i":
                print(f"\n  {RED}INCORRECT — enter the correct categories.{RESET}")
                show_category_reference()
                manual_ids = prompt_categories("Correct categories", prefill=None)
                notes = prompt_notes()
                reviews[key] = {
                    "verdict": "INCORRECT",
                    "llm_categories": [c["id"] for c in cats],
                    "manual_categories": manual_ids,
                    "notes": notes,
                }
                save_progress(progress)
                reviewed_this_session += 1
                idx += 1
                break

            else:
                print(f"  {RED}Unknown command. Use: c p i s b j o ? q{RESET}")

    print(f"\n{GREEN}All patches reviewed! Run with --export to generate CSV.{RESET}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(patches: list[dict], progress: dict):
    reviews = progress.get("reviews", {})
    llm_cache = progress.get("llm_cache", {})

    def _fmt_cats(ids):
        if not ids:
            return ""
        return "; ".join(f"{i}:{MUTATION_CATEGORIES[i][0]}" for i in ids if i in MUTATION_CATEGORIES)

    fieldnames = [
        "patch_num", "pr_id", "pr_url", "final_pattern", "final_sub_pattern",
        "sha", "filename", "additions", "deletions",
        "llm_categories", "llm_reasoning",
        "verdict", "manual_categories", "notes",
    ]

    rows = []
    for i, patch in enumerate(patches):
        key = patch_key(patch)
        llm_data = llm_cache.get(key, {})
        review = reviews.get(key, {})
        llm_ids = [c["id"] for c in llm_data.get("categories", [])]
        manual_ids = review.get("manual_categories")
        final_ids = manual_ids if manual_ids is not None else llm_ids
        rows.append({
            "patch_num": i + 1,
            "pr_id": patch["pr_id"],
            "pr_url": patch["pr_url"],
            "final_pattern": patch["final_pattern"],
            "final_sub_pattern": patch["final_sub_pattern"],
            "sha": patch["sha"],
            "filename": patch["filename"],
            "additions": patch["additions"],
            "deletions": patch["deletions"],
            "llm_categories": _fmt_cats(llm_ids),
            "llm_reasoning": llm_data.get("reasoning", ""),
            "verdict": review.get("verdict", ""),
            "manual_categories": _fmt_cats(final_ids),
            "notes": review.get("notes", ""),
        })

    with open(EXPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed = sum(1 for r in rows if r["verdict"])
    print(f"{GREEN}Exported {len(rows)} rows ({reviewed} reviewed) to {EXPORT_CSV}{RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    export_mode = "--export" in sys.argv

    print(f"{BOLD}Loading patches...{RESET}")
    patches = load_patches()
    print(f"Loaded {len(patches)} source patches across {len(set(p['pr_id'] for p in patches))} PRs.")

    progress = load_progress()

    if export_mode:
        export_csv(patches, progress)
        return

    progress = ensure_llm_classifications(patches, progress)
    run_review(patches, progress)


if __name__ == "__main__":
    main()
