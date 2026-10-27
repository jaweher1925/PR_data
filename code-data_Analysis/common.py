"""
common.py - small helpers used by every paper notebook.
You don't need to change anything here.
"""
import os, re, subprocess, json, gzip
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 60)

BLUE, ORANGE, GREY, LIGHT = "#1F4E79", "#C55A11", "#A6A6A6", "#D9E2F3"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})
# data of this notebook = Thesis_Data_Analysis/data/<this folder name>
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.join(DATA_ROOT, os.path.basename(os.getcwd()))
os.makedirs("charts", exist_ok=True)
os.makedirs("tables", exist_ok=True)


# ---------- get the data ----------
def get_repo(url, name):
    """Use the paper's data in ../../data/<paper>/<name>; download the GitHub repository there if it is missing."""
    target = os.path.join(DATA_DIR, name)
    if os.path.exists(target):
        print("Data folder:", target)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, target], check=True)
        print("Downloaded:", target)
    return target


def list_files(folder, ext=(".csv", ".json", ".jsonl", ".xlsx", ".parquet", ".gz")):
    """All data files in the repository, biggest first."""
    rows = []
    for root, _, files in os.walk(folder, followlinks=True):
        if ".git" in root:
            continue
        for f in files:
            if f.lower().endswith(ext):
                p = os.path.join(root, f)
                rows.append({"file": os.path.relpath(p, folder), "size_MB": round(os.path.getsize(p) / 1e6, 2)})
    return pd.DataFrame(rows).sort_values("size_MB", ascending=False).reset_index(drop=True)


def pick(path):
    """Return the file, or its compressed version (.gz) if that is what we have."""
    return path if os.path.exists(path) else path + ".gz"


def load_json(path):
    path = pick(path)
    with (gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------- first look ----------
def overview(df, name, n=5):
    """Size, some lines, and ALL the columns of a table."""
    display(Markdown(f"**{name}** — **{len(df):,} rows × {df.shape[1]} columns** "
                     f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory)"))
    display(Markdown(f"*First {n} lines:*"))
    display(df.head(n))
    cols = pd.DataFrame({
        "column": df.columns,
        "type": df.dtypes.astype(str).values,
        "missing_%": (df.isna().mean() * 100).round(1).values,
        "unique": [df[c].astype(str).nunique() for c in df.columns],
        "example": [str(df[c].dropna().iloc[0])[:50] if df[c].notna().any() else "" for c in df.columns],
    })
    display(Markdown("*All columns:*"))
    display(cols)
    return cols


# ---------- agents ----------
def agent_of(text):
    """Turn any author / tool text into a clean agent name."""
    t = str(text).lower()
    for key, name in [("claude", "Claude Code"), ("codex", "Codex"), ("openai", "Codex"), ("devin", "Devin"),
                      ("copilot", "Copilot"), ("cursor", "Cursor"), ("jules", "Jules"), ("gpt-4", "GPT-4"),
                      ("gpt4", "GPT-4"), ("gpt-3", "GPT-3"), ("gpt3", "GPT-3"), ("chatgpt", "ChatGPT")]:
        if key in t:
            return name
    return "Human / other"


# ---------- tables ----------
def save_table(df, name):
    df.to_csv(os.path.join("tables", f"{name}.csv"), index=False)
    return df


def summary_card(paper, unit, rows, repos, first, last, agents, involvement, fixes):
    """One line that answers Mohamed's questions for this paper."""
    card = pd.DataFrame([{"paper": paper, "unit (1 row =)": unit, "rows": rows, "repos": repos,
                          "first date": first, "latest date": last, "agents": agents,
                          "how the agent is involved": involvement, "fixes of AI code": fixes}])
    save_table(card, "summary_card")
    display(card.T.rename(columns={0: ""}))
    return card


# ---------- charts ----------
def save_chart(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join("charts", fname), dpi=150, bbox_inches="tight")
    plt.show()


def bar(counts, title, fname, xlabel="", color=BLUE, horizontal=False, pct=False):
    """Vertical bar chart with the number written on each bar."""
    counts = counts.copy()
    horizontal = False                      # all charts are vertical
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(counts) + 2), 4.2))
    total = counts.sum()
    if horizontal:
        counts = counts[::-1]
        ax.barh(counts.index.astype(str), counts.values, color=color)
        for i, v in enumerate(counts.values):
            ax.text(v, i, f" {v:,}" + (f" ({v / total:.0%})" if pct else ""), va="center", fontsize=9)
        ax.set_xlabel(xlabel)
    else:
        ax.bar(counts.index.astype(str), counts.values, color=color)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}" + (f"\n({v / total:.0%})" if pct else ""), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(xlabel)
        plt.xticks(rotation=35 if len(counts) > 5 else 15, ha="right", fontsize=8)
    ax.set_title(title)
    save_chart(fig, fname)
    show_numbers(counts.sort_values(ascending=False) if not counts.index.is_monotonic_increasing else counts, title)


def show_numbers(data, title=""):
    """Print the exact numbers behind a chart (count + %)."""
    if isinstance(data, pd.Series):
        t = data.rename("number").to_frame()
        if pd.api.types.is_numeric_dtype(t["number"]) and t["number"].sum() > 0:
            t["%"] = (100 * t["number"] / t["number"].sum()).round(1)
        t.loc["TOTAL"] = [t["number"].sum()] + ([100.0] if "%" in t else [])
    else:
        t = data.copy()
        if all(pd.api.types.is_numeric_dtype(t[c]) for c in t.columns):
            t["TOTAL"] = t.sum(axis=1)
            t.loc["TOTAL"] = t.sum()
    display(Markdown(f"*Exact numbers – {title}*"))
    display(t.round(1))
    return t


def stacked(table, title, fname, colors=None, horizontal=True, xlabel="count"):
    """Vertical stacked bars, e.g. fixes per agent split by who fixed them."""
    horizontal = False                      # all charts are vertical
    table = table.iloc[::-1] if len(table) > 1 else table
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(table) + 2.5), 4.2))
    table.plot(kind="barh" if horizontal else "bar", stacked=True, ax=ax,
               color=colors or [BLUE, ORANGE, GREY, LIGHT, "#70AD47"][: table.shape[1]])
    ax.set_title(title)
    ax.set_ylabel(xlabel); ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=30 if len(table) > 4 else 0, ha="right" if len(table) > 4 else "center", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_chart(fig, fname)
    show_numbers(table.iloc[::-1] if len(table) > 1 else table, title)


def hist(values, title, fname, xlabel, bins=30, color=BLUE):
    values = pd.Series(values).dropna()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(values, bins=bins, color=color)
    ax.axvline(values.median(), color=ORANGE, ls="--")
    ax.text(values.median(), ax.get_ylim()[1] * 0.92, f"  median = {values.median():.1f}", color=ORANGE, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    save_chart(fig, fname)
    st = values.describe(percentiles=[.25, .5, .75]).rename({"50%": "median"}).round(2)
    display(Markdown(f"*Exact numbers – {title}*")); display(st.to_frame(xlabel).T)


# ---------- final figure (the 4 charts on one page) ----------
def final_figure(n, title, cap1, cap2):
    """Put charts 1-4 of this paper on one page (2 rows), with captions. Saved as charts/Figure_<n>.png"""
    import glob
    import numpy as np
    from PIL import Image
    files = sorted(glob.glob(os.path.join("charts", "[1-4]_*.png")))[:4]
    def row(a, b, H=600):
        ims = []
        for p in (a, b):
            im = Image.open(p).convert("RGB")
            ims.append(im.resize((int(im.width * H / im.height), H)) if im.width > im.height
                       else im.resize((1000, int(im.height * 1000 / im.width))))
        H = max(i.height for i in ims)
        out = Image.new("RGB", (sum(i.width for i in ims) + 60, H), "white")
        x = 20
        for im in ims:
            out.paste(im, (x, (H - im.height) // 2)); x += im.width + 20
        return np.asarray(out)
    r1, r2 = row(files[0], files[1]), row(files[2], files[3])
    fig, axs = plt.subplots(2, 1, figsize=(16, 16 * (r1.shape[0] + r2.shape[0]) / max(r1.shape[1], r2.shape[1]) + 1.6),
                            gridspec_kw={"height_ratios": [r1.shape[0], r2.shape[0]]})
    for ax, im, cap in [(axs[0], r1, f"Figure {n}.1 — {cap1}"), (axs[1], r2, f"Figure {n}.2 — {cap2}")]:
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlabel(cap, fontsize=13, style="italic", color="#404040", labelpad=10)
    fig.suptitle(title, fontsize=18, fontweight="bold", color=BLUE)
    save_chart(fig, f"Figure_{n}.png")


# ---------- fixes in one common format (to merge all papers) ----------
FIX_COLS = ["paper", "fix_id", "agent", "repo", "language", "level", "ai_change", "fix_change",
            "fixed_by", "days_to_fix", "fix_label", "fix_reason", "has_code_patch"]
EXT = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript",
       ".go": "Go", ".rs": "Rust", ".java": "Java", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
       ".cpp": "C++", ".c": "C", ".kt": "Kotlin", ".swift": "Swift"}


def lang_of(files):
    """Main programming language of one file name or a list of file names."""
    files = [files] if isinstance(files, str) else list(files)
    langs = [EXT.get(os.path.splitext(str(f))[1].lower()) for f in files]
    langs = [l for l in langs if l]
    return pd.Series(langs).mode()[0] if langs else "Other"


def save_fixes(df, paper):
    """Save this paper's fixes of AI code with the SAME columns for every paper -> tables/fixes.csv"""
    df = df.copy()
    df["paper"] = paper
    df["fix_id"] = [f"{paper.split()[0][:3].upper()}-{i:05d}" for i in range(len(df))]
    df = df.reindex(columns=FIX_COLS)
    save_table(df, "fixes")
    display(Markdown(f"**{len(df):,} fixes saved to `tables/fixes.csv`** (same columns in every paper)"))
    display(df.head())
    return df
