"""
step2_task_type_v5.py - rebuild the PR task type (fix, feat, refactor...) for v5.

AIDev v5 has no pr_task_type. We rebuild it like this:
  1. if the title has a Conventional-Commits prefix ("fix: ...", "feat(api): ...") -> that type
     (on v3 this agrees 100% with AIDev's own labels);
  2. otherwise a classifier on the title, trained on the 40,214 v3 labels (agent + human PRs).
The classifier is first tested on 20% of v3 that it did not see (accuracy is printed).

Run:  uv run --project %EA% python step2_task_type_v5.py      (~5 min)
Output: raw_v5/pr_task_type_v5.parquet  (id, type, confidence, source)
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from paths import RAW_V3, RAW_V5

PREFIX = re.compile(r"^\s*\[?(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)\]?(\([^)]*\))?!?\s*:")


def title_prefix(title):
    t = str(title).lower()
    if t.startswith("revert"):
        return "revert"
    m = PREFIX.match(t)
    return m.group(1) if m else None


# 1. v3 labels (agent + human PRs)
lab = pd.concat([pd.read_parquet(os.path.join(RAW_V3, f), columns=["id", "title", "type"])
                 for f in ("pr_task_type.parquet", "human_pr_task_type.parquet")
                 if os.path.exists(os.path.join(RAW_V3, f))], ignore_index=True)
lab = lab.dropna(subset=["title", "type"])
print(f"v3 labels: {len(lab):,} titles, types: {lab.type.value_counts().to_dict()}")

def model():
    return make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        LogisticRegression(max_iter=2000, C=4))

# 2. test on v3 (20% held out)
tr, te = train_test_split(lab, test_size=0.2, random_state=0, stratify=lab.type)
clf = model().fit(tr.title, tr.type)
pred = clf.predict(te.title)
pref = te.title.map(title_prefix)
final = pref.fillna(pd.Series(pred, index=te.index))
for name, p in (("classifier only", pred), ("prefix + classifier (used)", final)):
    r = classification_report(te.type, p, output_dict=True, zero_division=0)
    print(f"  test on v3 - {name:27s}: accuracy {r['accuracy']:.1%} | 'fix' precision "
          f"{r['fix']['precision']:.1%}, recall {r['fix']['recall']:.1%}")

# 3. train on all v3, label v5
clf = model().fit(lab.title, lab.type)
v5 = pd.read_parquet(os.path.join(RAW_V5, "pull_request.parquet"), columns=["id", "title"])
proba = clf.predict_proba(v5.title.fillna(""))
v5["type"] = clf.classes_[proba.argmax(1)]
v5["confidence"] = proba.max(1).round(3)
v5["source"] = "classifier"
p = v5.title.map(title_prefix)
v5.loc[p.notna(), "type"] = p[p.notna()]
v5.loc[p.notna(), "confidence"] = 1.0
v5.loc[p.notna(), "source"] = "title_prefix"
out = os.path.join(RAW_V5, "pr_task_type_v5.parquet")
v5[["id", "type", "confidence", "source"]].to_parquet(out, index=False)
print(f"\nv5 PRs labelled: {len(v5):,} ({(v5.source == 'title_prefix').mean():.0%} from the title prefix)")
print(v5.type.value_counts().to_string())
print(f"\nSaved {out}\nNext: step3_build_lake_v5.py")
