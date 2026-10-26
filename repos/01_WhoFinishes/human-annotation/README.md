# Checking AI-agent follow-up fixes

## 1. Why we need this

When an AI coding agent opens a pull request (PR) and it gets merged, sometimes a
*later* PR shows up to fix it. We found these later "fix" PRs automatically — by looking for a PR that is **in the same repository**, **merged within 30 days** of the original, **changes at least one of the same files**, and is **labelled a fix** (or titled "Revert"). But a
computer can't be sure the later PR **really fixes the first one** (it might just touch
the same file for an unrelated reason).

Your job: look at **50 example pairs** and judge by eye whether the second PR genuinely
fixes the first. This tells us how trustworthy our automatic detection is.

**The one question to keep in mind for every pair:** *does PR&nbsp;Q fix PR&nbsp;P?*
(not "is Q a good PR?")

## 2. What's in this folder

```
human-annotation/
├── sample_pairs.py    # `gold`: draw+enrich the 50-pair set from the DB; `judge`: blind human-vs-judge draw from a judge CSV
├── gen_labeler.py     # render an offline HTML labeler (human sheets + human-vs-judge sheet)
├── score.py           # score the exported labels (humans vs humans, or human vs LLM judge)
├── rq1_verified_rates.py  # every verified-fix number in the paper (RQ1.2): rates (gold-backfilled) + kappa table
├── consolidation_gate.py # the Revise-and-Repair ledger gate: recomputes + ASSERTS every inline-verified number
├── data/              # per-pipeline artifacts, one folder per mode
│   ├── net_diff/      #   the paper's linker: master.csv, pairs_enriched.json, labeler.html
│   ├── judge/         #   human-vs-LLM-judge labelers (rq1_judge_sample{N}_{mode}[_{tag}]_labeler.html)
│   └── rq1_pairs_diff_snapshot.json   # local P/Q diff cache (gitignored, regenerable)
└── labels/            # the labels people (and the judge) produce, one folder per mode
    ├── humans/        #   exported annotator CSVs, agreed labels, blank template/
    └── judge/         #   per sample: blind sheet, exported annotator sheet,
                       #   model-key CSV, and _sample_meta.json (reweighting meta)
```

(`labels/` is the human-produced counterpart to the generated `data/`; derived
tables and figures land in `data/digests/` and `figures/` instead.)

The blank sheets in `labels/humans/template/` carry the wider sampling-stage
columns (`p_title`, `q_type`, `shared_files`, …), while the HTML labeler exports
the narrower `pair_id,agent,p_url,q_url,label,reason,annotator` schema — `score.py`
accepts both (it only needs `pair_id` and `label`).

## 3. How to label

1. **Open** the labeler sheet — just double-click `data/net_diff/labeler.html`; it
   opens in your browser (works offline, nothing to install). It shows the shared
   files' diffs inline, so you can judge without leaving the page.
2. **Type your name** in the box at the top right.
3. For **each of the 50 cards**:
   - Click the two links — **P** (the original PR) and **Q** (the later "fix"). On each,
     look at the **"Files changed"** tab, focusing on the *shared files* listed on the card.
   - Pick one button:
     - **Direct fix** — Q clearly repairs something P added or changed.
     - **Related touch** — Q changes the same area, but you can't be sure it fixes P.
     - **Unrelated / mislabeled** — same file, different purpose, or Q isn't really a fix.
     - **Indeterminate** — not enough to tell.
   - Optionally type a few words on why.
4. Your answers **save automatically** as you go. When the bar reads 50/50, click
   **Export CSV** and save the file into the **`labels/humans/`** folder next to this README.

If two or more people are labeling, **each person does all 50 on their own, without
discussing** — that lets us measure how much you agree.

## 4. How to score

Once everyone's exported CSV is in `labels/humans/`, run this from the repository root:

```
uv run python human-annotation/score.py humans
```

It auto-discovers every `rq1_followup_validation_annotator*.csv` in `labels/humans/`,
checks the sheets against `data/net_diff/master.csv` (the only labelled set), and reports:
- **inter-rater agreement** — Fleiss' κ (for any N ≥ 2 raters; plus Cohen's κ when exactly 2);
- **precision** of the detector on the **majority-vote** label — *strict* (Direct fix only) and
  *lenient* (Direct + Related), each with **Wilson 95% CIs**, reported **overall and split by
  high-/low-human-touch** (the bias check behind the +2.1 pp human-touch result);
- any pairs with **no majority**, listed for manual adjudication.

Only pairs labelled by *all* raters are scored.

## 4b. How to score a human against the LLM judge

The step above compares **two humans**. To compare **one human against the LLM
judge** on the `rq1_judge_sample{N}_{mode}` sheets (drawn by
`sample_pairs.py judge`, labelled via `data/judge/…_labeler.html`), use
the `judge` subcommand. From the repository root:

```
uv run python human-annotation/score.py judge
```

With no arguments it auto-discovers every sample set in `labels/judge/` and scores each
filled `…annotator*.csv` against that set's `…_model_key.csv`. To point it at
specific files instead:

```
uv run python human-annotation/score.py judge \
    -a labels/judge/rq1_judge_sample50_enriched_annotator_anon_NHATDUONG.csv \
    -k labels/judge/rq1_judge_sample50_enriched_model_key.csv
```

`-a` takes one or more human sheets (a blind sheet with the `label` column filled
in); `-k` is that sample's model key. The judge label is joined from the model key
by `pair_id`, so the scorer — not the labeler's exported `agree` column — is the
source of truth. For each annotator it reports:
- **pairs labelled**, and how many you **agreed** / **disagreed** with the LLM;
- the **agreement rate** (%);
- the **raw binary κ** — Cohen's κ on Direct fix vs not, over the enriched draw;
- the **LLM Direct-fix accuracy** — of the pairs the LLM called *Direct fix*, how
  many you confirmed, as a count and a %;
- the list of **disagreeing pairs**, with your label vs the LLM's for each.

## 4c. How to regenerate the paper's verified-fix numbers (paper RQ1.2)

Every verified-fix figure in the paper (verified fix rates, the kappa table, the
human-touch re-run) comes from one deterministic script; its name predates the
paper's RQ numbering. From the repository root:

```
uv run python human-annotation/rq1_verified_rates.py
```

It joins the population judge CSV onto the candidate-fix (paper RQ1.1)
uncensored frame (asserting
verified ⊆ linker-followed), backfills the 50 gold pairs with their human
consensus verdicts (the judge never scores them), and writes
`data/digests/rq1_verified_rates.csv` + `rq1_kappa_table.csv` (gitignored,
regenerable).

It also hard-fails if the rates, adjusted rates or precision bands it computes
drift from the paper's Table rq3-verified as typeset, so those numbers cannot
change without the script refusing to run. The pooled spot-check precision the
bands scale by is pinned separately — and without the lake — by
`tests/test_pinned_ledger_rows.py`.

## 4d. How to re-check the Revise-and-Repair ledger numbers

The paper's headline rows that were verified inline (the
agent-vs-human contrasts, fixer identity, and the RQ4 visibility probes) are
recomputed and asserted by one script. From the repository root:

```
uv run python human-annotation/consolidation_gate.py
```

It hard-fails on drift and writes `data/digests/consolidation_gate.csv`, plus
the *committed* pins it regenerates — re-commit those alongside any change to
the rows they back. The script's module docstring owns the frame/cut-off
design it enforces.

## 5. How to reproduce the examples

Only needed to regenerate a 50-pair sample from scratch (sampling requires the
project database; enrichment requires `gh`). From the repository root:

```
# the paper's (net-diff) set: data/net_diff/{master.csv, pairs_enriched.json, labeler.html}
uv run python human-annotation/sample_pairs.py gold --linker net-diff
uv run python human-annotation/gen_labeler.py --mode net-diff
```

(`sample_pairs.py gold --stage sample|enrich` runs a single stage.) To draw and
rebuild a human-vs-judge labeler (the sample is drawn by `sample_pairs.py judge`
over a population judge CSV):

```
uv run python human-annotation/sample_pairs.py judge --mode enriched --n 50 \
    --pop-csv data/judge/rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv
uv run python human-annotation/gen_labeler.py --mode judge --n 50 --sample-mode enriched
```

A second sample over a *different* judged population (e.g. the human-P incidence
baseline) needs `--tag` on both commands so its stem — and therefore its blind
sheet, model key and meta — never collides with an earlier draw. `score.py judge`
keys off the same stems, and skips a sheet that belongs to a longer stem with its
own model key, so the two sets are never scored against each other's key:

```
uv run python human-annotation/sample_pairs.py judge --mode enriched --n 50 --tag humanP \
    --pop-csv data/judge/rq1_llm_judge_pop_humanP_anthropic_claude-opus-4-8_low.csv
uv run python human-annotation/gen_labeler.py --mode judge --n 50 --sample-mode enriched --tag humanP \
    --cases data/judge/rq1_judge_cases_population_12k_humanP.json
```

`gen_labeler.py --mode judge` also reads the population cases file
`data/judge/rq1_judge_cases_population_12k.json` (the net-diff patches the judge
saw). `--tag` does **not** switch it — a non-agent sample must name its own cases
file with `--cases`, as above. Both cases files are committed,
so a fresh checkout has them; regenerate one only deliberately (the shipped
`judge.yaml` pins `auto_build: false`, so `run_judge` errors on a missing cases
file rather than rebuilding it; rebuild explicitly with
`uv run python -m prfix.rq1_fix_rates.judge.build_cases --mode population --judgeable-only
[--p-side human]`).
