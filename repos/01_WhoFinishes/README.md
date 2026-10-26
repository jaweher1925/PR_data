# Who Finishes the Job? — Replication Package

Replication package for the paper: [**Who Finishes the Job? A Study of Follow-Up Fixes and Commit Authorship on AI Coding Agent Pull Requests.**](https://arxiv.org/abs/2609.26847)

The study follows 6,774 merged AI-agent pull requests across five agents (Codex, Devin, GitHub Copilot, Cursor, Claude Code) from the AIDev-pop dataset into their follow-up fixes, against a baseline of 5,044 merged human PRs in the same repositories.

## Repo Structure

The package is organised by research question:

```
├── src/prfix/                     
│   ├── config.py                  
│   ├── common/
│   │   ├── loaders/                 // AIDev and human-PR enrichment (GitHub REST)
│   │   ├── metrics.py             
│   │   └── effect_sizes.py        
│   ├── rq1_fix_rates/               RQ1 — how often are agent merges fixed?
│   │   ├── linker.py                // the four heuristics of candidate fixes
│   │   ├── boilerplate.py           
│   │   ├── judge/                   // the LLM judge for verified fixes
│   │   └── fig_survival.py        
│   ├── rq2_fix_authors/             RQ2 — who authors the verified fix PRs?
│   │   └── fixer.py               
│   ├── rq3_commit_authors/          RQ3 — who authors the commits?
│   │   ├── classifier.py, rules.py  // the commit author-class rules (paper App. B)
│   │   └── commit_tables.py       
│   └── rq4_signals/                 RQ4 — merge-time signals
│       ├── per_pr_signals.py        // the four signals per merged PR
│       ├── fig_violins.py         
│       └── fig_forest.py     
|     
├── human-annotation/                annotation kits, labels, scorer, gates
│   ├── labels/humans/               // the two annotators' 50-pair validation + agreed
│   ├── labels/judge/                // the blind judge spot-checks, both cohorts (50+50)
│   ├── gen_labeler.py             
│   ├── sample_pairs.py            
│   ├── score.py                   
│   ├── rq1_verified_rates.py      
│   └── consolidation_gate.py      
├── data/
│   ├── judge/                     
│   └── digests/                  
└── tests/                         
```

## RQ → paper artifact map

| RQ | Paper artifact | Code | Pinned data |
|---|---|---|---|
| RQ1 | per-cohort verified rate table  | `human-annotation/rq1_verified_rates.py` | `data/digests/rq1_survival_frame.csv` |
| RQ1 | cross-cohort table | `prfix.common.metrics` | `data/digests/rq1_crosscohort_strata.csv` |
| RQ1 | PR survival figure | `prfix.rq1_fix_rates.fig_survival` | `data/digests/rq1_survival_frame.csv` |
| RQ1 | verification (judge) | `prfix.rq1_fix_rates.judge` | `data/judge/*` |
| RQ2 | fix-PR author table | `prfix.rq2_fix_authors.fixer` | `data/digests/rq2_fixer_frame.csv` |
| RQ3 | commit author-class tables | `prfix.rq3_commit_authors` | `data/digests/rq3_commit_classes.csv` |
| RQ4 | signal table | `prfix.rq4_signals` | `data/digests/rq4_signal_stats.csv` |

## Two ways to replicate results

Copy `.env.example` to `.env` and fill in only what the step you run needs.
Reproduction path A needs no credentials.

### A. Dry-run: Verify the paper's numbers (no downloads, no API keys)

```bash
uv sync --extra dev
uv run pytest
```

The test suite re-derives the headline numbers from the pinned digests and
asserts the exact values the paper typesets.

Reproduce the annotation agreement numbers:

```bash
uv run python human-annotation/score.py humans   # human–human agreement (round 1)
uv run python human-annotation/score.py judge    # human-vs-judge agreement + precision
```

`humans` scores the two independent annotator sheets on the 50-pair detector
set; `judge` scores both blind spot-checks and reports the judge's Direct-fix
precision as 27/30 = 90.0% on each (54/60 pooled). 

### B. Reproduce: Rebuild the pipeline from source data

1. **Ingest AIDev** (Hugging Face `hao-li/AIDev`, the June 22, 2025 snapshot
   used by the paper) into a local DuckDB lake:
   ```bash
   uv run python -m prfix.common.loaders.aidev
   ```
2. **Enrich the human cohort** — AIDev's `human_pull_request` table is
   PR-level only, so commits, diffs, and review events are re-fetched per PR
   (`GITHUB_TOKEN` required, see `.env.example`):
   ```bash
   uv run python -m prfix.common.loaders.human_enrich
   ```
3. **Net diffs** for the shared-file linker filter:
   ```bash
   uv run python -m prfix.common.loaders.fetch_netdiff
   uv run python -m prfix.common.loaders.build_net_files
   ```
4. **LLM judge** (RQ1 verification). The runner rebuilds the verdict CSVs:
   ```bash
   # agent cohort -> data/judge/rq1_llm_judge_pop_anthropic_claude-opus-4-8_low.csv
   uv run python -m prfix.rq1_fix_rates.judge.run_judge \
       --config src/prfix/rq1_fix_rates/judge/judge.yaml --consolidate-only
   # human-P incidence baseline -> ..._pop_humanP_....csv
   uv run python -m prfix.rq1_fix_rates.judge.run_judge \
       --config src/prfix/rq1_fix_rates/judge/judge.yaml --consolidate-only \
       --p-side human
   ```
   (A fresh live run needs `ANTHROPIC_API_KEY`; the judge is Claude Opus 4.8
   at temperature 0 — prompt in `src/prfix/rq1_fix_rates/judge/`.)
5. **Consolidation gate** — recompute and assert every headline number:
   ```bash
   uv run python human-annotation/consolidation_gate.py
   ```
6. **Figures** (written to `figures/`):
   ```bash
   uv run python -m prfix.rq1_fix_rates.fig_survival      # KM cumulative incidence
   uv run python -m prfix.rq4_signals.fig_violins         # split violins
   uv run python -m prfix.rq4_signals.fig_forest          # within-repository forest
   uv run python -m prfix.rq3_commit_authors.commit_tables # commit author-class tables
   ```

## Reference

If you use our material or results, please consider cite us:

<pre><code>@article{takerngsaksiri2026finishes,
  title={Who Finishes the Job? A Study of Follow-Up Fixes and Commit Authorship on AI Coding Agent Pull Requests},
  author={Takerngsaksiri, Wannita and Duong, Nhat and Barnett, Scott},
  journal={arXiv preprint arXiv:2609.26847},
  year={2026}
}</code></pre>




