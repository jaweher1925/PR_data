# Thesis_Data_Analysis

```
Thesis_Data_Analysis/
├── code-data_Analysis/          ← the code (open this in VS Code)
│   ├── common.py                  helpers used by every notebook (vertical charts)
│   ├── requirements.txt           pip install -r requirements.txt
│   ├── 00_AIDev/ … 09_Khemissi_References/
│   │   └── analysis.ipynb         1 first look · 2 data science (per agent) · 3 four charts · 4 final figure · 5 fixes
│   │   └── charts/  tables/       results of the notebook
│   ├── 99_ALL_PAPERS/analysis.ipynb   all papers together + FINAL DATASET (run last)
│   └── tools/                     miner.py (GitHub mining), merge_enrich.py, build_clean_pr_dataset.py, add_dates.py, clone_all.py
├── data/                        ← the data (one folder per paper, same names as the code folders)
│   ├── 01_WhoFinishes/ … 09_Khemissi_References/   (big files are .gz – the notebooks read them directly)
│   ├── ENRICH/                    data mined from public GitHub: agents for Xia & Miller, dates, code patches (patches_*.parquet)
│   ├── CLEAN_PR_DATASET/          323 clean AI PR → fix PR pairs (+ code patches)
│   └── FINAL_DATASET/             FINAL_AI_FIXES_DATASET.xlsx / .csv – 9,619 fixes of AI code (5,679 with code)
└── reports/                     Dataset_Report_per_Paper.docx, Code_Structure_per_Paper.docx, figures/
```

## How to run
1. `cd code-data_Analysis` → `pip install -r requirements.txt`
2. Open `<paper>/analysis.ipynb` → **Run All**. Each notebook reads its data from `../../data/<paper>/`
   (if a folder is missing, the notebook downloads the paper's GitHub repository there).
3. Run `99_ALL_PAPERS/analysis.ipynb` last → writes `data/FINAL_DATASET/`.
4. `00_AIDev`: set `RAW` at the top to your AIDev folder (e.g. evolution-assurance/data/raw).

| Folder | Paper | What we take |
|---|---|---|
| 01_WhoFinishes | Takerngsaksiri 2026 | linker.py + judge, 334 PR → fix PR (323 clean, with code) |
| 02_XiaMiller_PostMerge | Xia & Miller 2026 | 7,822 fixed AI commits (agents mined), maintenance classes (Q1) |
| 03_Sawada | Sawada 2026 | 216 fix commits on AI files |
| 04_Watanabe_ClaudeCode | Watanabe 2025 | 102 fix commits inside Claude Code PRs |
| 05_Yu_ChatGPT_Copilot | Yu 2024 | 113 files with fixes (baseline) |
| 06_Dovhoshliubnyi_Hunks | Dovhoshliubnyi 2026 | code-change taxonomy (Q3) |
| 07_Wang_LLM_Code_Errors | Wang 2025 | error taxonomy (Q2) |
| 08_Alam_AI_Fix_PRs | Alam 2026 | AIDev-pop with pr_task_type (for linker.py) |
| 09_Khemissi_References | Khemissi 2026 | 1,032 fixes of agent PRs (937 with code) |
