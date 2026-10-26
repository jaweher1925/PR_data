# Thesis_Data_Analysis

```
Thesis_Data_Analysis/
├── code-data_Analysis/          ← the code (open this in VS Code)
│   ├── common.py                  helpers used by every notebook (vertical charts)
│   ├── requirements.txt           pip install -r requirements.txt
│   ├── 00_AIDev/ … 09_Khemissi_References/
│   │   └── analysis.ipynb         one notebook per paper: first look, per agent, charts, final figure, fixes
│   │   └── charts/  tables/       results of the notebook
│   ├── 10_Build_AI_PRs_Dataset/   builds the list of AI-generated PRs (input of linker.py) → data/AI_PRS_DATASET/
│   ├── 11_Agent_Report/           one row per dataset: latest date, size, PRs per agent, how the agent is involved, Python
│   ├── 99_Fixes_All_Papers/       merges the fixes found by all papers → data/FINAL_DATASET/ (run after 01–09)
│   └── tools/                     miner.py (GitHub mining), build_clean_pr_dataset.py, add_dates.py, clone_all.py
├── data/                        ← the data (one folder per paper, same names as the code folders)
│   ├── 01_WhoFinishes/ … 09_Khemissi_References/   (big files are .gz – the notebooks read them directly)
│   ├── AI_PRS_DATASET/            34,352 AI PRs (AIDev-pop + Watanabe) with agent, language, task type, fixed later
│   ├── ENRICH/                    data mined from public GitHub: agents for Xia & Miller, dates, code patches
│   ├── CLEAN_PR_DATASET/          323 clean AI PR → fix PR pairs (+ code patches)
│   └── FINAL_DATASET/             9,619 fixes of AI code from all papers (5,679 with code)
└── repos/                       the papers' code (clone_all.py)
```

## The 3 summary notebooks

| Folder | Question it answers | 1 row = | Output |
|---|---|---|---|
| 10_Build_AI_PRs_Dataset | Which AI-generated PRs do we study? | 1 AI PR | data/AI_PRS_DATASET/ (input of linker.py) |
| 11_Agent_Report | For each dataset: latest date, size, agents, how the agent is involved, Python? | 1 dataset | tables/Agent_report.xlsx |
| 99_Fixes_All_Papers | Which fixes of AI code did the papers already find? | 1 fix | data/FINAL_DATASET/ (to check linker.py) |

## How to run
1. `cd code-data_Analysis` → `pip install -r requirements.txt`
2. Open `<folder>/analysis.ipynb` → **Run All**.
3. `00_AIDev`, `10_…`, `11_…`: set `RAW` to your AIDev folder (e.g. evolution-assurance/data/raw).

| Folder | Paper | What we take |
|---|---|---|
| 01_WhoFinishes | Takerngsaksiri 2026 | linker.py + judge, 334 PR → fix PR (323 clean, with code) |
| 02_XiaMiller_PostMerge | Xia & Miller 2026 | 7,822 fixed AI commits (agents mined), maintenance classes (Q1) |
| 03_Sawada | Sawada 2026 | 216 fix commits on AI files |
| 04_Watanabe_ClaudeCode | Watanabe 2025 | 797 Claude Code PRs, 102 fix commits |
| 05_Yu_ChatGPT_Copilot | Yu 2024 | 113 files with fixes (baseline) |
| 06_Dovhoshliubnyi_Hunks | Dovhoshliubnyi 2026 | code-change taxonomy (Q3) |
| 07_Wang_LLM_Code_Errors | Wang 2025 | error taxonomy (Q2) |
| 08_Alam_AI_Fix_PRs | Alam 2026 | AIDev-pop with pr_task_type |
| 09_Khemissi_References | Khemissi 2026 | 1,032 fixes of agent PRs (937 with code) |
