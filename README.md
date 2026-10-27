# Thesis_Data_Analysis

```
Thesis_Data_Analysis/
├── code-data_Analysis/          ← the code (open this in VS Code)
│   ├── common.py                  helpers used by every notebook
│   ├── requirements.txt           pip install -r requirements.txt
│   ├── 1_Papers/                  one notebook per paper (00_AIDev … 09_Khemissi_References)
│   ├── 2_AI_PRs_Dataset/          the AI-generated PRs (input of linker.py)
│   │   ├── 10_Build_AI_PRs_Dataset   AIDev-pop + Watanabe + paper labels  → data/AI_PRS_DATASET/
│   │   ├── 11_Agent_Report           per dataset: latest date, size, agents, Python
│   │   └── 13_Clean_AI_PRs_Dataset   CLEAN dataset: AIDev v4 + AIDev-pop + Watanabe (2.74M) → data/AI_PRS_CLEAN/
│   ├── 3_Fixes_Dataset/           the fixes the papers already found (to check linker.py)
│   │   └── 99_Fixes_All_Papers       9,619 fixes → data/FINAL_DATASET/
│   ├── 4_Linker_Test/             run Mohamed's linker.py and look at the results
│   │   └── 12_Run_Linker             step1 … step5 (see its README)
│   └── tools/                     miner.py, build_clean_pr_dataset.py, add_dates.py, clone_all.py
├── data/                        ← all the data (one folder per paper + the datasets we built)
└── repos/                       the papers' code (clone_all.py)
```

## The 3 summary notebooks

| Folder | Question it answers | 1 row = | Output |
|---|---|---|---|
| 2_AI_PRs_Dataset/10_Build_AI_PRs_Dataset | Which AI-generated PRs do we study? | 1 AI PR | data/AI_PRS_DATASET/ (input of linker.py) |
| 2_AI_PRs_Dataset/11_Agent_Report | For each dataset: latest date, size, agents, how the agent is involved, Python? | 1 dataset | tables/Agent_report.xlsx |
| 3_Fixes_Dataset/99_Fixes_All_Papers | Which fixes of AI code did the papers already find? | 1 fix | data/FINAL_DATASET/ (to check linker.py) |

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
