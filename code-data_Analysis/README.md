# code-data_Analysis – all the code of the thesis data work

Work only in this folder. The notebooks read and write the data in `../data/` automatically
(you never need to open it). Thesis question: **why / what / how are AI-generated PRs fixed later?**

```
AI-generated PRs (2_AI_PRs_Dataset)  ->  linker.py (4_Linker_Test)  ->  fix PRs  ->  manual analysis
                                   checked against the papers' fixes (3_Fixes_Dataset)
```

## Folders – run them in this order

| Folder | What it does | Main output (in ../data/) |
|---|---|---|
| **1_Papers/** | One notebook per paper (00_AIDev … 09_Khemissi_References): first look at the data, per agent, 4 charts, the fixes it gives | `tables/fixes.csv` in each folder |
| **2_AI_PRs_Dataset/10_Build_AI_PRs_Dataset** | AIDev-pop (33,596 AI PRs) + Watanabe + labels from the papers (task type, fixed later) | `AI_PRS_DATASET/` |
| **2_AI_PRs_Dataset/11_Agent_Report** | For each dataset: latest date, size, PRs per agent, how the agent is involved, Python | `tables/Agent_report.xlsx` |
| **2_AI_PRs_Dataset/13_Clean_AI_PRs_Dataset** | **The clean AI PR dataset**: AIDev v4 + AIDev-pop + Watanabe, one row per PR (2,744,996 PRs, 688,006 Python) | `AI_PRS_CLEAN/` |
| **3_Fixes_Dataset/99_Fixes_All_Papers** | All the fixes the papers already found (9,619) – the "answer sheet" to check linker.py | `FINAL_DATASET/` |
| **4_Linker_Test/12_Run_Linker** | Run Mohamed's linker.py step by step (step1 … step5) – see its README | `12_Run_Linker/results/` |
| **tools/** | Helper scripts: `miner.py` (GitHub mining), `clone_all.py`, `add_dates.py`, `build_clean_pr_dataset.py` | |
| `common.py` | Helpers used by every notebook (vertical charts with the exact numbers) – do not run it | |

## Setup (once)
```
cd C:\Users\GIGABYTE\Desktop\Thesis_Data_Analysis\code-data_Analysis
python -m pip install -r requirements.txt
```
Open a notebook → **Run All**. Notebooks that read the full AIDev v4 have a `RAW = …` line at the top
(`C:\Users\GIGABYTE\Desktop\code_theis\evolution-assurance\data\raw`).

## Outside this folder (do not move)
| Folder | Why it stays |
|---|---|
| `../data/` | all datasets; the notebooks find it automatically |
| `../repos/` | the papers' original code (made by `tools/clone_all.py`) |
| `../reports/` | Word reports – kept on your PC only (not pushed) |
| `code_theis/evolution-assurance` | Mohamed's repo with linker.py (used by 4_Linker_Test) |

## Where the data comes from
| Data | Source |
|---|---|
| AIDev v4 (2.74M AI PRs) | Hugging Face `hao-li/AIDev` |
| AIDev-pop 2025 (33,596 AI PRs, task type) | copy inside Alam's package "Agents for Fixing" (Zenodo) |
| Watanabe (797 Claude Code PRs) | the paper's repo `ontheuseofagenticcoding` |
| Fixes / labels | Who Finishes (`wannita901/pr-fix-authorship`), Khemissi (`replication_package_v2`), Xia & Miller, Sawada, Yu, Dovhoshliubnyi, Wang |
