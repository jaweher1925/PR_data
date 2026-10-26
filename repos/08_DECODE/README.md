# decode

This is the GitHub repository associated with the paper, *"Learning from 53.6K Real-World Developer Edits of AI-
Generated Code"*. The dataset from this project, **DECODE** (Developer Edits of Code Dataset), is accessible on Huggingface.

For any questions, please contact [jtliang@cs.cmu.edu](mailto:jtliang@cs.cmu.edu) or [mbairath@cs.cmu.edu](mailto:mbairath@cs.cmu.edu).

The codebase is structured as follows:

```
decode/
└── src/
    ├── analysis/
    ├── modeling/
    └── pipeline/
```

## `analysis/`
This folder contains the scripts used to analyze **DECODE**. This includes code to obtain insights on developer behavior (`data_analysis.ipynb`), as well as scripts to label the types of code edits `label_edit_types.py`.

## `modeling/`
This folder contains the scripts used to fine-tune models on **DECODE** for the classification (`classification.py`) and generation (`generation.py`) tasks, including common helper functions (`helpers.py`). It also contains code used to evaluate models on existing benchmarks, like HumanEval (`humaneval.py`) and MBPP (`mbpp.py`).

## `pipeline/`
This folder contains the scripts associated with creating the **DECODE** dataset. This includes the full dataset construction pipeline (`pipeline.py`), as well as scripts for extracting the edit from code (`extract_edit.py`). We also include code for validating the quality of the dataset (`dataset_validation.ipynb`).
