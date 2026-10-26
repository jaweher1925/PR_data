from collections import Counter
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainerCallback, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, concatenate_datasets
import argparse
import bitsandbytes
import config
import numpy as np
import os
import torch
import wandb
print(bitsandbytes.__version__)

TRAIN_DATASET_FILE = "/data/group_data/code_edits/trajectory_dataset_train.jsonl"
VAL_DATASET_FILE = "/data/group_data/code_edits/trajectory_dataset_val.jsonl"
TEST_DATASET_FILE = "/data/group_data/code_edits/trajectory_dataset_test.jsonl"
PREFIX_COLUMN = "outcome_prefix"
SUFFIX_COLUMN = "outcome_suffix"
PREV_COLUMN = "outcome_full_snippet"
TARGET_COLUMN = "edit_full_snippet"
LABEL_COLUMN = "code_remaining_label"
EXTRA_COMPLETIONS_COLUMN = "edit_full_snippets"
LOGGING_STEPS = 20
SAVE_TOTAL_LIMIT = 2
SEED = 8
USER = os.environ["USER"]

def preprocess_truncation_function(example, temp_tok, max_length, extra_instruction):
    completion_tokenized = temp_tok(extra_instruction+example["completion"])

    prompt_tokenized = temp_tok(
        example["prompt"], 
        truncation=True, 
        max_length=max(0, max_length-len(completion_tokenized["input_ids"])-16)
    )
    trimmed_prompt = temp_tok.decode(prompt_tokenized["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return {"prompt": trimmed_prompt + extra_instruction, "completion": example["completion"]}

def format_extra_edits(example, num_extra):
    edits = example[EXTRA_COMPLETIONS_COLUMN][:num_extra]
    if num_extra <= 0 or len(edits) == 0:
        return ""

    lines = []
    for i, e in enumerate(edits, start=1):
        if e == "":
            e = "<EDIT DELETED>"
        lines.append(f"EDIT {i}: {e}")

    return "\n".join(lines)

def create_model(model_name, use_lora, lora_r, lora_alpha, lora_dropout, target_modules):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )

    if use_lora:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model

def remap_label(row, num_labels):
    if num_labels == 2 and int(row[LABEL_COLUMN]) == 2:
        row[LABEL_COLUMN] = "1"
    return row

def rebalance_dataset(unbalanced_ds):
    counts = Counter(unbalanced_ds[LABEL_COLUMN])
    target_count = max(counts.values())
    parts = []
    for label, count in counts.items():
        subset = unbalanced_ds.filter(lambda x: x[LABEL_COLUMN] == label)
        parts.extend([subset] * (target_count//count))

    new_ds = concatenate_datasets(parts).shuffle(seed=SEED)
    print(f"New dataset counts: {dict(Counter(new_ds[LABEL_COLUMN]))}")
    return new_ds

class GenEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_ds, eval_fn, train_ds=None, do_sample=False, gen_max_new_tokens=4, label_desc=None):
        self.trainer = trainer
        self.eval_fn = eval_fn
        self.eval_ds = eval_ds
        self.train_ds = train_ds
        self.do_sample = do_sample
        self.gen_max_new_tokens = gen_max_new_tokens
        self.label_desc = label_desc

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        if self.train_ds is not None:
            gen_metrics = self.eval_fn(
                self.trainer.processing_class, 
                self.trainer.model, 
                self.train_ds, 
                do_sample=self.do_sample, 
                gen_max_new_tokens=self.gen_max_new_tokens,
                label_desc=self.label_desc,
                metric_prefix="train"
            )
            metrics.update(gen_metrics)
            self.trainer.log(gen_metrics)

        gen_metrics = self.eval_fn(
            self.trainer.processing_class, 
            self.trainer.model, 
            self.eval_ds, 
            do_sample=self.do_sample, 
            gen_max_new_tokens=self.gen_max_new_tokens,
            label_desc=self.label_desc
        )
        metrics.update(gen_metrics)
        self.trainer.log(gen_metrics)
        return control