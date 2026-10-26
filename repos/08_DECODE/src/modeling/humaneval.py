"""
Script that can load any model specified and then run the HumanEval evaluation on it.
Qwen reported results: https://qwenlm.github.io/blog/qwen2.5-coder-family/
Qwen code to run HE benchmark: https://github.com/QwenLM/Qwen3-Coder/blob/main/qwencoder-eval/instruct/eval_plus/model.py
Helpful code: https://github.com/abacaj/code-eval/tree/main
"""

import os
import torch
import csv
import re
import logging
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from evaluate import load as load_metric
from tqdm import tqdm
from src.modeling.helpers import *

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# load the model from the given path
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_model(
        model_path,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model.eval()

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

# generate the code solution for a prompt task
def generate_completion(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # remove the prompt part
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # some clean up
    blocks1 = re.findall(r"```python\s*(.*?)```", generated, re.DOTALL)
    blocks2 = re.findall(r"```\s*(.*?)```", generated, re.DOTALL)
    if blocks1:
        generated = blocks1[-1].strip() 
    elif blocks2:
        generated = blocks2[-1].strip()

    func = re.findall(r"(def\s+\w+\(.*?:\s*[\s\S]+)", generated)
    if func:
        generated = func[0].strip()
    
    return re.sub(r"\n?\s*```+\s*$", "", generated).rstrip()

# run over all examples in human eval
def evaluate_human_eval(model_path):
    log.info(f"Loading dataset...")
    human_eval = load_dataset("openai_humaneval")["test"]
    code_eval_metric = load_metric("code_eval", module_type="metric")
    k = 5

    log.info(f"Loading model from {model_path}...")
    tokenizer, model = load_model(model_path)
    hidden_test_cases, candidate_solutions = [], []

    with open("humaneval_generations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "generation"])

        for example in tqdm(human_eval):
            prompt = (
                example["prompt"] +
                "\n# Write the function body only, inside a code block.\n"
                "```python\n"
            )
            hidden_test_cases.append(example["test"])
            this_problem_solutions = []

            for _ in range(k):
                gen = generate_completion(prompt, tokenizer, model)
                this_problem_solutions.append(example["prompt"] + gen)
                writer.writerow([example["task_id"], gen])
            
            candidate_solutions.append(this_problem_solutions)

    log.info(f"Evaluation results for {model_path}")
    k_values = list(range(1, k+1))
    pass_at_k, _ = code_eval_metric.compute(
        references=hidden_test_cases[:len(candidate_solutions)],
        predictions=candidate_solutions,
        k=k_values
    )
    for k in k_values:
        log.info(f"Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    USE_LORA = True if args.use_lora else False
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    MODEL_NAME = args.model_name

    evaluate_human_eval(MODEL_NAME)