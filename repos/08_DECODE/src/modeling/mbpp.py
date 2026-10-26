"""
Script that can load any model specified and then run the MBPP evaluation on it.
Qwen reported results: https://qwenlm.github.io/blog/qwen2.5-coder-family/
Qwen code to run HE benchmark: https://github.com/QwenLM/Qwen3-Coder/blob/main/qwencoder-eval/instruct/eval_plus/model.py
Helpful code: https://github.com/abacaj/code-eval/tree/main
"""

import os
import torch
import csv
import re
import json
import logging
import sys
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

def get_function_name(header):
    m = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', header)
    return m.group(1) if m else None

# run over all examples in MBPP
def evaluate_mbpp(model_path):
    log.info(f"Loading dataset...")
    mbpp = [json.loads(line) for line in open("./data/mbpp.jsonl")]

    code_eval_metric = load_metric("code_eval", module_type="metric")
    k = 5

    log.info(f"Loading model from {model_path}...")
    tokenizer, model = load_model(model_path)
    hidden_test_cases, candidate_solutions = [], []

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "generation"])

        for example in tqdm(mbpp):
            code_lines = example["code"].splitlines()
            function_defn_index = [i for i, cl in enumerate(code_lines) if cl.startswith("def")][0] + 1
            function_defn = "\n".join(code_lines[0:function_defn_index])
            function_name = get_function_name(function_defn)

            function_header = function_defn + '\n\t"""\n\t' + '\n\t'.join(example["text"].splitlines()) + '\n\t"""\n\n'
            logging.info(f"\nEvaluating task {example['task_id']} with function header:\n{function_header}\n\n")

            prompt = (
                function_header +
                "\n# Write the function body only, inside a code block.\n"
                "```python\n"
            )

            test_code_lines = [t.replace(function_name, "candidate") for t in example["test_list"]]
            test_code = "def check(candidate):\n\t" + "\n\t".join(test_code_lines)
            logging.info(f"Evaluating task {example['task_id']} with prompt:\n{prompt}\nand test code:\n{test_code}\n\n")

            hidden_test_cases.append(test_code)
            this_problem_solutions = []

            for _ in range(k):
                gen = generate_completion(prompt, tokenizer, model)
                logging.info(f"Generated solution for task {example['task_id']}:\n{gen}\n\n")
                this_problem_solutions.append(function_header + gen)
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
    parser.add_argument("--output_file", type=str, default="mbpp_generations.csv")
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
    OUTPUT_FILE = args.output_file

    evaluate_mbpp(MODEL_NAME)