from experiment_helpers import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROMPT_HEADER = """You are a code-edit prediction model.

Given: COMPLETION (snippet of code inserted by the user), PREFIX (code before the COMPLETION), and SUFFIX (code after the COMPLETION).
Task: Classify how the COMPLETION will look like after the user makes their final edits later.
Rule: Output ONLY the final label (no code, no explanations).
"""

# used for few-shot prompting
def generate_examples(num_extra, label_desc, num_labels):
    examples = [
        {
            PREFIX_COLUMN: "# HW 7 question 2\nimport numpy as np\nfrom itertools import combinations\nfrom numpy.linalg import matrix_rank, inv\n\nA = np.random.uniform(-1, 1, (6, 50))\nb = np.random.uniform(-1, 1, 6)\nA_indices = [i for i in range(50)]\nA_combos = list(combinations(A_indices, 6))\n\nfor columns in A_combos:\n    A_sub = A[:, columns]\n    if matrix_rank(A_sub) == 6:",
            PREV_COLUMN: "        y_sub = inv(A_sub) @ b        y = np.zeros(50)\n        i = 0",
            SUFFIX_COLUMN: "        for idx in columns:\n            y[idx] = y_sub[i]\n            i += 1\n        \n",
            "EDIT 1": "        y_sub = inv(A_sub) @ b        y = np.zeros(50)\n        i = 0",
            "EDIT 2": "        y_sub = inv(A_sub) @ b        y = np.zeros(50)\n        i = 0",
            LABEL_COLUMN: "2" if num_labels == 3 else "1",
            TARGET_COLUMN: "        y_sub = inv(A_sub) @ b        y = np.zeros(50)\n        i = 0"
        },
        {
            PREFIX_COLUMN: "'use strict'\n\n.btn {\n    position: absolute;\n    top: 20px;",
            PREV_COLUMN: "    left: 20px;\n    width: 100px;\n    height: 30px;\n    background-color: #4CAF50;\n    color: white;\n    border: none;\n    cursor: pointer;",
            SUFFIX_COLUMN: "}\n",
            "EDIT 1": "    left: 250px;\n    width: 250px;\n    height: 30px;\n    background-color: #4CAF50;\n    color: white;\n    border: none;\n    cursor: pointer;",
            "EDIT 2": "    top: 20px;\n    left: 250px;\n    width: 50px;\n       background-color: #4CAF50;\n    color: white;\n    border: none;\n    cursor: pointer;",
            LABEL_COLUMN: "1",
            TARGET_COLUMN: "    top: 20px;\n    left: 250px;\n    height: 50px;\n    width: 50px;\n    border-radius: 50%;\n    border: none;\n    color: var(--btnFontColor);\n    background - color: var(--btnBg);\n}\n.btn:focus { outline-style: none;}\n\n.light-theme {\n    --bg: var(--green);\n    --fontColor: var(--black);\n    --btnBg: var(--black);\n    --btnFontColor: var(--white);\n    "
        },
        {
            PREFIX_COLUMN: "$(document).ready",
            PREV_COLUMN: "$(document).ready(function() {",
            SUFFIX_COLUMN: "$(\".form\").hover(function() {\n  $(this).css('background-color', '#f0f8ff');\n}, function() {\n  $(this).css('background-color', '');\n});",
            "EDIT 1": "$(document).ready",
            "EDIT 2": "",
            LABEL_COLUMN: "0",
            TARGET_COLUMN: "<EDIT DELETED>"
        }
    ]

    newline = "\n"
    return f"""=== EXAMPLE 1 ===
PREFIX:
{examples[0][PREFIX_COLUMN]}
COMPLETION:
{examples[0][PREV_COLUMN]}
{("EDIT 1:"+newline + examples[0]['EDIT 1']) if num_extra > 0 else ""}
{("EDIT 2:"+newline + examples[0]['EDIT 2']) if num_extra > 1 else ""}
SUFFIX:
{examples[0][SUFFIX_COLUMN]}
FINAL COMPLETION SNIPPET LABEL: {examples[0][LABEL_COLUMN]}: {label_desc[examples[0][LABEL_COLUMN]]}

=== EXAMPLE 2 ===
PREFIX:
{examples[1][PREFIX_COLUMN]}
COMPLETION:
{examples[1][PREV_COLUMN]}
{("EDIT 1:"+newline + examples[1]['EDIT 1']) if num_extra > 0 else ""}
{("EDIT 2:"+newline + examples[1]['EDIT 2']) if num_extra > 1 else ""}
SUFFIX:
{examples[1][SUFFIX_COLUMN]}
FINAL COMPLETION SNIPPET LABEL: {examples[1][LABEL_COLUMN]}: {label_desc[examples[1][LABEL_COLUMN]]}

=== EXAMPLE 3 ===
PREFIX:
{examples[2][PREFIX_COLUMN]}
COMPLETION:
{examples[2][PREV_COLUMN]}
{("EDIT 1:"+newline + examples[2]['EDIT 1']) if num_extra > 0 else ""}
{("EDIT 2:"+newline + examples[2]['EDIT 2']) if num_extra > 1 else ""}
SUFFIX:
{examples[2][SUFFIX_COLUMN]}
FINAL COMPLETION SNIPPET LABEL: {examples[2][LABEL_COLUMN]}: {label_desc[examples[2][LABEL_COLUMN]]}

"""

# get dataset into standard prompt-completion format
def preprocess_function(example, num_labels, label_col_map, label_desc, num_extra_completions=0, provide_examples=False):
    labels = "\n".join([f"{label_col_map[str(i)]} = {label_desc[label_col_map[str(i)]]}" for i in range(num_labels)])
    if provide_examples:
        prompt = PROMPT_HEADER + f"Labels:\n{labels}\n\n" + generate_examples(num_extra_completions, label_desc, num_labels) + "=== YOUR TURN ===\n\n"
    else:
        prompt = PROMPT_HEADER + f"Labels:\n{labels}\n\n"

    prompt += f"""PREFIX:
{example[PREFIX_COLUMN]}

COMPLETION:
{example[PREV_COLUMN]}

SUFFIX:
{example[SUFFIX_COLUMN]}

"""

    extra_block = format_extra_edits(example, num_extra_completions)
    if extra_block != "":
        prompt += f"""
Here are some initial edits to the COMPLETION (intermediate versions, oldest to newest). 
Use them as additional context to predict the FINAL edited completion label.
If an intermediate edit is <EDIT DELETED>, it does not necessarily mean the final completion is <EDIT DELETED>.

{extra_block}

"""
        
    return {"prompt": prompt, "completion": label_col_map[example[LABEL_COLUMN]]}

# create the dataset and filter out examples and process correctly
def create_dataset(overfit_check, max_length, model_name, num_labels, label_col_map, label_desc, provide_examples=False, num_extra_completions=0, remove_obvious_trajectories=False):
    ds = load_dataset("json", data_files={"val": VAL_DATASET_FILE, "train": TRAIN_DATASET_FILE})
    temp_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    final_instruction = "\nFINAL COMPLETION SNIPPET LABEL:\n\n" # use this for all finetuning experiments
    # final_instruction = "\nFINAL COMPLETION SNIPPET LABEL:\n" # use this for small llama baseline only

    if remove_obvious_trajectories:
        ds = ds.filter(lambda x: (
            int(x[LABEL_COLUMN]) != 0 or (int(x[LABEL_COLUMN]) == 0 and "" not in x[EXTRA_COMPLETIONS_COLUMN][:num_extra_completions])
        ))

    ds = ds.map(lambda x: preprocess_function(x, num_labels, label_col_map, label_desc, num_extra_completions=num_extra_completions, provide_examples=provide_examples))
    ds = ds.map(lambda x: preprocess_truncation_function(x, temp_tok, max_length, final_instruction))

    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["val"] = ds["train"]

    return ds

# run generation and compute metrics
def run_generation_eval(tokenizer, model, ds, metric_prefix="eval", label_desc=None, **kwargs):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for _, ex in enumerate(ds):
            inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=False).to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            gen_text = gen_text.split("<|end|>")[0]

            preds.append(extract_label(gen_text, label_desc))
            golds.append(ex["completion"])

            print("PRINTING OUT SAMPLES")
            print("GENERATED:\n", preds[-1])
            print("TARGET:\n", golds[-1])

    return compute_generation_metrics(preds, golds, metric_prefix)

# compute metrics
def compute_generation_metrics(preds, golds, metric_prefix):
    metrics = {
        f"{metric_prefix}_accuracy": accuracy_score(golds, preds),
        f"{metric_prefix}_precision_macro": precision_score(golds, preds, average="macro", zero_division=0),
        f"{metric_prefix}_recall_macro": recall_score(golds, preds, average="macro", zero_division=0),
        f"{metric_prefix}_f1_macro": f1_score(golds, preds, average="macro", zero_division=0),
    }

    precision_per_class = precision_score(golds, preds, average=None, zero_division=0)
    recall_per_class = recall_score(golds, preds, average=None, zero_division=0)
    f1_per_class = f1_score(golds, preds, average=None, zero_division=0)

    for i, (p, r, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f"{metric_prefix}_precision_class_{i}"] = p
        metrics[f"{metric_prefix}_recall_class_{i}"] = r
        metrics[f"{metric_prefix}_f1_class_{i}"] = f1
    
    print(metrics)
    return metrics

# extracts the best guess for a label, -1 if not possible
def extract_label(text, label_desc):
    counts = {k: text.count(k) for k in label_desc}
    if all(v == 0 for v in counts.values()):
        return "-1"  # no valid prediction found
    else:
        return max(counts, key=counts.get)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--overfit_check", action="store_true")
    parser.add_argument("--loss_on_everything", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_extra_completions", type=int, default=0)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--letter_labels", action="store_true")
    parser.add_argument("--rebalance_train", action="store_true")
    parser.add_argument("--remove_obvious_trajectories", action="store_true")
    args = parser.parse_args()

    EXPERIMENT_NAME = args.experiment_name
    MODEL_NAME = args.model_name
    OUTPUT_DIR = f"/data/user_data/{USER}/modeling-experiments/data/experiment_{EXPERIMENT_NAME}"
    MAX_LENGTH = args.max_length
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MICRO_BATCH_SIZE = args.micro_batch_size
    WARMUP_RATIO = args.warmup_ratio
    WEIGHT_DECAY = args.weight_decay
    RESUME_FROM_CHECKPOINT = os.path.isdir(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0
    OVERFIT_CHECK = True if args.overfit_check else False
    LOSS_ON_EVERYTHING = True if args.loss_on_everything else False
    USE_LORA = True if args.use_lora else False
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    NUM_EXTRA_COMPLETIONS = max(0, int(args.num_extra_completions))
    NUM_LABELS = int(args.num_labels)
    LETTER_LABELS = True if args.letter_labels else False
    REBALANCE_TRAIN = True if args.rebalance_train else False
    REMOVE_OBVIOUS_TRAJECTORIES = True if args.remove_obvious_trajectories else False

    # initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=f"code_edits_class_by_gen_{NUM_LABELS}labels_final", name=EXPERIMENT_NAME)

    # decide which labels to use
    if LETTER_LABELS and NUM_LABELS == 3:
        LABEL_DESC = {"A": "deleted by user", "B": "partially modified by user", "C": "unmodified by user"}
        LABEL_COL_MAP = {"0": "A", "1": "B", "2": "C"}
    elif not LETTER_LABELS and NUM_LABELS == 3:
        LABEL_DESC = {"0": "deleted by user", "1": "partially modified by user", "2": "unmodified by user"}
        LABEL_COL_MAP = {"0": "0", "1": "1", "2": "2"}
    elif LETTER_LABELS and NUM_LABELS == 2:
        LABEL_DESC = {"A": "deleted by user", "B": "partially or unmodified by user"}
        LABEL_COL_MAP = {"0": "A", "1": "B", "2": "B"}
    else:
        LABEL_DESC = {"0": "deleted by user", "1": "partially or unmodified by user"}
        LABEL_COL_MAP = {"0": "0", "1": "1", "2": "1"}

    # load the datasets for generation
    ds = create_dataset(
        OVERFIT_CHECK, 
        MAX_LENGTH, 
        MODEL_NAME, 
        NUM_LABELS,
        LABEL_COL_MAP,
        LABEL_DESC,
        num_extra_completions=NUM_EXTRA_COMPLETIONS, 
        remove_obvious_trajectories=REMOVE_OBVIOUS_TRAJECTORIES
    )
    ds = ds.map(lambda x: remap_label(x, NUM_LABELS))
    ds["val"] = rebalance_dataset(ds["val"])
    ds["train_original"] = rebalance_dataset(ds["train"])
    if REBALANCE_TRAIN:
        ds["train"] = ds["train_original"]

    # load the model for training
    model = create_model(MODEL_NAME, use_lora=USE_LORA, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES)

    # set training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=LOGGING_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR,
        model_init_kwargs={"dtype": torch.bfloat16},
        completion_only_loss=not LOSS_ON_EVERYTHING,
        max_length = MAX_LENGTH + 128, # add buffer
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=EXPERIMENT_NAME,
        save_total_limit=SAVE_TOTAL_LIMIT,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE, # small on purpose
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        remove_unused_columns=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["val"])
    trainer.add_callback(GenEvalCallback(trainer, ds["val"], run_generation_eval, train_ds=ds["train_original"], label_desc=LABEL_DESC))

    print("training model:")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(OUTPUT_DIR)

    run.finish()

    print("evaluating best model:")
    run_generation_eval(trainer.processing_class, trainer.model, ds["val"], label_desc=LABEL_DESC)

    print("done!")
