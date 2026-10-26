from src.modeling.helpers import *
import evaluate
import Levenshtein

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROMPT_HEADER = """You are a code-edit prediction model.

Given: COMPLETION (snippet of code inserted by the user), PREFIX (code before the COMPLETION), and SUFFIX (code after the COMPLETION).
Task: Predict what the COMPLETION will look like after the user makes their final edits later.

Rules:
- Output ONLY the final edited COMPLETION (no prefix/suffix, no explanations, no markdown fences).
- Preserve semantics; make the minimal necessary edits.
- Keep the same language and style as the surrounding code.
- If the completion would be completely deleted in the final version, output exactly: <EDIT DELETED>

"""

# prepare metric stuff
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# used for few-shot prompting
def generate_examples():
    examples = [
        {
            PREFIX_COLUMN: "import express from 'express';\nimport dotenv from \"dotenv\";\nimport './config/dotenv.js';\n\ndotenv.config();\n\nconst PORT = processs.env.PORT || 8080;\n\nconst app = express();\n",
            PREV_COLUMN: "app.listen(PORT, () => {\n  console.log(`Server is running on port ${PORT}`);\n})",
            SUFFIX_COLUMN: "",
            TARGET_COLUMN: "app.listen(PORT, () => {\n  console.log(`Server is running on port ${PORT}`);\n});"
        },
        {
            PREFIX_COLUMN: "from django.shortcuts import render\n\nfrom products.models import Product, ProductCategory, Slider\n\n#! \u0444\u0443\u043d\u043a\u0446\u0438\u0438 = \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u043b\u0435\u0440\u044b = \u0432\u044c\u044e\u0445\u0438\n\ndef index(request):\n  return render(request, 'products/index.html')\n\ndef products(request, category_id):\n  context = {\n    'categories': ProductCategory.objects.all(),\n    'sliders': Slider.objects.all(),\n  }",
            PREV_COLUMN: "  if category_id:\n    category = ProductCategory.objects.get(pk=category_id)\n    products = Product.objects.filter(category=category)\n    context['category'] = category\n  else:\n    context['category'] = None",
            SUFFIX_COLUMN: "  products = Product.objects.all(),\n\n  return render(request, 'products/products.html', context )",
            TARGET_COLUMN: "    'products': products\n  }    \n    "
        }
    ]
    return f"""=== EXAMPLE 1 ===
PREFIX:
{examples[0][PREFIX_COLUMN]}
COMPLETION:
{examples[0][PREV_COLUMN]}
SUFFIX:
{examples[0][SUFFIX_COLUMN]}
FINAL EDITED VERSION OF COMPLETION SNIPPET:
{examples[0][TARGET_COLUMN]}

=== EXAMPLE 2 ===
PREFIX:
{examples[1][PREFIX_COLUMN]}
COMPLETION:
{examples[1][PREV_COLUMN]}
SUFFIX:
{examples[1][SUFFIX_COLUMN]}
FINAL EDITED VERSION OF COMPLETION SNIPPET:
{examples[1][TARGET_COLUMN]}

"""

# get dataset into standard prompt-completion format
def preprocess_function(example, num_extra_completions=0, provide_examples=False):
    prompt = PROMPT_HEADER + generate_examples() + "\n=== YOUR TURN ===\n\n" if provide_examples else PROMPT_HEADER
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
Use them as additional context to predict the FINAL edited completion.
If an intermediate edit is <EDIT DELETED>, it does not mean the final output is <EDIT DELETED>.

{extra_block}

"""
        
    return {"prompt": prompt, "completion": example[TARGET_COLUMN]}

# create the dataset and filter out examples and process correctly
def create_dataset(overfit_check, max_length, model_name, provide_examples=False, num_extra_completions=0):
    ds = load_dataset("json", data_files={"val": VAL_DATASET_FILE, "train": TRAIN_DATASET_FILE})
    temp_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # final_instruction = "\nFINAL EDITED VERSION OF COMPLETION SNIPPET:\n\n" # use this for all finetuning experiments
    final_instruction = "\nFINAL EDITED VERSION OF COMPLETION SNIPPET:\n" # use this for small baselines

    ds = ds.filter(lambda x: int(x[LABEL_COLUMN]) != 0)
    ds = ds.map(lambda x: preprocess_function(x, num_extra_completions=num_extra_completions, provide_examples=provide_examples))
    ds = ds.map(lambda x: preprocess_truncation_function(x, temp_tok, max_length, final_instruction))

    if overfit_check:
        ds["train"] = ds["train"].select(range(3))
        ds["val"] = ds["train"]

    return ds

# run generation and compute metrics
def run_generation_eval(tokenizer, model, ds, do_sample=True, gen_max_new_tokens=256, **kwargs):
    model.eval()
    preds, golds = [], []

    with torch.no_grad():
        for _, ex in enumerate(ds):
            inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=False).to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                do_sample=do_sample,
                num_beams=3,
                early_stopping=True,
                length_penalty=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            gen_text = gen_text.split("<|end|>")[0]
            if gen_text.strip() == "<EDIT DELETED>":
                gen_text = ""

            preds.append(gen_text)
            golds.append(ex["completion"])

            print("PRINTING OUT SAMPLES")
            print("GENERATED:\n", preds[-1])
            print("TARGET:\n", golds[-1])

    return compute_generation_metrics(preds, golds)

# character overlap
def multiset_overlap_ratio(a, b):
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    denom = max(len(a), len(b))
    return inter / denom if denom > 0 else 1.0

# line overlap
def multiset_line_overlap_ratio(a, b):
    la = [ln.rstrip() for ln in a.splitlines() if ln.strip() != ""]
    lb = [ln.rstrip() for ln in b.splitlines() if ln.strip() != ""]
    ca, cb = Counter(la), Counter(lb)
    inter = sum((ca & cb).values())
    denom = max(len(la), len(lb))
    return inter / denom if denom > 0 else 1.0

# compute metrics
def compute_generation_metrics(preds, golds):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    levenshtein_scores = []
    char_overlap_scores = []
    line_overlap_scores = []

    for pred, gold in zip(preds, golds):
        if len(gold.strip()) == 0 and len(pred.strip()) == 0:
            rouge1_scores.append(1.0)
            rouge2_scores.append(1.0)
            rougeL_scores.append(1.0)
            bleu_scores.append(1.0)
            levenshtein_scores.append(1.0)
            char_overlap_scores.append(1.0)
            line_overlap_scores.append(1.0)
            continue
        elif len(gold.strip()) == 0 or len(pred.strip()) == 0:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
            bleu_scores.append(0.0)
            levenshtein_scores.append(0.0)
            char_overlap_scores.append(0.0)
            line_overlap_scores.append(0.0)
            continue

        r = rouge.compute(predictions=[pred], references=[gold])
        rouge1_scores.append(r["rouge1"])
        rouge2_scores.append(r["rouge2"])
        rougeL_scores.append(r["rougeL"])

        b = bleu.compute(predictions=[pred], references=[gold])
        bleu_scores.append(b["bleu"])

        dist = Levenshtein.distance(pred, gold)
        max_len = max(len(gold), len(pred))
        lev_sim = 1.0 - dist / max_len if max_len > 0 else 1.0
        levenshtein_scores.append(lev_sim)

        char_overlap_scores.append(multiset_overlap_ratio(pred, gold))
        line_overlap_scores.append(multiset_line_overlap_ratio(pred, gold))

    metrics = {
        "eval_bleu": float(np.mean(bleu_scores)),
        "eval_rouge1": float(np.mean(rouge1_scores)),
        "eval_rouge2": float(np.mean(rouge2_scores)),
        "eval_rougeL": float(np.mean(rougeL_scores)),
        "eval_levenshtein": float(np.mean(levenshtein_scores)),
        "eval_char_overlap": float(np.mean(char_overlap_scores)),
        "eval_line_overlap": float(np.mean(line_overlap_scores)),
    }
    print(metrics)
    return(metrics)

# train percentile length for output
def compute_max_new_tokens_from_train(ds_train, tokenizer, percentile=95, min_new_tokens=64, max_cap=512):
    lengths = []
    for ex in ds_train:
        lengths.append(len(tokenizer(ex["completion"], truncation=False)["input_ids"]))

    max_new = min(max(min_new_tokens, int(np.percentile(lengths, percentile))), max_cap)
    print("Computed max_new_tokens for generation: ", max_new)
    return int(max_new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--overfit_check", action="store_true")
    parser.add_argument("--loss_on_everything", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--sample_generation", action="store_true")
    parser.add_argument("--num_extra_completions", type=int, default=0)
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
    SAMPLE_GENERATION = True if args.sample_generation else False
    NUM_EXTRA_COMPLETIONS = max(0, int(args.num_extra_completions))

    # initialize wandb
    wandb.login(key=config.WANDB_KEY)
    run = wandb.init(project=f"code_edits_generation_final", name=EXPERIMENT_NAME)

    # load the datasets for generation
    ds = create_dataset(OVERFIT_CHECK, MAX_LENGTH, MODEL_NAME, num_extra_completions=NUM_EXTRA_COMPLETIONS)
    gen_max_new_tokens = compute_max_new_tokens_from_train(ds["train"], AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True))

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
        metric_for_best_model="eval_levenshtein",
        greater_is_better=True
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["val"])
    trainer.add_callback(GenEvalCallback(trainer, ds["val"], run_generation_eval, do_sample=SAMPLE_GENERATION, gen_max_new_tokens=gen_max_new_tokens))

    print("training model:")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(OUTPUT_DIR)

    run.finish()

    print("evaluating best model:")
    run_generation_eval(trainer.processing_class, trainer.model, ds["val"], do_sample=SAMPLE_GENERATION, gen_max_new_tokens=gen_max_new_tokens)

    print("done!")