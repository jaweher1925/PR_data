from openai import OpenAI
import numpy as np
import pandas as pd
import time
import json
import logging
from collections import Counter
import regex as re
import ast
import os
import sys
import argparse

def generate_data_to_annotate(input_csv_path):
    if not os.path.exists(input_csv_path):
        data_folder = "/data/group_data/code_edits"
        user_ids = json.load(open(f"{data_folder}/user_ids.json", "r"))
        num_batches = len(user_ids)

        datasets = []
        for i in range(num_batches):
            df = pd.read_csv(f"{data_folder}/dataset_batch{i}.csv")
            datasets.append(df)
        df = pd.concat(datasets, ignore_index=True)
        df = df[(df["codebert_score"] > 0.68) | (df["codebert_score"] == 0)]
        df_sampled = df.sort_values(["outcome_completion_id", "edit_timestamp"])

        df["outcome_full_snippet"] = df["outcome_full_snippet"].fillna("")
        df["edit_full_snippet"] = df["edit_full_snippet"].fillna("")
        df["edit_snippet"] = df["edit_snippet"].fillna("")

        # attempt to create triplets
        df_sampled["before_snippet"] = df_sampled.groupby("outcome_completion_id")["edit_full_snippet"].shift(1)
        df_sampled["before_snippet"] = df_sampled["before_snippet"].fillna(df_sampled["outcome_full_snippet"])
        df_sampled["after_snippet"] = df_sampled.groupby("outcome_completion_id")["edit_full_snippet"].shift(-1)
        df_sampled["after_snippet"] = df_sampled["after_snippet"].fillna(df_sampled["edit_full_snippet"])
        df_sampled["current_full_snippet"] = df_sampled["edit_full_snippet"]
        df_sampled["current_snippet"] = df_sampled["edit_snippet"]

        # special filtering
        df_sampled = df_sampled[~((df_sampled["current_full_snippet"].str.strip() != "") &
                    (df_sampled["before_snippet"].str.strip() != "") &
                    (df_sampled["current_snippet"].str.strip() == ""))]
        df.fillna("", inplace=True)

        # make the annotation file
        df_sampled.to_csv(input_csv_path, index=False)

    return pd.read_csv(input_csv_path)

def define_examples():
    # example 1 - complete code deletion
    outcome_snippet_1 = "import styles from '..components/../components/Banner.module.css'"
    current_full_snippet_1 = ""
    current_snippet_1 = ""
    prev_full_snippet_1 = "import styles from '..components/../components/Banner.module.css'"
    after_full_snippet_1 = ""
    output_1 = [{"line": "", "label": "start-over-commenting-out", "rationale": "The line was completely removed"}]
    
    # example 2
    outcome_snippet_2 = """
        .listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`)
        })
    } catch (error) {
    """
    current_full_snippet_2 = """ 
        app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`)
        })
    } catch (e) {
        console.log(e)
    """
    current_snippet_2 = """ 
        app.listen(PORT, () => {
    } catch (e) {
        console.log(e)
    """
    prev_full_snippet_2 = """
        .listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`)
        })
    } catch (error) {
    """
    after_full_snippet_2 = """ 
        app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`)
        })
    } catch (e) {
        console.log(e)
    """
    output_2 = [{"line": "    app.listen(PORT, () => {", "label": "modifying-functionality", "rationale": "modified how the API call gets made"}, {"line": "} catch (e) {", "label": "customizing-personalizing", "rationale": "changed the variable name only"}, {"line": "    console.log(e)", "label": "modifying-functionality", "rationale": "introduced a new method call"}]

    # example 3 - common source of initial label disagreements
    outcome_snippet_3 = """ 
        return '\n'.join(asm_list)  # type: ignore (I         return '\n'.join(asm_list)  # type: ignore (I        return '
    '.join(asm_list)  # type: ignore (I
        @property
        def registered_passes(self) -> List[BasePass]:
            "Get registered passes.")
    """
    current_full_snippet_3 = """ 
    # 'if statement' before that checked that
    # none of the statements are None
    # but PyLance
    print('hello')
    return '\n'.join(asm_list)  # type: ignore
    """
    current_snippet_3 = """ 
    # 'if statement' before that checked that
    # none of the statements are None
    # but PyLance
    print('hello')
    return '\n'.join(asm_list)  # type: ignore
    """
    prev_full_snippet_3 = """ 
    print('hello')
        return '\n'.join(asm_list)  # type: ignore (I         return '\n'.join(asm_list)  # type: ignore (I        return '
    '.join(asm_list)  # type: ignore (I
        @property
        def registered_passes(self) -> List[BasePass]:
            "Get registered passes.")
    """
    after_full_snippet_3 = """ 
    # ('if statement' before that checked that none of the statements are None)
    return '\n'.join(asm_list)  # type: ignore
    """
    output_3 = [{"line": "# 'if statement' before that checked that", "label": "improving-code-quality", "rationale": "added a new comment"}, {"line": "# none of the statements are None", "label": "improving-code-quality", "rationale": "added a new comment"}, {"line": "# but PyLance", "label": "improving-code-quality", "rationale": "added a new comment"}, {"line": "print('hello')", "label": "unchanged"}, {"line": "return '\n'.join(asm_list)  # type: ignore", "label": "improving-code-quality", "rationale": "partially deleted code to make syntax correct"}]

    # example 4
    outcome_snippet_4 = "instruction: `Move to the next lesson. you should respond in json. I.e {lesson_index: int} `"
    current_full_snippet_4 = "instruction: `Move to the next lesson. you should respond in json. I.e { less} `"
    current_snippet_4 = "instruction: `Move to the next lesson. you should respond in json. I.e { less} `"
    prev_full_snippet_4 = "instruction: `Move to the next lesson. you should respond in json. I.e {lesson_index: int} `"
    after_full_snippet_4 = "  instruction: `Move to the next lesson. you should respond in json using the followingg format. { nextLessonIndex: index} `,"
    output_4 = [{"line": "instruction: `Move to the next lesson. you should respond in json. I.e { less} `", "label": "customizing-personalizing", "rationale": "Based on current_snippet alone, it could be either customizing-personalizing or improving-code-quality, but after_full_snippet helps clarify that it's just a variable name change"}]
    
    # example 5
    outcome_snippet_5 = """ 
    const everyUnchecked = this.ringtones()
        .filter(this.notDefaultRingtone)
        .every(r => !r.needRemove());
    """
    current_full_snippet_5 = """ 
    const everyUnchecked = images
        .filter(image => image.rw)
        .every(image => !image.needRemove());

    this.isSelectAll(everyChecked);
    else
    """
    current_snippet_5 = """ 
    const everyUnchecked = images
        .filter(image => image.rw)
        .every(image => !image.needRemove());

    this.isSelectAll(everyChecked);
    else
    """
    prev_full_snippet_5 = """ 
    this.isSelectAll(everyChecked);
    else
    """
    after_full_snippet_5 = "this.isSelectAll(everyChecked);"
    output_5 = [{"line": "            const everyUnchecked = images", "label": "modifying-functionality", "rationale": "although not present in the previous snippet, we see that a similar line is in the outcome snippet but modified"}, 
                {"line": "                .filter(image => image.rw)", "label": "modifying-functionality", "rationale": "although not present in the previous snippet, we see that a similar line is in the outcome snippet but modified"}, 
                {"line": "                .every(image => !image.needRemove());", "label": "customizing-personalizing", "rationale": "although not present in the previous snippet, we see that a similar line is in the outcome snippet but customized, as the variable name is changed"}, 
                {"line": "            this.isSelectAll(everyChecked);", "label": "unchanged"}, {"line": "else", "label": "unchanged"}]

    return [outcome_snippet_1, current_full_snippet_1, current_snippet_1, prev_full_snippet_1, after_full_snippet_1, output_1,
            outcome_snippet_2, current_full_snippet_2, current_snippet_2, prev_full_snippet_2, after_full_snippet_2, output_2,
            outcome_snippet_3, current_full_snippet_3, current_snippet_3, prev_full_snippet_3, after_full_snippet_3, output_3,
            outcome_snippet_4, current_full_snippet_4, current_snippet_4, prev_full_snippet_4, after_full_snippet_4, output_4,
            outcome_snippet_5, current_full_snippet_5, current_snippet_5, prev_full_snippet_5, after_full_snippet_5, output_5]

def construct_user_prompt(outcome_snippet, current_full_snippet, current_snippet, prev_full_snippet, after_full_snippet):
    return f"""
    outcome_snippet:
    {outcome_snippet}

    prev_full_snippet:
    {prev_full_snippet}

    current_full_snippet:
    {current_full_snippet}

    current_snippet:
    {current_snippet}

    after_full_snippet:
    {after_full_snippet}
    """

def create_system_prompt(outcome_snippet_1, current_full_snippet_1, current_snippet_1, prev_full_snippet_1, after_full_snippet_1, output_1, 
                         outcome_snippet_2, current_full_snippet_2, current_snippet_2, prev_full_snippet_2, after_full_snippet_2, output_2,
                         outcome_snippet_3, current_full_snippet_3, current_snippet_3, prev_full_snippet_3, after_full_snippet_3, output_3,
                         outcome_snippet_4, current_full_snippet_4, current_snippet_4, prev_full_snippet_4, after_full_snippet_4, output_4,
                         outcome_snippet_5, current_full_snippet_5, current_snippet_5, prev_full_snippet_5, after_full_snippet_5, output_5):
    return f"""
    You are a code-edit classifier. You must determine how each line in 'current_snippet' changed relative to earlier versions of the code.

    You will receive:
    - outcome_snippet: the oldest known version of the snippet
    - prev_full_snippet: the version of the snippet immediately before the current edit. it may or may not simply be the outcome_snippet itself
    - current_full_snippet: the complete snippet at the current moment
    - current_snippet: ONLY the lines that changed in this edit (a subset of current_full_snippet)
    - after_full_snippet: the next version of the snippet, after the current edit. it may or may not simply be the current_full_snippet itself

    Timeline of code from oldest to newest:
    outcome_snippet, prev_full_snippet, current_full_snippet, after_full_snippet

    Your job:
    For each line in 'current_snippet', assign exactly one label using the taxonomy below. Use all surrounding snippets only to understand how the line changed.
    Prioritize using prev_full_snippet to understand how the line changed - only when you cannot do this use the other snippets as well. 

    Taxonomy:
    1. customizing-personalizing: The line keeps the same behavior but changes literal values or symbol names that customizes the code to fit the remaining file well, stylistically or otherwise.
       Examples: renaming variables, changing constant values/literal values, swapping variable names, replacing variables with literals, changing variable values.
       Changing the value of a comment string is improving-code-quality instead. 
    2. improving-code-quality: The line improves readability, formatting, comments, or syntax without changing semantic meaning.
       Examples: Includes fixing syntax such as closing code structures, adjusting tabbing, cleaning up the code (such as removing unnecessary comments, adding new empty lines), deleting partial code, and adding comments. 
       If the entire line was removed, use start-over-commenting-out instead.
    3. modifying-functionality: The line introduces new code that changes program behavior.
       Examples: Includes adding methods, changing code logic (such as adding new method calls, changing control flow), and changing API calls (such as changing the API, API methods used, parameters passed in). 
       Do not use if only names/values were changed but the logic remains the same, that is customizing-personalizing instead. 
    4. start-over-commenting-out: The line corresponds to complete deletion of an earlier line OR the line in current_snippet is an empty string representing removed code. This includes lines that were fully commented out.
       Partial deletions that still leave a meaningful line are improving-code-quality instead.
       If a line is empty but not due to code deletion, rather the empty line is a new addition, then this is improving-code-quality instead.
       If the entire snippet of code in current_snippet is an empty string or not provided, label the empty string under this category.
       If a new empty line is added but has a matching empty line in prev_full_snippet, then this is unchanged instead. 
    5. unchanged: The line is EXACTLY the same between prev_full_snippet and current_full_snippet. Even one character difference means it is NOT unchanged.

    Tie-breaking:  
    If a line seems ambiguous, apply this priority order:  
    modifying-functionality > customizing-personalizing > improving-code-quality > start-over-commenting-out > unchanged

    Output format:
    Return a JSON array, where each object includes:
    {{
        "line": "<the line from current_snippet>",
        "label": "<one label from the taxonomy>"
    }}
    Use strict JSON (no comments, no trailing commas, no code fences).

    Here are some examples to guide you. I have provided additional rationale per label, but you don't need to include that in your output.

    Example 1:
    outcome_snippet: {outcome_snippet_1}
    current_full_snippet: {current_full_snippet_1}
    current_snippet: {current_snippet_1}
    prev_full_snippet:{prev_full_snippet_1}
    after_full_snippet: {after_full_snippet_1} 
    output: {output_1}
    
    Example 2:
    outcome_snippet: {outcome_snippet_2}
    current_full_snippet: {current_full_snippet_2}
    current_snippet: {current_snippet_2}
    prev_full_snippet:{prev_full_snippet_2}
    after_full_snippet: {after_full_snippet_2} 
    output: {output_2}

    Example 3:
    outcome_snippet: {outcome_snippet_3}
    current_full_snippet: {current_full_snippet_3}
    current_snippet: {current_snippet_3}
    prev_full_snippet:{prev_full_snippet_3}
    after_full_snippet: {after_full_snippet_3} 
    output: {output_3}

    Example 4:
    outcome_snippet: {outcome_snippet_4}
    current_full_snippet: {current_full_snippet_4}
    current_snippet: {current_snippet_4}
    prev_full_snippet:{prev_full_snippet_4}
    after_full_snippet: {after_full_snippet_4} 
    output: {output_4}

    Example 5:
    outcome_snippet: {outcome_snippet_5}
    current_full_snippet: {current_full_snippet_5}
    current_snippet: {current_snippet_5}
    prev_full_snippet:{prev_full_snippet_5}
    after_full_snippet: {after_full_snippet_5} 
    output: {output_5}
    """

def call_model(system_prompt, dataset_df, client, output_csv_path, num_already_annotated, index_offset=0, model_name="gpt-5-mini"):
    for idx, row in dataset_df.iterrows():
        adjusted_idx = idx - index_offset
        if adjusted_idx < num_already_annotated:
            continue

        # call the model
        user_prompt = construct_user_prompt(row["outcome_full_snippet"], row["current_full_snippet"], row["current_snippet"], row["before_snippet"], row["after_snippet"])
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        end = time.perf_counter()
        completion_time = end - start
        log.info(f"Completion time for row {idx}: {completion_time:.3f} seconds")

        dataset_df.at[idx, "llm_edit_labels"] = completion.choices[0].message.content.strip()
        dataset_df.iloc[[adjusted_idx]].to_csv(output_csv_path, mode="a", header=False, index=False)
        
        log.info(f"Processed row {adjusted_idx+1}/{len(dataset_df)}")
        time.sleep(1)

    log.info("Finished processing all rows.")

def process_majorities(output_csv_path):
    final_df = pd.read_csv(output_csv_path)
    final_df["llm_final_label"] = ""

    for idx, row in final_df.iterrows():
        log.info(f"Processing row {idx}")
        log.info(f"Raw LLM answer: {row['llm_edit_labels']}")
        llm_ans = re.sub(r"^```json\s*|```$", "", row["llm_edit_labels"].strip(), flags=re.MULTILINE).strip()
        try:
            log.info(f"Processing row {idx}, raw LLM answer: {row['llm_edit_labels']}")
            llm_ans = json.loads(llm_ans)
        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON for row {idx}: {e}")
            llm_ans = ast.literal_eval(llm_ans)

        if not llm_ans or len(llm_ans) == 0:
            final_df.at[idx, "llm_final_label"] = "start-over-commenting-out"
            continue

        labels = [obj.get("label", "") for obj in llm_ans]
        if all(lbl == "unchanged" for lbl in labels):
            final_df.at[idx, "llm_final_label"] = "unchanged"
            continue

        effective = [obj for obj in llm_ans if obj.get("label") != "unchanged"]
        filtered_labels = [obj.get("label") for obj in effective]

        counts = Counter(filtered_labels)
        max_count = max(counts.values())
        tied = [lbl for lbl, cnt in counts.items() if cnt == max_count]
        final_label = None

        for p in ["modifying-functionality", "improving-code-quality", "customizing-personalizing", "start-over-commenting-out"]:
            if p in tied:
                p = "modifying-functionality" if p == "start-over-commenting-out" else p
                final_df.at[idx, "llm_final_label"] = p
                final_label = p
                break

        if not final_label:
            log.info("Somehow got no label for the edit")

    final_df.to_csv(output_csv_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='CLI for getting labels for code edits using an LLM')
    parser.add_argument("--batch-id", type=int, required=True, help="The batch id for this annotation")
    args = parser.parse_args()
    batch_id = args.batch_id
    batch_size = 500

    # Compute the start and end indices
    start_index = batch_id * batch_size
    end_index = (batch_id + 1) * batch_size
    api_keys = [
        "API_KEY_1",
        "API_KEY_2"
    ]

    # initialize things for the model call
    client = OpenAI(api_key=api_keys[batch_id % len(api_keys)])
    input_csv_path = "annotation_data/dataset_without_edit_labels.csv"
    output_csv_path = f"annotation_data/dataset_with_edit_labels_{batch_id}.csv"

    # get data to annotate
    df = generate_data_to_annotate(input_csv_path)
    end_index = min(end_index, len(df))
    log.info(f"Full dataset length: {len(df)}")

    df = df.iloc[start_index:end_index]
    log.info(f"Dataset: {df.head(1)}")
    log.info(f"Dataset length: {len(df)}")
    log.info(f"Dataset indicies: {start_index} to {end_index-1}")

    # initialize the output df
    df["llm_edit_labels"] = ""
    if not os.path.exists(output_csv_path):
        df.iloc[0:0].to_csv(output_csv_path, index=False)
    num_already_annotated = len(pd.read_csv(output_csv_path))
    log.info(f"Number already annotated from {output_csv_path}: {num_already_annotated}")

    # call the model
    system_prompt = create_system_prompt(*define_examples())
    call_model(system_prompt, df, client, output_csv_path, num_already_annotated, index_offset=start_index)

    # process majorities
    process_majorities(output_csv_path)
