"""
extract_edit.py
Takes in an original file, new file, start and end lines (1-indexed) of the chunk of code to track in the original file
Outputs the edit in the new file if possible
"""

from unidiff import PatchSet
import code_bert_score
import json
import logging
import os
import subprocess
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger(__name__)

def diff_from_strings(old_code, new_code, num_lines_context=3):
    """
    Given two pieces of texts, compute a diff of them using the git diff function and return it as a patch set.

    Args:
        old_text (string): The string corresponding to the old code
        new_text (string): The string corresponding to the new code
        num_lines_context (int): Number of context lines to include in the diff output

    Output explanation:
    A patch set contains one or more patches - one patch per file. The way we call this function, there will only be one patch. 
    A patch consists of several hunks. Each hunk is a modified part of the file, with some context lines before and after the changes.
    Each hunk consists of deletion lines from the old file, addition lines in the new file, and context lines that are unchanged.
    Ex: if a line was modified, then there will be one deletion and one addition line in the hunk, with context lines around them.

    Note that if the hunk is at the very start or end of the file, it may not have context lines for padding before/after.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f_old, tempfile.NamedTemporaryFile(mode='w+', delete=True) as f_new:
        f_old.write(old_code)
        f_old.flush()
        f_new.write(new_code)
        f_new.flush()

        # Use the Git Diff with Histogram algorithm
        # Documentation: https://adamj.eu/tech/2024/01/18/git-improve-diff-histogram/
        result = subprocess.run([
            'git', 'diff', '--no-index', '--histogram', f"-U{num_lines_context}", f_old.name, f_new.name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return PatchSet(result.stdout)

def transpile_ts_inline(ts_code):
    """
    Transpile a TypeScript snippet to JavaScript using ts.transpileModule.
    """
    js_code = f"""
        const ts = require('typescript');
        const result = ts.transpileModule({json.dumps(ts_code)}, {{
            compilerOptions: {{ target: ts.ScriptTarget.ES5 }}
        }});
        process.stdout.write(result.outputText);
    """

    with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False) as f:
        f.write(js_code)
        path = f.name

    proc = subprocess.run(["node", path], capture_output=True, text=True)
    os.remove(path)

    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())

    return proc.stdout.strip()

def compute_codebert_similarity(cand, ref, lang):
    """
    Computes CodeBERT F1 score between two code snippets. Returns 0.0 if the language is not supported. 
    This is used to check if the extracted edit snippet is similar to the outcome snippet. 
    This is an extra precaution, ideally checking that the files were similar or not before should be sufficient.

    Args:
        cand (string): The candidate code snippet
        ref (string): The reference code snippet
        lang (string): The programming language of the code snippets. Ex: python
    """
    # handle typescript separately
    if lang == "typescript":
        cand = transpile_ts_inline(cand)
        ref = transpile_ts_inline(ref)
        lang = "javascript"

    scores = code_bert_score.score(cands=[cand], refs=[ref], lang=lang)
    return scores[2].tolist()[0]  # Return F1 scores
    
def parse_patch(patch, outcome_snippet_line_start, outcome_snippet_line_end, outcome_file, snapshot_file):
    """
    Takes a patch (from git diff) and parses the output to a better format
    It also returns a separate list of all context and removed lines that were in the outcome snippet range

    Args:
        patch (PatchSet): The patch set from git diff
        outcome_snippet_line_start (int): The starting line number of the outcome snippet in the outcome file, 1-indexed
        outcome_snippet_line_end (int): The ending line number of the outcome snippet in the outcome file, 1-indexed
        outcome_file (string): The outcome file
        snapshot_file (string): The snapshot file
    """
    outcome_context_and_removed_lines = []
    all_changed_lines = []
    outcome_file_lines = outcome_file.splitlines()
    snapshot_file_lines = snapshot_file.splitlines()

    for patched_file in patch:
        # we're interested in the hunks that include changes to the outcome snippet (but we should ignore other edits in those hunks)
        for i, hunk in enumerate(patched_file):
            for line in hunk:
                line_data = {
                    "edit_snippet_line_number": line.target_line_no, # only present if the line was added or is context, 1-indexed
                    "outcome_snippet_line_number": line.source_line_no, # only present if the line was removed or is context, 1-indexed
                    "type": "context" if line.is_context else "removed" if line.is_removed else "added" if line.is_added else "unknown",
                    "text": line.value,
                    "hunk_id": i,
                }
                all_changed_lines.append(line_data)

                # check that the line actually comes from either file
                if line_data["outcome_snippet_line_number"] is not None:
                    assert(outcome_file_lines[line_data["outcome_snippet_line_number"] - 1].strip() == line_data["text"].strip())
                if line_data["edit_snippet_line_number"] is not None:
                    assert(snapshot_file_lines[line_data["edit_snippet_line_number"] - 1].strip() == line_data["text"].strip())

                # we add all context, removed lines that fall in the outcome snippet range
                if line_data["type"] not in {"context", "removed"}:
                    continue
                elif not(outcome_snippet_line_start <= line_data["outcome_snippet_line_number"] <= outcome_snippet_line_end):
                    continue
                elif line_data["type"] == "context":
                    line_data["from_start"] = line_data["outcome_snippet_line_number"] - outcome_snippet_line_start
                    line_data["from_end"] = outcome_snippet_line_end - line_data["outcome_snippet_line_number"]
                outcome_context_and_removed_lines.append(line_data)

    # sort line data by the outcome snippet line number - needed since we may have handled multiple hunks
    sort_lines = lambda x: x["edit_snippet_line_number"] if x["edit_snippet_line_number"] is not None else -1
    outcome_context_and_removed_lines = list(sorted(outcome_context_and_removed_lines, key=sort_lines))
    all_changed_lines = list(sorted(all_changed_lines, key=sort_lines))

    return outcome_context_and_removed_lines, all_changed_lines

def extract_edit_git(outcome_file, outcome_snippet_start_line, outcome_snippet_end_line, snapshot_file):
    """
    Extracts an edit from the snapshot using the Git diff algorithm. Returns None if no relevant changes were found.

    Args:
        outcome_file (string): The outcome file content
        outcome_snippet_start_line (int): The starting line number of the outcome snippet in the outcome file, 1-indexed
        outcome_snippet_end_line (int): The ending line number of the outcome snippet in the outcome file, 1-indexed
        snapshot_file (string): The snapshot file content
    """
    patch = diff_from_strings(outcome_file, snapshot_file)
    outcome_context_and_removed_lines, all_lines = parse_patch(
        patch, outcome_snippet_start_line, outcome_snippet_end_line, outcome_file, snapshot_file
    )

    # check if there were any relevant edits
    # It will be empty if: There were no changes in the outcome snippet or near it
    if len(outcome_context_and_removed_lines) == 0:
        return None

    # We now need to calculate the theoretical start and end of the edit snippet based on the information we have so far.
    edit_snippet_theoretical_start_index = min(
        [x['edit_snippet_line_number'] - x['from_start'] - 1 for x in outcome_context_and_removed_lines if x["type"] == "context"], 
        default=None
    )
    edit_snippet_theoretical_end_index = max(
        [x['edit_snippet_line_number'] + x['from_end'] - 1 for x in outcome_context_and_removed_lines if x["type"] == "context"], 
        default=None
    )

    # Build a mapping of context lines from the outcome file to the snapshot file
    lines_mapping = dict()
    hunk_ids = [x["hunk_id"] for x in outcome_context_and_removed_lines]
    for l in all_lines:
        if l["type"] == "context" and l["hunk_id"] in hunk_ids:
            lines_mapping[l["outcome_snippet_line_number"]] = l["edit_snippet_line_number"]
    sorted_outcome_lines = sorted(lines_mapping.keys())

    # Handle a deletion on the last line of the file
    max_entry = max(outcome_context_and_removed_lines, key=lambda d: d["outcome_snippet_line_number"])
    if max_entry["type"] == "removed" and max_entry["outcome_snippet_line_number"] == len(outcome_file.splitlines()):
        return None

    # Check each pair of context lines in this mapping, and see if there is a gap between them
    # If there is, we can assume the extra lines were lines that were edited/added
    # This will not work if the change is at the very start or end of the file, as it's not padded by context lines
    inferred_indices = set()
    for i in range(len(sorted_outcome_lines) - 1):
        o1 = sorted_outcome_lines[i]
        o2 = sorted_outcome_lines[i + 1]

        if o2 - o1 > 1 and not (o2 - 1 < outcome_snippet_start_line or o1 + 1 > outcome_snippet_end_line):
            # A relevant gap exists — assume lines between o1 and o2 (excluding o1 and o2) were edited
            s1 = lines_mapping[o1]
            s2 = lines_mapping[o2]
            inferred_indices.update(range(s1+1, s2))

    # Now that we know which lines in the snapshot file were edited, we extract those from all_lines, make them 0-indexed
    edit_snippet_line_indices = [
        l['edit_snippet_line_number'] - 1 for l in all_lines if l["type"] == "added" and l["edit_snippet_line_number"] in inferred_indices
    ]

    # This will be empty if the edits happened outside of the outcome snippet OR there were only deletions, no additions
    if len(edit_snippet_line_indices) == 0 and edit_snippet_theoretical_end_index == None:
        if len(outcome_context_and_removed_lines) > 0:
            return (outcome_context_and_removed_lines, "", "", all_lines, "")
        else:
            return None
    
    # calculate the start and end of the edit snippet - that is, the outcome snippet mapped over, not just the changed lines
    edit_snippet_start_index = min(x for x in edit_snippet_line_indices + [edit_snippet_theoretical_start_index] if x != None)
    edit_snippet_end_index = max(x for x in edit_snippet_line_indices + [edit_snippet_theoretical_end_index] if x != None)

    # Rebuild the edit snippet with just the relevant lines
    snapshot_lines = snapshot_file.splitlines()
    minimal_edit_snippet = '\n'.join([x for i, x in enumerate(snapshot_lines) if i in edit_snippet_line_indices])
    edit_snippet_full = '\n'.join(snapshot_lines[edit_snippet_start_index : edit_snippet_end_index + 1])
    edit_snippet_with_extra_context = '\n'.join(snapshot_lines[max(0, edit_snippet_start_index - 3) : edit_snippet_end_index + 1 + 3])

    # take care of situations where the edit snippet is empty
    if minimal_edit_snippet.strip() == "":
        # if outcome_context_and_removed_lines is all context lines, then we can return None
        if all(x["type"] == "context" for x in outcome_context_and_removed_lines):
            return None
    
    return (outcome_context_and_removed_lines, minimal_edit_snippet, edit_snippet_full, all_lines, edit_snippet_with_extra_context)