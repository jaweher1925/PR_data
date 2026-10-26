from firebase_admin import credentials, firestore
import argparse
import firebase_admin
import src.pipeline.extract_edit as extract_edit
import json
import logging
import os
import pandas as pd
import sys

# we want to consider outcomes that came from only these models
MODELS = [
    'gpt-4o-mini-2024-07-18',
    'llama-3.1-70b-instruct', 
    'llama-3.1-405b-instruct',
    'codestral-2405', 
    'deepseek-coder-fim', 
    'gemini-1.5-flash-002', 
    'gemini-1.5-pro-002',
    'claude-3-5-sonnet-20240620', 
    'gpt-4o-2024-08-06', 
    'gpt-4o-2024-11-20', 
    'qwen-2.5-coder-32b-instruct', 
    'claude-3-5-sonnet-20241022',
    'gemini-2.0-flash-exp', 
    'codestral-2501', 
    'deepseek-coder-v3-fim', 
    'anonymous-titan',
    'gemini-2.0-flash-001', 
    'gemini-2.0-pro-exp-02-05', 
    'anonymous-q', 
    'claude-3-7-sonnet-20250219'
]

# we want to consider outcomes that were written only in these languages
BERT_SCORE_THRESHOLDS = {"python": 0.68, "javascript": 0.68, "typescript": 0.68}
FILE_SIMILARITY_THRESHOLDS = {"python": 0.5, "javascript": 0.4, "typescript": 0.4}

# collections to load from firestore
COMPLETIONS_COLLECTIONS = ["autocomplete_completions_v1", "autocomplete_completions_5"]
OUTCOMES_COLLECTIONS = ["autocomplete_outcomes_v1", "autocomplete_outcomes_5"]
SNAPSHOTS_COLLECTIONS = ["all_autocomplete_completions_v1", "all_autocomplete_completions_5"]

# batch of user ids to process
USER_IDS_FILE = f"/data/group_data/code_edits/user_ids.json"

USER = os.environ["USER"]

def create_firestore_client():
    """
    Initialize Firebase (make sure you have the firebase-config.json file in the same directory)
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-config.json")
        firebase_admin.initialize_app(cred)

    return firestore.client()

# set up the Firestore client
db = create_firestore_client()

# set up logging as its better when running from a SLURM script
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger(__name__)

def firestore_collection_as_df(collection_name, docs):
    """
    Convert a Firestore collection to a Pandas DataFrame.
    """
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict["collection"] = collection_name  # Add collection name for reference
        data.append(doc_dict)

    return pd.DataFrame(data)

def fetch_firestore_collection(collection_name):
    """
    Given a collection name from Firebase, fetches all the associated data in the collection.
    Documentation: https://firebase.google.com/docs/firestore/query-data/get-data

    Args:
        collection_name (str): Name of the collection to query from Firebase
    """
    collection_ref = db.collection(collection_name)
    docs = collection_ref.stream()

    return firestore_collection_as_df(collection_name, docs)

def fetch_outcomes():
    """
    Fetch all outcomes from both collections in Firestore
    """
    outcome_dfs = [fetch_firestore_collection(collection_name) for collection_name in OUTCOMES_COLLECTIONS]
    outcomes_df = pd.concat(outcome_dfs, ignore_index=True)

    return outcomes_df

def fetch_completions_by_ids(completion_ids):
    """
    Fetch specific completions by their IDs from both collection databases in Firestore.

    Args:
        completion_ids (set): Set of completion IDs to fetch
    """
    completion_ids_list = list(completion_ids)

    # process in batches of 30 since that's the max count the "in" operator can work with in Firestore
    batch_size = 30
    for collection_name in COMPLETIONS_COLLECTIONS:
        log.info(f"\nProcessing collection: {collection_name}")

        for i in range(0, len(completion_ids_list), batch_size):
            # Firestore query for this batch of IDs
            batch_ids = completion_ids_list[i:i + batch_size]
            try:
                docs = db.collection(collection_name).where('completionId', 'in', batch_ids).stream()
                batch_df = firestore_collection_as_df(collection_name, docs)
                if not batch_df.empty:
                    yield batch_df
                    log.info(f"Processed batch of {len(batch_df)} documents from {collection_name}...")
            except Exception as e:
                log.warning(f"Error processing batch in {collection_name}: {e}")

def fetch_snapshots_by_user_ids_and_timestamps(user_ids_timestamps):
    """
    Fetch specific snapshots by their user IDs and timestamps from both collections in Firestore.
    
    Args:
        user_ids_timestamps (list of tuples): List of tuples of user IDs and timestamps to fetch
    """    
    for collection_name in SNAPSHOTS_COLLECTIONS:
        log.info(f"\nProcessing collection: {collection_name}")

        for user_id, timestamp in user_ids_timestamps:
            try:
                docs = db.collection(collection_name).where('userId', '==', user_id).where('timestamp', '>', timestamp).stream()
                batch_df = firestore_collection_as_df(collection_name, docs)
                if not batch_df.empty:
                    yield batch_df
                    log.info(f"Processed batch of {len(batch_df)} documents from {collection_name}...")
            except Exception as e:
                log.warning(f"Error processing batch in {collection_name}: {e}")

def get_matching_completion_ids_from_outcomes(outcomes_df):
    """
    Extract unique completion IDs from the outcomes DataFrame (completionItems column)

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    completion_ids = set()
    for completion_items in outcomes_df['completionItems']:
        for completion_item in completion_items:
            completion_ids.add(completion_item.get('completionId', None))

    completion_ids.discard(None)  # Remove None values if any
    log.info(f"Found {len(completion_ids)} unique completion IDs in outcomes")
    return completion_ids

def get_matching_completion_dfs_from_outcomes(outcomes_df):
    """
    Get matching completions for the outcomes DataFrame from Firestore.

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    completion_ids = get_matching_completion_ids_from_outcomes(outcomes_df)

    log.info("Starting to fetch matching completions...")
    completion_dfs_list = []
    for batch_df in fetch_completions_by_ids(completion_ids):
        completion_dfs_list.append(batch_df)

    if len(completion_dfs_list) > 0:
        completions_df = pd.concat(completion_dfs_list, ignore_index=True)
        log.info(f"\nTotal matching completions collected: {len(completions_df)}")
        log.info(f"Completions per collection: {completions_df['collection'].value_counts()}")

        # check for missing completions
        retrieved_ids = set(completions_df['completionId'])
        missing_ids = completion_ids - retrieved_ids
        if missing_ids:
            log.warning(f"\nWarning: {len(missing_ids)} completion IDs were not found in either collection")
    else:
        completions_df = pd.DataFrame()
        log.warning("No matching completions found.")

    return completions_df

def get_matching_snapshots_df_from_outcomes(outcomes_df):
    """
    Get matching snapshots for the outcomes DataFrame from Firestore based on user IDs and timestamps.

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    earliest_timestamps = (outcomes_df.groupby("userId")["timestamp"].min().reset_index())
    user_ids_timestamps = list(earliest_timestamps.itertuples(index=False, name=None))

    log.info("Starting to fetch matching snapshots...")
    snapshots_df_list = []
    for batch_df in fetch_snapshots_by_user_ids_and_timestamps(user_ids_timestamps):
        snapshots_df_list.append(batch_df)

    if len(snapshots_df_list) > 0:
        snapshots_df = pd.concat(snapshots_df_list, ignore_index=True)
        log.info(f"\nTotal matching snapshots collected: {len(snapshots_df)}")
        log.info(f"Completions per collection: {snapshots_df['collection'].value_counts()}")

        # check for missing snapshots
        retrieved_ids = set(snapshots_df['userId'])
        missing_ids = set(outcomes_df['userId']) - retrieved_ids
        if missing_ids:
            log.warning(f"\nWarning: {len(missing_ids)} user IDs were not found in either collection")
        
    else:
        snapshots_df = pd.DataFrame()
        log.warning("No matching completions found.")

    return snapshots_df

def get_users_from_outcomes(outcomes_df):
    """
    Returns list of all user IDs present in outcomes dataframe

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    users_df = outcomes_df.groupby("userId").size().reset_index(name="outcomes_count").sort_values(by="outcomes_count", ascending=False)
    return set(users_df['userId'])

def load_outcomes_and_user_ids():
    """
    Load outcomes from Firestore or a local CSV and return as a DataFrame.
    If loaded from Firestore, saves a local copy as well.
    Also return the set of user IDs loaded and filter outcomes to only those users.
    """

    # load outcomes
    if LOAD_FROM_FIRESTORE:
        # convert outcome completionsItems to strings before saving
        outcomes_df = fetch_outcomes()
        outcomes_df['completionItems'] = outcomes_df['completionItems'].apply(json.dumps)
    else:
        outcomes_df = pd.read_csv(OUTCOMES_INPUT_FILE)
    log.info(f"Loaded outcomes with {len(outcomes_df.columns)} columns")

    # load users
    num_users = 30
    if not os.path.isfile(USER_IDS_FILE):
        all_user_ids = list(get_users_from_outcomes(outcomes_df))
        user_id_batches = {
            f"batch{i}": all_user_ids[i*num_users:(i+1)*num_users] for i in range((len(all_user_ids) // num_users) + 1)
        }
        json.dump(user_id_batches, open(USER_IDS_FILE, "w"))
    else:
        user_id_batches = json.load(open(USER_IDS_FILE, "r"))
    user_ids = set(user_id_batches[f"batch{USER_IDS_BATCH_NUMBER}"])
    log.info(f"Loaded {len(user_ids)} user IDs from {USER_IDS_FILE}")
    
    # filter outcomes to only those users before saving
    outcomes_df = outcomes_df[outcomes_df['userId'].isin(user_ids)].reset_index(drop=True)
    outcomes_df.to_csv(OUTCOMES_OUTPUT_FILE, index=False)

    # convert completionItems back to list of dictionaries, filter columns
    outcomes_df['completionItems'] = outcomes_df['completionItems'].apply(json.loads)
    outcomes_df = outcomes_df[[
        'fileType', 'userId', 'privacy', 'acceptedIndex', 'pairId', 'completionItems', 'timestamp', 'collection',
        'completionId', 'pairCompletionId'
    ]]

    return outcomes_df, user_ids

def load_completions(outcomes_df):
    """
    Load completions from Firestore or a local CSV and return as a DataFrame.
    If loaded from Firestore, saves a local copy as well.
    Completions are matched based on the provided outcomes DataFrame.

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    if LOAD_FROM_FIRESTORE:
        completions_df = get_matching_completion_dfs_from_outcomes(outcomes_df)
        completions_df.to_csv(COMPLETIONS_OUTPUT_FILE, index=False)

        log.info(f"Saved completions with {len(completions_df.columns)} columns")
    else:
        completions_df = pd.read_csv(COMPLETIONS_INPUT_FILE)
        completion_ids = get_matching_completion_ids_from_outcomes(outcomes_df)
        filtered_completions_df = completions_df[completions_df['completionId'].isin(completion_ids)]

        log.info(f"Loaded original completions with {len(completions_df)} entries")
        log.info(f"Loaded filtered completions with {len(filtered_completions_df)} entries")

        completions_df = filtered_completions_df

    completions_df = completions_df[[
        'pairCompletionId', 'privacy', 'suffix', 'full_prefix', 'prefix', 'raw_completion', 'model', 'completionId', 
        'fileType', 'pairIndex', 'userId', 'full_suffix', 'completion', 'timestamp', 'collection'
    ]]
    return completions_df

def load_snapshots(outcomes_df):
    """
    Load snapshots from Firestore or a local CSV and return as a DataFrame.
    If loaded from Firestore, saves a local copy as well.
    Snapshots are matched based on the provided outcomes DataFrame.

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    if LOAD_FROM_FIRESTORE:
        snapshots_df = get_matching_snapshots_df_from_outcomes(outcomes_df)
        snapshots_df.to_csv(SNAPSHOTS_OUTPUT_FILE, index=False)

        log.info(f"Saved snapshots with {len(snapshots_df.columns)} columns")
    else:
        snapshots_df = pd.read_csv(SNAPSHOTS_INPUT_FILE)

        snapshots_df = snapshots_df[snapshots_df['userId'].isin(get_users_from_outcomes(outcomes_df))]

        earliest_timestamps = (outcomes_df.groupby("userId")["timestamp"].min().reset_index())
        user_ids_timestamps = dict(list(earliest_timestamps.itertuples(index=False, name=None)))
        snapshots_df = snapshots_df[snapshots_df.apply(lambda row: row['timestamp'] > user_ids_timestamps[row['userId']], axis=1)] 

        log.info(f"Loaded snapshots with {len(snapshots_df)} entries")


    snapshots_df = snapshots_df[[
        'privacy', 'pairCompletionId', 'suffix', 'model', 'full_prefix', 'completionId', 'pairIndex', 'userId', 'prefix', 'full_suffix',
        'completion', 'timestamp', 'collection', 'raw_completion'
    ]]
    return snapshots_df

def normalize(s):
    """
    Normalizes strings to replace Windows newlines with regular newlines

    Args:
        s (string): String to normalize newlines from
    """

    return s.replace('\r\n', '\n').replace('\r', '\n')

def construct_file_content(row, completions_df=None):
    """
    Construct the content of a file based on the row data and which df it is. 
    Returns an empty string if there's no valid associated file. 
    Args:
        row (pd.Series): The row data from the DataFrame
        completions_df (pd.DataFrame): DataFrame containing completions, if applicable
    """
    if completions_df is None or completions_df.empty:
        if 'prefix' in row:
            prefix_key, suffix_key = 'prefix', 'suffix'
        else:
            prefix_key, suffix_key = 'full_prefix', 'full_suffix'
        return normalize(f"{row[prefix_key] or ''}{row[suffix_key] or ''}")
    else:
        accepted_index = row['acceptedIndex']

        if pd.isna(accepted_index):
            return ""
        else:
            outcome_data = row['completionItems'][int(accepted_index)]
            accepted_completion_id = outcome_data['completionId']
            completion_data = completions_df.loc[completions_df['completionId'] == accepted_completion_id]
            completion_data = {} if completion_data.empty else completion_data.iloc[0].to_dict()

            prefix = completion_data.get('prefix', pd.NA)
            prefix = str(prefix) if pd.notna(prefix) else ""

            suffix = completion_data.get('suffix', pd.NA)
            suffix = str(suffix) if pd.notna(suffix) else ""

            return normalize(f"{prefix}{outcome_data.get('completion', '')}{suffix}")
        
def get_outcome(row, outcome_data_key, completions_df=None):
    """
    Extract the accepted outcome from the row based on the acceptedIndex

    Args:
        row (pd.Series): The row data from the DataFrame
        outcome_data_key (string): A key in the completion object for the corresponding outcome
        completions_df (pd.DataFrame): DataFrame containing completions, if applicable
    """
    accepted_index = row['acceptedIndex']
    
    if pd.notna(accepted_index):
        outcome_data = row['completionItems'][int(accepted_index)]
        outcome = outcome_data.get(outcome_data_key, '')

        if (outcome is pd.NA or outcome == '') and outcome_data_key == 'completion' and not completions_df.empty:
            # Let's see if we can pull this out directly from the dataframe instead...
            # This is rare but its good to be thorough
            completion_id = row["completionId"]

            if pd.notna(completion_id):
                completion_data = completions_df.loc[completions_df['completionId'] == completion_id]
                completion_data = {} if completion_data.empty else completion_data.iloc[0].to_dict()
                outcome_data = completion_data.get(outcome_data_key, pd.NA)
                outcome_data = outcome_data.values[0] if pd.notna(outcome_data) else ""

        outcome = "" if pd.isna(outcome) else outcome
        return normalize(outcome)
    else:
        # No valid completion snippet, return empty string
        return ''
    
def filter_outcomes(outcomes_df, completions_df):
    """
    Filter outcomes based on language, models, helper methods

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
        completions_df (pd.DataFrame): DataFrame containing completions
    """
    outcomes_df = outcomes_df[outcomes_df['completionItems'].apply(
        lambda items: len(items) >= 2 and items[0]['model'] in MODELS and items[1]['model'] in MODELS
    )]

    outcomes_df = outcomes_df[outcomes_df['fileType'].isin(BERT_SCORE_THRESHOLDS)]

    outcomes_df['fileContentWithOutcome'] = outcomes_df.apply(lambda x: construct_file_content(x, completions_df), axis=1)

    outcomes_df['outcome'] = outcomes_df.apply(lambda x: get_outcome(x, 'completion', completions_df), axis=1)
    outcomes_df['completionId'] = outcomes_df.apply(lambda x: get_outcome(x, 'completionId', completions_df), axis=1)
    outcomes_df = outcomes_df[(outcomes_df['completionId'].astype(str).str.strip() != '') & (outcomes_df['outcome'].astype(str).str.strip() != '')]

    outcomes_df['model'] = outcomes_df.apply(lambda x: get_outcome(x, 'model', completions_df), axis=1)

    return outcomes_df

def filter_completions(outcomes_df, completions_df):
    """
    Filter completions based on updated outcomes

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
    """
    completion_ids = get_matching_completion_ids_from_outcomes(outcomes_df)
    completions_df = completions_df[completions_df['completionId'].isin(completion_ids)]

    return completions_df

def filter_snapshots(outcomes_df, snapshots_df):
    """
    Filter snapshots based on user IDs
    We did create the snapshots based on user IDs and timestamps, but let's filter again to be sure since we've now filtered outcomes further

    Args:
        outcomes_df (pd.DataFrame): DataFrame containing outcomes
        snapshots_df (pd.DataFrame): DataFrame containing snapshots
    """
    snapshots_df = snapshots_df[snapshots_df['userId'].isin(get_users_from_outcomes(outcomes_df))]
    snapshots_df['fileContent'] = snapshots_df.apply(construct_file_content, axis=1)
    snapshots_df = snapshots_df[snapshots_df["fileContent"].astype(str).str.strip() != ""]

    return snapshots_df

def get_code_snippet_lines(code_snippet, code_file):
    """
    Returns the line numbers (1-indexed) in code_file where code_snippet matches exactly, inclusive.
    Returns the start and end of the entire file if no exact match is found.

    Args:
        code_snippet (string): Code snippet to match
        code_file (string): Code file to search 
    """
    snippet_lines = code_snippet.splitlines()
    file_lines = code_file.splitlines()

    for i in range(len(file_lines) - len(snippet_lines) + 1):
        # beginning and end need not be exact matches, could be substrings
        start_match = snippet_lines[0] in file_lines[i]
        end_match = snippet_lines[-1] in file_lines[i + len(snippet_lines) - 1]

        middle_match = all(file_lines[i + j] == snippet_lines[j] for j in range(1, len(snippet_lines) - 1))
        
        if start_match and end_match and middle_match:
            return i + 1, i + len(snippet_lines)
        
    if len(snippet_lines) != len(file_lines):
        log.warning("error getting code snippet lines")
        log.debug(f"Code snippet: {code_snippet}")
        log.debug(f"Code file: {code_file}")

    return 1, len(file_lines)

def drop_near_duplicates(df, my_outcomes_completion_ids, time_threshold=10):
    """
    Drops any near-duplicate snapshots based on timestamp within the given snapshot DataFrame. DataFrame rows associated 
    with known outcomes are saved.

    Args:
        df (DataFrame): DataFrame to de-duplicate rows for.
        my_outcomes_completion_ids (string[]): A list of completion IDs of known outcomes.
        time_threshold (int): The timethreshold used to determine near duplicates
    """

    df = df.sort_values(by="timestamp")
    time_diff = df["timestamp"].diff()

    # Keep rows spaced apart by time_threshold or more
    keep_mask = time_diff.isna() | (time_diff >= time_threshold)

    # Keep rows matching outcome_completion_id
    keep_mask |= df["completionId"].isin(my_outcomes_completion_ids)

    return df[keep_mask]

def is_similar(row, file, file_type, n=10, threshold=0.5):
    """
    Determines whether the snapshot row is a fuzzy match to the given outcome file. The fuzzy match is based on 
    the first N lines of the outcome file and snapshot file prefixes overlap past a threshold. 

    Args:
        row (Series): The snapshot row to determine similarity for.
        file (string): The outcome file to compare to.
        file_type (string): The outcome file language
        n (int): The number of lines to extract from the file prefixes.
        threshold (float): Threshold for similarity score

    Note: It might seem like this may not capture files that have edits at the very start very well
    However, the lower threshold actually accounts for this pretty well, and very few files don't make the cut when they should
    """
    def get_substantive_file_lines(this_file):
        lines = this_file.splitlines()
        return [l for l in lines if l.strip() != "" and len(l.strip()) >= 5]

    # Don't compute edits on identical files
    if row['fileContent'] == file:
        return False

    file_top_n_lines = set(get_substantive_file_lines(file)[:n])
    snapshot_top_n_lines = set(get_substantive_file_lines(row['fileContent'])[:n])
    
    overlap = len(file_top_n_lines & snapshot_top_n_lines) / max(len(file_top_n_lines), len(snapshot_top_n_lines), 1)
    return overlap >= FILE_SIMILARITY_THRESHOLDS[file_type]

def get_user_snapshots_df(snapshots_df, user_id, outcome_full_file, outcome_timestamp, known_completion_ids, file_type):
    """
    Filters the snapshots dataframe to extract the snapshots related to the current outcome.
    Returns an empty dataframe if no snapshots are found.

    Args:
        snapshots_df (DataFrame): The dataframe containing all of the code snapshots
        user_id (str): User ID associated with the outcome
        outcome_full_file (str): Full file containing outcome
        outcome_timestamp (int): Timestamp associated with the outcome
        known_completion_ids ([str]): Known completions IDs associated with the outcome
        file_type (str): File language
    """

    # Get all the snapshots from the user
    user_snapshots_df = snapshots_df[snapshots_df['userId'] == user_id]
    if user_snapshots_df.empty:
        log.warning(f"No snapshots found for user {user_id}")
        return user_snapshots_df

    # Filter snapshots based on the first 10 lines of the prefix
    user_snapshots_df = user_snapshots_df[user_snapshots_df.apply(lambda x: is_similar(x, outcome_full_file, file_type), axis=1)]
    if user_snapshots_df.empty:
        log.warning(f"No similar snapshots found for user {user_id}")
        return user_snapshots_df
    
    # Filter snapshots based on timestamp, keeping snapshots corresponding to known outcomes
    user_snapshots_df = drop_near_duplicates(user_snapshots_df, known_completion_ids)

    # Filter the snapshots that started AFTER this completion; this should filter the snapshot corresponding to the outcome
    user_snapshots_df = user_snapshots_df[user_snapshots_df["timestamp"] > outcome_timestamp]

    user_snapshots_df = user_snapshots_df.sort_values(by="timestamp", ascending=True)
    return user_snapshots_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI for the code edits project')
    parser.add_argument("--download-only", action="store_true", help="Only download the datasets, don't process them")
    parser.add_argument("--batch-number", type=int, required=True, help="Batch number of user IDs to process")
    parser.add_argument("--load-from-firestore", type=str, required=True, choices=['true', 'false'], help="Allow Firestore download")
    args = parser.parse_args()

    USER_IDS_BATCH_NUMBER = str(args.batch_number)
    LOAD_FROM_FIRESTORE = args.load_from_firestore.lower() == "true"
    
    # files to read from
    COMPLETIONS_INPUT_FILE = f"/data/user_data/{USER}/new_data/completions_batch{USER_IDS_BATCH_NUMBER}.csv"
    OUTCOMES_INPUT_FILE = f"/data/group_data/code_edits/outcomes_batch{USER_IDS_BATCH_NUMBER}.csv"
    SNAPSHOTS_INPUT_FILE = f"/data/user_data/{USER}/new_data/snapshots_batch{USER_IDS_BATCH_NUMBER}.csv"

    # files to write to 
    # note that these files will be overwritten if you keep the input and output file names the same
    COMPLETIONS_OUTPUT_FILE = f"./data/completions_batch{USER_IDS_BATCH_NUMBER}.csv"
    OUTCOMES_OUTPUT_FILE = f"./data/outcomes_batch{USER_IDS_BATCH_NUMBER}.csv"
    SNAPSHOTS_OUTPUT_FILE = f"./data/snapshots_batch{USER_IDS_BATCH_NUMBER}.csv"
    DATASET_FILE = f"./data/dataset_batch{USER_IDS_BATCH_NUMBER}.csv"

    global_outcomes_df, global_user_ids = load_outcomes_and_user_ids()
    log.info(global_outcomes_df.info())

    global_completions_df = load_completions(global_outcomes_df)
    log.info(global_completions_df.info())

    global_snapshots_df = load_snapshots(global_outcomes_df)
    log.info(global_snapshots_df.info())

    if args.download_only:
        log.info("Quitting after downloading files")
        sys.exit(0)

    global_outcomes_df = filter_outcomes(global_outcomes_df, global_completions_df)
    global_completions_df = filter_completions(global_outcomes_df, global_completions_df)
    global_snapshots_df = filter_snapshots(global_outcomes_df, global_snapshots_df)

    log.info(f"\nFiltered Outcomes DataFrame Length: {len(global_outcomes_df)}")
    log.info(f"\nFiltered Completions DataFrame Length: {len(global_completions_df)}")
    log.info(f"\nFiltered Snapshots DataFrame Length: {len(global_snapshots_df)}")

    # Prepare the edit dataset
    dataset_columns = [
        "outcome_pair_completion_id", "outcome_pair_id", "outcome_completion_id", "outcome", "outcome_snippet", "outcome_full_snippet", 
        "outcome_timestamp", "outcome_full_file", "outcome_snippet_start_line", "outcome_snippet_end_line",
        "edit_completion_id", "edit_snippet", "edit_full_file", "edit_timestamp", "edit_full_snippet",
        "all_changed_lines", "is_outcome", "user_id", "model", "codebert_score", "outcome_context_and_removed_lines", 
        "edit_snippet_with_extra_context"
    ]
    dataset_df = pd.DataFrame(columns=dataset_columns)
    dataset_df.to_csv(DATASET_FILE, mode='w', header=True, index=False)

    global_outcome_completion_ids = set(global_outcomes_df['completionId'].unique().tolist())
    error_reasons = {'no_outcome': 0, 'id_not_in_snapshots': 0, 'edit_info_is_none': 0, 'bert_low': 0, "outcome_at_start": 0}

    for i, outcome_row in global_outcomes_df.iterrows():
        outcome_data = []

        # Prepare relevant metadata (part 1)
        outcome_pair_completion_id = outcome_row["pairCompletionId"]
        outcome_pair_id = outcome_row["pairId"]
        outcome_completion_id = outcome_row["completionId"]
        outcome = outcome_row['outcome']
        outcome_timestamp = outcome_row['timestamp']
        outcome_full_file = outcome_row['fileContentWithOutcome']
        user_id = outcome_row['userId']
        model = outcome_row['model']

        # Get information about where the outcome is occurring in the file
        outcome_snippet_lines = outcome_full_file.splitlines()
        outcome_snippet_start_line, outcome_snippet_end_line = get_code_snippet_lines(outcome, outcome_full_file)
        outcome_snippet = '\n'.join(outcome_snippet_lines[outcome_snippet_start_line - 1: outcome_snippet_end_line])
        if outcome.strip() == '' or outcome_snippet.strip() == '':
            # No valid outcome that we can find, we continue
            error_reasons['no_outcome'] += 1
            continue
        elif outcome_snippet_start_line == 1:
            # we cannot handle this case
            error_reasons["outcome_at_start"] += 1
            continue

        # set up variables
        current_snippet = outcome_snippet
        current_snippet_file = outcome_full_file
        new_start_line, new_end_line = outcome_snippet_start_line, outcome_snippet_end_line

        # get relevant snapshots
        user_snapshots_df = get_user_snapshots_df(
            global_snapshots_df, user_id, outcome_full_file, outcome_timestamp, global_outcome_completion_ids, outcome_row['fileType']
        )
        if user_snapshots_df.empty:
            error_reasons['id_not_in_snapshots'] += 1
            continue

        for j, snapshot_row in user_snapshots_df.iterrows():
            snapshot_file = snapshot_row["fileContent"]

            # Get the edit if possible
            log.info(f"extracting edit for {outcome_pair_id}")
            edit_info = extract_edit.extract_edit_git(outcome_full_file, outcome_snippet_start_line, outcome_snippet_end_line, snapshot_file)
            if edit_info is None:
                error_reasons['edit_info_is_none'] += 1
                continue    
            outcome_context_and_removed_lines, minimal_edit_snippet, edit_snippet_full, all_lines, edit_snippet_with_extra_context = edit_info

            # safety check that the edit snippet is similar to the outcome snippet
            # we compare between the last edit snippet and the new snippet, since the code may diverge over time
            codebert_score = extract_edit.compute_codebert_similarity(minimal_edit_snippet.strip(), outcome_snippet.strip(), outcome_row['fileType'])
            if codebert_score < BERT_SCORE_THRESHOLDS[outcome_row['fileType']] and codebert_score > 0.0:
                log.warning(f"Edit snippet not similar enough to outcome snippet for completion {outcome_completion_id} of user {user_id}")
                error_reasons['bert_low'] += 1
                continue
            log.info('extracted valid edit')

            # Prepare relevant metadata (part 2)
            current_snippet = minimal_edit_snippet
            current_snippet_file = snapshot_file
            edit_completion_id = snapshot_row["completionId"]
            edit_snippet = current_snippet
            edit_full_snippet = edit_snippet_full
            edit_full_file = current_snippet_file
            edit_timestamp = snapshot_row["timestamp"]

            # compute the absolute start and end of the snippet over time
            for line in outcome_context_and_removed_lines:
                if line["type"] == "context" and line["outcome_snippet_line_number"] < new_start_line:
                    new_start_line = line["outcome_snippet_line_number"]
                elif line["type"] == "context" and line["outcome_snippet_line_number"] > new_end_line:
                    new_end_line = line["outcome_snippet_line_number"]

            # add to dataset
            dataset_row = {
                "outcome_pair_completion_id": outcome_pair_completion_id,
                "outcome_pair_id": outcome_pair_id,
                "outcome_completion_id": outcome_completion_id,
                "outcome": outcome,
                "outcome_snippet": outcome_snippet,
                "outcome_timestamp": outcome_timestamp,
                "outcome_full_file": outcome_full_file,
                "outcome_snippet_start_line": outcome_snippet_start_line,
                "outcome_snippet_end_line": outcome_snippet_end_line,
                "edit_completion_id": edit_completion_id,
                "edit_snippet": edit_snippet,
                "edit_full_file": edit_full_file,
                "edit_timestamp": edit_timestamp,
                "edit_full_snippet": edit_full_snippet,
                "is_outcome": edit_completion_id in global_outcome_completion_ids,
                "user_id": user_id,
                "model": model,
                "all_changed_lines": json.dumps(all_lines),
                "codebert_score": codebert_score,
                "outcome_context_and_removed_lines": json.dumps(outcome_context_and_removed_lines),
                "edit_snippet_with_extra_context": edit_snippet_with_extra_context
            }
            outcome_data.append(dataset_row)

        # Update the outcome snippet for all entries
        for od in outcome_data:
            od["outcome_full_snippet"] = '\n'.join(outcome_snippet_lines[new_start_line - 1: new_end_line])

        if len(outcome_data) == 0:
            log.warning(f"No edits extracted for completion {outcome_completion_id} of user {user_id}")
            continue
        else:
            temp_df = pd.DataFrame(outcome_data)
            temp_df.to_csv(DATASET_FILE, mode='a', header=False, index=False, columns=dataset_columns)

    log.info(f"Didn't extract edits for the following reasons: {error_reasons}")
    log.info("Done building dataset!")

    # Remove duplicates from dataset
    final_df = pd.read_csv(DATASET_FILE)
    duplicate_columns = [
        'outcome_pair_completion_id', 'outcome_pair_id', 'outcome_completion_id', 'is_outcome', 'user_id',
        'outcome', 'outcome_snippet', 'outcome_full_snippet', 'outcome_timestamp', 'edit_full_snippet',
        'outcome_full_file', 'outcome_snippet_start_line', 'outcome_snippet_end_line', 'edit_snippet'
    ]
    final_df_unique = final_df.sort_values(by='edit_timestamp').drop_duplicates(subset=duplicate_columns, keep='first')
    final_df_unique = final_df_unique[dataset_columns]

    log.info(f"Keeping {len(final_df_unique)} rows from {len(final_df)}")
    final_df_unique.to_csv(DATASET_FILE, index=False)
