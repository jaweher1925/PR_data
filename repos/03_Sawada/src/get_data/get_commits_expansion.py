"""
コミットデータ拡張プログラム
results_v5.csvのファイルに対して2025/11/11以降のコミットを取得して追加
"""

import os
import pandas as pd
from datetime import datetime
import time
from github import Github
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import pipeline

# componentsフォルダからインポート
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.AI_check import ai_check
from components.check_network import retry_with_network_check

# .envファイル読み込み
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '..', '.env')
load_dotenv(dotenv_path)

# コミット分類用モデル初期化
pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")
tokenizer = pipe.tokenizer


class CommitExpansion:
    def __init__(self, github_token):
        """初期化"""
        self.github_token = github_token
        self.g = Github(github_token)
        
        # 入出力パス
        project_root = os.path.join(script_dir, '../..')
        self.input_csv = os.path.join(project_root, 'results/EASE-results/results_v5.csv')
        self.output_csv = os.path.join(project_root, 'results/EASE-results/csv/results_v7_released_commits_restriction.csv')
        
        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
        print(f"入力ファイル: {self.input_csv}")
        print(f"出力ファイル: {self.output_csv}")
    
    def prepare_prompt(self, commit_message: str, git_diff: str, context_window: int = 1024):
        """コミット分類用プロンプト作成"""
        prompt_head = "<s>[INST] <<SYS>>\nYou are a commit classifier based on commit message and code diff.Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n**feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n**fix**: Code changes aim to fix bugs and faults within the codebase.\n**perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n**style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n**refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n**docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n**test**: Code changes that modify test files, including the addition or updating of tests.\n**ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n**build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n**chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n<</SYS>>\n\n"
        prompt_head_encoded = tokenizer.encode(prompt_head, add_special_tokens=False)

        prompt_message = f"- given commit message:\n{commit_message}\n"
        prompt_message_encoded = tokenizer.encode(prompt_message, max_length=64, truncation=True, add_special_tokens=False)

        prompt_diff = f"- given commit diff: \n{git_diff}\n"
        remaining_length = (context_window - len(prompt_head_encoded) - len(prompt_message_encoded) - 6)
        prompt_diff_encoded = tokenizer.encode(prompt_diff, max_length=remaining_length, truncation=True, add_special_tokens=False)

        prompt_end = tokenizer.encode(" [/INST]", add_special_tokens=False)
        return tokenizer.decode(prompt_head_encoded + prompt_message_encoded + prompt_diff_encoded + prompt_end)

    def classify_commit(self, commit_message: str, git_diff: str, context_window: int = 1024):
        """コミット分類"""
        if not pipe or not tokenizer:
            return "model_not_available"

        try:
            prompt = self.prepare_prompt(commit_message, git_diff, context_window)
            result = pipe(prompt, max_new_tokens=10, pad_token_id=pipe.tokenizer.eos_token_id)
            label = result[0]["generated_text"].split()[-1]
            return label
        except Exception as e:
            print(f"分類エラー: {e}")
            return "classification_error"
    
    @retry_with_network_check
    def get_repo(self, repo_name):
        """リポジトリ取得"""
        return self.g.get_repo(repo_name)
    
    @retry_with_network_check
    def get_new_commits(self, repo, file_path):
        """2025/11/1以降のコミット取得"""
        since_date = datetime(2025, 11, 1)
        commits = repo.get_commits(path=file_path, since=since_date)
        commit_list = list(commits)
        return commit_list
    
    @retry_with_network_check
    def get_commit_details(self, repo, commit_sha):
        """コミット詳細取得"""
        commit = repo.get_commit(commit_sha)
        return commit
    
    @retry_with_network_check
    def get_commit_patch(self, repo, commit_sha, file_path):
        """特定ファイルのpatch取得"""
        commit = repo.get_commit(commit_sha)
        for file in commit.files:
            if file.filename == file_path:
                return file.patch or "", file.changes
        return "", 0
    
    def process_commit(self, repo, commit, file_path, file_info):
        """コミット情報処理"""
        try:
            # コミット基本情報
            commit_sha = commit.sha
            author_name = commit.commit.author.name or "Unknown"
            commit_date = commit.commit.author.date.isoformat()
            message = commit.commit.message
            
            # コミット作成者（author + committer）
            all_authors = [author_name]
            if commit.commit.committer and commit.commit.committer.name:
                committer_name = commit.commit.committer.name
                if committer_name != author_name and committer_name not in all_authors:
                    all_authors.append(committer_name)
            
            # AI判定
            is_ai, commit_created_by = ai_check(all_authors)
            
            # コミット全体の変更行数
            commit_changed_lines = sum(file.changes for file in commit.files)
            
            # ファイル固有の変更行数取得
            patch, file_specific_changed_lines = self.get_commit_patch(repo, commit_sha, file_path)
            
            # コミット分類
            commit_classification = self.classify_commit(message, patch)
            
            # データ作成
            commit_data = {
                **file_info,  # ファイル情報をコピー
                'commit_hash': commit_sha,
                'commit_authors': ', '.join(all_authors),
                'commit_created_by': commit_created_by,
                'commit_changed_lines': commit_changed_lines,
                'commit_date': commit_date,
                'commit_classification': commit_classification,
                'file_specific_changed_lines': file_specific_changed_lines
            }
            
            time.sleep(0.05)  # API rate limit対策
            return commit_data
            
        except Exception as e:
            print(f"  エラー（コミット処理 {commit.sha[:8]}）: {e}")
            return None
    
    def run(self):
        """メイン処理"""
        print("\n" + "="*80)
        print("コミットデータ拡張開始")
        print("="*80)
        
        # 入力CSV読み込み
        df_v5 = pd.read_csv(self.input_csv)
        print(f"入力データ: {len(df_v5)}行")
        
        # リポジトリ×ファイル単位でグループ化（CSV上の順序を保持）
        grouped = df_v5.groupby(['repository_name', 'file_name'], sort=False)
        # 元のCSVの順序でグループをソート
        group_first_idx = {key: group.index.min() for key, group in grouped}
        sorted_groups = sorted(grouped, key=lambda x: group_first_idx[x[0]])
        total_files = len(sorted_groups)
        print(f"処理対象: {total_files}ファイル")
        
        # 出力用データフレーム初期化
        if os.path.exists(self.output_csv):
            df_output = pd.read_csv(self.output_csv)
            print(f"既存の出力ファイルを読み込み: {len(df_output)}行")
            # 処理済みファイルを特定
            processed = set(df_output.groupby(['repository_name', 'file_name']).groups.keys())
            print(f"処理済み: {len(processed)}ファイル")
        else:
            df_output = pd.DataFrame()
            processed = set()
        
        # 各ファイルを処理（CSV上の順序で）
        for (repo_name, file_name), group in tqdm(sorted_groups, desc="ファイル処理"):
            # 処理済みならスキップ
            if (repo_name, file_name) in processed:
                continue
            
            print(f"\n処理中: {repo_name} / {file_name}")
            
            try:
                # 1. 既存データをコピー
                file_data = group.copy()
                
                # ファイル情報（コミットに依存しない情報）
                file_info = {
                    'repository_name': repo_name,
                    'file_name': file_name,
                    'file_creators': group.iloc[0]['file_creators'],
                    'file_created_by': group.iloc[0]['file_created_by'],
                    'file_line_count': group.iloc[0]['file_line_count'],
                    'file_creation_date': group.iloc[0]['file_creation_date'],
                    'file_commit_count': group.iloc[0]['file_commit_count']
                }
                
                # 2. 新しいコミット取得
                repo = self.get_repo(repo_name)
                new_commits = self.get_new_commits(repo, file_name)
                
                # 3. 既存のコミットハッシュセット
                existing_hashes = set(group['commit_hash'].values)
                
                # 4. 新しいコミットを処理
                new_data = []
                for commit in new_commits:
                    if commit.sha not in existing_hashes:
                        commit_data = self.process_commit(repo, commit, file_name, file_info)
                        if commit_data:
                            new_data.append(commit_data)
                
                
                # 5. データ結合（既存+新規）
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    file_data = pd.concat([file_data, new_df], ignore_index=True)
                    # 日付順にソート
                    file_data = file_data.sort_values('commit_date')
                
                # 6. 出力データに追加
                df_output = pd.concat([df_output, file_data], ignore_index=True)
                
                # 7. 即座に保存（動作1と4のタイミング）
                df_output.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
                
            except Exception as e:
                print(f"  エラー（ファイル処理）: {e}")
                continue
        
        print("\n" + "="*80)
        print("処理完了")
        print(f"出力: {self.output_csv}")
        print(f"総行数: {len(df_output)}行")
        print("="*80)


def main():
    """メイン実行"""
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        print("エラー: GITHUB_TOKENが設定されていません")
        return
    
    expander = CommitExpansion(github_token)
    expander.run()


if __name__ == "__main__":
    main()
