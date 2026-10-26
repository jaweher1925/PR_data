"""
RQ1統合プログラム: AIコミット分析システム（GitHub API版）
機能: リポジトリをクローンせずにGitHub APIで分析
"""

import sys
import os # ファイルパスを扱うためのライブラリ
import pandas as pd # データフレームを扱うためのライブラリ
from datetime import datetime, timedelta # 日付取得や時間の計算のためのライブラリ
import numpy as np # 数値計算を行うためのライブラリ
from transformers import pipeline # 事前学習したモデルを扱うためのライブラリ
import requests # HTTPリクエストを扱うためのライブラリ
from github import Github # Github APIを扱うためのライブラリ
from dotenv import load_dotenv # .envファイルを読み込むためのライブラリ
from tqdm import tqdm # プログレスバーを表示するためのライブラリ
import time
import base64 # Base64エンコード/デコードを行うためのライブラリ

# src ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# componentsフォルダからインポート
from components.AI_check import ai_check
from components.check_network import retry_with_network_check, check_network_connectivity

# srcフォルダ内の.envファイルを読み込む
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path)

pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")
tokenizer = pipe.tokenizer

class RQ1AnalyzerAPI:
    def __init__(self, repo_name_full, github_token=None):
        """
        repo_name_full: 'owner/repo' 形式のリポジトリ名
        github_token: GitHub Personal Access Token
        """
        self.repo_name_full = repo_name_full
        self.repo_name = repo_name_full.split('/')[-1]
        self.github_token = github_token
        
        if not self.github_token:
            raise ValueError("GitHub tokenが必要です。.envファイルにGITHUB_TOKENを設定してください。")
        
        # GitHub API初期化
        self.g = Github(self.github_token)
        self.repo = self.g.get_repo(repo_name_full)
        
        # 出力ディレクトリ
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.final_output_dir = os.path.join(script_dir, "../data_list/RQ1/final_result")
        os.makedirs(self.final_output_dir, exist_ok=True)
        
        # 成功リポジトリリストのCSVパス
        dataset_dir = os.path.join(script_dir, "../dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        self.successful_repos_csv = os.path.join(dataset_dir, "successful_repository_list.csv")
        
        print(f"リポジトリ接続成功: {repo_name_full}")
        print(f"スター数: {self.repo.stargazers_count}, フォーク数: {self.repo.forks_count}")

    def save_successful_repository(self, ai_file_count):
        """分析成功したリポジトリ情報をCSVに記録
        
        Args:
            ai_file_count: AI作成ファイル数
        """
        owner, repo_name = self.repo_name_full.split('/')
        
        # 既存のCSVを読み込むか、新規作成
        if os.path.exists(self.successful_repos_csv):
            df = pd.read_csv(self.successful_repos_csv)
        else:
            df = pd.DataFrame(columns=['owner', 'repository_name', 'ai_file_count'])
        
        # 新しいレコードを追加
        new_record = pd.DataFrame([{
            'owner': owner,
            'repository_name': repo_name,
            'ai_file_count': ai_file_count
        }])
        
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.successful_repos_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 成功リポジトリを記録: {self.successful_repos_csv}")

    @retry_with_network_check
    def get_all_commits_with_file_additions_api(self):
        """GitHub APIで2025/1/1～2025/7/31の全コミットを取得（ファイル追加のみ）
        
        Returns:
            tuple: (commits_data, total_commits_count)
        """
        print(f"=== GitHub APIで全コミット取得中 ===")
        
        commits_data = []
        
        try:
            # 2025/1/1～2025/7/31の期間のコミットを取得
            since_date = datetime(2025, 1, 1)
            until_date = datetime(2025, 7, 31)
            print(f"コミット取得期間: {since_date.date()} ～ {until_date.date()}")
            print("全コミットを取得中...")
            
            commits = self.repo.get_commits(since=since_date, until=until_date)
            commits_list = list(commits)
            total_commits_count = len(commits_list)
            print(f"総コミット数: {total_commits_count}件")
            
            # 全コミットを処理
            for commit in tqdm(commits_list, desc="コミット処理"):
                try:
                    # コミット情報取得
                    commit_sha = commit.sha
                    author_name = commit.commit.author.name or "Unknown"
                    author_email = commit.commit.author.email or "unknown@example.com"
                    commit_date = commit.commit.author.date.isoformat()
                    message = commit.commit.message
                    
                    # コミットアカウントのみ取得（author + committer）
                    all_authors = [author_name]
                    
                    # committerも追加（authorと異なる場合）
                    if commit.commit.committer and commit.commit.committer.name:
                        committer_name = commit.commit.committer.name
                        if committer_name != author_name and committer_name not in all_authors:
                            all_authors.append(committer_name)
                    
                    # 追加されたファイルを検索
                    added_files = []
                    for file in commit.files:
                        if file.status == 'added':
                            added_files.append(file.filename)
                    
                    if added_files:
                        commit_info = {
                            'hash': commit_sha,
                            'author_name': author_name,
                            'author_email': author_email,
                            'all_authors': all_authors,  # 全作成者リスト
                            'date': commit_date,
                            'message': message,
                            'added_files': added_files
                        }
                        commits_data.append(commit_info)
                    
                    # API rate limit対策
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"\nコミット処理エラー {commit.sha[:8]}: {e}")
                    continue
            
            print(f"\nコミット取得完了: 2025/1/1～2025/7/31の総コミット数={total_commits_count}件, ファイル追加コミット={len(commits_data)}件")
            return commits_data, total_commits_count
            
        except Exception as e:
            print(f"GitHub API エラー: {e}")
            return [], 0

    def step1_find_added_files(self, target_ai_files=10, target_human_files=10):
        """ステップ1: ファイル追加分析（全コミット取得 + ランダムサンプリング）
        
        Args:
            target_ai_files: 目標AI作成ファイル数（最大10）
            target_human_files: 目標Human作成ファイル数（最大10）
        """
        print("\n=== ステップ1: ファイル追加分析 (全コミット取得 + ランダムサンプリング) ===")
        print(f"目標: AI={target_ai_files}件, Human={target_human_files}件")
        print("※ 2025/1/1～2025/7/31の全コミットを取得します")

        # 全コミット取得
        commits_data, total_commits = self.get_all_commits_with_file_additions_api()
        
        if total_commits == 0:
            print("指定期間のコミットが存在しません")
            return None, 'no_commits_90days'
        
        # 全データ作成
        all_csv_data = []
        for commit in commits_data:
            is_ai, ai_type = ai_check(commit['all_authors'])
            author_type = "AI" if is_ai else "Human"
            
            for file_path in commit['added_files']:
                all_csv_data.append({
                    'commit_hash': commit['hash'],
                    'commit_date': commit['date'],
                    'added_file': file_path,
                    'author_type': author_type,
                    'ai_type': ai_type,
                    'is_ai_generated': is_ai,
                    'all_authors': commit['all_authors'],  # ← この行を追加
                    'author_name': commit['author_name'],
                    'author_email': commit['author_email'],
                    'commit_message': commit['message']
                })
        
        # DataFrameに変換
        df = pd.DataFrame(all_csv_data)
        
        if len(df) == 0:
            print(f"ファイル追加コミット未発見（総コミット数: {total_commits}件）")
            return None, 'no_file_additions'
        
        # AI/人間ファイルを分離
        ai_df = df[df['author_type'] == 'AI'].copy()
        human_df = df[df['author_type'] == 'Human'].copy()
        
        ai_count = len(ai_df)
        human_count = len(human_df)
        
        print(f"\n取得結果 - AI作成ファイル: {ai_count}件, 人間作成ファイル: {human_count}件")
        
        # AI作成ファイルが見つからなかった場合
        if ai_count == 0:
            print("警告: AI作成ファイルが見つかりませんでした")
            return df, 'no_ai_files'
        
        # AI作成ファイルをランダムに最大10個選択
        num_ai_files = min(target_ai_files, ai_count)
        if ai_count > num_ai_files:
            ai_sampled = ai_df.sample(n=num_ai_files, random_state=None)
            print(f"AI作成ファイル: {ai_count}件から{num_ai_files}件をランダム選択")
        else:
            ai_sampled = ai_df
            print(f"AI作成ファイル: 全{ai_count}件を使用")
        
        # 人間作成ファイルを同数ランダムに選択
        num_human_files = num_ai_files  # AI作成ファイルと同数
        if human_count >= num_human_files:
            human_sampled = human_df.sample(n=num_human_files, random_state=None)
            print(f"人間作成ファイル: {human_count}件から{num_human_files}件をランダム選択")
        else:
            human_sampled = human_df
            print(f"警告: 人間作成ファイルが不足（{human_count}件のみ）")
        
        # 結合
        sampled_df = pd.concat([ai_sampled, human_sampled], ignore_index=True)
        
        print(f"\n最終選択 - 総計: {len(sampled_df)}件 (AI: {len(ai_sampled)}件, 人間: {len(human_sampled)}件)")
        
        return sampled_df, 'success'

    @retry_with_network_check
    def get_file_commits_api(self, file_path):
        """GitHub APIで特定ファイルのコミット履歴取得（2025/10/31まで）"""
        try:
            # 2025/10/31までのコミットを取得
            until_date = datetime(2025, 10, 31, 23, 59, 59)
            commits = self.repo.get_commits(path=file_path, until=until_date)
            commit_logs = []
            
            for commit in commits:
                author_name = commit.commit.author.name or "Unknown"
                
                # コミットアカウントのみ取得（author + committer）
                all_authors = [author_name]
                
                # committerも追加（authorと異なる場合）
                if commit.commit.committer and commit.commit.committer.name:
                    committer_name = commit.commit.committer.name
                    if committer_name != author_name and committer_name not in all_authors:
                        all_authors.append(committer_name)
                
                commit_logs.append({
                    'hash': commit.sha,
                    'date': commit.commit.author.date.isoformat(),
                    'author': author_name,
                    'all_authors': all_authors,  # 全作成者リスト
                    'email': commit.commit.author.email or "unknown@example.com",
                    'message': commit.commit.message
                })
                
                time.sleep(0.05)  # API rate limit対策
            
            return commit_logs
            
        except Exception as e:
            print(f"ファイル履歴取得エラー {file_path}: {e}")
            return []

    @retry_with_network_check
    def get_file_creation_info(self, file_path):
        """ファイルの作成情報を取得（最初のコミット）"""
        try:
            commits = self.repo.get_commits(path=file_path)
            commit_list = list(commits)
            if commit_list:
                # 最後のコミット（最初のコミット）を取得
                first_commit = commit_list[-1]
                author_name = first_commit.commit.author.name or "Unknown"
                
                # コミットアカウントのみ取得（author + committer）
                all_authors = [author_name]
                
                # committerも追加（authorと異なる場合）
                if first_commit.commit.committer and first_commit.commit.committer.name:
                    committer_name = first_commit.commit.committer.name
                    if committer_name != author_name and committer_name not in all_authors:
                        all_authors.append(committer_name)
                
                return {
                    'author_name': author_name,
                    'all_authors': all_authors,  # 全作成者リスト
                    'all_creator_names': all_authors,  # CSV出力用（ファイル作成者名）
                    'creation_date': first_commit.commit.author.date.isoformat(),
                    'commit_count': len(commit_list)
                }
            return None
        except Exception as e:
            print(f"ファイル作成情報取得エラー {file_path}: {e}")
            return None

    def get_file_line_count(self, file_path, commit_sha):
        """ファイルの行数を取得"""
        try:
            # 特定のコミットでのファイル内容を取得
            content = self.repo.get_contents(file_path, ref=commit_sha)
            if content.encoding == 'base64':
                decoded_content = base64.b64decode(content.content).decode('utf-8', errors='ignore')
                return len(decoded_content.splitlines())
            return 0
        except Exception as e:
            print(f"ファイル行数取得エラー {file_path}: {e}")
            return 0

    def get_files_by_author_type(self, df, target_ai_count=10, target_human_count=10):
        """AI/Humanファイル選択（ランダムサンプリング版）"""
        # AI作成ファイルと人間作成ファイルを分離
        ai_df = df[df['author_type'] == 'AI'].copy()
        human_df = df[df['author_type'] == 'Human'].copy()
        
        # ランダムサンプリング
        ai_files = []
        human_files = []
        
        # AI作成ファイルをランダムに選択
        if len(ai_df) > 0:
            sample_size_ai = min(target_ai_count, len(ai_df))
            ai_sampled = ai_df.sample(n=sample_size_ai, random_state=None)
            
            for _, row in ai_sampled.iterrows():
                ai_files.append({
                    'commit_hash': row['commit_hash'],
                    'added_file': row['added_file'],
                    'author_type': row['author_type'],
                    'ai_type': row['ai_type']
                })
        
        # 人間作成ファイルをランダムに選択
        if len(human_df) > 0:
            sample_size_human = min(target_human_count, len(human_df))
            human_sampled = human_df.sample(n=sample_size_human, random_state=None)
            
            for _, row in human_sampled.iterrows():
                human_files.append({
                    'commit_hash': row['commit_hash'],
                    'added_file': row['added_file'],
                    'author_type': row['author_type'],
                    'ai_type': row['ai_type']
                })
        
        # 同数に調整（数が小さい方合わせる）
        min_count = min(len(ai_files), len(human_files))
        ai_files = ai_files[:min_count]
        human_files = human_files[:min_count]
        
        print(f"ファイル数調整: AI={len(ai_files)} Human={len(human_files)} (同数に調整)")
        
        return ai_files + human_files

    def step2_find_commit_changed_files(self, df):
        """ステップ2: コミット履歴分析（API版）- エラーファイルを除外して同数に調整"""
        print("\n=== ステップ2: コミット履歴分析 (API版) ===")
        
        selected_files = self.get_files_by_author_type(df)
        ai_count = sum(1 for f in selected_files if f['author_type'] == 'AI')
        print(f"選択ファイル: {len(selected_files)} (AI:{ai_count} Human:{len(selected_files)-ai_count})")
        
        # ファイル情報記録用（成功したファイルのみ）
        file_info_records = []
        # 成功したファイルのリスト（AI/Human別に管理）
        successful_ai_files = []
        successful_human_files = []
        
        results = []
        
        # AI作成ファイルを先に処理
        ai_files = [f for f in selected_files if f['author_type'] == 'AI']
        print(f"\nAI作成ファイルの処理開始: {len(ai_files)}件")
        
        for file_info in tqdm(ai_files, desc="AI作成ファイル処理"):
            
            file_path = file_info['added_file']
            commit_hash = file_info['commit_hash']
            author_type = file_info['author_type']
            
            # ファイル作成情報を取得
            creation_info = self.get_file_creation_info(file_path)
            if creation_info:
                line_count = self.get_file_line_count(file_path, commit_hash)
                
                # ファイル情報を記録
                file_info_record = {
                    'repository_name': self.repo_name_full,
                    'file_name': file_path,
                    'all_creator_names': creation_info['all_creator_names'],
                    'line_count': line_count,
                    'created_by': author_type,
                    'creation_date': creation_info['creation_date'],
                    'commit_count': creation_info['commit_count']
                }
                file_info_records.append(file_info_record)
                successful_ai_files.append(file_path)
                
                # コミット履歴取得
                commit_logs = self.get_file_commits_api(file_path)
                
                if commit_logs:
                    for log in commit_logs:
                        is_ai, ai_type = self.is_ai_generated_commit(log['all_authors'])
                        results.append({
                            'original_commit_type': author_type,
                            'original_commit_hash': commit_hash,
                            'file_path': file_path,
                            'commit_hash': log['hash'],
                            'commit_date': log['date'],
                            'author': log['author'],
                            'all_authors': log['all_authors'],
                            'is_ai_generated': is_ai,
                            'ai_type': ai_type
                        })
            else:
                # 情報取得失敗時はスキップ
                tqdm.write(f"  警告: ファイル情報取得失敗 - {file_path} (スキップ)")
        
        # 人間作成ファイルを処理（成功したAI作成ファイル数に合わせる）
        human_files = [f for f in selected_files if f['author_type'] == 'Human']
        target_human_count = len(successful_ai_files)  # AI成功数に合わせる
        
        print(f"\n人間作成ファイルの処理開始: {len(human_files)}件中{target_human_count}件を処理")
        
        human_processed = 0
        for file_info in tqdm(human_files, desc="人間作成ファイル処理", total=target_human_count):
            if human_processed >= target_human_count:
                tqdm.write(f"  目標数{target_human_count}件に到達 - 残りの人間ファイルはスキップ")
                break
            
            file_path = file_info['added_file']
            commit_hash = file_info['commit_hash']
            author_type = file_info['author_type']
            
            # ファイル作成情報を取得
            creation_info = self.get_file_creation_info(file_path)
            if creation_info:
                line_count = self.get_file_line_count(file_path, commit_hash)
                
                # ファイル情報を記録
                file_info_record = {
                    'repository_name': self.repo_name_full,
                    'file_name': file_path,
                    'all_creator_names': creation_info['all_creator_names'],
                    'line_count': line_count,
                    'created_by': author_type,
                    'creation_date': creation_info['creation_date'],
                    'commit_count': creation_info['commit_count']
                }
                file_info_records.append(file_info_record)
                successful_human_files.append(file_path)
                human_processed += 1
                
                # コミット履歴取得
                commit_logs = self.get_file_commits_api(file_path)
                
                if commit_logs:
                    for log in commit_logs:
                        is_ai, ai_type = ai_check(log['all_authors'])
                        results.append({
                            'original_commit_type': author_type,
                            'original_commit_hash': commit_hash,
                            'file_path': file_path,
                            'commit_hash': log['hash'],
                            'commit_date': log['date'],
                            'author': log['author'],
                            'all_authors': log['all_authors'],
                            'is_ai_generated': is_ai,
                            'ai_type': ai_type
                        })
            else:
                # 情報取得失敗時はスキップ
                tqdm.write(f"  警告: ファイル情報取得失敗 - {file_path} (スキップ)")
        
        # 最終調整：AI/Humanの成功数を同数にする
        final_count = min(len(successful_ai_files), len(successful_human_files))
        
        if final_count < len(successful_ai_files) or final_count < len(successful_human_files):
            print(f"\n最終調整: AI={len(successful_ai_files)}件, Human={len(successful_human_files)}件 → 両方{final_count}件に調整")
            
            # 各タイプから最初のfinal_count件のみ残す
            keep_ai_files = set(successful_ai_files[:final_count])
            keep_human_files = set(successful_human_files[:final_count])
            keep_files = keep_ai_files | keep_human_files
            
            # resultsをフィルタリング
            results = [r for r in results if r['file_path'] in keep_files]
            
            # file_info_recordsをフィルタリング
            file_info_records = [f for f in file_info_records if f['file_name'] in keep_files]
        
        print(f"\n最終結果: AI={final_count}件, Human={final_count}件, 総ファイル数={final_count*2}件")
        print(f"総コミット数: {len(results)}件")
        
        # ファイル情報をインスタンス変数として保存
        self.file_info_records = file_info_records
        
        return pd.DataFrame(results) if results else None

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

    def fetch_message_and_diff(self, commit_sha):
        """GitHub API経由でコミット情報取得"""
        try:
            commit = self.repo.get_commit(commit_sha)
            
            if commit.parents:
                parent_sha = commit.parents[0].sha
                diff_url = self.repo.compare(parent_sha, commit_sha).diff_url
                return commit.commit.message, requests.get(diff_url).text
            return commit.commit.message, ""
        except Exception as e:
            print(f"GitHub取得エラー: {e}")
            return None, None

    def get_commit_changed_lines(self, commit_sha):
        """コミットの変更行数を取得"""
        try:
            commit = self.repo.get_commit(commit_sha)
            return commit.stats.additions + commit.stats.deletions
        except Exception as e:
            print(f"変更行数取得エラー {commit_sha[:8]}: {e}")
            return 0
    
    def save_results_to_csv_v4(self, df_classified):
        """結果をresults_v4.csvに保存（上書き・コミット単位）"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "../data_list/RQ1/final_result")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "results_v4.csv")
        
        # 既存のCSVを読み込む（存在する場合）
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
        else:
            existing_df = pd.DataFrame()
        
        # 新しいデータを作成
        csv_data = []
        
        for _, row in df_classified.iterrows():
            file_path = row['file_path']
            
            # ファイル情報を取得
            file_info = None
            for record in self.file_info_records:
                if record['file_name'] == file_path:
                    file_info = record
                    break
            
            if file_info is None:
                continue
            
            # コミット変更行数を取得
            commit_hash = row['commit_hash']
            changed_lines = 0
            if commit_hash != 'No commits found':
                changed_lines = self.get_commit_changed_lines(commit_hash)
            
            csv_data.append({
                'repository_name': self.repo_name_full,
                'file_name': file_path,
                'file_creators': ', '.join(file_info['all_creator_names']),
                'file_created_by': file_info['created_by'],
                'file_line_count': file_info['line_count'],
                'file_creation_date': file_info['creation_date'],
                'file_commit_count': file_info['commit_count'],
                'commit_hash': commit_hash,
                'commit_authors': ', '.join(row['all_authors']) if isinstance(row['all_authors'], list) else '',
                'commit_created_by': 'AI' if row['is_ai_generated'] else 'Human',
                'commit_changed_lines': changed_lines,
                'commit_date': row['commit_date'],
                'commit_classification': row.get('classification_label', 'not_classified')
            })
        
        # 新しいデータをDataFrameに変換
        new_df = pd.DataFrame(csv_data)
        
        # 既存データと結合して上書き保存
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"CSV保存完了: {csv_path}（総行数: {len(combined_df)}行）")

    def step3_classify_commits(self, df):
        """ステップ3: コミット分類"""
        print("\n=== ステップ3: コミット分類 ===")
        
        if not pipe:
            print("分類モデル利用不可 - 分類スキップ")
            df['classification_label'] = 'not_classified'
            return df
        
        results = []
        
        try:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="コミット分類"):
                commit_sha = row['commit_hash']
                
                base_result = {
                    'original_commit_type': row['original_commit_type'],
                    'commit_hash': commit_sha,
                    'file_path': row['file_path'],
                    'commit_date': row['commit_date'],
                    'author': row['author'],
                    'all_authors': row['all_authors'],  # ← この行を追加
                    'is_ai_generated': row['is_ai_generated'],
                    'ai_type': row['ai_type']
                }
                
                if commit_sha == 'No commits found':
                    base_result['classification_label'] = 'no_commits'
                else:
                    try:
                        message, diff = self.fetch_message_and_diff(commit_sha)
                        base_result['classification_label'] = self.classify_commit(message, diff) if message and diff else 'fetch_error'
                    except Exception as e:
                        tqdm.write(f"エラー {commit_sha[:8]}: {e}")
                        base_result['classification_label'] = 'error'
                
                results.append(base_result)
            
            print("分類処理完了")
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"\n✗✗✗ 致命的エラー（Segmentation fault等）: {type(e).__name__} ✗✗✗")
            print(f"詳細: {str(e)}")
            print("このリポジトリをスキップします")
            return None

    def analyze_subset(self, subset_df, label):
        """サブセット分析（日本語出力）"""
        if len(subset_df) == 0:
            return [f"{label} コミットが見つかりませんでした"]
        
        analysis = [f"{label.upper()} コミット分析", "-" * 30]
        
        # 1. ファイル統計
        commits_per_file = subset_df.groupby('file_path').size()
        analysis.extend([
            f"1. ファイルごとのコミット数: 平均={commits_per_file.mean():.2f} 標準偏差={commits_per_file.std():.2f} 最小={commits_per_file.min()} 最大={commits_per_file.max()}",
            ""
        ])
        
        # 2. 分類ラベル分布
        analysis.append("2. 分類ラベルの分布:")
        overall_labels = subset_df['classification_label'].value_counts()
        for lbl, count in overall_labels.items():
            analysis.append(f"   {lbl}: {count}件 ({count/len(subset_df)*100:.1f}%)")
        analysis.append("")
        
        # 3. コミット頻度分析
        analysis.append("3. コミット頻度分析:")
        file_frequencies = []
        for file_path in subset_df['file_path'].unique():
            file_commits = subset_df[subset_df['file_path'] == file_path].copy()
            file_commits['commit_date'] = pd.to_datetime(file_commits['commit_date'])
            if len(file_commits) > 1:
                dates = file_commits['commit_date'].sort_values()
                time_diffs = [(dates.iloc[i] - dates.iloc[i-1]).days 
                            for i in range(1, len(dates)) if (dates.iloc[i] - dates.iloc[i-1]).days > 0]
                if time_diffs:
                    file_frequencies.extend(time_diffs)
        
        if file_frequencies:
            analysis.append(f"   平均間隔: {np.mean(file_frequencies):.1f}日 (標準偏差: {np.std(file_frequencies):.1f})")
        else:
            analysis.append("   頻度分析に十分なデータがありません")
        analysis.append("")
        
        # 4. AI判定統計
        ai_count = len(subset_df[subset_df['is_ai_generated'] == True])
        analysis.extend([
            f"4. 作成者: AI={ai_count}件 ({ai_count/len(subset_df)*100:.1f}%) 人間={len(subset_df)-ai_count}件 ({(len(subset_df)-ai_count)/len(subset_df)*100:.1f}%)",
            ""
        ])
        
        # 5. AIツール/モデルの分布（AI起源の場合のみ）
        if label == "AI起源" and 'ai_type' in subset_df.columns:
            analysis.append("5. 使用されたAIツール/モデルの分布:")
            first_commits = subset_df.drop_duplicates(subset=['file_path'], keep='first')
            ai_type_counts = first_commits['ai_type'].value_counts()
            for tool, count in ai_type_counts.items():
                if tool != 'N/A':
                    analysis.append(f"   {tool}: {count}件 ({count/len(first_commits)*100:.1f}%)")
            analysis.append("")
        
        return analysis

    def step4_analyze_commit_data(self, df):
        """ステップ4: データ分析（日本語出力）"""
        print("\n=== ステップ4: データ分析 ===")
        
        df['commit_date'] = pd.to_datetime(df['commit_date'])
        ai_df = df[df['original_commit_type'] == 'AI']
        human_df = df[df['original_commit_type'] == 'Human']
        
        # 統計結果構築
        ai_generated = len(df[df['is_ai_generated'] == True])
        results = [
            f"{self.repo_name} コミット分析結果", "=" * 50, "",
            "全体統計", "-" * 20,
            f"総コミット数: {len(df)}件",
            f"AI起源: {len(ai_df)}件 ({len(ai_df)/len(df)*100:.1f}%)",
            f"人間起源: {len(human_df)}件 ({len(human_df)/len(df)*100:.1f}%)",
            f"ユニークファイル数: {df['file_path'].nunique()}件",
            f"AI生成コミット: {ai_generated}件 ({ai_generated/len(df)*100:.1f}%)", ""
        ]
        
        # サブセット分析
        results.extend(self.analyze_subset(ai_df, "AI起源"))
        results.extend(self.analyze_subset(human_df, "人間起源"))
        
        # 比較分析
        if len(ai_df) > 0 and len(human_df) > 0:
            ai_cpf = ai_df.groupby('file_path').size()
            human_cpf = human_df.groupby('file_path').size()
            results.extend([
                "比較分析", "-" * 20,
                f"ファイルごとの平均コミット数 - AI: {ai_cpf.mean():.2f}件 人間: {human_cpf.mean():.2f}件",
                f"最も活発なファイル - AI: {ai_cpf.idxmax()} ({ai_cpf.max()}件) 人間: {human_cpf.idxmax()} ({human_cpf.max()}件)"
            ])
        
        print(f"分析完了: {self.repo_name}")
        print("\n".join(results[:10]) + "\n...")
        return results

    def run_full_analysis(self):
        """全分析実行（API版）- エラーハンドリング強化版"""
        print(f"=== RQ1分析開始 (API版): {self.repo_name_full} ===")
        
        try:
            # step1: ファイル追加分析
            print("\n--- ステップ1: ファイル追加分析 ---")
            df_additions, step1_status = self.step1_find_added_files()
            if df_additions is None or len(df_additions) == 0:
                if step1_status == 'no_commits_90days':
                    print("⚠ ステップ1失敗: 7/31以前のコミットが存在しません")
                    return None, 'no_commits_90days'
                elif step1_status == 'no_file_additions':
                    print("⚠ ステップ1失敗: ファイル追加コミットが見つかりませんでした")
                    return None, 'no_file_additions'
                else:
                    print("⚠ ステップ1失敗: データが取得できませんでした")
                    return None, 'step1_failed'
            
            print(f"✓ ステップ1完了: {len(df_additions)}件のファイル追加を検出")
            
            # AI作成ファイルが見つからなかった場合
            if step1_status == 'no_ai_files':
                print("⚠ 警告: AI作成ファイルが見つかりませんでした")
                return None, 'no_ai_files'
            
            # step2: コミット履歴分析
            print("\n--- ステップ2: コミット履歴分析 ---")
            df_history = self.step2_find_commit_changed_files(df_additions)
            if df_history is None or len(df_history) == 0:
                print("⚠ ステップ2: コミット履歴が取得できませんでした")
                return None, 'step2_failed'
            
            print(f"✓ ステップ2完了: {len(df_history)}件のコミット履歴を取得")
            
            # step3: コミット分類
            print("\n--- ステップ3: コミット分類 ---")
            df_classified = self.step3_classify_commits(df_history)
            if df_classified is None or len(df_classified) == 0:
                print("⚠ ステップ3: コミット分類ができませんでした")
                return None, 'step3_failed'
            
            print(f"✓ ステップ3完了: {len(df_classified)}件のコミットを分類")
            
            # results_v4.csvに保存
            print("\n--- CSV保存 (results_v4.csv) ---")
            self.save_results_to_csv_v4(df_classified)
            print("✓ CSV保存完了")
            
            # AI作成ファイル数をカウント
            ai_file_count = len(df_additions[df_additions['is_ai_generated'] == True])
            
            # 成功リポジトリ情報を記録
            print("\n--- 成功リポジトリ記録 ---")
            self.save_successful_repository(ai_file_count)
            
            # 個別レポートは出力せず、統合分析でまとめて出力
            print(f"\n✓✓✓ 完了: {self.repo_name} ✓✓✓")
            
            # 結果を返す
            return {
                'df_additions': df_additions,
                'df_history': df_history,
                'df_classified': df_classified
            }, 'success'
            
        except Exception as e:
            print(f"\n✗✗✗ 予期しないエラー発生: {self.repo_name_full} ✗✗✗")
            print(f"エラー詳細: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f'exception: {type(e).__name__}'


def analyze_multiple_repositories(repo_list, start_index=0, num_repos=100):
    """複数リポジトリの分析を実行 - 成功数ベース版"""
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        print("エラー: GitHub tokenが設定されていません")
        print(".envファイルにGITHUB_TOKENを設定してください")
        return
    
    print(f"=" * 80)
    print(f"RQ1 複数リポジトリ分析 (GitHub API版)")
    print(f"=" * 80)
    print(f"開始位置: {start_index + 1}番目のリポジトリから")
    print(f"目標分析数: {num_repos}リポジトリ（成功基準）")
    print(f"GitHub API: OK")
    print(f"=" * 80)
    
    start_time = datetime.now()
    all_results = []
    all_classifications = []
    failed_repos = []
    
    # 成功したリポジトリがnum_repos個になるまで続ける
    idx = start_index
    while len(all_results) < num_repos and idx < len(repo_list):
        repo_info = repo_list[idx]
        repo_name_full = f"{repo_info['owner']}/{repo_info['repository_name']}"
        
        print(f"\n{'='*80}")
        print(f"[試行: {idx+1}] [成功: {len(all_results)}/{num_repos}] {repo_name_full}")
        print(f"スター数: {repo_info['stars']:,}")
        print(f"{'='*80}")
        
        try:
            # RQ1Analyzerの初期化
            print("リポジトリに接続中...")
            analyzer = RQ1AnalyzerAPI(repo_name_full, github_token)
            
            # 分析実行
            result, status = analyzer.run_full_analysis()
            
            if result is not None:
                all_results.append({
                    'repo': repo_name_full,
                    'stars': repo_info['stars'],
                    'analyzer': analyzer,
                    'data': result
                })
                all_classifications.append(result['df_classified'])
                print(f"\n✓✓✓ [成功: {len(all_results)}/{num_repos}] {repo_name_full} 分析成功 ✓✓✓")
                
            else:
                # 失敗理由を詳細に記録
                reason_map = {
                    'no_commits_90days': '90日以前のコミットが存在しない',
                    'no_file_additions': '90日以前にファイル追加コミットなし',
                    'no_ai_files': 'AI作成ファイルが見つからない',
                    'step1_failed': 'ステップ1失敗',
                    'step2_failed': 'ステップ2失敗',
                    'step3_failed': 'ステップ3失敗'
                }
                reason = reason_map.get(status, status)
                
                failed_repos.append({
                    'repo': repo_name_full,
                    'stars': repo_info['stars'],
                    'reason': reason
                })
                print(f"\n✗✗✗ {repo_name_full} 分析失敗: {reason} ✗✗✗")
                print(f"→ 次のリポジトリに進みます... (残り成功必要数: {num_repos - len(all_results)})")
                
        except Exception as e:
            failed_repos.append({
                'repo': repo_name_full,
                'stars': repo_info['stars'],
                'reason': f'{type(e).__name__}: {str(e)}'
            })
            print(f"\n✗✗✗ {repo_name_full} エラー発生 ✗✗✗")
            print(f"エラー詳細: {type(e).__name__}: {str(e)}")
            print(f"→ 次のリポジトリに進みます... (残り成功必要数: {num_repos - len(all_results)})")
        
        idx += 1
    
    # 結果サマリー
    print(f"\n{'='*80}")
    print("分析結果サマリー")
    print(f"{'='*80}")
    print(f"成功: {len(all_results)}件（目標: {num_repos}件）")
    print(f"失敗: {len(failed_repos)}件")
    print(f"試行総数: {idx}件")
    
    if len(all_results) < num_repos:
        print(f"\n⚠ 警告: 目標の{num_repos}件に達しませんでした（リポジトリリスト不足）")
    
    if failed_repos:
        print(f"\n失敗したリポジトリ:")
        for failed in failed_repos:
            print(f"  - {failed['repo']}: {failed['reason']}")
    
    total_time = datetime.now() - start_time
    print(f"\n{'='*80}")
    print(f"総処理時間: {total_time}")
    print(f"{'='*80}")


def main():
    """メイン実行"""
    # CSVからリポジトリリスト読み込み
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../dataset/repository_list.csv")
    
    print(f"リポジトリリスト読み込み: {csv_path}")
    repo_df = pd.read_csv(csv_path)
    repo_list = repo_df.to_dict('records')

    # 開始位置（上から何番目のリポジトリから始めるか）
    # 例: start_repo = 0 なら1番目から、start_repo = 100 なら101番目から開始
    start_repo = 2
    
    # 分析対象リポジトリ数
    num_repos = 100
    
    print(f"総リポジトリ数: {len(repo_list)}件")
    print(f"開始位置: {start_repo + 1}番目")
    print(f"分析対象: {num_repos}件")
    
    # 複数リポジトリ分析実行
    analyze_multiple_repositories(repo_list, start_repo, num_repos)


if __name__ == "__main__":
    main()
