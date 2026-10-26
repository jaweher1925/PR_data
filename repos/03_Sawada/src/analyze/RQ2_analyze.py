import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def analyze_rq2():
    """
    RQ2: AI作成ファイルの保守は誰が行っているのかを分析
    """
    # 2つの期間で分析を実行
    run_analysis(end_date="2026-01-31", suffix="_until_0131")

def run_analysis(end_date=None, suffix=""):
    """
    分析実行関数
    
    入力: results_v7_released_commits_restriction.csv
    出力: 
        - results/RQ2_results{suffix}.txt
    """
    
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "../../results/EASE-results/csv/results_v7_released_commits_restriction.csv")
    output_dir = os.path.join(script_dir, "../../results/EASE-results/summary")
    os.makedirs(output_dir, exist_ok=True)
    
    output_txt_path = os.path.join(output_dir, f"RQ2_results{suffix}.txt")

    df = pd.read_csv(input_dir)

    # 日付フィルタリング
    df['commit_date'] = pd.to_datetime(df['commit_date']).dt.tz_localize(None)
    df['file_creation_date'] = pd.to_datetime(df['file_creation_date']).dt.tz_localize(None)

    if end_date:
        end_date_dt = pd.to_datetime(end_date)
        print(f"分析期間: ～ {end_date}")
        df = df[df['commit_date'] <= end_date_dt].copy()

    # --- 全ファイル数のカウント（フィルタリング前） ---
    all_files_count = df[['repository_name', 'file_name']].drop_duplicates().shape[0]
    all_commits_count_raw = len(df)
    
    ai_files_total_df = df[df['file_created_by'] == 'AI']
    ai_files_total_count = ai_files_total_df[['repository_name', 'file_name']].drop_duplicates().shape[0]
    ai_commits_count_raw = len(ai_files_total_df)

    human_files_total_df = df[df['file_created_by'] == 'Human']
    human_files_total_count = human_files_total_df[['repository_name', 'file_name']].drop_duplicates().shape[0]
    human_commits_count_raw = len(human_files_total_df)

    # ファイル作成コミットを除外する
    mask_creation = df['commit_date'] == df['file_creation_date']
    creation_commit_indices = df[mask_creation].groupby(['repository_name', 'file_name']).head(1).index
    df = df.drop(creation_commit_indices)

    # 結果格納用リスト
    results_text = []
    results_text.append(f"RQ2 分析結果: AI作成ファイルの保守主体分析 (期間: ～ {end_date if end_date else '全期間'})")
    results_text.append("=" * 60)
    results_text.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 全体のファイル数
    results_text.append(f"総ファイル数: {all_files_count}")
    results_text.append(f"総コミット数 (CSV全体): {all_commits_count_raw}")
    results_text.append("")

    # ---------------------------------------------------------
    # 1. AI作成ファイルに対するコミット分析
    # ---------------------------------------------------------
    ai_files_df = df[df['file_created_by'] == 'AI']
    ai_commit_counts = ai_files_df['commit_created_by'].value_counts()
    
    ai_total_commits = ai_commit_counts.sum()
    ai_ai_commits = ai_commit_counts.get('AI', 0)
    ai_human_commits = ai_commit_counts.get('Human', 0)
    
    ai_ai_ratio = (ai_ai_commits / ai_total_commits * 100) if ai_total_commits > 0 else 0
    ai_human_ratio = (ai_human_commits / ai_total_commits * 100) if ai_total_commits > 0 else 0
    
    results_text.append("1. AI作成ファイルに対するコミット内訳")
    results_text.append("-" * 40)
    results_text.append(f"対象ファイル数: {ai_files_total_count}")
    results_text.append(f"総コミット数 (CSV全体): {ai_commits_count_raw}")
    results_text.append(f"分析対象コミット数 (作成除外後): {ai_total_commits}")
    results_text.append(f"AIによるコミット: {ai_ai_commits} ({ai_ai_ratio:.2f}%)")
    results_text.append(f"人間によるコミット: {ai_human_commits} ({ai_human_ratio:.2f}%)")
    results_text.append("")

    # ---------------------------------------------------------
    # 2. 人間作成ファイルに対するコミット分析
    # ---------------------------------------------------------
    human_files_df = df[df['file_created_by'] == 'Human']
    human_commit_counts = human_files_df['commit_created_by'].value_counts()

    print(human_commit_counts)
    
    human_total_commits = human_commit_counts.sum()
    human_ai_commits = human_commit_counts.get('AI', 0)
    human_human_commits = human_commit_counts.get('Human', 0)
    
    human_ai_ratio = (human_ai_commits / human_total_commits * 100) if human_total_commits > 0 else 0
    human_human_ratio = (human_human_commits / human_total_commits * 100) if human_total_commits > 0 else 0
    
    results_text.append("2. 人間作成ファイルに対するコミット内訳")
    results_text.append("-" * 40)
    results_text.append(f"対象ファイル数: {human_files_total_count}")
    results_text.append(f"総コミット数 (CSV全体): {human_commits_count_raw}")
    results_text.append(f"分析対象コミット数 (作成除外後): {human_total_commits}")
    results_text.append(f"AIによるコミット: {human_ai_commits} ({human_ai_ratio:.2f}%)")
    results_text.append(f"人間によるコミット: {human_human_commits} ({human_human_ratio:.2f}%)")
    results_text.append("")

    # 結果保存
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_text))
    
    print(f"分析完了。結果を保存しました: {output_txt_path}")

if __name__ == "__main__":
    analyze_rq2()
