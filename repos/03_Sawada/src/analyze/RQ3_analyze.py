import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def analyze_commit_classification(end_date=None, suffix=""):
    # ファイルパスの定義
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "../../results/EASE-results/csv/results_v7_released_commits_restriction.csv")
    output_dir = os.path.join(script_dir, "../../results/EASE-results/summary")
    output_txt = os.path.join(output_dir, f'RQ3_results{suffix}.txt')

    df = pd.read_csv(input_dir)

    # 日付比較のためにdatetime型に変換
    df['commit_date'] = pd.to_datetime(df['commit_date']).dt.tz_localize(None)
    df['file_creation_date'] = pd.to_datetime(df['file_creation_date']).dt.tz_localize(None)

    if end_date:
        end_date_dt = pd.to_datetime(end_date)
        print(f"分析期間: ～ {end_date}")
        df = df[df['commit_date'] <= end_date_dt].copy()
    
    # 作成日時とコミット日時が一致する行（作成コミット）を除外
    original_count = len(df)
    mask_creation = df['commit_date'] == df['file_creation_date']
    creation_commit_indices = df[mask_creation].groupby(['repository_name', 'file_name']).head(1).index
    df = df.drop(creation_commit_indices)

    # AIと人間が作成したファイルのデータをフィルタリング
    ai_df = df[df['file_created_by'] == 'AI']
    human_df = df[df['file_created_by'] == 'Human']

    # コミット分類の分析
    ai_counts = ai_df['commit_classification'].value_counts()
    human_counts = human_df['commit_classification'].value_counts()

    ai_total = len(ai_df)
    human_total = len(human_df)

    # 出力テキストの準備
    output_lines = []
    
    output_lines.append("AI Created Files")
    output_lines.append(f"Total Commits: {ai_total}")
    for commit_type, count in ai_counts.items():
        percentage = (count / ai_total) * 100 if ai_total > 0 else 0
        output_lines.append(f"{commit_type}: {count} ({percentage:.2f}%)")
    
    output_lines.append("\nHuman Created Files Commit Classification:")
    output_lines.append(f"Total Commits: {human_total}")
    for commit_type, count in human_counts.items():
        percentage = (count / human_total) * 100 if human_total > 0 else 0
        output_lines.append(f"{commit_type}: {count} ({percentage:.2f}%)")

    # テキストファイルへの書き込み
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Results written to {output_txt}")

if __name__ == "__main__":
    # 2つの期間で分析を実行
    analyze_commit_classification(end_date="2026-01-31", suffix="_until_0131")
