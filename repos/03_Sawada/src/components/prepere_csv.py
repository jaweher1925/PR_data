import os
import pandas as pd


def prepere_csv(step, repo):
    """
    CSVから指定されたリポジトリと著者の行を削除する
    
    Args:
        step: 処理するステップ番号（1の場合のみstep1_all_files.csvを処理）
        repo: リポジトリ名（例: "owner/repo_name"）
    """
    if step == 1:
        # step1_all_files.csvのパス
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(script_dir, "../results/EASE-results/csv/step1_all_files.csv")
        
        # CSVを読み込み
        df = pd.read_csv(csv_path)
        
        # repository_nameがrepoと一致する行を削除
        df = df[df['repository_name'] != repo]
        
        # CSVに保存
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

