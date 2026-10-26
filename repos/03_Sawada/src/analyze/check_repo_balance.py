import pandas as pd
import os

def check_repository_balance():
    """
    successful_repository_list.csv の ai_file_count (目標AIファイル数) と
    results_v4.csv に含まれる Human 作成ファイル数 (分析済みHumanファイル数) を比較し、
    Humanファイル数が目標AIファイル数より少ないリポジトリを特定する。
    """
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    success_list_path = os.path.join(script_dir, "../dataset/successful_repository_list.csv")
    results_path = os.path.join(script_dir, "../results/results_v4.csv")
    
    # ファイル存在確認
    if not os.path.exists(success_list_path):
        print(f"エラー: ファイルが見つかりません: {success_list_path}")
        return
    if not os.path.exists(results_path):
        print(f"エラー: ファイルが見つかりません: {results_path}")
        return

    print(f"読み込み中: {success_list_path}")
    try:
        df_success = pd.read_csv(success_list_path)
    except Exception as e:
        print(f"CSV読み込みエラー (successful_repository_list.csv): {e}")
        return

    print(f"読み込み中: {results_path}")
    try:
        df_results = pd.read_csv(results_path)
    except Exception as e:
        print(f"CSV読み込みエラー (results_v4.csv): {e}")
        return

    # successful_repository_list.csv の処理
    # owner と repository_name を結合してキーにする
    df_success['full_name'] = df_success['owner'] + '/' + df_success['repository_name']
    target_ai_counts = df_success.set_index('full_name')['ai_file_count']

    # results_v4.csv の処理
    # ファイル単位でユニークにする
    unique_files = df_results[['repository_name', 'file_name', 'file_created_by']].drop_duplicates()
    
    # Human作成ファイルをカウント
    human_files = unique_files[unique_files['file_created_by'] == 'Human']
    human_counts = human_files.groupby('repository_name').size()

    print("\n" + "="*100)
    print(f"{'Repository Name':<40} | {'Target AI Count':<15} | {'Actual Human Count':<18} | {'Diff':<5}")
    print("-" * 100)

    found_repos_count = 0
    
    # 比較
    for repo_name, target_ai in target_ai_counts.items():
        actual_human = human_counts.get(repo_name, 0)
        
        if actual_human < target_ai:
            diff = target_ai - actual_human
            print(f"{repo_name:<40} | {target_ai:<15} | {actual_human:<18} | -{diff:<5}")
            found_repos_count += 1

    print("-" * 100)
    print(f"条件に該当するリポジトリ数 (Human < Target AI): {found_repos_count}")
    print("="*100)

if __name__ == "__main__":
    check_repository_balance()
