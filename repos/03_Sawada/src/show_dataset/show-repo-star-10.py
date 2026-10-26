import pandas as pd

def check_parquet(df):

    print("\n=== 列名一覧 ===")
    print(df.columns.tolist())

    # データ確認
    print("リポジトリデータ件数:", len(df))
    
    # スター数でソートして上位10個を表示
    print("\n=== スター数上位10リポジトリ ===")
    top_repos = df.nlargest(10, 'stars')[['full_name', 'stars', 'language', 'forks', 'open_issues']]
    
    for i, (_, repo) in enumerate(top_repos.iterrows(), 1):
        print(f"{i:2d}. {repo['full_name']}")
        print(f"    Stars: {repo['stars']:,}")
        print(f"    Language: {repo['language']}")
        print(f"    Forks: {repo['forks']:,}")
        print(f"    Open Issues: {repo['open_issues']:,}")
        print()

if __name__ == '__main__':
    # df = pd.read_parquet("../data_list/all_pull_request_local.parquet")
    df = pd.read_parquet("../data_list/all_repository_local.parquet")
    # df = pd.read_parquet("../data_list/pr_commits_local.parquet")
    # df = pd.read_parquet("../data_list/pr_commit_details_local.parquet")

    check_parquet(df)