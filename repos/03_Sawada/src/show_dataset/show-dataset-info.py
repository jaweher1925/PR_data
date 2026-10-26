"""
Parquetファイルの内容確認用スクリプト
"""

import pandas as pd

def check_parquet(df):

    print("\n=== 列名一覧 ===")
    print(df.columns.tolist())

    # データ確認
    print("PRデータ件数:", len(df))

    print("\n=== 内容表示（all_repository_local） ===")
    license_counts = df['license'].value_counts()
    print(license_counts)
    print()
    full_name_counts = df['full_name'].value_counts()
    print(full_name_counts)
    print()
    language_counts = df['language'].value_counts()
    print(language_counts)
    print()
    forks_counts = df['forks'].value_counts()
    print(forks_counts)
    print()
    stars_counts = df['stars'].value_counts()
    print(stars_counts)
    print()

    # print("\n=== 内容表示（all_pull_request_local） ===")
    # author_counts = df['agent'].value_counts()
    # print(author_counts)
    # print()
    # commiter_counts = df['user'].value_counts()
    # print(commiter_counts)
    # print()
    # closed_at_counts = df['closed_at'].value_counts()
    # print(closed_at_counts)
    # print()
    # created_at_counts = df['created_at'].value_counts()
    # print(created_at_counts)
    # print()
    # state_counts = df['state'].value_counts()
    # print(state_counts)
    # print()

if __name__ == '__main__':
    # df = pd.read_parquet("dataset/all_pull_request_local.parquet")
    # df = pd.read_parquet("dataset/all_repository_local.parquet")
    # df = pd.read_parquet("dataset/pr_commits_local.parquet")
    # df = pd.read_parquet("dataset/pr_commit_details_local.parquet")
    df = pd.read_parquet("dataset/repository.parquet")


    check_parquet(df)

