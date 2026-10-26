"""
repository.parquetからスター順にリポジトリリストを作成するスクリプト
"""

import pandas as pd


def create_repository_list():
    """
    repository.parquetファイルを読み込み、スター順に並べてrepository_list.csvを作成
    """
    # Parquetファイルの読み込み
    print("repository.parquetを読み込んでいます...")
    df = pd.read_parquet("dataset/repository.parquet")
    
    print(f"総リポジトリ数: {len(df)}")
    
    # full_nameをowner, repository_nameに分割
    df[['owner', 'repository_name']] = df['full_name'].str.split('/', n=1, expand=True)
    
    # 必要な列のみを選択
    result_df = df[['owner', 'repository_name', 'url', 'stars']].copy()
    
    # stars順に降順ソート
    result_df = result_df.sort_values(by='stars', ascending=False)
    
    # インデックスをリセット
    result_df = result_df.reset_index(drop=True)
    
    # CSVファイルとして保存
    output_path = "dataset/repository_list.csv"
    result_df.to_csv(output_path, index=False)
    
    print(f"\n{output_path}を作成しました")
    print(f"出力件数: {len(result_df)}")
    
    # 上位10件を表示
    print("\n=== 上位10件のリポジトリ ===")
    print(result_df.head(10).to_string())
    
    # 統計情報を表示
    print(f"\n=== 統計情報 ===")
    print(f"最大スター数: {result_df['stars'].max()}")
    print(f"最小スター数: {result_df['stars'].min()}")
    print(f"平均スター数: {result_df['stars'].mean():.2f}")
    print(f"中央値スター数: {result_df['stars'].median():.0f}")


if __name__ == '__main__':
    create_repository_list()
