import pandas as pd

def check_parquet(df):
    print("\n=== 列名一覧 ===")
    print(df.columns.tolist())

    # データ確認
    print("PRデータ件数:", len(df))
    
    # AIエージェントによるPRをフィルタリング（agentが'ai'または何らかのAI識別子の場合）
    # agentカラムの値を確認
    print("\n=== agentカラムの値確認 ===")
    if 'agent' in df.columns:
        agent_counts = df['agent'].value_counts()
        print(agent_counts)
        
        # AIによるPRのみフィルタリング（agentカラムにAI関連の値がある場合）
        # ここでは一旦全てのPRを対象とし、必要に応じて条件を調整
        ai_prs = df[df['agent'].notna()]  # agent情報があるもの
        
        if len(ai_prs) > 0:
            print(f"\nAI関連PRデータ件数: {len(ai_prs)}")
            
            # 各AIエージェント別にランキングを表示
            unique_agents = ai_prs['agent'].unique()
            
            for agent in unique_agents:
                agent_prs = ai_prs[ai_prs['agent'] == agent]
                repo_pr_counts = agent_prs['repo_url'].value_counts()
                
                print(f"\n=== {agent} による PRが多いリポジトリ（上位10件） ===")
                print(f"{agent} のPR総数: {len(agent_prs)}件")
                
                for i, (repo_url, count) in enumerate(repo_pr_counts.head(10).items(), 1):
                    # repo_urlからリポジトリ名を抽出
                    repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                    print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
                
                print(f"\n=== {agent} による PRが少ないリポジトリ（下位10件） ===")
                for i, (repo_url, count) in enumerate(repo_pr_counts.tail(10).items(), 1):
                    # repo_urlからリポジトリ名を抽出
                    repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                    print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
            
            # 全AI PRを対象にした総合ランキング
            repo_pr_counts = ai_prs['repo_url'].value_counts()
            
            print("\n=== 全AIエージェントによるPRが多いリポジトリ（上位10件） ===")
            for i, (repo_url, count) in enumerate(repo_pr_counts.head(10).items(), 1):
                # repo_urlからリポジトリ名を抽出
                repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
            
            print("\n=== 全AIエージェントによるPRが少ないリポジトリ（下位10件） ===")
            for i, (repo_url, count) in enumerate(repo_pr_counts.tail(10).items(), 1):
                # repo_urlからリポジトリ名を抽出
                repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
                
        else:
            print("AI関連のPRが見つかりませんでした。")
            print("全PRを対象にリポジトリ別PR数を表示します。")
            
            # 全PRでリポジトリ別PR数をカウント
            repo_pr_counts = df['repo_url'].value_counts()
            
            print("\n=== PRが多いリポジトリ（上位10件） ===")
            for i, (repo_url, count) in enumerate(repo_pr_counts.head(10).items(), 1):
                repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
            
            print("\n=== PRが少ないリポジトリ（下位10件） ===")
            for i, (repo_url, count) in enumerate(repo_pr_counts.tail(10).items(), 1):
                repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
                print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
    else:
        print("agentカラムが見つかりません。")
        
        # agentカラムがない場合、全PRでリポジトリ別PR数をカウント
        repo_pr_counts = df['repo_url'].value_counts()
        
        print("\n=== PRが多いリポジトリ（上位10件） ===")
        for i, (repo_url, count) in enumerate(repo_pr_counts.head(10).items(), 1):
            repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
            print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")
        
        print("\n=== PRが少ないリポジトリ（下位10件） ===")
        for i, (repo_url, count) in enumerate(repo_pr_counts.tail(10).items(), 1):
            repo_name = repo_url.split('/')[-1] if repo_url else "Unknown"
            print(f"{i:2d}. {repo_name} ({count}件) - {repo_url}")

if __name__ == '__main__':
    df = pd.read_parquet("../dataset/all_pull_request_local.parquet")
    check_parquet(df)