import os
import pandas as pd

def update_results():
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "../../results/EASE-results/csv/results_v7_released_commits_restriction.csv")
    output_file = os.path.join(script_dir, "../../results/EASE-results/csv/results_v7_released_commits_restriction_updated.csv")

    df = pd.read_csv(input_file)

    # 値を変換
    df['commit_created_by'] = df['commit_created_by'].replace({
        'copilot': 'AI',
        'cursor': 'AI',
        'human': 'Human'
    })

    # 結果を保存
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    update_results()
