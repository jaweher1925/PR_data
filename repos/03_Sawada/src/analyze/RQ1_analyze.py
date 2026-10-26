import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Pythonデータを可視化するためのライブラリ，バイオリンプロットに使用
import os
import sys
from datetime import datetime, timedelta

# src ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.mannwhitneyu import perform_mannwhitneyu

def analyze_rq1():
    # パス設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dir = os.path.join(script_dir, "../../results/EASE-results/csv/results_v7_released_commits_restriction.csv")
    output_dir = os.path.join(script_dir, "../../results/EASE-results/summary")

    os.makedirs(output_dir, exist_ok=True)

    # CSVファイルを読み込む
    df = pd.read_csv(input_dir)
    print(f"読み込み完了: {input_dir}")
    print(f"データ数: {len(df)}")

    # 全ファイルリストを作成（作成コミット除外前に確保）
    print("全ファイルリストを作成中...")
    all_files_df = df.sort_values('commit_date', ascending=False).groupby(['repository_name', 'file_name']).first().reset_index()
    all_files_df = all_files_df[['repository_name', 'file_name', 'file_created_by', 'file_creation_date', 'file_line_count']]
    print(f"総ファイル数: {len(all_files_df)}")

    # ファイル作成コミットを除外 作成日とコミット日が完全に一致するものを除外する
    df = df[df['commit_date'] != df['file_creation_date']].copy()

    # 日付列を日付型に変換（タイムゾーン情報を削除してtz-naiveに統一）
    df['commit_date'] = pd.to_datetime(df['commit_date']).dt.tz_localize(None)
    df['file_creation_date'] = pd.to_datetime(df['file_creation_date']).dt.tz_localize(None)
    all_files_df['file_creation_date'] = pd.to_datetime(all_files_df['file_creation_date']).dt.tz_localize(None)

    # 分析終了日
    analysis_end_date = pd.to_datetime("2026-1-31")

    # 6ヶ月以内のデータに限定した分析
    df['days_diff'] = (df['commit_date'] - df['file_creation_date']).dt.days
    # 作成日(0日)〜90日未満(89日後)まで
    df = df[(df['days_diff'] >= 0) & (df['days_diff'] < 180)].copy()
    run_analysis_process(df, all_files_df, output_dir, analysis_end_date, suffix="_6months", period_months=6)

def run_analysis_process(df, all_files_df, output_dir, analysis_end_date, suffix="", period_months=None):
    """
    分析プロセスを実行する関数
    """
    output_txt_path = os.path.join(output_dir, f"RQ1_results{suffix}.txt")
    
    # 結果格納用リスト
    results_text = []
    title_suffix = f" (作成から{period_months}ヶ月間限定)"
    period_text = f"作成日 ～ 作成日+{period_months*30}日"
    
    results_text.append(f"RQ1 分析結果: AI生成ファイルの保守性分析{title_suffix}")
    results_text.append("=" * 60)
    results_text.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"分析対象期間: {period_text}")
    results_text.append("※ 変更規模の分析には「ファイル単位の変更行数」を使用しています。")
    results_text.append("")

    # ---------------------------------------------------------
    # 1. コミット数の分析
    # ---------------------------------------------------------
    # 各ファイルのコミット数を計算
    commit_counts = df.groupby(['repository_name', 'file_name', 'file_created_by']).size().reset_index(name='commit_count')
    
    # 全ファイルリストとマージして、コミットがないファイルを0にする
    commit_counts = pd.merge(all_files_df, commit_counts, on=['repository_name', 'file_name', 'file_created_by'], how='left')
    commit_counts['commit_count'] = commit_counts['commit_count'].fillna(0)
    
    # AI作成ファイルと人間作成ファイルのコミット数を取得
    ai_commit_counts = commit_counts[commit_counts['file_created_by'] == 'AI']['commit_count']
    human_commit_counts = commit_counts[commit_counts['file_created_by'] == 'Human']['commit_count']

    def get_stats(data, label):
        return {
            'label': label,
            'count': len(data),
            'sum': data.sum(),
            'mean': data.mean(),
            'min': data.min(),
            'max': data.max(),
            'median': data.median(),
            'std': data.std()
        }

    ai_stats_count = get_stats(ai_commit_counts, "AI作成ファイル")
    human_stats_count = get_stats(human_commit_counts, "人間作成ファイル")

    results_text.append("1. コミット数の分析")
    results_text.append("-" * 40)
    
    for stats in [ai_stats_count, human_stats_count]:
        results_text.append(f"■ {stats['label']}")
        results_text.append(f"  ファイル数: {stats['count']}")
        results_text.append(f"  総コミット数: {stats['sum']}")
        results_text.append(f"  平均値: {stats['mean']:.2f}")
        results_text.append(f"  最小値: {stats['min']}")
        results_text.append(f"  最大値: {stats['max']}")
        results_text.append(f"  中央値: {stats['median']}")
        results_text.append(f"  標準偏差: {stats['std']:.2f}")
        results_text.append("")

    # 有意差検定 (コミット数)
    results_text.append(perform_mannwhitneyu(ai_commit_counts, human_commit_counts, "コミット数"))
    results_text.append("")

    # ---------------------------------------------------------
    # 2. コミット頻度の分析 (1週間ごと & 1か月ごと)
    # ---------------------------------------------------------
    # 各ファイルごとの期間別コミット数の中央値を計算
    ai_weekly_medians = []
    ai_monthly_medians = []
    human_weekly_medians = []
    human_monthly_medians = []
    
    # ファイルごとに処理
    # all_files_df をベースにループする
    # df を groupby して辞書にする (高速化)
    commit_groups = {k: v for k, v in df.groupby(['repository_name', 'file_name'])}
    
    for _, row in all_files_df.iterrows():
        repo = row['repository_name']
        file_name = row['file_name']
        creator = row['file_created_by']
        creation_date = row['file_creation_date']
        
        key = (repo, file_name)
        if key in commit_groups:
            commit_dates = commit_groups[key]['commit_date'].sort_values()
        else:
            commit_dates = pd.Series([], dtype='datetime64[ns]')
        
        target_end_date = creation_date + timedelta(days=period_months*30)
        actual_end_date = min(target_end_date, analysis_end_date)
        
        weekly_median = calculate_period_median(commit_dates, creation_date, actual_end_date, days=7)
        monthly_median = calculate_period_median(commit_dates, creation_date, actual_end_date, days=30)
        
        if creator == 'AI':
            if weekly_median is not None: ai_weekly_medians.append(weekly_median)
            if monthly_median is not None: ai_monthly_medians.append(monthly_median)
        else:
            if weekly_median is not None: human_weekly_medians.append(weekly_median)
            if monthly_median is not None: human_monthly_medians.append(monthly_median)
            
    # 統計計算
    ai_weekly_stats = get_stats(pd.Series(ai_weekly_medians), "AI作成ファイル (週間)")
    human_weekly_stats = get_stats(pd.Series(human_weekly_medians), "人間作成ファイル (週間)")
    
    ai_monthly_stats = get_stats(pd.Series(ai_monthly_medians), "AI作成ファイル (月間)")
    human_monthly_stats = get_stats(pd.Series(human_monthly_medians), "人間作成ファイル (月間)")
    
    results_text.append("2. コミット頻度の分析 (各ファイルの期間ごとのコミット数の中央値の統計)")
    results_text.append("-" * 40)
    
    results_text.append("■ 1週間ごとの頻度")
    for stats in [ai_weekly_stats, human_weekly_stats]:
        results_text.append(f"  [{stats['label']}]")
        results_text.append(f"  対象ファイル数: {stats['count']}")
        results_text.append(f"  平均値: {stats['mean']:.4f}")
        results_text.append(f"  最小値: {stats['min']}")
        results_text.append(f"  最大値: {stats['max']}")
        results_text.append(f"  中央値: {stats['median']}")
        results_text.append(f"  標準偏差: {stats['std']:.4f}")
        results_text.append("")

    # 有意差検定 (週間頻度)
    results_text.append(perform_mannwhitneyu(ai_weekly_medians, human_weekly_medians, "週間コミット頻度"))
    results_text.append("")
        
    results_text.append("■ 1か月ごとの頻度")
    for stats in [ai_monthly_stats, human_monthly_stats]:
        results_text.append(f"  [{stats['label']}]")
        results_text.append(f"  対象ファイル数: {stats['count']}")
        results_text.append(f"  平均値: {stats['mean']:.4f}")
        results_text.append(f"  最小値: {stats['min']}")
        results_text.append(f"  最大値: {stats['max']}")
        results_text.append(f"  中央値: {stats['median']}")
        results_text.append(f"  標準偏差: {stats['std']:.4f}")
        results_text.append("")

    # 有意差検定 (月間頻度)
    results_text.append(perform_mannwhitneyu(ai_monthly_medians, human_monthly_medians, "月間コミット頻度"))
    results_text.append("")

    # ---------------------------------------------------------
    # 3. 時系列でのコミット数推移 (追加)
    # ---------------------------------------------------------
    
    # 経過日数を計算
    df_ts = df.copy()
    df_ts['days_diff'] = (df_ts['commit_date'] - df_ts['file_creation_date']).dt.days
    
    # 週番号 (0始まり: 0-6日がWeek 1)
    df_ts['week_num'] = df_ts['days_diff'] // 7
    # 月番号 (0始まり: 0-29日がMonth 1)
    df_ts['month_num'] = df_ts['days_diff'] // 30
    
    # ユニークなファイルIDを作成
    df_ts['file_id'] = df_ts['repository_name'] + "::" + df_ts['file_name']
    
    # 全ファイルリストと作成者情報のマッピング (all_files_dfを使用)
    all_files_df['file_id'] = all_files_df['repository_name'] + "::" + all_files_df['file_name']
    file_creators = all_files_df.set_index('file_id')['file_created_by']
    # AI/Humanに分割
    ai_files = file_creators[file_creators == 'AI'].index
    human_files = file_creators[file_creators == 'Human'].index

    # --- 月次集計 ---
    # ファイルごと、月ごとのコミット数
    monthly_counts = df_ts.groupby(['file_id', 'month_num']).size().unstack(fill_value=0)
    
    # 全ファイルを含めるようにreindex
    monthly_counts = monthly_counts.reindex(index=file_creators.index, fill_value=0)

    max_month_limit = period_months
    max_month_data = monthly_counts.columns.max() if not monthly_counts.empty else 0
    max_month = min(max_month_limit, max_month_data)
    
    target_months = range(int(max_month) + 1)
    monthly_counts = monthly_counts.reindex(columns=target_months, fill_value=0)
    
    ai_monthly_df = monthly_counts.loc[monthly_counts.index.intersection(ai_files)]
    human_monthly_df = monthly_counts.loc[monthly_counts.index.intersection(human_files)]
    
    results_text.append(f"■ 月次推移 (Month 1 = 最初の30日間, 最大Month {max_month+1}まで表示)")
    results_text.append(f"{'Month':<6} | {'AI Mean':<10} | {'AI Median':<10} | {'Human Mean':<10} | {'Human Median':<10}")
    results_text.append("-" * 60)
    
    for m in target_months:
        ai_col = ai_monthly_df[m] if m in ai_monthly_df.columns else pd.Series([], dtype=float)
        human_col = human_monthly_df[m] if m in human_monthly_df.columns else pd.Series([], dtype=float)
        
        ai_mean = ai_col.mean() if not ai_col.empty else 0
        ai_median = ai_col.median() if not ai_col.empty else 0
        human_mean = human_col.mean() if not human_col.empty else 0
        human_median = human_col.median() if not human_col.empty else 0
        
        results_text.append(f"{m+1:<6} | {ai_mean:<10.4f} | {ai_median:<10.1f} | {human_mean:<10.4f} | {human_median:<10.1f}")
    results_text.append("")

    # --- リポジトリごとの月次コミット数推移 ---
    if not monthly_counts.empty:
        # file_id から repository_name と file_created_by を取得するためのマッピング
        repo_map = all_files_df.set_index('file_id')[['repository_name', 'file_created_by']]
        
        # joinしてリポジトリ情報を付与
        repo_monthly = monthly_counts.join(repo_map)
        
        # 数値カラム（月番号）のみを抽出
        numeric_cols = [c for c in repo_monthly.columns if isinstance(c, int)]
        
        # リポジトリと作成者タイプごとに合計
        repo_monthly_sum = repo_monthly.groupby(['repository_name', 'file_created_by'])[numeric_cols].sum()
        
        # ロング形式に変換
        repo_long = repo_monthly_sum.reset_index().melt(
            id_vars=['repository_name', 'file_created_by'], 
            value_vars=numeric_cols,
            var_name='month_num', 
            value_name='commit_count'
        )
        
        # バイオリンプロットとして出力
        create_monthly_trend_violinplot(
            repo_long,
            'commit_count',
            "Commits per Repository",
            "Commits",
            os.path.join(output_dir, f"RQ1_commits_per_repo_violin{suffix}.pdf"),
            max_month,
            ylim=10,
        )


    # ---------------------------------------------------------
    # 4. 変更規模の分析 
    # ---------------------------------------------------------
    # 変更行数と割合の計算
    df_size = df.copy()
    df_size = df_size[df_size['file_line_count'] > 0]
    
    df_size = df_size[df_size['file_specific_changed_lines'] != -1]
    target_col = 'file_specific_changed_lines'

    df_size['change_ratio'] = (df_size[target_col] / df_size['file_line_count']) * 100
    # 100%を超える場合は100%にクリップする
    df_size['change_ratio'] = df_size['change_ratio'].clip(upper=100.0)
    
    # AI/Humanに分割
    ai_size_df = df_size[df_size['file_created_by'] == 'AI']
    human_size_df = df_size[df_size['file_created_by'] == 'Human']
    
    results_text.append("4. 変更規模の分析")
    results_text.append("-" * 40)

    # --- ファイル行数の分析 (追加) ---
    # all_files_df を使用して、変更がないファイルも含めた全ファイルのサイズを分析
    ai_file_sizes = all_files_df[all_files_df['file_created_by'] == 'AI']['file_line_count']
    human_file_sizes = all_files_df[all_files_df['file_created_by'] == 'Human']['file_line_count']

    results_text.append("■ ファイル行数 (ファイルサイズ)")
    results_text.append(f"  [AI作成ファイル]")
    results_text.append(f"  平均値: {ai_file_sizes.mean():.2f}")
    results_text.append(f"  中央値: {ai_file_sizes.median():.2f}")
    results_text.append(f"  [人間作成ファイル]")
    results_text.append(f"  平均値: {human_file_sizes.mean():.2f}")
    results_text.append(f"  中央値: {human_file_sizes.median():.2f}")
    results_text.append("")
    
    # 有意差検定 (ファイル行数)
    results_text.append(perform_mannwhitneyu(ai_file_sizes, human_file_sizes, "ファイル行数"))
    results_text.append("")
    # -----------------------------
    
    # 変更行数
    ai_lines = ai_size_df[target_col]
    human_lines = human_size_df[target_col]
    
    results_text.append("■ 1コミットあたりの変更行数")
    results_text.append(f"  [AI作成ファイル]")
    results_text.append(f"  平均値: {ai_lines.mean():.2f}")
    results_text.append(f"  中央値: {ai_lines.median():.2f}")
    results_text.append(f"  [人間作成ファイル]")
    results_text.append(f"  平均値: {human_lines.mean():.2f}")
    results_text.append(f"  中央値: {human_lines.median():.2f}")
    results_text.append("")
    
    # 有意差検定 (変更行数)
    results_text.append(perform_mannwhitneyu(ai_lines, human_lines, "変更行数"))
    results_text.append("")
    
    # 変更割合
    ai_ratio = ai_size_df['change_ratio']
    human_ratio = human_size_df['change_ratio']
    
    results_text.append("■ 1コミットあたりの変更割合 (%)")
    results_text.append(f"  [AI作成ファイル]")
    results_text.append(f"  平均値: {ai_ratio.mean():.2f}%")
    results_text.append(f"  中央値: {ai_ratio.median():.2f}%")
    results_text.append(f"  [人間作成ファイル]")
    results_text.append(f"  平均値: {human_ratio.mean():.2f}%")
    results_text.append(f"  中央値: {human_ratio.median():.2f}%")
    results_text.append("")

    # 有意差検定 (変更割合)
    results_text.append(perform_mannwhitneyu(ai_ratio, human_ratio, "変更割合"))
    results_text.append("")

    # --- 変更規模の月次推移 ---
    # 月番号の計算
    df_size['days_diff'] = (df_size['commit_date'] - df_size['file_creation_date']).dt.days
    df_size['month_num'] = df_size['days_diff'] // 30
    
    # 表示期間の制限
    max_month_limit = period_months
    
    df_size_filtered = df_size[df_size['month_num'] <= max_month_limit]
    
    # 1. 変更行数の推移
    create_monthly_trend_violinplot(
        df_size_filtered,
        target_col,
        "Lines Changed per Commit",
        "Lines Changed",
        os.path.join(output_dir, f"RQ1_lines_changed_per_commit{suffix}.pdf"),
        max_month_limit,
        ylim=200,
    )
    
    # 2. 変更割合の推移
    # ファイル単位の集計をやめて、コミット単位(ファイルごとのコミット)のデータをそのまま使う
    df_ratio_filtered = df_size[df_size['month_num'] <= max_month_limit].copy()

    create_monthly_trend_violinplot(
        df_ratio_filtered,
        'change_ratio',
        "Change Ratio per Commit",
        "Change Ratio(%)",
        os.path.join(output_dir, f"RQ1_change_ratio_per_commit{suffix}.pdf"),
        max_month_limit,
        ylim=100,
    )

    # --- 変更規模の月次推移 (テキスト出力) ---
    results_text.append("■ 月次推移 (変更行数の中央値)")
    results_text.append(f"{'Month':<6} | {'AI Median':<10} | {'Human Median':<10}")
    results_text.append("-" * 40)
    
    # 月ごとに集計
    for m in range(int(max_month_limit) + 1):
        month_data = df_size_filtered[df_size_filtered['month_num'] == m]
        if month_data.empty:
            continue
            
        ai_median = month_data[month_data['file_created_by'] == 'AI'][target_col].median()
        human_median = month_data[month_data['file_created_by'] == 'Human'][target_col].median()
        
        # NaNの場合は0または-にする
        ai_str = f"{ai_median:.1f}" if not pd.isna(ai_median) else "-"
        human_str = f"{human_median:.1f}" if not pd.isna(human_median) else "-"
        
        results_text.append(f"{m+1:<6} | {ai_str:<10} | {human_str:<10}")
    results_text.append("")

    results_text.append("■ 月次推移 (変更割合の中央値 %)")
    results_text.append(f"{'Month':<6} | {'AI Median':<10} | {'Human Median':<10}")
    results_text.append("-" * 40)
    
    for m in range(int(max_month_limit) + 1):
        month_data = df_ratio_filtered[df_ratio_filtered['month_num'] == m]
        if month_data.empty:
            continue
            
        ai_median = month_data[month_data['file_created_by'] == 'AI']['change_ratio'].median()
        human_median = month_data[month_data['file_created_by'] == 'Human']['change_ratio'].median()
        
        ai_str = f"{ai_median:.2f}" if not pd.isna(ai_median) else "-"
        human_str = f"{human_median:.2f}" if not pd.isna(human_median) else "-"
        
        results_text.append(f"{m+1:<6} | {ai_str:<10} | {human_str:<10}")
    results_text.append("")

    # 結果保存
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_text))

def calculate_period_median(commit_dates, start_date, end_date, days):
    """
    指定期間ごとのコミット数を集計し、その中央値を返す
    """
    if pd.isna(start_date):
        # 作成日が不明な場合は最初のコミット日を使用
        start_date = commit_dates.min()
        
    current_date = start_date
    counts = []
    
    # 期間が終了日を超えるまでループ
    while current_date + timedelta(days=days) <= end_date:
        next_date = current_date + timedelta(days=days)
        # 期間内のコミット数をカウント
        count = ((commit_dates >= current_date) & (commit_dates < next_date)).sum()
        counts.append(count)
        current_date = next_date
        
    if not counts:
        return None
        
    return np.median(counts)

def create_monthly_trend_violinplot(df, value_col, title, ylabel, output_path, max_month, ylim=None, labels=None):
    """
    月ごとの変更規模の推移をバイオリンプロットで可視化する
    """
    if labels is None:
        labels = {'AI': 'Agent-created files', 'Human': 'Human-created files'}

    # データが空の場合はスキップ
    if df.empty:
        print(f"Warning: No data available for {title}")
        return

    # 月番号が負のものを除外
    df = df[df['month_num'] >= 0].copy()
    
    # 月を1始まりにする
    df['Month'] = df['month_num'] + 1
    
    # 凡例用の名前変更
    df['Type'] = df['file_created_by'].map({
        'AI': labels['AI'],
        'Human': labels['Human']
    })
    
    # プロット用データ作成
    df_plot = df[['Month', value_col, 'Type']].rename(columns={value_col: 'Value'})
    
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    sns.set_style("white")
    
    # 月の順序を指定
    month_order = range(1, int(max_month) + 2)
    
    # バイオリンプロット
    sns.violinplot(
        data=df_plot,
        x='Month',
        y='Value',
        hue='Type',
        split=True,
        inner=None,
        palette={labels['AI']: "#FF9999", labels['Human']: "#99CCFF"},
        cut=0,
        order=month_order,
        ax=ax
    )
    
    # 箱ひげ図と平均値を重ねる
    box_width = 0.1
    offset = 0.1
    
    for i, m in enumerate(month_order):
        month_data = df_plot[df_plot['Month'] == m]
        if month_data.empty:
            continue
            
        data_ai = month_data[month_data['Type'] == labels['AI']]['Value']
        data_human = month_data[month_data['Type'] == labels['Human']]['Value']
        
        if not data_ai.empty:
            ax.boxplot(
                [data_ai],
                positions=[i - offset],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black'),
                showfliers=False,
                manage_ticks=False
            )
            # ax.scatter(x=[i - offset], y=[data_ai.mean()], color='white', marker='D', s=30, zorder=10, edgecolor='black')

        if not data_human.empty:
            ax.boxplot(
                [data_human],
                positions=[i + offset],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor='white', alpha=0.5, edgecolor='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black'),
                showfliers=False,
                manage_ticks=False
            )
            # ax.scatter(x=[i + offset], y=[data_human.mean()], color='white', marker='D', s=30, zorder=10, edgecolor='black')

    if ylim is not None:
        ax.set_ylim(0, ylim)

    # plt.title(title, fontsize=20)
    plt.xlabel("Months", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.legend(title="", fontsize=28, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    analyze_rq1()
