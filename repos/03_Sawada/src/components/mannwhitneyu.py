from scipy.stats import mannwhitneyu

def perform_mannwhitneyu(data1, data2, label):
    """Mann-Whitney U検定を実行して結果文字列を返す"""
    try:
        # データが空の場合は検定できない
        if len(data1) == 0 or len(data2) == 0:
            return f"■ {label}の検定: データ不足のため実行不可\n"
            
        statistic, p_value = mannwhitneyu(data1, data2, True, alternative='two-sided')
        result = f"■ {label}のMann-Whitney U検定結果\n"
        result += f"  検定統計量 U: {statistic}\n"
        result += f"  p値: {p_value}\n"
        if p_value < 0.01:
            result += "  判定: ** 1%水準で有意差あり\n"
        elif p_value < 0.05:
            result += "  判定: * 5%水準で有意差あり\n"
        else:
            result += "  判定: 有意差なし\n"
        return result
    except Exception as e:
        return f"■ {label}の検定エラー: {e}\n"