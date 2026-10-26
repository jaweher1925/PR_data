# AIボットアカウント定義（作成者名で判定）
AI_BOT_ACCOUNTS = {
    'copilot': ['copilot'],  # GitHub Copilot
    'cursor': ['cursor'],  # Cursor
    'devin': ['devin-ai-integration'],  # Devin
    'claude': ['claude']  # Claude
}

def ai_check(all_authors):
    """
    AIコミット判定（全アカウントをチェック）
        
    Args:
        all_authors: コミットに関与した全アカウント名（文字列またはリスト）
    Returns:
        tuple: (bool, str) - AIかどうか, AIの種類またはhuman
    """
    # 文字列の場合はリストに変換
    if isinstance(all_authors, str):
        all_authors = [all_authors]

    # 全アカウントをチェック
    for author_name in all_authors:
        author_lower = author_name.lower()
        for ai_type, bot_names in AI_BOT_ACCOUNTS.items():
            if any(bot_name.lower() in author_lower for bot_name in bot_names):
                return True, ai_type

    return False, "human"