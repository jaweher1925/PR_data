def get_file_line_count(self, file_path, commit_sha):
    """ファイルの行数を取得"""
    try:
        # 特定のコミットでのファイル内容を取得
        content = self.repo.get_contents(file_path, ref=commit_sha)
        if content.encoding == 'base64':
            import base64
            decoded_content = base64.b64decode(content.content).decode('utf-8', errors='ignore')
            return len(decoded_content.splitlines())
        return 0
    except Exception as e:
        print(f"ファイル行数取得エラー {file_path}: {e}")
        return 0