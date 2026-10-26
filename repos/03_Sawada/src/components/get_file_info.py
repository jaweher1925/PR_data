def get_file_creation_info(self, file_path):
        """ファイルの作成情報を取得（最初のコミット）"""
        try:
            commits = self.repo.get_commits(path=file_path)
            commit_list = list(commits)
            if commit_list:
                # 最後のコミット（最初のコミット）を取得
                first_commit = commit_list[-1]
                author_name = first_commit.commit.author.name or "Unknown"
                
                # コミットアカウントのみ取得（author + committer）
                all_authors = [author_name]
                
                # committerも追加（authorと異なる場合）
                if first_commit.commit.committer and first_commit.commit.committer.name:
                    committer_name = first_commit.commit.committer.name
                    if committer_name != author_name and committer_name not in all_authors:
                        all_authors.append(committer_name)
                
                return {
                    'author_name': author_name,
                    'all_authors': all_authors,  # 全作成者リスト
                    'all_creator_names': all_authors,  # CSV出力用（ファイル作成者名）
                    'creation_date': first_commit.commit.author.date.isoformat(),
                    'commit_count': len(commit_list)
                }
            return None
        except Exception as e:
            print(f"ファイル作成情報取得エラー {file_path}: {e}")
            return None