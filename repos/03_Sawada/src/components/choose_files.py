def get_files_by_author_type(self, df, target_ai_count=10, target_human_count=10):
        """AI/Humanファイル選択（ランダムサンプリング版）"""
        # AI作成ファイルと人間作成ファイルを分離
        ai_df = df[df['author_type'] == 'AI'].copy()
        human_df = df[df['author_type'] == 'Human'].copy()
        
        # ランダムサンプリング
        ai_files = []
        human_files = []
        
        # AI作成ファイルをランダムに選択
        if len(ai_df) > 0:
            sample_size_ai = min(target_ai_count, len(ai_df))
            ai_sampled = ai_df.sample(n=sample_size_ai, random_state=None)  # random_state=Noneで毎回異なる
            
            for _, row in ai_sampled.iterrows():
                ai_files.append({
                    'commit_hash': row['commit_hash'],
                    'added_file': row['added_file'],
                    'author_type': row['author_type'],
                    'ai_type': row['ai_type']
                })
        
        # 人間作成ファイルをランダムに選択
        if len(human_df) > 0:
            sample_size_human = min(target_human_count, len(human_df))
            human_sampled = human_df.sample(n=sample_size_human, random_state=None)
            
            for _, row in human_sampled.iterrows():
                human_files.append({
                    'commit_hash': row['commit_hash'],
                    'added_file': row['added_file'],
                    'author_type': row['author_type'],
                    'ai_type': row['ai_type']
                })
        
        # 同数に調整（数が小さい方合わせる）
        min_count = min(len(ai_files), len(human_files))
        ai_files = ai_files[:min_count]
        human_files = human_files[:min_count]
        
        print(f"ファイル数調整: AI={len(ai_files)} Human={len(human_files)} (同数に調整)")
        
        return ai_files + human_files