import socket
import time
from datetime import datetime # 日付取得や時間の計算のためのライブラリ

def retry_with_network_check(func):
    """
    ネットワークエラー時に自動的に再接続を試みるデコレータ
    
    Args:
        func: ラップする関数
    
    Returns:
        ラップされた関数
    """
    def wrapper(*args, **kwargs):
        max_wait = 60  # 最大待機時間（秒）
        wait_time = 10  # 初期待機時間（秒）
        
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # DNS解決エラーやConnectionErrorを検出
                if 'nameresolutionerror' in error_str or 'failed to resolve' in error_str or \
                    'connectionerror' in error_str or 'connection error' in error_str or \
                    'getaddrinfo failed' in error_str:
                    
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ネットワークエラーを検出しました: {e}")
                    print(f"ネットワーク接続を確認中...")
                    
                    # ネットワークが復旧するまで待機
                    while not check_network_connectivity():
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ネットワークに接続できません。{wait_time}秒後に再試行します...")
                        time.sleep(wait_time)
                        # 指数バックオフ（最大60秒まで）
                        wait_time = min(wait_time * 2, max_wait)
                    
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ネットワーク接続が復旧しました。処理を再開します...")
                    wait_time = 10  # 待機時間をリセット
                    continue  # 関数を再実行
                else:
                    # ネットワーク以外のエラーはそのまま送出
                    raise
    
    return wrapper

# ネットワーク再接続機能
def check_network_connectivity(host="api.github.com", port=443, timeout=5):
    """
    ネットワーク接続を確認する
    
    Args:
        host: 接続先ホスト
        port: 接続ポート
        timeout: タイムアウト（秒）
    
    Returns:
        bool: 接続可能ならTrue
    """
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False