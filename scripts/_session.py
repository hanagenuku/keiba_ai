"""JRAスクレイピング用の共通requestsセッション生成"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
JRA_BASE = 'https://www.jra.go.jp'


def create_session():
    sess = requests.Session()
    sess.headers.update(HEADERS)
    # 429(Too Many Requests)も対象に含める。Retry-Afterヘッダがあれば
    # urllib3が自動で尊重するため、追加のsleep実装は不要。
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    # suffix探索を並列化するため接続プールを広げる（同時接続数 ≒ スキャンの並列度）
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    sess.mount('https://', adapter)
    sess.get(f'{JRA_BASE}/keiba/thisweek/', timeout=15)
    return sess
