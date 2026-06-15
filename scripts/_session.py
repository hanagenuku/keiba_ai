"""JRAスクレイピング用の共通requestsセッション生成"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
JRA_BASE = 'https://www.jra.go.jp'


def create_session():
    sess = requests.Session()
    sess.headers.update(HEADERS)
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    sess.mount('https://', HTTPAdapter(max_retries=retry))
    sess.get(f'{JRA_BASE}/keiba/thisweek/', timeout=15)
    return sess
