#!/usr/bin/env python3
"""直前オッズログ取込：GAS の getOddsLog エンドポイントから直前確定オッズを取得し、
keiba.db の odds_snapshots に保存する。

スマホアプリの「直前オッズ取得」ボタンが押されるたびに、GAS 側がその時点の
単勝・複勝オッズを Google スプレッドシートへ追記している。本スクリプトはその
ログを取り込み、朝予想(race_predictions) vs 直前オッズ(odds_snapshots) vs
結果(history.db) の三者を後から突き合わせられるようにする。

環境変数:
    GAS_URL : GAS WebアプリのデプロイURL（?action=getOddsLog を叩く）
              未設定ならスキップ（CI/ローカルで安全に no-op）。
"""
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.db import (init_db, get_db_path,
                          save_odds_snapshots, get_latest_odds_snapshot_time)


def main():
    gas_url = os.environ.get('GAS_URL', '').strip()
    if not gas_url:
        print('⚠ GAS_URL 未設定のため直前オッズログ取込をスキップします')
        return

    init_db(ROOT)
    db_path = get_db_path(ROOT)
    since = get_latest_odds_snapshot_time(db_path)

    sep = '&' if '?' in gas_url else '?'
    url = gas_url + sep + urllib.parse.urlencode({'action': 'getOddsLog', 'since': since})

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'⚠ 直前オッズログ取得に失敗: {e}')
        return

    if data.get('status') != 'ok':
        print(f'⚠ GASエラー: {data.get("message", data)}')
        return

    rows = data.get('rows', [])
    saved = save_odds_snapshots(rows, ROOT)
    print(f'📥 直前オッズログ取込: {len(rows)}行受信 / {saved}行新規保存 '
          f'(since={since or "全件"})')


if __name__ == '__main__':
    main()
