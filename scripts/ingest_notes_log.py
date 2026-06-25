#!/usr/bin/env python3
"""不利メモログ取込：GAS の getNotesLog エンドポイントから手動入力の不利メモを取得し、
keiba.db の race_notes に保存する。

スマホアプリの「📝不利メモ」入力で送られた値を、GAS 側が Google スプレッドシートへ
追記している。本スクリプトはそのログを取り込み、過去走の不利・出遅れ・展開ロスを
total_handicap（スキーマ駆動の補正値合計）として蓄積する。次走以降、その馬の
f_unlucky_* 特徴量が立ち上がり、「前走不利だった強い馬」を評価できるようになる。

環境変数:
    GAS_URL : GAS WebアプリのデプロイURL（?action=getNotesLog を叩く）
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
                          save_race_notes, get_latest_note_time)


def main():
    gas_url = os.environ.get('GAS_URL', '').strip()
    if not gas_url:
        print('⚠ GAS_URL 未設定のため不利メモログ取込をスキップします')
        return

    init_db(ROOT)
    db_path = get_db_path(ROOT)
    since = get_latest_note_time(db_path)

    sep = '&' if '?' in gas_url else '?'
    url = gas_url + sep + urllib.parse.urlencode({'action': 'getNotesLog', 'since': since})

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f'⚠ 不利メモログ取得に失敗: {e}')
        return

    if data.get('status') != 'ok':
        print(f'⚠ GASエラー: {data.get("message", data)}')
        return

    rows = data.get('rows', [])
    saved = save_race_notes(rows, ROOT)
    print(f'📥 不利メモログ取込: {len(rows)}行受信 / {saved}行保存 '
          f'(since={since or "全件"})')


if __name__ == '__main__':
    main()
