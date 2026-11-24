#!/usr/bin/env python3
"""「朝に出した版」の買い目を data/latest.json の git 履歴から埋め戻す（2026-11-24）。

博士「今まで、推奨買い目の的中率と思って見てたよ」。成績ページの的中実績は
AI上位馬を各券種1点ずつ買った参考値で、画面に出した買い目（軸1頭ベース）とも
記事の2コースとも別物だった。ここで実際に出した版を displayed_bets に残す。

    python3 scripts/backfill_morning_bets.py --since 2026-07-25   # 全履歴が要る（fetch-depth 0）

- 各開催日について「コミット文が `Weekend refresh YYYYMMDD` で、latest.json のレース日が
  その日と同じ」最初のコミットを朝の版とする。
- アプリ（gumbel_bets）→ snapshot 'morning'、記事の2コース → 'article_steady' / 'article_aggr'。
  記事の買い目は**今の作り方**で朝の予想から作る（2026-11-21 より前は記事を公開していない）。
- 既に記録のあるレースは触らない（keep_first）。最後に決済する。
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.betting.displayed_bets import article_rows_from_app_json, rows_from_app_json  # noqa: E402
from src.utils.db import init_db, save_displayed_bets, settle_displayed_bets  # noqa: E402


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def morning_commits(since):
    """[(race_date 'YYYYMMDD', sha)] を古い順に。"""
    log = _git('log', '--reverse', f'--since={since}', '--format=%H %s', '--', 'data/latest.json')
    out, seen = [], set()
    for line in log.splitlines():
        sha, _, msg = line.partition(' ')
        m = re.search(r'Weekend refresh (\d{8})', msg)
        if not m or m.group(1) in seen:
            continue
        try:
            data = json.loads(_git('show', f'{sha}:data/latest.json'))
        except Exception:
            continue
        races = data.get('races') or {}
        first = next((r for rs in (races.values() if isinstance(races, dict) else [races]) for r in rs
                      if r.get('race_id')), None)
        if not first or str(first['race_id'])[:8] != m.group(1):
            continue
        seen.add(m.group(1))
        out.append((m.group(1), sha, data))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-07-25')
    a = ap.parse_args(argv)
    init_db(ROOT)
    tot_m = tot_a = 0
    for day, sha, data in morning_commits(a.since):
        n1 = save_displayed_bets(rows_from_app_json(data, snapshot='morning'), ROOT, keep_first=True)
        n2 = save_displayed_bets(article_rows_from_app_json(data), ROOT, keep_first=True)
        tot_m += n1
        tot_a += n2
        print(f'{day} {sha[:8]}  アプリ {n1}点・記事 {n2}点')
    hp = os.path.join(ROOT, 'data', 'history.db')
    print('決済:', settle_displayed_bets(ROOT, hist_db_path=hp))
    print(f'合計: アプリ {tot_m}点・記事 {tot_a}点')


if __name__ == '__main__':
    main()
