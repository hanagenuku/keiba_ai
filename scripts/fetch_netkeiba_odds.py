#!/usr/bin/env python3
"""過去レースの**全馬の単勝オッズ**を netkeiba から取得して data/private に貯める。

なぜ必要か
----------
D-2（残差学習の base_margin を「人気順位のZipf近似」から「実オッズ」へ）は
2026-08-06 に **+0.008〜0.020 AUC** と実測された、残っている中で桁が一つ大きい
唯一の案。ブロッカーは history.db の `win_odds` が全年0%充足で、JRA公式の
結果ページからは単勝オッズ列が消滅している（2026-08-03③）ため再取得もできず、
`race_predictions` の朝オッズが N=2,000 に達するまで約5ヶ月待ちだったこと。

netkeiba の結果ページには全馬の単勝オッズ・人気・馬体重が**無料で**載っている
（2026-08-24 のプローブで2023年のレースまで遡って確認）。
⚠ 有料なのは ﾀｲﾑ指数・ｽﾀｰﾄ指数・追走指数・上がり指数・調教ﾀｲﾑ・厩舎ｺﾒﾝﾄ（`**` or 空）。
**それらには触らない。**

保存先について
--------------
⚠ `hanagenuku/keiba_ai` は**公開リポジトリ**で weekend.yml / sunday-results.yml が
毎週 `git add data/` で data/ 配下を丸ごと自動pushする。netkeiba の規約は入手データの
用途を私的利用に限定しているため、**`data/private/`（gitignore済み）にのみ**書く。
詳細は `data/private/README.md`。

⚠ North Star #4: 件数と時間の二重上限を必ず持つ。中断しても再開できる。
"""
import argparse
import os
import re
import sqlite3
import sys
import time

import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

UA = ('keiba_ai-research/1.0 (personal use; '
      'contact via github.com/hanagenuku/keiba_ai)')
SLEEP = 2.0
OUT_DB = os.path.join(ROOT, 'data', 'private', 'netkeiba.db')

# 列は**見出しの文字で**引く。位置決め打ちにしない。
# 🔑 2026-08-03③でJRA公式の結果ページの列順が変わり、popularity/body_weight/trainer が
#    1ヶ月間サイレントに壊れた。同じ事故を繰り返さない。
_COL = {
    'horse_num': ('馬番',),
    'horse_name': ('馬名',),
    'win_odds': ('単勝',),
    'popularity': ('人気',),
    'body_weight': ('馬体重',),
    'place': ('着順',),
}


def init_out_db(path=OUT_DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('''CREATE TABLE IF NOT EXISTS netkeiba_odds (
        race_id TEXT, horse_num INTEGER, horse_name TEXT,
        win_odds REAL, popularity INTEGER, body_weight INTEGER, place INTEGER,
        nk_race_id TEXT, fetched_at TEXT,
        UNIQUE(race_id, horse_num))''')
    # 取得済み/取得不能を記録して再開できるようにする
    conn.execute('''CREATE TABLE IF NOT EXISTS fetch_log (
        race_id TEXT PRIMARY KEY, nk_race_id TEXT, status TEXT,
        n_horses INTEGER, fetched_at TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS day_map (
        date TEXT PRIMARY KEY, n_races INTEGER, fetched_at TEXT)''')
    conn.commit()
    return conn


def parse_result_table(html):
    """結果テーブルから全馬の単勝オッズ等を取り出す。見出しの文字で列を引く。"""
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if len(rows) < 3:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(['th', 'td'])]
        if '馬名' not in head or '単勝' not in head:
            continue
        idx = {}
        for key, names in _COL.items():
            for nm in names:
                if nm in head:
                    idx[key] = head.index(nm)
                    break
        if 'horse_num' not in idx or 'win_odds' not in idx:
            continue
        out = []
        for r in rows[1:]:
            cells = [c.get_text(strip=True) for c in r.find_all(['th', 'td'])]
            if len(cells) < len(head):
                continue
            rec = {}
            for key, i in idx.items():
                rec[key] = cells[i]
            out.append(_normalize(rec))
        return [x for x in out if x and x.get('horse_num')]
    return []


def _num(s, cast=float):
    try:
        return cast(str(s).replace(',', '').strip())
    except (TypeError, ValueError):
        return None


def _normalize(rec):
    bw = rec.get('body_weight') or ''
    m = re.match(r'^(\d+)', bw)
    return {
        'horse_num': _num(rec.get('horse_num'), int),
        'horse_name': (rec.get('horse_name') or '').strip(),
        # `**`（有料）や空は None になる。触らない
        'win_odds': _num(rec.get('win_odds')),
        'popularity': _num(rec.get('popularity'), int),
        'body_weight': int(m.group(1)) if m else None,
        'place': _num(rec.get('place'), int),
    }


class Fetcher:
    def __init__(self, max_requests, max_seconds):
        self.sess = requests.Session()
        self.sess.headers.update({'User-Agent': UA})
        self.max_requests = max_requests
        self.deadline = time.time() + max_seconds
        self.count = 0

    def budget_left(self):
        return self.count < self.max_requests and time.time() < self.deadline

    def get(self, url):
        if not self.budget_left():
            return None
        self.count += 1
        try:
            r = self.sess.get(url, timeout=25)
        except Exception as e:
            print(f'  ⚠ 通信失敗 {type(e).__name__}: {url}')
            return None
        time.sleep(SLEEP)
        if r.status_code != 200:
            print(f'  ⚠ HTTP {r.status_code}: {url}')
            return None
        r.encoding = r.apparent_encoding or 'euc-jp'
        return r.text


def build_day_map(f, conn, date_str):
    """開催日の netkeiba race_id 一覧を取り、(場コード, R) → nk_race_id を返す。"""
    html = f.get(f'https://db.netkeiba.com/race/list/{date_str}/')
    if html is None:
        return {}
    ids = sorted(set(re.findall(r'/race/(\d{12})', html)))
    m = {(i[4:6], i[10:12]): i for i in ids}
    conn.execute('INSERT OR REPLACE INTO day_map VALUES (?,?,datetime("now"))',
                 (date_str, len(ids)))
    conn.commit()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-07-01', help='この日以降のレースを対象')
    ap.add_argument('--end', default='2026-12-31')
    ap.add_argument('--max-requests', type=int, default=600)
    ap.add_argument('--max-minutes', type=int, default=25)
    a = ap.parse_args()

    hist = sqlite3.connect(os.path.join(ROOT, 'data', 'history.db'))
    races = hist.execute(
        'SELECT race_id, date FROM race_history WHERE date BETWEEN ? AND ? '
        'ORDER BY date, race_id', (a.start, a.end)).fetchall()
    hist.close()

    conn = init_out_db()
    done = {r[0] for r in conn.execute('SELECT race_id FROM fetch_log')}
    todo = [(rid, d) for rid, d in races if rid not in done]
    print(f'対象 {len(races):,}レース  取得済み {len(done):,}  残り {len(todo):,}')
    print(f'budget: {a.max_requests}リクエスト / {a.max_minutes}分  間隔{SLEEP}秒')

    f = Fetcher(a.max_requests, a.max_minutes * 60)
    day_maps, n_ok, n_ng = {}, 0, 0
    for rid, d in todo:
        if not f.budget_left():
            print('  → budget 到達。ここで中断（次回このレースから再開）')
            break
        date_str = rid.split('_')[0]
        if date_str not in day_maps:
            day_maps[date_str] = build_day_map(f, conn, date_str)
            print(f'  📅 {date_str}: netkeiba race_id {len(day_maps[date_str])}件')
        parts = rid.split('_')
        nk = day_maps[date_str].get((parts[1], parts[2].zfill(2)))
        if not nk:
            conn.execute('INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,datetime("now"))',
                         (rid, None, 'no_match', 0))
            conn.commit(); n_ng += 1
            continue
        html = f.get(f'https://db.netkeiba.com/race/{nk}/')
        if html is None:
            continue                      # budget切れ or 通信失敗。再開時に拾う
        rows = parse_result_table(html)
        got = sum(1 for r in rows if r.get('win_odds'))
        for r in rows:
            conn.execute(
                'INSERT OR REPLACE INTO netkeiba_odds VALUES (?,?,?,?,?,?,?,?,datetime("now"))',
                (rid, r['horse_num'], r['horse_name'], r['win_odds'],
                 r['popularity'], r['body_weight'], r['place'], nk))
        conn.execute('INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,datetime("now"))',
                     (rid, nk, 'ok' if got else 'no_odds', got))
        conn.commit()
        n_ok += 1 if got else 0
        n_ng += 0 if got else 1
        if (n_ok + n_ng) % 50 == 0:
            print(f'  … {n_ok + n_ng}レース処理  オッズ取得 {n_ok}  '
                  f'リクエスト {f.count}/{a.max_requests}')

    tot = conn.execute('SELECT COUNT(*) FROM netkeiba_odds WHERE win_odds > 0').fetchone()[0]
    nr = conn.execute('SELECT COUNT(DISTINCT race_id) FROM netkeiba_odds '
                      'WHERE win_odds > 0').fetchone()[0]
    print(f'\n今回: {n_ok}レース成功 / {n_ng}件スキップ / {f.count}リクエスト')
    print(f'累計: {nr:,}レース {tot:,}頭ぶんの実オッズ')
    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
