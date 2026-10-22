#!/usr/bin/env python3
"""yosounohone.com の「追い切り注目馬」記事を一括取得して `data/private/oikiri.db` に溜める。

事前登録カード: `oikiri/CRITERIA.md`（§10 取得の約束）。

    python scripts/collect_oikiri.py --start 2023-01 --end 2025-12 --budget 300

🔴 **North Star #4: budget 必須。** 2026-07-18 に budget 無しのスクレイピングを足して
   CI がタイムアウトし、その回のデータが丸ごと失われた。件数と時間の二重上限を持つ。

🔴 **D-2① の教訓: budget を守っていても「前に進んでいるか」は別問題。**
   2026-08-25 の netkeiba は artifact の復元に失敗して**毎回同じ556レースを取り直し**、
   ログの1行目が常に「取得済み 0」だった。ここは DB に取得済み URL を持ち、
   **復元できた取得済み件数を必ず印字する**。

⚠ 取得データは `data/private/`（`.gitignore` 済み）。記事本文を公開リポジトリに置かない。
⚠ 2026 も取得はできるが、**検定には使わない**（2026-10-21 の logic2026 で読んだ）。
   既定の `--end` は 2025-12。
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import yosounohone as yh  # noqa: E402

DB_REL = os.path.join('data', 'private', 'oikiri.db')
CACHE_REL = os.path.join('data', 'private', 'yosounohone_sitemaps')

DEFAULT_BUDGET = 300
DEFAULT_TIME_BUDGET_SEC = 2400
DEFAULT_INTERVAL_SEC = 1.5


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({'User-Agent': 'keiba_ai research bot (Claude-User)'})
    s.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


def _get(sess, url, timeout=30):
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or 'utf-8'
    return r.text


def init_db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT PRIMARY KEY,
        date TEXT, venue TEXT,
        published_og TEXT, published_ld TEXT,
        status TEXT, n_races INTEGER, n_horses INTEGER,
        dropped TEXT, fetched_at TEXT
    );
    CREATE TABLE IF NOT EXISTS marks (
        url TEXT, date TEXT, venue TEXT,
        race_num INTEGER, race_name TEXT,
        horse_name TEXT, stars INTEGER, training TEXT, comment TEXT,
        UNIQUE(url, race_num, horse_name)
    );
    CREATE INDEX IF NOT EXISTS idx_marks_key ON marks(date, venue, race_num);
    """)
    c.commit()
    return c


def _months(start_ym, end_ym):
    y, m = int(start_ym[:4]), int(start_ym[-2:])
    ey, em = int(end_ym[:4]), int(end_ym[-2:])
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def list_articles(sess, start_ym, end_ym, cache_dir):
    """月別 sitemap から (会場, 日付, URL) を列挙する。sitemap はキャッシュする。

    ⚠ 日付から URL を組み立てて総当たりしない。sitemap にあるものだけを対象にする。
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for y, m in _months(start_ym, end_ym):
        cache = os.path.join(cache_dir, f'{y}{m:02d}.xml')
        if os.path.exists(cache):
            with open(cache, encoding='utf-8') as f:
                xml = f.read()
        else:
            url = f'{yh.BASE_URL}/sitemap_periodical.xml?year={y}&month={m}'
            try:
                xml = _get(sess, url)
            except Exception as e:
                print(f'  ⚠ {y}-{m:02d} の sitemap 取得に失敗: {e}')
                continue
            with open(cache, 'w', encoding='utf-8') as f:
                f.write(xml)
            time.sleep(0.5)
        out += yh.list_month_articles(xml)
    # 同じ URL が複数月に出ることは無いが念のため
    seen, uniq = set(), []
    for v, d, u in sorted(out, key=lambda x: (x[1], x[0])):
        if u not in seen:
            seen.add(u)
            uniq.append((v, d, u))
    return uniq


def _done_urls(conn):
    return {r[0] for r in conn.execute('SELECT url FROM articles')}


def _save(conn, url, venue, date, art, og, ld, status):
    now = datetime.now(timezone.utc).isoformat()
    n_h = sum(len(r['horses']) for r in (art['races'] if art else []))
    conn.execute(
        'INSERT OR REPLACE INTO articles '
        '(url,date,venue,published_og,published_ld,status,n_races,n_horses,dropped,fetched_at) '
        'VALUES (?,?,?,?,?,?,?,?,?,?)',
        (url, date, venue,
         og.isoformat() if og else None, ld.isoformat() if ld else None,
         status, len(art['races']) if art else 0, n_h,
         ' / '.join(art['dropped'])[:2000] if art else None, now))
    for r in (art['races'] if art else []):
        for h in r['horses']:
            conn.execute(
                'INSERT OR REPLACE INTO marks '
                '(url,date,venue,race_num,race_name,horse_name,stars,training,comment) '
                'VALUES (?,?,?,?,?,?,?,?,?)',
                (url, date, venue, r['race_num'], r['race_name'],
                 h['name'], h['stars'], h['training'], h['comment']))
    conn.commit()


def collect(base_dir='.', start_ym='202301', end_ym='202512',
            budget=DEFAULT_BUDGET, time_budget_sec=DEFAULT_TIME_BUDGET_SEC,
            interval_sec=DEFAULT_INTERVAL_SEC):
    db_path = os.path.join(base_dir, DB_REL)
    conn = init_db(db_path)
    sess = _session()
    t0 = time.time()

    print(f'📚 sitemap から対象を列挙 ({start_ym}〜{end_ym})')
    articles = list_articles(sess, start_ym, end_ym,
                             os.path.join(base_dir, CACHE_REL))
    done = _done_urls(conn)
    todo = [a for a in articles if a[2] not in done]
    # 🔴 D-2① の教訓: 「取得済み」を必ず印字する。黙って0から始まるのを繰り返さない
    print(f'   対象 {len(articles)} 記事 / 取得済み {len(done)} / 残り {len(todo)}')
    if articles and not todo:
        print('✅ 残りゼロ。何もしない')
        return {'fetched': 0, 'ok': 0, 'failed': 0, 'remaining': 0}

    ok = failed = fetched = 0
    for venue, date, url in todo:
        if fetched >= budget:
            print(f'⏹ 件数の上限 {budget} に達したので今回はここまで')
            break
        if time.time() - t0 > time_budget_sec:
            print(f'⏹ 時間の上限 {time_budget_sec}s に達したので今回はここまで')
            break
        fetched += 1
        try:
            page = _get(sess, url)
        except Exception as e:
            print(f'  ❌ {date} {venue}: 取得失敗 {e}')
            _save(conn, url, venue, date, None, None, None, f'fetch_error: {e}'[:200])
            failed += 1
            time.sleep(interval_sec)
            continue

        og, ld = yh.parse_published_at(page)
        hdr = yh.parse_header_date_venue(page)
        if hdr != (venue, date):
            # 🔑 URL 由来と本文由来の突合（North Star #8）。一致しなければ採用しない
            print(f'  ⚠ {date} {venue}: 本文の見出しが {hdr} で URL と一致しない → 不採用')
            _save(conn, url, venue, date, None, og, ld, 'header_mismatch')
            failed += 1
            time.sleep(interval_sec)
            continue

        art = yh.parse_article(page, venue, date)
        if not art['races']:
            print(f'  ⚠ {date} {venue}: ★の付いたレースが0 → 不採用')
            _save(conn, url, venue, date, art, og, ld, 'no_marks')
            failed += 1
            time.sleep(interval_sec)
            continue

        _save(conn, url, venue, date, art, og, ld, 'ok')
        ok += 1
        n_h = sum(len(r['horses']) for r in art['races'])
        if ok % 25 == 0 or ok <= 3:
            print(f'  ✅ {date} {venue}: {len(art["races"])}R {n_h}頭 '
                  f'(ok {ok} / 失敗 {failed})')
        time.sleep(interval_sec)

    remaining = len(todo) - fetched
    tot = conn.execute('SELECT COUNT(*) FROM marks').fetchone()[0]
    arts = conn.execute("SELECT COUNT(*) FROM articles WHERE status='ok'").fetchone()[0]
    print(f'\n📊 今回 {fetched} 件取得（成功 {ok} / 失敗 {failed}）・残り {remaining}')
    print(f'   DB 累計: 記事 {arts} / 印 {tot} 頭')
    return {'fetched': fetched, 'ok': ok, 'failed': failed, 'remaining': remaining}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='.')
    p.add_argument('--start', default='2023-01')
    p.add_argument('--end', default='2025-12',
                   help='既定 2025-12。2026 は検定に使わない（logic2026 で読んだ）')
    p.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    p.add_argument('--time-budget', type=int, default=DEFAULT_TIME_BUDGET_SEC)
    p.add_argument('--interval', type=float, default=DEFAULT_INTERVAL_SEC)
    a = p.parse_args()
    r = collect(a.base_dir, a.start.replace('-', ''), a.end.replace('-', ''),
                a.budget, a.time_budget, a.interval)
    return 0 if r['ok'] or r['remaining'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
