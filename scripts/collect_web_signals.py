#!/usr/bin/env python3
"""kayochinkeiba の能力指数を過去分ふくめて集め、`data/private/web_signals.db` に入れる。

なぜ過去分が取れるか
--------------------
2026-09-24 に sitemap を確認した。月別アーカイブが **2016-12〜2026-09（119ヶ月）**
揃っており、`robots.txt` は `/wp-admin/` のみ Disallow・Crawl-delay なし。
history.db と重なる 2023-01〜2026-09 で **JRA記事 811本 / 開催日 399日**。
2023-09 の記事が 2026-09 と同一構造であることも実HTMLで確認済み。

→ 「溜まるまで待つ」必要がない。過去分で先に測れる。
→ 学習データが作れるので、後付け補正ではなく特徴量として入れられる。

⚠ North Star #4: 新しいリクエスト元には必ず件数の上限を置く
   2026-07-18 に budget 無しのスクレイピングを足してCIがタイムアウトし、
   その回のデータを丸ごと失った前例がある。ここでは
   ①記事数 ②実時間 の二重上限を置き、超えたらそこで正常終了する。

⚠ 2026-08-25 D-2① の教訓: **budget を守っていても「前に進んでいるか」は別問題**。
   netkeiba 取得は artifact の復元に失敗して毎回同じ556レースを取り直していた。
   ここでは `web_fetch_log` に取得済みURLを残し、実行のたびに
   **取得済み / 残り を必ず印字する**。黙って0から始まるのを繰り返さないため。

⚠ 取得データは `data/private/`（.gitignore 済み）に置く。
   公開リポジトリに他所のデータを置かない（netkeiba と同じ扱い）。

⚠ 失敗を無音にしない。パース却下・HTTPエラーも `web_fetch_log` に理由付きで残す。
   「取得できた件数」だけ見て成功と誤認しないため。

使い方
------
    python scripts/collect_web_signals.py --start 2023-01 --end 2026-09 --budget 60
    python scripts/collect_web_signals.py --dry-run          # 対象一覧だけ出す
"""
import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.kayochin import SOURCE_NAME, parse_article  # noqa: E402
from src.utils.config import PLACE_ENG  # noqa: E402

BASE_URL = 'https://kayochinkeiba.com'
SITEMAP_INDEX = f'{BASE_URL}/sitemap.xml'
DB_REL = os.path.join('data', 'private', 'web_signals.db')
CACHE_REL = os.path.join('data', 'private', 'kayochin_sitemaps')

# 1回の実行で取る記事数の上限（North Star #4）
DEFAULT_BUDGET = 60
# 1回の実行の実時間上限（秒）。ワークフローのタイムアウトを守る最後の砦
DEFAULT_TIME_BUDGET_SEC = 1200
# リクエスト間隔（秒）。相手のサーバに負荷をかけない
DEFAULT_INTERVAL_SEC = 2.0

_ARTICLE_RE = re.compile(
    r'https://kayochinkeiba\.com/(\d{6})(' + '|'.join(sorted(PLACE_ENG)) + r')/')
_MONTH_SITEMAP_RE = re.compile(
    r'<loc>(https://kayochinkeiba\.com/sitemap-posttype-post\.(\d{6})\.xml)</loc>')


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({'User-Agent': (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/124 Safari/537.36')})
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def _get(sess, url, timeout=30):
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or 'utf-8'
    return r.text


def _ymd_from_yymmdd(s):
    """'230930' → '2023-09-30'。このサイトは2016年以降しか無い。"""
    return f'20{s[0:2]}-{s[2:4]}-{s[4:6]}'


def list_articles(sess, start_ym, end_ym, cache_dir):
    """sitemap から (date, venue, url) を列挙する。月別sitemapはキャッシュする。

    ⚠ 日付を推測してURLを組み立てない。sitemap にあるものだけを対象にする
      （開催日カレンダーを別途持つ必要が無く、存在しないURLを叩かない）。
    """
    os.makedirs(cache_dir, exist_ok=True)
    idx = _get(sess, SITEMAP_INDEX)
    months = sorted({ym for _u, ym in _MONTH_SITEMAP_RE.findall(idx)
                     if start_ym <= ym <= end_ym})

    found, seen = [], set()
    for ym in months:
        path = os.path.join(cache_dir, f'{ym}.xml')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, encoding='utf-8', errors='replace') as f:
                txt = f.read()
        else:
            txt = _get(sess, f'{BASE_URL}/sitemap-posttype-post.{ym}.xml')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(txt)
            time.sleep(0.5)
        for yymmdd, venue in _ARTICLE_RE.findall(txt):
            date = _ymd_from_yymmdd(yymmdd)
            if not (f'{start_ym[:4]}-{start_ym[4:]}' <= date[:7]
                    <= f'{end_ym[:4]}-{end_ym[4:]}'):
                continue
            url = f'{BASE_URL}/{yymmdd}{venue}/'
            if url in seen:
                continue
            seen.add(url)
            found.append((date, venue, url))
    found.sort()
    return found


def init_db(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_signals (
            source       TEXT NOT NULL,
            race_id      TEXT NOT NULL,
            horse_num    INTEGER NOT NULL,
            date         TEXT NOT NULL,
            venue        TEXT NOT NULL,
            race_num     INTEGER NOT NULL,
            index_value  INTEGER NOT NULL,
            index_rank   INTEGER NOT NULL,
            n_horses     INTEGER NOT NULL,
            published_at TEXT,
            collected_at TEXT NOT NULL,
            UNIQUE(source, race_id, horse_num)
        )""")
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ws_race ON web_signals(race_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ws_date ON web_signals(date)')
    # 取得の履歴。再開可能にするためと、失敗を無音にしないため
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_fetch_log (
            url          TEXT PRIMARY KEY,
            source       TEXT NOT NULL,
            date         TEXT NOT NULL,
            venue        TEXT NOT NULL,
            status       TEXT NOT NULL,
            n_races      INTEGER NOT NULL DEFAULT 0,
            n_rows       INTEGER NOT NULL DEFAULT 0,
            rejected     TEXT,
            published_at TEXT,
            collected_at TEXT NOT NULL
        )""")
    conn.commit()
    return conn


def _done_urls(db_path):
    """すでに成功した URL。失敗は再試行の対象に残す。

    ⚠ DBが無ければ**作らずに**空集合を返す。`--dry-run` が副作用で
       ファイルを作らないようにするため（テストで固定）。
    """
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT url FROM web_fetch_log WHERE status = 'ok'").fetchall()}
    except sqlite3.OperationalError:
        return set()          # まだテーブルが無い
    finally:
        conn.close()


def _save(conn, url, date, venue, rows, meta, status):
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    conn.executemany("""
        INSERT OR REPLACE INTO web_signals
          (source, race_id, horse_num, date, venue, race_num,
           index_value, index_rank, n_horses, published_at, collected_at)
        VALUES (:source, :race_id, :horse_num, :date, :venue, :race_num,
                :index_value, :index_rank, :n_horses, :published_at, :collected_at)
    """, [dict(r, collected_at=now) for r in rows])
    rej = meta.get('rejected') or []
    conn.execute("""
        INSERT OR REPLACE INTO web_fetch_log
          (url, source, date, venue, status, n_races, n_rows,
           rejected, published_at, collected_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (url, SOURCE_NAME, date, venue, status, meta.get('n_races', 0), len(rows),
          '; '.join(f'{rn}R:{why}' for rn, why in rej) or None,
          meta.get('published_at'), now))
    conn.commit()


# 台帳がこの status を付けた情報源は収集しない。
# 🔑 「不採用」「取得不可」と決めたものを毎週取りに行かない
#    （相手のサーバに無駄な負荷をかけないため）。
# ⚠ 台帳に**載っていない**情報源は止めない。未登録は「決定していない」だけで、
#   「集めないと決めた」ではない。測る前はデータが無いと測れないので
#   untested / measuring も集める。
NON_COLLECTABLE_STATUS = ('rejected', 'unusable', 'blocked_pending_user')


def _status_of(base_dir, source):
    from src.utils.source_registry import load_registry
    for e in load_registry(base_dir).get('sources', []):
        if e.get('source') == source:
            return e.get('status')
    return None


def collect(base_dir='.', start_ym='202301', end_ym='202609',
            budget=DEFAULT_BUDGET, time_budget_sec=DEFAULT_TIME_BUDGET_SEC,
            interval_sec=DEFAULT_INTERVAL_SEC, dry_run=False, force=False):
    t0 = time.time()

    # 台帳が「集めてよい」と言っていなければ何もしない（--force で上書きできる）
    st = _status_of(base_dir, SOURCE_NAME)
    if not force and st in NON_COLLECTABLE_STATUS:
        print(f'⏭ {SOURCE_NAME} は台帳の status={st} なので収集しない'
              f'（{"/".join(NON_COLLECTABLE_STATUS)} は集めない）')
        print('   別の問いのために集め直すなら --force')
        return {'total': 0, 'done': 0, 'todo': 0, 'fetched': 0, 'rows': 0,
                'failed': 0, 'left': 0, 'skipped_reason': f'status={st}'}
    db_path = os.path.join(base_dir, DB_REL)
    cache_dir = os.path.join(base_dir, CACHE_REL)
    sess = _session()

    print(f'📚 sitemap から対象を列挙 ({start_ym}〜{end_ym})')
    articles = list_articles(sess, start_ym, end_ym, cache_dir)
    done = _done_urls(db_path)
    todo = [a for a in articles if a[2] not in done]

    # 🔑 D-2① の教訓: 前に進んでいるかを毎回印字する
    print(f'   対象 {len(articles)} 本  取得済み {len(done)}  残り {len(todo)}')
    if articles:
        print(f'   期間 {articles[0][0]} 〜 {articles[-1][0]}')
    if dry_run:
        for d, v, u in todo[:10]:
            print(f'   - {d} {v} {u}')
        if len(todo) > 10:
            print(f'   ... 他 {len(todo) - 10} 本')
        return {'total': len(articles), 'done': len(done), 'todo': len(todo)}

    conn = init_db(db_path)
    if not todo:
        conn.close()
        print('✅ 残りなし')
        return {'total': len(articles), 'done': len(done), 'todo': 0,
                'fetched': 0, 'rows': 0, 'failed': 0, 'left': 0}

    fetched = rows_total = failed = 0
    for date, venue, url in todo:
        if fetched >= budget:
            print(f'⏹ 件数の上限 {budget} に達したので今回はここまで')
            break
        if time.time() - t0 > time_budget_sec:
            print(f'⏹ 時間の上限 {time_budget_sec}s に達したので今回はここまで')
            break
        try:
            html = _get(sess, url)
        except Exception as e:
            failed += 1
            _save(conn, url, date, venue, [], {}, f'http_error: {type(e).__name__}')
            print(f'   ❌ {date} {venue}: 取得失敗 {type(e).__name__}')
            time.sleep(interval_sec)
            continue
        try:
            rows, meta = parse_article(html, date, venue)
        except Exception as e:
            failed += 1
            _save(conn, url, date, venue, [], {}, f'parse_error: {type(e).__name__}')
            print(f'   ❌ {date} {venue}: パース例外 {type(e).__name__}')
            time.sleep(interval_sec)
            continue

        status = 'ok' if rows else 'no_races'
        if not rows:
            failed += 1
        _save(conn, url, date, venue, rows, meta, status)
        fetched += 1
        rows_total += len(rows)
        rej = meta.get('rejected') or []
        mark = '✅' if status == 'ok' and not rej else ('⚠' if rej else '❌')
        print(f'   {mark} {date} {venue}  {meta.get("n_races", 0)}R / {len(rows)}頭'
              + (f'  却下 {len(rej)}R' if rej else ''))
        time.sleep(interval_sec)

    left = len(todo) - fetched
    n_sig = conn.execute('SELECT COUNT(*) FROM web_signals').fetchone()[0]
    print(f'\n📊 今回 {fetched} 本 / {rows_total} 頭  失敗 {failed}  未取得 {left}')
    print(f'   DB累計 {n_sig} 頭  ({db_path})')
    if left:
        print('   もう一度同じコマンドを実行すれば続きから進みます')
    conn.close()
    return {'total': len(articles), 'done': len(done), 'todo': len(todo),
            'fetched': fetched, 'rows': rows_total, 'failed': failed, 'left': left}


def collect_for_date(base_dir, target_date, budget=6, interval_sec=1.0):
    """予想を作る当日（または対象日）の記事だけを取る。推論経路から呼ぶ。

    🔑 過去分の一括収集（`collect`）と**同じパーサ・同じDB**を通す。
       推論用に別のパース経路を作ると、学習と推論で値がズレる
       （2026-09-15 のペースモデルで実際に起きた形）。

    ⚠ 台帳で採用済みの情報源が無ければ**何もしない**（サイトを叩かない）。
    ⚠ budget は既定6本（1日3会場＋余裕）。North Star #4。

    戻り値: {'fetched':, 'rows':, 'failed':, 'skipped_reason':}
    """
    from src.utils.source_registry import adopted_sources
    if not any(e.get('source') == SOURCE_NAME for e in adopted_sources(base_dir)):
        return {'fetched': 0, 'rows': 0, 'failed': 0,
                'skipped_reason': 'not_adopted'}

    ym = target_date.replace('-', '')[:6]
    db_path = os.path.join(base_dir, DB_REL)
    cache_dir = os.path.join(base_dir, CACHE_REL)
    sess = _session()
    articles = [a for a in list_articles(sess, ym, ym, cache_dir)
                if a[0] == target_date]
    if not articles:
        return {'fetched': 0, 'rows': 0, 'failed': 0,
                'skipped_reason': 'no_article_for_date'}

    done = _done_urls(db_path)
    todo = [a for a in articles if a[2] not in done][:budget]
    if not todo:
        return {'fetched': 0, 'rows': 0, 'failed': 0,
                'skipped_reason': 'already_collected'}

    conn = init_db(db_path)
    fetched = rows_total = failed = 0
    try:
        for date, venue, url in todo:
            try:
                html = _get(sess, url)
                rows, meta = parse_article(html, date, venue)
            except Exception as e:
                failed += 1
                _save(conn, url, date, venue, [], {},
                      f'fetch_or_parse_error: {type(e).__name__}')
                time.sleep(interval_sec)
                continue
            status = 'ok' if rows else 'no_races'
            if not rows:
                failed += 1
            _save(conn, url, date, venue, rows, meta, status)
            fetched += 1
            rows_total += len(rows)
            time.sleep(interval_sec)
    finally:
        conn.close()
    return {'fetched': fetched, 'rows': rows_total, 'failed': failed,
            'skipped_reason': None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='.')
    p.add_argument('--start', default='2023-01', help='YYYY-MM')
    p.add_argument('--end', default='2026-09', help='YYYY-MM')
    p.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    p.add_argument('--time-budget', type=int, default=DEFAULT_TIME_BUDGET_SEC)
    p.add_argument('--interval', type=float, default=DEFAULT_INTERVAL_SEC)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='台帳の status を無視して集める（別の問いのための再収集）')
    a = p.parse_args()
    collect(a.base_dir, a.start.replace('-', ''), a.end.replace('-', ''),
            a.budget, a.time_budget, a.interval, a.dry_run, a.force)


if __name__ == '__main__':
    main()
