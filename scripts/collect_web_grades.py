#!/usr/bin/env python3
"""web 情報源の「馬ごとの評価（A/B/C・オススメ）」を集める（GitHub ランナーで実行）。

    python3 scripts/collect_web_grades.py --source rihua-keiba --budget 300

保存先は data/private/web_grades.db（.gitignore 済み・公開リポジトリにコミットしない。artifact で持ち回る）。
🔴 robots を守る・件数上限（--budget）と間隔（SLEEP）を守る（North Star #4）・UA は正直に名乗る。
🔴 本文（短評）は保存しない。馬名・騎手・評価記号・オススメの有無だけ。
"""
import argparse
import datetime as dt
import os
import re
import sqlite3
import sys
import time
import urllib.robotparser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, 'data', 'private', 'web_grades.db')
UA = 'keiba-ai-research-bot/0.1 (+https://github.com/hanagenuku/keiba_ai)'
SLEEP = 2.0


def parsers():
    from src.scraper import rihua
    return {rihua.SOURCE_NAME: rihua}


def init(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS web_grade_articles (
        url TEXT PRIMARY KEY, source TEXT NOT NULL, published TEXT, title TEXT, one_week INTEGER,
        n_blocks INTEGER, n_horses INTEGER, status TEXT NOT NULL, fetched_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS web_grades (
        source TEXT NOT NULL, url TEXT NOT NULL, block INTEGER NOT NULL, race_label TEXT, published TEXT,
        one_week INTEGER, horse_name TEXT NOT NULL, jockey TEXT, grade TEXT, grade_value INTEGER,
        recommended INTEGER, UNIQUE(url, block, horse_name));
    """)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--budget', type=int, default=300)
    a = ap.parse_args()
    mod = parsers()[a.source]
    import requests
    s = requests.Session()
    s.headers['User-Agent'] = UA
    rp = urllib.robotparser.RobotFileParser()
    r = s.get(mod.BASE_URL + '/robots.txt', timeout=20)
    rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    if any(x in r.text.lower() for x in ('claudebot', 'claude-web', 'claude-user', 'anthropic-ai')):
        raise SystemExit('robots が Claude 系を名指ししているので取らない')

    def get(url):
        if not rp.can_fetch(UA, url):
            return None
        time.sleep(SLEEP)
        rr = s.get(url, timeout=25)
        rr.encoding = rr.apparent_encoding or 'utf-8'
        return rr.text if rr.status_code == 200 else None

    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB)
    init(conn)
    done = {u for (u,) in conn.execute('SELECT url FROM web_grade_articles WHERE source=?', (a.source,))}
    idx = get(mod.SITEMAP) or ''
    children = re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', idx) if '<sitemapindex' in idx else []
    urls = []
    for c in children or [mod.SITEMAP]:
        x = get(c) if children else idx
        urls += [u for u in re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', x or '') if mod.is_workout_url(u)]
    todo = [u for u in dict.fromkeys(urls) if u not in done]
    print(f'{a.source}: 記事 {len(set(urls))} / 取得済み {len(done)} / 今回 {min(len(todo), a.budget)}', flush=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    n_ok = n_rows = 0
    for u in todo[: a.budget]:
        try:
            h = get(u)
        except Exception as e:
            conn.execute('INSERT OR REPLACE INTO web_grade_articles VALUES (?,?,?,?,?,?,?,?,?)',
                         (u, a.source, None, None, None, 0, 0, f'error:{type(e).__name__}', now))
            continue
        if not h:
            conn.execute('INSERT OR REPLACE INTO web_grade_articles VALUES (?,?,?,?,?,?,?,?,?)',
                         (u, a.source, None, None, None, 0, 0, 'not_fetched', now))
            continue
        art = mod.parse_article(h)
        nh = 0
        for bi, b in enumerate(art['blocks']):
            for hh in b['horses']:
                conn.execute('INSERT OR IGNORE INTO web_grades VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                             (a.source, u, bi, b['race_label'], art['published'], int(art['one_week']),
                              hh['name'], hh['jockey'], hh['grade'], hh['grade_value'], int(hh['recommended'])))
                nh += 1
        conn.execute('INSERT OR REPLACE INTO web_grade_articles VALUES (?,?,?,?,?,?,?,?,?)',
                     (u, a.source, art['published'], art['title'][:80], int(art['one_week']),
                      len(art['blocks']), nh, 'ok', now))
        n_ok += 1
        n_rows += nh
        conn.commit()
    conn.commit()
    tot = conn.execute('SELECT COUNT(*), SUM(grade IS NOT NULL), SUM(recommended) FROM web_grades WHERE source=?',
                       (a.source,)).fetchone()
    print(f'今回 {n_ok}記事 / {n_rows}頭。累計 {tot[0]}頭（評価つき {tot[1]}・オススメ {tot[2]}）')


if __name__ == '__main__':
    main()
