#!/usr/bin/env python3
"""過去分がある web 情報源の「記事の作り」を下見する（GitHub ランナーで実行・手動）。

    python3 scripts/probe_web_source.py keibanomiryoku rihua-keiba speedkeiba

目的: 収集器（パーサ）を書く前に、記事の URL の種類と HTML の骨組みを知る。
🔴 公開リポジトリのログに出るので、**本文は出さない**。出すのは URL の型・件数・見出しの文字列・
   表の列名・印（◎○▲△★ A〜E など）の出現数・馬番らしき数字の並びの有無だけ。
🔴 robots を守る。1情報源あたり最大 MAX_REQ 回・間隔 SLEEP 秒。UA は正直に名乗る。
"""
import re
import sys
import time
import urllib.robotparser
from collections import Counter, defaultdict
from urllib.parse import unquote, urlparse

UA = 'keiba-ai-research-bot/0.1 (+https://github.com/hanagenuku/keiba_ai)'
MAX_REQ = 40
SLEEP = 2.0
BASES = {
    'keibanomiryoku': 'https://www.keibanomiryoku.com',
    'rihua-keiba': 'https://rihua-keiba.com',
    'speedkeiba': 'https://speedkeiba.com',
}
MARKS = ['◎', '○', '〇', '▲', '△', '☆', '★', '注', '✕', '消']


def locs(xml):
    return re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', xml or '')


def lastmods(xml):
    return re.findall(r'<lastmod>\s*(\d{4}-\d{2})', xml or '')


def pattern(u):
    p = unquote(urlparse(u).path)
    p = re.sub(r'\d{4}', 'YYYY', p)
    p = re.sub(r'\d+', 'N', p)
    parts = [x for x in p.split('/') if x]
    if parts and len(parts[-1]) > 12:
        tail = parts[-1]
        kw = [k for k in ('results', 'work-out', 'workout', 'yosou', 'prediction', 'kekka', '追い切り', '予想',
                          '調教', '結果', '血統', 'データ', '展開', '回顧') if k in tail]
        parts[-1] = '*' + ('[' + ','.join(kw) + ']' if kw else '')
    return '/' + '/'.join(parts[:4])


class Fetcher:
    def __init__(self, base):
        import requests
        self.s = requests.Session()
        self.s.headers['User-Agent'] = UA
        self.n = 0
        self.rp = urllib.robotparser.RobotFileParser()
        r = self.s.get(base + '/robots.txt', timeout=20)
        self.n += 1
        self.rp.parse(r.text.splitlines() if r.status_code == 200 else [])

    def get(self, url):
        if self.n >= MAX_REQ or not self.rp.can_fetch(UA, url):
            return None
        self.n += 1
        time.sleep(SLEEP)
        try:
            r = self.s.get(url, timeout=25)
            r.encoding = r.apparent_encoding or 'utf-8'
            return r.text if r.status_code == 200 else None
        except Exception:
            return None


def structure(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for t in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        t.decompose()
    body = soup.find('article') or soup.find('main') or soup.body or soup
    text = body.get_text('\n', strip=True)
    heads = [h.get_text(' ', strip=True)[:40] for h in body.find_all(['h2', 'h3'])][:14]
    tables = []
    for t in body.find_all('table')[:4]:
        rows = t.find_all('tr')
        hdr = [c.get_text(' ', strip=True)[:10] for c in (rows[0].find_all(['th', 'td']) if rows else [])][:10]
        tables.append(dict(rows=len(rows), header=hdr))
    marks = {m: text.count(m) for m in MARKS if text.count(m)}
    grades = Counter(re.findall(r'(?<![A-Za-z])([SABCDE])(?:評価|ランク|判定)?(?![A-Za-z])', text))
    nums = len(re.findall(r'(?:^|\n)\s*(?:[①-⑱]|\d{1,2}\s*[番.．:：])', text))
    return dict(chars=len(text), h=heads, tables=tables, marks=marks,
                grades={k: v for k, v in grades.items() if v >= 3}, horse_num_lines=nums,
                classes=Counter(c for el in body.find_all(class_=True) for c in el.get('class', []))
                .most_common(12))


def probe(name):
    base = BASES[name]
    f = Fetcher(base)
    print(f'\n######## {name}  {base}', flush=True)
    idx = f.get(base + '/sitemap.xml') or f.get(base + '/wp-sitemap.xml') or f.get(base + '/sitemap_index.xml')
    children = locs(idx) if idx and '<sitemapindex' in idx else []
    urls, months = [], Counter()
    for c in children[:12] if children else []:
        x = f.get(c)
        if x:
            urls += locs(x)
            months.update(m[:4] for m in lastmods(x))
    if not children and idx:
        urls = locs(idx)
        months.update(m[:4] for m in lastmods(idx))
    pats = Counter(pattern(u) for u in urls)
    print(f'sitemap children {len(children)} / 見た URL {len(urls)} / 年 {dict(sorted(months.items()))}')
    for p, n in pats.most_common(15):
        print(f'  {n:5d}  {p}')
    by = defaultdict(list)
    for u in urls:
        by[pattern(u)].append(u)
    for p, n in pats.most_common(4):
        for u in (by[p][0], by[p][len(by[p]) // 2]):
            h = f.get(u)
            if not h:
                continue
            s = structure(h)
            print(f'--- 型 {p}  例 {unquote(urlparse(u).path)[:90]}')
            print(f'    文字数 {s["chars"]}  馬番らしき行 {s["horse_num_lines"]}  印 {s["marks"]}  評価 {s["grades"]}')
            print(f'    見出し {s["h"]}')
            print(f'    表 {s["tables"]}')
            print(f'    class {s["classes"]}')
    print(f'requests {f.n}')


if __name__ == '__main__':
    for n in sys.argv[1:] or list(BASES):
        try:
            probe(n)
        except Exception as e:
            print(f'{n}: 失敗 {type(e).__name__}: {e}')
