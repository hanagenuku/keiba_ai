#!/usr/bin/env python3
"""web 情報源の一斉下見（GitHub ランナーで実行・手動）。

    python3 scripts/survey_web_sources.py --out websig/SURVEY_YYYY-MM-DD

2026-11-20 ユーザー指示「最初は広く広くとこから検証して削っていく」。
各情報源について、**取得してよいか（robots.txt）**と**過去分が機械で読める形で残っているか（sitemap）**だけを見る。
過去分があれば、すぐに大量の過去データで測れる（kayochin・yosounohone と同じ道）。無ければ朝の台帳に貯めるしかない。

🔴 中身（予想・指数）は取らない。robots.txt・sitemap だけ。1情報源あたり最大 MAX_REQ 回・間隔 SLEEP 秒（North Star #4）。
🔴 UA は正直に名乗る。robots で禁止されているパスは取りに行かない。
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.robotparser
from collections import Counter
from urllib.parse import urljoin, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = 'keiba-ai-research-bot/0.1 (+https://github.com/hanagenuku/keiba_ai)'
MAX_REQ = 8
SLEEP = 1.5
SKIP_STATUS = ('blocked_pending_user',)
SKIP_SOURCE = ('netkeiba',)          # 公開リポジトリに netkeiba 由来のものを書かない
EXTRA = [  # 台帳に出てくるが台帳（source_registry）に無いホスト
    ('jra-official', 'https://www.jra.go.jp'),
]
CANDIDATE_MAPS = ('/sitemap.xml', '/sitemap_index.xml', '/wp-sitemap.xml')


class Budget:
    def __init__(self, sess):
        self.sess, self.n = sess, 0

    def get(self, url):
        if self.n >= MAX_REQ:
            return None, 'budget'
        self.n += 1
        time.sleep(SLEEP)
        try:
            r = self.sess.get(url, timeout=25)
            return r, r.status_code
        except Exception as e:
            return None, type(e).__name__


def locs(xml):
    return re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', xml or '')


def years(xml):
    ys = re.findall(r'<lastmod>\s*(\d{4})-', xml or '')
    c = Counter(ys)
    return dict(sorted(c.items()))


def survey(name, base, sess):
    out = dict(source=name, url=base)
    b = Budget(sess)
    host = urlparse(base).scheme + '://' + urlparse(base).netloc
    r, st = b.get(host + '/robots.txt')
    out['robots_status'] = st
    rp = urllib.robotparser.RobotFileParser()
    text = r.text if (r is not None and r.status_code == 200) else ''
    rp.parse(text.splitlines())
    out['robots_disallow'] = [l.strip() for l in text.splitlines() if l.lower().startswith('disallow')][:15]
    out['allowed_root'] = rp.can_fetch(UA, host + '/') if text else True
    out['allowed_base'] = rp.can_fetch(UA, base) if text else True
    maps = [l.split(':', 1)[1].strip() for l in text.splitlines() if l.lower().startswith('sitemap:')]
    if not maps:
        maps = [host + p for p in CANDIDATE_MAPS]
    out['sitemap'] = None
    for m in maps:
        if not rp.can_fetch(UA, m) and text:
            continue
        r, st = b.get(m)
        if r is None or r.status_code != 200 or '<loc>' not in r.text:
            continue
        xml = r.text
        children = locs(xml) if '<sitemapindex' in xml else []
        info = dict(url=m, is_index=bool(children), n_children=len(children),
                    children_sample=children[:6] + (['…'] + children[-4:] if len(children) > 10 else []))
        if not children:
            urls = locs(xml)
            info.update(n_urls=len(urls), lastmod_years=years(xml), url_sample=urls[:5])
        else:
            # 投稿の sitemap を最大2つだけ見る（古いものと新しいもの）
            posts = [c for c in children if re.search(r'post|article|blog|\d{4}', c)] or children
            looked = []
            for c in [posts[0], posts[-1]][: 2 if len(posts) > 1 else 1]:
                if not rp.can_fetch(UA, c) and text:
                    continue
                rr, _ = b.get(c)
                if rr is not None and rr.status_code == 200:
                    us = locs(rr.text)
                    looked.append(dict(url=c, n_urls=len(us), lastmod_years=years(rr.text), url_sample=us[:4]))
            info['looked'] = looked
        out['sitemap'] = info
        break
    out['requests'] = b.n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import requests
    sess = requests.Session()
    sess.headers['User-Agent'] = UA
    reg = json.load(open(os.path.join(ROOT, 'data', 'source_registry.json'), encoding='utf-8'))
    src = reg.get('sources', reg)
    src = src if isinstance(src, list) else list(src.values())
    targets = [(e['source'], e['url']) for e in src if isinstance(e, dict) and e.get('url')
               and e.get('status') not in SKIP_STATUS and e.get('source') not in SKIP_SOURCE]
    targets += EXTRA
    res = []
    for name, url in targets:
        r = survey(name, url, sess)
        res.append(r)
        sm = r.get('sitemap') or {}
        print(f"{name:22s} robots={r['robots_status']} root={'OK' if r['allowed_root'] else 'NG'} "
              f"sitemap={'あり' if sm else 'なし'} children={sm.get('n_children')} req={r['requests']}", flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)) or '.', exist_ok=True)
    with open(os.path.join(ROOT, a.out + '.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print('書き出し:', a.out + '.json')


if __name__ == '__main__':
    main()
