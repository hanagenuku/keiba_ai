#!/usr/bin/env python3
"""JRA の結果ページで重賞の格付け（GⅠ/GⅡ/GⅢ）がどう書かれているかを見る（調査用・1回）。

2026-11-25 博士「各場のメインレースがＧ３以上の重賞のみ…」。history.db の race_class は
重賞でも 'OP' になっていて（`_extract_class` は '(G3)' という文字を探すが見つからない）、
格付けをどこから取るかを決めるために、実際のページの該当部分を出す。

    python3 scripts/probe_jra_grade.py 20261123 05 11    # 日付・場コード・レース番号
"""
import re
import sys

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, '.')
from src.scraper import jra_scraper as J  # noqa: E402


def main():
    date, pc, rn = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sess = requests.Session()
    r0 = sess.post(f'{J.JRA_BASE}/JRADB/accessS.html', data={'CNAME': 'pw01sli00/AF'}, timeout=15)
    r0.encoding = 'shift_jis'
    base = None
    for tag in BeautifulSoup(r0.text, 'lxml').find_all(onclick=True):
        m = re.search(r'pw01srl\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/(\w{2})', tag.get('onclick', ''))
        if m and m.group(1) == pc and m.group(5) == date:
            base = f'pw01sde10{pc}{m.group(2)}{m.group(3)}{m.group(4)}'
    print('base', base)
    if not base:
        return
    r01 = J.find_r01_result(base, date, sess)
    soup = None
    for sx in (J.calc_suffix(r01, rn), f'{(r01 + (rn - 1) * 181) % 256:02X}'):
        soup = J._try_fetch_result(sess, base, rn, date, sx)
        if soup is not None:
            break
    if soup is None:
        print('ページが取れない')
        return
    print('--- img（alt / src）')
    for im in soup.find_all('img'):
        print(repr(im.get('alt')), im.get('src'))
    print('--- class に grade / icon を含む要素')
    for el in soup.find_all(class_=re.compile('grade|icon', re.I)):
        print(el.name, el.get('class'), repr(el.get_text(' ', strip=True)[:60]))
    print('--- 「G」「Ｇ」「Ⅲ」を含む文字列')
    for t in soup.find_all(string=re.compile('G[1-3Ⅰ-Ⅲ]|Ｇ|GⅠ|GⅡ|GⅢ|Jpn')):
        print(repr(t.strip()[:80]), '<-', t.parent.name, t.parent.get('class'))
    h = soup.find(class_=re.compile('race_header|race_title|race_name', re.I))
    print('--- 見出しの HTML')
    print(str(h)[:3000] if h else str(soup.find_all('table')[0])[:2000])


if __name__ == '__main__':
    main()
