"""出走表が JRA 公式にいつ載るかを、生ページをダンプして確かめる。

## なぜ必要か

2026-09-08(火) に `get_kaisai_on_date` が今週末(9/12・9/13)を返さなかったことから
「JRAがまだ載せていない」と結論しかけたが、**それは推測だった**。
実際に確認できていたのは「`pw01dli00/F3` の中に `pw01drl00...` 形式のリンクが
無かった」ことだけで、

  ① 別の形式で載っている
  ② 枠順未定の出走表が別ページにある
  ③ thisweek 側には載っている

のいずれでもない、とは確かめていない。
CLAUDE.md が繰り返し警告している「**空振りと該当なしは別物**」（2026-08-16）の
再発なので、ページの中身そのものを出して判定する。

## 何を出すか

`get_kaisai_on_date` が見る2つのページを、パターンで絞らずに丸ごと要約する:
  - タイトル / 本文長 / table 数
  - onclick と href に現れる **すべての** pw01系トークン（種類ごとに件数）
  - ページ内に現れる **すべての8桁日付**（＝どの開催日が載っているか）
  - リンクの先頭40件（文字列つき）

⚠ 完全な読み取り専用。DB・モデル・latest.json には一切書き込まない。
⚠ リクエストは3件だけ。
"""
import os
import re
import sys
import unicodedata
from collections import Counter

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._session import create_session          # noqa: E402
from src.utils.config import JRA_BASE                # noqa: E402

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DATE_RE = re.compile(r'\b(20\d{6})\b')
TOKEN_RE = re.compile(r'(pw[0-9a-z]{5,7})')


def dump(name, html):
    soup = BeautifulSoup(html, 'lxml')
    txt = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    title = soup.title.get_text(strip=True) if soup.title else '(なし)'
    print(f'\n{"=" * 88}')
    print(f'■ {name}')
    print('=' * 88)
    print(f'  タイトル : {title[:70]}')
    print(f'  本文長   : {len(txt)} 字 / table {len(soup.find_all("table"))}個 '
          f'/ a {len(soup.find_all("a"))}個')

    blobs = []
    for tag in soup.find_all(True):
        for attr in ('onclick', 'href'):
            v = tag.get(attr)
            if v:
                blobs.append(v)
    joined = ' '.join(blobs)

    toks = Counter(TOKEN_RE.findall(joined))
    print(f'  pw系トークンの種類: {dict(toks) if toks else "（1件も無い）"}')

    dates = Counter(DATE_RE.findall(joined + ' ' + txt))
    if dates:
        print(f'  ページ内に現れる日付: '
              f'{", ".join(f"{d}({n})" for d, n in sorted(dates.items()))}')
    else:
        print('  ページ内に現れる日付: （1件も無い）')

    print(f'  --- リンク先頭40件 ---')
    for a in soup.find_all('a')[:40]:
        t = unicodedata.normalize('NFKC', a.get_text(' ', strip=True))[:26]
        blob = ' '.join(filter(None, [a.get('href', ''), a.get('onclick', '')]))[:70]
        if t or blob:
            print(f'    {t:<28}{blob}')
    print(f'  --- 本文冒頭 ---')
    print(f'    {txt[:300]}')


def main():
    sess = create_session()

    # ① 出走表一覧（get_kaisai_on_date が最初に見るページ）
    r = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                  data={'cname': 'pw01dli00/F3', 'CNAME': 'pw01dli00/F3'},
                  headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    dump('出走表一覧  accessD.html  cname=pw01dli00/F3', r.text)

    # ② thisweek（フォールバック先）
    r2 = sess.get(f'{JRA_BASE}/keiba/thisweek/', headers=HEADERS, timeout=15)
    r2.encoding = 'shift_jis'
    dump('今週の開催  /keiba/thisweek/', r2.text)

    # ③ 出馬表トップ（枠順未定の段階で別ページに出ていないかの確認）
    r3 = sess.get(f'{JRA_BASE}/keiba/', headers=HEADERS, timeout=15)
    r3.encoding = 'shift_jis'
    dump('競馬メニュー  /keiba/', r3.text)

    print(f'\n{"=" * 88}')
    print('■ 判定の目安')
    print('=' * 88)
    print('  ・今週末の日付が現れる → JRAは載せている。取れないのはこちらのパーサーの問題')
    print('  ・日付が1件も現れない → その時点では本当に載っていない')
    print('  ・pw系トークンが1件も無い → ページ自体が空 or 取得に失敗している（空振り）')


if __name__ == '__main__':
    main()
