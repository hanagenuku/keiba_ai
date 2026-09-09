"""出走表が JRA 公式にいつ載るかを、生ページをダンプして確かめる。

## なぜ必要か

2026-09-08頃 に `get_kaisai_on_date` が今週末(9/12・9/13)を返さなかったことから
「JRAがまだ載せていない」と結論しかけたが、**それは推測だった**。
実際に確認できていたのは「`pw01dli00/F3` の中に `pw01drl00...` 形式のリンクが
無かった」ことだけで、

  ① 別の形式で載っている
  ② 枠順未定の出走表が別ページにある
  ③ thisweek 側には載っている

のいずれでもない、とは確かめていない。
CLAUDE.md が繰り返し警告している「**空振りと該当なしは別物**」（2026-08-16）の
再発なので、ページの中身そのものを出して判定する。

## 第1回(2026-09-09 水)の結果と、それを受けた変更

出走表一覧ページは**空ではなかった**:
  pw01drl10×6 / pw17hde10×2 / pw01d5010×1 / pw01dde10×4 / pw151ou10×4
というトークンを持つリンクがあり、他2ページには無い。
一方 **8桁の日付は1件も現れない**。
`get_kaisai_on_date` が探しているのは `pw01drl00` + 8桁日付なので、
「探し方が違う」のか「今週末がまだ無い」のかが**この時点では区別できない**。

そこで:
  ① リンク先頭40件（＝全部グローバルナビだった）ではなく、
     **CNAME を持つリンクを全部**出す
  ② そのうち中身を持つものを **実際に開いて**、何が載っているかを見る
  ③ 枠順確定前の情報源の候補である「特別レース登録馬」(accessT) も見る

## 第2回(2026-09-09 水 09:09 JST)の結果 — 水曜については決着した

■ 今週末(20260912/20260913)は出馬表・オッズ・レース結果・払戻金の
  どのページにも1件も現れない。
  出馬表一覧に載っているのは先週末(20260905/20260906)の6件だけで、
  ラベルが「4回中山1日 **馬番確定**」＝終わった開催の出馬表。
  → 「JRAがまだ載せていない」は今回、全リンクを列挙した結果として確認できた。

■ ⚠ パーサーが探すのは pw01drl00 だが、水曜に載っているのは pw01drl10。
  00=今週分 / 10=過去分 らしい（結果ページも過去分は pw01srl10）。
  つまり水曜の空振りは「探し方が古い」のではなく「探すべきものがまだ無い」。

■ オッズは原理的に無い。パリミュチュエル方式なので発売前に盤が存在しない。
  オッズ選択ページに今週末のリンクはゼロ、あるのは先週末の確定オッズ4件のみ。

■ ✅ 唯一「特別レース登録馬」(accessT.html pw03trl00/29) だけが今週末を持つ。
  中山・阪神の R9/R10/R11 × 9/12・9/13 = 12レース。
  🔴 ただし中身は **馬名と負担重量の2列だけ**（実測）:
      | 馬名             | 負担重量 |
      | イブニングタイド   | 56.0   |
      | エリム           | 56.0   |
  騎手・枠番・馬番・オッズは無い。しかも「登録馬」なので出走馬とも一致しない。
  平場（未勝利・1勝クラス）は含まれない。

■ 🔴 未確認: 木曜・金曜のどこで出馬表が出るか。金曜夜は動くことが分かっている
  （本番が毎週それで動いている）が、境界は測っていない。
  ⚠ 火曜は測っていない。ただし JRA は一度載せたものを引っ込めないので、
    水曜に無い以上、火曜にも無いと考えてよい（これは推論であって実測ではない）。

⚠ 完全な読み取り専用。DB・モデル・latest.json には一切書き込まない。
⚠ リクエストは十数件（FOLLOW_LIMIT で上限を持つ）。
"""
import os
import re
import sys
import time
import unicodedata
from collections import Counter, OrderedDict

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._session import create_session          # noqa: E402
from src.utils.config import JRA_BASE                # noqa: E402

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DATE_RE = re.compile(r'\b(20\d{6})\b')
TOKEN_RE = re.compile(r'(pw[0-9a-z]{5,7})')
# CNAME は "pw01dde1006202609120120260912" のような英数字の並び。
# "/F3" のような suffix が付くこともあるので区切りごと拾う。
CNAME_RE = re.compile(r"['\"]([0-9A-Za-z]{8,}(?:/[0-9A-Za-z]+)?)['\"]")
ENDPOINT_RE = re.compile(r"(access[0-9A-Za-z]+\.html)")

FOLLOW_LIMIT = 4   # 開いてみるリンクの上限（リクエスト数の歯止め）


def _norm(s):
    return unicodedata.normalize('NFKC', s or '')


def links_with_cname(soup):
    """CNAME を持つリンクだけを (text, endpoint, cname) で返す。

    JRADB は href ではなく onclick の doAction('/JRADB/accessX.html','CNAME')
    に本体を埋めるので両方見る。グローバルナビは素の href なのでここに出ない。
    """
    out = OrderedDict()
    for a in soup.find_all('a'):
        blob = ' '.join(filter(None, [a.get('href', ''), a.get('onclick', '')]))
        cn = CNAME_RE.search(blob)
        if not cn:
            continue
        ep = ENDPOINT_RE.search(blob)
        key = (ep.group(1) if ep else '?', cn.group(1))
        if key not in out:
            out[key] = _norm(a.get_text(' ', strip=True))[:34]
    return [(t, ep, cn) for (ep, cn), t in out.items()]


def dump(name, html, show_body=300):
    soup = BeautifulSoup(html, 'lxml')
    txt = _norm(soup.get_text(' ', strip=True))
    title = soup.title.get_text(strip=True) if soup.title else '(なし)'
    print(f'\n{"=" * 92}')
    print(f'■ {name}')
    print('=' * 92)
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
    print(f'  pw系トークン: {dict(toks) if toks else "（1件も無い）"}')

    dates = Counter(DATE_RE.findall(joined + ' ' + txt))
    if dates:
        print('  ページ内の日付: '
              + ', '.join(f'{d}({n})' for d, n in sorted(dates.items())))
    else:
        print('  ページ内の日付: （1件も無い）')

    lk = links_with_cname(soup)
    print(f'  --- CNAMEを持つリンク 全{len(lk)}件 ---')
    for t, ep, cn in lk:
        print(f'    {t:<36}{ep:<18}{cn}')
    if show_body:
        print('  --- 本文冒頭 ---')
        print(f'    {txt[:show_body]}')
    return lk


def dump_table(name, html):
    """表がある前提のページ用。見出しと先頭数行をそのまま出す。

    「登録馬が並んでいるのか」「枠番の列があるのか」を目で見て判定するため、
    列名を推測せず生のセルを出す。
    """
    soup = BeautifulSoup(html, 'lxml')
    txt = _norm(soup.get_text(' ', strip=True))
    title = soup.title.get_text(strip=True) if soup.title else '(なし)'
    tables = soup.find_all('table')
    print(f'\n  {"-" * 88}')
    print(f'  ■ {name}')
    print(f'    タイトル : {title[:70]}')
    print(f'    本文長   : {len(txt)} 字 / table {len(tables)}個')
    dates = Counter(DATE_RE.findall(txt))
    print(f'    日付: {dict(dates) if dates else "（1件も無い）"}')
    for ti, tb in enumerate(tables[:2]):
        rows = tb.find_all('tr')
        print(f'    --- table[{ti}] {len(rows)}行 ---')
        for r in rows[:8]:
            cells = [_norm(c.get_text(' ', strip=True))[:18]
                     for c in r.find_all(['th', 'td'])]
            if any(cells):
                print('      | ' + ' | '.join(cells))
    if not tables:
        print(f'    --- 本文 ---\n    {txt[:600]}')


def post_cname(sess, endpoint, cname):
    r = sess.post(f'{JRA_BASE}/JRADB/{endpoint}',
                  data={'cname': cname, 'CNAME': cname},
                  headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    return r.text


def main():
    sess = create_session()

    # ① 出走表一覧（get_kaisai_on_date が最初に見るページ）
    html = post_cname(sess, 'accessD.html', 'pw01dli00/F3')
    top_links = dump('出走表一覧  accessD.html  cname=pw01dli00/F3', html)

    # ② thisweek（フォールバック先）
    r2 = sess.get(f'{JRA_BASE}/keiba/thisweek/', headers=HEADERS, timeout=15)
    r2.encoding = 'shift_jis'
    dump('今週の開催  /keiba/thisweek/', r2.text)

    # ③ 特別レース登録馬（枠順確定前の情報源の候補）
    html3 = ''
    try:
        html3 = post_cname(sess, 'accessT.html', 'pw03trl00/29')
        dump('特別レース登録馬  accessT.html  cname=pw03trl00/29', html3)
    except Exception as e:      # noqa: BLE001
        print(f'\n■ 特別レース登録馬: 取得失敗 {type(e).__name__}: {e}')

    # ③' 特別レース登録馬の「中身」を開く（水曜に今週末を持つ唯一の経路）
    #     ここに馬名・騎手・負担重量が並ぶのか、枠順が無いだけなのかを見る。
    print(f'\n{"=" * 92}')
    print('■ 特別レース登録馬の中身（最大2件）')
    print('=' * 92)
    try:
        tlinks = links_with_cname(BeautifulSoup(html3, 'lxml'))
    except Exception:      # noqa: BLE001
        tlinks = []
    n = 0
    for t, ep, cn in tlinks:
        if 'pw03tde' not in cn:
            continue
        if n >= 2:
            break
        try:
            detail = post_cname(sess, ep, cn)
        except Exception as e:      # noqa: BLE001
            print(f'\n■ {t}: 取得失敗 {type(e).__name__}: {e}')
            continue
        dump_table(f'└ {t}  [{ep} {cn}]', detail)
        n += 1
        time.sleep(0.5)
    if n == 0:
        print('  pw03tde 形式のリンクが1件も無い（＝この経路も空）')

    # ④ ①で見つかった中身つきリンクを実際に開く
    print(f'\n{"=" * 92}')
    print(f'■ ①のリンクを実際に開く（最大{FOLLOW_LIMIT}件）')
    print('=' * 92)
    per_token = Counter()
    followed = 0
    for t, ep, cn in top_links:
        if followed >= FOLLOW_LIMIT:
            break
        # 日付を持たない CNAME はグローバルナビなので開かない
        if not DATE_RE.search(cn):
            continue
        m = TOKEN_RE.search(cn)
        tok = m.group(1) if m else cn[:9]
        # トークンの種類ごとに2件までにして、種類を広く見る
        if per_token[tok] >= 2:
            continue
        per_token[tok] += 1
        try:
            sub = post_cname(sess, ep, cn)
        except Exception as e:      # noqa: BLE001
            print(f'\n■ {t} ({ep} {cn}): 取得失敗 {type(e).__name__}: {e}')
            continue
        dump_table(f'└ {t}  [{ep} {cn}]', sub)
        followed += 1
        time.sleep(0.5)

    print(f'\n{"=" * 92}')
    print('■ 判定の目安')
    print('=' * 92)
    print('  ・今週末(20260912/20260913)の日付が現れる → JRAは載せている。')
    print('    取れないのは get_kaisai_on_date の探し方の問題')
    print('  ・先週末(20260906/20260907)の日付しか現れない → まだ載っていない')
    print('  ・馬名の並んだ table が出る → 枠順未定でも出走馬は取れる')


if __name__ == '__main__':
    main()
