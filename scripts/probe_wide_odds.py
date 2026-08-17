"""ワイド（および他券種）のオッズ盤が JRA公式から取得できるかを実データで確認する。

## なぜ必要か

1レースで3着内に入るのは必ず3頭。したがって次が**厳密に**成立する:

    Σ_{B≠A} P(A,B がともに3着内) = 2 × P(A が3着内)

つまり「ワイドのオッズ盤」から各馬の3着内確率が導出でき、それは
「複勝のオッズ盤」に直接書いてある値と一致しなければならない。
一致しなければ、ワイドを買う群衆と複勝を買う群衆が同じ出来事に
違う値段を付けていることになる（＝予測を使わない裁定の芽）。

正規化定数（どちらも合計3）が厳密に分かっているため、
**控除率を推定する必要がない**のがこの手法の強み。

## 何を調べるか

現行の `fetch_odds_for_race` は accessO.html に

    CNAME = {odds_base}{race_num:02d}{date_str}Z/{suffix}

を投げて単勝・複勝を得ている。この **`Z` が券種の指定子ではないか**が仮説。
A〜Z を総当たりし、返ってきたページの構造（table数・見出し・
組番らしき文字列の有無）をログに出す。

⚠ 完全な読み取り専用。DB・モデル・latest.json には一切書き込まない。
⚠ 1レースぶんだけ・待機を入れて叩く（総リクエストは50件未満）。
"""
import re
import sys
import time
import unicodedata

from bs4 import BeautifulSoup

sys.path.insert(0, '.')
from scripts._session import create_session            # noqa: E402
from src.scraper.jra_scraper import (              # noqa: E402
    JRA_BASE, HEADERS, get_kaisai_on_date, find_r01_odds, calc_suffix,
)

SLEEP = 1.0
PROBE_DAYS_BACK = 9
RACE_NUM = 1


def _post(sess, endpoint, cname):
    r = sess.post(f'{JRA_BASE}/JRADB/{endpoint}',
                  data={'cname': cname, 'CNAME': cname},
                  headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    return r


def _describe(html):
    """ページの中身を要約する。組番らしき文字列があるかが最大の関心事。"""
    soup = BeautifulSoup(html, 'lxml')
    txt = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    tables = soup.find_all('table')
    # 「3-7」「3 - 7」のような組番、および「X.X - Y.Y」のオッズ範囲
    combos = re.findall(r'\b\d{1,2}\s*[-−]\s*\d{1,2}\b', txt)
    ranges = re.findall(r'\d{1,4}\.\d\s*[-−~〜]\s*\d{1,4}\.\d', txt)
    singles = re.findall(r'(?<![\d.])\d{1,4}\.\d(?![\d.])', txt)
    kws = {k: txt.count(k) for k in
           ['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '3連複', '3連単',
            '三連複', '三連単'] if txt.count(k)}
    title = soup.title.get_text(strip=True) if soup.title else ''
    return dict(title=title, n_table=len(tables), n_combo=len(combos),
                n_range=len(ranges), n_single=len(singles), kws=kws,
                head=txt[:120])


def main():
    sess = create_session()

    # 開催日を自動で探す（日付ハードコードで空振りした前科があるため）
    import datetime as dt
    base = date = None
    today = dt.date.today()
    for back in range(PROBE_DAYS_BACK):
        d = (today - dt.timedelta(days=back)).strftime('%Y%m%d')
        try:
            links = get_kaisai_on_date(sess, d)
        except Exception as e:
            print(f'  {d}: 取得失敗 {e}')
            continue
        if links:
            # ⚠ 戻り値は {base: 日付}。キーが base（2026-08-15に取り違えた）
            base, date = list(links.keys())[0], d
            print(f'📅 開催日 {d} / base={base}')
            break
        time.sleep(SLEEP)
    if not base:
        print(f'❌ 直近{PROBE_DAYS_BACK}日に開催が見つからない。中止。')
        return

    odds_base = base.replace('pw01dde', 'pw151ous')
    print(f'   odds_base={odds_base}')

    r01 = find_r01_odds(odds_base, date, sess)
    if r01 is None:
        print('❌ オッズR01のsuffixが見つからない。中止。')
        return
    sx = calc_suffix(r01, RACE_NUM)
    print(f'   R01 suffix={r01:02X} → R{RACE_NUM:02d} suffix={sx}\n')

    print('=' * 88)
    print('■ CNAME の "Z" を A〜Z に振って、返るページの中身を見る')
    print('=' * 88)
    print(f'{"文字":<5}{"table":>6}{"組番":>6}{"範囲":>6}{"単値":>6}  '
          f'{"券種キーワード":<28}タイトル')
    print('-' * 88)
    hits = []
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        cn = f'{odds_base}{RACE_NUM:02d}{date}{ch}/{sx}'
        try:
            resp = _post(sess, 'accessO.html', cn)
        except Exception as e:
            print(f'{ch:<5}  通信例外 {type(e).__name__}')
            time.sleep(SLEEP)
            continue
        if 'パラメータエラー' in resp.text:
            print(f'{ch:<5}  パラメータエラー')
            time.sleep(SLEEP)
            continue
        d_ = _describe(resp.text)
        kw = ' '.join(f'{k}{v}' for k, v in list(d_['kws'].items())[:5])
        print(f'{ch:<5}{d_["n_table"]:>6}{d_["n_combo"]:>6}{d_["n_range"]:>6}'
              f'{d_["n_single"]:>6}  {kw[:27]:<28}{d_["title"][:28]}')
        if d_['n_table']:
            hits.append((ch, d_, resp.text))
        time.sleep(SLEEP)

    print('\n' + '=' * 88)
    print('■ 中身のあったページの本文冒頭（券種を見分けるため）')
    print('=' * 88)
    for ch, d_, html in hits[:8]:
        print(f'\n--- "{ch}" ---')
        print(f'  {d_["head"]}')
        soup = BeautifulSoup(html, 'lxml')
        for t in soup.find_all('table')[:2]:
            rows = t.find_all('tr')[:3]
            for tr in rows:
                cells = [unicodedata.normalize('NFKC', c.get_text(strip=True))
                         for c in tr.find_all(['td', 'th'])]
                if cells:
                    print(f'    {cells[:12]}')

    print('\n' + '=' * 88)
    print('■ 判定の目安')
    print('=' * 88)
    print('  ・「組番」が多数 かつ「ワイド」を含む → ワイドのオッズ盤が取れる')
    print('  ・どの文字でも単勝・複勝しか出ない → accessO は単複専用。')
    print('    その場合はワイドの取得手段が別に必要（要追加調査）')


if __name__ == '__main__':
    main()
