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
⚠ 1レースぶんだけ叩く。ただし `find_r01_odds` は R01 の suffix を
   0x00〜0xFF で総当たりするため、リクエストは1開催日あたり最大256件になる
   （本番の週次ワークフローと同じ探索。当初「50件未満」と書いていたのは誤り）。

## ⚠ 実行できるタイミングが限られる（2026-08-20に判明）

このプローブは 2026-08-17 に2回走らせて**2回とも1件も盤を見ずに終了していた**。
1回目は `get_kaisai_on_date` の引数順の取り違え、2回目は**探索の向き**が原因。

`get_kaisai_on_date` が読む `pw01dli00` は「**今週これからの開催**」しか載せない。
過去に遡って探すと、月曜に走らせた時点で必ず空振りになる（8/15・8/16 は実際に
開催があり結果も取れているのに、この経路では見えない）。よって:

  1. 開催日は**前方（これからの開催）**へ探す
  2. オッズ盤は**発売中しか存在しない**ので、実行は**金曜夜〜日曜**に限る
     （前日発売の開始後。それより前に走らせると開催日は見つかっても
       `find_r01_odds` が空振りする）

## 🔴 3つ目のバグ（2026-08-24に判明・修正済み）

2026-08-23 08:30 JST の自動実行で、前方探索は成功して開催日3会場を見つけたのに
`find_r01_odds` が **256件すべてパラメータエラー**で終わった。
同じ時刻(08:20 JST)の本番 refresh は同じレースのオッズを**100%取得**しており、
盤は存在していた。原因は CNAME の組み立てを**本番と別に書いていた**こと:

    本番 `_to_odds_base()` : pw151ouS3 0420260302
    プローブ（自前の置換） : pw151ous  010420260302   ← 別物

`ODDS_PREFIX = 'pw151ouS3'` で先頭の `pw01dde01`（末尾の01を含む）を置き換える
のが正しい。**同じ導出を2箇所に書いたための取り違え**で、このプロジェクトで
繰り返している型（2026-08-09③「対になっている処理は片方だけ直される」）。
本番の `_to_odds_base()` をそのまま使う形に直した。
"""
import re
import sys
import time
import unicodedata

from bs4 import BeautifulSoup

sys.path.insert(0, '.')
from scripts._session import create_session            # noqa: E402
from src.scraper.calendar import get_kaisai_on_date    # noqa: E402
from src.scraper.jra_scraper import (                  # noqa: E402
    JRA_BASE, HEADERS, find_r01_odds, calc_suffix, _to_odds_base,
)

SLEEP = 1.0
# 前方に何日ぶん探すか（今日を含む）。次の開催まで最大でも1週間なので8日あれば届く。
PROBE_DAYS_AHEAD = 8
# オッズ探索(256件)を何開催日ぶんまで試すか。無制限だとリクエストが膨らむ。
MAX_ODDS_TRIES = 2
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


def find_kaisai_forward(sess, today=None, days_ahead=PROBE_DAYS_AHEAD):
    """今日から**前方**へ開催日を探し、[(date_str, base), ...] を返す。

    ⚠ 過去へ遡ってはいけない。`get_kaisai_on_date` が読む出走表一覧は
    「今週これからの開催」しか載せないため、過去日は必ず空振りする。
    """
    import datetime as dt
    today = today or dt.date.today()
    found = []
    for ahead in range(days_ahead):
        d = (today + dt.timedelta(days=ahead)).strftime('%Y%m%d')
        try:
            links = get_kaisai_on_date(d, sess)
        except Exception as e:
            print(f'  {d}: 取得失敗 {e}')
            continue
        if links:
            # 戻り値は {base: 日付}。キーが base（2026-08-15に取り違えた）
            for base in links:
                found.append((d, base))
            print(f'📅 開催日 {d} / 会場 {len(links)}件')
        else:
            time.sleep(SLEEP)
    return found


def main():
    sess = create_session()

    cands = find_kaisai_forward(sess)
    if not cands:
        print(f'❌ 今日から{PROBE_DAYS_AHEAD}日先までに開催が見つからない。中止。')
        return

    # オッズ盤は発売中しか存在しないので、開催日ごとに順に試す
    base = date = r01 = None
    for date_, base_ in cands[:MAX_ODDS_TRIES]:
        odds_base_ = _to_odds_base(base_)
        print(f'\n🔎 {date_} / odds_base={odds_base_} でオッズR01を探す')
        r01_ = find_r01_odds(odds_base_, date_, sess)
        if r01_ is not None:
            base, date, r01 = base_, date_, r01_
            break
        print('   → 未発見（まだ発売前の可能性）')
    if r01 is None:
        print('\n❌ オッズR01のsuffixが見つからない。中止。')
        print('   オッズ盤は発売中しか存在しない。**金曜夜〜日曜**に実行し直すこと')
        print(f'   （試した開催日: {[d for d, _ in cands[:MAX_ODDS_TRIES]]}）')
        return

    odds_base = _to_odds_base(base)
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
