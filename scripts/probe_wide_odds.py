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

を投げて単勝・複勝を得ている。

🔴 **当初の仮説「この `Z` が券種の指定子」は 2026-09-06 の実行で否定された。**
A〜Z を総当たりした結果、**A〜Y は全部パラメータエラー、`Z` だけ**が
「単勝・複勝オッズ（馬番順）」を返した（見出しは
枠/馬番/馬名/単勝/複勝(2着払い)/性齢/馬体重/負担重量/騎手名/調教師名）。
ログに「ワイド」が1回出るのはナビゲーションメニューのリンクであって表の中身ではない。

そこで方針を変え、**`Z` で取れる単複ページの `<a>` を全部棚卸しして
券種リンクの CNAME を特定する**。JRADB は href ではなく onclick の
doAction('/JRADB/accessX.html','CNAME') に本体を埋めることが多いので両方を見る。
ワイド系のリンクが見つかったら実際に叩いて、組番が並ぶ盤かどうかを確認する。

⚠ 「券種リンクが無い」と「CNAME を拾えていない」は別物。
   リンク総数と CNAME 取得数を必ず併記し、空振りを『該当なし』と読み違えない
   （2026-08-16 の調教調査で実際に踏みかけた誤り）。

⚠ 到達できた場合の進め方は `dprime/CRITERIA.md` に**結果を見る前に**登録してある。
   Step 0(到達) → 1(盤の健全性) → 2(恒等式) → 3(ノイズとの区別) → 4(実配当)
   の順に通すこと。到達しただけで「使える」「利益が出る」とは書かない。
   ⚠ 恒等式は **8頭立て以上**でしか成立しない（5〜7頭は複勝が2着内、ワイドは3着内）。

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


BET_WORDS = ['ワイド', '馬連', '馬単', '枠連', '3連複', '3連単', '三連複', '三連単',
             '単勝', '複勝', 'オッズ']

# CNAME は "pw151ouS30620260402" のような英数字の並びに "/15" のような
# suffix が付く形。区切りの / を含めて丸ごと拾わないと POST できない。
_CNAME_RE = re.compile(r"['\"]([0-9A-Za-z]{8,}(?:/[0-9A-Za-z]+)?)['\"]")
_ENDPOINT_RE = re.compile(r"(access[0-9A-Za-z]+\.html)")


def extract_links(html):
    """ページ内の <a> を全部拾い、endpoint と CNAME を復元する。

    JRADB は href ではなく onclick の doAction('/JRADB/accessX.html','CNAME')
    に本体を埋めることが多いので、href と onclick の両方を見る。
    ⚠ 「探した結果ゼロ」と「探せていない」を混同しないため、
       CNAME が取れなかったリンクも件数として必ず残す（2026-08-16 の教訓）。
    """
    soup = BeautifulSoup(html, 'lxml')
    out = []
    for a in soup.find_all('a'):
        blob = ' '.join(filter(None, [a.get('href', ''), a.get('onclick', '')]))
        ep = _ENDPOINT_RE.search(blob)
        cn = _CNAME_RE.search(blob)
        out.append(dict(
            text=unicodedata.normalize('NFKC', a.get_text(' ', strip=True))[:40],
            href=(a.get('href') or '')[:60],
            endpoint=ep.group(1) if ep else '',
            cname=cn.group(1) if cn else '',
        ))
    return out


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
    print('■ 単複オッズページ（Z）を1回だけ取得し、ページ内のリンクを全列挙する')
    print('=' * 88)
    print('  ⚠ A〜Z の総当たりは 2026-09-06 の実行で決着済み。')
    print('     A〜Y は全部パラメータエラー、Z だけが「単勝・複勝オッズ（馬番順）」を返した。')
    print('     つまり CNAME の Z は券種の指定子ではない。券種リンクから入口を探す。')

    cn_z = f'{odds_base}{RACE_NUM:02d}{date}Z/{sx}'
    try:
        resp = _post(sess, 'accessO.html', cn_z)
    except Exception as e:
        print(f'  ❌ 通信例外 {type(e).__name__}: {e}')
        return
    if 'パラメータエラー' in resp.text:
        print('  ❌ Z がパラメータエラー。オッズ未発売の可能性。金曜夜〜日曜に再実行すること')
        return
    d0 = _describe(resp.text)
    print(f'  取得: {d0["title"][:50]}  table {d0["n_table"]}個')

    links = extract_links(resp.text)
    with_cn = [l for l in links if l['cname']]
    print(f'\n  リンク総数 {len(links)}  / うち CNAME が取れたもの {len(with_cn)}')
    print(f'  endpoint の内訳: {sorted({l["endpoint"] for l in links if l["endpoint"]})}')

    print('\n' + '-' * 88)
    print(f'{"リンク文字":<22}{"endpoint":<20}{"CNAME":<26}href')
    print('-' * 88)
    bet_links = []
    for l in links:
        hit = any(w in l['text'] for w in BET_WORDS)
        if hit or l['cname']:
            print(f'{l["text"][:21]:<22}{l["endpoint"][:19]:<20}{l["cname"][:25]:<26}{l["href"]}')
        if hit and l['cname']:
            bet_links.append(l)

    print('\n' + '=' * 88)
    print('■ 券種リンクの CNAME を、いま使っている単複の CNAME と並べて差分を見る')
    print('=' * 88)
    print(f'  いま使っている（単複）: {cn_z}')
    for l in bet_links[:12]:
        print(f'  {l["text"][:16]:<18} {l["endpoint"]:<18} {l["cname"]}')

    # ワイド系のリンクがあれば、実際に叩いて中身を確認する（最大3件）
    targets = [l for l in bet_links
               if any(w in l['text'] for w in ['ワイド', '馬連', '3連複', '三連複'])]
    if not targets:
        print('\n❌ ワイド/馬連/三連複のリンクが CNAME 付きで見つからなかった。')
        print('   ⚠ 「リンクが無い」のか「CNAME が JS 側で組み立てられていて拾えない」のかは別問題。')
        print(f'   上の「リンク総数 {len(links)} / CNAME 取得 {len(with_cn)}」を根拠に判断すること。')
        return

    print('\n' + '=' * 88)
    print('■ 券種リンクを実際に叩いて中身を確認する')
    print('=' * 88)
    for l in targets[:3]:
        time.sleep(SLEEP)
        ep = l['endpoint'] or 'accessO.html'
        try:
            r2 = _post(sess, ep, l['cname'])
        except Exception as e:
            print(f'  {l["text"][:16]}: 通信例外 {type(e).__name__}')
            continue
        if 'パラメータエラー' in r2.text:
            print(f'  {l["text"][:16]}: パラメータエラー')
            continue
        d2 = _describe(r2.text)
        kw = ' '.join(f'{k}{v}' for k, v in list(d2['kws'].items())[:5])
        print(f'\n--- {l["text"][:20]} ({ep}) ---')
        print(f'  タイトル: {d2["title"][:50]}')
        print(f'  table {d2["n_table"]}  組番 {d2["n_combo"]}  範囲 {d2["n_range"]}  '
              f'単値 {d2["n_single"]}  券種KW: {kw[:40]}')
        soup2 = BeautifulSoup(r2.text, 'lxml')
        for t in soup2.find_all('table')[:2]:
            for tr in t.find_all('tr')[:4]:
                cells = [unicodedata.normalize('NFKC', c.get_text(strip=True))
                         for c in tr.find_all(['td', 'th'])]
                if cells:
                    print(f'    {cells[:12]}')

    print('\n' + '=' * 88)
    print('■ 判定の目安')
    print('=' * 88)
    print('  ・組番(1-2, 1-3…)が多数出るページに到達 → ワイド盤が取れる。')
    print('    🔴 ただし到達しただけで「使える」と書かないこと。')
    print("    dprime/CRITERIA.md の Step 1→2→3→4 を順番に通す（事前登録済み）。")
    print('    Step 4（実配当で5基準）を通るまで「利益が出る」とは書かない。')
    print('  ・券種リンクは在るが CNAME が取れない → JS 側で組み立てている。別途調査')
    print('  ・券種リンク自体が無い → JRA公式の無料経路ではワイド盤に到達できない。D\' は打ち切り')


if __name__ == '__main__':
    main()
