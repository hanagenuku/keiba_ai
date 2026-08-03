#!/usr/bin/env python3
"""結果ページの列崩れを実データで診断する読み取り専用スクリプト。

## なぜ要るか

2026-08-03の調査で、history.db の popularity/body_weight/trainer が
2026-06-28までは正常に埋まっていたのに2026-07-04以降は揃って0%充足に
転落していたと判明した（win_oddsは元々0%充足で無関係・既知の別問題）。
jra_scraper.py はこの間コミットされておらず、texts[0..10]相当
（着順〜上がり3F、jockey含む）は今も正常なため、実際のJRADB結果ページの
列構成が texts[11]（単勝オッズ）以降のどこかで変わった可能性が高い。

この環境（Claude Code on the web）からは www.jra.go.jp への到達がブロック
されているため実HTMLを確認できない。本スクリプトは GitHub Actions
（ネットワーク到達可）から実行し、実際に結果ページを取得・パースすることで、
parse_result_soup() 内の診断ログ diag_result_row_columns()
（jra_scraper.py, 2026-08-03追加）を発火させ、実際の texts 配列を
ログに残す。

## 使い方（probe-result-columns.yml から実行。ローカルからはJRAに届かない）

    TARGET_DATE=20260802 PLACES=04 python scripts/probe_result_columns.py

⚠ 読み取り専用。DB・モデル・latest.json には一切書き込まない。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._session import create_session
from src.scraper.calendar import get_base_from_calendar
from src.scraper.jra_scraper import (
    find_r01_result, _try_fetch_result, parse_result_soup, calc_suffix,
)

PLACE_NAMES = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
               '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}


def main():
    date = (os.environ.get('TARGET_DATE') or '').strip()
    if not (len(date) == 8 and date.isdigit()):
        raise SystemExit('TARGET_DATE=YYYYMMDD を指定してください')
    places = [p.strip() for p in
              (os.environ.get('PLACES') or '').split(',') if p.strip()]
    if not places:
        raise SystemExit('PLACES を指定してください（例: 04）')

    sess = create_session()

    print(f'🔎 対象日: {date}  対象場: {places}')
    print('─' * 62)

    for pc in places:
        name = PLACE_NAMES.get(pc, pc)
        cal_base = get_base_from_calendar(pc, date)
        if not cal_base:
            print(f'\n{name}({pc}): カレンダーにbaseが無い（非開催日の可能性）')
            continue
        base = 'pw01sde10' + cal_base[len('pw01dde01'):]
        print(f'\n── {name}({pc})  base={base} ──')

        sx = find_r01_result(base, date, sess)
        if not sx:
            print('  R01 未発見のためスキップ')
            continue
        r01 = int(sx, 16)
        print(f'  R01 suffix = {sx}')

        # 診断には数レースあれば十分（列構成はレースごとに変わらない前提）
        for r in range(1, 3):
            s = sx if r == 1 else calc_suffix(r01, r)
            soup = _try_fetch_result(sess, base, r, date, s)
            if soup is None:
                print(f'  R{r:02d}: 取得できず')
                continue
            result = parse_result_soup(soup, name, r, date, pc)
            if result is None:
                print(f'  R{r:02d}: parse失敗（障害レース等）')
                continue
            finishers = result['finishers']
            print(f'  R{r:02d}: {len(finishers)}頭取得')
            if finishers:
                h0 = finishers[0]
                print(f'    先頭馬: trainer={h0["trainer"]!r} '
                      f'popularity={h0["popularity"]!r} '
                      f'body_weight={h0["body_weight"]!r} '
                      f'win_odds={h0["win_odds"]!r}')
                # ↑ ここが壊れていれば diag_result_row_columns() が
                #    上の行で実際のtexts配列を自動的にログへ出しているはず

    print('\n' + '─' * 62)
    print('※ 読み取り専用。DB・モデル・latest.json には一切書き込んでいない。')


if __name__ == '__main__':
    main()
