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
（ネットワーク到達可）から実行し、本番と全く同じ経路（fetch_results()、
結果一覧pw01sli00/AF由来のbase解決）で実際に結果ページを取得・パースする
ことで、parse_result_soup() 内の診断ログ diag_result_row_columns()
（jra_scraper.py, 2026-08-03追加）を発火させ、実際の texts 配列をログに残す。

⚠ 静的カレンダー(KAISAI_CALENDAR)経由のbase解決は2026-03-28以降83日分
しかカバーしておらず直近日を解決できないため使わない（実際に失敗を確認済み）。
fetch_results() が使う「結果一覧ページからの逆算」は本番の日曜結果取得と
完全に同じ経路で、対象日が過去であれば静的カレンダーの範囲外でも解決できる。

## 使い方（probe-result-columns.yml から実行。ローカルからはJRAに届かない）

    TARGET_DATE=20260802 python scripts/probe_result_columns.py

⚠ 読み取り専用。fetch_results() は hist_db_path を渡さない限り DB・モデル・
latest.json に一切書き込まない（結果は戻り値としてメモリ上に返るのみ）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._session import create_session
from src.scraper.jra_scraper import fetch_results


def main():
    date = (os.environ.get('TARGET_DATE') or '').strip()
    if not (len(date) == 8 and date.isdigit()):
        raise SystemExit('TARGET_DATE=YYYYMMDD を指定してください')

    sess = create_session()

    print(f'🔎 対象日: {date}')
    print('─' * 62)

    # fetch_results() は本番(sunday_results.py/weekend.py)と全く同じ経路。
    # hist_db_path を渡さないため DB には一切書き込まない（戻り値のみ）。
    results = fetch_results(sess, date)

    print('\n' + '─' * 62)
    print(f'取得レース数: {len(results)}')
    for r in results[:3]:
        finishers = r.get('finishers', [])
        if not finishers:
            continue
        h0 = finishers[0]
        print(f"  {r.get('racecourse', '?')} {r.get('race_name', '')}: "
              f"trainer={h0.get('trainer')!r} popularity={h0.get('popularity')!r} "
              f"body_weight={h0.get('body_weight')!r} win_odds={h0.get('win_odds')!r}")
        # ↑ ここが壊れていれば diag_result_row_columns() が
        #    parse_result_soup() 内で実際のtexts配列を既にログへ出しているはず

    print('\n※ 読み取り専用。DB・モデル・latest.json には一切書き込んでいない。')


if __name__ == '__main__':
    main()
