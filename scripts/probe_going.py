"""出馬表ページに「馬場状態・天候」が載っているかを実データで確認する調査用。

背景（2026-08-27発見）:
  結果ページ parse_result_soup は track_condition / weather を取っているが、
  出馬表側 parse_header は date/racecourse/surface/distance/direction/class/
  start_time しか取っていない。そのため推論時は
      race.get('track_condition', '良')  →  常に「良」
  になり、f_track_cond は学習では実測値・推論では常に0.0 という
  学習/推論パリティ違反になっている。良でないレースは全体の27.2%。

このスクリプトは**読み取り専用**。DB・モデル・latest.json には一切書かない。
出馬表ページの生テキストから馬場・天候の表記を探し、
実際に何が載っているかをログに出すだけ。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._session import make_session          # noqa: E402
from src.scraper.jra_scraper import (              # noqa: E402
    get_kaisai_on_date, find_r01_shutuba, _try_fetch_shutuba,
)

TARGET = os.environ.get('TARGET_DATE', '').strip()

# 探す表記。結果ページ側は header に対して (良|稍重|重|不良) を単純検索している。
GOING = re.compile(r'(良|稍重|重|不良)')
WEATHER = re.compile(r'(晴|曇|小雨|雨|小雪|雪)')
LABELS = ['馬場状態', '馬場', '天候', '天気', 'コンディション', '含水率']


def main():
    if not TARGET:
        print('TARGET_DATE を指定してください（YYYYMMDD）')
        return
    sess = make_session()
    kaisai = get_kaisai_on_date(sess, TARGET)
    if not kaisai:
        print(f'❌ {TARGET} の開催情報が見つかりません')
        print('   出馬表一覧は「これから開催されるレース」しか載らない。')
        print('   金曜〜日曜に、これからの開催日を指定して実行すること。')
        return
    print(f'開催: {kaisai}')

    for base, venue in list(kaisai.items())[:2]:
        print(f'\n{"="*60}\n■ {venue}  base={base}')
        suffix = find_r01_shutuba(sess, base)
        if not suffix:
            print('  ❌ R01の出馬表が見つからない')
            continue
        soup = _try_fetch_shutuba(sess, base, 1, suffix)
        if soup is None:
            print('  ❌ 取得できない')
            continue
        text = soup.get_text(' ', strip=True)
        print(f'  本文 {len(text):,} 字')

        # ① ラベルの有無
        print('\n  --- ラベル検索 ---')
        for lab in LABELS:
            idx = text.find(lab)
            if idx >= 0:
                print(f'    ✅ 「{lab}」 → …{text[max(0,idx-30):idx+60]}…')
            else:
                print(f'    ❌ 「{lab}」 なし')

        # ② 馬場・天候の語そのもの
        #    ⚠ 「重」は「重賞」「斤量」等にも出るので前後を必ず出す
        print('\n  --- 馬場らしき語（前後20字つき。誤検出に注意） ---')
        hits = list(GOING.finditer(text))[:8]
        if not hits:
            print('    ❌ 良/稍重/重/不良 が1つも無い')
        for m in hits:
            s = max(0, m.start() - 20)
            print(f'    「{m.group(1)}」 … {text[s:m.end()+20]} …')

        print('\n  --- 天候らしき語 ---')
        wh = list(WEATHER.finditer(text))[:6]
        if not wh:
            print('    ❌ 晴/曇/雨/雪 が1つも無い')
        for m in wh:
            s = max(0, m.start() - 20)
            print(f'    「{m.group(1)}」 … {text[s:m.end()+20]} …')

        # ③ 冒頭を丸ごと出す（ヘッダに載っていれば目視で分かる）
        print(f'\n  --- 冒頭400字 ---\n    {text[:400]}')

    print('\n' + '=' * 60)
    print('判定の仕方:')
    print('  ラベル「馬場状態」があり、その直後に 良/稍重/重/不良 が続く → 取得可能')
    print('  「重」しか出ず、前後が「重賞」「斤量」等 → 出馬表には載っていない')


if __name__ == '__main__':
    main()
