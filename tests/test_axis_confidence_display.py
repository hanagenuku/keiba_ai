"""アプリの「AI xx%」信頼度バッジを出さないことの回帰テスト。

2026-08-20 に本番の実予想536レース × 実着順で測ったところ、このバッジは
軸(RL1)が実際に来るかどうかと**無関係**だった:

    conf 62%帯 → 軸の3着内 55.0%
    conf 96%帯 → 軸の3着内 53.4%
    AUC = 0.483（95%CI [0.434, 0.534]。0.5＝情報なし）

正体は `60 + (本命の人気順位 - 2)*4 + スコア差*20` という手製の式で、
**本命が人気薄なほど数字が上がる**（人気順位との相関 +0.382）。
つまり「AIの自信」ではなく「AIが市場と食い違っている度合い」を、
自信として見せていた。2026-08-07⑥の「ROI予測 ~150%」固定値、
2026-08-09①の単勝EV表示と同じ型（作り話をしない）。
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.app_json import to_app_json

JST = timezone(timedelta(hours=9))
ROOT = os.path.join(os.path.dirname(__file__), '..')


def _races(*horse_counts):
    return [
        {'racecourse': '新潟', 'race_num': i + 1, 'id': f'r{i + 1}',
         'race_name': 'テストR', 'distance': 1200, 'surface': 'ダート',
         'horses': [{} for _ in range(n)]}
        for i, n in enumerate(horse_counts)
    ]


def test_race_entries_have_no_conf_field():
    """latest.json のレースに conf を出さない。"""
    jst_now = datetime(2026, 8, 20, 20, 0, tzinfo=JST)
    with patch('src.betting.app_json.calc_all', return_value=[]):
        result = to_app_json([], _races(8, 10), None, jst_now, day_type='sunday')
    for lst in result['races'].values():
        for race in lst:
            assert 'conf' not in race, race


def test_app_json_does_not_compute_conf():
    """算出式そのものを残さない（消し忘れが復活する事故を防ぐ）。"""
    src = open(os.path.join(ROOT, 'src/betting/app_json.py'), encoding='utf-8').read()
    assert 'conf' not in src


def test_index_html_does_not_render_confidence_badge():
    """画面に「AI xx%」を出さない。★推奨のラベル自体は残す。"""
    html = open(os.path.join(ROOT, 'index.html'), encoding='utf-8').read()
    assert 'race.conf' not in html
    assert 'class="conf"' not in html
    assert 'rec-lbl' in html          # 推奨ラベルまで消していないこと
