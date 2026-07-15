"""
shadow.py のユニットテスト（朝予想スナップショット参照・リーク防止）

2026-07-06 のリーク修正を検証する:
- calc_all の事後再実行ではなく race_predictions の朝予想から RL1-3 を引く
- 朝予想がないレースは記録しない
"""

import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import init_db, save_race_predictions
from src.betting.shadow import record_all_shadow_bets


@pytest.fixture
def tmp_base(tmp_path):
    """data/keiba.db 付きの一時プロジェクトルート。"""
    base = tmp_path
    (base / 'data').mkdir()
    init_db(str(base))
    return str(base)


def _save_morning(base_dir, race_id, date, horses):
    """朝予想スナップショットを保存する。"""
    race = {'id': race_id, 'date': date, 'racecourse': '東京', 'race_num': 1}
    save_race_predictions(race, horses, base_dir=base_dir)


def _make_result(race_id, date, finishers, tansho_payout=500):
    return {
        'id': race_id,
        'date': date,
        'racecourse': '東京',
        'race_num': 1,
        'race_class': '3勝クラス',
        'surface': '芝',
        'distance': 1600,
        'finishers': finishers,
        'dividends': {
            'tansho': {'payout': tansho_payout},
            'fukusho': [{'num': f['num'], 'payout': 150} for f in finishers[:3]],
            'umaren': {'payout': 1200},
            'wide': [],
            'sanrenpuku': {'payout': 3000},
        },
    }


def _fetch_shadow_rows(base_dir):
    conn = sqlite3.connect(os.path.join(base_dir, 'data', 'keiba.db'))
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM shadow_bets').fetchall()
    conn.close()
    return rows


def test_uses_morning_predictions_not_recompute(tmp_base):
    """RL1-3 は race_predictions の朝予想から取られる。"""
    horses = [
        {'horse_num': 5, 'name': '朝の本命', 'rl_rank': 1,
         'win_prob': 0.30, 'cal_prob': 0.45, 'win_odds': 4.0},
        {'horse_num': 2, 'name': '朝の対抗', 'rl_rank': 2,
         'win_prob': 0.20, 'cal_prob': 0.35, 'win_odds': 6.0},
        {'horse_num': 8, 'name': '朝の三番手', 'rl_rank': 3,
         'win_prob': 0.15, 'cal_prob': 0.30, 'win_odds': 8.0},
    ]
    _save_morning(tmp_base, 'R001', '2026-07-06', horses)

    finishers = [
        {'num': 2, 'name': '朝の対抗', 'place': 1, 'win_odds': 5.5},
        {'num': 5, 'name': '朝の本命', 'place': 2, 'win_odds': 3.8},
        {'num': 8, 'name': '朝の三番手', 'place': 3, 'win_odds': 9.0},
    ]
    record_all_shadow_bets([_make_result('R001', '2026-07-06', finishers)],
                           tmp_base)

    rows = _fetch_shadow_rows(tmp_base)
    assert len(rows) == 1
    row = rows[0]
    # RL1 は朝予想の #5（結果ページの再計算ではない）
    assert row['rl1_num'] == 5
    assert row['rl1_name'] == '朝の本命'
    assert row['rl2_num'] == 2
    assert row['rl3_num'] == 8
    # RL1(#5) は2着 → 単勝外れ・複勝的中
    assert row['shadow_tansho_hit'] == 0
    assert row['shadow_fukusho_hit'] == 1
    # RL1-2 = {5,2} = 実際の1-2着 → 馬連的中
    assert row['shadow_umaren_hit'] == 1
    # RL1-3 = {5,2,8} = 実際の1-2-3着 → 三連複的中
    assert row['shadow_sanrenp_hit'] == 1


def test_skips_race_without_morning_prediction(tmp_base):
    """朝予想がないレースは記録されない（リーク行を作らない）。"""
    finishers = [
        {'num': 1, 'name': 'A', 'place': 1, 'win_odds': 2.0},
        {'num': 2, 'name': 'B', 'place': 2, 'win_odds': 4.0},
        {'num': 3, 'name': 'C', 'place': 3, 'win_odds': 6.0},
    ]
    record_all_shadow_bets([_make_result('R_NO_MORNING', '2026-07-06', finishers)],
                           tmp_base)
    assert _fetch_shadow_rows(tmp_base) == []


def test_winner_pop_none_when_odds_missing(tmp_base):
    """結果ページのオッズが欠損している場合 winner_pop は None。"""
    horses = [
        {'horse_num': 1, 'name': 'A', 'rl_rank': 1,
         'win_prob': 0.30, 'cal_prob': 0.45, 'win_odds': 2.0},
        {'horse_num': 2, 'name': 'B', 'rl_rank': 2,
         'win_prob': 0.20, 'cal_prob': 0.35, 'win_odds': 4.0},
        {'horse_num': 3, 'name': 'C', 'rl_rank': 3,
         'win_prob': 0.10, 'cal_prob': 0.25, 'win_odds': 8.0},
    ]
    _save_morning(tmp_base, 'R002', '2026-07-06', horses)

    finishers = [
        {'num': 3, 'name': 'C', 'place': 1},   # win_odds なし
        {'num': 1, 'name': 'A', 'place': 2},
        {'num': 2, 'name': 'B', 'place': 3},
    ]
    record_all_shadow_bets([_make_result('R002', '2026-07-06', finishers)],
                           tmp_base)

    rows = _fetch_shadow_rows(tmp_base)
    assert len(rows) == 1
    assert rows[0]['winner_pop'] is None


def test_winner_pop_computed_when_odds_present(tmp_base):
    """オッズがあれば winner_pop は正しい人気順位になる。"""
    horses = [
        {'horse_num': 1, 'name': 'A', 'rl_rank': 1,
         'win_prob': 0.30, 'cal_prob': 0.45, 'win_odds': 2.0},
        {'horse_num': 2, 'name': 'B', 'rl_rank': 2,
         'win_prob': 0.20, 'cal_prob': 0.35, 'win_odds': 4.0},
        {'horse_num': 3, 'name': 'C', 'rl_rank': 3,
         'win_prob': 0.10, 'cal_prob': 0.25, 'win_odds': 8.0},
    ]
    _save_morning(tmp_base, 'R003', '2026-07-06', horses)

    # 勝ったのはオッズ8.0の3番人気 #3
    finishers = [
        {'num': 3, 'name': 'C', 'place': 1, 'win_odds': 8.0},
        {'num': 1, 'name': 'A', 'place': 2, 'win_odds': 2.0},
        {'num': 2, 'name': 'B', 'place': 3, 'win_odds': 4.0},
    ]
    record_all_shadow_bets([_make_result('R003', '2026-07-06', finishers)],
                           tmp_base)

    rows = _fetch_shadow_rows(tmp_base)
    assert rows[0]['winner_pop'] == 3


def test_empty_results_noop(tmp_base):
    """空リストでもクラッシュしない。"""
    record_all_shadow_bets([], tmp_base)
    assert _fetch_shadow_rows(tmp_base) == []


def test_rerun_does_not_duplicate_row(tmp_base):
    """同一レースを2回記録しても shadow_bets には1行しか残らない（idx_shadow_bets_uniq）。

    workflow_dispatch の再実行や --force 再生成で record_all_shadow_bets が
    同じレースに対して2回呼ばれても、rl_accuracy 等の集計が二重カウントされないことを保証する。
    """
    horses = [
        {'horse_num': 5, 'name': '朝の本命', 'rl_rank': 1,
         'win_prob': 0.30, 'cal_prob': 0.45, 'win_odds': 4.0},
        {'horse_num': 2, 'name': '朝の対抗', 'rl_rank': 2,
         'win_prob': 0.20, 'cal_prob': 0.35, 'win_odds': 6.0},
        {'horse_num': 8, 'name': '朝の三番手', 'rl_rank': 3,
         'win_prob': 0.15, 'cal_prob': 0.30, 'win_odds': 8.0},
    ]
    _save_morning(tmp_base, 'R_RERUN', '2026-07-06', horses)

    finishers = [
        {'num': 2, 'name': '朝の対抗', 'place': 1, 'win_odds': 5.5},
        {'num': 5, 'name': '朝の本命', 'place': 2, 'win_odds': 3.8},
        {'num': 8, 'name': '朝の三番手', 'place': 3, 'win_odds': 9.0},
    ]
    result = _make_result('R_RERUN', '2026-07-06', finishers)

    record_all_shadow_bets([result], tmp_base)
    record_all_shadow_bets([result], tmp_base)  # ワークフロー再実行を模擬

    rows = _fetch_shadow_rows(tmp_base)
    assert len(rows) == 1
