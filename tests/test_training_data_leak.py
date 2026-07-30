"""学習データに「当該レースの結果由来の値」が混入しないことの回帰テスト
（2026-07-28導入）。

build_training_data は対象レースの行を horse_history から読むため、
うっかりすると **レースが終わって初めて分かる値** を特徴量計算に
渡してしまう。実際に以下2件が混入していた:

  - race['first_3f']    : そのレースの実際の前半3F → f_early_speed
                          （推論時は出馬表に存在せず常に 36.0）
  - h['running_style']  : その馬がそのレースで実際に取った脚質
                          （推論時は過去走から _infer_running_style で推定）

いずれも「学習時だけ正解を見て、推論時は別の値」という
学習/推論パリティ違反であり、AUCでは検出できない。

North Starに従い、手打ちdictではなく実際に save_history_db() で
history.db に書き込み、build_training_data() を通して出力CSVを検証する。
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.db import save_history_db
from src.tools.build_training_data import build_training_data


def _race(rid, date, style, first_3f, corner_3=None):
    """1レース分の結果を本番と同じ形で作る。

    style / first_3f が「そのレースの実際の結果」に相当する。
    """
    return {
        'id': rid, 'race_id': rid, 'date': date,
        'racecourse': '東京', 'race_num': 1, 'race_name': 'T',
        'distance': 1600, 'surface': '芝', 'first_3f': first_3f,
        'track_condition': '良', 'race_class': '1勝クラス', 'weather': '晴',
        'finishers': [
            {'place': i, 'num': i, 'name': f'H{i}', 'popularity': i,
             'running_style': style, 'agari3f': 35.0,
             'corner_3': corner_3 if corner_3 is not None else i,
             'tansho_payout': 300 if i == 1 else 0,
             'fukusho_payout': 150 if i <= 3 else 0,
             'jockey': 'J', 'trainer': 'T',
             'weight_load': 56.0, 'sex': '牡', 'age': 4}
            for i in range(1, 9)
        ],
    }


def _build(base_dir, target_style, target_first_3f, hist_style='先行',
           hist_corner_3=2):
    """過去走5走 + 対象レース1走 を書いて特徴量CSVを生成する。

    hist_* が過去走（推論時にも見える情報）、
    target_* が対象レースの結果（推論時には存在しない情報）。
    """
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    races = [_race(f'2025010{d}_05_01', f'2025-01-0{d}',
                   hist_style, 35.0, hist_corner_3) for d in range(1, 6)]
    races.append(_race('20250601_05_01', '2025-06-01',
                       target_style, target_first_3f))
    save_history_db(races, db_path=os.path.join(base_dir, 'data', 'history.db'))
    build_training_data(base_dir)
    df = pd.read_csv(os.path.join(base_dir, 'data', 'horse_features.csv'))
    return df[df.race_id == '20250601_05_01'].reset_index(drop=True)


class TestTargetRaceResultDoesNotLeak:

    def test_f_early_speed_ignores_actual_race_pace(self, tmp_path):
        """f_early_speed に当該レースの実際の前半3Fが入らないこと。

        推論時は出馬表に first_3f が無く 36.0 になるため、
        学習時も 36.0 でなければ分布が食い違う。
        """
        df = _build(str(tmp_path / 'a'), target_style='追込',
                    target_first_3f=33.0)     # 極端に速い前半
        assert set(df.f_early_speed.unique()) == {36.0}, (
            f'当該レースの実測ペースが漏れている: {df.f_early_speed.unique()}'
        )

    def test_features_invariant_to_actual_race_pace(self, tmp_path):
        """実際の前半3Fを変えても特徴量が1つも変わらないこと。"""
        slow = _build(str(tmp_path / 'slow'), '先行', 39.0)
        fast = _build(str(tmp_path / 'fast'), '先行', 33.0)
        cols = [c for c in slow.columns if c.startswith(('f_', 'rl_', 'cl_'))]
        pd.testing.assert_frame_equal(slow[cols], fast[cols])

    def test_features_invariant_to_actual_running_style(self, tmp_path):
        """その馬がそのレースで実際にどう走ったかで特徴量が変わらないこと。

        過去走(先行)は同一で、当該レースの実測脚質だけを変える。
        """
        a = _build(str(tmp_path / 'nige'), '逃げ', 35.0, hist_style='先行')
        b = _build(str(tmp_path / 'oikomi'), '追込', 35.0, hist_style='先行')
        cols = [c for c in a.columns if c.startswith(('f_', 'rl_', 'cl_'))]
        pd.testing.assert_frame_equal(a[cols], b[cols])

    def test_style_still_reflects_past_races(self, tmp_path):
        """一方で、過去走の脚質は特徴量に反映されること。

        （上のテストが「脚質を単に無視している」だけで通ってしまう
        のを防ぐ。過去走が変われば特徴量は変わらねばならない。）
        """
        nige = _build(str(tmp_path / 'h_nige'), '先行', 35.0,
                      hist_style='逃げ', hist_corner_3=1)
        oikomi = _build(str(tmp_path / 'h_oikomi'), '先行', 35.0,
                        hist_style='追込', hist_corner_3=8)
        cols = [c for c in nige.columns if c.startswith(('f_', 'rl_', 'cl_'))]
        assert not nige[cols].equals(oikomi[cols]), (
            '過去走の脚質が特徴量に一切反映されていない'
        )
