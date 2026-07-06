"""市場特徴量（f_popularity / f_pop_last / f_pop_avg / f_beat_market_rate）のテスト

2026-07-06 追加: 市場（人気）情報をXGB特徴量に取り込む。
history.db の win_odds は0%欠損のため popularity（99.2%充足）を使う。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import calc_features_for_xgb


def _race(horses=None):
    return {
        'date': '2026-07-06',
        'racecourse': '東京',
        'distance': 1600,
        'surface': '芝',
        'race_class': '3勝クラス',
        'track_condition': '良',
        'horses': horses or [],
    }


def _hist_run(place, popularity, **kw):
    base = {
        'place': place,
        'popularity': popularity,
        'distance': 1600,
        'surface': '芝',
        'racecourse': '東京',
        'race_class': '3勝クラス',
        'class': '3勝クラス',
        'track_condition': '良',
        'condition': '良',
        'date': '2026-06-01',
        'last_3f': 34.5,
        'agari3f': 34.5,
        'finishers': 12,
        'num_finishers': 12,
        'margin': 0.3,
        'corner_3': 5,
        'running_style': '差し',
    }
    base.update(kw)
    return base


def test_f_popularity_from_current_race():
    """現走の popularity が f_popularity になる。"""
    h = {'name': 'テスト馬', 'popularity': 3, 'history': []}
    feats = calc_features_for_xgb(h, _race())
    assert feats['f_popularity'] == 3.0


def test_f_popularity_nan_when_unknown():
    """popularity 未設定(0/99)は NaN（→学習時に fillna される）。"""
    for pop in [None, 0, 99]:
        h = {'name': 'テスト馬', 'history': []}
        if pop is not None:
            h['popularity'] = pop
        feats = calc_features_for_xgb(h, _race())
        assert math.isnan(feats['f_popularity']), f'pop={pop} で NaN でない'


def test_f_pop_last_and_avg():
    """直近5走の人気から f_pop_last / f_pop_avg を計算する。"""
    h = {'name': 'テスト馬', 'popularity': 2,
         'history': [
             _hist_run(place=1, popularity=4),   # 直近
             _hist_run(place=5, popularity=2),
             _hist_run(place=3, popularity=6),
         ]}
    feats = calc_features_for_xgb(h, _race())
    assert feats['f_pop_last'] == 4.0
    assert feats['f_pop_avg'] == (4 + 2 + 6) / 3


def test_f_beat_market_rate():
    """着順 < 人気 だった走の割合。"""
    h = {'name': 'テスト馬', 'popularity': 2,
         'history': [
             _hist_run(place=1, popularity=4),   # 市場より上 (1<4)
             _hist_run(place=5, popularity=2),   # 市場より下 (5>2)
             _hist_run(place=3, popularity=6),   # 市場より上 (3<6)
             _hist_run(place=2, popularity=2),   # 同等 (2==2 → 上回りではない)
         ]}
    feats = calc_features_for_xgb(h, _race())
    assert feats['f_beat_market_rate'] == 2 / 4


def test_market_features_nan_without_history():
    """過去走がなければ市場履歴系は NaN。"""
    h = {'name': '新馬', 'popularity': 1, 'history': []}
    feats = calc_features_for_xgb(h, _race())
    assert math.isnan(feats['f_pop_last'])
    assert math.isnan(feats['f_pop_avg'])
    assert math.isnan(feats['f_beat_market_rate'])


def test_history_popularity_99_excluded():
    """popularity=99（未取得）の過去走は集計から除外される。"""
    h = {'name': 'テスト馬', 'popularity': 2,
         'history': [
             _hist_run(place=1, popularity=99),
             _hist_run(place=2, popularity=3),
         ]}
    feats = calc_features_for_xgb(h, _race())
    assert feats['f_pop_last'] == 3.0
    assert feats['f_pop_avg'] == 3.0


def _init_engine_globals():
    """init_engine を通さず calc_all を動かすための最小グローバル設定。"""
    import src.features.engine as eng
    for attr in ['_horse_dist_dict', '_horse_course_dict',
                 '_horse_venue_dist_dict', '_post_zone_bias',
                 '_jockey_dict', '_trainer_dict']:
        if not hasattr(eng, attr) or getattr(eng, attr) is None:
            setattr(eng, attr, {})


def test_calc_all_derives_popularity_before_features():
    """calc_all が win_odds から popularity を導出し、出力にも引き継ぐ。"""
    _init_engine_globals()
    from src.features.engine import calc_all
    horses = [
        {'num': 1, 'horse_num': 1, 'name': 'A', 'win_odds': 8.0,
         'running_style': '差し', 'history': []},
        {'num': 2, 'horse_num': 2, 'name': 'B', 'win_odds': 2.0,
         'running_style': '先行', 'history': []},
        {'num': 3, 'horse_num': 3, 'name': 'C', 'win_odds': 15.0,
         'running_style': '逃げ', 'history': []},
    ]
    race = _race(horses)
    out = calc_all(race)
    pop_by_name = {h['name']: h.get('popularity') for h in out}
    assert pop_by_name['B'] == 1   # 最低オッズ = 1番人気
    assert pop_by_name['A'] == 2
    assert pop_by_name['C'] == 3


def test_calc_all_keeps_existing_popularity():
    """結果ページ由来の確定人気（既存値）は上書きされない。"""
    _init_engine_globals()
    from src.features.engine import calc_all
    horses = [
        {'num': 1, 'horse_num': 1, 'name': 'A', 'win_odds': 8.0,
         'popularity': 1,   # 確定人気が既に入っている
         'running_style': '差し', 'history': []},
        {'num': 2, 'horse_num': 2, 'name': 'B', 'win_odds': 2.0,
         'popularity': 2,
         'running_style': '先行', 'history': []},
    ]
    race = _race(horses)
    out = calc_all(race)
    pop_by_name = {h['name']: h.get('popularity') for h in out}
    assert pop_by_name['A'] == 1
    assert pop_by_name['B'] == 2
