"""「AI単」列を単独モデル（市場なし）の順位にする変更が、表示だけを変えること。

hitrate/CRITERIA_SOLO.md。実際に save_model / load_model した Booster で確かめる（North Star #6）。
"""
import copy
import json
import pickle

import numpy as np
import pytest
import xgboost as xgb

COLS = ['f_post', 'f_age', 'f_weight_load', 'f_career_runs', 'f_ws_kayochinkeiba_rank']


def _booster(seed):
    rng = np.random.RandomState(seed)
    X = rng.rand(60, len(COLS)) * 10
    y = (X[:, 0] + rng.rand(60) * 5 > 7).astype(int)
    return xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                     xgb.DMatrix(X, label=y, feature_names=COLS), num_boost_round=4)


def _setup(tmp_path, solo_cols=None, with_solo=True):
    d = tmp_path / 'data'
    d.mkdir()
    for name in ['horse_dist_dict.pkl', 'horse_course_dict.pkl', 'horse_venue_dist_dict.pkl', 'post_zone_bias.pkl']:
        with open(d / name, 'wb') as f:
            pickle.dump({}, f)
    _booster(1).save_model(str(d / 'xgb_fukusho_model.pkl'))
    (d / 'xgb_feature_cols.json').write_text(json.dumps({'feature_cols': COLS, 'residual': True}))
    if with_solo:
        _booster(2).save_model(str(d / 'xgb_solo_model.pkl'))
        (d / 'xgb_solo_cols.json').write_text(json.dumps({'feature_cols': solo_cols or COLS}))
    from src.features import engine
    engine.init_engine(str(tmp_path))
    return engine


def _race(odds=(2.0, 8.0, 15.0, 30.0)):
    # 本番の parser が返す形（馬番は num / post_position。horse_num は無い）
    horses = [{'num': i + 1, 'post_position': i + 1, 'name': 'ABCD'[i], 'win_odds': o, 'age': 3 + i,
               'weight_load': 54.0 + i, 'running_style': '差し', 'history': []} for i, o in enumerate(odds)]
    return {'date': '20260726', 'id': 'x', 'racecourse': '新潟', 'distance': 1600, 'surface': '芝',
            'race_class': '3勝クラス', 'track_condition': '良', 'horses': horses}


def _by_num(out):
    return {h['num']: h for h in out}


def test_solo_margin_is_emitted_and_changes_nothing_else(tmp_path):
    eng = _setup(tmp_path, with_solo=True)
    assert eng._XGB_SOLO_MODEL is not None
    with_solo = _by_num(eng.calc_all(_race()))
    eng._XGB_SOLO_MODEL = None
    without = _by_num(eng.calc_all(_race()))
    for n, h in with_solo.items():
        assert h['solo_margin'] is not None and without[n]['solo_margin'] is None
        a = {k: v for k, v in h.items() if k != 'solo_margin'}
        b = {k: v for k, v in without[n].items() if k != 'solo_margin'}
        assert a == b, f'単独モデルの有無で solo_margin 以外が変わった: 馬番{n}'


def test_solo_margin_does_not_see_todays_market(tmp_path):
    eng = _setup(tmp_path)
    a = _by_num(eng.calc_all(_race(odds=(2.0, 8.0, 15.0, 30.0))))
    b = _by_num(eng.calc_all(_race(odds=(30.0, 15.0, 8.0, 2.0))))     # 人気を逆にする
    assert [a[n]['popularity'] for n in a] != [b[n]['popularity'] for n in b]
    for n in a:
        assert a[n]['solo_margin'] == pytest.approx(b[n]['solo_margin'], abs=1e-6)


def test_solo_model_is_ignored_when_its_columns_differ_from_production(tmp_path):
    eng = _setup(tmp_path, solo_cols=COLS[:-1])
    assert eng._XGB_SOLO_MODEL is None
    assert all(h['solo_margin'] is None and h['ability_margin'] is not None for h in eng.calc_all(_race()))


def test_no_solo_files_keeps_the_old_behaviour(tmp_path):
    eng = _setup(tmp_path, with_solo=False)
    assert eng._XGB_SOLO_MODEL is None


def test_solo_rank_uses_solo_margin_only_when_every_horse_has_it():
    from src.betting.app_json import _build_solo_ranks
    scored = [{'num': 1, 'ability_margin': 0.9, 'solo_margin': -1.0},
              {'num': 2, 'ability_margin': 0.1, 'solo_margin': 2.0},
              {'num': 3, 'ability_margin': 0.5, 'solo_margin': 0.0}]
    assert _build_solo_ranks(scored) == {2: 1, 3: 2, 1: 3}
    partial = copy.deepcopy(scored)
    partial[0]['solo_margin'] = None                    # 1頭欠けたらレース全体で従来の順位
    assert _build_solo_ranks(partial) == {1: 1, 3: 2, 2: 3}
    old = [{k: v for k, v in h.items() if k != 'solo_margin'} for h in scored]
    assert _build_solo_ranks(old) == {1: 1, 3: 2, 2: 3}
    assert _build_solo_ranks([{'num': 1, 'ability_margin': None}]) == {}


def test_real_solo_model_matches_the_production_columns():
    """リポジトリに単独モデルがあるなら、列は本番モデルと同じ（片方だけ作り直すと engine が黙って使わなくなる）。"""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = os.path.join(root, 'data', 'xgb_solo_cols.json')
    if not os.path.exists(sp):
        pytest.skip('単独モデルはまだ無い')
    with open(sp, encoding='utf-8') as f:
        solo = json.load(f)['feature_cols']
    with open(os.path.join(root, 'data', 'xgb_feature_cols.json'), encoding='utf-8') as f:
        prod = json.load(f)['feature_cols']
    assert solo == prod
    b = xgb.Booster()
    b.load_model(os.path.join(root, 'data', 'xgb_solo_model.pkl'))
    assert b.num_boosted_rounds() >= 1
