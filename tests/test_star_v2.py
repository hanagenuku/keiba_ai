"""☆ v2（2026-11-20・hitrate/CRITERIA_STAR_V2.md）。実際に保存した成果物の形で確かめる。"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features import star


def _artifacts(tmp_path, tau1=0.5, tau2=0.05):
    d = tmp_path / 'data'
    d.mkdir()
    import xgboost as xgb
    cols = ['f_a', 'f_ws_kayochinkeiba_rank']
    rng = np.random.RandomState(0)
    X = rng.rand(80, 2)
    y = (X[:, 0] > 0.5).astype(int)
    bst = xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                    xgb.DMatrix(X, label=y, feature_names=cols), 5)
    bst.save_model(str(d / 'star_model.ubj'))
    ident = IsotonicRegression(out_of_bounds='clip').fit([0, 1], [0, 1])
    with open(d / 'star_calibrator.pkl', 'wb') as f:
        pickle.dump({'version': 2, 'iso1': ident, 'cm': ident, 'ck': ident}, f)
    (d / 'star_feature_cols.json').write_text(json.dumps(
        {'version': 2, 'feature_cols': cols, 'tau1': tau1, 'tau2': tau2, 'star_fill': 5.0}))
    star._DIR = None
    return str(tmp_path)


def test_web_columns_stay_nan_in_both_paths():
    recs = [{'f_a': None, 'f_ws_x': None}, {'f_a': 1.0, 'f_ws_x': 2.0}]
    a = star.to_matrix(recs, ['f_a', 'f_ws_x'])
    b = star.matrix_from_frame(pd.DataFrame(recs), ['f_a', 'f_ws_x'])
    np.testing.assert_array_equal(a, b)
    assert a[0, 0] == star.STAR_FILL and np.isnan(a[0, 1])


def test_s1_wins_when_over_tau1(tmp_path):
    base = _artifacts(tmp_path, tau1=0.5)
    assert star.pick_star_v2([1, 2, 3], [None, 0.7, 0.4], [None, 0.0, 0.2], base) == (2, 'S1', 0.7)


def test_s2_used_when_no_s1_over_tau1(tmp_path):
    base = _artifacts(tmp_path, tau1=0.9, tau2=0.05)
    num, kind, _ = star.pick_star_v2([1, 2, 3], [None, 0.7, 0.4], [None, 0.01, 0.2], base)
    assert (num, kind) == (3, 'S2')


def test_no_star_when_both_below(tmp_path):
    base = _artifacts(tmp_path, tau1=0.9, tau2=0.5)
    assert star.pick_star_v2([1, 2], [None, 0.7], [None, 0.2], base) == (None, None, None)


def test_s2_is_model_minus_market_and_none_for_favourite(tmp_path):
    base = _artifacts(tmp_path)
    out = star.star_s2([0.0, 1.0, None], [0.0, 0.0, 0.0], [1, 2, 3], base)
    assert out[0] is None and out[2] is None
    sig = 1 / (1 + np.exp(-1.0))
    assert abs(out[1] - (sig - 0.5)) < 1e-9


def test_v1_artifacts_keep_old_behaviour(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    (d / 'star_feature_cols.json').write_text(json.dumps({'feature_cols': ['f_a'], 'star_tau': 0.6}))
    star._DIR = None
    assert star.pick_star_v2([1, 2], [None, 0.61], [None, 0.9], str(tmp_path)) == (2, 'S1', 0.61)
    assert star.pick_star_v2([1, 2], [None, 0.59], [None, 0.9], str(tmp_path)) == (None, None, None)


def test_pop_range_limits_candidates(tmp_path):
    """STAR-NARROW: 成果物に pop_lo/pop_hi があれば 4〜9番人気からだけ選ぶ。人気不明は候補にしない。"""
    base = _artifacts(tmp_path, tau1=0.5, tau2=0.05)
    fp = tmp_path / 'data' / 'star_feature_cols.json'
    m = json.loads(fp.read_text())
    m.update(pop_lo=4, pop_hi=9)
    fp.write_text(json.dumps(m))
    star._DIR = None
    nums, s1, s2 = [1, 2, 3, 4], [None, 0.9, 0.6, 0.8], [None, 0.3, 0.1, 0.2]
    assert star.pick_star_v2(nums, s1, s2, base, pops=[1, 2, 5, None]) == (3, 'S1', 0.6)
    assert star.pick_star_v2(nums, s1, s2, base) == (None, None, None)
    assert star.star_note(m) == star.STAR_NOTE and '4〜9番人気' in star.STAR_NOTE
    assert star.star_note({'version': 2}) == star.STAR_NOTE_V2
