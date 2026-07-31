"""base_margin の学習/推論パリティ（人気ドリフト注入）のテスト（2026-07-28導入）。

学習データの popularity は結果ページ由来の**確定人気**だが、本番の推論時に
渡るのは予想生成時点（朝〜前夜）の薄いオッズ由来の人気。実測すると
平均1.51位ずれ、順位が変わらない馬は34.5%しかない。確定人気のまま学習すると
本番でだけ base_margin が劣化して届くため、実測ドリフトを注入して
学習側の情報量を推論側に揃える。

North Starに従い、手打ちのdictではなく実際に sqlite3 の keiba.db を作り
race_predictions / odds_snapshots から分布を組み立てて検証する。
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.train_xgb import load_popularity_drift, _apply_popularity_drift


def _make_db(path, n_races, n_horses=10, shuffle=True):
    """本番と同じ2テーブルを持つ keiba.db を作る。

    odds_snapshots が「直前（確定に近い）」、race_predictions が「朝」。
    shuffle=True のとき朝と直前で順位がずれる（＝ドリフトがある）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE race_predictions
                 (race_id TEXT, horse_num INT, tansho_odds REAL)""")
    c.execute("""CREATE TABLE odds_snapshots
                 (race_id TEXT, horse_num INT, tansho REAL, captured_at TEXT)""")
    rng = np.random.default_rng(0)
    for r in range(n_races):
        rid = f'R{r:04d}'
        late = np.arange(1, n_horses + 1) * 2.0            # 直前オッズ
        morn = late.copy()
        if shuffle:
            morn = morn + rng.normal(0, 3.0, n_horses)     # 朝はズレている
        for i in range(n_horses):
            c.execute('INSERT INTO race_predictions VALUES (?,?,?)',
                      (rid, i + 1, max(1.0, float(morn[i]))))
            c.execute('INSERT INTO odds_snapshots VALUES (?,?,?,?)',
                      (rid, i + 1, float(late[i]), '2026-07-01T09:00'))
    c.commit(); c.close()


class TestLoadPopularityDrift:

    def test_returns_none_without_db(self, tmp_path):
        assert load_popularity_drift(str(tmp_path)) is None

    def test_returns_none_when_sample_too_small(self, tmp_path):
        """標本が薄いうちは注入しない（誤ったノイズを学習に入れないため）。"""
        _make_db(str(tmp_path / 'data' / 'keiba.db'), n_races=5)
        assert load_popularity_drift(str(tmp_path)) is None

    def test_builds_distribution_from_real_tables(self, tmp_path):
        _make_db(str(tmp_path / 'data' / 'keiba.db'), n_races=120)
        d = load_popularity_drift(str(tmp_path))
        assert d is not None
        assert '_all' in d
        assert len(d['_all']) >= 500
        # ドリフトが実際に発生している（全部0なら注入する意味がない）
        assert np.abs(d['_all']).mean() > 0

    def test_no_drift_when_morning_equals_final(self, tmp_path):
        """朝と直前が同一なら、ズレの標本はすべて0になる。"""
        _make_db(str(tmp_path / 'data' / 'keiba.db'), n_races=120, shuffle=False)
        d = load_popularity_drift(str(tmp_path))
        assert d is not None
        assert np.abs(d['_all']).max() == 0


class TestApplyPopularityDrift:

    @pytest.fixture
    def drift(self, tmp_path):
        _make_db(str(tmp_path / 'data' / 'keiba.db'), n_races=120)
        return load_popularity_drift(str(tmp_path))

    def test_output_is_valid_permutation_per_race(self, drift):
        """レースごとに 1..N の順列でなければ base_margin が壊れる。"""
        pop = pd.Series([1, 2, 3, 4, 5, 6, 7, 8] * 3)
        rid = ['A'] * 8 + ['B'] * 8 + ['C'] * 8
        out = _apply_popularity_drift(pop, rid, drift, seed=1)
        for r in ('A', 'B', 'C'):
            v = sorted(out[[x == r for x in rid]].astype(int))
            assert v == list(range(1, 9)), f'{r} が順列でない: {v}'

    def test_actually_changes_order(self, drift):
        """ドリフトがある分布なら順位は変わる（無変化なら注入の意味がない）。"""
        pop = pd.Series(list(range(1, 13)))
        rid = ['A'] * 12
        out = _apply_popularity_drift(pop, rid, drift, seed=3)
        assert not np.array_equal(out.astype(int), pop.values)

    def test_deterministic_for_same_seed(self, drift):
        pop = pd.Series(list(range(1, 13)))
        rid = ['A'] * 12
        a = _apply_popularity_drift(pop, rid, drift, seed=7)
        b = _apply_popularity_drift(pop, rid, drift, seed=7)
        np.testing.assert_array_equal(a, b)


class TestLoadXgbModelAny:
    """旧モデル比較が残差モデルで一度も動いていなかった件（2026-07-31）。

    残差モデルは `Booster.save_model()` の UBJ 形式で保存されるのに、
    比較側が `pickle.load()` していたため必ず
    `invalid load key, '{'` で失敗し、「旧モデル評価スキップ」と出ていた。
    """

    def _residual_model(self, path):
        import numpy as np
        import xgboost as xgb
        X = np.random.default_rng(0).normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)
        b = xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                      xgb.DMatrix(X, label=y), 3)
        b.save_model(str(path))
        return b

    def test_pickle_cannot_read_residual_model(self, tmp_path):
        """前提の固定: 残差モデルは pickle では読めない。"""
        import pickle
        p = tmp_path / 'res.pkl'
        self._residual_model(p)
        with pytest.raises(Exception):
            pickle.load(open(p, 'rb'))

    def test_loads_residual_model(self, tmp_path):
        import xgboost as xgb
        from src.tools.train_xgb import _load_xgb_model_any
        p = tmp_path / 'res.pkl'
        self._residual_model(p)
        assert isinstance(_load_xgb_model_any(str(p), True), xgb.Booster)

    def test_loads_residual_model_even_if_flag_wrong(self, tmp_path):
        """residual フラグが欠けている古いメタでも読めること。"""
        import xgboost as xgb
        from src.tools.train_xgb import _load_xgb_model_any
        p = tmp_path / 'res.pkl'
        self._residual_model(p)
        assert isinstance(_load_xgb_model_any(str(p), False), xgb.Booster)

    def test_loads_pickled_sklearn_model(self, tmp_path):
        import pickle
        import numpy as np
        import xgboost as xgb
        from src.tools.train_xgb import _load_xgb_model_any
        X = np.random.default_rng(0).normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)
        m = xgb.XGBClassifier(n_estimators=3, max_depth=2).fit(X, y)
        p = tmp_path / 'norm.pkl'
        pickle.dump(m, open(p, 'wb'))
        assert isinstance(_load_xgb_model_any(str(p), False), xgb.XGBClassifier)

    def test_raises_for_garbage_file(self, tmp_path):
        from src.tools.train_xgb import _load_xgb_model_any
        p = tmp_path / 'junk.pkl'
        p.write_bytes(b'not a model')
        with pytest.raises(Exception):
            _load_xgb_model_any(str(p), True)
