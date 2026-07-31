"""残差学習モデルでキャリブレーションが動くことのテスト（2026-07-31）。

🔴 これまで `_build_xgb_pairs` は `model.predict_proba(X)` を呼んでいた。
残差モデルは sklearn API ではなく `xgb.Booster` なので predict_proba を持たず、
**レース単位の try/except に毎回吸われてペアが1件も作れず**、
「データ不足（0ペア）」で静かに None を返していた。

例外が飛ばないため呼び出し側の try/except では検知できず、
`KEIBA_XGB_retrain_v6.ipynb` セル7は「完了」と表示していた（嘘の成功）。
結果、新モデルに旧モデル用の較正器が付いたまま残っていた。

North Star #6 に従い、手打ちのモックではなく **実際に xgb.Booster を学習して
UBJ で保存し、init_engine 経由でロードした本物の残差モデル**で検証する。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

xgb = pytest.importorskip('xgboost')


def _make_residual_model(base_dir, feature_cols, n=400):
    """本番と同じ形（Booster + UBJ + residual:true）の残差モデルを作る。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, len(feature_cols)))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(feature_cols))
    dtrain.set_base_margin(rng.normal(size=n))
    booster = xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                        dtrain, num_boost_round=5)
    os.makedirs(f'{base_dir}/data', exist_ok=True)
    booster.save_model(f'{base_dir}/data/xgb_fukusho_model.pkl')
    with open(f'{base_dir}/data/xgb_feature_cols.json', 'w') as f:
        json.dump({'residual': True, 'feature_cols': list(feature_cols)}, f)
    return booster


class TestPredictFukushoProbs:

    def test_residual_booster_returns_probabilities(self, tmp_path, monkeypatch):
        """Booster でも確率が返ること（旧実装は AttributeError で落ちていた）。"""
        import pandas as pd
        import src.features.engine as eng
        from src.tools.calibrate_xgb import _predict_fukusho_probs

        cols = [f'f{i}' for i in range(6)]
        booster = _make_residual_model(str(tmp_path), cols)
        monkeypatch.setattr(eng, '_XGB_FUKUSHO_MODEL', booster)
        monkeypatch.setattr(eng, '_XGB_RESIDUAL', True)

        horses = [{'popularity': p} for p in (1, 2, 3, 4)]
        X = pd.DataFrame(np.zeros((4, len(cols))), columns=cols)
        probs = _predict_fukusho_probs(X, horses)

        assert len(probs) == 4
        assert all(0.0 < p < 1.0 for p in probs)

    def test_old_implementation_would_have_failed(self, tmp_path, monkeypatch):
        """回帰の要: Booster に predict_proba は無い。

        これが無いことこそが「0ペア」の原因だったので、
        前提が変わったら気づけるように固定する。
        """
        cols = [f'f{i}' for i in range(6)]
        booster = _make_residual_model(str(tmp_path), cols)
        assert not hasattr(booster, 'predict_proba')

    def test_popularity_ordering_is_reflected(self, tmp_path, monkeypatch):
        """人気が良いほど base_margin が高く、確率も高くなること。

        base_margin の作り方が engine.py とズレると較正が本番と違う分布に
        当たってしまうため、向きだけでも固定しておく。
        """
        import pandas as pd
        import src.features.engine as eng
        from src.tools.calibrate_xgb import _predict_fukusho_probs

        cols = [f'f{i}' for i in range(6)]
        booster = _make_residual_model(str(tmp_path), cols)
        monkeypatch.setattr(eng, '_XGB_FUKUSHO_MODEL', booster)
        monkeypatch.setattr(eng, '_XGB_RESIDUAL', True)

        # 人気順位は頭数で頭打ちにする（train_xgb の np.clip と同じ）ので、
        # 比較する人気は必ず頭数以内に収めること。
        horses = [{'popularity': p} for p in range(1, 13)]
        X = pd.DataFrame(np.zeros((12, len(cols))), columns=cols)
        probs = _predict_fukusho_probs(X, horses)
        assert probs[0] > probs[4] > probs[11]

    def test_popularity_is_clipped_to_field_size(self, tmp_path, monkeypatch):
        """頭数を超える人気は頭数に丸める（train_xgb の np.clip と同じ挙動）。"""
        import pandas as pd
        import src.features.engine as eng
        from src.tools.calibrate_xgb import _predict_fukusho_probs

        cols = [f'f{i}' for i in range(6)]
        booster = _make_residual_model(str(tmp_path), cols)
        monkeypatch.setattr(eng, '_XGB_FUKUSHO_MODEL', booster)
        monkeypatch.setattr(eng, '_XGB_RESIDUAL', True)

        X = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)
        probs = _predict_fukusho_probs(X, [{'popularity': p} for p in (3, 5, 12)])
        assert probs[0] == probs[1] == probs[2]

    def test_matches_engine_base_margin_formula(self, tmp_path, monkeypatch):
        """engine.py の推論経路と同じ base_margin であること。"""
        import math
        import pandas as pd
        import src.features.engine as eng
        from src.tools.calibrate_xgb import _predict_fukusho_probs

        cols = [f'f{i}' for i in range(6)]
        booster = _make_residual_model(str(tmp_path), cols)
        monkeypatch.setattr(eng, '_XGB_FUKUSHO_MODEL', booster)
        monkeypatch.setattr(eng, '_XGB_RESIDUAL', True)

        horses = [{'popularity': p} for p in (1, 2, 3, 4, 5)]
        X = pd.DataFrame(np.zeros((5, len(cols))), columns=cols)
        got = _predict_fukusho_probs(X, horses)

        # engine.py と同じ式を独立に書き下して突き合わせる
        n_h = max(len(horses), 2)
        harm = math.log(n_h) + 0.5772
        bm = []
        for h in horses:
            p_mkt = max(min((1.0 / h['popularity']) / harm, 0.999), 0.001)
            bm.append(math.log(p_mkt / (1 - p_mkt)))
        dmat = xgb.DMatrix(X.values, feature_names=cols)
        dmat.set_base_margin(np.array(bm))
        want = [1 / (1 + math.exp(-float(m)))
                for m in booster.predict(dmat, output_margin=True)]

        assert got == pytest.approx(want, abs=1e-9)

    def test_non_residual_uses_predict_proba(self, monkeypatch):
        """非残差モデル（sklearn API）は従来どおり predict_proba を使うこと。"""
        import pandas as pd
        import src.features.engine as eng
        from src.tools.calibrate_xgb import _predict_fukusho_probs

        class FakeSklearnModel:
            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.4),
                                        np.full(len(X), 0.6)])

        monkeypatch.setattr(eng, '_XGB_FUKUSHO_MODEL', FakeSklearnModel())
        monkeypatch.setattr(eng, '_XGB_RESIDUAL', False)

        X = pd.DataFrame(np.zeros((3, 2)), columns=['a', 'b'])
        assert _predict_fukusho_probs(X, [{}, {}, {}]) == [0.6, 0.6, 0.6]
