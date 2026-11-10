"""本番の残差モデルを「何本の木で」予測するかを固定するテスト。

🔴 なぜ要るか（実測は `hitrate/RESULTS_B.md`）:
  本番の残差モデルは early stopping が **切片**で 10本目に止まる（信号の飽和ではない）。
  木は 61本あり、`Booster.predict()` の既定は **全木**。止まったあとの50本が
  順位の信号を担っていて、実測で

      全木 61本           → プール AUC 0.7879（3窓9,735行・3シード平均）
      best_iteration まで → プール AUC 0.7799

  と **0.008** 違う。`requirements.txt` は `xgboost>=1.7` で版を固定していないので、
  将来の版で `predict()` の既定が best_iteration までに変わると、
  **本番の精度がこの幅で静かに落ちる**。

North Star #6 に従い、合成モデルではなく**実際に保存されている本番の UBJ ファイル**で検証する。
"""
import json
import os

import numpy as np
import pytest

xgb = pytest.importorskip('xgboost')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'data', 'xgb_fukusho_model_residual.pkl')
COLS = os.path.join(ROOT, 'data', 'xgb_feature_cols_residual.json')


@pytest.fixture(scope='module')
def booster():
    if not os.path.exists(MODEL):
        pytest.skip('本番の残差モデルが無い環境')
    b = xgb.Booster()
    b.load_model(MODEL)          # UBJ 形式（pickle では読めない）
    return b


@pytest.fixture(scope='module')
def dmat():
    cols = json.load(open(COLS, encoding='utf-8'))['feature_cols']
    rng = np.random.default_rng(0)
    d = xgb.DMatrix(rng.normal(5, 2, (64, len(cols))), feature_names=cols)
    d.set_base_margin(np.zeros(64))
    return d


def test_early_stopping_left_trees_after_best_iteration(booster):
    """「止まったあとにも木がある」状況が実在することを固定する。"""
    best = int(booster.attributes()['best_iteration'])
    total = booster.num_boosted_rounds()
    assert best < total - 1, (
        f'best_iteration={best} / 木 {total}本。'
        'この前提が崩れたら本テストの意味を見直すこと')


def test_predict_default_uses_all_trees(booster, dmat):
    """`predict()` の既定が全木であること。

    将来の xgboost で既定が best_iteration までに変わったらここで落ちる。
    落ちたときは「テストを緩める」のではなく、engine.py が明示している
    `iteration_range=(0, num_boosted_rounds())` のままで良いかを**実測で**確かめる。
    """
    default = booster.predict(dmat, output_margin=True)
    all_trees = booster.predict(
        dmat, output_margin=True,
        iteration_range=(0, booster.num_boosted_rounds()))
    assert np.allclose(default, all_trees, atol=0, rtol=0)


def test_best_iteration_only_is_materially_different(booster, dmat):
    """全木と best_iteration までが**実質的に違う**ことを記録として固定する。"""
    best = int(booster.attributes()['best_iteration'])
    all_trees = booster.predict(
        dmat, output_margin=True,
        iteration_range=(0, booster.num_boosted_rounds()))
    only_best = booster.predict(
        dmat, output_margin=True, iteration_range=(0, best + 1))
    assert np.abs(all_trees - only_best).max() > 0.5


def test_engine_passes_iteration_range_explicitly():
    """engine.py が木の本数を版の既定に任せていないこと（ソースレベル）。"""
    src = open(os.path.join(ROOT, 'src', 'features', 'engine.py'),
               encoding='utf-8').read()
    i = src.index('raw_margin = float(_XGB_FUKUSHO_MODEL.predict(')
    block = src[i:i + 200]
    assert 'iteration_range=(0, _ntree)' in block, \
        'engine.py の残差推論から iteration_range が消えている'
    assert 'num_boosted_rounds()' in src[i - 400:i]


def test_shap_breakdown_passes_iteration_range_explicitly():
    """SHAP 分解も同じ本数で計算すること（合計が ability_margin と一致するため）。"""
    src = open(os.path.join(ROOT, 'src', 'features', 'shap_explain.py'),
               encoding='utf-8').read()
    i = src.index('contribs = booster.predict(')
    assert 'iteration_range=(0, booster.num_boosted_rounds())' in src[i:i + 220]
