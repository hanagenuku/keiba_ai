"""単独モデル（AI単 列用）の学習が、本番の列をそのまま使い、市場を見ないこと。"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def _frame(cols, n_races=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        date = f'2026-0{1 + r % 8}-1{r % 9}'
        for h in range(1, 11):
            x = rng.normal(size=len(cols))
            rows.append(dict(zip(cols, x), race_id=f'r{r}', date=date, horse_num=h,
                             f_popularity=h, is_fukusho=int(x[0] + rng.normal() > 0.8)))
    df = pd.DataFrame(rows)
    df.loc[df.index % 3 == 0, 'f_ws_kayochinkeiba_rank'] = np.nan   # 指数が無い馬は NaN のまま
    return df


def test_train_solo_uses_production_cols_and_no_market(tmp_path):
    from src.tools.train_solo import COLS_NAME, MODEL_NAME, train_solo
    cols = ['f_a', 'f_b', 'f_ws_kayochinkeiba_rank']
    os.makedirs(tmp_path / 'data')
    with open(tmp_path / 'data' / 'xgb_feature_cols.json', 'w') as f:
        json.dump({'feature_cols': cols, 'residual': True}, f)
    df = _frame(cols)
    b, meta = train_solo(str(tmp_path), train_end='2026-06-30', val_start='2026-07-01', val_end='2026-08-31', df=df)
    assert meta['feature_cols'] == cols and 'f_popularity' not in meta['feature_cols']
    # 実際に保存したファイルを読み直して予測できる（保存形式ごと確かめる）
    b2 = xgb.Booster()
    b2.load_model(str(tmp_path / 'data' / MODEL_NAME))
    assert b2.num_boosted_rounds() == meta['n_trees'] >= 1
    with open(tmp_path / 'data' / COLS_NAME) as f:
        assert json.load(f)['feature_cols'] == cols
    # 保存した全木で予測できる
    X = df[cols].head(20)
    p1 = b2.predict(xgb.DMatrix(X, feature_names=cols), output_margin=True)
    assert np.isfinite(p1).all()


def test_train_solo_fails_when_training_data_lacks_a_production_col(tmp_path):
    import pytest
    from src.tools.train_solo import train_solo
    os.makedirs(tmp_path / 'data')
    with open(tmp_path / 'data' / 'xgb_feature_cols.json', 'w') as f:
        json.dump({'feature_cols': ['f_a', 'f_missing']}, f)
    with pytest.raises(ValueError):
        train_solo(str(tmp_path), df=_frame(['f_a']))
