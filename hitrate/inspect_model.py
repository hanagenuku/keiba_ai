#!/usr/bin/env python3
"""本番の残差モデルを「ファイルだけ」から調べる（レース結果は一切使わない）。

使い方: python3 hitrate/inspect_model.py
見るもの:
  1. 木の本数と best_iteration
  2. 各木が全馬に一律で足している量（葉の値の cover 加重平均＝切片の移動）
  3. 各木が馬ごとに付けている差（葉の値の cover 加重標準偏差＝特徴量の信号）
  4. Booster.predict() が既定で何本の木を使うか
"""
import json
import os
import warnings

import numpy as np
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'data', 'xgb_fukusho_model_residual.pkl')
COLS = os.path.join(ROOT, 'data', 'xgb_feature_cols_residual.json')


def main():
    warnings.simplefilter('ignore')
    b = xgb.Booster()
    b.load_model(MODEL)
    cfg = json.loads(b.save_config())['learner']
    print('xgboost            :', xgb.__version__)
    print('木の本数            :', b.num_boosted_rounds())
    print('attributes         :', b.attributes())
    print('scale_pos_weight   :', cfg['objective']['reg_loss_param']['scale_pos_weight'])

    df = b.trees_to_dataframe()
    leaf = df[df.Feature == 'Leaf']
    shift, spread = [], []
    for _, g in leaf.groupby('Tree'):
        m = np.average(g.Gain, weights=g.Cover)
        shift.append(m)
        spread.append(np.sqrt(np.average((g.Gain - m) ** 2, weights=g.Cover)))
    shift, spread = np.array(shift), np.array(spread)
    print('\n木ごとの一律の移動（先頭12本）:', np.round(shift[:12], 3))
    print('木ごとの馬間の差 SD（先頭12本）:', np.round(spread[:12], 3))
    for k in (10, 11, 20, 40, len(shift)):
        print(f'  先頭{k:3d}本まで: 一律の移動の累計 {shift[:k].sum():+.3f}'
              f' / 馬間の差SDの単純和 {spread[:k].sum():.3f}')

    cols = json.load(open(COLS))['feature_cols']
    rng = np.random.default_rng(0)
    d = xgb.DMatrix(rng.normal(5, 2, (200, len(cols))), feature_names=cols)
    d.set_base_margin(np.zeros(200))
    p_all = b.predict(d, output_margin=True)
    p_best = b.predict(d, output_margin=True,
                       iteration_range=(0, int(b.attributes()['best_iteration']) + 1))
    p_full = b.predict(d, output_margin=True, iteration_range=(0, b.num_boosted_rounds()))
    print('\npredict() 既定 と 全木 の最大差        :', float(np.abs(p_all - p_full).max()))
    print('predict() 既定 と best_iteration まで の最大差:', float(np.abs(p_all - p_best).max()))


if __name__ == '__main__':
    main()
