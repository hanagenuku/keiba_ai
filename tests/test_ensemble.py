import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools.train_xgb import _popularity_to_base_margin


def test_train_ensemble_import():
    """train_ensemble が正しくインポートできる"""
    from src.tools.train_xgb import train_ensemble
    assert callable(train_ensemble)


def test_popularity_to_base_margin_monotonic():
    """人気が高い(小さい)ほど base_margin が大きい"""
    import numpy as np
    pops = np.array([1, 5, 10, 16])
    ns   = np.array([16, 16, 16, 16])
    bm = _popularity_to_base_margin(
        type('S', (), {'values': pops})(),
        type('S', (), {'values': ns})(),
    )
    for i in range(len(bm) - 1):
        assert bm[i] > bm[i + 1], f'bm[{i}]={bm[i]:.3f} <= bm[{i+1}]={bm[i+1]:.3f}'


def test_pace_features_in_xgb_output():
    """calc_features_for_xgb がペースシナリオ特徴量3個を出力する"""
    from src.features.engine import calc_features_for_xgb
    horses = [
        {'name': f'H{i}', 'horse_num': i + 1, 'running_style': '差し',
         'history': [{'running_style': '差し', 'corner_3': 8}]}
        for i in range(8)
    ]
    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 1600,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': horses, 'date': '2026-01-01',
    }
    feats = calc_features_for_xgb(horses[0], race)
    assert 'f_pace_prob_fast' in feats
    assert 'f_pace_prob_slow' in feats
    assert 'f_pace_x_style' in feats
    assert 0.0 <= feats['f_pace_prob_fast'] <= 1.0
    assert 0.0 <= feats['f_pace_prob_slow'] <= 1.0


if __name__ == '__main__':
    test_train_ensemble_import()
    print('✅ test_train_ensemble_import passed')
    test_popularity_to_base_margin_monotonic()
    print('✅ test_popularity_to_base_margin_monotonic passed')
    test_pace_features_in_xgb_output()
    print('✅ test_pace_features_in_xgb_output passed')
