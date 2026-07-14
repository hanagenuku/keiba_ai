import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.horse_type import (
    calc_agari_ability, calc_stamina_score, calc_speed_score,
    calc_optimal_distance, calc_distance_features,
)


def test_agari_ability_basic():
    history = [
        {'agari_rank': 1, 'field_size': 16},
        {'agari_rank': 8, 'field_size': 16},
    ]
    val = calc_agari_ability(history)
    assert 0.5 < val <= 1.0


def test_agari_ability_no_data():
    assert calc_agari_ability([]) == 0.5
    assert calc_agari_ability([{'distance': 1600}]) == 0.5


def test_stamina_score_long_dist_winner():
    history = [
        {'distance': 2200, 'place': 1, 'field_size': 14},
        {'distance': 2000, 'place': 2, 'field_size': 15},
        {'distance': 1400, 'place': 10, 'field_size': 16},
    ]
    val = calc_stamina_score(history)
    assert val > 0.5


def test_speed_score_short_dist_winner():
    history = [
        {'distance': 1200, 'place': 1, 'field_size': 14},
        {'distance': 1400, 'place': 2, 'field_size': 15},
        {'distance': 2000, 'place': 8, 'field_size': 16},
    ]
    val = calc_speed_score(history)
    assert val > 0.5


def test_optimal_distance_sprint():
    history = [
        {'distance': 1200, 'place': 1, 'field_size': 14},
        {'distance': 1200, 'place': 2, 'field_size': 14},
        {'distance': 1400, 'place': 1, 'field_size': 14},
        {'distance': 2000, 'place': 10, 'field_size': 16},
    ]
    opt, conf = calc_optimal_distance(history)
    assert 1100 <= opt <= 1400
    assert 0.0 < conf <= 1.0


def test_optimal_distance_no_data():
    opt, conf = calc_optimal_distance([])
    assert opt == 1600
    assert conf == 0.0


def test_speed_type_shortening():
    """スピード型の距離短縮を検出"""
    history = [
        {'distance': 2000, 'place': 8, 'field_size': 16, 'date': '2026-05-01'},
        {'distance': 1400, 'place': 1, 'field_size': 14, 'date': '2026-03-01'},
        {'distance': 1400, 'place': 2, 'field_size': 15, 'date': '2026-01-01'},
    ]
    race = {'distance': 1600, 'date': '2026-07-05'}
    feats = calc_distance_features({}, race, history)

    assert feats['f_speed_score'] > 0.5
    assert feats['f_stamina_score'] < 0.5
    assert 1300 <= feats['f_optimal_distance'] <= 1500
    assert feats['f_dist_change'] == -400
    assert feats['f_speed_x_shortening'] > 0.3


def test_stamina_type_extension():
    """スタミナ型の距離延長を検出"""
    history = [
        {'distance': 1400, 'place': 10, 'field_size': 16, 'date': '2026-05-01'},
        {'distance': 2200, 'place': 1, 'field_size': 14, 'date': '2026-03-01'},
        {'distance': 2000, 'place': 2, 'field_size': 15, 'date': '2026-01-01'},
    ]
    race = {'distance': 2400, 'date': '2026-07-05'}
    feats = calc_distance_features({}, race, history)
    assert feats['f_stamina_score'] > 0.5
    assert feats['f_dist_change'] == 1000
    assert feats['f_stamina_x_extension'] > 0.3


def test_no_history_defaults():
    """データなし時のデフォルト値"""
    feats = calc_distance_features({}, {'distance': 1600}, [])
    assert feats['f_agari_ability'] == 0.5
    assert feats['f_optimal_distance'] == 1600
    assert feats['f_dist_confidence'] == 0.0
    assert feats['f_dist_change'] == 0
    assert feats['f_speed_x_shortening'] == 0.0
    assert feats['f_stamina_x_extension'] == 0.0


def test_all_nine_features_present():
    """9特徴量がすべて含まれるか"""
    feats = calc_distance_features({}, {'distance': 1600}, [])
    expected = [
        'f_agari_ability', 'f_stamina_score', 'f_speed_score',
        'f_optimal_distance', 'f_dist_vs_optimal', 'f_dist_change',
        'f_speed_x_shortening', 'f_stamina_x_extension', 'f_dist_confidence',
    ]
    for key in expected:
        assert key in feats, f'{key} missing'


def test_data_leak_prevention():
    """calc_features_for_xgb でデータリークが起きないことを確認"""
    from src.features.engine import calc_features_for_xgb

    horse = {
        'name': 'TestHorse', 'horse_num': 1, 'running_style': '差し',
        'history': [
            {'distance': 1600, 'place': 1, 'field_size': 16,
             'agari_rank': 1, 'date': '2026-08-01',
             'running_style': '差し', 'corner_3': 5},
            {'distance': 1400, 'place': 5, 'field_size': 14,
             'agari_rank': 7, 'date': '2026-03-01',
             'running_style': '差し', 'corner_3': 6},
        ],
    }
    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 1600,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': [horse], 'date': '2026-07-05',
    }
    feats = calc_features_for_xgb(horse, race)

    # 2026-08-01 のデータは除外されるので、
    # agari_ability は 2026-03-01 の1走分のみから算出される
    # (agari_rank=7, field_size=14 → 1 - 6/13 ≈ 0.538)
    assert 0.4 <= feats['f_agari_ability'] <= 0.6
    # optimal_distance は 2026-03-01 の1走分のみ → 1400
    assert feats['f_optimal_distance'] == 1400


def test_interaction_zero_when_no_change():
    """距離変化なしなら相互作用項は0"""
    history = [
        {'distance': 1600, 'place': 1, 'field_size': 14, 'date': '2026-05-01'},
    ]
    race = {'distance': 1600, 'date': '2026-07-05'}
    feats = calc_distance_features({}, race, history)
    assert feats['f_dist_change'] == 0
    assert feats['f_speed_x_shortening'] == 0.0
    assert feats['f_stamina_x_extension'] == 0.0


def test_finishers_alias():
    """field_size がなくても finishers で動作する"""
    history = [
        {'agari_rank': 2, 'finishers': 16},
        {'distance': 1200, 'place': 1, 'finishers': 14},
    ]
    val = calc_agari_ability(history)
    assert val > 0.5


if __name__ == '__main__':
    test_agari_ability_basic()
    print('ok test_agari_ability_basic')
    test_agari_ability_no_data()
    print('ok test_agari_ability_no_data')
    test_stamina_score_long_dist_winner()
    print('ok test_stamina_score_long_dist_winner')
    test_speed_score_short_dist_winner()
    print('ok test_speed_score_short_dist_winner')
    test_optimal_distance_sprint()
    print('ok test_optimal_distance_sprint')
    test_optimal_distance_no_data()
    print('ok test_optimal_distance_no_data')
    test_speed_type_shortening()
    print('ok test_speed_type_shortening')
    test_stamina_type_extension()
    print('ok test_stamina_type_extension')
    test_no_history_defaults()
    print('ok test_no_history_defaults')
    test_all_nine_features_present()
    print('ok test_all_nine_features_present')
    test_data_leak_prevention()
    print('ok test_data_leak_prevention')
    test_interaction_zero_when_no_change()
    print('ok test_interaction_zero_when_no_change')
    test_finishers_alias()
    print('ok test_finishers_alias')
    print('\nAll tests passed!')
