"""脚質×コース適性 + 展開予想特徴量のテスト"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (
    estimate_horse_style,
    predict_race_pace,
    calc_style_course_fit,
    calc_pace_fit,
)


def _base_dir():
    return os.path.join(os.path.dirname(__file__), '..')


# ── estimate_horse_style ──────────────────────────────────────────────────

def test_estimate_style_from_running_style_japanese():
    """日本語 running_style から英語キーを返す"""
    horse = {'history': [{'running_style': '先行'}] * 3 + [{'running_style': '逃げ'}] * 2}
    assert estimate_horse_style(horse) == 'front'


def test_estimate_style_from_running_style_english():
    """英語 running_style もそのまま受け取れる（テスト用パス）"""
    horse = {'history': [{'running_style': 'stalk'}] * 5}
    assert estimate_horse_style(horse) == 'stalk'


def test_estimate_style_fallback_corner():
    """running_style がなくても corner_3 で推定できる"""
    horse = {'history': [{'corner_3': 1.0}, {'corner_3': 2.0}, {'corner_3': 1.5}]}
    assert estimate_horse_style(horse) == 'escape'


def test_estimate_style_no_history():
    """過去走なしは None を返す"""
    assert estimate_horse_style({'history': []}) is None


# ── predict_race_pace ──────────────────────────────────────────────────────

def test_predict_pace_high_many_escape():
    """逃げ馬4頭以上 → ハイペース、差し有利"""
    horses = [{'history': [{'running_style': 'escape'}] * 5} for _ in range(4)]
    pace = predict_race_pace(horses)
    assert pace['pace'] == 'high'
    assert pace['favored_style'] == 'stalk'
    assert pace['n_escape'] == 4


def test_predict_pace_slow_all_stalk():
    """差し馬だけ → スローペース（逃げ0頭）"""
    horses = [{'history': [{'running_style': 'stalk'}] * 5} for _ in range(10)]
    pace = predict_race_pace(horses)
    assert pace['pace'] == 'slow'


def test_predict_pace_slow_japanese_style():
    """日本語 running_style を渡してもペース判定が正しい"""
    horses = [{'history': [{'running_style': '差し'}] * 5} for _ in range(8)]
    pace = predict_race_pace(horses)
    assert pace['pace'] == 'slow'


def test_predict_pace_empty_horses():
    """馬なしでもクラッシュしない"""
    pace = predict_race_pace([])
    assert pace['pace'] in ('high', 'middle', 'slow')


# ── calc_style_course_fit ────────────────────────────────────────────────

def test_style_course_fit_stalk_at_hakodate():
    """函館芝（先行有利）の差し馬 → fit < 0.3"""
    race = {'racecourse': '函館', 'surface': '芝'}
    horse = {'history': [{'running_style': 'stalk'}] * 5}
    fit = calc_style_course_fit(horse, race, _base_dir())
    assert fit < 0.3, f'函館芝の差しは0.16程度のはず: got {fit}'


def test_style_course_fit_front_at_hakodate():
    """函館芝（先行有利）の先行馬 → fit > 0.4"""
    race = {'racecourse': '函館', 'surface': '芝'}
    horse = {'history': [{'running_style': 'front'}] * 5}
    fit = calc_style_course_fit(horse, race, _base_dir())
    assert fit > 0.4, f'函館芝の先行は0.47程度のはず: got {fit}'


def test_style_course_fit_stalk_at_tokyo():
    """東京芝（差し有利）の差し馬 → fit > 0.30"""
    race = {'racecourse': '東京', 'surface': '芝'}
    horse = {'history': [{'running_style': 'stalk'}] * 5}
    fit = calc_style_course_fit(horse, race, _base_dir())
    assert fit > 0.30, f'東京芝の差しは0.35程度のはず: got {fit}'


def test_style_course_fit_unknown_style():
    """脚質不明時は 0.25 を返す"""
    race = {'racecourse': '函館', 'surface': '芝'}
    horse = {'history': []}
    assert calc_style_course_fit(horse, race, _base_dir()) == 0.25


def test_style_course_fit_unknown_course():
    """course_profiles に未定義のコースは 0.25 を返す"""
    race = {'racecourse': '仮設競馬場', 'surface': '芝'}
    horse = {'history': [{'running_style': 'escape'}] * 5}
    assert calc_style_course_fit(horse, race, _base_dir()) == 0.25


# ── calc_pace_fit ────────────────────────────────────────────────────────

def test_pace_fit_stalk_at_highpace():
    """ハイペースで差し馬 → fit = 0.70"""
    horse = {'history': [{'running_style': 'stalk'}] * 5}
    pace_info = {'pace': 'high', 'favored_style': 'stalk'}
    assert calc_pace_fit(horse, pace_info) == 0.70


def test_pace_fit_escape_at_slowpace():
    """スローペースで逃げ馬 → fit = 0.75"""
    horse = {'history': [{'running_style': 'escape'}] * 5}
    pace_info = {'pace': 'slow', 'favored_style': 'front'}
    assert calc_pace_fit(horse, pace_info) == 0.75


def test_pace_fit_unknown_style():
    """脚質不明時は 0.5 を返す"""
    horse = {'history': []}
    pace_info = {'pace': 'high', 'favored_style': 'stalk'}
    assert calc_pace_fit(horse, pace_info) == 0.5


# ── 統合: calc_features_for_xgb に3特徴量が含まれる ────────────────────

def _set_base_dir():
    """XGBモデルロードをスキップしつつ _BASE_DIR だけ設定する（CI環境用）。"""
    import src.features.engine as _eng
    _eng._BASE_DIR = _base_dir()
    _eng._COURSE_PROFILES = None   # キャッシュをクリアして次回ロード時に再読み込みさせる


def test_features_for_xgb_includes_style_features():
    """calc_features_for_xgb の出力に3つの脚質特徴量が含まれる"""
    import src.features.engine as _eng
    from src.features.engine import calc_features_for_xgb
    _set_base_dir()
    _eng._XGB_FUKUSHO_MODEL = None   # XGBなしでルールベースパス

    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 2000,
        'race_class': '3勝クラス', 'track_condition': '良',
        'first_3f': 36.0, 'date': '2026-06-01',
        'horses': [
            {'name': 'テスト馬A', 'horse_num': 1,
             'running_style': '差し',
             'history': [{'running_style': 'stalk', 'place': 2, 'distance': 2000,
                          'surface': '芝', 'racecourse': '東京',
                          'last_3f': 34.5, 'corner_3': 8}]},
        ],
    }
    horse = race['horses'][0]
    horse.update({'jockey': '福永', 'trainer': '藤原英', 'weight_load': 56.0,
                  'jockey_rate': 0.15, 'trainer_rate': 0.12, 'age': 4, 'sex': '牡'})

    feats = calc_features_for_xgb(horse, race)

    assert 'f_style_course_fit' in feats
    assert 'f_pace_fit' in feats
    assert 'f_style_total_fit' in feats
    total = feats['f_style_total_fit']
    assert abs(total - (feats['f_style_course_fit'] + feats['f_pace_fit']) / 2) < 0.001


# ── pace_info キャッシュ（レース単位で1回計算）──────────────────────────

def test_pace_info_cached_in_race_dict():
    """predict_race_pace がレース dict にキャッシュされ2回計算されない"""
    import src.features.engine as _eng
    from src.features.engine import calc_features_for_xgb
    _set_base_dir()
    _eng._XGB_FUKUSHO_MODEL = None

    horses = []
    for i in range(3):
        horses.append({
            'name': f'馬{i}', 'horse_num': i + 1, 'running_style': '先行',
            'history': [{'running_style': 'front', 'place': i + 1, 'distance': 1800,
                         'surface': '芝', 'racecourse': '阪神',
                         'last_3f': 35.0, 'corner_3': 4}],
            'jockey': '', 'trainer': '', 'weight_load': 56.0,
            'jockey_rate': 0.15, 'trainer_rate': 0.12, 'age': 4, 'sex': '牡',
        })

    race = {
        'racecourse': '阪神', 'surface': '芝', 'distance': 1800,
        'race_class': '2勝クラス', 'track_condition': '良',
        'first_3f': 36.0, 'date': '2026-06-01', 'horses': horses,
    }

    for h in horses:
        calc_features_for_xgb(h, race)

    assert '_pace_info_cache' in race, 'pace_info がキャッシュされていない'


if __name__ == '__main__':
    for fn_name in [n for n in dir() if n.startswith('test_')]:
        fn = eval(fn_name)
        fn()
        print(f'✅ {fn_name}')
