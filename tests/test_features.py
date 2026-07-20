import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (
    calc_performance_index, f_recent, calc_chaos_score,
    auto_comment, dist_zone_label, dz,
    calc_course_aptitude_features, load_course_profiles, get_course_profile,
    calc_features_for_xgb, _ensure_escape_front_count, f_blood, _bayes_rate,
)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def test_calc_performance_index_basic():
    pi = calc_performance_index(34.0, distance=1600, surface='芝', condition='良')
    assert isinstance(pi, float)
    assert 0.0 <= pi <= 100.0


def test_calc_performance_index_pace_correction():
    # ハイペース（前半速い）の場合、上りが遅くても高い指数になる
    pi_high = calc_performance_index(35.5, first_3f=33.0, corner_pos=2, distance=1600, surface='芝')
    pi_slow = calc_performance_index(35.5, first_3f=36.5, corner_pos=10, distance=1600, surface='芝')
    assert pi_high > pi_slow


def test_f_recent_no_history():
    h = {'win_odds': 5.0, 'history': []}
    race = {'distance': 1600, 'surface': '芝', 'num_horses': 16}
    score = f_recent(h, race)
    assert 0.0 <= score <= 10.0


def test_f_recent_with_history():
    h = {
        'win_odds': 5.0,
        'history': [
            {'place': 1, 'finishers': 16, 'margin': 0.0, 'last_3f': 33.5,
             'first_3f': 35.0, 'corner_3': 2, 'surface': '芝', 'distance': 1600},
            {'place': 2, 'finishers': 16, 'margin': 0.2, 'last_3f': 34.0,
             'first_3f': 35.5, 'corner_3': 5, 'surface': '芝', 'distance': 1600},
        ],
    }
    race = {'distance': 1600, 'surface': '芝', 'num_horses': 16}
    score = f_recent(h, race)
    assert 0.0 <= score <= 10.0
    assert score > 5.0  # 1着・2着の実績があるので5以上


def test_calc_chaos_score_clear():
    scored = [
        {'total': 9.0}, {'total': 5.0}, {'total': 4.0},
    ]
    chaos = calc_chaos_score({}, scored)
    assert chaos < 0.5  # 差が大きいので混戦度低い


def test_calc_chaos_score_tight():
    scored = [
        {'total': 6.0}, {'total': 5.9}, {'total': 5.8},
    ]
    chaos = calc_chaos_score({}, scored)
    assert chaos > 0.5  # 差が小さいので混戦度高い


def test_dist_zone_label():
    assert dist_zone_label(1200) == '短距離'
    assert dist_zone_label(1600) == 'マイル'
    assert dist_zone_label(2000) == '中距離'
    assert dist_zone_label(3000) == '長距離'


def test_dz():
    assert dz(1200) == 'sp'
    assert dz(1600) == 'mi'
    assert dz(2000) == 'md'
    assert dz(3200) == 'lo'


# ── コース適性特徴量 ──────────────────────────────────────────────
def test_course_profiles_loads_all():
    profiles = load_course_profiles(ROOT)
    assert profiles is not None
    assert len(profiles['courses']) >= 20  # 10競馬場 × 芝/ダート
    assert get_course_profile('東京', '芝', ROOT)['straight_class'] == 'long'
    assert get_course_profile('中山', '芝', ROOT)['straight_class'] == 'short'


def test_course_aptitude_tokyo_specialist():
    history = [
        {'racecourse': '東京', 'surface': '芝', 'place': 1, 'agari3f': 33.2},
        {'racecourse': '東京', 'surface': '芝', 'place': 2, 'agari3f': 33.5},
        {'racecourse': '中山', 'surface': '芝', 'place': 12, 'agari3f': 36.8},
        {'racecourse': '中山', 'surface': '芝', 'place': 10, 'agari3f': 37.1},
    ]
    # 今日が東京芝 → 東京で2戦2好走（ベイズ縮小により1.0ではなく中立値寄りの0.598）
    feats = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    assert feats['f_same_course_rate'] == _bayes_rate([1, 1])
    assert feats['f_course_coverage'] == 2
    # 今日が中山芝 → 中山で2戦2凡走（同様に0.0ではなく0.198）
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', history, ROOT)
    assert feats['f_same_course_rate'] == _bayes_rate([0, 0])
    assert feats['f_course_coverage'] == 2


def test_course_aptitude_straight_match():
    # 東京(long)で好走。今日が新潟(very_long)なら直線クラスは異なるが、
    # 同じlong同士のマッチを検証するため今日も東京で確認する。
    history = [
        {'racecourse': '東京', 'surface': '芝', 'place': 1, 'agari3f': 33.2},
        {'racecourse': '函館', 'surface': '芝', 'place': 8, 'agari3f': 35.0},
    ]
    feats = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    # straight_class=long の過去走は東京の1走（好走）のみ
    # → ベイズ縮小で1.0ではなく中立値寄りの0.497（n=1は信頼度が低いため）
    assert feats['f_straight_match'] == _bayes_rate([1])
    # long コースの最速上がりは 33.2
    assert feats['f_agari_at_similar'] == 33.2


def test_course_aptitude_no_history():
    feats = calc_course_aptitude_features('新馬', '東京', '芝', [], ROOT)
    # 未経験(n=0)はベイズ縮小のprior(0.33)。0.0（=確実に凡走）と区別する
    assert feats['f_same_course_rate'] == _bayes_rate([])
    assert feats['f_course_coverage'] == 0
    assert feats['f_agari_at_similar'] == 99.0


def test_course_aptitude_unknown_course():
    # 未定義の競馬場（地方など）はデフォルト返却
    feats = calc_course_aptitude_features('テスト馬', '大井', 'ダート', [], ROOT)
    assert feats == {
        'f_same_course_rate': 0.0, 'f_same_turn_rate': 0.0,
        'f_straight_match': 0.0, 'f_uphill_match': 0.0,
        'f_agari_at_similar': 99.0, 'f_course_coverage': 0,
        'f_course_type_rate': 0.0, 'f_tight_vs_spacious': 0.0,
        'f_uphill_severity_rate': 0.0, 'f_corner_position_change': 0.0,
        'f_agari_rank_at_type': 0.5,
    }


def test_course_type_rate_tight_specialist():
    """小回りコース(tight/flat_tight)で好走、大箱(spacious)で凡走する馬"""
    history = [
        {'racecourse': '中山', 'surface': '芝', 'place': 1, 'corner_all': '3-3-2-1'},
        {'racecourse': '福島', 'surface': '芝', 'place': 2, 'corner_all': '5-4-3-2'},
        {'racecourse': '函館', 'surface': '芝', 'place': 1, 'corner_all': '4-3-2-1'},
        {'racecourse': '東京', 'surface': '芝', 'place': 10},
        {'racecourse': '新潟', 'surface': '芝', 'place': 12},
    ]
    # 今日が中山(tight) → tight系3戦3好走（ベイズ縮小で1.0ではなく0.598）
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', history, ROOT)
    assert feats['f_course_type_rate'] == _bayes_rate([1, 1])  # 中山+福島=2好走/2走
    assert feats['f_tight_vs_spacious'] > 0    # tight側 > spacious側 → 正（縮小後も方向は不変）
    assert feats['f_corner_position_change'] > 0  # 3角→4角で前に出ている

    # 今日が東京(spacious)
    feats2 = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    assert feats2['f_course_type_rate'] == _bayes_rate([0, 0])  # spacious: 東京0+新潟0 / 2走


def test_uphill_severity_rate():
    """急坂コース(steep)で好走する馬"""
    history = [
        {'racecourse': '中山', 'surface': '芝', 'place': 1},  # steep
        {'racecourse': '阪神', 'surface': '芝', 'place': 2},  # steep
        {'racecourse': '京都', 'surface': '芝', 'place': 8},  # none
        {'racecourse': '新潟', 'surface': '芝', 'place': 10}, # none
    ]
    # 今日が中京(steep)
    feats = calc_course_aptitude_features('テスト馬', '中京', '芝', history, ROOT)
    assert feats['f_uphill_severity_rate'] == _bayes_rate([1, 1])  # steep: 2好走/2走


def test_corner_position_change():
    """小回りコースでの3→4角の位置変動"""
    history = [
        # 3角5位→4角2位 = +3（前に出る = 器用）
        {'racecourse': '中山', 'surface': '芝', 'place': 2, 'corner_all': '6-5-5-2'},
        # 3角3位→4角1位 = +2
        {'racecourse': '福島', 'surface': '芝', 'place': 1, 'corner_all': '4-3-3-1'},
    ]
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', history, ROOT)
    assert feats['f_corner_position_change'] == 2.5  # (3+2)/2


def test_agari_rank_at_type():
    """同タイプコースでの上がり相対順位"""
    history = [
        {'racecourse': '東京', 'surface': '芝', 'place': 2,
         'agari_rank': 1, 'num_finishers': 16},  # spacious, 上がり1位/16頭
        {'racecourse': '新潟', 'surface': '芝', 'place': 3,
         'agari_rank': 2, 'num_finishers': 18},  # spacious, 上がり2位/18頭
    ]
    feats = calc_course_aptitude_features('テスト馬', '東京', '芝', history, ROOT)
    # (1/16 + 2/18) / 2 ≈ 0.087
    assert 0.08 < feats['f_agari_rank_at_type'] < 0.09


def test_tight_vs_spacious_no_data():
    """片方のコースタイプしかデータがない場合は0.0"""
    history = [
        {'racecourse': '中山', 'surface': '芝', 'place': 1},
    ]
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', history, ROOT)
    # spaciousのデータなし → tight_vs_spacious = 0.0
    assert feats['f_tight_vs_spacious'] == 0.0


# ── ベイズ縮小レート（少走数での極端値対策） ──────────────────────────
def test_bayes_rate_no_data_returns_prior():
    """未経験(n=0)はpriorをそのまま返す（0.0固定にしない）"""
    assert _bayes_rate([]) == 0.33
    assert _bayes_rate([], prior=0.3) == 0.3


def test_bayes_rate_small_sample_shrinks_toward_prior():
    """1走2走の全勝/全敗は1.0/0.0まで振れず、priorへ引き寄せられる"""
    assert 0.33 < _bayes_rate([1]) < 1.0
    assert 0.0 < _bayes_rate([0]) < 0.33
    # サンプルが増えるほど実測値（1.0）に近づく（縮小幅が小さくなる）
    assert _bayes_rate([1]) < _bayes_rate([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_bayes_rate_large_sample_converges_to_observed():
    """十分な走数があれば実測レートにほぼ収束する"""
    hits = [1] * 47 + [0] * 3  # 50走47勝
    assert abs(_bayes_rate(hits) - 0.94) < 0.04


def test_course_aptitude_single_race_less_extreme_than_two():
    """同じ全勝でも経験走数が少ないほどベイズ縮小の影響が大きい"""
    history_1race = [{'racecourse': '東京', 'surface': '芝', 'place': 1}]
    history_2race = [
        {'racecourse': '東京', 'surface': '芝', 'place': 1},
        {'racecourse': '東京', 'surface': '芝', 'place': 1},
    ]
    feats_1 = calc_course_aptitude_features('テスト馬', '東京', '芝', history_1race, ROOT)
    feats_2 = calc_course_aptitude_features('テスト馬', '東京', '芝', history_2race, ROOT)
    assert feats_1['f_same_course_rate'] < feats_2['f_same_course_rate'] < 1.0


def test_fukusho_rate_features_shrink_for_thin_history():
    """f_dist_fukusho / f_course_fukusho / f_recent_fukusho も1走のみでは1.0にならない"""
    horse = {
        'name': 'テスト馬', 'horse_num': 1, 'running_style': '差し',
        'history': [
            {'racecourse': '東京', 'surface': '芝', 'distance': 1600, 'place': 1, 'corner_3': 3},
        ],
    }
    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 1600,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': [horse], 'date': '2026-01-01',
    }
    feats = calc_features_for_xgb(horse, race)
    assert 0.33 < feats['f_dist_fukusho'] < 1.0
    assert 0.33 < feats['f_course_fukusho'] < 1.0
    assert 0.33 < feats['f_recent_fukusho'] < 1.0


# ── ペースシナリオ特徴量 ──────────────────────────────────────────────
def test_ensure_escape_front_count():
    """horses の running_style から escape/front count を自動算出"""
    race = {
        'horses': [
            {'running_style': '逃げ'},
            {'running_style': '先行'},
            {'running_style': '先行'},
            {'running_style': '差し'},
            {'running_style': '追込'},
        ],
    }
    _ensure_escape_front_count(race)
    assert race['escape_count'] == 1
    assert race['front_count'] == 2


def test_ensure_escape_front_count_already_set():
    """既に設定済みなら上書きしない"""
    race = {'escape_count': 3, 'front_count': 5, 'horses': []}
    _ensure_escape_front_count(race)
    assert race['escape_count'] == 3
    assert race['front_count'] == 5


def test_pace_scenario_features_sashi():
    """差し馬はハイペース期待時に f_pace_x_style > 0"""
    horses = [
        {'name': 'A', 'horse_num': 1, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 1}]},
        {'name': 'B', 'horse_num': 2, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 1}]},
        {'name': 'C', 'horse_num': 3, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 2}]},
        {'name': 'D', 'horse_num': 4, 'running_style': '差し',
         'history': [{'running_style': '差し', 'corner_3': 8}]},
        {'name': 'E', 'horse_num': 5, 'running_style': '追込',
         'history': [{'running_style': '追込', 'corner_3': 12}]},
    ]
    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 1600,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': horses, 'date': '2026-01-01',
    }
    feats = calc_features_for_xgb(horses[3], race)  # 差し馬
    assert 'f_pace_prob_fast' in feats
    assert 'f_pace_prob_slow' in feats
    assert 'f_pace_x_style' in feats
    assert feats['f_pace_x_style'] > 0  # 逃げ3頭→ハイペース→差し有利


def test_pace_scenario_features_nige():
    """逃げ馬はハイペース予想時に f_pace_x_style < 0（不利）"""
    horses = [
        {'name': 'A', 'horse_num': 1, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 1}]},
        {'name': 'B', 'horse_num': 2, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 2}]},
        {'name': 'C', 'horse_num': 3, 'running_style': '逃げ',
         'history': [{'running_style': '逃げ', 'corner_3': 1}]},
        {'name': 'D', 'horse_num': 4, 'running_style': '差し',
         'history': [{'running_style': '差し', 'corner_3': 8}]},
        {'name': 'E', 'horse_num': 5, 'running_style': '追込',
         'history': [{'running_style': '追込', 'corner_3': 13}]},
    ]
    race = {
        'racecourse': '東京', 'surface': '芝', 'distance': 1600,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': horses, 'date': '2026-01-01',
    }
    feats = calc_features_for_xgb(horses[0], race)  # 逃げ馬
    assert feats['f_pace_x_style'] < 0  # 逃げ3頭→ハイペース→逃げ不利


class TestFBlood:
    """f_blood: 母父(dam_sire)は現状スクレイピング未対応で常に空文字になる。
    空の場合に DEF_SIRE(汎用値) とブレンドして父側の実データを希釈しないことを確認する。
    """

    def test_no_dam_sire_avoids_dilution(self):
        # ロードカナロア: 短距離特化(sp:1.0)・長距離は苦手(lo:0.3)。
        # 母父が空のときは父の実データがそのまま反映され、DEF_SIRE(lo:0.8=平均的)と
        # ブレンドするより長距離適性は低く評価されるはず。
        race = {'surface': '芝', 'distance': 3000}  # 'lo'ゾーン
        h_no_dam = {'sire': 'ロードカナロア', 'dam_sire': '', 'age': 2.5}
        h_unknown_dam = {'sire': 'ロードカナロア', 'dam_sire': '存在しない架空馬名', 'age': 2.5}

        score_no_dam = f_blood(h_no_dam, race)
        score_diluted = f_blood(h_unknown_dam, race)

        assert score_no_dam < score_diluted, (
            "母父が空(未取得)の場合は父の実データをそのまま使うべきで、"
            "DEF_SIREとブレンドした場合より苦手条件の評価が下がる（＝薄まらない）はず"
        )

    def test_dam_sire_present_still_blends(self):
        """母父が実際に取得できているケース（将来のスクレイピング対応後）は
        従来通り父70%・母父30%でブレンドする（後方互換の確認）。
        """
        race = {'surface': '芝', 'distance': 3000}
        h_same = {'sire': 'ロードカナロア', 'dam_sire': 'ロードカナロア', 'age': 2.5}
        h_no_dam = {'sire': 'ロードカナロア', 'dam_sire': '', 'age': 2.5}

        # 母父も父と同一プロファイルなら、ブレンドしても無ブレンドと同じ値になる
        assert abs(f_blood(h_same, race) - f_blood(h_no_dam, race)) < 1e-9


if __name__ == '__main__':
    test_calc_performance_index_basic()
    print('✅ test_calc_performance_index_basic passed')
    test_calc_performance_index_pace_correction()
    print('✅ test_calc_performance_index_pace_correction passed')
    test_f_recent_no_history()
    print('✅ test_f_recent_no_history passed')
    test_f_recent_with_history()
    print('✅ test_f_recent_with_history passed')
    test_calc_chaos_score_clear()
    print('✅ test_calc_chaos_score_clear passed')
    test_calc_chaos_score_tight()
    print('✅ test_calc_chaos_score_tight passed')
    test_dist_zone_label()
    print('✅ test_dist_zone_label passed')
    test_dz()
    print('✅ test_dz passed')
    test_course_profiles_loads_all()
    print('✅ test_course_profiles_loads_all passed')
    test_course_aptitude_tokyo_specialist()
    print('✅ test_course_aptitude_tokyo_specialist passed')
    test_course_aptitude_straight_match()
    print('✅ test_course_aptitude_straight_match passed')
    test_course_aptitude_no_history()
    print('✅ test_course_aptitude_no_history passed')
    test_course_aptitude_unknown_course()
    print('✅ test_course_aptitude_unknown_course passed')
