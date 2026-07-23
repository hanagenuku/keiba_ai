import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engine import (
    calc_performance_index, f_recent, calc_chaos_score,
    auto_comment, dist_zone_label, dz,
    calc_course_aptitude_features, load_course_profiles, get_course_profile,
    calc_features_for_xgb, _ensure_escape_front_count, f_blood, _bayes_rate,
    _bayes_shrink, _check_xgb_feature_coverage, _warn_xgb_inference_fallback,
    calc_course_distance_features, load_course_distance_profiles, _resolve_turf_loop,
)
import src.features.engine as engine

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


# ── 距離依存コース特徴量（course_distance_profiles.json） ────────────────
def test_course_distance_profiles_loads():
    profiles = load_course_distance_profiles(ROOT)
    assert profiles is not None
    assert '中山' in profiles['dirt_turf_start']
    assert 1200 in profiles['dirt_turf_start']['中山']


def test_dirt_turf_start_true_for_nakayama_1200():
    # 中山ダート1200mはJRA公式で「芝スタート」と明記されている距離
    feats = calc_course_distance_features('中山', 'ダート', 1200, ROOT)
    assert feats['f_dirt_turf_start'] == 1.0


def test_dirt_turf_start_false_for_nakayama_1800():
    feats = calc_course_distance_features('中山', 'ダート', 1800, ROOT)
    assert feats['f_dirt_turf_start'] == 0.0


def test_dirt_turf_start_hanshin_two_distances():
    # 阪神は1400m・2000mの2距離が芝スタート
    assert calc_course_distance_features('阪神', 'ダート', 1400, ROOT)['f_dirt_turf_start'] == 1.0
    assert calc_course_distance_features('阪神', 'ダート', 2000, ROOT)['f_dirt_turf_start'] == 1.0
    assert calc_course_distance_features('阪神', 'ダート', 1800, ROOT)['f_dirt_turf_start'] == 0.0


def test_dirt_turf_start_venue_without_it():
    # 札幌はダートの芝スタート区間が無い
    feats = calc_course_distance_features('札幌', 'ダート', 1700, ROOT)
    assert feats['f_dirt_turf_start'] == 0.0


def test_course_hill_diff_nakayama():
    feats = calc_course_distance_features('中山', '芝', 2500, ROOT)
    assert feats['f_course_hill_diff'] == 2.2


def test_course_hill_diff_unknown_course_defaults_zero():
    feats = calc_course_distance_features('存在しない場', '芝', 1600, ROOT)
    assert feats['f_course_hill_diff'] == 0.0
    assert feats['f_dirt_turf_start'] == 0.0
    assert feats['f_course_corner_tight'] == 2.0  # Normal相当のデフォルト


def test_resolve_turf_loop_nakayama():
    profiles = load_course_distance_profiles(ROOT)
    # 中山芝2500m(有馬記念)はJRA公式の発走距離表記で(内)タグ＝内回りが舞台
    # （発走地点自体は外回り上にあるが、周回するのは内回りコース）
    assert _resolve_turf_loop(profiles, '中山', 2500) == '内回り'
    assert _resolve_turf_loop(profiles, '中山', 1200) == '外回り'


def test_resolve_turf_loop_no_split_venue_returns_none():
    profiles = load_course_distance_profiles(ROOT)
    # 東京は内回り/外回りの区別が無い競馬場
    assert _resolve_turf_loop(profiles, '東京', 1600) is None


def test_corner_tightness_differs_by_loop_hanshin():
    # 阪神は内回りNormal・外回りWideで、コーナータイト度の値が距離によって変わる
    inner = calc_course_distance_features('阪神', '芝', 1200, ROOT)   # 内回り
    outer = calc_course_distance_features('阪神', '芝', 1800, ROOT)   # 外回り
    assert inner['f_course_corner_tight'] < outer['f_course_corner_tight']


def test_calc_features_for_xgb_includes_course_distance_features():
    horses = [{'name': f'Horse{i}', 'num': i, 'running_style': '差し'} for i in range(1, 9)]
    race = {'racecourse': '中山', 'surface': 'ダート', 'distance': 1200, 'horses': horses}
    feats = calc_features_for_xgb(horses[0], race)
    assert feats['f_dirt_turf_start'] == 1.0
    assert 'f_course_hill_diff' in feats
    assert 'f_course_corner_tight' in feats


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


def test_bayes_shrink_counts_matches_bayes_rate():
    """(的中数, 試行数) 版と list 版は同じ値を返す（_bayes_rateの内部実装）"""
    assert _bayes_shrink(2, 5, prior=0.2, k=4) == _bayes_rate([1, 1, 0, 0, 0], prior=0.2, k=4)
    assert _bayes_shrink(0, 0, prior=0.15, k=10) == 0.15


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


# ── 枠順バイアス（f_post）の3段階フォールバック ─────────────────────
def _make_post_race(racecourse, distance, horse_num, n_horses=16):
    horse = {'name': 'テスト馬', 'horse_num': horse_num, 'running_style': '差し'}
    race = {
        'racecourse': racecourse, 'surface': '芝', 'distance': distance,
        'track_condition': '良', 'race_class': '1勝', 'first_3f': 35.0,
        'horses': [dict(horse, horse_num=i) for i in range(1, n_horses + 1)],
        'date': '2026-01-01',
    }
    return horse, race


def test_f_post_falls_back_to_zone_when_no_real_data():
    """_post_zone_bias に実データが無ければ POST_BIAS_BY_ZONE を使う。
    中山の長距離は+2.0（強く内枠有利）で、競馬場単体の POST_BIAS(+0.5、
    旧規約で外枠有利)とは示す傾向が異なるため、区別できる。"""
    engine._post_zone_bias.pop(('中山', '長距離'), None)
    inner, race = _make_post_race('中山', 2500, horse_num=1)
    outer, _ = _make_post_race('中山', 2500, horse_num=16)
    f_inner = calc_features_for_xgb(inner, race)['f_post']
    f_outer = calc_features_for_xgb(outer, race)['f_post']
    assert f_inner > f_outer  # 長距離の中山は内枠有利(+2.0)なので内枠が高スコア


def test_f_post_prefers_real_data_over_zone_fallback():
    """_post_zone_bias に実データがあれば、POST_BIAS_BY_ZONEより優先される。"""
    try:
        engine._post_zone_bias[('中山', '長距離')] = -3.0  # 実データが強い外枠有利を示す想定
        inner, race = _make_post_race('中山', 2500, horse_num=1)
        outer, _ = _make_post_race('中山', 2500, horse_num=16)
        f_inner = calc_features_for_xgb(inner, race)['f_post']
        f_outer = calc_features_for_xgb(outer, race)['f_post']
        assert f_outer > f_inner  # 実データが優先され、外枠有利の判定に反転する
    finally:
        engine._post_zone_bias.pop(('中山', '長距離'), None)


def test_f_post_falls_back_to_venue_only_when_unknown_venue():
    """実データもゾーン別設定も無い競馬場は POST_BIAS（無ければ0=中立）にフォールバック。"""
    inner, race = _make_post_race('存在しない場', 2500, horse_num=1)
    outer, _ = _make_post_race('存在しない場', 2500, horse_num=16)
    f_inner = calc_features_for_xgb(inner, race)['f_post']
    f_outer = calc_features_for_xgb(outer, race)['f_post']
    assert f_inner == f_outer == 5.0


# ── 騎手・調教師勝率のベイズ縮小（_build_horse_dicts） ────────────────
def _make_history_db(base_dir, rows):
    """テスト用の最小history.db（horse_history）を作成する。"""
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, 'history.db')
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE horse_history (
            race_id TEXT, horse_name TEXT, distance INTEGER, surface TEXT,
            racecourse TEXT, place INTEGER, horse_num INTEGER,
            jockey TEXT, trainer TEXT
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO horse_history (race_id, horse_name, distance, surface, "
            "racecourse, place, horse_num, jockey, trainer) VALUES (?,?,?,?,?,?,?,?,?)",
            (r['race_id'], r.get('horse_name', 'テスト馬'), r.get('distance', 1600),
             r.get('surface', '芝'), r.get('racecourse', '東京'), r['place'],
             r.get('horse_num', 1), r.get('jockey', ''), r.get('trainer', '')),
        )
    conn.commit()
    conn.close()
    return str(base_dir)


def test_build_horse_dicts_jockey_low_sample_shrinks_toward_prior(tmp_path):
    """runs<10の騎手も除外されず、prior(0.15)と実測値の間の値が入る
    （旧実装はrunsな>=10のハードカットオフで丸ごと除外し、lookup側の
    デフォルト0.15に一律フォールバックしていた）"""
    rows = [{'race_id': f'R{i}', 'place': 1, 'jockey': '若手騎手'} for i in range(2)]
    base_dir = _make_history_db(tmp_path, rows)
    engine._jockey_dict = {}
    engine._trainer_dict = {}
    engine._build_horse_dicts(base_dir)
    key = ('若手騎手', '東京', '芝')
    assert key in engine._jockey_dict
    assert 0.15 < engine._jockey_dict[key] < 1.0


def test_build_horse_dicts_jockey_high_sample_close_to_observed(tmp_path):
    """十分な走数があれば実測勝率に近づく"""
    rows = ([{'race_id': f'W{i}', 'place': 1, 'jockey': 'ベテラン騎手'} for i in range(30)]
            + [{'race_id': f'L{i}', 'place': 5, 'jockey': 'ベテラン騎手'} for i in range(20)])
    base_dir = _make_history_db(tmp_path, rows)
    engine._jockey_dict = {}
    engine._trainer_dict = {}
    engine._build_horse_dicts(base_dir)
    val = engine._jockey_dict[('ベテラン騎手', '東京', '芝')]
    assert abs(val - 0.6) < 0.1  # 30勝/50走=0.6に近い


def test_build_horse_dicts_trainer_low_sample_shrinks_toward_prior():
    """調教師も同様にruns<10で除外されず、prior(0.12)寄りの値になる"""
    assert _bayes_shrink(1, 1, prior=0.12, k=10) > 0.12
    assert _bayes_shrink(1, 1, prior=0.12, k=10) < 1.0


# ── XGB特徴量カバレッジ検証（学習時と違う特徴量の静かな混入を検知） ──────
def test_check_xgb_feature_coverage_warns_on_missing_column(capsys):
    """feature_colsにあってxfeatsに無い列があれば警告を1回出す"""
    engine._XGB_MISSING_FEATS_WARNED = set()
    _check_xgb_feature_coverage({'f_a': 1.0}, ['f_a', 'f_b'])
    out = capsys.readouterr().out
    assert 'f_b' in out
    assert 'XGB特徴量' in out


def test_check_xgb_feature_coverage_silent_when_complete(capsys):
    """全列が揃っていれば何も出力しない"""
    engine._XGB_MISSING_FEATS_WARNED = set()
    _check_xgb_feature_coverage({'f_a': 1.0, 'f_b': 2.0}, ['f_a', 'f_b'])
    assert capsys.readouterr().out == ''


def test_check_xgb_feature_coverage_warns_only_once(capsys):
    """同じ欠落列の組み合わせは2回目以降は警告しない（毎頭ごとのログ洪水防止）"""
    engine._XGB_MISSING_FEATS_WARNED = set()
    _check_xgb_feature_coverage({'f_a': 1.0}, ['f_a', 'f_b'])
    _check_xgb_feature_coverage({'f_a': 1.0}, ['f_a', 'f_b'])
    out = capsys.readouterr().out
    assert out.count('XGB特徴量') == 1


# ── XGB推論の例外フォールバック検知（無警告フォールバック事故の再発防止） ──
def test_warn_xgb_inference_fallback_warns_once(capsys):
    """XGB推論が例外で失敗しルールベースへフォールバックしたら警告を出す"""
    engine._XGB_INFERENCE_ERRORS_WARNED = set()
    _warn_xgb_inference_fallback('テスト馬', TypeError('boom'))
    out = capsys.readouterr().out
    assert 'テスト馬' in out
    assert 'XGB推論失敗' in out
    assert 'TypeError' in out


def test_warn_xgb_inference_fallback_dedupes_same_error(capsys):
    """同じ例外種別+メッセージは2回目以降は警告しない（毎頭ごとのログ洪水防止）"""
    engine._XGB_INFERENCE_ERRORS_WARNED = set()
    _warn_xgb_inference_fallback('馬A', TypeError('boom'))
    _warn_xgb_inference_fallback('馬B', TypeError('boom'))
    out = capsys.readouterr().out
    assert out.count('XGB推論失敗') == 1


def test_warn_xgb_inference_fallback_warns_again_for_different_error(capsys):
    """例外の種類が異なれば別途警告する"""
    engine._XGB_INFERENCE_ERRORS_WARNED = set()
    _warn_xgb_inference_fallback('馬A', TypeError('boom'))
    _warn_xgb_inference_fallback('馬B', ValueError('other'))
    out = capsys.readouterr().out
    assert out.count('XGB推論失敗') == 2


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
