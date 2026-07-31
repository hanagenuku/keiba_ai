"""残差学習モデルのability_margin（市場非依存のAI評価）をSHAP寄与度で
人間可読カテゴリに分解する説明可能性(Explainability)レイヤー。

ユーザーの「AIが独自に予想を立てた後、オッズを見て“なぜこの馬は人気があるのに
AIは評価が低いのか”を要因分解して考える仕組みが欲しい」という要望に対応する。
モデル自体・学習ロジックには一切触れず、既に学習済みのxgb.Boosterに対して
推論後の解釈を追加するだけの層。
"""

import xgboost as xgb

# 特徴量名 → 人間可読カテゴリ のマッピング。xgb_feature_cols.json の138列を
# 網羅する（新しい特徴量を追加した場合はここにも追記すること。未登録の列は
# 自動的に「その他」カテゴリに割り当てられ、集計から漏れることはない）。
FEATURE_CATEGORY_MAP = {
    # 騎手・厩舎
    'f_jockey': '騎手', 'f_jockey_rate': '騎手', 'f_trainer': '厩舎',
    'cl_f_jockey_rank': '騎手', 'cl_f_jockey_vs_field': '騎手',
    'cl_f_trainer_rank': '厩舎', 'cl_f_trainer_vs_field': '厩舎',
    # 距離適性
    'f_dist_fukusho': '距離適性', 'cl_f_dist_fukusho_rank': '距離適性',
    'cl_f_dist_fukusho_vs_field': '距離適性', 'f_optimal_distance': '距離適性',
    'f_dist_vs_optimal': '距離適性', 'f_dist_change': '距離適性',
    'f_speed_x_shortening': '距離適性', 'f_stamina_x_extension': '距離適性',
    'f_dist_confidence': '距離適性',
    # コース適性
    'f_course_fukusho': 'コース適性', 'cl_f_course_fukusho_rank': 'コース適性',
    'cl_f_course_fukusho_vs_field': 'コース適性', 'f_same_course_rate': 'コース適性',
    'f_same_turn_rate': 'コース適性', 'f_straight_match': 'コース適性',
    'f_uphill_match': 'コース適性', 'f_agari_at_similar': 'コース適性',
    'f_course_coverage': 'コース適性', 'f_course_type_rate': 'コース適性',
    'f_tight_vs_spacious': 'コース適性', 'f_uphill_severity_rate': 'コース適性',
    'f_corner_position_change': 'コース適性', 'f_agari_rank_at_type': 'コース適性',
    'f_dirt_turf_start': 'コース適性', 'f_course_hill_diff': 'コース適性',
    'f_course_corner_tight': 'コース適性',
    # ペース・脚質
    'f_p_front': 'ペース適性', 'f_p_mid': 'ペース適性', 'f_p_back': 'ペース適性',
    'f_pace': 'ペース適性', 'f_perf_highpace': 'ペース適性', 'f_perf_slowpace': 'ペース適性',
    'f_pace_fit': 'ペース適性', 'f_pace_prob_fast': 'ペース適性',
    'f_pace_prob_slow': 'ペース適性', 'f_pace_x_style': 'ペース適性',
    'f_style_course_fit': 'ペース適性', 'f_style_total_fit': 'ペース適性',
    'cl_f_style_course_fit_rank': 'ペース適性', 'cl_f_style_course_fit_vs_field': 'ペース適性',
    'cl_f_pace_fit_rank': 'ペース適性', 'cl_f_pace_fit_vs_field': 'ペース適性',
    'cl_f_style_total_fit_rank': 'ペース適性', 'cl_f_style_total_fit_vs_field': 'ペース適性',
    # スピード能力
    'f_speed_avg': 'スピード能力', 'f_speed_max': 'スピード能力', 'f_speed_last': 'スピード能力',
    'f_speed_trend': 'スピード能力', 'f_speed_fig_last': 'スピード能力',
    'f_speed_fig_avg': 'スピード能力', 'f_speed_fig_max': 'スピード能力',
    'rl_f_speed_avg_rank': 'スピード能力', 'rl_f_speed_avg_vs_field': 'スピード能力',
    'rl_f_speed_last_rank': 'スピード能力', 'rl_f_speed_last_vs_field': 'スピード能力',
    'rl_f_speed_fig_last_rank': 'スピード能力', 'rl_f_speed_fig_last_vs_field': 'スピード能力',
    'rl_f_speed_fig_avg_rank': 'スピード能力', 'rl_f_speed_fig_avg_vs_field': 'スピード能力',
    'rl_f_speed_fig_max_rank': 'スピード能力', 'rl_f_speed_fig_max_vs_field': 'スピード能力',
    'f_agari_ability': 'スピード能力', 'f_stamina_score': 'スピード能力',
    'f_speed_score': 'スピード能力', 'f_late_speed': 'スピード能力', 'f_early_speed': 'スピード能力',
    'rl_f_late_speed_rank': 'スピード能力', 'rl_f_late_speed_vs_field': 'スピード能力',
    'f_last1_3f': 'スピード能力', 'f_last2_3f': 'スピード能力', 'f_last3_3f': 'スピード能力',
    # 近走成績
    'f_pos_avg_3': '近走成績', 'f_pos_std_3': '近走成績',
    'f_last1_pos3c': '近走成績', 'f_last2_pos3c': '近走成績', 'f_last3_pos3c': '近走成績',
    'f_recent': '近走成績', 'f_recent_fukusho': '近走成績',
    'rl_f_recent_rank': '近走成績', 'rl_f_recent_vs_field': '近走成績',
    'rl_f_recent_fukusho_rank': '近走成績', 'rl_f_recent_fukusho_vs_field': '近走成績',
    'f_last1_rank': '近走成績', 'f_last2_rank': '近走成績', 'f_last3_rank': '近走成績',
    'f_career_runs': '近走成績', 'f_is_debut': '近走成績',
    'f_finish_time_avg': '近走成績', 'f_time_diff_avg': '近走成績',
    'rl_f_finish_time_rank': '近走成績', 'rl_f_finish_time_vs_field': '近走成績',
    'rl_f_time_diff_rank': '近走成績', 'rl_f_time_diff_vs_field': '近走成績',
    # クラス・メンバーレベル
    'f_class_level': 'クラス適性', 'f_class_jump': 'クラス適性',
    # f_member_level_* は未来参照リークのため2026-07-28に特徴量から削除
    # （旧モデルを読み込んだ場合に備え、キー自体は残さない）
    'f_competitiveness': 'クラス適性', 'f_competitive_best': 'クラス適性',
    'f_cl_rank': 'クラス適性', 'f_rl_rank': 'クラス適性',
    # 血統
    'f_blood': '血統', 'cl_f_blood_rank': '血統', 'cl_f_blood_vs_field': '血統',
    # 馬体・斤量
    'f_weight_load': '斤量・馬体', 'cl_f_weight_load_rank': '斤量・馬体',
    'cl_f_weight_load_vs_field': '斤量・馬体', 'f_sex': '斤量・馬体', 'f_age': '斤量・馬体',
    # 馬体重推移（2026-07-24追加）。旧モデルは追加前日の学習で列を持たず、
    # 2026-07-31の再学習で初めて本番入りしたためカテゴリ登録が漏れていた。
    'f_weight_trend_avg': '斤量・馬体', 'f_weight_last_diff': '斤量・馬体',
    # 枠順
    'f_post': '枠順',
    # 馬場適性
    'f_track_cond': '馬場適性', 'f_heavy_track_rate': '馬場適性',
    'cl_f_heavy_track_rank': '馬場適性', 'cl_f_heavy_track_vs_field': '馬場適性',
    # ローテーション
    'f_days_since_last': 'ローテーション', 'f_interval_score': 'ローテーション',
    # AIの自己反省的補正（過去の予想外れの蓄積から学習する項目）
    # f_pred_gap_* は「乖離が繰り返す」前提が実データで否定され2026-07-28に削除
    # （カテゴリ「AI自己補正」は該当列がなくなったため消滅）
    # 過去の人気推移（市場情報だが「現在の」オッズではなく履歴のため残差モデルの
    # 特徴量として残置されている。f_popularity（現在の人気）自体は除外済み）
    'f_pop_last': '過去人気推移', 'f_pop_avg': '過去人気推移',
    'f_beat_market_rate': '過去人気推移',
}

# 意図的に廃止した特徴量。旧モデルの xgb_feature_cols.json にはまだ残るため、
# カテゴリ網羅チェックの対象外にする（次回の再学習で列自体が消える）。
REMOVED_FEATURE_COLS = {
    # 対戦相手の「その後」の成績を使う未来参照リークのため2026-07-28に削除
    'f_member_level_avg', 'f_member_level_max', 'f_member_level_last',
    'rl_f_member_level_avg_rank', 'rl_f_member_level_avg_vs_field',
    'rl_f_member_level_last_rank', 'rl_f_member_level_last_vs_field',
    # 「AIの予測乖離は繰り返す」という前提が実データで否定され2026-07-28に削除
    'f_pred_gap_avg', 'f_pred_gap_worst', 'f_pred_gap_consistency',
    'rl_f_pred_gap_avg_rank', 'rl_f_pred_gap_avg_vs_field',
    'rl_f_pred_gap_worst_rank', 'rl_f_pred_gap_worst_vs_field',
}

BASELINE_LABEL = '基準値'


def compute_ability_breakdown(booster, X_pred, feature_cols, base_margin):
    """1頭ぶんの ability_margin（raw_margin - base_margin）をカテゴリ別寄与度に分解する。

    XGBoostのTreeSHAP（`pred_contribs=True`）は特徴量ごとの寄与度+バイアス項を返し、
    その合計は常にraw_margin（base_margin込みのモデル出力）と厳密に一致する
    （`sum(pred_contribs) == raw_margin`）。バイアス項からbase_marginを差し引いた
    残りを「木モデルの基準値」として扱うことで、特徴量ごとの寄与度の合計が
    ability_margin（= raw_margin - base_margin）とほぼ一致するように分解する。

    Returns: [{'category': str, 'contrib': float}, ...]  contrib降順。
    失敗時（モデル未ロード等）は空リストを返す。
    """
    dmat = xgb.DMatrix(X_pred, feature_names=list(feature_cols))
    dmat.set_base_margin([base_margin])
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    feature_contribs = contribs[:-1]
    tree_bias = float(contribs[-1]) - base_margin

    totals = {}
    for name, val in zip(feature_cols, feature_contribs):
        cat = FEATURE_CATEGORY_MAP.get(name, 'その他')
        totals[cat] = totals.get(cat, 0.0) + float(val)
    if abs(tree_bias) > 1e-6:
        totals[BASELINE_LABEL] = totals.get(BASELINE_LABEL, 0.0) + tree_bias

    breakdown = [{'category': k, 'contrib': round(v, 4)} for k, v in totals.items()]
    breakdown.sort(key=lambda x: x['contrib'], reverse=True)
    return breakdown
