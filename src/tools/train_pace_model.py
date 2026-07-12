"""
展開予測モデル（ペース分類器）の学習スクリプト

history.db のレースデータからペース（slow/mid/high）を予測する
XGBClassifier を学習し、data/pace_model.pkl に保存する。

従来の8特徴量に加え、枠順×脚質・騎手ペースメイク傾向・コース特性・
ペース耐性を追加した強化版。

使い方（Colab）:
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.train_pace_model import train_pace_model
    result = train_pace_model(BASE_DIR)
"""

import os
import json
import math
import pickle
import sqlite3

# ペースラベル: first_3f を距離で正規化し分類
# first_3f / (distance/1000) で 1000m あたりの前半3Fペースを算出
# 閾値は JRA の平均的なペース分布に基づく
_PACE_THRESHOLDS = {
    '芝':   {'high': 34.5, 'slow': 36.0},
    'ダート': {'high': 36.0, 'slow': 37.5},
}


def _classify_pace(first_3f, distance, surface, pace_label=None):
    """前半3Fからペースを3分類する。pace_label が明示されていればそれを優先。"""
    if pace_label and pace_label in ('slow', 'mid', 'high'):
        return pace_label
    if pace_label in ('S', 'スロー'):
        return 'slow'
    if pace_label in ('H', 'ハイ'):
        return 'high'
    if pace_label in ('M', 'ミドル'):
        return 'mid'

    if not first_3f or first_3f <= 0 or not distance or distance <= 0:
        return None
    # 距離で正規化（1200m基準）
    ratio = 1200.0 / max(distance, 800)
    normalized = first_3f * ratio
    th = _PACE_THRESHOLDS.get(surface, _PACE_THRESHOLDS['芝'])
    if normalized <= th['high']:
        return 'high'
    elif normalized >= th['slow']:
        return 'slow'
    return 'mid'


def _build_jockey_pace_stats(conn):
    """騎手ごとの逃げ時ペースメイク傾向を算出する。

    Returns: dict[str, dict] — {jockey_name: {median_first3f_norm, n_escape, escape_rate}}
    """
    rows = conn.execute("""
        SELECT h.jockey, h.running_style, r.first_3f, r.distance, r.surface
        FROM horse_history h
        JOIN race_history r ON h.race_id = r.race_id
        WHERE h.jockey IS NOT NULL
          AND r.first_3f > 0
          AND r.distance > 0
          AND h.place IS NOT NULL AND h.place < 99
    """).fetchall()

    from collections import defaultdict
    jockey_escape_3f = defaultdict(list)   # 逃げ時の正規化前半3F
    jockey_total = defaultdict(int)        # 全騎乗数

    for row in rows:
        jockey = (row['jockey'] or '').replace(' ', '').replace('　', '')
        if not jockey:
            continue
        jockey_total[jockey] += 1
        rs = row['running_style'] or ''
        if rs in ('逃げ', 'escape'):
            ratio = 1200.0 / max(row['distance'], 800)
            norm_3f = row['first_3f'] * ratio
            jockey_escape_3f[jockey].append(norm_3f)

    stats = {}
    for jockey, vals in jockey_escape_3f.items():
        if len(vals) < 3:
            continue
        vals_sorted = sorted(vals)
        median_val = vals_sorted[len(vals_sorted) // 2]
        stats[jockey] = {
            'median_first3f_norm': round(median_val, 2),
            'n_escape': len(vals),
            'escape_rate': round(len(vals) / max(jockey_total[jockey], 1), 3),
        }
    return stats


def _build_features(conn, course_profiles, jockey_pace_stats):
    """history.db から全レースの展開予測特徴量を構築する。"""
    import numpy as np

    races = conn.execute("""
        SELECT r.race_id, r.date, r.racecourse, r.distance, r.surface,
               r.first_3f, r.pace_label, r.track_condition,
               r.num_finishers
        FROM race_history r
        WHERE r.distance > 0
          AND r.surface IN ('芝', 'ダート')
        ORDER BY r.date
    """).fetchall()

    feature_rows = []
    labels = []

    for race_row in races:
        race_id = race_row['race_id']
        distance = int(race_row['distance'] or 1600)
        surface = race_row['surface'] or '芝'
        racecourse = race_row['racecourse'] or ''
        first_3f = float(race_row['first_3f'] or 0)
        condition = race_row['track_condition'] or '良'

        pace = _classify_pace(first_3f, distance, surface, race_row['pace_label'])
        if pace is None:
            continue

        horses = conn.execute("""
            SELECT horse_num, running_style, jockey, agari3f,
                   COALESCE(bracket, 0) AS bracket,
                   COALESCE(win_odds, 0) AS win_odds,
                   COALESCE(popularity, 0) AS popularity
            FROM horse_history
            WHERE race_id = ?
            ORDER BY horse_num
        """, (race_id,)).fetchall()

        if len(horses) < 3:
            continue

        n = len(horses)

        # --- 既存特徴量 ---
        escape_count = 0
        front_count = 0
        agari_vals = []
        for h in horses:
            rs = h['running_style'] or ''
            if rs in ('逃げ', 'escape'):
                escape_count += 1
            elif rs in ('先行', 'front'):
                front_count += 1
            a3f = h['agari3f']
            if a3f and float(a3f) > 0:
                agari_vals.append(float(a3f))

        front_density = (escape_count + front_count) / n
        avg_agari = sum(agari_vals) / len(agari_vals) if agari_vals else 36.0
        std_agari = (sum((x - avg_agari) ** 2 for x in agari_vals) / len(agari_vals)) ** 0.5 if len(agari_vals) > 1 else 1.5
        surface_num = 1 if surface == '芝' else 0

        # --- 新特徴量 4: 逃げ馬の枠順分布 ---
        escape_positions = []
        for h in horses:
            rs = h['running_style'] or ''
            if rs in ('逃げ', 'escape'):
                pos = int(h['horse_num'] or 8)
                escape_positions.append(pos)

        if escape_positions:
            escape_avg_pos = sum(escape_positions) / len(escape_positions)
            escape_outer_ratio = sum(1 for p in escape_positions if p > n * 0.6) / len(escape_positions)
        else:
            escape_avg_pos = n / 2
            escape_outer_ratio = 0.0

        # --- 新特徴量 5: 逃げ馬のペース耐性 ---
        # 逃げ馬が過去にハイペースでも好走したか（history.db から算出済みの
        # ペース耐性は build_training_data 経由では使えないため、簡易指標を使う）
        # → 逃げ馬の平均人気（人気があるほど実力がある＝ペース耐性が高い）
        escape_pop = []
        for h in horses:
            rs = h['running_style'] or ''
            if rs in ('逃げ', 'escape'):
                pop = int(h['popularity'] or 0)
                if 0 < pop < 99:
                    escape_pop.append(pop)
        escape_avg_pop = sum(escape_pop) / len(escape_pop) if escape_pop else 8.0

        # --- 新特徴量 6: コース特性 ---
        course_key = f'{racecourse}_{surface}'
        profile = course_profiles.get(course_key, {})
        straight_length = profile.get('straight_length', 350)
        has_uphill = 1 if profile.get('has_uphill', False) else 0
        straight_class = {'very_long': 4, 'long': 3, 'medium': 2, 'short': 1}.get(
            profile.get('straight_class', 'medium'), 2
        )

        # コーナー数（距離とコースから推定）
        if distance <= 1400:
            n_corners = 2 if straight_length < 500 else 1
        elif distance <= 2000:
            n_corners = 3 if straight_length < 400 else 2
        else:
            n_corners = 4

        # --- 新特徴量 7: 騎手のペースメイク傾向 ---
        jockey_pace_vals = []
        for h in horses:
            rs = h['running_style'] or ''
            if rs in ('逃げ', 'escape'):
                jn = (h['jockey'] or '').replace(' ', '').replace('　', '')
                js = jockey_pace_stats.get(jn, {})
                if 'median_first3f_norm' in js:
                    jockey_pace_vals.append(js['median_first3f_norm'])

        jockey_pace_median = (
            sum(jockey_pace_vals) / len(jockey_pace_vals)
            if jockey_pace_vals else 35.5
        )
        jockey_escape_pct = 0.0
        for h in horses:
            jn = (h['jockey'] or '').replace(' ', '').replace('　', '')
            js = jockey_pace_stats.get(jn, {})
            jockey_escape_pct += js.get('escape_rate', 0.05)
        jockey_escape_pct /= max(n, 1)

        # 馬場状態（重/不良は前有利＝スローになりにくい）
        condition_num = {'良': 0, '稍重': 1, '重': 2, '不良': 3}.get(condition, 0)

        feature_rows.append({
            # 既存8特徴量
            'escape_count': escape_count,
            'front_count': front_count,
            'front_density': round(front_density, 3),
            'avg_agari3f': round(avg_agari, 2),
            'std_agari3f': round(std_agari, 2),
            'runner_count': n,
            'distance': distance,
            'surface_num': surface_num,
            # 新特徴量: 枠順×脚質
            'escape_avg_pos': round(escape_avg_pos, 2),
            'escape_outer_ratio': round(escape_outer_ratio, 3),
            # 新特徴量: ペース耐性
            'escape_avg_pop': round(escape_avg_pop, 2),
            # 新特徴量: コース特性
            'straight_length': straight_length,
            'straight_class': straight_class,
            'has_uphill': has_uphill,
            'n_corners': n_corners,
            # 新特徴量: 騎手ペースメイク
            'jockey_pace_median': round(jockey_pace_median, 2),
            'jockey_escape_pct': round(jockey_escape_pct, 4),
            # 馬場状態
            'condition_num': condition_num,
        })
        labels.append(pace)

    return feature_rows, labels


def train_pace_model(base_dir,
                     train_end='2026-03-31',
                     val_start='2026-04-01',
                     val_end=None,
                     n_estimators=300,
                     max_depth=5):
    """
    展開予測モデルを学習する。

    Returns
    -------
    dict: accuracy, confusion_matrix 等
    """
    import pandas as pd
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    db_path = os.path.join(base_dir, 'data', 'history.db')
    profiles_path = os.path.join(base_dir, 'data', 'course_profiles.json')
    model_path = os.path.join(base_dir, 'data', 'pace_model.pkl')
    bak_path = os.path.join(base_dir, 'data', 'pace_model_old.pkl')
    stats_path = os.path.join(base_dir, 'data', 'jockey_pace_stats.json')

    # コースプロファイル読み込み
    course_profiles = {}
    if os.path.exists(profiles_path):
        with open(profiles_path, encoding='utf-8') as f:
            cp = json.load(f)
            course_profiles = cp.get('courses', {})

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 騎手ペースメイク統計を構築
    print('騎手ペースメイク統計を構築中...')
    jockey_pace_stats = _build_jockey_pace_stats(conn)
    print(f'  逃げ騎手: {len(jockey_pace_stats)} 名')

    # 統計を保存（推論時に使用）
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(jockey_pace_stats, f, ensure_ascii=False, indent=2)
    print(f'  保存: {stats_path}')

    # 特徴量構築
    print('特徴量を構築中...')
    feature_rows, labels = _build_features(conn, course_profiles, jockey_pace_stats)
    conn.close()

    if not feature_rows:
        raise ValueError('学習データが0件です。history.db の first_3f を確認してください。')

    df = pd.DataFrame(feature_rows)
    df['pace'] = labels
    print(f'全データ: {len(df)} レース')
    print(f'ペース分布: {df["pace"].value_counts().to_dict()}')

    # 日付フィルタ用に race_id からdate推定が難しいので、indexで分割
    # → train_end/val_start は build_features 内で date ORDER されているため
    #   全データの70%をtrain、30%をvalにする（簡易分割）
    # ただし train_end が指定されている場合は date ベースで分割
    # feature_rows には date がないので、race_history から取得し直す必要がある
    # → 簡易実装として比率分割を使う
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    print(f'Train: {len(train_df)} レース')
    print(f'Val  : {len(val_df)} レース')

    # ラベルエンコード
    le = LabelEncoder()
    le.fit(['high', 'mid', 'slow'])

    feat_cols = [c for c in df.columns if c != 'pace']
    X_train = train_df[feat_cols].fillna(0)
    y_train = le.transform(train_df['pace'])
    X_val = val_df[feat_cols].fillna(0)
    y_val = le.transform(val_df['pace'])

    print(f'特徴量数: {len(feat_cols)}')
    print(f'Train ペース分布: {dict(zip(*np.unique(y_train, return_counts=True)))}')

    # XGB学習
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # 評価
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    report = classification_report(y_val, val_pred,
                                   target_names=le.classes_,
                                   output_dict=True)

    print(f'\n── Val 評価 ──')
    print(f'  Accuracy: {acc:.4f}')
    print(classification_report(y_val, val_pred, target_names=le.classes_))

    # 旧モデルと比較
    old_acc = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                old_model = pickle.load(f)
            # 旧モデルの特徴量列数が異なる場合があるため、共通列で評価
            old_cols = list(getattr(old_model, 'feature_names_in_', feat_cols[:8]))
            old_X = X_val.reindex(columns=old_cols, fill_value=0)
            old_pred = old_model.predict(old_X)
            old_acc = accuracy_score(y_val, old_pred)
            print(f'  旧モデル Accuracy: {old_acc:.4f}  '
                  f'({"↑改善" if acc > old_acc else "↓悪化"})')
        except Exception as e:
            print(f'  旧モデル比較スキップ: {e}')

    # 保存
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, bak_path)
        print(f'旧モデル退避: {bak_path}')

    # LabelEncoder を model に添付（推論時に classes_ が必要）
    model._pace_label_encoder = le
    model._pace_feature_cols = feat_cols

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'新モデル保存: {model_path}')

    # 特徴量重要度
    importances = sorted(zip(feat_cols, model.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print('\n── 特徴量重要度 ──')
    for name, imp in importances:
        print(f'  {name:<25} {imp*100:.2f}%')

    return {
        'accuracy': round(acc, 4),
        'old_accuracy': round(old_acc, 4) if old_acc else None,
        'n_features': len(feat_cols),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'report': report,
        'feature_importance': {name: round(float(imp), 4) for name, imp in importances},
    }


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    train_pace_model(base)
