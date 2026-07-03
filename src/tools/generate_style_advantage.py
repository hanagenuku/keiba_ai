"""
course_profiles.json の style_advantage を history.db の実データから更新するスクリプト。

使い方（Colab）:
    from src.tools.generate_style_advantage import generate_style_advantage
    generate_style_advantage(BASE_DIR)
    # → data/course_profiles.json の style_advantage が実データで上書きされる
"""

import os
import json
import sqlite3


def generate_style_advantage(base_dir, min_samples=30):
    """
    history.db の horse_history を集計して course_profiles.json の
    style_advantage（コース×脚質別3着内率）を更新する。

    Parameters
    ----------
    base_dir    : プロジェクトルート
    min_samples : 更新に必要な最低サンプル数（これ未満のコース×脚質はデフォルト値を維持）
    """
    db_path      = os.path.join(base_dir, 'data', 'history.db')
    profile_path = os.path.join(base_dir, 'data', 'course_profiles.json')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f'history.db が見つかりません: {db_path}')
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f'course_profiles.json が見つかりません: {profile_path}')

    # ── history.db 集計 ─────────────────────────────────────────────────
    _STYLE_NORM = {
        '逃げ': 'escape', '先行': 'front', '差し': 'stalk', '追込': 'closer',
    }

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT racecourse, surface, running_style,
               AVG(CASE WHEN place <= 3 THEN 1.0 ELSE 0.0 END) AS top3_rate,
               COUNT(*) AS n
        FROM horse_history
        WHERE running_style IS NOT NULL
          AND running_style != ''
          AND place > 0
          AND place < 99
        GROUP BY racecourse, surface, running_style
    """).fetchall()
    conn.close()

    # (racecourse, surface) → {style: top3_rate} （min_samples 以上のみ）
    stats = {}
    for rc, sf, rs, rate, n in rows:
        style = _STYLE_NORM.get(rs)
        if not style or not rc or not sf:
            continue
        if n < min_samples:
            continue
        key = f'{rc}_{sf}'
        stats.setdefault(key, {})[style] = round(float(rate), 3)

    # ── course_profiles.json 更新 ────────────────────────────────────────
    with open(profile_path, encoding='utf-8') as f:
        profiles = json.load(f)

    updated = 0
    for course_key, adv in stats.items():
        if course_key not in profiles.get('courses', {}):
            continue
        existing = profiles['courses'][course_key].get('style_advantage', {})
        merged = {**existing, **adv}   # 実データ優先、サンプル不足はデフォルト維持
        profiles['courses'][course_key]['style_advantage'] = merged
        updated += 1
        print(f'  {course_key}: {merged}')

    profiles['_generated_from_db'] = True

    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f'\n✅ course_profiles.json 更新完了: {updated} コース')
    return stats
