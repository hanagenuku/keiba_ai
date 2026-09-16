"""能力軸は既存データで「分かれて」いるか。

設計指示書は「馬能力を1スコアにまとめず、能力軸（スピード/持続/瞬発/スタミナ/
脚質/末脚/コース形状対応）に分解する。ただし人間が型を決め打ちしない。
既存データから定量化できるか検証する」と言っている。

その手前で答えるべき問いがある: **いま持っている列は本当に別々の軸か**。
全部が1つの「強さ」の言い換えなら、分解しても軸は増えない。

  ① 軸候補どうしの相関
  ② PCA: 第1主成分が分散の何%を占めるか（高いほど「1軸しかない」）

使い方: python3 noryoku/inv2.py <db_path> [n_races]
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.features.engine import init_engine, calc_features_for_xgb, add_relative_features
from noryoku.inv1 import build_race

# 設計指示書が挙げた能力軸に、いまの列を当てる（当たらない軸は空にする）
AXES = {
    'スピード（絶対時計）': ['f_speed_fig_avg', 'f_speed_fig_max', 'f_speed_avg', 'f_speed_max'],
    '持続（平均ペース維持）': ['f_finish_time_avg', 'f_time_diff_avg'],
    '瞬発（上がり）': ['f_late_speed', 'f_last1_3f', 'f_agari_ability'],
    'スタミナ': ['f_stamina_score', 'f_optimal_distance'],
    '脚質（位置取り）': ['f_pos_avg_3', 'f_p_front', 'f_p_back'],
    'コース形状対応': ['f_same_turn_rate', 'f_straight_match', 'f_uphill_match',
                 'f_agari_at_similar', 'f_course_type_rate'],
    '総合レーティング': ['f_pl_rating', 'f_recent'],
}


def run(db, n_races=400):
    init_engine('.')
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    ids = [r[0] for r in conn.execute(
        """SELECT race_id FROM race_history WHERE date >= '2026-01-01'
           ORDER BY race_id LIMIT ?""", (n_races,))]
    rows = []
    for rid in ids:
        race = build_race(conn, rid)
        if len(race['horses']) < 6:
            continue
        xs = []
        for h in race['horses']:
            try:
                xs.append(calc_features_for_xgb(h, race))
            except Exception:
                xs.append({})
        add_relative_features(xs)
        rows.extend(xs)
    conn.close()
    D = pd.DataFrame(rows)
    print(f'■ {len(ids)}レース / {len(D):,}頭\n')

    cols = [c for v in AXES.values() for c in v if c in D.columns]
    X = D[cols].apply(pd.to_numeric, errors='coerce')
    keep = [c for c in cols if X[c].notna().mean() > 0.5 and X[c].std() > 0]
    X = X[keep].dropna()
    print(f'欠損50%超/定数を除いて {len(keep)}列 / {len(X):,}頭が残った')
    drop = [c for c in cols if c not in keep]
    if drop:
        print(f'   除外: {" ".join(drop)}')

    # 向きを揃える（小さいほど良い列は符号反転）。相関の符号を読みやすくするだけ。
    LOWER_BETTER = {'f_finish_time_avg', 'f_time_diff_avg', 'f_late_speed',
                    'f_last1_3f', 'f_agari_at_similar', 'f_pos_avg_3'}
    Z = X.copy()
    for c in Z.columns:
        if c in LOWER_BETTER:
            Z[c] = -Z[c]
    Z = (Z - Z.mean()) / Z.std()

    print('\n■ 軸どうしの相関（各軸は列の平均）')
    axis_vals = {}
    for name, cs in AXES.items():
        cs = [c for c in cs if c in Z.columns]
        if cs:
            axis_vals[name] = Z[cs].mean(axis=1)
    A = pd.DataFrame(axis_vals)
    C = A.corr()
    names = list(C.columns)
    print('        ' + ' '.join(f'{i+1:>6d}' for i in range(len(names))))
    for i, n in enumerate(names):
        print(f'{i+1} {n:14s}' + ' '.join(f'{C.iloc[i, j]:+6.2f}' for j in range(len(names))))

    off = C.values[np.triu_indices(len(names), 1)]
    print(f'\n   軸間相関の |中央値| {np.median(np.abs(off)):.2f}   最大 {np.abs(off).max():.2f}')

    print('\n■ PCA（軸の平均ベクトル・標準化済み）')
    M = (A - A.mean()) / A.std()
    ev = np.linalg.eigvalsh(np.cov(M.dropna().values.T))[::-1]
    r = ev / ev.sum()
    print('   寄与率 ' + ' '.join(f'{x:.0%}' for x in r))
    print(f'   第1主成分が {r[0]:.0%} を占める  '
          f'{"→ 実質1軸" if r[0] > 0.6 else "→ 複数軸ある"}')
    print(f'   第1・第2で {r[0] + r[1]:.0%}')

    print('\n■ 生の列どうし（軸内の重複を見る）')
    CC = Z.corr()
    pairs = [(abs(CC.iloc[i, j]), CC.iloc[i, j], CC.index[i], CC.columns[j])
             for i in range(len(CC)) for j in range(i + 1, len(CC))]
    pairs.sort(reverse=True)
    for a, v, x, y in pairs[:8]:
        print(f'   {v:+.2f}  {x} × {y}')


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
