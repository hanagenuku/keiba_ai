"""展開予想（隊列・ペース）の2段モデルを学習する。

段1: 各馬が最初のコーナーを先頭で通過する確率 P(lead)  — ロジスティック
段2: レースの前半600m（実数・端数区間を揃えた量）          — XGBoost 回帰

3分類（high/mid/slow）は段2の予測値と距離帯×表面のパーセンタイル閾値から
**正規分布で導く**（ハードなクラスではなく確率を返すため）。

なぜ作り直したか / 実測は tenkai/RESULTS.md と src/features/race_shape.py を参照。
要点だけ:
  - 旧モデルは学習時に「そのレースの実際の脚質・実測agari3f」を見て、
    推論時は「過去走からの推定・定数36.0」を渡されていた（一致率 39.8%）
  - 3分類 学習時入力 52.80% / **推論時入力 50.45%** / 揃えると 52.88%
  - 目的変数 `first_3f` も壊れていた（端数区間の有無で同じ距離が二峰・差 5.48s）。
    `early_pace_seconds()` で揃えると検定期 RMSE 2.2590s → 0.8459s
  - 揃えた目的変数では**レース構成の特徴量は足すと悪化する**（2窓とも）ので、
    段2は**レース条件だけ**を使う。段1の P(先頭) はペースには使わず、
    馬ごとの量として本体モデルの A/B に回す
"""
import json
import math
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd

from src.features.race_shape import (
    summarize_horse_shape, race_shape_features, first_corner_position,
    early_pace_seconds, pace_model_inputs, dist_zone,
    PACE_INPUT_COLS, RACE_FEATURE_COLS,
)

LEAD_COLS = ['p_led', 'p_mean', 'p_min', 'p_last', 'p_std', 'p_n',
             'd_dist', 'draw', 'fs', 'j_led', 'dist']
# 🔑 ペース（前半600m）は**レース条件だけ**で決める。
#    出走馬の構成（P(先頭)の分布・過去の通過位置・過去の上がり・逃げ馬の人気）は
#    2窓とも足すと悪化した（2026-09-14 実測。詳細は tenkai/RESULTS.md）:
#        条件だけ 窓A 0.8272s / 窓B 0.8325s
#        + P(先頭)由来   +0.0145 / +0.0077
#        + 過去の通過位置  +0.0121 / +0.0167
#        + 過去の上がり   +0.0130 / +0.0051
#        + 全部        +0.0151 / +0.0210
#    ✅ 副産物として、条件だけなら**学習時と推論時の入力が同一**になり、
#       旧モデルのパリティ違反（実際の脚質 vs 推定脚質・一致率39.8%）が構造的に消える。
#    ⚠ cond（馬場状態）は**推論時つねに「良」**（出馬表に載っていない既知の
#      パリティ違反）。古い違反を直しながら新しい違反を入れないよう外す。
#      外すコストは RMSE +0.011s だけ（0.8292→0.8404 / 0.8316→0.8417）。
#      web から実際の馬場が取れるようになったら戻せる。
#    🔑 列の中身は race_shape.pace_model_inputs() が作る。**推論も同じ関数**を通す。
PACE_COLS = PACE_INPUT_COLS
MIN_JOCKEY_RUNS = 30


def _load(db_path):
    con = sqlite3.connect(db_path)
    H = pd.read_sql("""SELECT h.race_id, h.date, h.horse_name, h.horse_num, h.corner_all,
                              h.agari3f, h.popularity pop, h.jockey, h.distance dist,
                              h.place, r.num_finishers fs
                       FROM horse_history h LEFT JOIN race_history r USING(race_id)
                       WHERE h.place > 0 AND h.place < 99""", con)
    R = pd.read_sql("""SELECT race_id, date, racecourse rc, surface sf, distance dist,
                              first_3f, lap_times, track_condition tc, race_class
                       FROM race_history WHERE lap_times IS NOT NULL""", con)
    cnt = pd.read_sql('SELECT race_id, COUNT(*) n_horses FROM horse_history '
                      'WHERE place > 0 GROUP BY race_id', con)
    con.close()
    R = R.merge(cnt, on='race_id', how='left')
    # 🔴 first_3f は端数区間の有無で同じ距離でも 5秒以上ずれる。揃えた量を使う。
    R['early'] = [early_pace_seconds(l, d) for l, d in zip(R.lap_times, R.dist)]
    R = R[R.early.notna()].copy()
    H['fs'] = H.fs.fillna(H.groupby('race_id')['horse_num'].transform('max'))
    H = H[H.fs.between(5, 18)].copy()
    H['pos_n'] = [first_corner_position(c, f) for c, f in zip(H.corner_all, H.fs)]
    H = H[H.pos_n.notna()].copy()
    H['led'] = (H.pos_n == 0).astype(int)
    return H.sort_values(['horse_name', 'date']).reset_index(drop=True), R


def _horse_features(H):
    """各行に『その馬のそれ以前の走り』から作ったサマリを付ける（shift で当該レース除外）。"""
    g = H.groupby('horse_name')
    H['p_led'] = g['led'].transform(lambda s: s.shift().expanding().mean())
    H['p_mean'] = g['pos_n'].transform(lambda s: s.shift().expanding().mean())
    H['p_min'] = g['pos_n'].transform(lambda s: s.shift().expanding().min())
    H['p_last'] = g['pos_n'].transform(lambda s: s.shift())
    H['p_std'] = g['pos_n'].transform(lambda s: s.shift().expanding().std())
    H['p_ag'] = g['agari3f'].transform(lambda s: s.shift().expanding().mean())
    H['p_n'] = g['led'].transform(lambda s: s.shift().expanding().count())
    H['d_dist'] = H.dist - g['dist'].transform(lambda s: s.shift())
    H = H.sort_values(['jockey', 'date'])
    H['j_led'] = H.groupby('jockey')['led'].transform(lambda s: s.shift().expanding().mean())
    H['draw'] = H.horse_num / H.fs
    return H.sort_values('date').reset_index(drop=True)


def train_race_shape(base_dir, train_end='2026-06-30', seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    H, R = _load(os.path.join(base_dir, 'data', 'history.db'))
    H = _horse_features(H)
    tr = H[H.date <= train_end]
    print(f'段1 学習 {len(tr):,}頭 / 全 {len(H):,}頭  先頭通過率 {100*H.led.mean():.1f}%')

    med = tr[LEAD_COLS].median()
    X = tr[LEAD_COLS].fillna(med)
    mu, sd = X.mean(), X.std().replace(0, 1)
    lm = LogisticRegression(max_iter=3000).fit((X - mu) / sd, tr.led)
    te = H[H.date > train_end]
    auc = None
    if len(te) > 500:
        Z = (te[LEAD_COLS].fillna(med) - mu) / sd
        auc = roc_auc_score(te.led, lm.predict_proba(Z)[:, 1])
        print(f'   P(先頭に立つ) OOS AUC {auc:.4f}  (検定 {len(te):,}頭)')

    H['P'] = lm.predict_proba((H[LEAD_COLS].fillna(med) - mu) / sd)[:, 1]

    # 段2: レース単位に集約（推論と同じ race_shape の関数を通す）
    rows = []
    for rid, s in H.groupby('race_id'):
        summ = [{'p_led': r.p_led, 'p_mean': r.p_mean, 'p_min': r.p_min,
                 'p_last': r.p_last, 'p_std': r.p_std, 'p_ag': r.p_ag, 'p_n': r.p_n}
                for r in s.itertuples()]
        f = race_shape_features(summ, list(s.P), list(s['pop']))
        f['race_id'] = rid
        rows.append(f)
    F = pd.DataFrame(rows)

    d = R.merge(F, on='race_id', how='inner', suffixes=('', '_shape'))
    # 🔑 推論と同じ関数で列を作る（engine 側も pace_model_inputs を呼ぶ）
    pin = pd.DataFrame([pace_model_inputs(dist, sf, cls, nh) for dist, sf, cls, nh
                        in zip(d.dist, d.sf, d.race_class, d.n_horses)])
    for c in PACE_COLS:
        d[c] = pin[c].values
    d['dz'] = d.dist.map(dist_zone)
    trd = d[d.date <= train_end]
    ted = d[d.date > train_end]
    print(f'段2 学習 {len(trd):,}レース / 検定 {len(ted):,}レース')

    m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, random_state=seed,
                         verbosity=0)
    m.fit(trd[PACE_COLS].fillna(-1), trd.early)
    # 🔑 sigma は学習残差ではなく OOS 残差で決める。学習残差だと過信する
    #    （実測 学習 0.60s / OOS 1.5s で 2.5倍違った）
    sigma = float((trd.early - m.predict(trd[PACE_COLS].fillna(-1))).std())
    rmse = None
    if len(ted) > 50:
        p = m.predict(ted[PACE_COLS].fillna(-1))
        rmse = float(np.sqrt(((p - ted.early) ** 2).mean()))
        base = float(np.sqrt(((trd.early.mean() - ted.early) ** 2).mean()))
        sigma = rmse                                    # 過信しないよう OOS を採る
        print(f'   前半600m OOS RMSE {rmse:.4f}s  (学習平均を答える {base:.4f}s'
              f' / 改善 {base - rmse:+.4f}s)')

    # 3分類の閾値（距離帯×表面のパーセンタイル）。学習期だけで決める
    th = {}
    for (sf, dz), g in trd.groupby(['sf', 'dz']):
        if len(g) >= 30:
            th[f'{sf}|{dz}'] = [float(g.early.quantile(.33)),
                                float(g.early.quantile(.67))]

    out = {
        'version': 1,
        'lead': {'cols': LEAD_COLS, 'coef': lm.coef_[0].tolist(),
                 'intercept': float(lm.intercept_[0]),
                 'mu': mu.tolist(), 'sd': sd.tolist(), 'median': med.tolist(),
                 'auc': auc},
        'pace_model': m, 'pace_cols': PACE_COLS,
        'sigma': sigma, 'thresholds': th,
        'meta': {'train_end': train_end, 'n_horses': int(len(tr)),
                 'n_races': int(len(trd)), 'rmse': rmse, 'seed': seed},
    }
    path = os.path.join(base_dir, 'data', 'race_shape_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    print(f'✅ {path}')
    return out


if __name__ == '__main__':
    import sys
    train_race_shape(sys.argv[1] if len(sys.argv) > 1 else '.')
