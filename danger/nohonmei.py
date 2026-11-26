#!/usr/bin/env python3
"""「1番人気なのに◎が付いていない馬」は消せるか。

🔴 **2026-11-26: この計測器は本番の `rl_rank` を再現していない。そのまま使い回さない。**
   本番は `engine.calc_all` で
       total = 残差モデル × (1-RELATIVE_WEIGHT) + f_relative_score × RELATIVE_WEIGHT
       win_prob = softmax(total, 3.5) → rl_rank = win_prob の降順
   だが、ここは `rl_rank = 残差モデル margin の降順` にしていた。そのため
   対象（1番人気 かつ rl_rank>=2）が本番 11.6% に対し **0.47%（14頭）**しか出ず、
   測定不能で終わった。経緯は danger/RESULTS_NOHONMEI.md。

事前登録カード: danger/CRITERIA_NOHONMEI.md（コミット済み）

🔴 **測定は1回**。窓・指標・合格線はカードのとおりで、結果を見てから変えない。
🔴 中止条件を**率より手前で**印字して判定する。
🔴 本番の設定は1つも読み書きしない（読むのは horse_features.csv と history.db だけ）。

使い方:
    python3 danger/nohonmei.py <horse_features.csv>
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

# 🔑 学習・base_margin は logic2026/eval.py の実装をそのまま import する
#    （同じ量を2か所で作らない）。
from logic2026.eval import (  # noqa: E402
    _base_margin, _feat_cols, _xgb_params, N_ROUNDS, EARLY_STOP,
)
from src.tools.train_xgb import load_popularity_drift  # noqa: E402

TRAIN_END = '2023-12-31'
VAL_START = '2024-01-01'
VAL_END   = '2024-11-24'     # ⚠ 2024-11-25〜12-31 は history.db に1件も無い
SEEDS     = (42, 43, 44)
N_SHUFFLE = 200

#: カード §4 の合格線（danger/CRITERIA.md と同一）
TOP3_MAX  = 40.0
MIN_N     = 300


def load_features(path):
    df = pd.read_csv(path)
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    return df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)


def load_results(db, lo, hi):
    conn = sqlite3.connect(db)
    q = """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= ? AND r.date <= ?"""
    out = pd.read_sql_query(q, conn, params=(lo, hi))
    conn.close()
    return out


def fit_and_rank(df, cols, drift, seed):
    """評価期間の各馬に rl_rank（AI+市場の順位）を付けて返す。

    🔑 使うのは**順位**だけ。確率の水準には依存しない
       （2026-11-25 の conf=55 が止まったのは水準を再現できなかったため）。
    """
    tr = df[df['date_obj'] <= TRAIN_END]
    va = df[(df['date_obj'] >= VAL_START) & (df['date_obj'] <= VAL_END)].copy()
    Xtr, ytr = tr[cols].fillna(5.0), tr['is_fukusho'].values
    Xva, yva = va[cols].fillna(5.0), va['is_fukusho'].values
    spw = round((1 - ytr.mean()) / max(ytr.mean(), 0.01), 2)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=cols)
    dtr.set_base_margin(_base_margin(tr, drift, 11))
    dva.set_base_margin(_base_margin(va, drift, 12))
    bst = xgb.train(_xgb_params(spw, seed), dtr, num_boost_round=N_ROUNDS,
                    evals=[(dva, 'val')], early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
    margin = bst.predict(dva, output_margin=True)
    va['_score'] = margin
    # レース内でスコア降順＝rl_rank（同値は馬番の小さい方を上に）
    va = va.sort_values(['race_id', '_score', 'horse_num'],
                        ascending=[True, False, True])
    va['rl_rank'] = va.groupby('race_id').cumcount() + 1
    auc = roc_auc_score(yva, 1 / (1 + np.exp(-margin)))
    return va, auc


def pop1_frame(va, res):
    """市場1番人気の馬だけを着順つきで返す。"""
    m = va[['race_id', 'horse_num', 'rl_rank', 'f_popularity', 'date_obj']].merge(
        res, on=['race_id', 'horse_num'], how='left')
    p1 = m[(m['f_popularity'] == 1) & m['place'].notna()].copy()
    p1['top3'] = p1['place'].between(1, 3)
    p1['win'] = p1['place'] == 1
    return p1


def rate(sel):
    return round(float(sel['top3'].mean()) * 100, 2) if len(sel) else None


def main(argv):
    hf = argv[0] if argv else 'data/horse_features.csv'
    df = load_features(hf)
    cols = _feat_cols(df)
    drift = load_popularity_drift('.')
    res = load_results('data/history.db', VAL_START, VAL_END)
    print(f'特徴量 {len(df):,}行 / {df["race_id"].nunique():,}レース / {len(cols)}列')
    print(f'学習 〜{TRAIN_END} / 評価 {VAL_START}〜{VAL_END}')

    frames, aucs = [], []
    for sd in SEEDS:
        va, auc = fit_and_rank(df, cols, drift, sd)
        frames.append(pop1_frame(va, res))
        aucs.append(auc)
    auc = float(np.mean(aucs))
    nrace = frames[0]['race_id'].nunique()
    npop1 = len(frames[0])

    print('\n' + '=' * 72)
    print('■ 中止条件（率より手前で判定する）')
    print('=' * 72)
    stop = []
    print(f'評価レース {nrace:,} / 市場1番人気 {npop1:,}頭 / 軸AUC {auc:.4f}')
    if nrace < 2000:
        stop.append(f'評価レース {nrace} < 2,000（中止条件1）')
    if not (0.75 <= auc <= 0.82):
        stop.append(f'軸AUC {auc:.4f} が 0.75〜0.82 の外（中止条件2）')
    if npop1 < 2000:
        stop.append(f'市場1番人気 {npop1} < 2,000頭（中止条件4）')

    # 対照群B（オラクル）— 率を読む前に
    print('\n■ 対照群B（実際に3着内を外した1番人気からラベルを付ける）')
    okB = True
    for i, p1 in enumerate(frames):
        n_target = int((p1['rl_rank'] >= 2).sum())
        miss = p1[~p1['top3']]
        oracle = miss.head(n_target)
        r = rate(oracle)
        ok = r is not None and abs(r - 0.0) <= 0.01
        okB &= ok
        print(f'  seed{SEEDS[i]}: オラクル {len(oracle)}頭の3着内率 {r}% '
              f'（閉じた式では 0.00%）{"✅" if ok else "❌"}')
    if not okB:
        stop.append('対照群B が 0.00% を返さない（中止条件3）')

    print('\n中止条件: ' + ('発火なし ✅' if not stop else '🔴 発火'))
    for s in stop:
        print('  🔴', s)
    if stop:
        print('\n率は読まない（カード §6）。')
        return 1

    # ── 本測定 ───────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 主指標: 対象馬（1番人気 かつ rl_rank>=2）の3着内率')
    print('=' * 72)
    base_rates, tgt_rates, ns, wins, rois = [], [], [], [], []
    for p1 in frames:
        tgt = p1[p1['rl_rank'] >= 2]
        base_rates.append(rate(p1))
        tgt_rates.append(rate(tgt))
        ns.append(len(tgt))
        wins.append(round(float(tgt['win'].mean()) * 100, 2) if len(tgt) else None)
        pay = np.where(tgt['top3'], tgt['fukusho_payout'].fillna(0).values, 0.0)
        rois.append(round(float(pay.sum()) / (len(tgt) * 100) * 100, 2) if len(tgt) else None)
    n_avg = int(np.mean(ns))
    t_avg = float(np.mean(tgt_rates))
    b_avg = float(np.mean(base_rates))
    print(f'  1番人気すべて      {npop1:>5}頭  3着内率 {b_avg:.2f}%')
    print(f'  うち ◎が付かない   {n_avg:>5}頭  3着内率 {t_avg:.2f}%  '
          f'（Δ {t_avg - b_avg:+.2f}pt）')
    print(f'  （参考）1着率 {np.mean(wins):.2f}% / 複勝回収率 {np.mean(rois):.2f}%'
          f' ← 合否に使わない')

    print('\n■ 安定性（評価期間を3等分）')
    thirds = []
    for p1 in frames:
        d = p1.sort_values('date_obj')
        # ⚠ np.array_split は DataFrame を ndarray にする（2026-11-26 に踏んだ）。
        #    期間を**時系列で**3等分する（間引きではない）。
        e = np.linspace(0, len(d), 4).astype(int)
        cut = [d.iloc[e[k]:e[k + 1]] for k in range(3)]
        thirds.append([rate(x[x['rl_rank'] >= 2]) for x in cut])
    for k in range(3):
        vals = [t[k] for t in thirds if t[k] is not None]
        ns_k = []
        for p1 in frames:
            d = p1.sort_values('date_obj')
            e = np.linspace(0, len(d), 4).astype(int)
            part = d.iloc[e[k]:e[k + 1]]
            ns_k.append(len(part[part['rl_rank'] >= 2]))
        print(f'  U{k + 1}: 3着内率 {np.mean(vals):.2f}%  (N {int(np.mean(ns_k))})')
    u_rates = [float(np.mean([t[k] for t in thirds if t[k] is not None]))
               for k in range(3)]

    print('\n■ 対照群A（1番人気の中でラベルを無作為に付け替える × 200）')
    rng = np.random.default_rng(0)
    p1 = frames[0]
    n_t = int((p1['rl_rank'] >= 2).sum())
    obs = rate(p1[p1['rl_rank'] >= 2]) - rate(p1)
    vals = []
    for _ in range(N_SHUFFLE):
        idx = rng.choice(len(p1), size=n_t, replace=False)
        vals.append(rate(p1.iloc[idx]) - rate(p1))
    m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
    line = 2 * s / np.sqrt(len(vals))
    okA = abs(m) < line
    print(f'  帰無 mean {m:+.3f}pt / 線 {line:.3f} '
          f'{"✅ 0中心" if okA else "❌ 0中心でない"}  (std {s:.2f})')
    print(f'  実測 Δ {obs:+.2f}pt （帰無の {abs(obs) / s:.1f}σ）')

    print('\n' + '=' * 72)
    print('■ 判定（カード §4 の5基準・1つでも欠けたら不採用）')
    print('=' * 72)
    c1 = t_avg < TOP3_MAX
    c2 = n_avg >= MIN_N
    c3 = all(r < TOP3_MAX for r in u_rates)
    print(f'  基準1 3着内率 {t_avg:.2f}% < {TOP3_MAX}%     : {"✅" if c1 else "❌"}')
    print(f'  基準2 N {n_avg} >= {MIN_N}                : {"✅" if c2 else "❌"}')
    print(f'  基準3 3期間すべて {TOP3_MAX}% 未満         : {"✅" if c3 else "❌"}'
          f'  ({[round(r, 1) for r in u_rates]})')
    print(f'  基準4 対照群A が0中心              : {"✅" if okA else "❌"}')
    print(f'  基準5 対照群B がオラクルで 0.00%   : {"✅" if okB else "❌"}')
    ok = all((c1, c2, c3, okA, okB))
    print(f'\n判定: {"✅ 合格（✕印を出してよい）" if ok else "🔴 不採用"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
