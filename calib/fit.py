"""較正検証: 6窓 × 2モデル腕 の out-of-sample 予測を作る（seed=42 単一モデル）。

腕A 市場フリー131  : plain XGB（画面の AI勝率/AI複勝 の元）
腕B 本番同型134    : sigmoid(base_margin(市場人気) + f_AI)（cal_prob / EV の元）
                     ドリフト注入ON（seed 11/12 は train_xgb と同じ）

内側HO（学習期間の末尾42日）の**生確率**も保存する。較正器はここで学習するので、
後段で Platt / Isotonic / レース内正規化 を作り直せるようにしておく。

⚠ 窓の作り方は prep.py と完全に同一（同じ dropna/reset_index を通す）。
   行順がずれると全部が静かに壊れるので、ここは複製ではなく同じ手続きを書く。
"""
import os, sys, pickle, time
import numpy as np, pandas as pd
import xgboost as xgb
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai')
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)

CID82 = dict(max_depth=5, learning_rate=0.02, min_child_weight=300, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1)
N_EST, ES, SEED, INNER_DAYS = 4000, 50, 42, 42
MARKET = ['f_pop_last', 'f_pop_avg', 'f_beat_market_rate']
WINDOWS = [('T1', '2025-02-27', '2025-03-01', '2025-04-30'),
           ('T2', '2025-05-30', '2025-06-01', '2025-07-31'),
           ('T3', '2025-09-29', '2025-10-01', '2025-11-30'),
           ('W1', '2026-06-30', '2026-07-04', '2026-09-06'),
           ('W2', '2026-04-30', '2026-05-02', '2026-06-28'),
           ('W3', '2026-02-28', '2026-03-01', '2026-04-26')]

D = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))
FEAT, META = D['feat'], D['meta']

df = pd.read_csv(f'{BASE}/data/horse_features.csv')
df['date_obj'] = pd.to_datetime(df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
                                format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['date_obj']).reset_index(drop=True)
assert len(df) == len(META), '行数が meta と食い違う（prep.py と同じ手続きになっていない）'

FIELD = META['field'].to_numpy(float)
RID   = META['race_id'].to_numpy()
# f_popularity は 2.1% 欠損。train_xgb と同じく頭数の半分で埋める
POP = df['f_popularity'].to_numpy(float)
POP = np.where(np.isfinite(POP), POP, FIELD / 2.0)

XA = df[FEAT].fillna(5.0).to_numpy(np.float32)
XB = df[FEAT + MARKET].fillna(5.0).to_numpy(np.float32)
Y3 = df['is_fukusho'].to_numpy(np.int8)
DT = df['date_obj'].to_numpy()

DRIFT = load_popularity_drift(BASE)
assert DRIFT is not None, \
    '🔴 ドリフト分布が無い。確定人気で測ると符号が反転する（2026-07-31/08-21に2度踏んだ罠）'
print('ドリフト分布: あり', flush=True)


def base_margin(idx, seed):
    """その行集合の base_margin。確定人気にドリフトを注入し朝の人気を再現する。"""
    pop = pd.Series(_apply_popularity_drift(pd.Series(POP[idx]), RID[idx], DRIFT, seed=seed))
    return _popularity_to_base_margin(pop, pd.Series(FIELD[idx]))


def run(name, te, v0, v1):
    te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
    inner_from = te - pd.Timedelta(days=INNER_DAYS)
    m_fit = DT <= np.datetime64(inner_from)
    m_in  = (DT > np.datetime64(inner_from)) & (DT <= np.datetime64(te))
    m_va  = (DT >= np.datetime64(v0)) & (DT <= np.datetime64(v1))
    assert not (m_fit & m_va).any() and not (m_in & m_va).any()
    fi, ii, vi = np.where(m_fit)[0], np.where(m_in)[0], np.where(m_va)[0]
    # prep.py の窓と一致していることを検算（ずれたら全部静かに壊れる）
    assert np.array_equal(vi, D['windows'][name]['va_idx']), f'{name}: 評価行がずれている'
    assert np.array_equal(ii, D['windows'][name]['in_idx']), f'{name}: 内側HOがずれている'

    out = {'y3i': Y3[ii], 'y3v': Y3[vi], 'in_idx': ii, 'va_idx': vi,
           'span': (str(te.date()), str(v0.date()), str(v1.date()))}

    # ── 腕A: 市場フリー131（plain XGB・Isotonic前の生確率）─────────────
    m = xgb.XGBClassifier(n_estimators=N_EST, **CID82, scale_pos_weight=1.0,
                          eval_metric='logloss', early_stopping_rounds=ES,
                          random_state=SEED, n_jobs=4, verbosity=0, tree_method='hist')
    m.fit(XA[fi], Y3[fi], eval_set=[(XA[ii], Y3[ii])], verbose=False)
    out['A'] = dict(pi=m.predict_proba(XA[ii])[:, 1],
                    pv=m.predict_proba(XA[vi])[:, 1], trees=int(m.best_iteration))

    # ── 腕B: 本番同型134 + residual + ドリフト注入 ──────────────────────
    dtr = xgb.DMatrix(XB[fi], label=Y3[fi]); dtr.set_base_margin(base_margin(fi, 11))
    din = xgb.DMatrix(XB[ii], label=Y3[ii]); din.set_base_margin(base_margin(ii, 12))
    dva = xgb.DMatrix(XB[vi]);               dva.set_base_margin(base_margin(vi, 12))
    prm = dict(objective='binary:logistic', eval_metric='logloss', seed=SEED,
               nthread=4, tree_method='hist', **CID82)
    bst = xgb.train(prm, dtr, num_boost_round=N_EST, evals=[(din, 'in')],
                    early_stopping_rounds=ES, verbose_eval=False)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    out['B'] = dict(pi=sig(bst.predict(din, output_margin=True, iteration_range=(0, bst.best_iteration + 1))),
                    pv=sig(bst.predict(dva, output_margin=True, iteration_range=(0, bst.best_iteration + 1))),
                    trees=int(bst.best_iteration))
    return out


if __name__ == '__main__':
    res = {}
    for name, te, v0, v1 in WINDOWS:
        t0 = time.time()
        res[name] = run(name, te, v0, v1)
        r = res[name]
        print(f'{name}: A木{r["A"]["trees"]} B木{r["B"]["trees"]}  '
              f'n_val={len(r["A"]["pv"]):,}  実測3着内 {r["y3v"].mean()*100:.2f}%  '
              f'A平均{r["A"]["pv"].mean()*100:.2f}% B平均{r["B"]["pv"].mean()*100:.2f}%  '
              f'{time.time()-t0:.0f}s', flush=True)
    with open(f'{BASE}/calib/pred.pkl', 'wb') as f:
        pickle.dump(res, f, protocol=4)
    print('-> calib/pred.pkl')
