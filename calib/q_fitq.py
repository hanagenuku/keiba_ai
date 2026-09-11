"""AI勝率の品質評価: 3分割(較正train / 較正val / 最終OOS)用のOOS予測を1本作る。

本番と同型: 134特徴量(131 + 過去市場3) / residual / ドリフト注入ON / cid82 / seed 42。
train_end を 2025-06-30 に戻すことで、本番の較正器には存在しないOOSを作る。
"""
import os, sys, pickle, time
import numpy as np, pandas as pd, xgboost as xgb
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai')
from src.tools.train_xgb import (load_popularity_drift, _apply_popularity_drift,
                                 _popularity_to_base_margin)
CID82 = dict(max_depth=5, learning_rate=0.02, min_child_weight=300, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1)
N_EST, ES, SEED, INNER_DAYS = 4000, 50, 42, 42
MARKET = ['f_pop_last', 'f_pop_avg', 'f_beat_market_rate']
TE, SPLITS = '2025-06-30', [('CT', '2025-07-01', '2025-10-31'),
                            ('CV', '2025-11-01', '2025-12-31'),
                            ('OOS', '2026-01-01', '2026-09-06')]

D = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))
FEAT, META = D['feat'], D['meta']
df = pd.read_csv(f'{BASE}/data/horse_features.csv')
df['date_obj'] = pd.to_datetime(df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
                                format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['date_obj']).reset_index(drop=True)
assert len(df) == len(META), '行数が meta と食い違う'
FIELD, RID = META['field'].to_numpy(float), META['race_id'].to_numpy()
POP = df['f_popularity'].to_numpy(float); POP = np.where(np.isfinite(POP), POP, FIELD / 2.0)
XB = df[FEAT + MARKET].fillna(5.0).to_numpy(np.float32)
Y3 = df['is_fukusho'].to_numpy(np.int8).astype(int)          # 🔴 int8 のまま集計しない
Y1 = (df['place'].to_numpy() == 1).astype(int)
DT = df['date_obj'].to_numpy()
DRIFT = load_popularity_drift(BASE)
assert DRIFT is not None, '🔴 ドリフト分布が無い（確定人気で測ると符号が反転する）'

def bm_of(idx, seed):
    pop = pd.Series(_apply_popularity_drift(pd.Series(POP[idx]), RID[idx], DRIFT, seed=seed))
    return np.asarray(_popularity_to_base_margin(pop, pd.Series(FIELD[idx])), dtype=float)

te = pd.Timestamp(TE); inner_from = te - pd.Timedelta(days=INNER_DAYS)
m_fit = DT <= np.datetime64(inner_from)
m_in = (DT > np.datetime64(inner_from)) & (DT <= np.datetime64(te))
fi, ii = np.where(m_fit)[0], np.where(m_in)[0]
blocks = {k: np.where((DT >= np.datetime64(a)) & (DT <= np.datetime64(b)))[0] for k, a, b in SPLITS}
# 関門3: 行数が CRITERIA の表と一致すること
EXP = {'CT': 14910, 'CV': 8140, 'OOS': 32904}
for k, n in EXP.items():
    assert len(blocks[k]) == n, f'{k}: {len(blocks[k])} != {n}（CRITERIAの表とずれている）'
# 関門4: 学習側と評価側が一切重ならないこと
for k, v in blocks.items():
    assert not np.intersect1d(v, np.concatenate([fi, ii])).size, f'{k} が学習側と重なる'
print(f'学習 {len(fi):,} / 内側HO {len(ii):,} / ' +
      ' / '.join(f'{k} {len(v):,}' for k, v in blocks.items()), flush=True)

t0 = time.time()
dtr = xgb.DMatrix(XB[fi], label=Y3[fi]); dtr.set_base_margin(bm_of(fi, 11))
din = xgb.DMatrix(XB[ii], label=Y3[ii]); din.set_base_margin(bm_of(ii, 12))
prm = dict(objective='binary:logistic', eval_metric='logloss', seed=SEED,
           nthread=4, tree_method='hist', **CID82)
bst = xgb.train(prm, dtr, num_boost_round=N_EST, evals=[(din, 'in')],
                early_stopping_rounds=ES, verbose_eval=False)
IR = (0, bst.best_iteration + 1)
print(f'木 {bst.best_iteration}  {time.time()-t0:.0f}s', flush=True)

out = {'span': {'train_end': TE, **{k: (a, b) for k, a, b in SPLITS}},
       'trees': int(bst.best_iteration)}
for k, idx in blocks.items():
    bm = bm_of(idx, 12)
    d = xgb.DMatrix(XB[idx]); d.set_base_margin(bm)
    raw = bst.predict(d, output_margin=True, iteration_range=IR)
    out[k] = dict(idx=idx, raw=raw, bm=bm, y1=Y1[idx], y3=Y3[idx],
                  field=FIELD[idx], rid=RID[idx], pop=POP[idx],
                  date=pd.Series(DT[idx]).dt.strftime('%Y-%m-%d').to_numpy())
# 関門1: ability が base_margin の取り方に依存しないこと（±3 振る）
idx = blocks['OOS'][:4000]
ab = []
for sh in (-3.0, 0.0, 3.0):
    d = xgb.DMatrix(XB[idx]); d.set_base_margin(bm_of(idx, 12) + sh)
    ab.append(bst.predict(d, output_margin=True, iteration_range=IR) - (bm_of(idx, 12) + sh))
dev = float(max(np.abs(ab[0] - ab[1]).max(), np.abs(ab[2] - ab[1]).max()))
assert dev <= 1e-4, f'🔴 ability が土台に依存している: 最大ズレ {dev}'
# 🔴 assert 形では `assert dev <= t` が正しい。`assert not (dev > t)` は NaN が素通りする
#    （`if not (dev <= t): raise` と assert では真偽が逆。自分の関門破りテストで検出した）
print(f'関門1 OK: ability の base_margin 依存 最大ズレ {dev:.3e}', flush=True)
with open(f'{BASE}/q/predq.pkl', 'wb') as f:
    pickle.dump(out, f, protocol=4)
print('-> q/predq.pkl')
