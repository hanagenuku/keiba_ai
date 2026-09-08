"""Phase 2: 市場フリーモデルの特徴量グループ・アブレーション。

M-1 のハーネス(mf1/fit.py)と設定(cid 82)をそのまま使う。
過去に「市場アンカー型で効果なし」と判定された群も、市場フリー体制では
一度も測っていないので全部測り直す（2026-09-08 ユーザー方針）。
"""
import os, sys, pickle, itertools, json
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from mf1.fit import fit_one
from src.features.shap_explain import FEATURE_CATEGORY_MAP

CID82 = dict(max_depth=5, learning_rate=0.02, min_child_weight=300, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.1,
             spw='none', es_metric='logloss')
WINS, SEEDS = ['W1', 'W2', 'W3'], [42, 7, 2026]

D = pickle.load(open(f'{BASE}/mf1/data.pkl', 'rb'))
FEAT = D['feat']
COLIDX = {c: i for i, c in enumerate(FEAT)}

groups = defaultdict(list)
for c, cat in FEATURE_CATEGORY_MAP.items():
    if c in COLIDX:
        groups[cat].append(COLIDX[c])
assert '過去人気推移' not in groups, '市場列が残っている'
missing = [c for c in FEAT if c not in FEATURE_CATEGORY_MAP]
assert not missing, f'カテゴリ未登録の列: {missing}'
GROUPS = {k: sorted(v) for k, v in groups.items()}
print(f'{len(FEAT)}列 / {len(GROUPS)}グループ  ' +
      ' '.join(f'{k}:{len(v)}' for k, v in sorted(GROUPS.items(), key=lambda x: -len(x[1]))))

TASKS = [(g, w, s) for g in [None] + sorted(GROUPS, key=lambda k: -len(GROUPS[k]))
         for w in WINS for s in SEEDS]


def job(t):
    grp, w, seed = t
    keep = np.ones(len(FEAT), bool)
    if grp is not None:
        keep[GROUPS[grp]] = False
    win = D['windows'][w]
    sub = dict(win)
    for k in ('Xf', 'Xi', 'Xv'):
        sub[k] = win[k][:, keep]
    r = fit_one(sub, CID82, seed=seed)
    return dict(drop=grp or '(なし)', win=w, seed=seed, n_feat=int(keep.sum()),
                auc3=r['auc3'], auc1=r['auc1'], ll=r['ll'], ece=r['ece'],
                brier=r['brier'], trees=r['best'])


if __name__ == '__main__':
    with Pool(4) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(job, TASKS), 1):
            rows.append(r)
            if i % 9 == 0:
                print(f'  {i}/{len(TASKS)} 完了', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{BASE}/ablation_raw.csv', index=False)
    print('saved ablation_raw.csv')
