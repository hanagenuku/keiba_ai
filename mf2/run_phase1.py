import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_eval import run_arm, load_df, WINDOWS, SEEDS, MARKET_COLS
from scipy.stats import spearmanr

load_df()
rows, preds = [], {}
for wname, tr_end, vs, ve in WINDOWS:
    for arm in ['M', 'R', 'Fplus', 'F']:
        seeds = [0] if arm == 'M' else SEEDS
        for sd in seeds:
            r = run_arm(arm, tr_end, vs, ve, sd, es_mode='inner')
            preds[(wname, arm, sd)] = (r['va_idx'], r['prob'])
            rows.append(dict(win=wname, arm=arm, seed=sd, n_val=r['n_val'],
                             n_races=r['n_races'], auc=r['auc'], logloss=r['logloss'],
                             brier=r['brier'], ece=r['ece'], best_iter=r['best_iter'],
                             n_feat=r['n_feat']))
            print(f"{wname} {arm:<6} seed={sd:<5} AUC {r['auc']:.4f}  LL {r['logloss']:.4f}  "
                  f"Brier {r['brier']:.4f}  ECE {r['ece']:.4f}  iter {r['best_iter']:>3}  feat {r['n_feat']}",
                  flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_raw.csv'), index=False)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_preds.npy'),
        {k: v for k, v in preds.items()}, allow_pickle=True)

print('\n===== 集計（シード平均 ± std）=====')
agg = df.groupby(['arm', 'win']).agg(auc=('auc','mean'), auc_sd=('auc','std'),
                                     ll=('logloss','mean'), brier=('brier','mean'),
                                     ece=('ece','mean'), it=('best_iter','mean')).reset_index()
for arm in ['M','R','Fplus','F']:
    s = agg[agg.arm==arm]
    line = ' | '.join(f"{r.win} {r.auc:.4f}" + (f"±{r.auc_sd:.4f}" if r.auc_sd==r.auc_sd else "")
                      for r in s.itertuples())
    print(f"{arm:<6} {line}   3窓平均 AUC {s.auc.mean():.4f}  LL {s.ll.mean():.4f}  ECE {s.ece.mean():.4f}")
