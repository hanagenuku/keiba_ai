import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_eval import run_arm, load_df, SEEDS
BASE = os.path.dirname(os.path.abspath(__file__))
load_df()
EXTRA = [('W0', '2026-08-16', '2026-08-22', '2026-09-06'),
         ('W4', '2025-12-31', '2026-01-04', '2026-02-22')]
rows, preds = [], {}
for wname, te, vs, ve in EXTRA:
    for arm in ['M', 'R', 'Fplus', 'F']:
        for sd in ([0] if arm == 'M' else SEEDS):
            r = run_arm(arm, te, vs, ve, sd, es_mode='inner')
            preds[(wname, arm, sd)] = (r['va_idx'], r['prob'])
            rows.append(dict(win=wname, arm=arm, seed=sd, n_val=r['n_val'],
                             n_races=r['n_races'], auc=r['auc'], logloss=r['logloss'],
                             brier=r['brier'], ece=r['ece'], best_iter=r['best_iter'],
                             n_feat=r['n_feat']))
            print(f"{wname} {arm:<6} seed={sd:<5} AUC {r['auc']:.4f} LL {r['logloss']:.4f} "
                  f"n_val {r['n_val']} iter {r['best_iter']}", flush=True)
pd.DataFrame(rows).to_csv(f'{BASE}/phase1b_raw.csv', index=False)
np.save(f'{BASE}/phase1b_preds.npy', preds, allow_pickle=True)
print('done')
