"""再学習A の測定ハーネス（事前登録カード retrain/CRITERIA.md の設計どおり）。

P（本番モデル・学習〜2026-06-30）と N（学習期間を延長して再学習）を、
**同一の行・同一の順序・同一の base_margin 配列**で比較する。

🔴 base_margin の作り方は `train_xgb` 本体の関数をそのまま import して使う。
   自前で書き直すと、ドリフト注入が行ごとに乱数を引く性質のせいで
   行順の違いだけで AUC が 0.0016 動く（2026-09-15 に実測）。

使い方: python3 retrain/ab.py <workspace_dir>
⚠ workspace_dir は本番リポジトリではなく隔離コピーを渡すこと。
   train_xgb の保存処理がシンボリックリンク経由で本番モデルを書き換えた事故が
   2026-09-15 に実際に起きている。
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

WINDOWS = [('W1', '2026-07-31', '2026-08-01', '2026-08-31'),
           ('W2', '2026-08-31', '2026-09-01', '2026-09-30'),
           ('W3', '2026-09-30', '2026-10-01', '2026-10-18')]
SEEDS = [42, 7, 2026]
# train_xgb の既定ハイパラ（本番同型を崩さないため値を直書きせず明示的に揃える）
HP = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1,
          reg_lambda=1.0, early_stopping_rounds=50)


def metrics(y, p):
    return (round(roc_auc_score(y, p), 4), round(brier_score_loss(y, p), 4),
            round(log_loss(y, p), 4))


def main(W):
    sys.path.insert(0, W)
    import src.tools.train_xgb as T

    df = pd.read_csv(os.path.join(W, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj'])

    exclude = T._EXCLUDE_COLS | {'date_obj'} | T._MARKET_FEAT_COLS
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]
    drift = T.load_popularity_drift(W)
    print(f'CSV {len(df):,}行 × {len(df.columns)}列 / 特徴量 {len(feat_cols)} / '
          f'drift {"ON" if drift is not None else "OFF"}')

    meta = json.load(open(os.path.join(W, 'data', 'xgb_feature_cols_residual.json')))
    old_feats = list(meta['feature_cols'])
    P = T._load_xgb_model_any(os.path.join(W, 'data', 'xgb_fukusho_model_residual.pkl'), True)
    print(f'P: 学習{meta.get("n_train")}行 / 記録 val_auc {meta.get("val_auc")} / {len(old_feats)}列')
    # ── 中止条件4: 特徴量列の一致 ──
    same_set = set(feat_cols) == set(old_feats)
    same_ord = feat_cols == old_feats
    print(f'中止条件4 列集合の一致: {same_set} / 順序の一致: {same_ord}')
    if not same_set:
        print('  差分 N-P:', sorted(set(feat_cols) - set(old_feats)))
        print('  差分 P-N:', sorted(set(old_feats) - set(feat_cols)))

    def prep(sub, dseed):
        sub = sub.copy()
        sub['_n_horses'] = sub.groupby('race_id')['horse_num'].transform('count')
        pop = sub['f_popularity'].fillna(sub['_n_horses'] / 2)
        if drift is not None:
            pop = pd.Series(T._apply_popularity_drift(
                pop, sub['race_id'].values, drift, seed=dseed))
        bm = np.asarray(T._popularity_to_base_margin(pop, sub['_n_horses']), dtype=float)
        return sub[feat_cols].fillna(5.0), sub['is_fukusho'].values, bm

    # ── 参考値（旧 中止条件2・合否には使わない）──
    ref = df[(df.date_obj >= pd.Timestamp('2026-07-01')) &
             (df.date_obj <= pd.Timestamp('2026-09-15'))]
    Xr, yr, bmr = prep(ref, 12)
    dr = xgb.DMatrix(Xr.reindex(columns=old_feats, fill_value=5.0), feature_names=old_feats)
    dr.set_base_margin(bmr)
    pr = 1 / (1 + np.exp(-P.predict(dr, output_margin=True)))
    print(f'\n[参考] P を 2026-07-01〜09-15（{len(ref):,}行）で評価: '
          f'AUC {metrics(yr, pr)[0]}  ← 記録 {meta.get("val_auc")}'
          f'（ドリフト標本が当時と違うため一致は保証されない）')

    pool = {s: {'y': [], 'p': [], 'n': []} for s in SEEDS}
    per_window = []
    for wname, te, vs, ve in WINDOWS:
        tr = df[df.date_obj <= pd.Timestamp(te)]
        va = df[(df.date_obj >= pd.Timestamp(vs)) & (df.date_obj <= pd.Timestamp(ve))]
        Xtr, ytr, bmtr = prep(tr, 11)
        Xva, yva, bmva = prep(va, 12)
        bm_md5 = hashlib.md5(bmva.tobytes()).hexdigest()[:12]
        spw = round((1 - ytr.mean()) / max(ytr.mean(), 0.01), 2)
        print(f'\n━━ {wname}  学習 {len(Xtr):,}行(〜{te}) / 検定 {len(Xva):,}行({vs}〜{ve})'
              f'  spw {spw}  bm_val md5 {bm_md5}')

        # P（同じ bmva をそのまま使う ＝ 中止条件3）
        dP = xgb.DMatrix(Xva.reindex(columns=old_feats, fill_value=5.0), feature_names=old_feats)
        dP.set_base_margin(bmva)
        pP = 1 / (1 + np.exp(-P.predict(dP, output_margin=True)))
        aP, bP, lP = metrics(yva, pP)
        print(f'   P : AUC {aP}  Brier {bP}  LogLoss {lP}')

        for s in SEEDS:
            clf = XGBClassifier(scale_pos_weight=spw, eval_metric='logloss',
                                use_label_encoder=False, random_state=s, n_jobs=-1, **HP)
            params = clf.get_xgb_params()
            params['eval_metric'] = 'logloss'
            dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_cols)
            dtr.set_base_margin(bmtr)
            dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_cols)
            dva.set_base_margin(bmva)
            bst = xgb.train(params, dtr, num_boost_round=HP['n_estimators'],
                            evals=[(dva, 'val')],
                            early_stopping_rounds=HP['early_stopping_rounds'],
                            verbose_eval=False)
            pN = 1 / (1 + np.exp(-bst.predict(dva, output_margin=True)))
            aN, bN, lN = metrics(yva, pN)
            print(f'   N seed={s:<5} AUC {aN} ({aN-aP:+.4f})  Brier {bN} ({bN-bP:+.4f})  '
                  f'LogLoss {lN} ({lN-lP:+.4f})  best_iter {bst.best_iteration}')
            per_window.append(dict(window=wname, seed=s, n=len(yva),
                                   auc_P=aP, auc_N=aN, brier_P=bP, brier_N=bN,
                                   ll_P=lP, ll_N=lN))
            pool[s]['y'].append(yva); pool[s]['p'].append(pP); pool[s]['n'].append(pN)

    print('\n' + '=' * 72)
    print('■ 主測定: 3窓をプールした walk-forward')
    rows = []
    for s in SEEDS:
        y = np.concatenate(pool[s]['y']); pP = np.concatenate(pool[s]['p'])
        pN = np.concatenate(pool[s]['n'])
        aP, bP, lP = metrics(y, pP); aN, bN, lN = metrics(y, pN)
        rows.append((s, len(y), aP, aN, bP, bN, lP, lN))
        print(f'  seed={s:<5} N={len(y):,}  AUC P {aP} → N {aN} ({aN-aP:+.4f})  '
              f'Brier {bP}→{bN} ({bN-bP:+.4f})  LogLoss {lP}→{lN} ({lN-lP:+.4f})')
    dA = [r[3] - r[2] for r in rows]
    sd = float(np.std([r[3] for r in rows], ddof=1))
    print(f'\n  ΔAUC 平均 {np.mean(dA):+.4f}  （シード std {sd:.4f}）')
    print('  ── 採用基準（カード §5・4つすべて必要） ──')
    c1 = all(d > 0 for d in dA)
    c2 = np.mean(dA) >= 0.002
    c3 = abs(np.mean(dA)) >= 2 * sd
    c4 = all(r[5] < r[4] for r in rows) and all(r[7] < r[6] for r in rows)
    for i, (lbl, ok) in enumerate([
            ('1 3シードすべて N>P（悪化ゼロ）', c1),
            ('2 ΔAUC 平均 >= +0.002', c2),
            ('3 ΔAUC >= シードstd の2倍', c3),
            ('4 Brier・LogLoss も3シードとも改善', c4)], 1):
        print(f'   {"✅" if ok else "❌"} 基準{lbl}')
    print(f'\n  判定: {"採用候補（4基準すべて通過）" if all([c1,c2,c3,c4]) else "不採用（基準未達）"}')
    json.dump({'per_window': per_window,
               'pooled': [dict(seed=r[0], n=r[1], auc_P=r[2], auc_N=r[3],
                               brier_P=r[4], brier_N=r[5], ll_P=r[6], ll_N=r[7])
                          for r in rows],
               'criteria': {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4},
               'delta_auc_mean': float(np.mean(dA)), 'seed_std': sd},
              open(os.path.join(W, 'ab_result.json'), 'w'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main(sys.argv[1])
