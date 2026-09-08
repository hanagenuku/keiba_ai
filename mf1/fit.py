"""M-1 の1配置ぶんの学習・評価。

指標はユーザー指定により5つ全部残す（木数の実験で LogLoss と AUC が逆に動くと
実証されたため、単一指標で選ばない）。選抜は 3着内AUC で行う（事前宣言）。
確率としての指標(LogLoss/Brier/ECE)は、scale_pos_weight が探索軸に入っていて
生の確率がアームごとに別スケールになるので、**内側ホールドアウトで当てた
Isotonic を通した後**の値を主に見る（本番も較正層を通す）。
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

N_EST = 4000
ES_ROUNDS = 50

DEFAULT = dict(max_depth=6, learning_rate=0.05, min_child_weight=10, subsample=0.8,
               colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
               spw='prior', es_metric='logloss')


def ece(y, p, bins=10):
    """等頻度10分割の較正誤差。"""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(p, q[1:-1])
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot


def fit_one(w, cfg, seed=42):
    Xf, y3f, Xi, y3i = w['Xf'], w['y3f'], w['Xi'], w['y3i']
    Xv, y3v, y1v = w['Xv'], w['y3v'], w['y1v']
    pr = float(y3f.mean())
    spw = round((1 - pr) / max(pr, .01), 2) if cfg['spw'] == 'prior' else 1.0
    m = XGBClassifier(
        n_estimators=N_EST, max_depth=cfg['max_depth'], learning_rate=cfg['learning_rate'],
        min_child_weight=cfg['min_child_weight'], subsample=cfg['subsample'],
        colsample_bytree=cfg['colsample_bytree'], reg_lambda=cfg['reg_lambda'],
        reg_alpha=cfg['reg_alpha'], scale_pos_weight=spw,
        eval_metric=cfg['es_metric'], early_stopping_rounds=ES_ROUNDS,
        random_state=seed, n_jobs=1, verbosity=0, tree_method='hist')
    # early stopping は内側ホールドアウトだけで行う（外側の窓は評価にしか使わない）
    m.fit(Xf, y3f, eval_set=[(Xi, y3i)], verbose=False)

    pi = m.predict_proba(Xi)[:, 1]
    pv = m.predict_proba(Xv)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(pi, y3i)
    cv = np.clip(iso.predict(pv), 1e-6, 1 - 1e-6)

    return dict(
        auc3=roc_auc_score(y3v, pv),
        auc1=roc_auc_score(y1v, pv),
        ll=log_loss(y3v, cv), brier=brier_score_loss(y3v, cv), ece=ece(y3v, cv),
        ll_raw=log_loss(y3v, np.clip(pv, 1e-6, 1 - 1e-6)),
        best=int(m.best_iteration), spw=spw)


def market_auc(w):
    """同じ母集団での市場だけの基準線（人気が欠損する行は落とす）。"""
    ok = np.isfinite(w['popv']) & (w['popv'] > 0)
    return dict(auc3=roc_auc_score(w['y3v'][ok], -w['popv'][ok]),
                auc1=roc_auc_score(w['y1v'][ok], -w['popv'][ok]),
                n=int(ok.sum()), n_all=len(ok))
