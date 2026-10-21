#!/usr/bin/env python3
"""2026年の完全OOSで予想生成ロジック（P: 本番 / F: 市場フリー）を比べる。

事前登録カード: logic2026/CRITERIA.md（これをコミットしてから実行する）

🔴 **測定は1回**。閾値・アーム・期間・母集団・評価指標はカードのとおりで、
   結果を見てから変えない。

🔴 中止条件は**回収率より手前で印字して判定する**。
   2026-10-19② で中止条件を最後に印字する作りにしたせいで ECE を先に
   見てしまった。同じ作りにしない。

使い方:
    cd <隔離ワークスペース>   # data/horse_features.csv と history.db がある場所
    python3 logic2026/eval.py
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.getcwd())

from src.tools.train_xgb import (  # noqa: E402
    _EXCLUDE_COLS, _MARKET_FEAT_COLS, _apply_popularity_drift,
    _popularity_to_base_margin, load_popularity_drift,
)

TRAIN_END = '2025-12-31'
VAL_START = '2026-01-01'
VAL_END   = '2026-10-18'
SEEDS     = (42, 7, 2026)

# 本番と同じハイパラ（train_xgb の既定値）
PARAMS = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=10,
              reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss')
N_ROUNDS = 500
EARLY_STOP = 50


# ── データ ──────────────────────────────────────────────────────────────
def load_frames(base='.'):
    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    tr = df[df['date_obj'] <= TRAIN_END].copy()
    va = df[(df['date_obj'] >= VAL_START) & (df['date_obj'] <= VAL_END)].copy()
    return df, tr, va


def load_payouts(base='.'):
    """history.db から着順と払戻を引く。

    ⚠ `race_dividends`（払戻表）は 2026-06-06 以降しか無いので、2026年を
      通しで測るにはここを使うしかない。`horse_history.fukusho_payout` は
      3着内なのに 0/NULL の行が 1.73% あり、**少頭数で複勝が発売されて
      いないケースとスクレイプ漏れを区別できない**（2026-10-02 に確認）。
      よって「3着内だが払戻が無い」行は**評価不能として分母から外す**
      （0円として混ぜると実力を過小評価する。2026-08-18③）。
      両アームでまったく同じ扱いなので **P vs F の比較には影響しない**。
    """
    hp = os.path.join(base, 'data', 'history.db')
    conn = sqlite3.connect(hp)
    q = """SELECT h.race_id, h.horse_num, h.place,
                  h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= ? AND r.date <= ?"""
    pay = pd.read_sql_query(q, conn, params=(VAL_START, VAL_END))
    conn.close()
    return pay


# ── 学習 ────────────────────────────────────────────────────────────────
def _feat_cols(df):
    exclude = _EXCLUDE_COLS | {'date_obj'} | _MARKET_FEAT_COLS
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]


def _base_margin(part, drift, seed):
    n = part.groupby('race_id')['horse_num'].transform('count')
    pop = part['f_popularity'].fillna(n / 2)
    if drift is not None:
        pop = pd.Series(_apply_popularity_drift(
            pop, part['race_id'].values, drift, seed=seed))
    return _popularity_to_base_margin(pop, n)


def train_arm(arm, tr, va, cols, drift, seed):
    """arm='P' なら本番と同じ残差学習、'F' なら市場を一切使わない独立モデル。"""
    Xtr, ytr = tr[cols].fillna(5.0), tr['is_fukusho'].values
    Xva, yva = va[cols].fillna(5.0), va['is_fukusho'].values
    spw = round((1 - ytr.mean()) / max(ytr.mean(), 0.01), 2)
    params = dict(PARAMS, scale_pos_weight=spw, random_state=seed, seed=seed)

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=cols)
    if arm == 'P':
        bm_tr = _base_margin(tr, drift, 11)
        bm_va = _base_margin(va, drift, 12)
        dtr.set_base_margin(bm_tr)
        dva.set_base_margin(bm_va)
    bst = xgb.train(params, dtr, num_boost_round=N_ROUNDS,
                    evals=[(dva, 'val')], early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
    margin = bst.predict(dva, output_margin=True)      # 2重sigmoid回避
    prob = 1 / (1 + np.exp(-margin))
    return prob, bst.best_iteration


# ── 評価 ────────────────────────────────────────────────────────────────
def pick_axis(va, score):
    """レースごとに score 最大の馬を軸にする（同値は馬番の小さい方）。"""
    d = pd.DataFrame({'race_id': va['race_id'].values,
                      'horse_num': va['horse_num'].values,
                      'score': score})
    d = d.sort_values(['race_id', 'score', 'horse_num'],
                      ascending=[True, False, True])
    return d.groupby('race_id', as_index=False).first()


def roi(axis, pay, kind='fukusho', bonus=0.0):
    """軸1頭を1点買った時の回収率。

    kind='fukusho' なら3着内で払戻、'tansho' なら1着で払戻。
    **払戻が取れていない的中行は分母から外す**（評価不能）。
    bonus は対照群B用。全ての評価対象行に一律で足す（ΔR = bonus pt になる）。
    """
    m = axis.merge(pay, on=['race_id', 'horse_num'], how='left')
    col = f'{kind}_payout'
    hit_cond = m['place'].between(1, 3) if kind == 'fukusho' else (m['place'] == 1)
    has_result = m['place'].notna()
    bad = has_result & hit_cond & (m[col].fillna(0) <= 0)   # 的中だが配当なし
    ok = has_result & ~bad
    pays = np.where(hit_cond[ok], m.loc[ok, col].fillna(0).values, 0.0) + bonus
    n = int(ok.sum())
    if n == 0:
        return {'n': 0, 'hit': 0, 'rate': None, 'roi': None, 'cut3': None,
                'unevaluable': int(bad.sum()), 'no_result': int((~has_result).sum())}
    srt = np.sort(pays)[::-1]
    return {
        'n': n,
        'hit': int((hit_cond[ok]).sum()),
        'rate': round(float(hit_cond[ok].mean()) * 100, 2),
        'roi': round(float(pays.sum()) / (n * 100) * 100, 2),
        'cut3': round(float(srt[3:].sum()) / (n * 100) * 100, 2),
        'unevaluable': int(bad.sum()),
        'no_result': int((~has_result).sum()),
    }


def monthly_roi(axis, pay, va):
    dates = va.drop_duplicates('race_id')[['race_id', 'date_obj']]
    a = axis.merge(dates, on='race_id')
    out = []
    for m, g in a.groupby(a['date_obj'].dt.strftime('%Y-%m')):
        r = roi(g[['race_id', 'horse_num']], pay)
        out.append({'month': m, 'races': len(g), 'roi': r['roi'],
                    'rate': r['rate'], 'n': r['n']})
    return out


def main():
    base = '.'
    print('=' * 72)
    print('■ 中止条件の判定（回収率より手前で出す）')
    print('=' * 72)
    df, tr, va = load_frames(base)
    cols = _feat_cols(df)
    drift = load_popularity_drift(base)

    stop = []
    # 1. 学習に検定期の行が混ざっていないか
    leak = int((tr['date_obj'] > TRAIN_END).sum())
    print(f'  1. 学習行に {TRAIN_END} より後の行: {leak}（0 でなければ中止）')
    if leak:
        stop.append('学習データに検定期の行が混ざっている')
    # 2. 両アームの母集団が同一か（同じ df から同じ行を使うので構造的に同一）
    print(f'  2. 学習 {len(tr):,}行 / 検定 {len(va):,}行 / 列 {len(cols)}'
          f'  期間 {tr["date_obj"].min().date()}〜{tr["date_obj"].max().date()}'
          f' → {va["date_obj"].min().date()}〜{va["date_obj"].max().date()}')
    if len(va) == 0:
        stop.append('検定データが空')
    print(f'     ドリフト標本: {"あり" if drift is not None else "なし"}')
    if drift is None:
        stop.append('ドリフト標本が無い（本番と同じ base_margin を作れない）')

    pay = load_payouts(base)
    print(f'  3. 払戻テーブル: {len(pay):,}行')

    # P を1シードだけ先に学習して AUC 帯を確認する
    probP0, bi0 = train_arm('P', tr, va, cols, drift, SEEDS[0])
    aucP0 = roc_auc_score(va['is_fukusho'].values, probP0)
    print(f'  4. P の検定窓 AUC = {aucP0:.4f}（0.75〜0.82 の外なら中止）'
          f'  best_iteration={bi0}')
    if not (0.75 <= aucP0 <= 0.82):
        stop.append(f'P の AUC {aucP0:.4f} が想定帯の外')

    if stop:
        print('\n🔴 中止条件に該当。結果を読まずに破棄する:')
        for s in stop:
            print('   -', s)
        return

    print('\n  ✅ 中止条件はすべて通過。測定に進む。')

    # ── 対照群B: 注入の復元 ───────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 対照群B（注入の復元）')
    print('=' * 72)
    axis0 = pick_axis(va, probP0)
    base_r = roi(axis0, pay)
    inj = roi(axis0, pay, bonus=20.0)
    delta = round(inj['roi'] - base_r['roi'], 4)
    print(f'  注入前 {base_r["roi"]}%  注入後 {inj["roi"]}%  差 {delta}pt'
          f'（注入量 20.0pt と一致すべき）')
    okB = abs(delta - 20.0) < 1e-3
    print(f'  判定: {"✅ 復元した" if okB else "🔴 復元できていない"}')

    # 独立検算: 1着馬を必ず選べば R_tan は単勝払戻の平均になるはず
    truth = pay[pay['place'] == 1][['race_id', 'horse_num']]
    orc = roi(truth, pay, kind='tansho')
    exp = pay[(pay['place'] == 1) & (pay['tansho_payout'] > 0)]['tansho_payout'].mean()
    print(f'  検算(オラクル単勝): R={orc["roi"]}%  単勝払戻の平均/100={exp/100*100:.2f}%'
          f'  → {"✅" if abs(orc["roi"] - exp) < 0.05 else "🔴"}')

    # ── 本測定 ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 本測定（P: 本番構成 / F: 市場フリー）')
    print('=' * 72)
    res = {}
    preds = {}
    for arm in ('P', 'F'):
        rows = []
        for sd in SEEDS:
            prob = probP0 if (arm == 'P' and sd == SEEDS[0]) else \
                train_arm(arm, tr, va, cols, drift, sd)[0]
            ax = pick_axis(va, prob)
            r = roi(ax, pay)
            rt = roi(ax, pay, kind='tansho')
            rows.append({
                'seed': sd,
                'auc': round(roc_auc_score(va['is_fukusho'].values, prob), 4),
                'brier': round(brier_score_loss(va['is_fukusho'].values, prob), 4),
                'logloss': round(log_loss(va['is_fukusho'].values, prob), 4),
                'fuku': r, 'tan': rt,
            })
            preds.setdefault(arm, {})[sd] = prob
            print(f'  {arm} seed={sd:4d}  AUC {rows[-1]["auc"]}  '
                  f'複勝 {r["hit"]}/{r["n"]} ({r["rate"]}%) 回収 {r["roi"]}% '
                  f'3本抜 {r["cut3"]}%  単勝回収 {rt["roi"]}%')
        res[arm] = rows

    def avg(arm, path):
        vals = [r[path[0]][path[1]] if len(path) == 2 else r[path[0]]
                for r in res[arm]]
        return round(float(np.mean([v for v in vals if v is not None])), 3)

    print(f'\n  平均  P: AUC {avg("P", ["auc"])}  複勝回収 {avg("P", ["fuku", "roi"])}%'
          f'  3本抜 {avg("P", ["fuku", "cut3"])}%')
    print(f'        F: AUC {avg("F", ["auc"])}  複勝回収 {avg("F", ["fuku", "roi"])}%'
          f'  3本抜 {avg("F", ["fuku", "cut3"])}%')
    print(f'  F − P = {round(avg("F", ["fuku", "roi"]) - avg("P", ["fuku", "roi"]), 2)}pt'
          f'（採用基準は +2.0pt 以上）')

    # ── 市場比較ゲート ────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 市場比較ゲート')
    print('=' * 72)
    mkt_score = -va['f_popularity'].fillna(99).values     # 人気が良いほど高い
    ax_m = pick_axis(va, mkt_score)
    r_m = roi(ax_m, pay)
    auc_m = roc_auc_score(va['is_fukusho'].values, mkt_score)
    print(f'  市場単独   AUC {auc_m:.4f}  複勝 {r_m["hit"]}/{r_m["n"]} '
          f'({r_m["rate"]}%) 回収 {r_m["roi"]}%  3本抜 {r_m["cut3"]}%')
    print(f'  AI単独(F)  AUC {avg("F", ["auc"])}  回収 {avg("F", ["fuku", "roi"])}%')
    print(f'  AI×市場(P) AUC {avg("P", ["auc"])}  回収 {avg("P", ["fuku", "roi"])}%')

    # 乖離の方向別（P の軸が市場より上か下か）
    print('\n  ── 乖離の方向別（P の軸・seed=42）──')
    pr = pd.DataFrame({'race_id': va['race_id'].values,
                       'horse_num': va['horse_num'].values,
                       'p': probP0, 'pop': va['f_popularity'].fillna(99).values})
    pr['ai_rank'] = pr.groupby('race_id')['p'].rank(ascending=False, method='first')
    ax = pr[pr['ai_rank'] == 1]
    for lab, sel in [('AIが強気(人気>=4)', ax['pop'] >= 4),
                     ('やや強気(人気2-3)', ax['pop'].between(2, 3)),
                     ('一致(人気1)', ax['pop'] == 1)]:
        g = ax[sel][['race_id', 'horse_num']]
        r = roi(g, pay)
        print(f'    {lab:18s} N={r["n"]:5d}  複勝率 {r["rate"]}%  回収 {r["roi"]}%')

    # ── 市場アンカーが壊れたレース（フォールバック仮説） ────────────────
    print('\n' + '=' * 72)
    print('■ フォールバック仮説（市場アンカーが壊れたレースで F > P か）')
    print('=' * 72)
    bad_races = va.groupby('race_id')['f_popularity'].apply(
        lambda s: s.isna().mean() > 0.3)
    bad_ids = set(bad_races[bad_races].index)
    print(f'  f_popularity が3割以上欠けるレース: {len(bad_ids)}')
    if bad_ids:
        for arm in ('P', 'F'):
            ax = pick_axis(va, preds[arm][SEEDS[0]])
            sub = ax[ax['race_id'].isin(bad_ids)]
            r = roi(sub, pay)
            print(f'    {arm}: N={r["n"]}  複勝率 {r["rate"]}%  回収 {r["roi"]}%')

    # ── 対照群A: 並べ替え分布 ─────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 対照群A（レース内で予測を並べ替え 200回）')
    print('=' * 72)
    rng = np.random.default_rng(20261021)
    rid = va['race_id'].values
    base_roi = roi(pick_axis(va, probP0), pay)['roi']
    deltas = []
    for _ in range(200):
        sh = rng.random(len(probP0))
        deltas.append(roi(pick_axis(va, sh), pay)['roi'] - base_roi)
    deltas = np.array(deltas)
    lim = 2 * deltas.std() / np.sqrt(len(deltas))
    print(f'  帰無の平均 {deltas.mean():+.3f}pt  std {deltas.std():.3f}'
          f'  合格線 |mean| < {lim:.3f}')
    print(f'  🔑 これは「無作為に軸を選んだ時の回収率」との差。'
          f'帰無が0中心になるのは**無作為と同じ**という意味ではない。')
    print(f'  実測 P − 無作為 = {-deltas.mean():+.3f}pt'
          f'  並べ替え P値 = {float((deltas >= 0).mean()):.4f}')

    # ── 月別 ──────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('■ 月別（複勝回収率）')
    print('=' * 72)
    mp = monthly_roi(pick_axis(va, probP0), pay, va)
    mf = {m['month']: m for m in monthly_roi(pick_axis(va, preds['F'][SEEDS[0]]), pay, va)}
    win = 0
    for m in mp:
        f = mf.get(m['month'], {})
        better = (f.get('roi') or 0) > (m['roi'] or 0)
        win += 1 if better else 0
        print(f"  {m['month']}  R={m['races']:4d}  P {m['roi']:6.1f}%  "
              f"F {f.get('roi', float('nan')):6.1f}%  {'F勝ち' if better else ''}")
    print(f'  F が上回った月: {win}/{len(mp)}（採用基準は6以上）')

    out = {'train_rows': len(tr), 'val_rows': len(va), 'cols': len(cols),
           'auc_market': round(auc_m, 4), 'roi_market': r_m,
           'arms': res, 'monthly_P': mp, 'monthly_F': list(mf.values()),
           'controlB_delta': delta, 'controlA_mean': float(deltas.mean()),
           'controlA_std': float(deltas.std()), 'months_F_wins': win}
    with open('logic2026_result.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print('\n✅ logic2026_result.json に保存')


if __name__ == '__main__':
    main()
