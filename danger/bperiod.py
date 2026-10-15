"""「危険な人気馬」の条件を B期間で測る。事前登録は `danger/CRITERIA.md`。

🔴 測定は1回。期間・閾値・母集団・推定仕様を結果を見てから動かさない。

母集団は **2025-01-01 以降**（カードの定義そのまま）。
⚠ カードは「5,932レース」と書いたが、これは Step 1 を閉じた 2026-09-22 時点の
   件数。その後の結果取得で増えており、同じ日付定義で 6,077 レースある。
   増えたのは時間で、母集団の定義は動かしていない。

実装上の選択（**結果を読む前に**ここに書いた）

- walk-forward 3窓。各窓はその窓が始まる前のデータだけで学習する。
    W1 train <=2024-12-31 → test 2025-01-01..2025-06-30
    W2 train <=2025-06-30 → test 2025-07-01..2025-12-31
    W3 train <=2025-12-31 → test 2026-01-01..(最終日)
- 学習は本番と同型: residual=True / `xgb_feature_cols.json` の134列そのまま /
  既定ハイパラ / seed=42 / base_margin に実測ドリフト注入(seed=11) /
  早期打ち切りは**テスト窓の直前61日**を内部valにして行う（本番も「直近を val に
  して early stopping」なので同型。テスト窓は一切見ない）。
- 主測定の `pop` と test 側 base_margin は **確定人気**。
  感度分析として**ドリフト注入人気**(seed=12 ＝本番の朝の情報量)でも測る。
  `ability_margin` は base_margin に依存しないので solo は両者で同じになるはず。
  → それ自体を突合して計測器の検査にする（North Star #8）。
- 前半/後半は **2025年 / 2026年**（構造で決める。中央値で割らない）。
- 🔴 基準4のオッズ帯層別は**実行不能**。`history.db` の `win_odds` は B期間
   83,547行すべて欠損（0.0%）。代わりに**頭数帯とクラス**で層別する。
   データの制約であって、結果を見てから選んだ層ではない。
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/keiba_ai')

from src.features.aptitude import APT_COLS, aptitude_scores          # noqa: E402
from src.tools.train_xgb import (                                     # noqa: E402
    _apply_popularity_drift, _popularity_to_base_margin,
    load_popularity_drift,
)

BASE = '/home/user/keiba_ai'
CSV = os.environ.get(
    'DANGER_CSV', '/tmp/claude-0/-home-user-keiba-ai/'
    '00b2457c-aac1-5a17-b0c5-39bb7cf987a1/scratchpad/danger/feat.csv')
B_START = '2025-01-01'
# 中間生成物（行単位の表・対照群の記録）はリポジトリに置かない（North Star #5）
OUT = os.environ.get('DANGER_OUT', os.path.dirname(CSV))
WINDOWS = [('2025-01-01', '2025-06-30'),
           ('2025-07-01', '2025-12-31'),
           ('2026-01-01', '2099-12-31')]
VAL_DAYS = 61
SEED_MODEL = 42
SEED_DRIFT_TRAIN = 11
SEED_DRIFT_TEST = 12
N_PERM = 200
PERM_SEED = 20261015

# 条件 → (選別マスク, 比較プール)。プールは「その条件が狙う市場人気帯」。
CONDS = [
    ('V0  pop==1 & solo>=6',          lambda d: d['solo'] >= 6,                      1),
    ('V1a pop==1 & solo>=8',          lambda d: d['solo'] >= 8,                      1),
    ('V1b pop==1 & solo>=10',         lambda d: d['solo'] >= 10,                     1),
    ('V2  pop==1 & solo>=6 & rl>=2',  lambda d: (d['solo'] >= 6) & (d['rl'] >= 2),   1),
    ('V3  pop==1 & solo>=6 & fit>=6', lambda d: (d['solo'] >= 6) & (d['fit'] >= 6),  1),
    ('V4  pop<=2 & solo>=6',          lambda d: d['solo'] >= 6,                      2),
]


def _rank_desc(vals, nums):
    """大きいほど1位。同値は馬番の昇順（着順は絶対に使わない）。"""
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], nums[i]))
    out = [0] * len(vals)
    for r, i in enumerate(order, 1):
        out[i] = r
    return out


def build_table():
    import xgboost as xgb
    from xgboost import XGBClassifier

    df = pd.read_csv(CSV, low_memory=False)
    df['date'] = df['date'].astype(str).str[:10]
    cols = json.load(open(f'{BASE}/data/xgb_feature_cols.json'))['feature_cols']
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f'CSVに本番の特徴量が無い: {miss[:8]} ({len(miss)}列)')
    df['_n'] = df.groupby('race_id')['horse_num'].transform('size')
    df['_pop'] = df['f_popularity'].fillna(df['_n'] / 2)

    drift = load_popularity_drift(BASE)
    if drift is None:
        raise SystemExit('ドリフト標本が無い。本番と同じ条件で学習できないので中止。')

    out = []
    for w, (te_s, te_e) in enumerate(WINDOWS, 1):
        cut = (pd.Timestamp(te_s) - pd.Timedelta(days=VAL_DAYS)).strftime('%Y-%m-%d')
        tr = df[(df['date'] < cut) & (df['place'] > 0) & (df['place'] < 99)]
        va = df[(df['date'] >= cut) & (df['date'] < te_s) &
                (df['place'] > 0) & (df['place'] < 99)]
        te = df[(df['date'] >= te_s) & (df['date'] <= te_e)]
        if len(te) == 0:
            continue

        def _bm(d, seed):
            p = pd.Series(_apply_popularity_drift(
                d['_pop'], d['race_id'].values, drift, seed=seed))
            return _popularity_to_base_margin(p, d['_n'])

        spw = round((1 - tr['is_fukusho'].mean()) / max(tr['is_fukusho'].mean(), .01), 2)
        params = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1,
            reg_lambda=1.0, scale_pos_weight=spw, eval_metric='logloss',
            early_stopping_rounds=50, random_state=SEED_MODEL, n_jobs=-1,
        ).get_xgb_params()
        params['eval_metric'] = 'logloss'

        dtr = xgb.DMatrix(tr[cols].fillna(5.0), label=tr['is_fukusho'],
                          feature_names=cols)
        dtr.set_base_margin(_bm(tr, SEED_DRIFT_TRAIN))
        dva = xgb.DMatrix(va[cols].fillna(5.0), label=va['is_fukusho'],
                          feature_names=cols)
        dva.set_base_margin(_bm(va, SEED_DRIFT_TRAIN))
        bst = xgb.train(params, dtr, num_boost_round=500, evals=[(dva, 'val')],
                        early_stopping_rounds=50, verbose_eval=False)

        for tag, seed in (('confirmed', None), ('drifted', SEED_DRIFT_TEST)):
            if seed is None:
                pop_te = te['_pop'].values.astype(float)
                bm_te = _popularity_to_base_margin(te['_pop'], te['_n'])
            else:
                p = pd.Series(_apply_popularity_drift(
                    te['_pop'], te['race_id'].values, drift, seed=seed))
                pop_te = p.values.astype(float)
                bm_te = _popularity_to_base_margin(p, te['_n'])
            dte = xgb.DMatrix(te[cols].fillna(5.0), feature_names=cols)
            dte.set_base_margin(bm_te)
            raw = bst.predict(dte, output_margin=True)
            out.append(pd.DataFrame({
                'win': w, 'variant': tag, 'race_id': te['race_id'].values,
                'date': te['date'].values, 'horse_num': te['horse_num'].values,
                'place': te['place'].values, 'n': te['_n'].values,
                'pop_raw': pop_te, 'ability': raw - np.asarray(bm_te),
                'total': raw,
            }))
        print(f'  W{w}: train {len(tr):,}行(<{cut}) / val {len(va):,}行 → '
              f'test {len(te):,}行 ({te["date"].min()}..{te["date"].max()}) '
              f'best_iter={bst.best_iteration}', flush=True)

    t = pd.concat(out, ignore_index=True)

    # 適性順位（本番の表示列 `src/features/aptitude.py` と同じ5列・同じ z 化）
    apt = [c for c, _ in APT_COLS]
    fit_map = {}
    for rid, g in df.groupby('race_id'):
        nums = [int(v) for v in g['horse_num'].values]
        sc, _used = aptitude_scores(g[apt].to_dict('records'))
        if not sc:
            fit_map.update({(rid, n): 99 for n in nums})
            continue
        for n, r in zip(nums, _rank_desc(list(sc), nums)):
            fit_map[(rid, n)] = r

    rows = []
    for (rid, var), g in t.groupby(['race_id', 'variant'], sort=False):
        nums = [int(v) for v in g['horse_num'].values]
        solo = _rank_desc(list(g['ability'].values), nums)
        rl = _rank_desc(list(g['total'].values), nums)
        pop = _rank_desc(list(-g['pop_raw'].values), nums)
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({
                'race_id': rid, 'variant': var, 'win': int(r['win']),
                'date': r['date'], 'horse_num': nums[i], 'n': int(r['n']),
                'place': int(r['place']), 'pop': pop[i],
                'pop_raw': float(r['pop_raw']),
                'solo': solo[i], 'rl': rl[i],
                'fit': fit_map.get((rid, nums[i]), 99),
                'ability': float(r['ability']),
            })
    tab = pd.DataFrame(rows)
    tab['top3'] = tab['place'].between(1, 3).astype(int)
    tab['year'] = tab['date'].str[:4]

    con = sqlite3.connect(f'file:{BASE}/data/history.db?mode=ro', uri=True)
    pay = pd.read_sql("SELECT race_id, horse_num, fukusho_payout FROM horse_history"
                      f" WHERE date >= '{B_START}'", con)
    cls = pd.read_sql("SELECT race_id, race_class FROM race_history"
                      f" WHERE date >= '{B_START}'", con)
    con.close()
    tab = tab.merge(pay, on=['race_id', 'horse_num'], how='left')
    return tab.merge(cls, on='race_id', how='left')


# ───────── 統計量 ─────────

def stat(d, mask, pool_pop):
    """β = 3着内率(該当) − 3着内率(同じ人気帯の非該当)  [pt]"""
    pool = d[d['pop'] <= pool_pop]
    sel = pool[mask(pool)]
    rest = pool[~mask(pool)]
    if len(sel) == 0 or len(rest) == 0:
        return float('nan'), float('nan'), len(sel)
    return (100 * sel['top3'].mean(),
            100 * (sel['top3'].mean() - rest['top3'].mean()), len(sel))


def recovery(d):
    """複勝1点100円の回収率[%]。3着内なのに払戻が無い行は評価不能として除く。"""
    bad = d[(d['top3'] == 1) & ~(d['fukusho_payout'] > 0)]
    ok = d.drop(bad.index)
    if len(ok) == 0:
        return float('nan'), 0, len(bad)
    ret = np.where(ok['top3'] == 1, ok['fukusho_payout'].fillna(0), 0.0).sum()
    return 100 * ret / (100 * len(ok)), len(ok), len(bad)


def wilson(k, n):
    if n == 0:
        return (float('nan'), float('nan'))
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


# ───────── 対照群（**本データの結果を読む前に**通す） ─────────

def control_b(tab):
    """対照群B＝注入の復元。合格は「注入後 − 注入前 = 注入量」。"""
    d = tab[tab['variant'] == 'confirmed'].copy()
    rng = np.random.default_rng(777)
    print('\n■ 対照群B（注入の復元）')

    # B1 経路の復元: 既知の群Gだけが solo>=6 になるよう作り替える
    pool = d[d['pop'] == 1].copy()
    g_idx = rng.choice(pool.index.values, size=int(len(pool) * 0.10), replace=False)
    d2 = d.copy()
    d2.loc[d2['pop'] == 1, 'solo'] = 1
    d2.loc[g_idx, 'solo'] = 18
    rate, _b, n = stat(d2, lambda x: x['solo'] >= 6, 1)
    truth = 100 * d.loc[g_idx, 'top3'].mean()
    ok1 = (n == len(g_idx)) and abs(rate - truth) < 1e-9
    print(f'  B1 経路  : 選別 {n} = 注入 {len(g_idx)} / '
          f'3着内率 {rate:.4f} vs 独立計算 {truth:.4f}  → {"✅" if ok1 else "🔴"}')

    # B2 効果量の復元: 該当群の outcome に +δ を注入
    delta = 20.0
    before = stat(d, lambda x: x['solo'] >= 6, 1)[1]
    d3 = d.copy()
    d3['top3'] = d3['top3'].astype(float)
    m = (d3['pop'] == 1) & (d3['solo'] >= 6)
    d3.loc[m, 'top3'] = d3.loc[m, 'top3'] + delta / 100.0
    after = stat(d3, lambda x: x['solo'] >= 6, 1)[1]
    ok2 = abs((after - before) - delta) < 1e-9
    print(f'  B2 効果量: 注入後 {after:+.4f} − 注入前 {before:+.4f} = '
          f'{after - before:+.4f}pt（注入量 {delta:+.1f}） → {"✅" if ok2 else "🔴"}')
    return ok1 and ok2


def control_a(tab, conds):
    """対照群A＝レース内シャッフル200回。合格は帰無が0中心。"""
    d = tab[tab['variant'] == 'confirmed'].copy()
    rng = np.random.default_rng(PERM_SEED)
    groups = [g.index.values for _, g in d.groupby('race_id', sort=False)]
    print(f'\n■ 対照群A（レース内シャッフル {N_PERM}回・solo を振り直す）')
    res = {}
    base_solo = d['solo'].values.copy()
    pos = {ix: i for i, ix in enumerate(d.index.values)}
    gpos = [np.array([pos[i] for i in g]) for g in groups]
    real = {name: stat(d, f, p)[1] for name, f, p in conds}
    shuf = {name: [] for name, _, _ in conds}
    for _k in range(N_PERM):
        s = base_solo.copy()
        for gp in gpos:
            s[gp] = rng.permutation(s[gp])
        d['solo'] = s
        for name, f, p in conds:
            shuf[name].append(stat(d, f, p)[1])
    d['solo'] = base_solo
    for name, _f, _p in conds:
        a = np.array(shuf[name], dtype=float)
        a = a[~np.isnan(a)]
        line = 2 * a.std(ddof=1) / np.sqrt(len(a))
        ok = abs(a.mean()) < line
        pv = (np.sum(a <= real[name]) + 1) / (len(a) + 1)   # 片側（下がる向き）
        res[name] = dict(mean=a.mean(), std=a.std(ddof=1), line=line,
                         ok=bool(ok), p=pv, real=real[name])
        print(f'  {name:32} 帰無 mean {a.mean():+.3f}pt (std {a.std(ddof=1):.3f}) '
              f'/ 合格線 {line:.3f} → {"✅" if ok else "🔴"}   '
              f'実測 β {real[name]:+.2f}pt  並べ替えP={pv:.4f}')
    return res


# ───────── 本測定 ─────────

def main():
    tab = build_table()
    conf = tab[tab['variant'] == 'confirmed']
    drif = tab[tab['variant'] == 'drifted']
    print(f'\nB期間: {conf["date"].min()}〜{conf["date"].max()}  '
          f'{conf["race_id"].nunique():,}レース / {len(conf):,}頭')

    # 計測器の検査: ability は base_margin に依存しないはず（別経路で突合）
    j = conf.merge(drif, on=['race_id', 'horse_num'], suffixes=('_c', '_d'))
    dmax = (j['ability_c'] - j['ability_d']).abs().max()
    agree = (j['solo_c'] == j['solo_d']).mean()
    print(f'検査: |ability(確定) − ability(ドリフト)| 最大 {dmax:.2e} / '
          f'solo 一致率 {100*agree:.2f}%  → {"✅" if dmax < 1e-4 else "🔴"}')
    # pop は「レース内での人気値の順位」。確定人気が綺麗なら値そのものと一致する
    same = (conf['pop'] == conf['pop_raw'].round()).mean()
    print(f'検査: pop(順位) == 確定人気の値  {100*same:.2f}%'
          f'  → {"✅" if same > 0.98 else "🔴"}')

    if not control_b(tab):
        raise SystemExit('🔴 対照群B 不合格。計測器が壊れているので本データを読まない。')
    ctrl_a = control_a(tab, CONDS)

    print('\n■ 主測定（確定人気）')
    print(f'{"条件":34} {"N":>6} {"3着内率":>8} {"95%CI":>16} {"β":>8} '
          f'{"2025":>8} {"2026":>8} {"複勝回収":>8}')
    rows = []
    for name, f, p in CONDS:
        for tag, d in (('確定', conf), ('ドリフト', drif)):
            pool = d[d['pop'] <= p]
            sel = pool[f(pool)]
            rate, b, n = stat(d, f, p)
            ci = wilson(int(sel['top3'].sum()), len(sel))
            r25 = stat(d[d['year'] == '2025'], f, p)[0]
            r26 = stat(d[d['year'] == '2026'], f, p)[0]
            rec, nok, nbad = recovery(sel)
            if tag == '確定':
                print(f'{name:34} {n:6,} {rate:7.1f}% '
                      f'[{ci[0]:5.1f},{ci[1]:5.1f}] {b:+7.1f} '
                      f'{r25:7.1f}% {r26:7.1f}% {rec:7.1f}%')
            rows.append(dict(cond=name, variant=tag, n=n, rate=rate, beta=b,
                             ci_lo=ci[0], ci_hi=ci[1], r2025=r25, r2026=r26,
                             rec=rec, n_rec=nok, n_bad=nbad))
    pd.DataFrame(rows).to_csv(f'{OUT}/_main.csv', index=False)

    print('\n■ 標準ゲート（市場単独 / AI単独 / AI×市場）')
    allp = conf
    for label, sub in (
            ('市場単独  pop==1（全体）', allp[allp['pop'] == 1]),
            ('AI単独    solo>=6（人気を問わない）', allp[allp['solo'] >= 6]),
            ('AI×市場  pop==1 & solo>=6 (=V0)',
             allp[(allp['pop'] == 1) & (allp['solo'] >= 6)]),
            ('（参考）全馬', allp)):
        rec, _n, _b = recovery(sub)
        print(f'  {label:36} N {len(sub):7,}  3着内率 {100*sub["top3"].mean():5.1f}%'
              f'  複勝回収 {rec:6.1f}%')

    print('\n■ 方向別（gap = 市場人気 − AI単独順位。負＝AIが market より低く見る）')
    g = allp.copy()
    g['gap'] = g['pop'] - g['solo']
    bins = [(-99, -4, 'AI弱気 -4以下'), (-3, -1, 'AI弱気 -3〜-1'), (0, 0, '一致 0'),
            (1, 3, 'AI強気 +1〜+3'), (4, 99, 'AI強気 +4以上')]
    for lo, hi, lab in bins:
        s = g[(g['gap'] >= lo) & (g['gap'] <= hi)]
        rec, _n, _b = recovery(s)
        print(f'  {lab:16} N {len(s):7,}  3着内率 {100*s["top3"].mean():5.1f}%'
              f'  複勝回収 {rec:6.1f}%')

    print('\n■ 基準4 層別（オッズ帯は win_odds 0%充足で不能 → 頭数帯・クラス）')
    v0 = lambda x: x['solo'] >= 6   # noqa: E731
    for lab, sub in (('〜9頭', conf[conf['n'] <= 9]),
                     ('10-13頭', conf[(conf['n'] >= 10) & (conf['n'] <= 13)]),
                     ('14頭〜', conf[conf['n'] >= 14])):
        rate, b, n = stat(sub, v0, 1)
        print(f'  {lab:10} N {n:5,}  3着内率 {rate:5.1f}%  β {b:+6.1f}pt')
    for cl in ('新馬', '未勝利', '1勝', '2勝', '3勝', 'OP'):
        sub = conf[conf['race_class'].astype(str).str.contains(cl, na=False)]
        if len(sub) == 0:
            continue
        rate, b, n = stat(sub, v0, 1)
        print(f'  {cl:10} N {n:5,}  3着内率 {rate:5.1f}%  β {b:+6.1f}pt')

    print('\n■ 警告の価値（1番人気を複勝1点で買う／V0該当だけ買わない）')
    p1 = conf[conf['pop'] == 1]
    for lab, sub in (('全部買う', p1),
                     ('V0該当を外す', p1[~(p1['solo'] >= 6)]),
                     ('（参考）V0該当だけ買う', p1[p1['solo'] >= 6])):
        rec, nok, nbad = recovery(sub)
        print(f'  {lab:24} N {nok:6,}（評価不能 {nbad}）  複勝回収 {rec:6.1f}%')

    print('\n■ ◎（rl_rank==1）との重なり')
    for name, f, p in CONDS:
        pool = conf[conf['pop'] <= p]
        sel = pool[f(pool)]
        if len(sel) == 0:
            continue
        ax = sel[sel['rl'] == 1]
        ot = sel[sel['rl'] != 1]
        print(f'  {name:34} ◎と同じ {len(ax):5,}頭 3着内 {100*ax["top3"].mean() if len(ax) else float("nan"):5.1f}%'
              f'  ／ ◎でない {len(ot):5,}頭 {100*ot["top3"].mean() if len(ot) else float("nan"):5.1f}%')

    tab.to_csv(f'{OUT}/_rows.csv.gz', index=False, compression='gzip')
    json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                   for kk, vv in v.items()} for k, v in ctrl_a.items()},
              open(f'{OUT}/_ctrl_a.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
