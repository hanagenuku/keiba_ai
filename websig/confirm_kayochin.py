#!/usr/bin/env python3
"""kayochin 能力指数の確認測定（事前登録カード `websig/CRITERIA_CONFIRM_kayochin.md`）。

🔴 測定は1回。窓・腕・基準はカードのとおりで、結果を見てから変えない。
🔴 中止条件は AUC・的中率・回収率を読む**手前**で印字して判定する。
🔴 隔離ワークスペースで実行する（data/ に1バイトも書かない）。

実行場所: `.github/workflows/confirm-kayochin.yml`（GitHub のランナー）。指数DBは非公開の
artifact にしか無く、開発環境からは取得できないため。ランナーは毎回まっさらなので隔離になる。

使い方:
    cd <隔離ワークスペース>   # data/horse_features.csv・data/history.db・data/private/web_signals.db
    python3 websig/confirm_kayochin.py
"""
import importlib.util
import json
import os
import sqlite3
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

from src.features.web_signals import _ranks_and_z, feature_cols_for  # noqa: E402
from src.tools.train_xgb import load_popularity_drift  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.getcwd(), rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 既存ハーネスを再利用する（自前で作り直さない・2026-08-08 の教訓）
B = _load('hitrate_eval_b', 'hitrate/eval_b.py')     # feat_cols / base_margins / fit_arm(P0)
L = _load('logic2026_eval', 'logic2026/eval.py')     # pick_axis / roi

SOURCE = 'kayochinkeiba'
RANK_COL, Z_COL = feature_cols_for(SOURCE)
# カード §1 の窓: (名前, 学習の末尾, 評価の先頭, 評価の末尾)
WINDOWS = [('W1', '2026-08-21', '2026-08-22', '2026-09-20'),
           ('W2', '2026-09-20', '2026-09-21', '2026-10-18'),
           ('W3', '2026-10-18', '2026-10-19', '2026-11-08')]
SEEDS = (42, 7, 2026)
INNER_DAYS = 42
# カードは30回。環境変数は**配管の試運転（偽の指数DB）専用**で、本測定では設定しない
N_SHUFFLE = int(os.environ.get('CONFIRM_SMOKE_SHUFFLE', '30'))
SHUFFLE_SEED = 20261111
INJECT_SIGMA, INJECT_SEED = 1.8, 20261109
JST = timezone(timedelta(hours=9))
CUTOFF_HHMM = (8, 50)


def _published_before_cutoff(published_at, race_date):
    """記事の公開時刻が、レース当日の 08:50 JST より前か。読めなければ False（使わない）。"""
    if not published_at:
        return False
    try:
        t = datetime.fromisoformat(str(published_at).replace('Z', '+00:00'))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=JST)
    d = str(race_date).replace('-', '')[:8]
    cut = datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), *CUTOFF_HHMM, tzinfo=JST)
    return t < cut


def load_index(db_path):
    """{race_id: {horse_num: 指数}} と、落とした件数。"""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        'SELECT race_id, horse_num, index_value, published_at, date FROM web_signals '
        'WHERE source = ?', (SOURCE,)).fetchall()
    con.close()
    by_race, dropped_late, dropped_nopub = {}, 0, 0
    for rid, num, val, pub, date in rows:
        if num is None or val is None:
            continue
        if not pub:
            dropped_nopub += 1
            continue
        if not _published_before_cutoff(pub, date):
            dropped_late += 1
            continue
        by_race.setdefault(rid, {})[int(num)] = float(val)
    return by_race, dict(total=len(rows), late=dropped_late, no_published_at=dropped_nopub)


def index_columns(df, by_race, rng=None):
    """df の行順で (rank, z) を返す。rng があればレース内で指数を並べ替えてから作る（対照群A）。"""
    rank = np.full(len(df), np.nan, dtype=np.float32)
    z = np.full(len(df), np.nan, dtype=np.float32)
    rid_arr = df['race_id'].to_numpy()
    num_arr = df['horse_num'].to_numpy()
    cache = {}
    for i in range(len(df)):
        rid = rid_arr[i]
        vals = by_race.get(rid)
        if not vals:
            continue
        if rid not in cache:
            v = vals
            if rng is not None:
                nums = sorted(v)
                perm = rng.permutation(len(nums))
                v = {nums[k]: vals[nums[perm[k]]] for k in range(len(nums))}
            cache[rid] = _ranks_and_z(v)
        ranks, zs = cache[rid]
        try:
            n = int(num_arr[i])
        except (TypeError, ValueError):
            continue
        if n in ranks:
            rank[i], z[i] = ranks[n], zs[n]
    return rank, z


def main():
    base = os.getcwd()
    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date_obj'] = pd.to_datetime(
        df['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date_obj', 'is_fukusho']).reset_index(drop=True)
    cols = B.feat_cols(df)
    prod_cols = list(json.load(open(os.path.join(
        base, 'data', 'xgb_feature_cols_residual.json')))['feature_cols'])
    drift = load_popularity_drift(base)

    XA = df[cols].fillna(5.0).to_numpy(dtype=np.float32)
    y = df['is_fukusho'].to_numpy(dtype=np.int8)
    d = df['date_obj'].to_numpy()
    bm_tr = B.base_margins(df, drift, B.DRIFT_SEED_TRAIN)
    bm_ev = B.base_margins(df, drift, B.DRIFT_SEED_EVAL)

    by_race, drop = load_index(os.path.join(base, 'data', 'private', 'web_signals.db'))
    rank, z = index_columns(df, by_race)
    XX = np.column_stack([XA, rank, z]).astype(np.float32)        # NaN は NaN のまま
    colsX = cols + [RANK_COL, Z_COL]

    rng_inj = np.random.default_rng(INJECT_SEED)
    zinj = y.astype(float) + rng_inj.normal(0, INJECT_SIGMA, len(y))
    zinj_rank = (pd.Series(-zinj).groupby(df['race_id']).rank(method='average')
                 .to_numpy(dtype=np.float32))
    XZ = np.column_stack([XA, zinj_rank, zinj]).astype(np.float32)

    masks = {}
    for name, te, v0, v1 in WINDOWS:
        te, v0, v1 = map(pd.Timestamp, (te, v0, v1))
        inner_from = te - pd.Timedelta(days=INNER_DAYS)
        masks[name] = (d <= np.datetime64(inner_from),
                       (d > np.datetime64(inner_from)) & (d <= np.datetime64(te)),
                       (d >= np.datetime64(v0)) & (d <= np.datetime64(v1)))

    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    pay = pd.read_sql_query(
        """SELECT h.race_id, h.horse_num, h.place, h.fukusho_payout, h.tansho_payout
           FROM horse_history h JOIN race_history r USING(race_id)
           WHERE r.date >= ? AND r.date <= ?""",
        con, params=(WINDOWS[0][2], WINDOWS[-1][3]))
    con.close()

    def fit(Xm, cs, w, seed):
        mf, mi, mv = masks[w]
        return B.fit_arm('P0', Xm[mf], y[mf], bm_tr[mf], Xm[mi], y[mi], bm_tr[mi],
                         Xm[mv], bm_ev[mv], cs, seed)

    fits = {}
    for w, *_ in WINDOWS:
        for sd in SEEDS:
            fits[('A', w, sd)] = fit(XA, cols, w, sd)
            fits[('X', w, sd)] = fit(XX, colsX, w, sd)
            print(f'  …fit A/X {w} seed={sd}', flush=True)
        fits[('XZ', w, 42)] = fit(XZ, colsX, w, 42)

    # ── 中止条件（結果より手前）─────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §4）— AUC の差・的中率・回収率を読む手前で判定する')
    print('=' * 78)
    stop = []
    ok = set(cols) == set(prod_cols)
    print(f'  1. 特徴量列 {len(cols)}（本番 {len(prod_cols)}）集合一致: {ok}')
    if not ok:
        stop.append('特徴量列が本番と一致しない')

    def pooled(arm, sd):
        ys = np.concatenate([y[masks[w][2]] for w, *_ in WINDOWS])
        ps = np.concatenate([fits[(arm, w, sd)]['p_all'] for w, *_ in WINDOWS])
        return ys, ps

    a_auc = [roc_auc_score(*pooled('A', sd)) for sd in SEEDS]
    ok = all(0.77 <= a <= 0.81 for a in a_auc)
    print(f'  2. A のプール AUC = {[round(a, 4) for a in a_auc]}  0.77〜0.81 の内か: {ok}')
    if not ok:
        stop.append(f'A のプール AUC が帯の外: {a_auc}')
    its = [fits[('A', w, sd)]['it'] for w, *_ in WINDOWS for sd in SEEDS]
    ok = all(7 <= i <= 15 for i in its)
    print(f'  3. A の best_iteration = {its}  7〜15 の内か: {ok}')
    if not ok:
        stop.append(f'A の best_iteration が帯の外: {its}')
    cover = {}
    for w, *_ in WINDOWS:
        mv = masks[w][2]
        g = pd.DataFrame({'r': df['race_id'].to_numpy()[mv], 'has': ~np.isnan(z[mv])})
        per = g.groupby('r')['has'].any()
        cover[w] = dict(races=int(len(per)), with_index=int(per.sum()),
                        race_cov=round(float(per.mean()), 4),
                        horse_cov=round(float(g['has'].mean()), 4))
        print(f'  4. {w} 評価 {int(mv.sum()):,}頭 / {len(per)}レース  '
              f'指数ありレース {per.mean():.1%}  指数あり頭 {g["has"].mean():.1%}')
        if per.mean() < 0.60:
            stop.append(f'{w} の被覆率が 60% 未満')
    mv_all = np.zeros(len(df), bool)
    for w, *_ in WINDOWS:
        mv_all |= masks[w][2]
    has = mv_all & ~np.isnan(z)
    solo = roc_auc_score(y[has], z[has]) if has.sum() > 100 else float('nan')
    print(f'  5. 指数単独（z）の評価窓 AUC = {solo:.4f}  0.70 以上か: {solo >= 0.70}')
    if not (solo >= 0.70):
        stop.append(f'指数単独の AUC {solo} が 0.70 未満')
    print(f'  6. 指数 {drop["total"]:,}行のうち 当日08:50以降公開で落とした {drop["late"]:,} / '
          f'公開時刻なしで落とした {drop["no_published_at"]:,}（残した行はすべて締切より前）')
    inj_auc = roc_auc_score(y[mv_all], zinj[mv_all])
    ok = 0.63 <= inj_auc <= 0.68
    print(f'  7. 注入列 z 単独の AUC = {inj_auc:.4f}  0.63〜0.68 の内か: {ok}')
    if not ok:
        stop.append(f'注入列の AUC {inj_auc} が帯の外')

    out = {'windows': WINDOWS, 'coverage': cover, 'dropped': drop,
           'a_best_iteration': its, 'solo_auc': solo, 'inject_solo_auc': inj_auc}
    if stop:
        print('\n🔴 中止条件に該当。結果を読まずに破棄する:')
        for s in stop:
            print('   -', s)
        out['stopped'] = stop
        json.dump(out, open('confirm_kayochin_result.json', 'w'), ensure_ascii=False, indent=2)
        return
    print('\n  ✅ 中止条件はすべて通過。測定に進む。')

    # ── 対照群A: レース内で指数を並べ替える（30回・seed 42）───────────────
    rng = np.random.default_rng(SHUFFLE_SEED)
    a42 = roc_auc_score(*pooled('A', 42))
    null = []
    for r in range(N_SHUFFLE):
        rk, zz = index_columns(df, by_race, rng=rng)
        XR = np.column_stack([XA, rk, zz]).astype(np.float32)
        ps = np.concatenate([fit(XR, colsX, w, 42)['p_all'] for w, *_ in WINDOWS])
        ys = np.concatenate([y[masks[w][2]] for w, *_ in WINDOWS])
        null.append(roc_auc_score(ys, ps) - a42)
        if (r + 1) % 5 == 0:
            print(f'  …shuffle {r + 1}/{N_SHUFFLE}', flush=True)
    null = np.array(null)

    # ── 本測定 ─────────────────────────────────────────────
    ev = pd.concat([df.loc[masks[w][2], ['race_id', 'horse_num']] for w, *_ in WINDOWS])

    def metrics(arm, sd):
        ys, ps = pooled(arm, sd)
        ax = L.pick_axis(ev, ps)[['race_id', 'horse_num']]
        f, t = L.roi(ax, pay, 'fukusho'), L.roi(ax, pay, 'tansho')
        return dict(auc=roc_auc_score(ys, ps), brier=brier_score_loss(ys, ps),
                    logloss=log_loss(ys, ps), fuku=f, tan=t)

    res = {arm: {sd: metrics(arm, sd) for sd in SEEDS} for arm in ('A', 'X')}
    win = {}
    for w, *_ in WINDOWS:
        mv = masks[w][2]
        dA = [roc_auc_score(y[mv], fits[('X', w, sd)]['p_all'])
              - roc_auc_score(y[mv], fits[('A', w, sd)]['p_all']) for sd in SEEDS]
        win[w] = dict(delta_by_seed=[round(x, 5) for x in dA], delta_mean=float(np.mean(dA)),
                      auc_A=float(np.mean([roc_auc_score(y[mv], fits[('A', w, sd)]['p_all'])
                                           for sd in SEEDS])))
    d_pool = [res['X'][sd]['auc'] - res['A'][sd]['auc'] for sd in SEEDS]
    d_mean, d_std = float(np.mean(d_pool)), float(np.std(d_pool, ddof=1))

    def avg(arm, k1, k2=None):
        return float(np.mean([res[arm][sd][k1] if k2 is None else res[arm][sd][k1][k2]
                              for sd in SEEDS]))

    # 指数がある行だけ（報告のみ）
    sub = ~np.isnan(np.concatenate([z[masks[w][2]] for w, *_ in WINDOWS]))
    d_sub = float(np.mean([roc_auc_score(pooled('X', sd)[0][sub], pooled('X', sd)[1][sub])
                           - roc_auc_score(pooled('A', sd)[0][sub], pooled('A', sd)[1][sub])
                           for sd in SEEDS]))
    # 対照群B
    ysz = np.concatenate([y[masks[w][2]] for w, *_ in WINDOWS])
    psz = np.concatenate([fits[('XZ', w, 42)]['p_all'] for w, *_ in WINDOWS])
    d_inj = roc_auc_score(ysz, psz) - a42
    axA = L.pick_axis(ev, pooled('A', 42)[1])[['race_id', 'horse_num']]
    bonus = L.roi(axA, pay, 'fukusho', bonus=5.0)['roi'] - L.roi(axA, pay, 'fukusho')['roi']
    mkt = roc_auc_score(ysz, np.concatenate([bm_ev[masks[w][2]] for w, *_ in WINDOWS]))

    print('\n' + '=' * 78)
    print('■ 結果')
    print('=' * 78)
    for w in win:
        print(f'  {w}: ΔAUC（X−A）シード別 {win[w]["delta_by_seed"]}  平均 {win[w]["delta_mean"]:+.5f}'
              f'  （A の AUC {win[w]["auc_A"]:.4f}）')
    print(f'  プール ΔAUC シード別 {[round(x, 5) for x in d_pool]}  平均 {d_mean:+.5f}  std {d_std:.5f}')
    print(f'  プール AUC  A {avg("A", "auc"):.4f} / X {avg("X", "auc"):.4f} / 市場単独 {mkt:.4f} / 指数単独 {solo:.4f}')
    print(f'  指数がある行だけの ΔAUC（報告のみ） {d_sub:+.5f}')
    print(f'  Brier  A {avg("A", "brier"):.5f} / X {avg("X", "brier"):.5f}   '
          f'LogLoss  A {avg("A", "logloss"):.5f} / X {avg("X", "logloss"):.5f}')
    for kind, lab in (('fuku', '複勝'), ('tan', '単勝')):
        print(f'  ◎{lab}  的中率 A {avg("A", kind, "rate"):.2f}% / X {avg("X", kind, "rate"):.2f}%   '
              f'回収率 A {avg("A", kind, "roi"):.2f}% / X {avg("X", kind, "roi"):.2f}%   '
              f'3本抜き A {avg("A", kind, "cut3"):.2f}% / X {avg("X", kind, "cut3"):.2f}%   '
              f'N {res["A"][42][kind]["n"]}')
    print(f'  対照群A（並べ替え {N_SHUFFLE}回）ΔAUC 平均 {null.mean():+.5f}  std {null.std(ddof=1):.5f}'
          f'  線 {2 * null.std(ddof=1) / np.sqrt(N_SHUFFLE):.5f}'
          f'  実測の並べ替え P = {(np.sum(null >= (res["X"][42]["auc"] - a42)) + 1) / (N_SHUFFLE + 1):.3f}')
    print(f'  対照群B  注入列の ΔAUC {d_inj:+.4f}（+0.01 以上か）  roi(bonus=5) の差 {bonus:+.2f}pt')

    c1a = all(win[w]['delta_mean'] > 0 for w in win)
    c1b = d_mean >= 0.002 and d_mean >= 2 * d_std
    c2 = avg('X', 'fuku', 'rate') >= avg('A', 'fuku', 'rate')
    c3a = abs(null.mean()) < 2 * null.std(ddof=1) / np.sqrt(N_SHUFFLE)
    c3b = d_inj >= 0.01 and abs(bonus - 5.0) < 1e-6
    c4 = (avg('X', 'fuku', 'roi') - avg('A', 'fuku', 'roi') >= -2.0
          and avg('X', 'fuku', 'cut3') - avg('A', 'fuku', 'cut3') >= -2.0)
    print('\n■ 基準（カード §3）')
    for k, v in (('1a 3窓すべて ΔAUC>0', c1a), ('1b プール +0.002 以上かつ std×2 以上', c1b),
                 ('2  ◎の複勝的中率 X>=A', c2), ('3  対照群A', c3a), ('3  対照群B', c3b),
                 ('4  複勝回収率の悪化が2pt以内（3本抜きも）', c4)):
        print(f'  {"✅" if v else "🔴"} {k}')
    if not (c3a and c3b):
        verdict = '計測器の不具合（採否に使わない）'
    elif c1a and c1b and c2 and c4:
        verdict = '採用'
    elif c1b and c2 and c4 and not c1a:
        verdict = '保留'
    else:
        verdict = '不採用（v2）'
    print(f'\n■ 判定（カード §5）: {verdict}')

    out.update(dict(
        windows_result=win, pooled_delta=d_pool, pooled_delta_mean=d_mean, pooled_delta_std=d_std,
        arms={a: {str(sd): res[a][sd] for sd in SEEDS} for a in res}, market_auc=mkt,
        subset_delta=d_sub, control_a=dict(mean=float(null.mean()), std=float(null.std(ddof=1)),
                                           values=[float(x) for x in null]),
        control_b=dict(inject_delta=float(d_inj), bonus_delta=float(bonus)),
        criteria=dict(c1a=bool(c1a), c1b=bool(c1b), c2=bool(c2), c3a=bool(c3a), c3b=bool(c3b),
                      c4=bool(c4)), verdict=verdict))
    json.dump(out, open('confirm_kayochin_result.json', 'w'), ensure_ascii=False, indent=2,
              default=float)


if __name__ == '__main__':
    main()
