"""「市場の支持ほどAIは評価していない馬」は1着に来ないか。事前登録は `danger/CRITERIA_WIN.md`。

🔴 **モデルの再学習はしない。** 2026-10-16 の3着内の測定（`danger/bperiod.py`）が
   出した行をそのまま読み、結果変数だけ `top3` → `place == 1` に差し替える。
   同じ3窓・同じ seed・同じ134列・同じ `best_iteration=10`。

🔴 **B期間は3着内で既に読んでいる。** 1着は3着内の部分集合なので、これは独立した
   証拠ではなく「既に見たデータの別の切り口」である（カード §0）。

実装上の選択（**結果を読む前に**ここに書いた）

- 主判定は **V0（本番の条件）**だけ。他5条件は参考として同じ表に出すが、
  N は結果変数に依らず3着内の測定と同一なので、V1b(155) / V2(100) は
  基準2（N>=300）を**最初から満たさない**。この事実はカードを書く前から確定している。
- 前半/後半は **2025年 / 2026年**（構造で決める。中央値で割らない）。
- 変種は `confirmed`（確定人気＝主）と `drifted`（朝の情報量＝副次）。
  3着内の測定で drifted の「1番人気」は31.6%が別の馬だと分かっているので、
  **確定1番人気に揃えた場合も併記する**。
- 単勝回収率は `history.db` の `tansho_payout` から取る。**判定には使わない**。
  単勝プール平均は控除率20%で80%なので、構造上100%は超えない。
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

SCR = ('/tmp/claude-0/-home-user-keiba-ai/'
       '00b2457c-aac1-5a17-b0c5-39bb7cf987a1/scratchpad/danger')
ROWS = os.environ.get('DANGER_ROWS', os.path.join(SCR, '_rows.csv.gz'))
HDB = os.environ.get('DANGER_HDB', os.path.join(SCR, 'history_body.db'))
OUT = os.environ.get('DANGER_OUT', SCR)

N_PERM = 200
PERM_SEED = 20261016

# カード §1 の条件は V0。他は3着内カードから持ち越した参考。
CONDS = [
    ('V0  pop==1 & solo>=6',          lambda d: d['solo'] >= 6,                      1),
    ('V1a pop==1 & solo>=8',          lambda d: d['solo'] >= 8,                      1),
    ('V1b pop==1 & solo>=10',         lambda d: d['solo'] >= 10,                     1),
    ('V2  pop==1 & solo>=6 & rl>=2',  lambda d: (d['solo'] >= 6) & (d['rl'] >= 2),   1),
    ('V3  pop==1 & solo>=6 & fit>=6', lambda d: (d['solo'] >= 6) & (d['fit'] >= 6),  1),
    ('V4  pop<=2 & solo>=6',          lambda d: d['solo'] >= 6,                      2),
]


def stat(d, mask, pool_pop):
    """β = 勝率(該当) − 勝率(同じ人気帯の非該当)  [pt]"""
    pool = d[d['pop'] <= pool_pop]
    sel = pool[mask(pool)]
    rest = pool[~mask(pool)]
    if len(sel) == 0 or len(rest) == 0:
        return float('nan'), float('nan'), len(sel)
    return (100 * sel['win1'].mean(),
            100 * (sel['win1'].mean() - rest['win1'].mean()), len(sel))


def wilson(k, n):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def tansho_recovery(d):
    """単勝1点100円の回収率[%]。1着なのに払戻が無い行は評価不能として除く。"""
    bad = d[(d['win1'] == 1) & ~(d['tansho_payout'] > 0)]
    ok = d.drop(bad.index)
    if len(ok) == 0:
        return float('nan'), 0, len(bad)
    ret = np.where(ok['win1'] == 1, ok['tansho_payout'].fillna(0), 0.0).sum()
    return 100 * ret / (100 * len(ok)), len(ok), len(bad)


def control_b(tab):
    """対照群B＝注入の復元。合格は「注入後 − 注入前 = 注入量」。"""
    d = tab[tab['variant'] == 'confirmed'].copy()
    rng = np.random.default_rng(777)
    print('\n■ 対照群B（注入の復元）')

    pool = d[d['pop'] == 1].copy()
    g_idx = rng.choice(pool.index.values, size=int(len(pool) * 0.10), replace=False)
    d2 = d.copy()
    d2.loc[d2['pop'] == 1, 'solo'] = 1
    d2.loc[g_idx, 'solo'] = 18
    rate, _b, n = stat(d2, lambda x: x['solo'] >= 6, 1)
    truth = 100 * d.loc[g_idx, 'win1'].mean()
    ok1 = (n == len(g_idx)) and abs(rate - truth) < 1e-9
    print(f'  B1 経路  : 選別 {n} = 注入 {len(g_idx)} / '
          f'勝率 {rate:.4f} vs 独立計算 {truth:.4f}  → {"✅" if ok1 else "🔴"}')

    delta = 20.0
    before = stat(d, lambda x: x['solo'] >= 6, 1)[1]
    d3 = d.copy()
    d3['win1'] = d3['win1'].astype(float)
    m = (d3['pop'] == 1) & (d3['solo'] >= 6)
    d3.loc[m, 'win1'] = d3.loc[m, 'win1'] + delta / 100.0
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
        pv = (np.sum(a <= real[name]) + 1) / (len(a) + 1)
        res[name] = dict(mean=a.mean(), std=a.std(ddof=1), line=line,
                         ok=bool(ok), p=pv, real=real[name])
        print(f'  {name:32} 帰無 mean {a.mean():+.3f}pt (std {a.std(ddof=1):.3f}) '
              f'/ 合格線 {line:.3f} → {"✅" if ok else "🔴"}   '
              f'実測 β {real[name]:+.2f}pt  並べ替えP={pv:.4f}')
    return res


def main():
    tab = pd.read_csv(ROWS)
    tab['win1'] = (tab['place'] == 1).astype(int)

    # ── 中止条件（カード §6）──────────────────────────────
    d = tab[tab['variant'] == 'confirmed'].copy()
    n_v0 = len(d[(d['pop'] == 1) & (d['solo'] >= 6)])
    n_fav = len(d[d['pop'] == 1])
    print('■ 中止条件の検査（カード §6）')
    print(f'  レース {d["race_id"].nunique():,} / 行 {len(d):,} / 1番人気 {n_fav:,}')
    print(f'  V0 該当 {n_v0}  （RESULTS.md の記録 849）'
          f' → {"✅" if n_v0 == 849 else "🔴 中止"}')
    if n_v0 != 849:
        print('  🔴 行が記録と食い違う。勝率を見る前に中止する。')
        return
    rest_rate = 100 * d[(d['pop'] == 1) & (d['solo'] < 6)]['win1'].mean()
    all_rate = 100 * d[d['pop'] == 1]['win1'].mean()
    ok_rest = 20.0 <= rest_rate <= 40.0
    print(f'  非該当1番人気の勝率 {rest_rate:.2f}%（記録 26.5〜34.9%）'
          f' → {"✅" if ok_rest else "🔴 計測器を疑う"}')
    print(f'  （参考）1番人気 全体の勝率 {all_rate:.2f}%'
          f'  ／ 全馬 {100 * d["win1"].mean():.2f}%')
    if not ok_rest:
        print('  🔴 判定に使わない。')
        return

    # ── 対照群（本データの主測定より前に通す）──────────────
    okb = control_b(tab)
    ctrl_a = control_a(tab, CONDS)

    # ── 主測定 ────────────────────────────────────────
    print('\n■ 主測定（confirmed・勝率）')
    print(f'{"条件":34} {"N":>6} {"勝率":>8} {"95%CI":>16} {"β":>8} '
          f'{"2025":>8} {"2026":>8}  判定')
    rows = []
    for name, f, p in CONDS:
        rate, beta, n = stat(d, f, p)
        pool = d[d['pop'] <= p]
        sel = pool[f(pool)]
        lo, hi = wilson(int(sel['win1'].sum()), len(sel))
        h1 = sel[sel['year'] == 2025]
        h2 = sel[sel['year'] == 2026]
        r1 = 100 * h1['win1'].mean() if len(h1) else float('nan')
        r2 = 100 * h2['win1'].mean() if len(h2) else float('nan')
        c1a = rate < 10.0
        c1b = rate < 16.0
        c2 = n >= 300
        c3 = (r1 < 16.0) and (r2 < 16.0)
        verdict = []
        verdict.append('1a✅' if c1a else ('1b✅' if c1b else '1❌'))
        verdict.append('2✅' if c2 else f'2❌(N={n})')
        verdict.append('3✅' if c3 else '3❌')
        print(f'{name:34} {n:>6} {rate:>7.1f}% [{lo:>5.1f},{hi:>5.1f}] '
              f'{beta:>+7.2f} {r1:>7.1f}% {r2:>7.1f}%  {" ".join(verdict)}')
        rows.append(dict(cond=name, n=n, rate=rate, lo=lo, hi=hi, beta=beta,
                         h2025=r1, h2026=r2, n2025=len(h1), n2026=len(h2),
                         c1a=bool(c1a), c1b=bool(c1b), c2=bool(c2), c3=bool(c3)))

    # ── V0 を層別（基準4）────────────────────────────
    print('\n■ V0 の層別（基準4：符号が反転しないか）')
    fav = d[d['pop'] == 1].copy()
    fav['grp'] = np.where(fav['solo'] >= 6, 'sel', 'rest')
    fav['nband'] = pd.cut(fav['n'], [0, 9, 13, 99],
                          labels=['〜9頭', '10-13頭', '14頭〜'])
    for key in ['nband', 'race_class']:
        print(f'  ── {key} ──')
        for v, g in fav.groupby(key, observed=True):
            s = g[g['grp'] == 'sel']
            r = g[g['grp'] == 'rest']
            if len(s) < 10 or len(r) < 10:
                continue
            print(f'    {str(v):10} 該当 {len(s):>5}頭 {100 * s["win1"].mean():>5.1f}%'
                  f' / 非該当 {len(r):>5}頭 {100 * r["win1"].mean():>5.1f}%'
                  f'  β {100 * (s["win1"].mean() - r["win1"].mean()):>+6.2f}pt')

    # ── ◎との重なり ───────────────────────────────
    print('\n■ ◎（rl_rank==1）との重なり')
    sel = fav[fav['grp'] == 'sel']
    is_axis = sel[sel['rl'] == 1]
    not_axis = sel[sel['rl'] >= 2]
    print(f'  該当 {len(sel)}頭 のうち ◎自身 {len(is_axis)}頭 '
          f'({100 * len(is_axis) / len(sel):.1f}%) 勝率 {100 * is_axis["win1"].mean():.1f}%')
    lo, hi = wilson(int(not_axis['win1'].sum()), len(not_axis))
    print(f'  ◎ではない {len(not_axis)}頭 '
          f'({100 * len(not_axis) / len(sel):.1f}%) 勝率 '
          f'{100 * not_axis["win1"].mean():.1f}% [95%CI {lo:.1f}, {hi:.1f}]')
    for y in [2025, 2026]:
        g = not_axis[not_axis['year'] == y]
        if len(g):
            print(f'    {y}年 {len(g)}頭 勝率 {100 * g["win1"].mean():.1f}%')

    # ── drifted（副次）────────────────────────────
    print('\n■ 感度分析（drifted＝朝の情報量・副次）')
    dd = tab[tab['variant'] == 'drifted'].copy()
    conf_fav = set(d[d['pop'] == 1]['race_id'] + '_' +
                   d[d['pop'] == 1]['horse_num'].astype(str))
    for name, f, p in CONDS[:4]:
        rate, beta, n = stat(dd, f, p)
        pool = dd[dd['pop'] <= p]
        s = pool[f(pool)]
        key = s['race_id'] + '_' + s['horse_num'].astype(str)
        same = key.isin(conf_fav)
        r_same = 100 * s[same]['win1'].mean() if same.any() else float('nan')
        print(f'  {name:34} N {n:>5} 勝率 {rate:>5.1f}% β {beta:>+6.2f}pt'
              f'  ／ 確定1番人気と同じ馬 {100 * same.mean():>5.1f}% '
              f'（その群の勝率 {r_same:.1f}%）')

    # ── 単勝回収率（副次・判定に使わない）────────────────
    print('\n■ 単勝回収率（副次・判定に使わない）')
    if os.path.exists(HDB):
        con = sqlite3.connect(f'file:{HDB}?mode=ro', uri=True)
        pay = pd.read_sql_query(
            'SELECT race_id, horse_num, tansho_payout FROM horse_history', con)
        con.close()
        m = fav.merge(pay, on=['race_id', 'horse_num'], how='left')
        for label, g in [('1番人気 全部', m),
                         ('V0 該当', m[m['grp'] == 'sel']),
                         ('V0 非該当', m[m['grp'] == 'rest'])]:
            ret, nok, nbad = tansho_recovery(g)
            print(f'  {label:14} N {nok:>6}（評価不能 {nbad}） 回収 {ret:>6.1f}%')
        a = m[m['grp'] == 'sel']
        b = m[m['grp'] == 'rest']
        ra, na, _ = tansho_recovery(a)
        rb, nb, _ = tansho_recovery(b)
        print(f'  → 該当を外すと {100 * (np.where(m["win1"] == 1, m["tansho_payout"].fillna(0), 0).sum()) / (100 * len(m)):.1f}%'
              f' → {rb:.1f}%（{nb}頭）')
    else:
        print(f'  🔴 {HDB} が無いので計算していない。')

    json.dump(dict(rows=rows, control_a=ctrl_a, control_b=okb,
                   rest_rate=rest_rate, all_fav_rate=all_rate),
              open(os.path.join(OUT, '_win.json'), 'w'),
              ensure_ascii=False, indent=1, default=float)
    print(f'\n→ {os.path.join(OUT, "_win.json")}')


if __name__ == '__main__':
    main()
