"""Betting Policy Step 1 — 能力 × 適性 → 結果（3着内）。

事前登録: betting/CRITERIA_STEP1.md（測定より前にコミット済み・edb7b26）。
🔴 閾値・構成列・符号・重み・期間・母集団・セルの切り方を結果を見てから変更しない。

順序（Gate 0 と同じ。対照群を本データより先に通す）:
  1) 特徴量を作る（目的変数を読まない）
  2) 対照群B 注入の復元      合格 注入後 − 注入前 = 注入量（許容差 1e-3）
  3) 対照群A 並べ替え分布200回 合格 |mean| < 2*std/sqrt(200)
  4) 本データ 基準1〜3

使い方: python3 betting/step1.py <history.db> [cache.npz]
"""
import json
import os
import pickle
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'betting')
import src.features.engine as eng
from src.features.engine import init_engine
from src.features.pl_rating import PLRatings
from noryoku.inv1 import build_race, feats_of
from gate0 import (ABILITY_COL, MIN_HORSES, T_COLS, _signed, _zrace,
                   desc_rank, preload_finish, spearman)

A1 = ('2023-01-07', '2023-12-31')
A2 = ('2024-01-01', '2024-12-31')
N_SHUFFLE = 200
N_PERM_RHO = 10000
SEED = 20260922

# 感度分析（事前登録）: f_course_fit_score を外した4列版
T_COLS_4 = [(c, m) for c, m in T_COLS if c != 'f_course_fit_score']


def t_from(xs, cols):
    parts = []
    for c, mode in cols:
        raw = [x.get(c) for x in xs]
        if any(v is None for v in raw):
            continue
        z = _zrace(_signed(raw, mode))
        if z is not None:
            parts.append(z)
    return (np.mean(parts, axis=0) if parts else None)


def tercile_w(rank, n):
    """レース内パーセンタイルの3分位への**重み**（各行の合計 1）。0=上位 1=中位 2=下位。

    🔴 2026-09-22・対照群A が捕まえた計測器の偏りの修正。
      当初は pct=(rank-0.5)/n の中点で離散的に割り当てていたが、**同値は平均順位**
      （Gate 0 で確定）なので、**同値グループは必ず同じ分位に入る**。t5 は
      800レース中 738レース（92%）で同値を持ち（中央で4頭が潰れる）、その結果
      レースごとに T上位/下位の群サイズが大きく揺れ、T をレース内でシャッフルしても
      帰無が 0 中心にならなかった（実測 mean **−0.43pt** / 合格線 0.137）。
      機序は切り分け済み:
          本実装(同値あり)        mean −0.4310  ❌
          同値を 1e-9 で壊す       mean +0.0111  ✅
          t5 を完全に乱数化(同値なし) mean −0.1368  ✅
      → 同値グループが**本来占める順位区間** [i/n, (i+m)/n] を 3分位の境界と
        重ねて比例配分する。乱数を使わないので決定的で、分位サイズが
        レースによらず n/3 ずつになる。
    ⚠ これは閾値・構成列・符号・期間・母集団の変更ではない。事前登録した
      「pct=(rank-0.5)/n の3分位」の**同値の扱いを定義しただけ**（カードは
      同値の扱いを書いていなかった）。Gate 0 の desc_rank 修正と同型。
    """
    rank = np.asarray(rank, float)
    k = len(rank)
    W = np.zeros((k, 3))
    order = np.argsort(rank, kind='stable')
    bounds = ((0.0, 1.0 / 3), (1.0 / 3, 2.0 / 3), (2.0 / 3, 1.0))
    i = 0
    while i < k:
        j = i
        while j + 1 < k and rank[order[j + 1]] == rank[order[i]]:
            j += 1
        lo, hi = i / k, (j + 1) / k          # この同値グループが占める区間
        for b, (b0, b1) in enumerate(bounds):
            ov = min(hi, b1) - max(lo, b0)
            if ov > 0:
                W[order[i:j + 1], b] = ov / (hi - lo)
        i = j + 1
    return W


# ── 1) 特徴量（目的変数を読まない） ──────────────────────────
def build_cache(db_path, cache_path):
    cols = json.load(open('data/xgb_feature_cols.json'))['feature_cols']
    init_engine('.')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""SELECT race_id, date FROM race_history
                           WHERE date>=? AND date<=? ORDER BY date, race_id""",
                        (A1[0], A2[1])).fetchall()
    print(f'A期間 全 {len(rows):,} レース（A1+A2・サンプリングしない）', flush=True)
    fin_of, _day_of = preload_finish(conn)
    pl = PLRatings()
    eng._PL_RATINGS = pl
    cur = None
    recs, n_skip, t0 = [], 0, time.time()
    for i, row in enumerate(rows):
        rid, d = row['race_id'], str(row['date'])[:10]
        if cur is not None and d != cur:
            pl.advance_day(cur)
        cur = d
        if i % 500 == 0:
            print(f'  {i}/{len(rows)} ({d})  {time.time()-t0:.0f}s', flush=True)
        try:
            race = build_race(conn, rid)
        except Exception:
            race = None
        if race is None or len(race['horses']) < MIN_HORSES:
            n_skip += 1
        else:
            base = feats_of(race, cols)
            abil = [x.get(ABILITY_COL) for x in base]
            t5 = t_from(base, T_COLS)
            t4 = t_from(base, T_COLS_4)
            if t5 is None or any(v is None for v in abil):
                n_skip += 1
            else:
                hn = [h['horse_num'] for h in race['horses']]
                recs.append({
                    'race_id': rid, 'date': d, 'n': len(hn), 'horse_num': hn,
                    'a_rank': desc_rank(np.asarray(abil, float)),
                    't5': np.asarray(t5, float),
                    't4': (None if t4 is None else np.asarray(t4, float)),
                })

        # 🔴 θ の更新は「特徴量を作った後」かつ「スキップしたレースでも」行う。
        #    同日の結果は advance_day までは反映されない
        #    （同日出走馬が互いの結果を見ないようにする）。
        f = fin_of.get(rid)
        if f:
            pl.queue_result(f)

        # スモーク専用の上限（本測定では設定しない）
        _lim = os.environ.get('STEP1_LIMIT')
        if _lim and len(recs) >= int(_lim):
            print(f'  ⚠ STEP1_LIMIT={_lim} に到達（スモーク）')
            break
    if cur is not None:
        pl.advance_day(cur)
    print(f'使用 {len(recs):,} レース（スキップ {n_skip}）  {time.time()-t0:.0f}s')
    with open(cache_path, 'wb') as f:
        pickle.dump(recs, f)
    return recs


# ── 目的変数（対照群を通した後に使う） ───────────────────────
def load_outcome(db_path, recs):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    want = {r['race_id'] for r in recs}
    out = {}
    for r in conn.execute("""SELECT race_id, horse_num, place, tansho_payout,
                                    fukusho_payout FROM horse_history
                             WHERE date>=? AND date<=?""", (A1[0], A2[1])):
        if r['race_id'] in want:
            out[(r['race_id'], int(r['horse_num'] or 0))] = (
                r['place'], r['tansho_payout'], r['fukusho_payout'])
    return out


def attach(recs, out):
    keep = []
    for r in recs:
        y, tp, fp = [], [], []
        ok = True
        for hn in r['horse_num']:
            v = out.get((r['race_id'], int(hn)))
            if v is None:
                ok = False
                break
            pl_, t_, f_ = v
            if pl_ is None:
                ok = False
                break
            y.append(1 if 1 <= int(pl_) <= 3 else 0)
            tp.append(float(t_) if t_ else 0.0)
            fp.append(float(f_) if f_ else 0.0)
        if ok:
            r = dict(r)
            r['y'] = np.asarray(y, int)
            r['tansho'] = np.asarray(tp, float)
            r['fukusho'] = np.asarray(fp, float)
            keep.append(r)
    return keep


# ── 計測器 ───────────────────────────────────────────────────
def cells(recs, tkey='t5', tvals=None, yvals=None):
    """(能力3分位, 適性3分位) ごとの値の重み付き合計と重み。9セル。

    yvals を渡すと y の代わりに任意の連続量へ同じ計測器を当てられる
    （対照群B の線形性の検算に使う）。"""
    s = np.zeros((3, 3))
    c = np.zeros((3, 3))
    for i, r in enumerate(recs):
        t = tvals[i] if tvals is not None else r[tkey]
        if t is None:
            continue
        A = tercile_w(r['a_rank'], r['n'])
        B = tercile_w(desc_rank(np.asarray(t, float)), r['n'])
        y = np.asarray(r['y'] if yvals is None else yvals[i], float)
        # 外積で (能力分位 × 適性分位) の重みを作り、馬ごとに足し込む
        w = A[:, :, None] * B[:, None, :]        # (k, 3, 3)
        s += (w * y[:, None, None]).sum(axis=0)
        c += w.sum(axis=0)
    return s, c


def main_effect(s, c):
    """能力分位を統制した「T上位 − T下位」の3着内率差（pt）。各能力分位のN重み。"""
    num = den = 0.0
    for ai in range(3):
        if c[ai, 0] < 1 or c[ai, 2] < 1:
            continue
        d = s[ai, 0] / c[ai, 0] - s[ai, 2] / c[ai, 2]
        w = c[ai, 0] + c[ai, 2]
        num += d * w
        den += w
    return (num / den * 100.0) if den else float('nan')


def logodds(s, c):
    p = np.divide(s, np.maximum(c, 1))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ── 2) 対照群B — 注入の復元（本データの結果を読む前に通す） ───
def _D(recs, vals, tvals=None):
    """計測器そのもの（能力分位で統制した T上位 − T下位 の加重差）。
    y の代わりに任意の連続量 vals に当てられる。単位は pt（×100）。"""
    return main_effect(*cells(recs, tvals=tvals, yvals=vals))


def control_B(recs, rng):
    """🔑 合格は「注入後 − 注入前 = 注入量」。

    「注入後 ≈ 注入量」と書くと「その変数の素の効果は 0」という**測っていない
    仮定**が入る（North Star #11 / 2026-09-17 P3-01 で実際に踏んだ）。
    ここでは素の効果 D(v) を実測し、注入量を計測器自身の言葉 δ·D(u) で定義する。

    計測器 D は加重平均の差なので**線形**。よって
        D(v + δu) − D(v) = δ·D(u)
    が厳密に成立する。これは代数恒等式なので許容差 1e-3 でよい
    （North Star #11: 対照群B の許容差は統計量の SE の話とは別）。
    """
    print('\n── 対照群B: 注入の復元 ──')
    ok_all = True
    # u = T上位の重み（「T上位ほど効果が大きい」という意味のある注入）
    U = []
    for r in recs:
        B = tercile_w(desc_rank(r['t5']), r['n'])
        U.append(B[:, 0])
    Du = _D(recs, U)
    V = [rng.normal(size=r['n']) for r in recs]
    Dv = _D(recs, V)
    for delta in (0.5, 2.0, -1.5):
        Dvu = _D(recs, [v + delta * u for v, u in zip(V, U)])
        got, want = Dvu - Dv, delta * Du
        err = abs(got - want)
        ok = err < 1e-3
        ok_all &= ok
        print(f'  δ={delta:+.2f}  注入前 {Dv:+.6f}  注入後 {Dvu:+.6f}  '
              f'差 {got:+.6f}  注入量 δ·D(u) {want:+.6f}  '
              f'誤差 {err:.2e}  {"✅" if ok else "❌"}')

    # 感度検査: 本物があったらこの計測器は検出できるのか（North Star #11）
    # 合成二値 y を T 依存で作り、真値 D(真の効果) を復元できるかを見る
    eff = [0.10 * tercile_w(desc_rank(r['t5']), r['n'])[:, 0]
           - 0.10 * tercile_w(desc_rank(r['t5']), r['n'])[:, 2] for r in recs]
    truth = _D(recs, eff)
    synth = [(rng.random(r['n']) < np.clip(0.22 + e, 0.01, 0.99)).astype(float)
             for r, e in zip(recs, eff)]
    got = _D(recs, synth)
    ok2 = abs(got - truth) < 2.0          # 二値サンプリングの誤差を許容
    ok_all &= ok2
    print(f'  感度検査(合成二値)  真値 {truth:+.2f}pt  検出 {got:+.2f}pt  '
          f'{"✅" if ok2 else "❌"}')
    return ok_all


# ── 3) 対照群A — 並べ替え分布 ────────────────────────────────
def control_A(recs, rng, n=N_SHUFFLE, label=''):
    """T をレース内でシャッフル。合格は「帰無が 0 中心」
    ＝ |mean(効果)| < 2*std/sqrt(n)。1回の引きの大小比較は書かない（North Star #11）。"""
    print(f'\n── 対照群A: 並べ替え分布 {n}回 {label} ──')
    vals = []
    for _ in range(n):
        tv = [rng.permutation(r['t5']) for r in recs]
        s, c = cells(recs, tvals=tv)
        vals.append(main_effect(s, c))
    v = np.asarray(vals, float)
    lim = 2 * v.std(ddof=1) / np.sqrt(n)
    ok = abs(v.mean()) < lim
    print(f'  mean {v.mean():+.5f}pt  std {v.std(ddof=1):.5f}  '
          f'合格線 {lim:.5f}  {"✅" if ok else "❌"}')
    print(f'  分布 5% {np.percentile(v,5):+.4f} / 中央 {np.median(v):+.4f} '
          f'/ 95% {np.percentile(v,95):+.4f}')
    return ok, v


def perm_p(obs, null):
    """観測値の並べ替え P（両側）。"""
    null = np.asarray(null, float)
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)


# ── 4) 本データ ──────────────────────────────────────────────
def report(recs, label, tkey='t5'):
    s, c = cells(recs, tkey=tkey)
    d = main_effect(s, c)
    p = np.divide(s, np.maximum(c, 1)) * 100
    print(f'\n[{label}] {len(recs):,}レース {int(c.sum()):,}頭   主効果 {d:+.3f}pt')
    print('        T上位    T中位    T下位   | 上−下')
    mono = 0
    for ai, nm in enumerate(('能力上位', '能力中位', '能力下位')):
        dd = p[ai, 0] - p[ai, 2]
        m = p[ai, 0] >= p[ai, 1] >= p[ai, 2]
        mono += m
        print(f'  {nm}  {p[ai,0]:6.2f}%  {p[ai,1]:6.2f}%  {p[ai,2]:6.2f}%  | '
              f'{dd:+6.2f}pt {"単調" if m else "  - "}  (N {int(c[ai].sum()):,})')
    return {'d': d, 'p': p, 's': s, 'c': c, 'mono': int(mono),
            'd_by_a': [p[ai, 0] - p[ai, 2] for ai in range(3)]}


def payout(recs, label):
    """記述のみ。判定に使わない（North Star #9: 的中率と回収率は別物）。
    3着内なのに複勝配当が欠損する行は評価不能として除外（2026-08-18③）。"""
    finv = np.zeros((3, 3)); fret = np.zeros((3, 3)); drop = 0
    tinv = np.zeros((3, 3)); tret = np.zeros((3, 3))
    for r in recs:
        A = tercile_w(r['a_rank'], r['n'])
        B = tercile_w(desc_rank(r['t5']), r['n'])
        for k in range(r['n']):
            w = A[k][:, None] * B[k][None, :]   # (3,3) 重み
            y = r['y'][k]
            if y and r['fukusho'][k] <= 0:
                drop += 1                       # 評価不能。0円として混ぜない
            else:
                finv += 100 * w
                fret += (r['fukusho'][k] if y else 0.0) * w
            tinv += 100 * w                     # tansho_payout は1着馬にのみ入る
            tret += (r['tansho'][k] if r['tansho'][k] > 0 else 0.0) * w
    for nm2, iv, rt in (('複勝', finv, fret), ('単勝', tinv, tret)):
        print(f'\n[{label}] {nm2}回収率（記述のみ・判定に使わない）'
              + (f'  除外 {drop} 行' if nm2 == '複勝' else ''))
        print('        T上位    T中位    T下位')
        for ai, nm in enumerate(('能力上位', '能力中位', '能力下位')):
            print(f'  {nm}  ' + '  '.join(
                f'{rt[ai,bi]/max(iv[ai,bi],1)*100:6.1f}%' for bi in range(3)))


def main():
    db = sys.argv[1]
    cache = sys.argv[2] if len(sys.argv) > 2 else 'step1_cache.pkl'
    rng = np.random.default_rng(SEED)

    # 1) 特徴量（目的変数を読まない）
    if os.path.exists(cache):
        print(f'キャッシュを使う: {cache}')
        recs = pickle.load(open(cache, 'rb'))
    else:
        recs = build_cache(db, cache)

    # 2) 対照群B → 3) 対照群A → 4) 本データ の順（Gate 0 と同じ）
    okB = control_B(recs, rng)
    if not okB:
        print('\n🔴 対照群B が落ちた。計測器が通っていない測定の値は読まない。')
        return 1

    out = load_outcome(db, recs)
    recs = attach(recs, out)
    a1 = [r for r in recs if A1[0] <= r['date'] <= A1[1]]
    a2 = [r for r in recs if A2[0] <= r['date'] <= A2[1]]
    print(f'\nA1 {len(a1):,}レース / A2 {len(a2):,}レース')

    # 🔑 帰無分布は期間別に作る。A1+A2 を合成すると N が倍になり null の std が
    #    1/√2 に縮むため、A1 単独の観測値に当てると P が甘く出る。
    okA1, null1 = control_A(a1, rng, label='A1')
    okA2, null2 = control_A(a2, rng, label='A2')
    if not (okA1 and okA2):
        print('\n🔴 対照群A が落ちた。計測器が通っていない測定の値は読まない。')
        return 1

    print('\n' + '=' * 62)
    print('本データ（5列版 T・事前登録の構成）')
    print('=' * 62)
    r1 = report(a1, 'A1 2023（規則を作る）')
    r2 = report(a2, 'A2 2024（再現を見る）')
    payout(a1, 'A1'); payout(a2, 'A2')

    # 基準1 主効果
    c1 = (r1['d'] > 0 and r2['d'] > 0 and r1['d'] >= 2.0 and r2['d'] >= 2.0)
    p1, p2 = perm_p(r1['d'], null1), perm_p(r2['d'], null2)
    # 基準2 持ち越し（9セル log-odds の Spearman ρ を並べ替えで P 値化）
    l1, l2 = logodds(r1['s'], r1['c']).ravel(), logodds(r2['s'], r2['c']).ravel()
    rho = spearman(np.argsort(np.argsort(-l1)), np.argsort(np.argsort(-l2)))
    perm = [spearman(np.argsort(np.argsort(-l1)),
                     rng.permutation(np.argsort(np.argsort(-l2))))
            for _ in range(N_PERM_RHO)]
    p_rho = (np.sum(np.asarray(perm) >= rho) + 1) / (N_PERM_RHO + 1)
    c2 = p_rho < 0.05
    # 基準3 単調性
    c3 = (r1['mono'] >= 2 and r2['mono'] >= 2)

    print('\n' + '=' * 62)
    print('事前登録した基準（betting/CRITERIA_STEP1.md §5）')
    print('=' * 62)
    print(f'  基準1 主効果    A1 {r1["d"]:+.2f}pt (並べ替えP {p1:.4f}) / '
          f'A2 {r2["d"]:+.2f}pt (P {p2:.4f})  合格線 両期間 >= +2.0pt  '
          f'{"✅" if c1 else "❌"}')
    print(f'  基準2 持ち越し  9セル log-odds の ρ = {rho:+.3f}  '
          f'並べ替え P = {p_rho:.4f}  合格線 P < 0.05  {"✅" if c2 else "❌"}')
    print(f'  基準3 単調性    A1 {r1["mono"]}/3 / A2 {r2["mono"]}/3  '
          f'合格線 両期間 2以上  {"✅" if c3 else "❌"}')
    print(f'  基準4 対照群A   ✅（上記）')
    print(f'  基準5 対照群B   ✅（上記）')

    # 基準1が落ちた場合に見る交互作用（事前登録・事後に増やさない）
    if not c1:
        d1, d2 = r1['d_by_a'], r2['d_by_a']
        o1, o2 = tuple(np.argsort(np.argsort(-np.asarray(d1)))), \
                 tuple(np.argsort(np.argsort(-np.asarray(d2))))
        i1 = (o1 == o2)
        i2 = max(max(d1), max(d2)) >= 3.0
        print('\n  ── 基準1 が落ちたので事前登録した交互作用を見る ──')
        print(f'  d_top/d_mid/d_bot  A1 ' +
              ' / '.join(f'{v:+.2f}' for v in d1) + '   A2 ' +
              ' / '.join(f'{v:+.2f}' for v in d2))
        print(f'  I-1 大小順序が A1→A2 で一致  {o1} vs {o2}  '
              f'{"✅" if i1 else "❌"}（偶然なら 1/6）')
        print(f'  I-2 いずれかの能力分位で >= +3.0pt  '
              f'最大 {max(max(d1), max(d2)):+.2f}pt  {"✅" if i2 else "❌"}')

    # 感度分析（事前登録）: f_course_fit_score を外した4列版
    print('\n' + '=' * 62)
    print('感度分析（事前登録）: f_course_fit_score を外した4列版 T')
    print('🔴 良い結果が出てもこちらを採用しない。1列依存かを見るためだけ')
    print('=' * 62)
    s1, c1_ = cells([r for r in a1 if r['t4'] is not None], tkey='t4')
    s2, c2_ = cells([r for r in a2 if r['t4'] is not None], tkey='t4')
    print(f'  主効果  A1 {main_effect(s1, c1_):+.2f}pt / '
          f'A2 {main_effect(s2, c2_):+.2f}pt  （5列版 '
          f'{r1["d"]:+.2f} / {r2["d"]:+.2f}）')

    print('\n' + '=' * 62)
    ok = c1 and c2 and c3
    print('判定: ' + ('Step 2 へ進む' if ok else
                      'Step 1 で閉じる（§9 の文を使う）'))
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())
