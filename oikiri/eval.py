#!/usr/bin/env python3
"""yosounohone の追い切り★を 2023-2025 で測るハーネス。

事前登録カード: `oikiri/CRITERIA.md`。**実行前にコミットする。**

🔴 **中止条件（カード §7）を、結果（3着内率・回収率）より手前に全部印字する。**
   2026-10-19② で中止条件を最後に印字する作りにしたため ECE を先に見てしまった。
   同じ失敗をしないよう、このファイルは

       ① 中止条件 → ② 対照群B → ③ 対照群A → ④ 主効果 → ⑤ 回収率

   の順でしか印字しない。①で止まったら②以降を計算しない。

🔴 **信号ライン（基準1〜4）が落ちたら以降を読まない**（カード §3 に明記）。
   基準7（本番同型 XGB への上乗せ）は `build_training_data` に15分かかるので
   **信号ラインを通ってから**別ステップで回す。

使い方:
    python3 oikiri/eval.py                 # Phase 1（中止条件 → 対照群 → 信号ライン → 回収率）
"""

import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.scraper.yosounohone import VENUE_CODE  # noqa: E402

OIKIRI_DB = os.path.join(BASE, 'data', 'private', 'oikiri.db')
HIST_DB = os.path.join(BASE, 'data', 'history.db')
OUT_JSON = os.path.join(BASE, 'oikiri', 'result.json')

START, END = '2023-01-01', '2025-12-31'
JST = timezone(timedelta(hours=9))

N_PERM = 200
PERM_SEED = 20261022
INJECT_PT = 20.0          # 対照群B の注入量
INJECT_TOL = 1e-3

# カード §3 の合格線（結果を見てから動かさない）
MIN_EFFECT_PT = 5.0
MIN_N_MARKED = 1000
MIN_MARKET_CONTROLLED_PT = 3.0
MIN_ROI_PCT = 100.0

# カード §7 の中止線
MIN_PARSE_OK = 0.90
MIN_NAME_MATCH = 0.95
MAX_PLACE_NULL = 0.05
MAX_BASELINE_DEV_PT = 3.0
RACE_START_JST_HOUR = 9.5   # 1R目は概ね 09:50。同日公開は 09:30 より前なら発走前


def _p(s=''):
    print(s, flush=True)


# ───────────────────────── データの読み込みと突合 ─────────────────────────

def load_rows():
    """marks × history.db を突合して、M が1頭以上居るレースの全馬を返す。"""
    oc = sqlite3.connect(OIKIRI_DB)
    oc.row_factory = sqlite3.Row
    hc = sqlite3.connect(HIST_DB)
    hc.row_factory = sqlite3.Row

    arts = oc.execute('SELECT * FROM articles').fetchall()
    marks = oc.execute(
        'SELECT * FROM marks WHERE date >= ? AND date <= ?', (START, END)).fetchall()

    # (date, venue, race_num) -> race_id
    rid = {}
    for r in hc.execute(
            'SELECT race_id, date, racecourse, race_num FROM race_history '
            'WHERE date >= ? AND date <= ?', (START, END)):
        rid[(r['date'], r['racecourse'], int(r['race_num']))] = r['race_id']

    # race_id -> {horse_name: row}
    horses = defaultdict(dict)
    for r in hc.execute(
            'SELECT race_id, horse_name, horse_num, place, popularity, '
            '       fukusho_payout, field_size '
            'FROM horse_history WHERE date >= ? AND date <= ?', (START, END)):
        horses[r['race_id']][r['horse_name']] = dict(r)

    matched, unmatched = [], []
    marked_by_race = defaultdict(dict)
    for m in marks:
        key = (m['date'], m['venue'], int(m['race_num']))
        r_id = rid.get(key)
        if r_id is None:
            unmatched.append((key, m['horse_name'], 'レースが history.db に無い'))
            continue
        h = horses[r_id].get(m['horse_name'])
        if h is None:
            unmatched.append((key, m['horse_name'], '馬が同レースに居ない'))
            continue
        marked_by_race[r_id][m['horse_name']] = int(m['stars'])
        matched.append((r_id, m['horse_name'], int(m['stars'])))

    rows = []
    for r_id, stars_map in marked_by_race.items():
        for name, h in horses[r_id].items():
            rows.append({
                'race_id': r_id,
                'name': name,
                'stars': stars_map.get(name),            # None なら対照群C
                'marked': name in stars_map,
                'place': h['place'],
                'pop': h['popularity'],
                'payout': h['fukusho_payout'],
                'field': len(horses[r_id]),
            })
    return arts, marks, matched, unmatched, rows


# ───────────────────────── 指標 ─────────────────────────

def top3_rate(rows):
    v = [1.0 if (r['place'] is not None and 1 <= r['place'] <= 3) else 0.0
         for r in rows if r['place'] is not None]
    return (100.0 * sum(v) / len(v), len(v)) if v else (float('nan'), 0)


def fuku_roi(rows):
    """複勝1点100円。payout が無い3着内（少頭数で複勝が売られていない等）は評価不能。

    ⚠ 2026-10-02 の実測で「3着内なのに fukusho_payout が 0/NULL」が 1.73% ある。
      0円として混ぜると実力を過小評価する方向に静かにずれる（2026-08-18③）。
    """
    bets, ret, skipped, payouts = 0, 0.0, 0, []
    for r in rows:
        if r['place'] is None:
            continue
        hit = r['place'] is not None and 1 <= r['place'] <= 3
        pay = r['payout']
        if hit and (pay is None or pay <= 0):
            skipped += 1
            continue
        bets += 1
        if hit:
            ret += float(pay)
            payouts.append(float(pay))
    if not bets:
        return float('nan'), float('nan'), 0, skipped, 0
    roi = 100.0 * ret / (100.0 * bets)
    cut = sorted(payouts, reverse=True)[:3]
    roi3 = 100.0 * (ret - sum(cut)) / (100.0 * bets)
    return roi, roi3, bets, skipped, len(payouts)


def _ci95(p_pct, n):
    if not n:
        return (float('nan'), float('nan'))
    p = p_pct / 100.0
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return (100 * (p - 1.96 * se), 100 * (p + 1.96 * se))


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float('nan')
    o = np.argsort(s, kind='mergesort')
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    # 同値は平均順位
    i = 0
    ss = s[o]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ───────────────────────── ① 中止条件 ─────────────────────────

def stop_conditions(arts, marks, matched, unmatched, rows):
    _p('=' * 78)
    _p('① 中止条件（カード §7）— 結果より手前に全部印字する')
    _p('=' * 78)
    stop = []

    n_art = len(arts)
    n_ok = sum(1 for a in arts if a['status'] == 'ok')
    rate = n_ok / n_art if n_art else 0.0
    _p(f'  1. パース成功率 = {n_ok}/{n_art} = {rate:.4f}（線 >= {MIN_PARSE_OK}）')
    for a in arts:
        if a['status'] != 'ok':
            _p(f'       不採用: {a["date"]} {a["venue"]} / {a["status"]}')
    if rate < MIN_PARSE_OK:
        stop.append(f'パース成功率 {rate:.4f} < {MIN_PARSE_OK}')

    n_m = len(marks)
    mrate = len(matched) / n_m if n_m else 0.0
    _p(f'  2. 馬名の突合率 = {len(matched)}/{n_m} = {mrate:.4f}（線 >= {MIN_NAME_MATCH}）')
    why = defaultdict(int)
    for _, _, w in unmatched:
        why[w] += 1
    for w, c in sorted(why.items(), key=lambda x: -x[1]):
        _p(f'       内訳: {w} = {c}')
    if mrate < MIN_NAME_MATCH:
        stop.append(f'突合率 {mrate:.4f} < {MIN_NAME_MATCH}')

    late, miss = [], []
    for a in arts:
        if a['status'] != 'ok':
            continue
        if not a['published_og']:
            miss.append(a['url'])
            continue
        og = datetime.fromisoformat(a['published_og']).astimezone(JST)
        rd = datetime.fromisoformat(a['date'] + 'T00:00:00').replace(tzinfo=JST)
        hours = og.hour + og.minute / 60.0
        if og.date() > rd.date() or (og.date() == rd.date()
                                     and hours >= RACE_START_JST_HOUR):
            late.append((a['date'], a['venue'], og.isoformat()))
    _p(f'  3. 公開時刻が発走前でない記事 = {len(late)} 件（線 = 0）')
    for x in late[:10]:
        _p(f'       {x}')
    if miss:
        _p(f'     ⚠ 公開時刻が読めない記事 {len(miss)} 件')
    if late or miss:
        stop.append(f'公開時刻が発走前でない/読めない記事 {len(late) + len(miss)} 件')

    mism = []
    for a in arts:
        if a['status'] != 'ok' or not a['published_og'] or not a['published_ld']:
            continue
        if datetime.fromisoformat(a['published_og']) != datetime.fromisoformat(a['published_ld']):
            mism.append((a['date'], a['venue'], a['published_og'], a['published_ld']))
    _p(f'  4. OG と JSON-LD が別の瞬間を指す記事 = {len(mism)} 件（線 = 0）')
    for x in mism[:5]:
        _p(f'       {x}')
    if mism:
        stop.append(f'OG と LD が食い違う記事 {len(mism)} 件')

    bad = [m['stars'] for m in marks if not (1 <= int(m['stars']) <= 5)]
    _p(f'  5. ★が 1〜5 の外 = {len(bad)} 件（線 = 0）')
    if bad:
        stop.append(f'★が値域外 {len(bad)} 件')

    A = rows
    a_rate, a_n = top3_rate(A)
    fields = [r['field'] for r in A]
    exp = 100.0 * 3.0 / (sum(fields) / len(fields)) if fields else float('nan')
    _p(f'  6. A群の3着内率 = {a_rate:.2f}%（N={a_n}） vs 3/平均頭数 = {exp:.2f}%'
       f'（許容 ±{MAX_BASELINE_DEV_PT}pt）')
    if not math.isnan(a_rate) and abs(a_rate - exp) > MAX_BASELINE_DEV_PT:
        stop.append(f'A群 {a_rate:.2f}% が期待 {exp:.2f}% から {MAX_BASELINE_DEV_PT}pt 超乖離')

    M = [r for r in rows if r['marked']]
    nullrate = sum(1 for r in M if r['place'] is None) / len(M) if M else 1.0
    _p(f'  7. M の place が NULL = {nullrate:.4f}（線 < {MAX_PLACE_NULL}）')
    if nullrate >= MAX_PLACE_NULL:
        stop.append(f'M の place NULL率 {nullrate:.4f} >= {MAX_PLACE_NULL}')

    _p()
    if stop:
        _p('🔴 中止条件に当たった:')
        for s in stop:
            _p(f'     - {s}')
        _p('   → 結果を読まずにここで止める。原因を直して**同じ線で**測り直す')
    else:
        _p('✅ 中止条件は7つとも通過')
    return stop


# ───────────────────────── ② 対照群B（注入の復元） ─────────────────────────

def control_b(rows):
    _p()
    _p('=' * 78)
    _p('② 対照群B ＝ 注入の復元（合格は「注入後 − 注入前 = 注入量」）')
    _p('=' * 78)
    M = [r for r in rows if r['marked'] and r['place'] is not None]
    C = [r for r in rows if not r['marked'] and r['place'] is not None]
    before = top3_rate(M)[0] - top3_rate(C)[0]

    # M の 20% 相当を「必ず3着内」に書き換えると、M の3着内率はちょうど +20pt 動く
    k = int(round(len(M) * INJECT_PT / 100.0))
    not3 = [i for i, r in enumerate(M) if not (1 <= r['place'] <= 3)]
    rng = np.random.default_rng(PERM_SEED)
    pick = rng.choice(len(not3), size=min(k, len(not3)), replace=False)
    inj = [dict(r) for r in M]
    for i in pick:
        inj[not3[i]]['place'] = 1
    after = top3_rate(inj)[0] - top3_rate(C)[0]
    delta = after - before
    expect = 100.0 * len(pick) / len(M)

    _p(f'  経路の検算: 注入対象 {len(pick)} 頭 / M {len(M)} 頭 = {expect:.4f}pt')
    _p(f'  注入前の β = {before:+.4f}pt')
    _p(f'  注入後の β = {after:+.4f}pt')
    _p(f'  注入後 − 注入前 = {delta:+.4f}pt  /  注入量 = {expect:+.4f}pt'
       f'  /  差 = {abs(delta - expect):.3e}（許容 {INJECT_TOL}）')
    ok = abs(delta - expect) < INJECT_TOL
    _p(f'  {"✅ 通過" if ok else "🔴 未通過"}')
    return {'ok': bool(ok), 'before': before, 'after': after,
            'delta': delta, 'expect': expect}


# ───────────────────────── ③ 対照群A（並べ替え分布） ─────────────────────────

def control_a(rows):
    _p()
    _p('=' * 78)
    _p(f'③ 対照群A ＝ 並べ替え分布（レース内で★の割り当てをシャッフル・{N_PERM}回）')
    _p('=' * 78)
    _p('  統計量 β = (M の3着内率) − (C の3着内率)')
    _p('  合格は「帰無が0中心」: |mean(β_shuffle)| < 2*std/sqrt(N)')

    by_race = defaultdict(list)
    for r in rows:
        if r['place'] is not None:
            by_race[r['race_id']].append(r)

    def beta(flags):
        m = [v for v, f in zip(flags[0], flags[1]) if f]
        c = [v for v, f in zip(flags[0], flags[1]) if not f]
        if not m or not c:
            return float('nan')
        return 100.0 * (sum(m) / len(m)) - 100.0 * (sum(c) / len(c))

    y, mk = [], []
    race_slices = []
    i = 0
    for rid, rs in by_race.items():
        for r in rs:
            y.append(1.0 if 1 <= r['place'] <= 3 else 0.0)
            mk.append(r['marked'])
        race_slices.append((i, i + len(rs), sum(1 for r in rs if r['marked'])))
        i += len(rs)
    y = np.asarray(y, float)
    obs = beta((y, np.asarray(mk, bool)))

    rng = np.random.default_rng(PERM_SEED)
    nulls = []
    for _ in range(N_PERM):
        f = np.zeros(len(y), bool)
        for a, b, nm in race_slices:
            if nm:
                f[a + rng.choice(b - a, size=nm, replace=False)] = True
        nulls.append(beta((y, f)))
    nulls = np.asarray(nulls, float)
    mean, std = float(nulls.mean()), float(nulls.std(ddof=1))
    line = 2 * std / math.sqrt(N_PERM)
    pval = float((np.abs(nulls) >= abs(obs)).sum() + 1) / (N_PERM + 1)

    _p(f'  実測 β = {obs:+.4f}pt')
    _p(f'  帰無 mean = {mean:+.4f}pt / std = {std:.4f} / 合格線 = {line:.4f}')
    ok = abs(mean) < line
    _p(f'  {"✅ 帰無は0中心" if ok else "🔴 帰無が0中心にならない（原因を特定するまで結果を読まない）"}')
    _p(f'  並べ替え P = {pval:.4f}')
    return {'ok': bool(ok), 'beta': obs, 'null_mean': mean,
            'null_std': std, 'line': line, 'perm_p': pval}


# ───────────────────────── ④ 信号ライン ─────────────────────────

def signal(rows):
    _p()
    _p('=' * 78)
    _p('④ 信号ライン（カード §3 基準1〜4）')
    _p('=' * 78)
    M = [r for r in rows if r['marked']]
    C = [r for r in rows if not r['marked']]
    mr, mn = top3_rate(M)
    cr, cn = top3_rate(C)
    eff = mr - cr
    _p(f'  基準1 主効果: M {mr:.2f}% (N={mn}) − C {cr:.2f}% (N={cn}) '
       f'= {eff:+.2f}pt（線 >= +{MIN_EFFECT_PT}）')
    ci = _ci95(mr, mn)
    _p(f'        M の95%CI [{ci[0]:.2f}, {ci[1]:.2f}]')
    _p(f'  基準2 標本: N(M) = {mn}（線 >= {MIN_N_MARKED}）')

    bands = [('★1', (1, 1)), ('★2', (2, 2)), ('★3', (3, 3)), ('★4-5', (4, 5))]
    brates = []
    _p('  基準3 単調性:')
    for lbl, (lo, hi) in bands:
        g = [r for r in M if r['stars'] is not None and lo <= r['stars'] <= hi]
        r_, n_ = top3_rate(g)
        brates.append(r_)
        _p(f'        {lbl:5s} {r_:6.2f}%  (N={n_})')
    mono = all((not math.isnan(brates[i + 1]) and not math.isnan(brates[i])
                and brates[i + 1] > brates[i]) for i in range(len(brates) - 1))
    _p(f'        3境界すべて単調増加: {"✅" if mono else "🔴"}')

    _p('  基準4 市場統制（市場人気3分位の各分位内）:')
    pops = sorted(r['pop'] for r in rows if r['pop'] and 0 < r['pop'] < 99)
    if len(pops) < 30:
        _p('        🔴 人気が足りず統制できない')
        q = []
    else:
        q = [pops[len(pops) // 3], pops[2 * len(pops) // 3]]
    diffs = []
    for i, (lo, hi) in enumerate([(0, q[0]), (q[0] + 1, q[1]), (q[1] + 1, 99)] if q else []):
        gm = [r for r in M if r['pop'] and lo <= r['pop'] <= hi and r['pop'] < 99]
        gc = [r for r in C if r['pop'] and lo <= r['pop'] <= hi and r['pop'] < 99]
        a_, na = top3_rate(gm)
        b_, nb = top3_rate(gc)
        d = a_ - b_
        diffs.append(d)
        _p(f'        人気{lo}-{min(hi,99)}: M {a_:6.2f}% (N={na:5d}) '
           f'− C {b_:6.2f}% (N={nb:5d}) = {d:+6.2f}pt')
    same = bool(diffs) and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs))
    avg = float(np.mean(diffs)) if diffs else float('nan')
    _p(f'        各分位で同符号: {"✅" if same else "🔴"} / 平均 {avg:+.2f}pt'
       f'（線 >= +{MIN_MARKET_CONTROLLED_PT}）')

    res = {
        'effect_pt': eff, 'm_rate': mr, 'c_rate': cr, 'n_m': mn, 'n_c': cn,
        'm_ci': ci, 'band_rates': brates, 'monotone': mono,
        'market_diffs': diffs, 'market_same_sign': same, 'market_avg': avg,
    }
    res['pass1'] = bool(eff >= MIN_EFFECT_PT)
    res['pass2'] = bool(mn >= MIN_N_MARKED)
    res['pass3'] = bool(mono)
    res['pass4'] = bool(same and avg >= MIN_MARKET_CONTROLLED_PT)
    _p()
    _p(f'  信号ライン: 基準1 {"✅" if res["pass1"] else "🔴"} / '
       f'基準2 {"✅" if res["pass2"] else "🔴"} / '
       f'基準3 {"✅" if res["pass3"] else "🔴"} / '
       f'基準4 {"✅" if res["pass4"] else "🔴"}')
    return res


# ───────────────────────── ⑤ 回収率と安定性 ─────────────────────────

def roi_block(rows):
    _p()
    _p('=' * 78)
    _p('⑤ 回収率（カード §3 基準6＝判定指標）と安定性（基準5）')
    _p('=' * 78)
    _p('  ⚠ 複勝プール平均は控除率20%で 80%。100%超が合格線')
    out = {}
    for lbl, g in (('M 印あり', [r for r in rows if r['marked']]),
                   ('C 印なし', [r for r in rows if not r['marked']]),
                   ('A 全馬', rows)):
        roi, roi3, bets, skip, hits = fuku_roi(g)
        ci = _ci95(roi, bets)
        _p(f'  {lbl:8s} 回収 {roi:7.2f}%  3本抜き {roi3:7.2f}%  '
           f'（買い {bets:5d} / 的中 {hits:4d} / 評価不能 {skip}）')
        out[lbl] = {'roi': roi, 'roi_cut3': roi3, 'bets': bets,
                    'hits': hits, 'skipped': skip, 'ci95': ci}
    _p()
    _p('  ★帯ごと（M の内訳）:')
    for lbl, lo, hi in (('★1', 1, 1), ('★2', 2, 2), ('★3', 3, 3), ('★4-5', 4, 5)):
        g = [r for r in rows if r['marked'] and r['stars'] and lo <= r['stars'] <= hi]
        roi, roi3, bets, _s, hits = fuku_roi(g)
        _p(f'        {lbl:5s} 回収 {roi:7.2f}%  3本抜き {roi3:7.2f}%  (買い {bets:5d})')
        out[f'band_{lbl}'] = {'roi': roi, 'roi_cut3': roi3, 'bets': bets}

    _p()
    _p('  基準5 安定性（前半 2023 / 後半 2024-2025）:')
    for lbl, lo, hi in (('2023', '2023-01-01', '2023-12-31'),
                        ('2024-2025', '2024-01-01', '2025-12-31')):
        sub = [r for r in rows if lo <= r['race_id'][:4] + '-' + r['race_id'][4:6]
               + '-' + r['race_id'][6:8] <= hi]
        m = [r for r in sub if r['marked']]
        c = [r for r in sub if not r['marked']]
        mr, mn = top3_rate(m)
        cr, _cn = top3_rate(c)
        roi, roi3, bets, _s, _h = fuku_roi(m)
        _p(f'        {lbl:10s} 主効果 {mr - cr:+6.2f}pt (N={mn:5d}) / '
           f'M 回収 {roi:7.2f}% 3本抜き {roi3:7.2f}%')
        out[f'half_{lbl}'] = {'effect_pt': mr - cr, 'n_m': mn,
                              'roi': roi, 'roi_cut3': roi3, 'bets': bets}
    return out


# ───────────────────────── ⑥ 市場比較ゲート ─────────────────────────

def market_gate(rows):
    _p()
    _p('=' * 78)
    _p('⑥ 市場比較ゲート（カード §4）')
    _p('=' * 78)
    _p('  ⚠ ここは本番134列ではなく「市場人気」と「★」だけの軽い比較。')
    _p('    基準7（本番同型 XGB への上乗せ）は信号ラインを通ってから別ステップで回す。')
    use = [r for r in rows if r['place'] is not None and r['pop'] and 0 < r['pop'] < 99]
    y = [1.0 if 1 <= r['place'] <= 3 else 0.0 for r in use]
    s_mkt = [-float(r['pop']) for r in use]
    s_star = [float(r['stars'] or 0) for r in use]
    # 市場 + ★（★は市場の中での小さな上乗せとして足す。係数は振らない）
    s_both = [m + 0.5 * st for m, st in zip(s_mkt, s_star)]
    _p(f'  市場単独（−確定人気）      複勝AUC {auc(y, s_mkt):.4f}  (N={len(use)})')
    _p(f'  ★単独                     複勝AUC {auc(y, s_star):.4f}')
    _p(f'  市場 + ★                  複勝AUC {auc(y, s_both):.4f}')

    _p()
    _p('  乖離の方向別（★の順位 − 市場人気順位。+ は「★のほうが高く見ている」）:')
    _p('  🔴 「★強気」側で改善が出たら、まず計測器を疑う（5回ともこの側が最悪だった）')
    by_race = defaultdict(list)
    for r in use:
        by_race[r['race_id']].append(r)
    buckets = defaultdict(list)
    for rid, rs in by_race.items():
        ranked = sorted(rs, key=lambda x: (-(x['stars'] or 0), x['pop']))
        for i, r in enumerate(ranked, 1):
            if not r['marked']:
                continue
            gap = r['pop'] - i
            k = ('★強気 (+3以上)' if gap >= 3 else
                 '一致 (±2)' if gap >= -2 else '★弱気 (−3以下)')
            buckets[k].append(r)
    gate = {}
    for k in ('★強気 (+3以上)', '一致 (±2)', '★弱気 (−3以下)'):
        g = buckets.get(k, [])
        t, n = top3_rate(g)
        roi, roi3, bets, _s, _h = fuku_roi(g)
        _p(f'        {k:16s} 3着内 {t:6.2f}% (N={n:5d}) / 回収 {roi:7.2f}% '
           f'3本抜き {roi3:7.2f}%')
        gate[k] = {'top3': t, 'n': n, 'roi': roi, 'roi_cut3': roi3}
    return {'auc_market': auc(y, s_mkt), 'auc_star': auc(y, s_star),
            'auc_both': auc(y, s_both), 'n': len(use), 'direction': gate}


def main():
    if not os.path.exists(OIKIRI_DB):
        _p(f'🔴 {OIKIRI_DB} が無い。先に scripts/collect_oikiri.py を回すこと')
        return 1
    _p('yosounohone の追い切り★ — 2023-2025 の検定（カード: oikiri/CRITERIA.md）')
    _p(f'期間 {START} 〜 {END}')
    _p('🔴 2026 は 2026-10-21 の logic2026 で読んだので使わない')
    _p()

    arts, marks, matched, unmatched, rows = load_rows()
    _p(f'記事 {len(arts)} / 印 {len(marks)} / 突合できた印 {len(matched)} / '
       f'対象レースの全馬 {len(rows)}')
    _p(f'レース数 {len({r["race_id"] for r in rows})}')

    res = {'period': [START, END], 'n_articles': len(arts), 'n_marks': len(marks),
           'n_matched': len(matched), 'n_rows': len(rows),
           'n_races': len({r['race_id'] for r in rows})}

    stop = stop_conditions(arts, marks, matched, unmatched, rows)
    res['stop'] = stop
    if stop:
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        return 2

    res['control_b'] = control_b(rows)
    res['control_a'] = control_a(rows)
    res['signal'] = signal(rows)
    res['roi'] = roi_block(rows)
    res['market_gate'] = market_gate(rows)

    sg = res['signal']
    _p()
    _p('=' * 78)
    _p('⑦ 判定')
    _p('=' * 78)
    passed = all(sg[f'pass{i}'] for i in (1, 2, 3, 4))
    _p(f'  信号ライン（基準1〜4）: {"✅ 通過" if passed else "🔴 未通過"}')
    if not passed:
        _p('  → カード §3 のとおり、ここが落ちたので採用ラインは読まない。')
        _p('     基準7（本番同型 XGB への上乗せ）は回さない。')
    else:
        m = res['roi']['M 印あり']
        _p(f'  基準6 回収率: M {m["roi"]:.2f}% / 3本抜き {m["roi_cut3"]:.2f}%'
           f'（線 > {MIN_ROI_PCT}）')
        _p('  → 基準7（本番同型 XGB）を別ステップで回す')
    res['signal_passed'] = bool(passed)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    _p(f'\n📄 {OUT_JSON} に書いた')
    return 0


if __name__ == '__main__':
    sys.exit(main())
