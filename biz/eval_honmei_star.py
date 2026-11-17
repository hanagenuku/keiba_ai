#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_HONMEI_STAR.md の測定（1回）。

    cd <隔離ワークスペース（horse_features.csv・history.db・keiba.db が実体）>
    python3 <repo>/biz/eval_honmei_star.py <repo のパス>

☆は hitrate/eval_star.py の 2026年の窓（V2・V3）をシード42で学習し直して作る（各窓の手前までのデータだけ）。
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = sys.argv[1]
WS = os.getcwd()
sys.path.insert(0, WS)
sys.path.insert(0, os.path.join(REPO, 'biz'))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


S = _load('eval_star', os.path.join(WS, 'hitrate', 'eval_star.py'))
S.SEEDS = (42,)
S.C.SEEDS = (42,)
import eval_points as E  # noqa: E402
from src.product import tickets as T  # noqa: E402

WINS = [w for w in S.Y2026 if w[0] in ('V2', 'V3')]
FROM, TO = '2026-06-27', '2026-11-08'
N_BOOT = 2000
STEADY = {'単勝': 'tansho', '複勝': 'fukusho'}


def star_set():
    df, cols, X, d, rid, hn, pop_tr, pop_ev, place, pay = S.load()
    _, pairs, fits = S.fit_period(WINS, cols, X, d, rid, hn, pop_tr, pop_ev, place, '2026')
    out = {}
    for name, _, _, _ in WINS:
        s = S.select(S.picks(pairs, fits, name, 42), 'S1')
        for r, n, p in zip(s['r'], s['n'], s['p']):
            out[str(r)] = (int(n), float(p))
    return out


def course_lines(r, axis):
    hs = r['horses']
    partners, solid, ana = T.split_partners(hs, axis, 3, 2)
    bets = T.build_bet_lines(axis, partners, solid, ana, allow_trifecta=True)
    agg, steady = {}, {}
    for b in bets:
        g = b.get('group') or b['type']
        if g in E.BT and g not in agg:
            agg[g] = [tuple(c) for c in b['combos']]
        if g in STEADY and g not in steady:
            steady[g] = [tuple(c) for c in b['combos']]
    return steady, agg


def pay_steady(r, t, combos):
    return sum(r['div'].get((STEADY[t], str(c[0])), 0) for c in combos)


def score(r, axis):
    st, ag = course_lines(r, axis)
    s_stake = sum(len(c) * 100 for c in st.values())
    s_pay = sum(pay_steady(r, t, c) for t, c in st.items())
    fuku = pay_steady(r, '複勝', st.get('複勝', []))
    a_stake = sum(len(c) * 100 for c in ag.values())
    a_pay = sum(E.pay_of(r, t, c) for t, c in ag.items())
    return dict(s_stake=s_stake, s_pay=s_pay, fuku=fuku, a_stake=a_stake, a_pay=a_pay)


def summ(st, py):
    hit = py > 0
    i = int(np.argmax(py))
    return dict(hit_rate=round(float(hit.mean() * 100), 1), roi=round(float(py.sum() / st.sum() * 100), 1),
                roi_wo_top1=round(float((py.sum() - py[i]) / max(st.sum() - st[i], 1) * 100), 1),
                loss_on_hit=round(float(((py < st) & hit).sum() / max(hit.sum(), 1) * 100), 1))


def diff(a_st, a_py, b_st, b_py, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a_st)
    dh, dr = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        dh.append((b_py[i] > 0).mean() - (a_py[i] > 0).mean())
        dr.append(b_py[i].sum() / b_st[i].sum() - a_py[i].sum() / a_st[i].sum())
    q = lambda v: [round(float(np.percentile(v, 2.5)) * 100, 1), round(float(np.percentile(v, 97.5)) * 100, 1)]
    return dict(d_hit=round(float(((b_py > 0).mean() - (a_py > 0).mean()) * 100), 1), d_hit_ci=q(dh),
                d_roi=round(float(b_py.sum() / b_st.sum() - a_py.sum() / a_st.sum()) * 100, 1), d_roi_ci=q(dr),
                d_roi_se=round(float(np.std(dr)) * 100, 1))


def evaluate(races):
    cur = [score(r, r['axis']) for r in races]
    new = [score(r, r['star']) for r in races]
    out = {'races': len(races)}
    for key, sk, pk in (('steady', 's_stake', 's_pay'), ('aggressive', 'a_stake', 'a_pay')):
        a_st = np.array([x[sk] for x in cur], float); a_py = np.array([x[pk] for x in cur], float)
        b_st = np.array([x[sk] for x in new], float); b_py = np.array([x[pk] for x in new], float)
        out[key] = dict(current=summ(a_st, a_py), star=summ(b_st, b_py), diff=diff(a_st, a_py, b_st, b_py))
    one = np.full(len(races), 100.0)
    a_f = np.array([x['fuku'] for x in cur], float); b_f = np.array([x['fuku'] for x in new], float)
    out['fukusho'] = dict(current=summ(one, a_f), star=summ(one, b_f), diff=diff(one, a_f, one, b_f))
    return out


def main():
    stars = star_set()
    E.DB = os.path.join(WS, 'data', 'keiba.db')
    races = [r for r in E.load() if FROM <= r['date'] <= TO]
    n_all = len(races)
    sw = []
    for r in races:
        s = stars.get(r['rid'])
        if s and s[0] != r['axis'] and any(h['n'] == s[0] for h in r['horses']):
            r = dict(r, star=s[0], star_p=s[1])
            sw.append(r)
    n_star = sum(1 for r in races if r['rid'] in stars)
    rec = [r for r in sw if r['conf'] >= E.CONF_REC]
    res = dict(period=[FROM, TO], n_races=n_all, n_star_races=n_star, n_switch=len(sw), n_switch_rec=len(rec),
               all=evaluate(sw), rec=evaluate(rec) if len(rec) >= 10 else None)
    a = res['all']
    stop = []
    if a['aggressive']['diff']['d_hit_ci'][1] < 0:
        stop.append('1 勝負コースの的中率がはっきり下がる')
    if a['fukusho']['diff']['d_hit_ci'][1] < 0:
        stop.append('2 複勝の的中率がはっきり下がる')
    if len(sw) < 50:
        stop.append('3 差し替えが起きるレースが50本未満')
    res['stop'] = stop
    res['verdict'] = '止める' if stop else '実装へ進む'
    with open(os.path.join(REPO, 'biz', 'honmei_star_result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
