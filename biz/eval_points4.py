#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_POINTS4.md の測定（1回）。

    python3 biz/eval_points4.py <朝の latest.json を置いたディレクトリ> <keiba.db>

ディレクトリには YYYYMMDD.json（その日最初の Weekend refresh コミットの data/latest.json）を置く。
"""
import glob
import itertools
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROM, TO = '20261003', '20261115'
CONF_REC = 66
N_BOOT = 2000
SE_MAX = 10.0


def load(snap_dir, db):
    c = sqlite3.connect(db)
    tri = defaultdict(int)
    for rid, combo, pay in c.execute("SELECT race_id, combo, payout FROM race_dividends WHERE bet_type='sanrenpuku'"):
        tri[(rid, combo)] += int(pay or 0)
    have = {rid for rid, _ in tri}
    races = []
    for f in sorted(glob.glob(os.path.join(snap_dir, '*.json'))):
        d8 = os.path.basename(f)[:8]
        if not (FROM <= d8 <= TO):
            continue
        R = json.load(open(f, encoding='utf-8')).get('races', {})
        for venue, rs in R.items():
            for r in rs:
                rid = r.get('race_id')
                hs = [h for h in r.get('horses', []) if isinstance(h.get('rl_rank'), int) and 0 < h['rl_rank'] < 99]
                if not rid or rid not in have or len(hs) < 6 or r.get('conf') is None:
                    continue
                races.append(dict(rid=rid, date=d8, conf=r['conf'], horses=hs, star=(r.get('star') or {}).get('n')))
    return races, tri


def by_rank(hs, key):
    return [h['n'] for h in sorted(hs, key=lambda h: (h.get(key) or 99, h['n']))]


def compose(r, size):
    hs = r['horses']
    rl = by_rank(hs, 'rl_rank')
    core = list(rl[:3])
    extra = []
    fit = [h['n'] for h in hs if h.get('fit_rank') == 1]
    solo = [h['n'] for h in hs if h.get('solo_rank') == 1]
    for n in fit[:1] + solo[:1] + ([r['star']] if r['star'] else []):
        if n not in core and n not in extra:
            extra.append(n)
    s = core + extra
    for n in rl:
        if len(s) >= size:
            break
        if n not in s:
            s.append(n)
    return s, len(extra)


def score(r, tri, nums):
    combos = list(itertools.combinations(sorted(nums), 3))
    pay = sum(tri.get((r['rid'], '-'.join(map(str, c))), 0) for c in combos)
    return len(combos) * 100, pay


def summarize(st, py):
    hit = py > 0
    i = int(np.argmax(py))
    return dict(points=round(float(st.mean() / 100), 1), hit_rate=round(float(hit.mean() * 100), 1),
                roi=round(float(py.sum() / st.sum() * 100), 1),
                roi_wo_top1=round(float((py.sum() - py[i]) / (st.sum() - st[i]) * 100), 1),
                loss_on_hit=round(float(((py < st) & hit).sum() / max(hit.sum(), 1) * 100), 1))


def compare(races, tri, size):
    a = [score(r, tri, compose(r, size)[0]) for r in races]
    b = [score(r, tri, by_rank(r['horses'], 'rl_rank')[:size]) for r in races]
    sa, pa = np.array([x[0] for x in a], float), np.array([x[1] for x in a], float)
    sb, pb = np.array([x[0] for x in b], float), np.array([x[1] for x in b], float)
    rng = np.random.default_rng(0)
    dh, dr = [], []
    n = len(races)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        dh.append((pa[i] > 0).mean() - (pb[i] > 0).mean())
        dr.append(pa[i].sum() / sa[i].sum() - pb[i].sum() / sb[i].sum())
    q = lambda v: [round(float(np.percentile(v, 2.5)) * 100, 1), round(float(np.percentile(v, 97.5)) * 100, 1)]
    hci, rci, rse = q(dh), q(dr), round(float(np.std(dr)) * 100, 1)
    hit_v = '上がる' if hci[0] > 0 else ('下がる' if hci[1] < 0 else '差は言えない')
    roi_v = '測れない' if rse > SE_MAX else ('下がる' if rci[1] < 0 else '下がるとは言えない')
    extras = [compose(r, size)[1] for r in races]
    return dict(size=size, races=n, A=summarize(sa, pa), B=summarize(sb, pb),
                d_hit=round(float(((pa > 0).mean() - (pb > 0).mean()) * 100), 1), d_hit_ci=hci,
                d_roi=round(float(pa.sum() / sa.sum() - pb.sum() / sb.sum()) * 100, 1), d_roi_ci=rci, d_roi_se=rse,
                verdict=dict(hit=hit_v, roi=roi_v),
                extra_horses={str(k): extras.count(k) for k in sorted(set(extras))},
                same_as_B=int(sum(set(compose(r, size)[0]) == set(by_rank(r['horses'], 'rl_rank')[:size]) for r in races)),
                star_races=int(sum(1 for r in races if r['star'])))


def main():
    races, tri = load(sys.argv[1], sys.argv[2])
    rec = [r for r in races if r['conf'] >= CONF_REC]
    out = dict(period=[min(r['date'] for r in races), max(r['date'] for r in races)],
               days=len({r['date'] for r in races}), n_all=len(races), n_rec=len(rec),
               rec=[compare(rec, tri, s) for s in (5, 6)], all=[compare(races, tri, s) for s in (5, 6)])
    with open(os.path.join(ROOT, 'biz', 'points4_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
