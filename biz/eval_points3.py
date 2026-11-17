#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_POINTS3.md の測定（1回）。

    python3 biz/eval_points3.py <keiba.db のパス>
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'biz'))
import eval_points as E  # noqa: E402
import eval_points2 as P  # noqa: E402
from src.product import tickets as T  # noqa: E402

COURSE = {'馬連', 'ワイド', '三連複', '三連単'}
N_RAND = 1000


def per_race(races):
    """レースごとに 5頭BOX / 6頭BOX のコース（点数・払戻）と、ルールの入力値を並べる。"""
    out = []
    for r in races:
        a = P.rows_for([r], COURSE, tri='T5')
        b = P.rows_for([r], COURSE, tri='T6')
        if not a or not b:
            continue
        ps = P.partners5(r)
        odds = {h['n']: T.usable_odds(h) for h in r['horses']}
        fav = min([o for o in odds.values() if o] or [np.nan])
        p6 = odds.get(ps[4]) if len(ps) >= 5 else None
        out.append(dict(date=r['date'], s5=a[0]['stake'], y5=a[0]['pay'], s6=b[0]['stake'], y6=b[0]['pay'],
                        field=len(r['horses']), fav=fav, p6=p6 if p6 else np.nan, conf=r['conf']))
    return out


def course(rows, pick6):
    st = np.array([x['s6'] if k else x['s5'] for x, k in zip(rows, pick6)], float)
    py = np.array([x['y6'] if k else x['y5'] for x, k in zip(rows, pick6)], float)
    hit = py > 0
    return dict(points=st.mean() / 100, hit=hit.mean() * 100, roi=py.sum() / st.sum() * 100,
                loss=((py < st) & hit).sum() / max(hit.sum(), 1) * 100)


def eff(rows, pick6):
    c5, c = course(rows, [False] * len(rows)), course(rows, pick6)
    dp = c['points'] - c5['points']
    return (c['hit'] - c5['hit']) / dp if dp > 0 else float('nan')


def rules(rows):
    med = {k: float(np.nanmedian([x[k] for x in rows])) for k in ('field', 'fav', 'p6', 'conf')}
    return med, {
        'R1 頭数': [x['field'] > med['field'] for x in rows],
        'R2 オッズの割れ': [x['fav'] > med['fav'] for x in rows],
        'R3 6頭目の力': [(not np.isnan(x['p6'])) and x['p6'] <= med['p6'] for x in rows],
        'R4 ◎の信頼度': [x['conf'] < med['conf'] for x in rows],
    }


def r1(v):
    return round(float(v), 2)


def run(rows, seed=0):
    med, rs = rules(rows)
    rng = np.random.default_rng(seed)
    dates = sorted(x['date'] for x in rows)
    cut = dates[len(dates) // 2]
    halves = [[i for i, x in enumerate(rows) if x['date'] < cut], [i for i, x in enumerate(rows) if x['date'] >= cut]]
    always6 = [True] * len(rows)
    base = {k: {m: r1(v) for m, v in course(rows, p).items()} for k, p in
            (('常に5頭', [False] * len(rows)), ('常に6頭', always6))}
    base['常に6頭']['eff'] = r1(eff(rows, always6))
    res = {'n': len(rows), 'median': {k: r1(v) for k, v in med.items()}, 'split_date': cut, 'baseline': base, 'rules': []}
    for name, pick in rs.items():
        k = int(sum(pick))
        e = eff(rows, pick)
        rand = []
        for _ in range(N_RAND):
            p = np.zeros(len(rows), bool)
            p[rng.choice(len(rows), k, replace=False)] = True
            rand.append(eff(rows, list(p)))
        p95 = float(np.nanpercentile(rand, 95))
        half = []
        for idx in halves:
            sub = [rows[i] for i in idx]
            sp = [pick[i] for i in idx]
            half.append(dict(rule=r1(eff(sub, sp)), always6=r1(eff(sub, [True] * len(sub)))))
        ok1 = e > p95
        ok2 = all(h['rule'] > h['always6'] for h in half)
        c = course(rows, pick)
        res['rules'].append(dict(rule=name, n_6box=k, **{m: r1(v) for m, v in c.items()}, eff=r1(e),
                                 rand_eff_mean=r1(np.nanmean(rand)), rand_eff_p95=r1(p95), halves=half,
                                 ok_beats_random=bool(ok1), ok_both_halves=bool(ok2),
                                 verdict='使える' if ok1 and ok2 else '使えない'))
    return res


def main():
    races = E.load()
    rec = [r for r in races if r['conf'] >= E.CONF_REC]
    out = dict(period=[min(r['date'] for r in races), max(r['date'] for r in races)],
               rec=run(per_race(rec)), all=run(per_race(races)))
    with open(os.path.join(ROOT, 'biz', 'points3_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    E.DB = sys.argv[1] if len(sys.argv) > 1 else E.DB
    main()
