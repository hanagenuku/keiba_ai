#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_POINTS2.md の測定（1回）。

    python3 biz/eval_points2.py <keiba.db のパス>

POINTS-01 の読み込み・払戻照合・ブートストラップをそのまま使う。結果は biz/points2_result.json。
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'biz'))
import eval_points as E  # noqa: E402
from src.product import tickets as T  # noqa: E402

SE_MAX = 10.0


def partners5(r):
    p, _, _ = T.split_partners(r['horses'], r['axis'], 3, 2)
    return p


def box3(nums):
    return [tuple(c) for c in itertools.combinations(nums, 3)]


def multi(axis, ps):
    out = []
    for a, b in itertools.combinations(ps, 2):
        out += list(itertools.permutations((axis, a, b)))
    return out


def lines(r, tri='cur', tan='cur'):
    base = E.lines_for(r, 5, 2)
    ps = partners5(r)
    if tri == 'T5' and len(ps) >= 2:
        base['三連複'] = box3([r['axis']] + ps[:4])
    if tri == 'T6' and len(ps) >= 2:
        base['三連複'] = box3([r['axis']] + ps[:5])
    if tan == 'M5' and len(ps) >= 2:
        base['三連単'] = multi(r['axis'], ps[:5])
    return base


def rows_for(races, pick, **kw):
    out = []
    for r in races:
        ls = lines(r, **kw)
        ls = {t: c for t, c in ls.items() if t in pick}
        if not ls:
            continue
        per = {t: (len(c) * E.UNIT, E.pay_of(r, t, c)) for t, c in ls.items()}
        out.append(dict(rid=r['rid'], stake=sum(s for s, _ in per.values()), pay=sum(p for _, p in per.values()), per=per))
    return out


def compare(races, pick, kw_new, label):
    r0 = rows_for(races, pick)
    r1 = rows_for(races, pick, **kw_new)
    ids = {x['rid'] for x in r0} & {x['rid'] for x in r1}
    r0 = [x for x in r0 if x['rid'] in ids]
    r1 = [x for x in r1 if x['rid'] in ids]
    s0, st0, py0 = E.summarize(r0)
    s1, st1, py1 = E.summarize(r1)
    ci = E.paired_ci(st0, py0, st1, py1)
    hit_up = ci['d_hit_ci'][0] > 0
    if ci['d_roi_se'] > SE_MAX:
        roi = 'このデータ量では測れない'
    else:
        roi = '回収率が下がる' if ci['d_roi_ci'][1] < 0 else '下がるとは言えない'
    loss = s1['loss_on_hit'] - s0['loss_on_hit']
    keep = ('points_avg', 'hit_rate', 'roi', 'roi_wo_top1', 'loss_on_hit')
    return dict(label=label, current={k: s0[k] for k in keep}, new={k: s1[k] for k in keep}, vs_current=ci,
                verdict=dict(hit_rate_up=hit_up, roi=roi, loss_on_hit_diff=round(loss, 1),
                             loss_increase_over_5pt=loss > 5.0))


def run(races):
    tri, tan = {'三連複'}, {'三連単'}
    course = {'馬連', 'ワイド', '三連複', '三連単'}
    return [
        compare(races, tri, dict(tri='T5'), 'T5 三連複5頭BOX（vs ◎軸流し10点）'),
        compare(races, tri, dict(tri='T6'), 'T6 三連複6頭BOX（vs ◎軸流し10点）'),
        compare(races, tan, dict(tan='M5'), 'M5 三連単◎軸5頭流しマルチ（vs ◎1着固定20点）'),
        compare(races, course, dict(tri='T5'), 'C1 勝負コース: 三連複をT5に'),
        compare(races, course, dict(tri='T6'), 'C2 勝負コース: 三連複をT6に'),
        compare(races, course, dict(tan='M5'), 'C3 勝負コース: 三連単をM5に'),
        compare(races, course, dict(tri='T6', tan='M5'), 'C4 勝負コース: T6とM5'),
    ]


def main():
    races = E.load()
    rec = [r for r in races if r['conf'] >= E.CONF_REC]
    out = dict(period=[min(r['date'] for r in races), max(r['date'] for r in races)], n_rec=len(rec), n_all=len(races),
               rec=run(rec), all=run(races))
    with open(os.path.join(ROOT, 'biz', 'points2_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    E.DB = sys.argv[1] if len(sys.argv) > 1 else E.DB
    main()
