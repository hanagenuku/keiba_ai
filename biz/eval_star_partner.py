#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_STAR_PARTNER.md の測定（1回）。

    cd <隔離ワークスペース>
    python3 <repo>/biz/eval_star_partner.py <repo のパス>
"""
import importlib.util
import json
import os
import sys

import numpy as np

REPO = sys.argv[1]
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(REPO, 'biz'))
_s = importlib.util.spec_from_file_location('h', os.path.join(REPO, 'biz', 'eval_honmei_star.py'))
H = importlib.util.module_from_spec(_s)
_s.loader.exec_module(H)
E = H.E
T = H.T


def lines(r, star):
    hs, ax = r['horses'], r['axis']
    p, so, an = T.split_partners(hs, ax, T.SOLID_SHARE_STRONG, T.ANA_SHARE_STRONG)
    star_in = False
    if star is not None:
        p, so, an, star_in = T.include_star(hs, ax, p, so, an, star)
    bets = T.build_bet_lines(ax, p, so, an, allow_trifecta=True, star_num=star if star_in else None)
    out = {}
    for b in bets:
        g = b.get('group') or b['type']
        if g in E.BT and g not in out:
            out[g] = [tuple(c) for c in b['combos']]
    return out


def score(r, ls, types):
    st = sum(len(c) * 100 for t, c in ls.items() if t in types)
    py = sum(E.pay_of(r, t, c) for t, c in ls.items() if t in types)
    return st, py


def evaluate(pairs):
    out = {'races': len(pairs)}
    for key, types in (('aggressive', set(E.BT)), ('wide', {'ワイド'})):
        a = [score(r, c, types) for r, c, _ in pairs]
        b = [score(r, n, types) for r, _, n in pairs]
        a_st, a_py = np.array([x[0] for x in a], float), np.array([x[1] for x in a], float)
        b_st, b_py = np.array([x[0] for x in b], float), np.array([x[1] for x in b], float)
        out[key] = dict(current=H.summ(a_st, a_py), star=H.summ(b_st, b_py), diff=H.diff(a_st, a_py, b_st, b_py))
    return out


def main():
    stars = H.star_set()
    E.DB = os.path.join(os.getcwd(), 'data', 'keiba.db')
    races = [r for r in E.load() if H.FROM <= r['date'] <= H.TO]
    pairs = []
    for r in races:
        s = stars.get(r['rid'])
        if not s or s[0] == r['axis'] or not any(h['n'] == s[0] for h in r['horses']):
            continue
        cur, new = lines(r, None), lines(r, s[0])
        if cur != new:
            pairs.append((r, cur, new))
    rec = [x for x in pairs if x[0]['conf'] >= E.CONF_REC]
    res = dict(period=[H.FROM, H.TO], n_races=len(races), n_changed=len(pairs), n_changed_rec=len(rec),
               all=evaluate(pairs) if pairs else None, rec=evaluate(rec) if len(rec) >= 10 else None)
    stop = []
    if res['all'] and res['all']['aggressive']['diff']['d_hit_ci'][1] < 0:
        stop.append('1 勝負コースの的中率がはっきり下がる')
    if len(pairs) < 30:
        stop.append('2 買い目が変わるレースが30本未満')
    res['stop'] = stop
    res['verdict'] = '止める' if stop else '入れる'
    with open(os.path.join(REPO, 'biz', 'star_partner_result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
