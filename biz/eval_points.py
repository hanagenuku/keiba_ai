#!/usr/bin/env python3
"""事前登録カード biz/CRITERIA_POINTS.md の測定（1回）。

    python3 biz/eval_points.py <keiba.db のパス>

本番の買い目組み立て（src/product/tickets.py）をそのまま呼び、相手の頭数とワイドの点数を変えて
確定払戻で答え合わせする。🔴 本番のファイルには何も書かない。結果は biz/points_result.json。
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.product import tickets as T  # noqa: E402

DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'keiba.db')
CONF_REC = T.CONF_STRONG
UNIT = 100
N_BOOT = 2000
BT = {'馬連': 'umaren', 'ワイド': 'wide', '三連複': 'sanrenpuku', '三連単': 'sanrentan'}
ORDERED = {'三連単'}
CUR_P, CUR_W = 5, 2


def load():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    horses = defaultdict(list)
    for r in c.execute('SELECT date, race_id, horse_num, rl_rank, tansho_odds, fuku_prob '
                       'FROM race_predictions WHERE actual_place IS NOT NULL'):
        horses[r['race_id']].append(dict(n=int(r['horse_num']), rl_rank=r['rl_rank'],
                                         odds=r['tansho_odds'], fuku=r['fuku_prob'], date=r['date']))
    div = defaultdict(lambda: defaultdict(int))
    for r in c.execute('SELECT race_id, bet_type, combo, payout FROM race_dividends'):
        div[r['race_id']][(r['bet_type'], r['combo'])] += int(r['payout'] or 0)
    races = []
    for rid, hs in horses.items():
        if rid not in div:
            continue
        ax = [h for h in hs if h['rl_rank'] == 1]
        if len(ax) != 1 or ax[0]['fuku'] is None:
            continue
        conf = min(99, max(0, int(round(float(ax[0]['fuku']) * 100))))
        races.append(dict(rid=rid, date=hs[0]['date'], horses=hs, axis=ax[0]['n'], conf=conf, div=div[rid]))
    races.sort(key=lambda r: r['rid'])
    return races


def key(type_, combo):
    if type_ in ORDERED:
        return '-'.join(str(x) for x in combo)
    return '-'.join(str(x) for x in sorted(combo))


def pay_of(race, type_, combos):
    d = race['div']
    return sum(d.get((BT[type_], key(type_, cb)), 0) for cb in combos) * UNIT // 100


def lines_for(race, n_partners, n_wide):
    """本番と同じ組み方で勝負コースの4行を作る。{券種: combos}"""
    ana_n = int(round(0.4 * n_partners))
    partners, solid, ana = T.split_partners(race['horses'], race['axis'], n_partners - ana_n, ana_n)
    bets = T.build_bet_lines(race['axis'], partners, solid, ana, allow_trifecta=True)
    out = {}
    for b in bets:
        g = b.get('group') or b['type']
        if g in BT and g not in out:            # 本線1行だけ（build_courses と同じ）
            out[g] = [tuple(cb) for cb in b['combos']]
    if 'ワイド' in out and n_wide != 2:
        ax = race['axis']
        base = [cb[1] for cb in out['ワイド']]
        extra = [p for p in partners if p not in base]
        nums = (base + extra)[:n_wide] if n_wide > 2 else base[:n_wide]
        out['ワイド'] = [(ax, p) for p in nums]
    return out


def evaluate(races, n_partners, n_wide):
    rows = []
    for r in races:
        ls = lines_for(r, n_partners, n_wide)
        per = {t: (len(cb) * UNIT, pay_of(r, t, cb)) for t, cb in ls.items()}
        stake = sum(s for s, _ in per.values())
        pay = sum(p for _, p in per.values())
        rows.append(dict(stake=stake, pay=pay, per=per))
    return rows


def summarize(rows):
    st = np.array([x['stake'] for x in rows], float)
    py = np.array([x['pay'] for x in rows], float)
    hit = py > 0
    out = dict(races=int(len(rows)), points_avg=round(float(st.mean() / UNIT), 1),
               hit_rate=round(float(hit.mean()) * 100, 1),
               roi=round(float(py.sum() / st.sum()) * 100, 1),
               roi_wo_top1=round(float((py.sum() - py.max()) / (st.sum() - st[py.argmax()])) * 100, 1),
               loss_on_hit=round(float(((py < st) & hit).sum() / max(hit.sum(), 1)) * 100, 1))
    per = {}
    for t in BT:
        s = np.array([x['per'][t][0] for x in rows if t in x['per']], float)
        p = np.array([x['per'][t][1] for x in rows if t in x['per']], float)
        if len(s):
            per[t] = dict(points_avg=round(float(s.mean() / UNIT), 1), hit_rate=round(float((p > 0).mean()) * 100, 1),
                          roi=round(float(p.sum() / s.sum()) * 100, 1))
    out['by_type'] = per
    return out, st, py


def paired_ci(st0, py0, st1, py1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(st0)
    d_roi, d_hit = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        d_roi.append(py1[i].sum() / st1[i].sum() - py0[i].sum() / st0[i].sum())
        d_hit.append((py1[i] > 0).mean() - (py0[i] > 0).mean())
    q = lambda a: [round(float(np.percentile(a, 2.5)) * 100, 1), round(float(np.percentile(a, 97.5)) * 100, 1)]
    return dict(d_roi=round(float(py1.sum() / st1.sum() - py0.sum() / st0.sum()) * 100, 1), d_roi_ci=q(d_roi),
                d_roi_se=round(float(np.std(d_roi)) * 100, 1),
                d_hit=round(float((py1 > 0).mean() - (py0 > 0).mean()) * 100, 1), d_hit_ci=q(d_hit))


def run(races, label):
    arms = [('P', p, CUR_W) for p in (3, 4, 5, 6, 7)] + [('W', CUR_P, w) for w in (1, 2, 3, 4)]
    base, st0, py0 = summarize(evaluate(races, CUR_P, CUR_W))
    res = {'label': label, 'current': base, 'arms': []}
    for kind, p, w in arms:
        s, st1, py1 = summarize(evaluate(races, p, w))
        ci = paired_ci(st0, py0, st1, py1)
        ok1 = ci['d_roi_ci'][1] >= 0
        ok2 = (s['loss_on_hit'] - base['loss_on_hit']) <= 5.0
        res['arms'].append(dict(kind=kind, partners=p, wide=w, **s, vs_current=ci,
                                criteria=dict(roi_not_clearly_lower=ok1, loss_on_hit_ok=ok2),
                                verdict=('現行' if (p, w) == (CUR_P, CUR_W) else ('増やしてよい' if ok1 and ok2 else '増やさない'))))
    return res


def main():
    races = load()
    rec = [r for r in races if r['conf'] >= CONF_REC]
    out = dict(db=os.path.basename(DB), period=[min(r['date'] for r in races), max(r['date'] for r in races)],
               n_all=len(races), n_rec=len(rec), n_boot=N_BOOT,
               rec=run(rec, f'推奨レース（conf>={CONF_REC}）'), all=run(races, '全レース（参考）'))
    with open(os.path.join(ROOT, 'biz', 'points_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
