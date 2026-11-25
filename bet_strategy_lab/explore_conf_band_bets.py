#!/usr/bin/env python3
"""信頼度帯 × 買い方の実測（記述統計）。

🔴 これは判定ではない。`bet_strategy_lab/EXPLORE_conf_band_bets.md` §3 のとおり、
   組み合わせ券種の実配当は 2026-06 以降しか無く、確認期が存在しない。
   ここの数字で本番の買い目を変えない。

使い方:
    python3 bet_strategy_lab/explore_conf_band_bets.py <keiba.db> > out.txt
"""

import itertools
import random
import sqlite3
import sys

BANDS = (('conf>=66', lambda c: c >= 66),
         ('55<=conf<66', lambda c: 55 <= c < 66),
         ('conf<55', lambda c: c < 55))

#: 券種ごとの払戻テーブルのキーの作り方。
#:   'set'   … 組み合わせ（昇順で突き合わせる）
#:   'order' … 着順どおり（三連単）
TYPE_KIND = {'tansho': 'set', 'fukusho': 'set', 'wide': 'set',
             'umaren': 'set', 'sanrenpuku': 'set', 'sanrentan': 'order'}


def _key(nums, kind):
    if kind == 'order':
        return '-'.join(str(n) for n in nums)
    return '-'.join(str(n) for n in sorted(nums))


def load(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    # 払戻表
    pays = {}
    for r in c.execute('SELECT race_id, bet_type, combo, payout FROM race_dividends'):
        pays.setdefault(r['race_id'], {}).setdefault(r['bet_type'], {})[
            str(r['combo'])] = float(r['payout'] or 0)
    # 予想（全頭）
    races = {}
    for r in c.execute('SELECT race_id, horse_num, popularity, rl_rank '
                       'FROM race_predictions ORDER BY race_id, rl_rank'):
        races.setdefault(r['race_id'], []).append(dict(r))
    # conf（generate_stats._axis_conf_map と同じ式）
    conf = {}
    for r in c.execute('SELECT race_id, fuku_prob FROM race_predictions WHERE rl_rank = 1'):
        if r['fuku_prob'] is not None:
            conf[r['race_id']] = min(99, max(0, int(round(float(r['fuku_prob']) * 100))))
    return races, pays, conf


# ── 買い方（EXPLORE_conf_band_bets.md §4 の12＋3個。結果を見て足さない）──────
def _ai(hs, lo, hi):
    """AI順位 lo〜hi 位の馬番（1始まり・両端含む）。"""
    out = [h['horse_num'] for h in hs
           if h['rl_rank'] and lo <= h['rl_rank'] <= hi]
    return out


def _ana(hs, n):
    """人気5〜12番のうち AI 順位が上の n 頭。"""
    c = [h for h in hs if h['popularity'] and 5 <= h['popularity'] <= 12]
    c.sort(key=lambda h: h['rl_rank'] or 99)
    return [h['horse_num'] for h in c[:n]]


def _pop(hs, lo, hi):
    out = [(h['popularity'], h['horse_num']) for h in hs
           if h['popularity'] and lo <= h['popularity'] <= hi]
    return [n for _p, n in sorted(out)]


def build(hs):
    """1レースぶんの {キー: (券種, [買い目, ...])}。組めない構成は入れない。"""
    axis = _ai(hs, 1, 1)
    if not axis:
        return {}
    a = axis[0]
    p2_3, p2_4, p2_6, p2_9 = _ai(hs, 2, 3), _ai(hs, 2, 4), _ai(hs, 2, 6), _ai(hs, 2, 9)
    top5 = _ai(hs, 1, 5)
    ana2, ana3 = _ana(hs, 2), _ana(hs, 3)
    pop3, pop5 = _pop(hs, 1, 3), _pop(hs, 1, 5)
    out = {}

    def put(key, bt, combos, need):
        if need and combos:
            out[key] = (bt, combos)

    put('tan_axis', 'tansho', [[a]], True)
    put('fuku_axis', 'fukusho', [[a]], True)
    put('wide_ai2', 'wide', [[a, x] for x in p2_3], len(p2_3) == 2)
    put('umaren_ai3', 'umaren', [[a, x] for x in p2_4], len(p2_4) == 3)
    put('umaren_ai5', 'umaren', [[a, x] for x in p2_6], len(p2_6) == 5)
    put('trio_ai5', 'sanrenpuku',
        [[a, x, y] for x, y in itertools.combinations(p2_6, 2)], len(p2_6) == 5)
    put('trio_box5', 'sanrenpuku',
        [list(c) for c in itertools.combinations(top5, 3)], len(top5) == 5)
    put('trio_form_wide', 'sanrenpuku',
        sorted({tuple(sorted((a, x, y))) for x in p2_6 for y in p2_9 if x != y}),
        len(p2_6) == 5 and len(p2_9) == 8)
    put('trifecta_ai3', 'sanrentan',
        [[a, x, y] for x in p2_4 for y in p2_4 if x != y], len(p2_4) == 3)
    put('wide_ana', 'wide', [[a, x] for x in ana2], len(ana2) == 2)
    put('trio_ana', 'sanrenpuku',
        [[a, x, y] for x in p2_3 for y in ana3], len(p2_3) == 2 and len(ana3) == 3)
    put('trio_noaxis', 'sanrenpuku',
        [list(c) for c in itertools.combinations(p2_6, 3)], len(p2_6) == 5)
    # 市場ベースライン
    put('umaren_pop3', 'umaren',
        [list(c) for c in itertools.combinations(pop3, 2)], len(pop3) == 3)
    put('trio_pop5', 'sanrenpuku',
        [list(c) for c in itertools.combinations(pop5, 3)], len(pop5) == 5)
    put('tan_pop1', 'tansho', [[pop3[0]]] if pop3 else [], bool(pop3))
    return out


def race_rows(races, pays, conf):
    """(帯, キー, 点数, 払戻) の行。払戻が取れない券種のレースは落とす。"""
    rows = []
    for rid, hs in races.items():
        if rid not in conf or rid not in pays:
            continue
        band = next((nm for nm, f in BANDS if f(conf[rid])), None)
        if band is None:
            continue
        for key, (bt, combos) in build(hs).items():
            table = pays[rid].get(bt)
            if not table:                 # その券種の払戻表が無い＝評価不能
                continue
            kind = TYPE_KIND[bt]
            pay = sum(table.get(_key(c, kind), 0.0) for c in combos)
            rows.append((band, key, len(combos), pay))
    return rows


def _roi(rs):
    inv = sum(n for _b, _k, n, _p in rs) * 100
    pay = sum(p for _b, _k, _n, p in rs)
    return (pay / inv * 100) if inv else None


def boot(rs, n=2000, seed=0):
    if not rs:
        return None, None
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rs[rnd.randrange(len(rs))] for _ in range(len(rs))]
        v = _roi(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(len(vals) * 0.025)], vals[int(len(vals) * 0.975)]


def main(argv):
    races, pays, conf = load(argv[0])
    rows = race_rows(races, pays, conf)
    keys = ['tan_axis', 'fuku_axis', 'wide_ai2', 'umaren_ai3', 'umaren_ai5',
            'trio_ai5', 'trio_box5', 'trio_form_wide', 'trifecta_ai3',
            'wide_ana', 'trio_ana', 'trio_noaxis',
            'umaren_pop3', 'trio_pop5', 'tan_pop1']
    print(f'母集団: {len(races)}レース / 行 {len(rows)}')
    for band, _f in BANDS:
        brs = [r for r in rows if r[0] == band]
        nrace = len({None for _ in ()}) or 0
        print(f'\n### {band}（この帯のレース数 '
              f'{len([1 for rid in races if rid in conf and next((nm for nm, f in BANDS if f(conf[rid])), None) == band])}）')
        print(f"{'買い方':<16}{'レース':>6}{'点数':>5}{'的中率':>8}{'回収率':>9}"
              f"{'3本抜き':>9}  95%CI")
        for k in keys:
            rs = [r for r in brs if r[1] == k]
            if not rs:
                continue
            hit = sum(1 for _b, _k, _n, p in rs if p > 0)
            roi = _roi(rs)
            cut = _roi(sorted(rs, key=lambda r: -r[3])[3:])
            lo, hi = boot(rs)
            w = (hi - lo) if (lo is not None and hi is not None) else None
            ci = (f'[{lo:6.1f},{hi:7.1f}] 幅{w:5.1f}'
                  + ('  🔴幅60pt超＝点推定を読まない' if w and w > 60 else '')) if w else ''
            print(f'{k:<16}{len(rs):>6}{rs[0][2]:>5}{hit / len(rs) * 100:>7.1f}%'
                  f'{roi:>8.1f}%{cut:>8.1f}%  {ci}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
