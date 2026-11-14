#!/usr/bin/env python3
"""相手の中穴を「前に行けそうな馬」から選ぶ案の測定（カード `tenkai/CRITERIA_PARTNER_FRONT.md`）。

🔴 測定は1回。ルール・母集団・基準はカードのとおりで、結果を見てから変えない。
🔴 中止条件は結果より手前で印字して判定する。data/ には1バイトも書かない。

使い方:
    cd <隔離ワークスペース>   # data/history.db・data/keiba.db の実体がある場所
    python3 tenkai/eval_partner_front.py
"""
import json
import os
import sqlite3
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.getcwd())

from scripts.generate_stats import _axis_conf_map, _dividend_map  # noqa: E402
from src.product import tickets  # noqa: E402
from src.scraper.jra_scraper import _infer_running_style  # noqa: E402
from src.tools.build_training_data import _get_history_before  # noqa: E402

FRONT_STYLES = ('逃げ', '先行')
N_SHUFFLE = 200
SEED = 20261113
N_BOOT = 1000


def split_partners_variant(horses, axis_num, n_solid, n_ana, ana_order):
    """`tickets.split_partners` と同じ規則で、中穴の候補の並べ方だけ差し替える。

    ana_order(cands_ana) -> 並べ替えた馬番リスト。cands_ana は [(rl_rank, 馬番, オッズ)]（rl_rank 昇順）。
    """
    cands = tickets._partner_candidates(horses, axis_num)
    total = n_solid + n_ana
    solid = [n for _, n, o in cands if o < tickets.ANA_ODDS_MIN]
    ana = ana_order([c for c in cands if tickets.ANA_ODDS_MIN <= c[2] <= tickets.ANA_ODDS_MAX])
    got_solid, got_ana = solid[:n_solid], ana[:n_ana]
    short = total - len(got_solid) - len(got_ana)
    if short > 0:
        got_solid += solid[len(got_solid):len(got_solid) + short]
    short = total - len(got_solid) - len(got_ana)
    if short > 0:
        got_ana += ana[len(got_ana):len(got_ana) + short]
    order = {n: i for i, (_, n, _o) in enumerate(cands)}
    partners = sorted(got_solid + got_ana, key=lambda n: order.get(n, 99))
    return partners, got_solid, got_ana


def lines(axis, partners, solid, ana):
    """馬連・ワイド・三連複(流し) の組番（本番 `build_bet_lines` と同じ形）。"""
    bets = {b['type']: b['combos'] for b in
            tickets.build_bet_lines(axis, partners, solid, ana, allow_trifecta=False)}
    return {k: bets.get(k, []) for k in ('馬連', 'ワイド', '三連複')}


KEY = {'馬連': 'umaren', 'ワイド': 'wide', '三連複': 'sanrenpuku'}


def settle(combos, table):
    """(購入点数, 払戻合計, 的中したか)。table は {組番: 払戻}。"""
    pay = 0
    for c in combos:
        pay += table.get('-'.join(str(n) for n in sorted(c)), 0) or 0
    return len(combos), pay, pay > 0


def main():
    base = os.getcwd()
    k = sqlite3.connect(os.path.join(base, 'data', 'keiba.db'))
    k.row_factory = sqlite3.Row
    h = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    h.row_factory = sqlite3.Row
    confs, divs = _axis_conf_map(k), _dividend_map(k)

    by_race = {}
    for r in k.execute('SELECT race_id, date, horse_num, horse_name, tansho_odds, rl_rank '
                       'FROM race_predictions WHERE tansho_odds > 0'):
        by_race.setdefault(r['race_id'], []).append(dict(r))
    place = {(r['race_id'], r['horse_num']): r['place'] for r in
             h.execute("SELECT race_id, horse_num, place FROM horse_history WHERE race_id >= '20260601'")}

    races, n_style_missing, n_horses = [], 0, 0
    for rid, rows in sorted(by_race.items()):
        conf = confs.get(rid)
        d = divs.get(rid) or {}
        if conf is None or 'umaren' not in d or 'sanrenpuku' not in d:
            continue
        v = tickets.verdict(conf)
        date = str(rows[0]['date'])[:10]
        horses = []
        for r in rows:
            hist = _get_history_before(h, r['horse_name'], date)
            st = _infer_running_style(r['horse_name'], hist, r['horse_num'])
            n_horses += 1
            n_style_missing += st is None
            horses.append({'n': int(r['horse_num']), 'rl_rank': int(r['rl_rank']) if r['rl_rank'] else None,
                           'odds': float(r['tansho_odds']), 'style': st,
                           'place': place.get((rid, r['horse_num']))})
        axis = next((x for x in horses if x['rl_rank'] == 1), None)
        if axis is None:
            continue
        races.append(dict(rid=rid, conf=conf, verdict=v, horses=horses, axis=axis, div=d))

    main_set = [r for r in races if r['verdict'] in ('strong', 'light')]
    strong = [r for r in races if r['verdict'] == 'strong']

    def shares(r):
        return ((tickets.SOLID_SHARE_STRONG, tickets.ANA_SHARE_STRONG) if r['verdict'] == 'strong'
                else (tickets.SOLID_SHARE_LIGHT, tickets.ANA_SHARE_LIGHT))

    r0_order = lambda c: [n for _, n, _o in c]                                     # noqa: E731
    style_of = lambda r: {x['n']: x['style'] for x in r['horses']}                 # noqa: E731

    def r1_order_for(r):
        st = style_of(r)
        return lambda c: ([n for _, n, _o in c if st.get(n) in FRONT_STYLES]
                          + [n for _, n, _o in c if st.get(n) not in FRONT_STYLES])

    # ── 中止条件 ─────────────────────────────────────────────
    print('=' * 78)
    print('■ 中止条件（カード §5）— 結果を読む手前で判定する')
    print('=' * 78)
    stop = []
    print(f'  1. 対象レース strong {len(strong)} + light {len(main_set) - len(strong)} = {len(main_set)}'
          f'（全 {len(races)}）  300 以上か: {len(main_set) >= 300}')
    if len(main_set) < 300:
        stop.append('対象レースが300未満')
    fuku = [(r['axis']['place'] is not None and 1 <= r['axis']['place'] <= 3) for r in strong]
    rate = float(np.mean(fuku)) * 100 if fuku else float('nan')
    ok = 60 <= rate <= 71
    print(f'  2. strong の ◎ 3着内率 = {rate:.1f}%  60〜71% の内か: {ok}')
    if not ok:
        stop.append(f'strong の◎3着内率 {rate:.1f}% が帯の外')
    miss = n_style_missing / max(n_horses, 1)
    print(f'  3. style が付かない馬 {n_style_missing}/{n_horses} = {miss:.2%}  1% 以下か: {miss <= 0.01}')
    if miss > 0.01:
        stop.append('style の欠損が1%超')
    same = True
    for r in main_set:
        ns, na = shares(r)
        a = tickets.split_partners(r['horses'], r['axis']['n'], ns, na)
        b = split_partners_variant(r['horses'], r['axis']['n'], ns, na, r0_order)
        same = same and (list(a[0]), list(a[1]), list(a[2])) == (list(b[0]), list(b[1]), list(b[2]))
    print(f'  4. R0 の相手が本番の `tickets.split_partners` と全レースで一致: {same}')
    if not same:
        stop.append('R0 の再現が本番と一致しない')
    out = {'n_races': len(races), 'n_main': len(main_set), 'n_strong': len(strong),
           'strong_axis_top3': rate, 'style_missing': miss}
    if stop:
        print('\n🔴 中止条件に該当。結果を読まずに破棄する:', stop)
        out['stopped'] = stop
        json.dump(out, open('partner_front_result.json', 'w'), ensure_ascii=False, indent=2)
        return
    print('\n  ✅ 中止条件はすべて通過。測定に進む。')

    def run(order_for, pop):
        """ルールを母集団に当てる。戻り値: レースごとの記録のリスト。"""
        recs = []
        for r in pop:
            ns, na = shares(r)
            partners, solid, ana = split_partners_variant(
                r['horses'], r['axis']['n'], ns, na, order_for(r))
            byn = {x['n']: x for x in r['horses']}
            rec = {'rid': r['rid'], 'ana': list(ana),
                   'ana_top3': [int(byn[n]['place'] is not None and 1 <= byn[n]['place'] <= 3) for n in ana],
                   'ana_odds': [byn[n]['odds'] for n in ana]}
            ls = lines(r['axis']['n'], partners, solid, ana)
            for t, combos in ls.items():
                table = r['div'].get(KEY[t])
                rec[t] = None if (not table or not combos) else settle(combos, table)
            recs.append(rec)
        return recs

    def ana_rate(recs):
        v = [x for rec in recs for x in rec['ana_top3']]
        return float(np.mean(v)) * 100, len(v)

    def summarize(recs):
        s = {}
        v = [x for rec in recs for x in rec['ana_top3']]
        o = [x for rec in recs for x in rec['ana_odds']]
        s['ana'] = dict(n=len(v), top3=float(np.mean(v)) * 100, odds=float(np.mean(o)))
        tot_pts = tot_pay = 0
        for t in KEY:
            got = [rec[t] for rec in recs if rec[t] is not None]
            pts, pay, hit = sum(g[0] for g in got), sum(g[1] for g in got), sum(g[2] for g in got)
            s[t] = dict(races=len(got), hit=int(hit), hit_rate=hit / len(got) * 100,
                        points=pts, roi=pay / (pts * 100) * 100)
            tot_pts += pts; tot_pay += pay
        s['roi_all'] = tot_pay / (tot_pts * 100) * 100
        return s

    R0 = run(lambda r: r0_order, main_set)
    R1 = run(r1_order_for, main_set)
    s0, s1 = summarize(R0), summarize(R1)
    changed = sum(1 for a, b in zip(R0, R1) if a['ana'] != b['ana'])

    # 対照群A: 中穴の候補をでたらめな順にする
    rng = np.random.default_rng(SEED)
    base_rate = ana_rate(R0)[0]
    null = []
    for _ in range(N_SHUFFLE):
        def rand_for(r):
            return lambda c: [c[i][1] for i in rng.permutation(len(c))]
        null.append(ana_rate(run(rand_for, main_set))[0] - base_rate)
    null = np.array(null)
    d1 = s1['ana']['top3'] - s0['ana']['top3']
    p95 = float(np.percentile(null, 95))

    # 対照群B: 検算
    v = [x for rec in R0 for x in rec['ana_top3']]
    kflip = 10
    v2 = list(v); done = 0
    for i, x in enumerate(v2):
        if x == 0 and done < kflip:
            v2[i] = 1; done += 1
    b1 = (np.mean(v2) - np.mean(v)) - kflip / len(v)
    got = [rec['馬連'] for rec in R0 if rec['馬連'] is not None]
    pts = sum(g[0] for g in got); pay = sum(g[1] for g in got)
    b2 = ((pay + 7 * pts) / (pts * 100) * 100 - pay / (pts * 100) * 100) - 7.0

    # 回収率の差のブートストラップ（レースを復元抽出）
    def roi_all(recs, idx):
        p = q = 0
        for i in idx:
            for t in KEY:
                g = recs[i][t]
                if g is not None:
                    p += g[0]; q += g[1]
        return q / (p * 100) * 100
    n = len(R0)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(roi_all(R1, idx) - roi_all(R0, idx))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    print('\n' + '=' * 78)
    print(f'■ 結果（strong＋light {len(main_set)}レース・買い目が変わったレース {changed}）')
    print('=' * 78)
    for lab, s in (('R0 現行', s0), ('R1 前に行けそうな馬を先', s1)):
        print(f'  {lab}')
        print(f'      中穴枠の馬  N={s["ana"]["n"]}  3着内率 {s["ana"]["top3"]:.2f}%  平均単勝オッズ {s["ana"]["odds"]:.2f}倍')
        for t in KEY:
            x = s[t]
            print(f'      {t:4s} 的中 {x["hit"]:3d}/{x["races"]}（{x["hit_rate"]:.2f}%）  回収率 {x["roi"]:.1f}%  {x["points"]}点')
        print(f'      3券種合算の回収率 {s["roi_all"]:.1f}%')
    hr0 = np.mean([s0[t]['hit_rate'] for t in KEY]); hr1 = np.mean([s1[t]['hit_rate'] for t in KEY])
    worst = min(s1[t]['hit_rate'] - s0[t]['hit_rate'] for t in KEY)
    droi = s1['roi_all'] - s0['roi_all']
    print(f'\n  中穴枠の3着内率の差 {d1:+.2f}pt   対照群A（でたらめな順 {N_SHUFFLE}回）平均 {null.mean():+.2f}pt・sd {null.std():.2f}pt・上側5%点 {p95:+.2f}pt')
    print(f'  3券種の的中率の平均 R0 {hr0:.2f}% → R1 {hr1:.2f}%   最も悪い券種の差 {worst:+.2f}pt')
    print(f'  3券種合算の回収率の差 {droi:+.1f}pt   ブートストラップ95%区間 [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    print(f'  対照群B  3着内率の書き換えの誤差 {b1:.2e} / 払戻 +7円の誤差 {b2:.2e}')

    c1 = d1 >= 1.0 and d1 > p95
    c2 = hr1 > hr0 and worst >= -0.5
    c3 = s1['ana']['odds'] >= 0.9 * s0['ana']['odds']
    c4 = droi >= -5.0
    cb = abs(b1) < 1e-9 and abs(b2) < 1e-9
    print('\n■ 基準（カード §4）')
    for lab, ok in (('1 中穴枠の3着内率 +1.0pt 以上・対照群Aの上側5%点超', c1),
                    ('2 3券種の的中率（平均で上・どれも −0.5pt より悪化しない）', c2),
                    ('3 平均単勝オッズが R0 の 0.9倍以上', c3),
                    ('4 3券種合算の回収率の差 −5.0pt 以上', c4), ('対照群B', cb)):
        print(f'  {"✅" if ok else "🔴"} {lab}')
    verdict = ('計測器の不具合' if not cb else '採用' if (c1 and c2 and c3 and c4) else '不採用')
    print(f'\n■ 判定（カード §6）: {verdict}')

    # 参考: strong だけ
    sS0 = summarize([a for a, r in zip(R0, main_set) if r['verdict'] == 'strong'])
    sS1 = summarize([a for a, r in zip(R1, main_set) if r['verdict'] == 'strong'])
    print(f'\n  参考（strong {len(strong)}レースだけ） 中穴枠 {sS0["ana"]["top3"]:.2f}% → {sS1["ana"]["top3"]:.2f}%   '
          + ' / '.join(f'{t} {sS0[t]["hit_rate"]:.1f}→{sS1[t]["hit_rate"]:.1f}%' for t in KEY))

    out.update(dict(changed=changed, R0=s0, R1=s1, ana_delta=d1,
                    control_a=dict(mean=float(null.mean()), sd=float(null.std()), p95=p95),
                    control_b=dict(flip_err=float(b1), bonus_err=float(b2)),
                    roi_delta=droi, roi_delta_ci=ci, strong_R0=sS0, strong_R1=sS1,
                    criteria=dict(c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4), cb=bool(cb)),
                    verdict=verdict))
    json.dump(out, open('partner_front_result.json', 'w'), ensure_ascii=False, indent=2, default=float)


if __name__ == '__main__':
    main()
