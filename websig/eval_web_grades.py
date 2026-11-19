#!/usr/bin/env python3
"""事前登録カード websig/CRITERIA_GRADES_<source>.md の測定（1回・GitHub ランナー）。

    python3 websig/eval_web_grades.py --source rihua-keiba

結果は websig/grades_<source>_result.json（集計だけ。馬ごとの評価は書かない）。
"""
import argparse
import datetime as dt
import json
import os
import sqlite3
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT, N_PERM = 2000, 200


def fs_bucket(n):
    return 0 if n <= 12 else (1 if n <= 15 else 2)


def load_history():
    con = sqlite3.connect(os.path.join(ROOT, 'data', 'history.db'))
    rows = con.execute('SELECT race_id, date, horse_name, place, popularity, field_size, fukusho_payout '
                       'FROM horse_history WHERE place > 0').fetchall()
    races = defaultdict(list)
    for rid, d, name, place, pop, fs, fuku in rows:
        races[rid].append(dict(date=str(d)[:10], name=name, place=place, pop=pop, fs=fs, fuku=fuku or 0))
    base_t3, base_beat = defaultdict(list), defaultdict(list)
    for rid, hs in races.items():
        n = len(hs)
        fav = [h for h in hs if h['pop'] == 1]
        for h in hs:
            if not h['pop'] or not 1 <= h['pop'] <= 18:
                continue
            k = (min(h['pop'], 12), fs_bucket(n))
            base_t3[k].append(h['place'] <= 3)
            if len(fav) == 1 and h['pop'] != 1:
                base_beat[k].append(h['place'] < fav[0]['place'])
    t3 = {k: float(np.mean(v)) for k, v in base_t3.items()}
    bt = {k: float(np.mean(v)) for k, v in base_beat.items()}
    by_date = defaultdict(list)
    for rid, hs in races.items():
        by_date[hs[0]['date']].append(rid)
    return races, t3, bt, by_date


def _year_of(title):
    import re
    import unicodedata
    m = re.search(r'(20\d{2})', unicodedata.normalize('NFKC', title or ''))
    return int(m.group(1)) if m else None


def match(blocks, races, by_date, titles=None):
    """記事の塊 → race_id。公開日から14日以内で馬名が最も重なるレース。

    公開日が取れていない記事は、題名の年（例【…２０２４】）の1年間から探す（2026-11-20 追記・カード §データ）。
    """
    out = {}
    titles = titles or {}
    for key, b in blocks.items():
        names = {h['horse_name'] for h in b}
        days = []
        try:
            d0 = dt.date.fromisoformat(b[0]['published'])
            days = [d0 + dt.timedelta(days=k) for k in range(0, 15)]
        except Exception:
            y = _year_of(titles.get(key[0]))
            if y:
                d0 = dt.date(y, 1, 1)
                days = [d0 + dt.timedelta(days=k) for k in range(0, 366)]
        best, best_n = None, 0
        for day in days:
            for rid in by_date.get(str(day), []):
                n = len(names & {h['name'] for h in races[rid]})
                if n > best_n:
                    best, best_n = rid, n
        if best and best_n >= max(3, 0.5 * len(names)):
            out[key] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    a = ap.parse_args()
    races, t3, bt, by_date = load_history()
    g = sqlite3.connect(os.path.join(ROOT, 'data', 'private', 'web_grades.db'))
    g.row_factory = sqlite3.Row
    rows = [dict(r) for r in g.execute('SELECT * FROM web_grades WHERE source=? AND one_week=0', (a.source,))]
    blocks = defaultdict(list)
    for r in rows:
        blocks[(r['url'], r['block'])].append(r)
    titles = {u: t for u, t in g.execute('SELECT url, title FROM web_grade_articles WHERE source=?', (a.source,))}
    n_pub = sum(1 for b in blocks.values() if b[0]['published'])
    print(f'塊 {len(blocks)}（公開日あり {n_pub}）', flush=True)
    m = match(blocks, races, by_date, titles)
    data = []
    for key, rid in m.items():
        hs = {h['name']: h for h in races[rid]}
        n = len(races[rid])
        fav = [h for h in races[rid] if h['pop'] == 1]
        for r in blocks[key]:
            h = hs.get(r['horse_name'])
            if not h or not h['pop'] or not 1 <= h['pop'] <= 18:
                continue
            k = (min(h['pop'], 12), fs_bucket(n))
            beat = None
            if len(fav) == 1 and h['pop'] != 1:
                beat = float(h['place'] < fav[0]['place']) - bt.get(k, np.nan)
            data.append(dict(rid=rid, grade=r['grade'], rec=r['recommended'], pop=h['pop'],
                             top3=float(h['place'] <= 3), ex_t3=float(h['place'] <= 3) - t3.get(k, np.nan),
                             ex_beat=beat, fuku=h['fuku']))
    res = dict(source=a.source, articles_rows=len(rows), blocks=len(blocks), blocks_with_date=n_pub, matched_blocks=len(m),
               match_rate=round(len(m) / max(len(blocks), 1), 3), horses=len(data),
               period=[min((races[r][0]['date'] for r in m.values()), default=None),
                       max((races[r][0]['date'] for r in m.values()), default=None)])

    def summ(sel):
        if not sel:
            return dict(n=0)
        ex = [x['ex_beat'] for x in sel if x['ex_beat'] is not None and not np.isnan(x['ex_beat'])]
        return dict(n=len(sel), races=len({x['rid'] for x in sel}), top3=round(np.mean([x['top3'] for x in sel]) * 100, 1),
                    ex_t3=round(np.mean([x['ex_t3'] for x in sel]) * 100, 2),
                    ex_beat=round(np.mean(ex) * 100, 2) if ex else None, n_beat=len(ex),
                    fuku_roi=round(sum(x['fuku'] for x in sel) / len(sel), 1),
                    pop_mean=round(np.mean([x['pop'] for x in sel]), 2))
    res['by_grade'] = {gr: summ([x for x in data if x['grade'] == gr]) for gr in ('S', 'A', 'B', 'C', None)}
    res['by_recommended'] = {k: summ([x for x in data if bool(x['rec']) == v]) for k, v in (('yes', True), ('no', False))}

    graded = [x for x in data if x['grade'] in ('A', 'C')]
    by_race = defaultdict(list)
    for x in [x for x in data if x['grade']]:
        by_race[x['rid']].append(x)
    rids = list(by_race)

    def D(sample):
        A = [x['ex_t3'] for x in sample if x['grade'] == 'A']
        C = [x['ex_t3'] for x in sample if x['grade'] == 'C']
        return (np.mean(A) - np.mean(C)) if A and C else np.nan
    allg = [x for r in rids for x in by_race[r]]
    d = D(allg)
    rng = np.random.default_rng(0)
    boots, bootA = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(rids), len(rids))
        smp = [x for i in pick for x in by_race[rids[i]]]
        boots.append(D(smp))
        A = [x['ex_t3'] for x in smp if x['grade'] == 'A']
        bootA.append(np.mean(A) if A else np.nan)
    perm = []
    for _ in range(N_PERM):
        smp = []
        for r in rids:
            gs = [x['grade'] for x in by_race[r]]
            rng.shuffle(gs)
            smp += [dict(x, grade=gg) for x, gg in zip(by_race[r], gs)]
        perm.append(D(smp))
    q = lambda v: [round(float(np.nanpercentile(v, 2.5)) * 100, 2), round(float(np.nanpercentile(v, 97.5)) * 100, 2)]
    pm, ps = float(np.nanmean(perm)), float(np.nanstd(perm))
    exA = res['by_grade']['A'].get('ex_t3')
    c1 = bool(d > 0 and q(boots)[0] > 0)
    c2 = bool(d > np.nanpercentile(perm, 95))
    c3 = bool(exA is not None and q(bootA)[0] > 0)
    res['D'] = dict(value=round(float(d) * 100, 2), ci=q(boots), perm_mean=round(pm * 100, 3), perm_sd=round(ps * 100, 3),
                    perm_p95=round(float(np.nanpercentile(perm, 95)) * 100, 2),
                    perm_centered=bool(abs(pm) < 2 * ps / np.sqrt(N_PERM)))
    res['A_ex_t3_ci'] = q(bootA)
    res['criteria'] = {'1_D_ci_above_0': c1, '2_beats_permutation_p95': c2, '3_A_beats_market': c3}
    res['verdict'] = '信号あり' if (c1 and c2 and c3) else '信号なし'
    out = os.path.join(ROOT, 'websig', f'grades_{a.source}_result.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
