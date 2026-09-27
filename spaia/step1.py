"""SPAIA指南書「順番待ちスコア」の天井測定（事前登録: spaia/CRITERIA.md）。

本番コード・モデル・買い目・表示には一切触れない。history.db を読むだけ。
"""
import sqlite3, math, random, sys
from collections import defaultdict

DB = 'data/history.db'
CLASSRANK = {'新馬': 0, '未勝利': 1, '1勝': 2, '1勝クラス': 2, '2勝': 3, '2勝クラス': 3,
             '3勝': 4, '3勝クラス': 4, 'OP': 5, 'オープン': 5}
SMALL = {'中山', '福島', '小倉', '函館', '札幌'}

TERMS = [  # (名前, 点数)
    ('直近3走で3着内2回以上', +2), ('同クラスで好走経験', +2), ('同距離で好走', +1),
    ('同コースで好走', +1), ('前走が近い条件', +1), ('着順が安定', +1),
    ('前走より条件好転', +1), ('前走6着以下', -2), ('10週以上休養', -2),
    ('小回りで後方脚質', -1), ('大幅な条件悪化', -2),
]

def _days(a, b):
    from datetime import date
    ya, ma, da = (int(x) for x in a.split('-')); yb, mb, db_ = (int(x) for x in b.split('-'))
    return (date(ya, ma, da) - date(yb, mb, db_)).days

def load():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    nf = {r['race_id']: r['num_finishers'] for r in c.execute(
        "SELECT race_id, num_finishers FROM race_history")}
    rows = [dict(r) for r in c.execute(
        "SELECT race_id,date,racecourse,distance,surface,place,horse_name,popularity,"
        "fukusho_payout,corner_all,class_grade,field_size "
        "FROM horse_history ORDER BY horse_name, date, race_id")]
    for r in rows:
        r['_nf'] = r['field_size'] if (r['field_size'] or 0) > 1 else nf.get(r['race_id'])
    return rows

def terms_for(cur, past):
    """past は cur より前の日付の走り（新しい順）。文書の11項目を機械的に判定。"""
    t = {}
    last3 = past[:3]
    pl3 = [p['place'] for p in last3 if p['place']]
    t['直近3走で3着内2回以上'] = len(pl3) >= 2 and sum(1 for p in pl3 if p <= 3) >= 2
    cg = cur['class_grade']
    same_cls = [p for p in past if p['class_grade'] == cg and cg]
    t['同クラスで好走経験'] = len(same_cls) >= 2 and any(p['place'] and p['place'] <= 5 for p in same_cls)
    t['同距離で好走'] = any(p['distance'] == cur['distance'] and p['place'] and p['place'] <= 3 for p in past)
    t['同コースで好走'] = any(p['racecourse'] == cur['racecourse'] and p['surface'] == cur['surface']
                        and p['distance'] == cur['distance'] and p['place'] and p['place'] <= 3 for p in past)
    prev = past[0]
    t['前走が近い条件'] = (prev['surface'] == cur['surface']
                    and abs((prev['distance'] or 0) - (cur['distance'] or 0)) <= 200)
    if len(pl3) == 3:
        m = sum(pl3) / 3.0
        t['着順が安定'] = math.sqrt(sum((p - m) ** 2 for p in pl3) / 3.0) <= 2.0
    else:
        t['着順が安定'] = False
    pr, cr = CLASSRANK.get(prev['class_grade']), CLASSRANK.get(cg)
    t['前走より条件好転'] = pr is not None and cr is not None and pr > cr
    t['前走6着以下'] = bool(prev['place']) and prev['place'] >= 6
    t['10週以上休養'] = _days(cur['date'], prev['date']) >= 70
    back = False
    if cur['racecourse'] in SMALL and prev['corner_all'] and prev['_nf']:
        try:
            c3 = int(str(prev['corner_all']).split('-')[0])
            back = (c3 / float(prev['_nf'])) >= 0.7
        except Exception:
            back = False
    t['小回りで後方脚質'] = back
    worse = (pr is not None and cr is not None and pr < cr) or \
            abs((prev['distance'] or 0) - (cur['distance'] or 0)) >= 400
    t['大幅な条件悪化'] = worse
    return t

def build():
    rows = load()
    by_horse = defaultdict(list)
    for r in rows:
        by_horse[r['horse_name']].append(r)
    out = []
    for name, rs in by_horse.items():
        for i, cur in enumerate(rs):
            if i == 0:
                continue                              # 過去走なし → 対象外
            if not cur['place'] or cur['place'] <= 0:
                continue
            if cur['place'] <= 3 and not (cur['fukusho_payout'] or 0) > 0:
                continue                             # 3着内なのに配当欠損 → 評価不能で除外
            past = [p for p in rs[:i] if p['date'] < cur['date']]
            if not past:
                continue
            past = list(reversed(past))
            t = terms_for(cur, past)
            S = sum(pt for nm, pt in TERMS if t[nm])
            out.append({
                'race_id': cur['race_id'], 'date': cur['date'], 'S': S, 'terms': t,
                'hit': 1 if cur['place'] <= 3 else 0,
                'pay': (cur['fukusho_payout'] or 0) if cur['place'] <= 3 else 0,
                'pop': cur['popularity'] if (cur['popularity'] or 0) and 1 <= cur['popularity'] <= 98 else None,
                'nf': cur['_nf'],
            })
    return out

def roi(rs):
    return (sum(r['pay'] for r in rs) / (100.0 * len(rs)) * 100) if rs else float('nan')

def hitrate(rs):
    return (sum(r['hit'] for r in rs) / len(rs) * 100) if rs else float('nan')

def auc(rs, key):
    xs = [(key(r), r['hit']) for r in rs if key(r) is not None]
    pos = [x for x, y in xs if y == 1]; neg = [x for x, y in xs if y == 0]
    if not pos or not neg:
        return float('nan')
    allv = sorted(x for x, _ in xs)
    # rank-based (tie-aware) Mann-Whitney
    import bisect
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2.0 + 1
        rank[allv[i]] = r
        i = j + 1
    sr = sum(rank[x] for x in pos)
    return (sr - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

def band(S):
    return 'S<=3' if S <= 3 else ('S4-6' if S <= 6 else 'S>=7')

def main():
    data = build()
    print(f"■ 母集団: {len(data)}行（過去走1走以上・配当欠損の3着内を除外）")
    print(f"  全体 複勝回収率 {roi(data):.1f}% / 3着内率 {hitrate(data):.1f}%")
    print()
    print("■ S の分布")
    dist = defaultdict(int)
    for r in data:
        dist[r['S']] += 1
    for s in sorted(dist):
        print(f"  S={s:+3d}  {dist[s]:6d}")
    print()

    # ---- 中止条件: S単独の複勝AUC
    a_s = auc(data, lambda r: r['S'])
    a_m = auc(data, lambda r: (-r['pop']) if r['pop'] else None)
    print(f"■ 中止条件  S単独 複勝AUC = {a_s:.4f}   （参考）市場単独(-人気) = {a_m:.4f}")
    if a_s < 0.50:
        print("  🔴 0.50 を下回った → 事前登録どおり中止。以降の帯分析は採用根拠にしない")
    print()

    # ---- Primary 1-3
    print("■ 基準1-3  帯別（全期間）")
    print(f"  {'帯':6s} {'N':>7s} {'3着内率':>8s} {'複勝回収':>9s}")
    bands = {}
    for b in ('S<=3', 'S4-6', 'S>=7'):
        rs = [r for r in data if band(r['S']) == b]
        bands[b] = rs
        print(f"  {b:6s} {len(rs):7d} {hitrate(rs):7.1f}% {roi(rs):8.1f}%")
    base = roi(data)
    d7 = roi(bands['S>=7']) - base
    mono = roi(bands['S<=3']) < roi(bands['S4-6']) < roi(bands['S>=7'])
    print(f"  基準1 単調性 {'✅' if mono else '❌'}")
    print(f"  基準2 効果量 S>=7 − 全体 = {d7:+.1f}pt （合格線 +5.0pt）{'✅' if d7 >= 5.0 else '❌'}")
    print(f"  基準3 N      {len(bands['S>=7'])} （合格線 2,000）{'✅' if len(bands['S>=7']) >= 2000 else '❌'}")
    print()

    # ---- 基準4 市場比較ゲート
    print("■ 基準4  市場人気で統制（3着内率の差 S>=7 − S<=3）")
    for label, keyf in (('人気帯(1-3/4-7/8+)', lambda r: 0 if r['pop'] <= 3 else (1 if r['pop'] <= 7 else 2)),
                        ('レース内%3分位', None)):
        if keyf is None:
            pc = [(r, r['pop'] / r['nf']) for r in data if r['pop'] and r['nf']]
            pcs = sorted(p for _, p in pc)
            q1, q2 = pcs[len(pcs) // 3], pcs[2 * len(pcs) // 3]
            groups = defaultdict(list)
            for r, p in pc:
                groups[0 if p <= q1 else (1 if p <= q2 else 2)].append(r)
        else:
            groups = defaultdict(list)
            for r in data:
                if r['pop']:
                    groups[keyf(r)].append(r)
        diffs = []
        print(f"  [{label}]")
        for g in sorted(groups):
            rs = groups[g]
            hi = [r for r in rs if band(r['S']) == 'S>=7']
            lo = [r for r in rs if band(r['S']) == 'S<=3']
            d = hitrate(hi) - hitrate(lo)
            diffs.append(d)
            print(f"    分位{g}: N={len(rs):6d}  S>=7 {hitrate(hi):5.1f}%(n={len(hi):5d}) "
                  f"S<=3 {hitrate(lo):5.1f}%(n={len(lo):5d})  差 {d:+6.2f}pt")
        same = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        print(f"    全分位同符号 {'✅' if same else '❌'} / 最小 {min(diffs, key=abs):+.2f}pt "
              f"（合格線 +2.0pt）{'✅' if same and min(diffs) >= 2.0 else '❌'}")
    print()

    # ---- 三者比較
    print("■ 標準ゲート: 三者比較（複勝AUC）")
    print(f"  市場単独        {a_m:.4f}")
    print(f"  S単独           {a_s:.4f}")
    mm = [r for r in data if r['pop'] and r['nf']]
    a_ms = auc(mm, lambda r: -(r['pop'] / r['nf']) * 10 + r['S'] * 0.0)  # 市場のみ（正規化）
    a_both = auc(mm, lambda r: -(r['pop'] / r['nf']) * 10 + r['S'] * 1.0)
    print(f"  市場(正規化)単独 {a_ms:.4f}")
    print(f"  市場 + S        {a_both:.4f}   差 {a_both - a_ms:+.4f}")
    print()

    # ---- 方向別
    print("■ 方向別（S のレース内順位 − 人気順位。＋なら S が市場より強気）")
    byrace = defaultdict(list)
    for r in data:
        byrace[r['race_id']].append(r)
    for r in data:
        r['gap'] = None
    for rid, rs in byrace.items():
        rs2 = [r for r in rs if r['pop']]
        if len(rs2) < 4:
            continue
        srank = {id(r): i + 1 for i, r in enumerate(sorted(rs2, key=lambda x: -x['S']))}
        prank = {id(r): i + 1 for i, r in enumerate(sorted(rs2, key=lambda x: x['pop']))}
        for r in rs2:
            r['gap'] = prank[id(r)] - srank[id(r)]
    for lab, f in (('S強気 +3以上', lambda g: g >= 3), ('一致 -2〜+2', lambda g: -2 <= g <= 2),
                   ('S弱気 -3以下', lambda g: g <= -3)):
        rs = [r for r in data if r['gap'] is not None and f(r['gap'])]
        print(f"  {lab:12s} N={len(rs):6d}  3着内 {hitrate(rs):5.1f}%  複勝回収 {roi(rs):6.1f}%")
    print()

    # ---- 基準5 頑健性
    print("■ 基準5  前後半")
    for lab, f in (('前半 2023-2024', lambda d: d < '2025-01-01'), ('後半 2025-2026', lambda d: d >= '2025-01-01')):
        sub = [r for r in data if f(r['date'])]
        b = {x: [r for r in sub if band(r['S']) == x] for x in ('S<=3', 'S4-6', 'S>=7')}
        mo = roi(b['S<=3']) < roi(b['S4-6']) < roi(b['S>=7'])
        dd = roi(b['S>=7']) - roi(sub)
        print(f"  {lab}: N={len(sub):6d} 全体{roi(sub):6.1f}%  "
              f"S<=3 {roi(b['S<=3']):6.1f}% / S4-6 {roi(b['S4-6']):6.1f}% / S>=7 {roi(b['S>=7']):6.1f}%"
              f"  単調{'✅' if mo else '❌'} 効果量{dd:+.1f}pt{'✅' if dd >= 5.0 else '❌'}")
    print()

    # ---- 基準6 CI（日ブロックbootstrap）
    rs7 = bands['S>=7']
    bydate = defaultdict(list)
    for r in rs7:
        bydate[r['date']].append(r)
    days = list(bydate)
    rnd = random.Random(20260927)
    boots = []
    for _ in range(2000):
        pick = [bydate[rnd.choice(days)] for _ in days]
        flat = [x for g in pick for x in g]
        boots.append(roi(flat))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    print(f"■ 基準6  S>=7 複勝回収 {roi(rs7):.1f}%  95%CI(日ブロック{len(days)}日) [{lo:.1f}, {hi:.1f}]"
          f"  {'✅' if lo > 100 else '❌'}")
    print()

    # ---- 対照群A（並べ替え分布）
    print("■ 対照群A  レース内 S シャッフル 200回")
    obs = roi(bands['S>=7']) - base
    rnd = random.Random(1)
    ds = []
    for _ in range(200):
        hi = []
        for rid, rs in byrace.items():
            ss = [r['S'] for r in rs]
            rnd.shuffle(ss)
            for r, s in zip(rs, ss):
                if band(s) == 'S>=7':
                    hi.append(r)
        ds.append(roi(hi) - base)
    m = sum(ds) / len(ds)
    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / (len(ds) - 1))
    line = 2 * sd / math.sqrt(len(ds))
    p = (sum(1 for d in ds if d >= obs) + 1) / (len(ds) + 1)
    print(f"  帰無 mean {m:+.3f}pt  std {sd:.3f}  合格線 |mean|<{line:.3f}  "
          f"{'✅' if abs(m) < line else '❌'}")
    print(f"  実測 Δ {obs:+.2f}pt  並べ替え P = {p:.4f}")
    print()

    # ---- 対照群B（注入の復元）
    print("■ 対照群B  S>=7 の3着内率を +5.0pt 注入して復元できるか")
    before = hitrate(bands['S>=7']) - hitrate(bands['S<=3'])
    rnd = random.Random(2)
    zeros = [r for r in bands['S>=7'] if r['hit'] == 0]
    need = int(round(0.05 * len(bands['S>=7'])))
    flip = set(id(r) for r in rnd.sample(zeros, need))
    hi2 = [1 if (r['hit'] == 1 or id(r) in flip) else 0 for r in bands['S>=7']]
    after = sum(hi2) / len(hi2) * 100 - hitrate(bands['S<=3'])
    print(f"  注入前 帯差 {before:+.2f}pt → 注入後 {after:+.2f}pt  "
          f"差 {after - before:+.3f}pt（期待 +5.000）"
          f"{'✅' if abs((after - before) - 5.0) < 0.2 else '❌'}")
    print()

    # ---- Secondary 符号検査
    print("■ Secondary  11項目の符号検査（採用判断には使わない・多重比較）")
    print(f"  {'項目':22s} {'文書':>4s} {'成立N':>7s} {'成立時回収':>10s} {'不成立回収':>10s} {'差':>8s} 符号")
    for nm, pt in TERMS:
        on = [r for r in data if r['terms'][nm]]
        off = [r for r in data if not r['terms'][nm]]
        d = roi(on) - roi(off)
        agree = '一致' if (pt > 0) == (d > 0) else '🔴逆'
        print(f"  {nm:22s} {pt:+4d} {len(on):7d} {roi(on):9.1f}% {roi(off):9.1f}% {d:+7.1f}pt {agree}")

if __name__ == '__main__':
    main()
