"""PLレーティング θ のハイパーパラメータを振るための、高速な再実装（検証用ツール）。

⚠ **本番コードではない。** 本番は `src/features/pl_rating.py`。
   これは θ を別パラメータで作り直して比較するための道具で、
   1配置あたり約1.7秒で 164,328頭ぶんの「各レース開始前のθ」を返す。

## 本番との一致は検証済み（2026-09-08）

`build_training_data` の θ ループを厳密に再現したものと全164,328行を突合し、
**f_pl_rating / f_pl_rating_n / rl_f_pl_rating / rl_f_pl_rating_z /
rl_f_pl_rating_rank の5列すべてで最大差 1.2e-15**（浮動小数点誤差のみ）。

## 本番との違いは2点だけ（どちらも厳密に同値）

1. 時間減衰を「全馬に毎日掛ける」代わりに**大域スケール factor** で持つ（高速化）
2. 各レース時点の特徴量を（更新前に）全部記録して返す

## 🔴 ここで2回間違えたので、直した箇所にコメントを残す

- **SQL は `ORDER BY date, race_id, horse_num`**。`ORDER BY place` にすると
  θ同値の馬（13.8%）で安定ソートが着順順を保ち、**順位が着順を漏らす**
  （2026-08-08 の「AUC +0.037」事故と同型。単勝回収率162.8%で発覚）
- **更新に渡す着順リストは `key=lambda x: x[0]`（着順のみ）でソートする**。
  `sorted([(着順, 馬名)])` にすると**同着294組**で並びが変わり、
  θのズレが対戦相手の勾配を通じて全行の34%に伝播する

## 使う前に必ず通す関門

- レース内でθをシャッフルした対照群で AUC ≈ 0.500 になること
- **θ同値行で corr(順位, 着順) が本番の健全値 +0.17 付近であること**（+0.87 ならリーク）

詳細と測定結果は `pl/CRITERIA.md` / `pl/RESULTS.md`。
"""
import math, sqlite3
from collections import defaultdict

MIN_FIELD = 3

def build(db_path, LR=0.30, EXP_DAMP=0.05, PRIOR_SHRINK=0.02, DECAY_DAYS=400.0,
          key='name'):
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT race_id, date, horse_name, place, surface, distance, horse_num
        FROM horse_history
        WHERE horse_name != '' AND date IS NOT NULL AND date != ''
        ORDER BY date, race_id, horse_num
    """).fetchall()
    conn.close()

    raw = defaultdict(float)      # theta = raw * scale
    nrun = defaultdict(int)
    scale = 1.0
    last_date = None
    out = []                      # 記録した特徴量

    def theta(n): return raw[n] * scale

    def apply_race(names):
        nonlocal scale
        k = len(names)
        th = [theta(n) for n in names]
        m = max(th)
        e = [math.exp(t - m) for t in th]
        tail = [0.0]*k; acc = 0.0
        for i in range(k-1, -1, -1):
            acc += e[i]; tail[i] = acc
        grad = [0.0]*k
        for pos in range(k-1):
            t = tail[pos]
            if t <= 0: continue
            grad[pos] += 1.0
            for j in range(pos, k):
                grad[j] -= e[j]/t
        for i, n in enumerate(names):
            nrun[n] += 1
            step = LR / (1.0 + EXP_DAMP*nrun[n])
            v = (theta(n) + step*grad[i]) * (1.0 - PRIOR_SHRINK)
            raw[n] = v / scale

    def days_between(a, b):
        from datetime import date
        def p(s):
            s = str(s)[:10].replace('/', '-'); y, mm, d = s.split('-')
            return date(int(y), int(mm), int(d))
        try: return (p(b) - p(a)).days
        except Exception: return 0

    def advance(day, pending):
        nonlocal scale, last_date, raw
        for names in pending: apply_race(names)
        if last_date and DECAY_DAYS:
            dt = days_between(last_date, day)
            if dt > 0:
                scale *= math.exp(-dt/DECAY_DAYS)
                if scale < 1e-100:                      # 再正規化（厳密に同値）
                    for n in list(raw): raw[n] *= scale
                    scale = 1.0
        last_date = day

    def record(rid, day, members):
        """members: [(horse_name, place, surface, dist)] — 更新前のθで特徴量を作る"""
        k = len(members)
        th = [theta(n) for n, _, _, _ in members]
        mean = sum(th)/k
        var = sum((t-mean)**2 for t in th)/k
        sd = math.sqrt(var)
        order = sorted(range(k), key=lambda i: -th[i])
        rank = [0]*k
        for r, i in enumerate(order, 1): rank[i] = r
        for i, (nm, pl, sf, dist) in enumerate(members):
            out.append((day, rid, nm, th[i], nrun[nm], th[i]-mean,
                        (th[i]-mean)/sd if sd > 1e-6 else 0.0, rank[i], k, pl, sf, dist))

    cur = None
    by_race = {}
    for row in rows:
        d = str(row['date'])[:10]
        if cur is not None and d != cur:
            pend = []
            for rid, mem in by_race.items():
                record(rid, cur, mem)
                fin = sorted([(p, n) for n, p, _, _ in mem if p and 0 < p < 99], key=lambda x: x[0])
                if len(fin) >= MIN_FIELD: pend.append([n for _, n in fin])
            advance(cur, pend)
            by_race = {}
        cur = d
        p = row['place']
        p = int(p) if p is not None else 99
        by_race.setdefault(row['race_id'], []).append(
            (row['horse_name'], p, row['surface'] or '', row['distance'] or 0))
    if cur is not None:
        pend = []
        for rid, mem in by_race.items():
            record(rid, cur, mem)
            fin = sorted([(p, n) for n, p, _, _ in mem if p and 0 < p < 99], key=lambda x: x[0])
            if len(fin) >= MIN_FIELD: pend.append([n for _, n in fin])
        advance(cur, pend)
    return out
