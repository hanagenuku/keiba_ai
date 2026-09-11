"""ユーザーの問い(2026-09-11): 勝率予測 18%以上 かつ EV(=勝率×オッズ) > 1.00 の回収率。

🔴 これは事前登録した検証ではない。ユーザーが指定した1つのセルの実測。
   採用の根拠にはしない。対照群と、閾値を振ったときの形まで併記する。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
BASE = os.path.abspath('.'); SCR = os.path.dirname(BASE)
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
ab_src = open(f'{BASE}/calib/ability.py').read().rsplit("\nif __name__ ==", 1)[0]
ab = type(sys)('ab'); ab.__dict__['__file__'] = f'{BASE}/calib/ability.py'
exec(compile(ab_src, 'ability.py', 'exec'), ab.__dict__)
gather, ARMS1 = ab.gather, ab.ARMS1

cal, y, rid, hn, pop = gather(['W1'], 'y1')
c = sqlite3.connect(f'{SCR}/keiba.db')
rp = pd.read_sql('SELECT race_id, horse_num, tansho_odds FROM race_predictions '
                 'WHERE tansho_odds >= 1.0', c); c.close()
c2 = sqlite3.connect(f'{BASE}/data/history.db')
pay = pd.read_sql('SELECT race_id, horse_num, tansho_payout FROM horse_history', c2); c2.close()
key = pd.MultiIndex.from_arrays([rid, hn])
odds = rp.set_index(['race_id','horse_num'])['tansho_odds'].reindex(key).to_numpy(float)
pyt = pay.set_index(['race_id','horse_num'])['tansho_payout'].reindex(key).to_numpy(float)
m = np.isfinite(odds) & (odds >= 1.0)
m &= ~((y == 1) & (~np.isfinite(pyt) | (pyt <= 0)))
O, Y, PY, R = odds[m], y[m], pyt[m], rid[m]
DAY = pd.Series(R).str.slice(0, 8).to_numpy()
days = np.unique(DAY)
rng = np.random.default_rng(42)

def roi(sel):
    return PY[sel & (Y == 1)].sum() / max(sel.sum(), 1)

def drop3(sel):
    w = np.sort(PY[sel & (Y == 1)])[::-1]
    return (w[3:].sum() if len(w) > 3 else 0.0) / max(sel.sum(), 1)

def ci(sel):
    """開催日ブロック bootstrap の95%CI"""
    if sel.sum() < 30: return (float('nan'), float('nan'))
    out = []
    for _ in range(2000):
        d = rng.choice(days, len(days))
        mm = np.concatenate([np.where(sel & (DAY == x))[0] for x in d])
        if len(mm) == 0: continue
        out.append(PY[mm][Y[mm] == 1].sum() / len(mm))
    return tuple(np.percentile(out, [2.5, 97.5]))

def half(sel):
    mid = days[len(days)//2]
    a, b = sel & (DAY < mid), sel & (DAY >= mid)
    return roi(a), roi(b)

def plus_days(sel):
    n = k = 0
    for d in days:
        s = sel & (DAY == d)
        if s.sum() == 0: continue
        n += 1; k += (roi(s) >= 100)
    return k, n

print(f'{"="*104}\n■ 母集団: 単勝・実オッズ W1(2026-07-04〜09-06) {m.sum():,}頭 / '
      f'{len(np.unique(R)):,}レース / {len(days)}開催日')
print(f'  買う判定=朝オッズ / 払戻=確定配当。全部買う 回収 {roi(np.ones(len(Y),bool)):.1f}% '
      f'（的中 {Y.mean()*100:.2f}%）\n{"="*104}')

for a in ['E1 Isotonic', '参考S 現行画面', '参考M 市場']:
    p = cal[a][m]; ev = p * O
    print(f'\n■ {a}   予測勝率の平均 {p.mean()*100:.2f}% / 18%以上は {int((p>=.18).sum()):,}頭')
    print(f'  {"条件":<34}{"N":>7}{"的中率":>9}{"回収率":>9}{"3本抜":>8}'
          f'{"前半":>8}{"後半":>8}{"+の日":>8}{"95%CI":>18}')
    rows = [('★ 勝率>=18% かつ EV>1.00', (p >= .18) & (ev > 1.0)),
            ('【対照】勝率>=18% かつ EV<=1.00', (p >= .18) & (ev <= 1.0)),
            ('【対照】勝率>=18%（EV条件なし）', p >= .18),
            ('【対照】EV>1.00（勝率条件なし）', ev > 1.0),
            ('【対照】全部買う', np.ones(len(Y), bool))]
    for lab, s in rows:
        if s.sum() < 30:
            print(f'  {lab:<34}{int(s.sum()):>7}{"N不足":>9}'); continue
        lo, hi = ci(s); h1, h2 = half(s); k, n = plus_days(s)
        print(f'  {lab:<34}{int(s.sum()):>7}{Y[s].mean()*100:>8.2f}%{roi(s):>8.1f}%'
              f'{drop3(s):>7.1f}%{h1:>7.1f}%{h2:>7.1f}%{k:>5}/{n:<3}'
              f'{f"[{lo:.0f}, {hi:.0f}]":>18}')

print(f'\n{"="*104}\n■ 閾値を振ったときの形（E1 Isotonic・回収率%／括弧内N）'
      f'\n  🔴 18%が特別かどうかを見るためのもの。最良セルを選ばない\n{"="*104}')
p = cal['E1 Isotonic'][m]; ev = p * O
print(f'  {"勝率の下限":<12}' + ''.join(f'{f"EV>{t:.2f}":>18}' for t in (1.0, 1.2, 1.5)))
for fl in (0.0, .08, .10, .12, .15, .18, .20, .25, .30):
    line = f'  {f">={fl*100:.0f}%":<12}'
    for t in (1.0, 1.2, 1.5):
        s = (p >= fl) & (ev > t)
        line += f'{"N不足":>18}' if s.sum() < 30 else f'{f"{roi(s):.1f}% ({int(s.sum())})":>18}'
    print(line)
