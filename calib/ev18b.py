"""★セルと対照群が 79.1% で一致した件の検算（North Star #8: 出来すぎた数字は疑う）"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
BASE = os.path.abspath('.'); SCR = os.path.dirname(BASE)
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
ab_src = open(f'{BASE}/calib/ability.py').read().rsplit("\nif __name__ ==", 1)[0]
ab = type(sys)('ab'); ab.__dict__['__file__'] = f'{BASE}/calib/ability.py'
exec(compile(ab_src, 'ability.py', 'exec'), ab.__dict__)
cal, y, rid, hn, pop = ab.gather(['W1'], 'y1')
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
O, Y, PY = odds[m], y[m], pyt[m]
p = cal['E1 Isotonic'][m]; ev = p * O
for lab, s in [('EV>1.00', (p>=.18)&(ev>1.0)), ('EV<=1.00', (p>=.18)&(ev<=1.0)),
               ('全体(>=18%)', p>=.18)]:
    won = s & (Y==1)
    print(f'  {lab:<14} N={int(s.sum()):>5}  的中={int(won.sum()):>4}  '
          f'払戻合計=¥{PY[won].sum():>10,.0f}  回収={PY[won].sum()/s.sum():.4f}%  '
          f'オッズ中央値={np.median(O[s]):.1f}倍')
print()
print(f'  勝率>=18% の EV>1.00 側は odds > 1.2/0.18 = {1.0/0.18:.2f}倍 より上の馬')
print(f'  → 「勝率18%以上」の中を**オッズだけで**2つに割っている（pがほぼ一定なので）')
print(f'  p の中央値: EV>1.00 側 {np.median(p[(p>=.18)&(ev>1.0)]):.4f} / '
      f'EV<=1.00 側 {np.median(p[(p>=.18)&(ev<=1.0)]):.4f}')
print(f'  corr(EV, オッズ) = {np.corrcoef(ev[p>=.18], O[p>=.18])[0,1]:+.4f}  '
      f'corr(EV, 勝率) = {np.corrcoef(ev[p>=.18], p[p>=.18])[0,1]:+.4f}')

print()
pm = cal['参考M 市場'][m]
ok = np.isfinite(pm)
s = (p >= .18) & (ev > 1.0)
print('■ ★セルの正体（人気での内訳）')
popv = pop[m]
for lo, hi, lab in [(1,1,'1番人気'),(2,3,'2-3番人気'),(4,5,'4-5番人気'),(6,9,'6-9番人気'),(10,99,'10番人気〜')]:
    b = (popv >= lo) & (popv <= hi)
    print(f'  {lab:<10} 全体 {int(b.sum()):>5}頭 / ★セル {int((s&b).sum()):>4}頭 '
          f'({(s&b).sum()/max(s.sum(),1)*100:>5.1f}%)')
print(f'\n■ AIの「勝率>=18%」と市場の「示唆>=18%」の重なり')
a18, m18 = (p >= .18), (ok & (pm >= .18))
print(f'  AIのみ {int((a18&~m18).sum()):>4}頭 回収 {PY[(a18&~m18)&(Y==1)].sum()/max((a18&~m18).sum(),1):>5.1f}%')
print(f'  両方   {int((a18&m18).sum()):>4}頭 回収 {PY[(a18&m18)&(Y==1)].sum()/max((a18&m18).sum(),1):>5.1f}%')
print(f'  市場のみ {int((~a18&m18).sum()):>4}頭 回収 {PY[(~a18&m18)&(Y==1)].sum()/max((~a18&m18).sum(),1):>5.1f}%')
