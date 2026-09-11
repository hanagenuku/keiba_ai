"""推奨ゲートは頭数を見ていない。少頭数を機械的に選んでいないかを測る。

🔴 JRA: 8頭以上→3着まで / 5〜7頭→2着まで / 4頭以下→複勝の発売なし。
   ゲートは top3_prob（3着内確率）を頭数と無関係に使っている。
   7頭立てでは「3着内」に入っても**複勝は付かない**ことがある。
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/keiba_ai'); sys.path.insert(0, BASE)
from src.models.predict import softmax_probs
from src.features.engine import calc_harville_probs
GATE = 0.55
Q = pickle.load(open(f'{BASE}/q/predq.pkl', 'rb'))
META = pickle.load(open(f'{BASE}/cut_data.pkl', 'rb'))['meta']
B = Q['OOS']; p = 1/(1+np.exp(-B['raw'])); md = META.iloc[B['idx']]
h = sqlite3.connect(f'{BASE}/data/history.db')
pl = pd.read_sql('SELECT race_id, horse_num, place, fukusho_payout FROM horse_history', h); h.close()
PL = pl.set_index(['race_id', 'horse_num'])
d = pd.DataFrame({'rid': B['rid'], 'hn': md['horse_num'].to_numpy(), 'p': p,
                  'y3': B['y3'], 'field': B['field'], 'cls': md['race_class'].fillna('').to_numpy()})
rows = []
for rid, g in d.groupby('rid', sort=False):
    wp = softmax_probs([v*10 for v in g['p']], temperature=3.5)
    t3 = [b for _, b in calc_harville_probs(wp)]
    i = int(np.argmax(g['p'].to_numpy()))
    rows.append(dict(rid=rid, cls=g['cls'].iloc[0], n=int(len(g)),
                     ax_p=t3[i], ax_y3=int(g['y3'].iloc[i]), ax_hn=int(g['hn'].iloc[i])))
R = pd.DataFrame(rows); R['day'] = [r[:8] for r in R['rid']]
k = pd.MultiIndex.from_arrays([R['rid'], R['ax_hn']])
R['pay'] = PL['fukusho_payout'].reindex(k).to_numpy(float)
R['place'] = PL['place'].reindex(k).to_numpy(float)
# 🔴 3着内なのに配当欠損は評価不能（0円として混ぜない）
R['ok'] = ~((R['ax_y3'] == 1) & (~np.isfinite(R['pay']) | (R['pay'] <= 0)))
# 実際に複勝が付く着順（JRAルール）
R['payable'] = np.where(R['n'] >= 8, R['place'] <= 3,
                np.where(R['n'] >= 5, R['place'] <= 2, False))

def roi(g):
    m = g[g['ok']]
    return m.loc[m['ax_y3'] == 1, 'pay'].sum() / len(m) if len(m) else np.nan

W = 100
print(f'{"="*W}\n■ 推奨ゲート(軸のtop3_prob>={GATE})は頭数を見ていない — 少頭数を機械的に選んでいないか')
print(f'  最終OOS {len(R):,}レース（2026-01-04〜09-06）\n{"="*W}')
pas = R[R['ax_p'] >= GATE].copy()
top = pas.sort_values('ax_p', ascending=False).groupby('day').head(6)
print(f'\n■ ① 頭数の分布')
print(f'  {"":<14}{"全レース":>10}{"ゲート通過":>12}{"推奨6本に選抜":>14}')
for lab, lo, hi in [('〜7頭(複勝2着まで)', 0, 7), ('8〜11頭', 8, 11), ('12〜15頭', 12, 15), ('16頭〜', 16, 99)]:
    f = lambda g: (g['n'].between(lo, hi)).mean()*100
    print(f'  {lab:<14}{f(R):>9.1f}%{f(pas):>11.1f}%{f(top):>13.1f}%')
print(f'  {"平均頭数":<14}{R["n"].mean():>9.1f} {pas["n"].mean():>10.1f} {top["n"].mean():>12.1f}')

print(f'\n■ ② 頭数帯別 — ゲートは何を測っているか')
print(f'  {"頭数":<16}{"R数":>6}{"軸予測":>8}{"軸3着内":>9}{"ズレ":>8}{"通過率":>8}'
      f'{"複勝が付いた率":>15}{"複勝回収":>10}')
for lab, lo, hi in [('5〜7頭', 5, 7), ('8〜9頭', 8, 9), ('10〜11頭', 10, 11),
                    ('12〜13頭', 12, 13), ('14〜15頭', 14, 15), ('16頭〜', 16, 99)]:
    g = R[R['n'].between(lo, hi)]
    if len(g) < 30: print(f'  {lab:<16}{len(g):>6}  N不足'); continue
    pg = g[g['ax_p'] >= GATE]
    print(f'  {lab:<16}{len(g):>6,}{g["ax_p"].mean()*100:>7.1f}%{g["ax_y3"].mean()*100:>8.1f}%'
          f'{(g["ax_y3"].mean()-g["ax_p"].mean())*100:>+7.1f}{len(pg)/len(g)*100:>7.1f}%'
          f'{g["payable"].mean()*100:>14.1f}%{roi(g):>9.1f}%')

print(f'\n■ ③ 🔴 「3着内」と「複勝が付く」がズレる帯（5〜7頭）')
s = R[R['n'].between(5, 7)]
if len(s):
    gap = (s['ax_y3'].mean() - s['payable'].mean())*100
    print(f'  5〜7頭 {len(s):,}レース: 軸の3着内 {s["ax_y3"].mean()*100:.1f}% だが'
          f' 実際に複勝が付いたのは {s["payable"].mean()*100:.1f}% ＝ **{gap:.1f}pt の差**')
    print(f'  ゲートは top3_prob を頭数と無関係に使うので、この {gap:.1f}pt を知らない')
print(f'\n■ ④ 推奨6本に選ばれたレースの成績を頭数で割る')
print(f'  {"頭数":<16}{"本数":>6}{"軸3着内":>9}{"複勝が付いた率":>15}{"複勝回収":>10}')
for lab, lo, hi in [('〜9頭', 0, 9), ('10〜13頭', 10, 13), ('14頭〜', 14, 99)]:
    g = top[top['n'].between(lo, hi)]
    if len(g) < 20: print(f'  {lab:<16}{len(g):>6}  N不足'); continue
    print(f'  {lab:<16}{len(g):>6}{g["ax_y3"].mean()*100:>8.1f}%{g["payable"].mean()*100:>14.1f}%{roi(g):>9.1f}%')
print(f'  {"（全444本）":<16}{len(top):>6}{top["ax_y3"].mean()*100:>8.1f}%'
      f'{top["payable"].mean()*100:>14.1f}%{roi(top):>9.1f}%')
