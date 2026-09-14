"""L-2: 決定打。「逃げると予測できた馬」にも市場残差は残っているか。

  実際に逃げた馬の残差 +15pt のうち、
    (a) 事前に予測できる部分  → 市場も予測できるので既に価格に入っているはず
    (b) 予測できない部分（当日ハナを切れた）→ 事前には取れない
  どちらが残差を作っているかを分ける。
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
d=pd.read_pickle(f'{B}/lead/d.pkl')
d['y3']=(d.place<=3).astype(int)
cut='2025-07-01'
tr,te=d[d.date<cut].copy(),d[d.date>=cut].copy()

# 市場残差の基準表は**学習期だけ**で作る（検定期に当てはめる）
tr['fsb']=pd.cut(tr.fs,[0,9,13,18],labels=['a','b','c'])
te['fsb']=pd.cut(te.fs,[0,9,13,18],labels=['a','b','c'])
tbl=tr[tr['pop'].between(1,18)].groupby(['pop','fsb'],observed=True)['y3'].mean()
te['exp3']=[tbl.get((p,b),np.nan) for p,b in zip(te['pop'],te.fsb)]
te=te.dropna(subset=['exp3']).copy()
te['resid']=te.y3-te.exp3

FE=['h_led_rate','h_pos_mean','h_pos_last','h_pos_min','h_n','d_dist','draw','fs',
    'j_led_rate','j_n','dist']
X=tr[FE].fillna(tr[FE].median()); mu,sd=X.mean(),X.std().replace(0,1)
m=LogisticRegression(max_iter=3000).fit((X-mu)/sd,tr.led)
te['p_lead']=m.predict_proba((te[FE].fillna(tr[FE].median())-mu)/sd)[:,1]
te['p_rank']=te.groupby('race_id')['p_lead'].rank(ascending=False,method='first')
print(f'検定期 {len(te):,}頭 / {te.race_id.nunique():,}レース  '
      f'実際の先頭通過率 {100*te.led.mean():.1f}%')

print('\n■ 事後（実際に逃げたか）で切る — これが +15pt の正体')
for k,s in te.groupby(te.led):
    print(f'   {"実際に先頭":10s}' if k else f'   {"それ以外":10s}',
          f'N={len(s):6,d}  複勝率{100*s.y3.mean():5.1f}%  '
          f'市場残差{100*s.resid.mean():+6.2f}pt')

print('\n■ 事前（予測した逃げやすさ）で切る — こちらが使える量')
q=pd.qcut(te.p_lead,5,labels=['Q1低','Q2','Q3','Q4','Q5高'])
for k,s in te.groupby(q,observed=True):
    print(f'   {str(k):8s} N={len(s):6,d}  実際に先頭{100*s.led.mean():5.1f}%  '
          f'複勝率{100*s.y3.mean():5.1f}%  市場残差{100*s.resid.mean():+6.2f}pt')

print('\n🔑 決定打: 予測が同じ馬どうしを、実際に逃げたかで割る')
print(f'{"":10s} {"":>14s} {"逃げた":>22s} {"逃げなかった":>22s}')
for k,s in te.groupby(q,observed=True):
    a=s[s.led==1]; b=s[s.led==0]
    if len(a)<50: continue
    print(f'   {str(k):8s} N={len(s):6,d}  N={len(a):5,d} 残差{100*a.resid.mean():+6.2f}pt'
          f'   N={len(b):6,d} 残差{100*b.resid.mean():+6.2f}pt'
          f'   差{100*(a.resid.mean()-b.resid.mean()):+6.2f}pt')

print('\n■ レース内で「最も逃げそう」と予測した1頭（＝実際に買える単位）')
top=te[te.p_rank==1]
print(f'   N={len(top):,}  的中(実際に先頭){100*top.led.mean():.1f}%  '
      f'複勝率{100*top.y3.mean():.1f}%  市場残差{100*top.resid.mean():+.2f}pt')
for k,s in top.groupby(top.led):
    print(f'      {"→ 実際に逃げた" if k else "→ 逃げられなかった"}: '
          f'N={len(s):5,d} 複勝率{100*s.y3.mean():5.1f}% 残差{100*s.resid.mean():+6.2f}pt')
