"""L-3: 「予測精度をどこまで上げれば元が取れるか」を算数で出す。"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
exec(open(f'{B}/lead/l2.py').read().split("print(f'検定期")[0])

top=te[te.p_rank==1]
a=top[top.led==1].resid.mean()*100; b=top[top.led==0].resid.mean()*100
p=top.led.mean()
print(f'■ レース内で最も逃げそうと予測した1頭（N={len(top):,}）')
print(f'   的中率 p = {100*p:.1f}%')
print(f'   当たったとき a = {a:+.2f}pt / 外したとき b = {b:+.2f}pt')
print(f'   期待値 = p·a + (1-p)·b = {p*a+(1-p)*b:+.2f}pt')
be=-b/(a-b)
print(f'\n   損益分岐の的中率 = -b/(a-b) = {100*be:.1f}%   （いま {100*p:.1f}%）')
print(f'   → 予測を {100*p:.0f}% から {100*be:.0f}% まで上げて、ようやく残差ゼロ')

print('\n🔴 しかも予測が上手くなるほど、当たったときの取り分が減る')
q=pd.qcut(te.p_lead,5,labels=['Q1','Q2','Q3','Q4','Q5'])
r=[]
for k,s in te.groupby(q,observed=True):
    x=s[s.led==1]
    if len(x)<50: continue
    r.append((str(k),100*s.led.mean(),100*x.resid.mean()))
for k,pl,rr in r:
    print(f'   {k}  実際に先頭 {pl:5.1f}%  →  逃げたときの残差 {rr:+6.2f}pt')
c=np.corrcoef([x[1] for x in r],[x[2] for x in r])[0,1]
print(f'   corr(逃げやすさ, 逃げたときの残差) = {c:+.3f}')
print('   ＝ 価値は「**意外な逃げ**」に集中している。予測できるほど価値が消える。')

print('\n■ 参考: 完璧に当てられた場合（AUC 1.0 の反実仮想）')
print(f'   実際に先頭の残差 {100*te[te.led==1].resid.mean():+.2f}pt  N={int(te.led.sum()):,}')
print(f'   ⚠ ただし上の相関どおり、完璧に当てる＝全部Q5に寄るので')
print(f'      実際に取れるのは Q5 の {r[-1][2]:+.2f}pt 側に縮む')
