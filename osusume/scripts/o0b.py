"""検算2が81-92%止まりの原因を切り分ける。同点・丸め・f_pace のどれか。"""
import numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
d=pd.read_pickle(f'{B}/osusume/d.pkl')

# cal_prob の同点（キャリブレーターの床）がどれだけあるか
tie=d.groupby('race_id')['cal_prob'].transform(lambda s:s.duplicated(keep=False))
print(f'cal_prob が同点の馬 {tie.mean():.2%}')

def conc(sub, a, b):
    """同一レース内のペアで順序が一致する率（同点ペアは除外）"""
    ok=tot=0
    for _,x in sub.groupby('race_id'):
        va,vb=x[a].to_numpy(),x[b].to_numpy()
        n=len(va)
        for i in range(n):
            for j in range(i+1,n):
                if va[i]==va[j] or vb[i]==vb[j]: continue
                tot+=1
                ok+= (va[i]>va[j])==(vb[i]>vb[j])
    return ok,tot

for lo,hi,lab in (('2026-06-27','2026-08-31','error_tags ON期'),
                  ('2026-09-01','2026-12-31','error_tags OFF期')):
    s=d[(d.date>=lo)&(d.date<=hi)]
    if not len(s): continue
    ok,tot=conc(s,'tOFF','cal_prob')
    ok2,tot2=conc(s,'tON','cal_prob')
    print(f'{lab:16s} ペア一致  tON-pb vs cal {ok/tot:.4%} ({tot}ペア)  '
          f'| 補正しない tON vs cal {ok2/tot2:.4%}')
