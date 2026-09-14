"""O' Step 3: 「推」という仕組みそのものに価値があるか（pace_bonus とは無関係）。

本番が実際に出した 推 マークと、同じレース・同じ人気帯の対照を比べる。
"""
import numpy as np, pandas as pd, sqlite3
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
exec(open(f'{B}/osusume/o1_mark.py').read().split("for lo,lab in")[0])
sub=d.reset_index(drop=True)
r_on,pn_on=arms['ON']
osu=marks(r_on,pn_on,sub)
sub['osusume']=osu
sub['ev']=pn_on*sub.tansho_odds.to_numpy()
sub['pop_band']=pd.cut(sub.popularity,[0,1,3,5,9,99],
                       labels=['1人気','2-3','4-5','6-9','10+'])

def stat(s,lab):
    if not len(s): return
    bad=(s.actual_place<=3)&(s.fukusho_payout.fillna(0)<=0)
    ok=s[~bad]
    print(f'  {lab:38s} N={len(s):5d} 勝{100*(s.actual_place==1).mean():5.1f}% '
          f'複{100*(s.actual_place<=3).mean():5.1f}% '
          f'単ROI {100*s.tansho_payout.fillna(0).sum()/(100*len(s)):6.1f}% '
          f'複ROI {100*ok.fukusho_payout.fillna(0).sum()/(100*max(len(ok),1)):6.1f}%')

print(f'\n■ 推 vs 対照  （{sub.race_id.nunique()}レース / {len(sub)}頭 / 推 {int(osu.sum())}頭）\n')
stat(sub[osu],'★推マーク')
# 対照1: 推が出たレースの、同じ人気帯の他馬
osu_races=set(sub[osu].race_id)
osu_bands=set(zip(sub[osu].race_id,sub[osu].pop_band.astype(str)))
c1=sub[[ (r,str(b)) in osu_bands for r,b in zip(sub.race_id,sub.pop_band)]&(~osu)]
stat(c1,'対照1 同レース・同人気帯の他馬')
# 対照2: 構造条件は満たすが sim_ev で落ちた馬（＝EV閾値の寄与を分離）
struct=(r_on<=sub.popularity.to_numpy()-3)&(r_on<=8)
stat(sub[struct&(~osu)],'対照2 AI強気だが sim_ev<1.2 で落選')
stat(sub[struct],'  （参考）AI強気な馬すべて')
# 対照3: 推と同じ人気分布を持つ「市場最上位」＝各推の人気帯でその帯の1番人気馬
print()
for b in ['1人気','2-3','4-5','6-9','10+']:
    m=osu&(sub.pop_band.astype(str)==b)
    if m.sum()==0: continue
    stat(sub[m],f'  推 × {b}')
    same=sub[(sub.pop_band.astype(str)==b)&sub.race_id.isin(osu_races)&(~osu)]
    stat(same,f'    対照 同帯・推以外')
