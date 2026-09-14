"""O' Step 4: 推 × 10番人気以下 を、2026-09-01 を殺した対照に当てる＋期間分割。

⚠ 2026-09-01 の教訓: 「6人気以下のAI最上位 72.9% vs **同じ帯の市場最上位 73.0%**」
   ＝AIの選別は何もしていなかった。同じ対照をここにも当てる。
"""
import numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
exec(open(f'{B}/osusume/o1_mark.py').read().split("for lo,lab in")[0])
sub=d.reset_index(drop=True)
r_on,pn_on=arms['ON']; r_off,pn_off=arms['OFF']
sub['osu_on']=marks(r_on,pn_on,sub); sub['osu_off']=marks(r_off,pn_off,sub)
sub['ai_on']=r_on; sub['ai_off']=r_off
LS=sub.popularity>=10                      # 10番人気以下
print(f'10番人気以下 {int(LS.sum())}頭 / {sub[LS].race_id.nunique()}レース')

def pick(mask_col):
    """レースごとに1頭選ぶ系の対照用インデックス集合"""
    return set()

sel={}
sel['★推(ON)']     = sub.osu_on & LS
sel['★推(OFF)']    = sub.osu_off & LS
# C1: 同帯で市場人気が最上位（＝その帯で最も人気のある馬）— 09-01を殺した対照
i1=sub[LS].groupby('race_id')['popularity'].idxmin()
sel['C1 同帯の市場最上位'] = sub.index.isin(i1)
# C2: 同帯でAI順位が最上位（推の条件を使わない素のAI）
i2=sub[LS].loc[sub[LS].groupby('race_id')['ai_on'].idxmin().values].index
sel['C2 同帯のAI最上位']  = sub.index.isin(i2)
# C3: 同帯から無作為1頭
rng=np.random.default_rng(11)
i3=sub[LS].groupby('race_id').apply(lambda x: rng.choice(x.index),include_groups=False)
sel['C3 同帯から無作為']  = sub.index.isin(i3.values)
# C4: 帯全部
sel['C4 同帯すべて']     = LS

def stat(m, days=None, drop=0):
    s=sub[m] if days is None else sub[m & sub.date.isin(days)]
    if not len(s): return None
    bad=(s.actual_place<=3)&(s.fukusho_payout.fillna(0)<=0)
    ok=s[~bad]; fp=ok.fukusho_payout.fillna(0).to_numpy().copy()
    tp=s.tansho_payout.fillna(0).to_numpy().copy()
    for _ in range(drop):
        if len(fp) and fp.max()>0: fp[fp.argmax()]=0
        if len(tp) and tp.max()>0: tp[tp.argmax()]=0
    return dict(n=len(s), f=int((s.actual_place<=3).sum()),
                fr=100*fp.sum()/(100*max(len(ok),1)),
                tr=100*tp.sum()/(100*len(s)))

udays=np.sort(sub.date.unique()); half=len(udays)//2
EX,CF=udays[:half],udays[half:]
print(f'探索期 {EX[0]}〜{EX[-1]} ({len(EX)}日) / 確認期 {CF[0]}〜{CF[-1]} ({len(CF)}日)\n')
print(f'{"":22s} {"N":>5s} {"複ROI":>7s} {"3本抜":>7s} | {"探索期":>14s} | {"確認期":>14s}')
for k,m in sel.items():
    a=stat(m); b=stat(m,drop=3); e=stat(m,EX); c=stat(m,CF)
    if a is None: continue
    print(f'{k:22s} {a["n"]:5d} {a["fr"]:6.1f}% {b["fr"]:6.1f}% | '
          f'N={e["n"]:4d} {e["fr"]:6.1f}% | N={c["n"]:4d} {c["fr"]:6.1f}%')

print('\n── 日ブロックbootstrap: 推(ON) − 各対照（複勝回収）──')
rng2=np.random.default_rng(3)
for k in ['C1 同帯の市場最上位','C2 同帯のAI最上位','C3 同帯から無作為','C4 同帯すべて','★推(OFF)']:
    diffs=[]
    for _ in range(2000):
        ds=rng2.choice(udays,len(udays),replace=True)
        a=stat(sel['★推(ON)'],ds); b=stat(sel[k],ds)
        if a and b: diffs.append(a['fr']-b['fr'])
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
    base=stat(sel['★推(ON)'])['fr']-stat(sel[k])['fr']
    print(f'   vs {k:22s} {base:+7.1f}pt  CI[{lo:+7.1f},{hi:+7.1f}]  上回る {(diffs>0).mean():5.1%}')
