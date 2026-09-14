"""L-1: 「先頭に立つ馬」は事前にどこまで当てられるか。

ユーザーの提案の土台になる数字。過去走だけから作った特徴量で
「この馬が最初のコーナーを先頭で通過するか」を予測し、AUC を測る。
"""
import sqlite3, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
con=sqlite3.connect(f'{B}/hist3.db')
d=pd.read_sql("""SELECT h.race_id,h.date,h.horse_name,h.horse_num,h.corner_all,h.running_style,
                        h.place,h.popularity pop,h.jockey,h.distance dist,h.surface sf,
                        h.racecourse rc,h.agari3f,r.num_finishers fs
                 FROM horse_history h LEFT JOIN race_history r USING(race_id)
                 WHERE h.place>0 AND h.place<99""",con); con.close()
d['fs']=d.fs.fillna(d.groupby('race_id')['horse_num'].transform('max'))
d=d[d.fs.between(5,18)].copy()
ca=d.corner_all.fillna('')
d['npt']=np.where(ca=='',0,ca.str.count('-')+1)
d['c1']=pd.to_numeric(ca.str.split('-').str[0],errors='coerce')
d=d[(d.npt>=2)&d.c1.notna()].copy()
d['led']=(d.c1==1).astype(int)
d['pos_n']=(d.c1-1)/np.maximum(d.fs-1,1)          # 0=先頭, 1=最後方
d=d.sort_values(['horse_name','date']).reset_index(drop=True)
print(f'{len(d):,}行 / {d.race_id.nunique():,}レース  先頭通過率 {100*d.led.mean():.1f}%')
print(f'  4地点そろう行 {100*(d.npt==4).mean():.1f}%（残りは3-4角のみ＝「最初の」が3角）')

g=d.groupby('horse_name')
# ── 過去走のみ（shift で自分の当該レースを除く）
for src,name in ((d.led,'led'),(d.pos_n,'pos')):
    cs=g[src.name if hasattr(src,'name') else None]
d['h_led_rate']=g['led'].transform(lambda s: s.shift().expanding().mean())
d['h_pos_mean']=g['pos_n'].transform(lambda s: s.shift().expanding().mean())
d['h_pos_last']=g['pos_n'].transform(lambda s: s.shift())
d['h_pos_min']=g['pos_n'].transform(lambda s: s.shift().expanding().min())
d['h_n']=g['led'].transform(lambda s: s.shift().expanding().count())
d['h_dist_last']=g['dist'].transform(lambda s: s.shift())
d['d_dist']=d.dist-d.h_dist_last
# 騎手の逃げ率（過去のみ・日付順）
d=d.sort_values(['jockey','date'])
d['j_led_rate']=d.groupby('jockey')['led'].transform(lambda s: s.shift().expanding().mean())
d['j_n']=d.groupby('jockey')['led'].transform(lambda s: s.shift().expanding().count())
d=d.sort_values('date').reset_index(drop=True)
d['draw']=d.horse_num/d.fs

cut='2025-07-01'
tr,te=d[d.date<cut],d[d.date>=cut]
FE=['h_led_rate','h_pos_mean','h_pos_last','h_pos_min','h_n','d_dist','draw','fs',
    'j_led_rate','j_n','dist']
def fit(cols,tr,te):
    X=tr[cols].fillna(tr[cols].median()); Xe=te[cols].fillna(tr[cols].median())
    m=LogisticRegression(max_iter=3000).fit((X-X.mean())/X.std().replace(0,1),tr.led)
    Z=(Xe-X.mean())/X.std().replace(0,1)
    return roc_auc_score(te.led,m.predict_proba(Z)[:,1])
print(f'\n学習 {len(tr):,}行 (〜{cut}) / 検定 {len(te):,}行 ({cut}〜)  '
      f'検定の先頭通過率 {100*te.led.mean():.1f}%')
print('\n■ 「先頭で通過するか」を当てるAUC（すべて過去走のみ・完全OOS）')
sets=[(['pop'],'市場人気だけ'),
      (['draw','fs','dist'],'枠・頭数・距離だけ'),
      (['h_led_rate','h_n'],'過去の逃げ率だけ'),
      (['h_pos_mean','h_pos_last','h_pos_min','h_n'],'過去の通過位置'),
      (['j_led_rate','j_n'],'騎手の逃げ率だけ'),
      (FE,'全部（騎手・枠・距離変化込み）'),
      (FE+['pop'],'全部＋市場人気')]
for cols,nm in sets:
    print(f'   {nm:26s} AUC {fit(cols,tr,te):.4f}')
d.to_pickle(f'{B}/lead/d.pkl')
