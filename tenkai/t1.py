"""T-1: 展開予想（ペース）の実際の精度を測る。

  学習時: running_style=実際の脚質 / agari3f=そのレースの実測
  推論時: running_style=過去走からの推定 / agari3f=定数36.0,1.5
  → この差が精度をどれだけ食っているか。
"""
import sqlite3, numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'
con=sqlite3.connect(f'{B}/hist3.db')
R=pd.read_sql("""SELECT race_id,date,racecourse rc,surface sf,distance dist,first_3f,
                        num_finishers fs,track_condition tc,race_class
                 FROM race_history WHERE first_3f IS NOT NULL AND first_3f>0""",con)
H=pd.read_sql("""SELECT race_id,date,horse_name,horse_num,running_style,agari3f,
                        popularity pop,jockey,corner_all,place,distance dist
                 FROM horse_history WHERE place>0 AND place<99""",con); con.close()
print(f'first_3f のあるレース {len(R):,}  {R.date.min()}〜{R.date.max()}')
print(f'  年別: '+' / '.join(f'{y} {n}' for y,n in R.date.str[:4].value_counts().sort_index().items()))

# ── ペース3分類ラベル（距離帯×表面のパーセンタイル。本番と同じ定義）
def dz(x): return '~1400' if x<=1400 else '1401-1800' if x<=1800 else '1801-2200' if x<=2200 else '2201~'
R['dz']=R.dist.map(dz)
q=R.groupby(['sf','dz'])['first_3f'].quantile([.33,.67]).unstack()
def lab(r):
    t=q.loc[(r.sf,r.dz)]
    return 'high' if r.first_3f<=t[0.33] else 'slow' if r.first_3f>=t[0.67] else 'mid'
R['pace']=R.apply(lab,axis=1)
print(f'  ラベル分布 {R.pace.value_counts(normalize=True).round(3).to_dict()}')

# ── 馬ごとの「過去走からの推定脚質」（当該レースを除く）
H=H.sort_values(['horse_name','date']).reset_index(drop=True)
g=H.groupby('horse_name')['running_style']
def modal_prev(s):
    out=[]; from collections import Counter; c=Counter()
    for v in s:
        out.append(c.most_common(1)[0][0] if c else None)
        if pd.notna(v) and v: c[v]+=1
    return pd.Series(out,index=s.index)
H['est_style']=g.transform(modal_prev)
cov=H.est_style.notna().mean()
print(f'\n過去走から脚質を推定できた率 {100*cov:.1f}%  '
      f'（推定と実際の一致 {100*(H.est_style==H.running_style).mean():.1f}%）')

# ── レース単位で集約: 実際版(T) と 推論版(S)
def agg(df, style_col, use_real_agari):
    o=df.groupby('race_id').agg(n=('horse_num','size'))
    st=df.set_index('race_id')[style_col]
    o['escape_count']=df.assign(x=(df[style_col]=='逃げ').astype(int)).groupby('race_id')['x'].sum()
    o['front_count'] =df.assign(x=(df[style_col]=='先行').astype(int)).groupby('race_id')['x'].sum()
    o['front_density']=(o.escape_count+o.front_count)/o.n
    if use_real_agari:
        a=df.groupby('race_id')['agari3f']
        o['avg_agari3f']=a.mean(); o['std_agari3f']=a.std().fillna(1.5)
    else:
        o['avg_agari3f']=36.0; o['std_agari3f']=1.5      # 本番の定数
    e=df[df[style_col]=='逃げ']
    o['escape_avg_pos']=e.groupby('race_id')['horse_num'].mean()
    o['escape_avg_pop']=e.groupby('race_id')['pop'].mean()
    return o.reset_index()

T=agg(H,'running_style',True)
S=agg(H.dropna(subset=['est_style']),'est_style',False)
print(f'\n集約: 実際版 {len(T):,}レース / 推論版 {len(S):,}レース')
T.to_pickle(f'{B}/tenkai/T.pkl'); S.to_pickle(f'{B}/tenkai/S.pkl')
R.to_pickle(f'{B}/tenkai/R.pkl'); H.to_pickle(f'{B}/tenkai/H.pkl')
print('保存完了')
