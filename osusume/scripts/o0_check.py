"""Step 0: 計測器の検算。本番の race_predictions から pace_bonus を復元できるか。

  総スコア total は softmax(T=3.5) を通って win_prob になる。
  よってレース内で  3.5*ln(win_prob) = total + const。
  total_OFF = total_ON - pace_bonus であり、
  cal_prob は pace_bonus が乗る**前**の生マージンの単調変換。
  → rank(total_ON - pb) が rank(cal_prob) と一致しなければ復元は失敗。
"""
import sqlite3, numpy as np, pandas as pd
B='/tmp/claude-0/-home-user-keiba-ai/e7ea39b9-e1d1-58ea-9dba-4df19f2e8216/scratchpad'

con=sqlite3.connect(f'{B}/k2.db')
rp=pd.read_sql("""SELECT date,race_id,racecourse,race_num,horse_num,horse_name,popularity,
                         tansho_odds,rl_rank,win_prob,cal_prob,actual_place
                  FROM race_predictions""",con); con.close()
print(f'race_predictions {len(rp)}頭 / {rp.race_id.nunique()}レース')

hf=pd.read_csv(f'{B}/cutrace/data/horse_features.csv',
               usecols=['race_id','horse_num','f_pace','f_pace_prob_slow','f_pace_prob_fast'])
d=rp.merge(hf,on=['race_id','horse_num'],how='inner')
print(f'特徴量が引けた   {len(d)}頭 / {d.race_id.nunique()}レース')

d=d[(d.win_prob>0)&(d.cal_prob.notna())].copy()
g=d.groupby('race_id')
d['pb']=(d.f_pace-g['f_pace'].transform('mean'))*(d.f_pace_prob_slow-d.f_pace_prob_fast)*0.5
d['tON']=3.5*np.log(d.win_prob)
d['tOFF']=d.tON-d.pb

for c,nm in (('win_prob','rank(win_prob)'),('cal_prob','rank(cal_prob)'),
             ('tOFF','rank(tON-pb)')):
    d[f'r_{c}']=d.groupby('race_id')[c].rank(ascending=False,method='first').astype(int)

print('\n■ 検算1: rank(win_prob) == rl_rank（DBの整合）')
ok=(d.r_win_prob==d.rl_rank)
print(f'   一致 {ok.mean():.2%}  ({int(ok.sum())}/{len(d)})')

print('\n■ 検算2: rank(tON - pb) == rank(cal_prob)  ← pace_bonus 復元の正しさ')
for lo,hi,lab in (('2026-06-27','2026-08-31','error_tags ON期'),
                  ('2026-09-01','2026-12-31','error_tags OFF期')):
    s=d[(d.date>=lo)&(d.date<=hi)]
    if not len(s): continue
    m=(s.r_tOFF==s.r_cal_prob)
    # レース単位で「全頭一致」も見る
    per=s.groupby('race_id').apply(lambda x:(x.r_tOFF==x.r_cal_prob).all(),include_groups=False)
    print(f'   {lab:16s} 馬単位 {m.mean():6.2%}  レース単位(全頭一致) {per.mean():6.2%} '
          f'({int(per.sum())}/{len(per)}R)')
d.to_pickle(f'{B}/osusume/d.pkl')
print('\n保存: osusume/d.pkl')
