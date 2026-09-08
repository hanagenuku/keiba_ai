import os, sys, json
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from p4_analyze import summarize, race_struct
pd.set_option('display.width', 220)

f = pd.read_pickle(f'{BASE}/p4_frame.pkl')
CP = json.load(open(f'{BASE}/data/course_profiles.json'))['courses']

# ── 条件列を作る（分析軸。特徴量ではない）──────────────────────────
f['surf'] = f['surface'].fillna('不明')
def dband(r):
    d = r['distance']
    if not np.isfinite(d): return '不明'
    if r['surf'] == '芝':
        return ('芝〜1200' if d<=1200 else '芝1400' if d<=1400 else '芝1600' if d<=1600 else
                '芝1800' if d<=1800 else '芝2000' if d<=2000 else '芝2200-2400' if d<=2400 else '芝2500〜')
    return ('ダ〜1200' if d<=1200 else 'ダ1400' if d<=1400 else 'ダ1600' if d<=1600 else
            'ダ1700' if d<=1700 else 'ダ1800' if d<=1800 else 'ダ2000' if d<=2000 else 'ダ2100〜')
f['dist_band'] = f.apply(dband, axis=1)
f['field_band'] = pd.cut(f['field'], [0,9,12,15,99], labels=['〜9頭','10-12頭','13-15頭','16頭〜'])
def cls(x):
    x = str(x)
    for k in ['新馬','未勝利','1勝','2勝','3勝','オープン','OP']:
        if k in x: return 'オープン' if k in ('OP','オープン') else k
    return 'その他'
f['cls'] = f['race_class'].map(cls)
f['going'] = f['track_condition'].fillna('不明')
key = f['racecourse'] + '_' + f['surf']
f['turn'] = key.map(lambda k: CP.get(k, {}).get('turn', '不明'))
f['straight'] = key.map(lambda k: CP.get(k, {}).get('straight_class', '不明'))
f['uphill'] = key.map(lambda k: {True:'坂あり',False:'坂なし'}.get(CP.get(k,{}).get('has_uphill'), '不明'))
f['ctype'] = key.map(lambda k: CP.get(k, {}).get('course_type', '不明'))

# ── レース構造 ────────────────────────────────────────────────
rs = race_struct(f)
f = f.merge(rs, on=['win','race_id'], how='left')
for c, lab in [('gap12','AI1位2位差'), ('top3','AI上位3頭集中'), ('entropy','AIエントロピー')]:
    q = f.groupby('period')[c].transform(lambda s: pd.qcut(s.rank(method='first'), 4, labels=['Q1低','Q2','Q3','Q4高']))
    f[c+'_q'] = q.astype(str)
f['ai1_vs_mkt'] = f['ai1_pop'].map(lambda p: 'AI1位=市場1番人気' if p==1 else
                                   'AI1位=市場2-3番人気' if p<=3 else 'AI1位=市場4番人気以下')
f['agree3_lab'] = f['agree3'].map(lambda a: f'上位3頭一致 {int(a)}頭' if np.isfinite(a) else '不明')

AXES = [('racecourse','競馬場'), ('surf','芝ダート'), ('dist_band','距離帯'),
        ('field_band','頭数'), ('cls','クラス'), ('going','馬場状態'),
        ('turn','回り'), ('straight','直線'), ('uphill','坂'), ('ctype','コース形態'),
        ('gap12_q','AI1位2位差'), ('top3_q','AI上位3頭集中'), ('entropy_q','AIエントロピー'),
        ('ai1_vs_mkt','AI1位の市場人気'), ('agree3_lab','AI上位3頭と市場上位3頭の一致')]
out = []
for col, lab in AXES:
    out.append(summarize(f, col, lab))
    print(f'  {lab} 完了', flush=True)
res = pd.concat(out, ignore_index=True)
res.to_csv(f'{BASE}/p4_conditions.csv', index=False)
print(f'-> p4_conditions.csv  ({len(res)}条件)')
