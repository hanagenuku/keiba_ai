import os, sys
import numpy as np, pandas as pd
BASE = os.path.dirname(os.path.abspath(__file__))
r = pd.read_csv(f'{BASE}/p4_conditions.csv')
BASE_D3 = -0.0197   # 確認期 全体の AI−市場（3着内AUC）
r = r[r.CONF_d3.notna() & r.TUNE_d3.notna()].copy()
r['sign_match'] = np.sign(r.TUNE_d3 - BASE_D3) == np.sign(r.CONF_d3 - BASE_D3)

def show(df, title):
    print(f'\n===== {title} =====')
    print(f"{'軸':<12}{'条件':<22}{'R数(確認)':>9}{'頭数':>8}{'AI3':>8}{'市場3':>8}{'差3':>9}"
          f"{'探索差3':>9}{'差1着':>9}{'CI(確認)':>18} 再現")
    for x in df.itertuples():
        ci = (f'[{x.CONF_lo:+.4f},{x.CONF_hi:+.4f}]'
              if hasattr(x,'CONF_lo') and np.isfinite(getattr(x,'CONF_lo',np.nan)) else '')
        print(f'{x.axis:<12}{x.cond:<22}{int(x.CONF_race):>9}{int(x.CONF_horse):>8}'
              f'{x.CONF_ai3:>8.4f}{x.CONF_mk3:>8.4f}{x.CONF_d3:>+9.4f}'
              f'{x.TUNE_d3:>+9.4f}{x.CONF_d1:>+9.4f}{ci:>18} {"○" if x.sign_match else "×"}')

r = r.sort_values('CONF_d3', ascending=False)
show(r.head(12), 'A. AIが市場に相対的に強い条件（確認期・差の大きい順 上位12）')
show(r.tail(10).iloc[::-1], 'C. AIが市場より明確に弱い条件（下位10）')
print(f'\n全体基準線（確認期）: AI−市場 3着内 {BASE_D3:+.4f}')
both = r[(r.CONF_d3 > BASE_D3) & (r.TUNE_d3 > BASE_D3)]
print(f'探索期・確認期の両方で全体基準線を上回った条件: {len(both)} / {len(r)}')
