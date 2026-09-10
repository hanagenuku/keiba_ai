"""②-A 補足: 負の残差（＝市場が過大評価している側）を「切る」と回収率はどこまで行くか。

上位分位の正の残差は最大 +3.23pt だったが、負の側は -4.39pt とより大きい。
負の残差は直接は買えない（空売りできない）ので、**除外**でしか使えない。
補集合の式で天井を出す:  r_good = (r_pool - w*r_bad) / (1 - w)
"""
import pandas as pd, numpy as np
POOL = 69.9
r = pd.read_csv('gap_a.csv')
N_CONF = 22399
a = r[(r['mode']=='事前指定')].copy()
a['w'] = a.n / N_CONF
a['roi_bad'] = a.roi_lin
a['roi_good'] = (POOL - a.w*a.roi_bad) / (1 - a.w)
a['gain'] = a.roi_good - POOL
b = a.sort_values('gain', ascending=False)
print(f'{"軸":<30}{"分位":<8}{"N":>7}{"除外率":>8}{"残差":>9}{"切った側":>9}{"残り":>8}{"改善":>8}')
for _, x in b.head(10).iterrows():
    print(f'{x.feat:<30}{x.side:<8}{x.n:>7,}{x.w*100:>7.1f}%{x.res:>+8.2f}pt'
          f'{x.roi_bad:>8.1f}%{x.roi_good:>7.1f}%{x.gain:>+7.2f}pt')
# 複数軸を同時に切る上限（重複を無視した楽観的な上界）
top = b.head(5)
w = top.w.sum(); rb = (top.w*top.roi_bad).sum()/w
print(f'\n上位5軸を全部切った場合（重複を無視＝楽観的な上界）:')
print(f'  除外率 {w*100:.1f}%  切った側 {rb:.1f}%  残り {(POOL-w*rb)/(1-w):.1f}%  '
      f'改善 {(POOL-w*rb)/(1-w)-POOL:+.2f}pt')
print(f'\n100% に到達するのに必要な除外条件:')
for w_ in (0.1, 0.2, 0.3, 0.4):
    need = (POOL - 100*(1-w_))/w_
    print(f'  除外率 {w_*100:.0f}% なら、切る側の回収率が {need:.1f}% でなければならない'
          f'   （実測の最悪は 50.8%）')
