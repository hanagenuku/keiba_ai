"""T'-C2: 競馬場 one-hot が効いた理由を分解する（診断。採用判断はしない）。

ユーザー指摘:
  「コースが重要」までは強く言えるが「何が重要なのか」はまだ分かっていない。
  どの競馬場がどれだけ寄与しているか、できれば 競馬場×距離 まで見たい。

⚠ one-hot から1会場分の列を落とす ablation は**使えない**。
   残りが全部0であることでその会場だと分かるので、情報が消えない（ダミー変数の罠）。
   代わりに ①会場ごとの改善量 ②改善の中身（水準のズレか交互作用か）を測る。

使い方: python3 tenkai/c4.py <db_path>
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from tenkai.c1 import build, WINDOWS
from src.features.race_shape import PACE_INPUT_COLS, VENUES, dist_zone

SEEDS = (42, 7, 2026)
BASE4 = ['dist', 'surface_num', 'cls', 'n_horses']
OH = [f'venue_{v}' for v in VENUES]


def fit_predict(trd, ted, cols, seed):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0)
    m.fit(trd[cols].astype(float), trd.early)
    return m.predict(trd[cols].astype(float)), m.predict(ted[cols].astype(float))


def rmse(p, y):
    return float(np.sqrt(((np.asarray(p) - np.asarray(y)) ** 2).mean()))


def run(db):
    H, R = build(db, '.')
    for v in VENUES:
        R[f'venue_{v}'] = (R.rc == v).astype(float)
    R['dz'] = R.dist.map(dist_zone)

    per_venue = {}
    per_cell = {}
    decomp = {}
    for wname, tend, vend in WINDOWS:
        trd = R[R.date <= tend].copy()
        ted = R[(R.date > tend) & (R.date <= vend)].copy()
        p0_tr = np.mean([fit_predict(trd, ted, BASE4, s)[0] for s in SEEDS], axis=0)
        p0 = np.mean([fit_predict(trd, ted, BASE4, s)[1] for s in SEEDS], axis=0)
        p1 = np.mean([fit_predict(trd, ted, BASE4 + OH, s)[1] for s in SEEDS], axis=0)
        ted['p0'], ted['p1'] = p0, p1
        trd['r0'] = trd.early.values - p0_tr

        # ① 会場ごとの改善
        for v, g in ted.groupby('rc'):
            r0, r1 = rmse(g.p0, g.early), rmse(g.p1, g.early)
            per_venue.setdefault(v, {})[wname] = (len(g), r0, r1, r1 - r0)

        # ② 会場×距離帯
        for (v, dz), g in ted.groupby(['rc', 'dz']):
            if len(g) < 30:
                continue
            per_cell.setdefault((v, dz), {})[wname] = (
                len(g), rmse(g.p0, g.early), rmse(g.p1, g.early),
                rmse(g.p1, g.early) - rmse(g.p0, g.early))

        # ③ 改善の中身: 「会場ごとの定数ずらし」でどこまで再現できるか
        #    学習期の残差の平均を会場ごとに取り、検定期の予測に足すだけ
        off_v = trd.groupby('rc')['r0'].mean()
        off_vs = trd.groupby(['rc', 'sf'])['r0'].mean()
        off_vd = trd.groupby(['rc', 'sf', 'dz'])['r0'].mean()
        shift_v = ted.p0 + ted.rc.map(off_v).fillna(0.0).values
        shift_vs = ted.p0 + pd.Series(list(zip(ted.rc, ted.sf))).map(off_vs).fillna(0.0).values
        shift_vd = ted.p0 + pd.Series(list(zip(ted.rc, ted.sf, ted.dz))).map(off_vd).fillna(0.0).values
        decomp[wname] = {
            'M0 会場なし': rmse(ted.p0, ted.early),
            'M0 + 会場ごとの定数': rmse(shift_v, ted.early),
            'M0 + 会場×馬場の定数': rmse(shift_vs, ted.early),
            'M0 + 会場×馬場×距離帯の定数': rmse(shift_vd, ted.early),
            'M1h 会場をモデルに入れる': rmse(ted.p1, ted.early),
        }
        # 会場ごとの水準ずれ（秒）も出す＝「何が違うのか」の中身
        decomp[wname]['_offsets'] = off_v.to_dict()

    print('■ ① 会場ごとの改善（RMSE の変化・マイナスが改善／3シード平均）\n')
    print(f'{"会場":6s} {"窓A n":>6s} {"M0":>8s} {"M1h":>8s} {"差":>9s}   '
          f'{"窓B n":>6s} {"M0":>8s} {"M1h":>8s} {"差":>9s}   判定')
    rows = []
    for v in VENUES:
        d = per_venue.get(v, {})
        a, b = d.get('窓A'), d.get('窓B')
        if not a or not b:
            continue
        mark = '✅両窓で改善' if a[3] < 0 and b[3] < 0 else ('❌両窓で悪化' if a[3] > 0 and b[3] > 0 else '△不一致')
        print(f'{v:6s} {a[0]:6d} {a[1]:8.4f} {a[2]:8.4f} {a[3]:+9.4f}   '
              f'{b[0]:6d} {b[1]:8.4f} {b[2]:8.4f} {b[3]:+9.4f}   {mark}')
        rows.append((v, a, b))

    print('\n■ ② 会場×距離帯（n>=30 のセルのみ・両窓そろうもの）\n')
    cells = []
    for (v, dz), d in per_cell.items():
        if '窓A' in d and '窓B' in d:
            cells.append((v, dz, d['窓A'], d['窓B']))
    cells.sort(key=lambda x: x[2][3] + x[3][3])
    print(f'{"会場×距離帯":22s} {"窓A n":>6s} {"窓A 差":>9s}  {"窓B n":>6s} {"窓B 差":>9s}   判定')
    for v, dz, a, b in cells:
        mark = '✅両窓で改善' if a[3] < 0 and b[3] < 0 else ('❌両窓で悪化' if a[3] > 0 and b[3] > 0 else '△不一致')
        print(f'{v+" "+dz:22s} {a[0]:6d} {a[3]:+9.4f}  {b[0]:6d} {b[3]:+9.4f}   {mark}')

    print('\n■ ③ 改善の中身: 定数ずらしでどこまで再現できるか（RMSE・秒）\n')
    keys = ['M0 会場なし', 'M0 + 会場ごとの定数', 'M0 + 会場×馬場の定数',
            'M0 + 会場×馬場×距離帯の定数', 'M1h 会場をモデルに入れる']
    print(f'{"":30s} {"窓A":>9s} {"窓B":>9s}')
    for k in keys:
        print(f'{k:30s} {decomp["窓A"][k]:9.4f} {decomp["窓B"][k]:9.4f}')
    for w in ('窓A', '窓B'):
        tot = decomp[w]['M0 会場なし'] - decomp[w]['M1h 会場をモデルに入れる']
        got = decomp[w]['M0 会場なし'] - decomp[w]['M0 + 会場ごとの定数']
        print(f'  {w}: 会場の定数ずらしだけで改善の {100*got/tot:.0f}% を回収')

    print('\n■ 会場ごとの水準ずれ（学習期の残差平均・秒。+ は M0 の予測より実際は遅い）\n')
    oa, ob = decomp['窓A']['_offsets'], decomp['窓B']['_offsets']
    print(f'{"会場":6s} {"窓A":>8s} {"窓B":>8s}')
    for v in sorted(VENUES, key=lambda x: -abs(oa.get(x, 0))):
        if v in oa and v in ob:
            print(f'{v:6s} {oa[v]:+8.3f} {ob[v]:+8.3f}')


if __name__ == '__main__':
    run(sys.argv[1])
