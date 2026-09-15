"""T'-G: 物理特徴に「未見コースへの一般化能力」はあるか（基準は CRITERIA_GENERALIZE.md）。

leave-one-course-out: (会場,馬場,実距離) を丸ごと学習から外して、その組み合わせを予測する。
  A  base4 + 会場one-hot        ← 会場は知っているが、その距離の癖は知らない
  B  A + 物理特徴               ← 物理構造から推測できるか

使い方: python3 tenkai/g1.py <db_path>
"""
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, '.')
from tenkai.c1 import build
from src.features.race_shape import VENUES

MIN_R = 30
SEED = 42
N_SHUFFLE = 24
START_LOCS = ['向正面', '正面スタンド前', 'ホームストレッチ', 'ホームストレッチ中ほど',
              '4角奥ポケット', '2角付近', '2角奥', '3コーナー付近', '1〜2コーナー間',
              '1コーナー奥', '1コーナー奥の引き込み線', '1〜2角間の引き込み線',
              '4角〜直線入口付近', '4角奥', '外回り2コーナー付近', '外回りコース上',
              '正面直線半ば', '直線半ば', 'スタンド前', 'スタンド前付近',
              '向正面中央付近', '直線コース']


def phys_cols(starts, courses):
    """(rc,sf,dist) -> 物理特徴の dict。数値は official+stated のみ入っている。"""
    out = {}
    for k, v in starts.items():
        rc, sf, dist = v['racecourse'], v['surface'], int(v['distance'])
        base = courses.get(f'{rc}_{sf}', {})
        row = {
            'ph_s2c': v.get('start_to_first_corner_m'),
            'ph_fc': v.get('first_corner'),
            'ph_slope': {'上り': 1.0, '下り': -1.0}.get(v.get('start_slope')),
            'ph_loop': {'内回り': 1.0, '外回り': 2.0, '直線': 3.0}.get(v.get('loop')),
            'ph_spiral': 1.0 if v.get('spiral_corner') else 0.0,
            'ph_turfstart': 1.0 if v.get('turf_start') else 0.0,
            'ph_straight': base.get('final_straight_m'),
            'ph_elev': base.get('course_elevation_range_m'),
            'ph_lap': base.get('lap_m'),
        }
        for loc in START_LOCS:
            row[f'ph_loc_{loc}'] = 1.0 if v.get('start_location') == loc else 0.0
        out[(rc, sf, dist)] = row
    return out


def fit_rmse(tr, te, cols, seed=SEED):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0)
    m.fit(tr[cols].astype(float), tr.early)
    p = m.predict(te[cols].astype(float))
    return float(np.sqrt(((p - te.early.values) ** 2).mean()))


def run(db):
    _, R = build(db, '.')
    for v in VENUES:
        R[f'venue_{v}'] = (R.rc == v).astype(float)
    OH = [f'venue_{v}' for v in VENUES]
    BASE = ['dist', 'surface_num', 'cls', 'n_horses']
    A = BASE + OH

    doc = json.load(open('data/course_starts.json'))['starts']
    ph_src = json.load(open('data/course_physical.json'))['courses']
    P = phys_cols(doc, ph_src)
    PH = sorted(next(iter(P.values())).keys())

    R['key'] = list(zip(R.rc, R.sf, R.dist.astype(int)))
    R = R[R.key.isin(P)].copy()
    for c in PH:
        R[c] = [P[k][c] if P[k][c] is not None else np.nan for k in R.key]
    sz = R.groupby('key').size()
    targets = [k for k in sz.index if sz[k] >= MIN_R]
    print(f'物理データがあり n>={MIN_R}R の組み合わせ: {len(targets)}  （対象レース {sz[targets].sum():,}）')
    print(f'物理特徴 {len(PH)}列  うち数値が埋まっている組: '
          f'ph_s2c {sum(1 for k in targets if not np.isnan(R[R.key==k].ph_s2c.iloc[0]))}/{len(targets)}\n')

    # 対照群: 物理特徴を組み合わせ間でシャッフル
    # 🔴 1回だけでは足りない。31列を足すこと自体で RMSE が動くので、
    #    シャッフルの**分布**と比べないと「物理量が効いた」とは言えない。
    #    （初版は1回だけ回して、緩い判定式で ✅ を出してしまった）
    def shuffled_cols(seed):
        rng = np.random.default_rng(seed)
        ks = list(P.keys()); perm = list(ks); rng.shuffle(perm)
        m = {k: P[v] for k, v in zip(ks, perm)}
        for c in PH:
            R['sh_' + c] = [m[k][c] if m[k][c] is not None else np.nan for k in R.key]
        return ['sh_' + c for c in PH]

    SH = shuffled_cols(0)
    # 注入: 真のオフセット（学習側から見える形で）を1列足す
    R['oracle'] = R.groupby('key')['early'].transform('mean')

    rows = []
    for k in targets:
        te = R[R.key == k]
        tr = R[R.key != k]
        rows.append((k, len(te),
                     fit_rmse(tr, te, A),
                     fit_rmse(tr, te, A + PH),
                     fit_rmse(tr, te, A + SH),
                     fit_rmse(tr, te, A + ['oracle'])))
    D = pd.DataFrame(rows, columns=['key', 'n', 'A', 'B', 'Bshuf', 'Boracle'])
    w = D.n / D.n.sum()

    def agg(c):   # レース数で重み付けした RMSE
        return float(np.sqrt((w * D[c] ** 2).sum()))

    print('■ 未見コース（leave-one-course-out）の RMSE\n')
    print(f'   A  会場one-hotのみ        {agg("A"):.4f}s')
    print(f'   B  + 物理特徴             {agg("B"):.4f}s   A比 {agg("B")-agg("A"):+.4f}s')
    print(f'   対照 物理特徴をシャッフル     {agg("Bshuf"):.4f}s   A比 {agg("Bshuf")-agg("A"):+.4f}s')
    print(f'   注入 真のオフセットを渡す     {agg("Boracle"):.4f}s   A比 {agg("Boracle")-agg("A"):+.4f}s')

    win = int((D.B < D.A).sum())
    print(f'\n■ 基準1 全体で B < A            {"✅" if agg("B") < agg("A") else "❌"}')
    print(f'■ 基準2 個別で B < A が過半      {win}/{len(D)}  '
          f'{"✅" if win > len(D)/2 else "❌"}')
    print(f'■ 基準3 改善が 0.01s 以上        {agg("A")-agg("B"):+.4f}s  '
          f'{"✅" if agg("A")-agg("B") >= 0.01 else "❌"}')
    # 基準4 は分布で判定する（1回の比較では決まらない）
    real = agg('B') - agg('A')
    sh_deltas = []
    for i in range(N_SHUFFLE):
        cols = shuffled_cols(200 + i)
        r = [fit_rmse(R[R.key != k], R[R.key == k], A + cols) for k in targets]
        sh_deltas.append(float(np.sqrt((w * np.array(r) ** 2).sum())) - agg('A'))
    sh_deltas = np.array(sh_deltas)
    better = int((sh_deltas <= real).sum())
    pval = (better + 1) / (N_SHUFFLE + 1)
    print(f'■ 基準4 対照群で効果が消える      シャッフル{N_SHUFFLE}回 平均 {sh_deltas.mean():+.4f}s '
          f'(std {sh_deltas.std():.4f}) / 本物 {real:+.4f}s  P={pval:.2f}  '
          f'{"✅ 消える" if pval < 0.05 else "❌ 消えない（列を足すこと自体で動く）"}')
    print(f'■ 基準5 注入を検出できる         {"✅" if agg("A")-agg("Boracle") > 0.02 else "❌ 計測器が弱い"}')

    D['gain'] = D.A - D.B
    D = D.sort_values('gain', ascending=False)
    print('\n■ 改善が大きい/小さい組み合わせ（上下5件）')
    print(f'{"会場 馬場 距離":22s} {"n":>5s} {"A":>8s} {"B":>8s} {"改善":>8s}')
    for r in list(D.head(5).itertuples()) + list(D.tail(5).itertuples()):
        print(f'{" ".join(map(str, r.key)):22s} {r.n:5d} {r.A:8.4f} {r.B:8.4f} {r.gain:+8.4f}')


if __name__ == '__main__':
    run(sys.argv[1])
