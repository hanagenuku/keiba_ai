"""コース基礎台帳を history.db の実データから作る（参照用・予想には繋がない）。

なぜ必要か
----------
`data/course_profiles.json` の `style_advantage` は**一度もデータで検証されて
いない手書きの表**だった（生成器 `generate_style_advantage.py` は実行されると
`_generated_from_db: True` を書き込むが、本番の JSON にそのキーが無い）。
その値は `f_style_course_fit` → `f_style_total_fit` と相対列を通じて
本番134特徴量のうち最大6列を支えている。

🔑 生の「脚質別3着内率」は**馬の強さと交絡する**（ハナを切れる馬はそもそも強い）。
   そこで人気×頭数から期待される3着内率を基準線にした **市場残差** で測る。
   ＝「同じ人気の馬が、このコースでこの脚質・この枠だと、市場の想定より走るか」

⚠ 脚質の残差は**事後の脚質**（実際にどう走ったか）で測っている。事前に
   「誰が逃げるか」は当てられないので、この数字をそのまま買い目に使わないこと。
   2026-08-27: 位置取りを完璧に知れば +0.0479 だが、その94.6%は4角＝着順の
   言い換えで、過去走から予測した位置取りでは +0.0001。
   **事前に分かるのは「コース側の差」と「枠」**。
"""
import os, json, sqlite3
import numpy as np
import pandas as pd

STYLE_MAP = {'逃げ': '逃', '先行': '先', '差し': '差', '追込': '追'}
STYLES = ['逃', '先', '差', '追']
ZONES = ['内', '中', '外']
DIST_BANDS = [(0, 1400, '~1400'), (1401, 1800, '1401-1800'),
              (1801, 2200, '1801-2200'), (2201, 9999, '2201~')]
MIN_CELL_HORSES = 200      # セルとして出す下限
MIN_GROUP_HORSES = 60      # 脚質・枠の内訳を出す下限
MIN_FAV_HORSES = 30        # 1番人気の集計を出す下限


def _dist_band(x):
    for lo, hi, label in DIST_BANDS:
        if lo <= x <= hi:
            return label
    return None


def _load(db_path):
    con = sqlite3.connect(db_path)
    d = pd.read_sql("""
        SELECT h.race_id, h.date, h.racecourse rc, h.surface sf, h.distance dist,
               h.running_style rs, h.place, h.popularity pop, h.horse_num,
               h.agari3f, r.num_finishers fs
        FROM horse_history h LEFT JOIN race_history r USING(race_id)
        WHERE h.place > 0 AND h.place < 99 AND h.distance > 0
    """, con)
    con.close()
    d['fs'] = d.fs.fillna(d.groupby('race_id')['horse_num'].transform('max'))
    return d[d.fs.between(5, 18)].copy()


def _add_market_residual(d):
    """人気×頭数帯から期待される3着内率を引いた残差を付ける（交絡の除去）。"""
    d['y3'] = (d.place <= 3).astype(int)
    d['fsb'] = pd.cut(d.fs, [0, 9, 13, 18], labels=['~9', '10-13', '14-18'])
    ok = d['pop'].between(1, 18)
    tbl = d[ok].groupby(['pop', 'fsb'], observed=True)['y3'].mean()
    d['exp3'] = [tbl.get((p, b), np.nan) for p, b in zip(d['pop'], d.fsb)]
    d['resid'] = d.y3 - d.exp3
    return d


def _cell(s):
    if len(s) < MIN_CELL_HORSES:
        return None
    fav = s[s['pop'] == 1]
    out = {
        'n_horses': int(len(s)),
        'n_races': int(s.race_id.nunique()),
        'fs_median': float(s.fs.median()),
        'fav_top3': round(100 * float(fav.y3.mean()), 1) if len(fav) >= MIN_FAV_HORSES else None,
        'agari_median': round(float(s.agari3f.median()), 1) if s.agari3f.notna().sum() >= 100 else None,
    }
    for key, col in (('style', 'st'), ('zone', 'zone')):
        grp = {}
        for name, x in s.groupby(col, observed=True):
            if len(x) < MIN_GROUP_HORSES:
                continue
            grp[str(name)] = {'n': int(len(x)),
                              'top3': round(100 * float(x.y3.mean()), 1),
                              'resid_pt': round(100 * float(x.resid.mean()), 2)}
        out[key] = grp
    return out


def build_course_ledger(base_dir, out_name='course_ledger.json'):
    """history.db から台帳を作り data/<out_name> に書く。戻り値はセル辞書。"""
    db_path = os.path.join(base_dir, 'data', 'history.db')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'history.db が見つかりません: {db_path}')

    d = _add_market_residual(_load(db_path))
    d['st'] = d.rs.map(STYLE_MAP)
    d['zone'] = pd.cut(d.horse_num / d.fs, [0, 1 / 3, 2 / 3, 1.01], labels=ZONES)
    d['db'] = d.dist.map(_dist_band)

    cells = {}
    for (rc, sf), s in d.groupby(['rc', 'sf']):
        cell = _cell(s)
        if cell:
            cells[f'{rc}_{sf}'] = cell
        for band, t in s.groupby('db', observed=True):
            cell = _cell(t)
            if cell:
                cells[f'{rc}_{sf}_{band}'] = cell

    payload = {
        '_note': ('実データ由来の参照用台帳。予想には繋いでいない。'
                  'resid_pt は人気×頭数から期待される3着内率からの残差(pt)で、'
                  '＋なら市場の想定より走る。生の top3 は馬の強さと交絡するので '
                  'resid_pt を見ること。⚠ 脚質の残差は事後の脚質で測っており、'
                  '事前に誰が逃げるかは当てられない（2026-08-27）。'
                  '事前に分かるのはコース側の差と枠。'),
        '_source': f'history.db {d.date.min()}〜{d.date.max()} {len(d)}行 '
                   f'{d.race_id.nunique()}レース',
        'cells': cells,
    }
    with open(os.path.join(base_dir, 'data', out_name), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f'✅ data/{out_name}: {len(cells)} セル（{d.race_id.nunique():,}レース）')
    return cells


if __name__ == '__main__':
    import sys
    build_course_ledger(sys.argv[1] if len(sys.argv) > 1 else '.')
