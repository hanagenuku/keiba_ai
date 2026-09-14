"""コース基礎台帳のテスト。

North Star #6 に従い、手打ち dict ではなく **実際に sqlite3 に INSERT した
history.db** を作って build_course_ledger() をエンドツーエンドで通す。
"""
import json
import os
import sqlite3

import pytest

from src.tools.build_course_ledger import (
    build_course_ledger, _dist_band, _add_market_residual, STYLE_MAP,
)
import pandas as pd


def _make_db(path, n_races=400):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE race_history (race_id TEXT PRIMARY KEY,
                   num_finishers INTEGER)""")
    con.execute("""CREATE TABLE horse_history (race_id TEXT, date TEXT,
                   racecourse TEXT, surface TEXT, distance INTEGER,
                   running_style TEXT, place INTEGER, popularity INTEGER,
                   horse_num INTEGER, agari3f REAL)""")
    styles = ['逃げ', '先行', '差し', '追込']
    for r in range(n_races):
        rid = f'2025010{r % 9 + 1}_01_{r:03d}'
        fs = 12
        con.execute('INSERT INTO race_history VALUES (?,?)', (rid, fs))
        # 🔑 決定的なデータだと基準表が同じ関係を丸ごと吸収して残差が0になる。
        #    「不人気4頭のうち誰か1頭が逃げて勝つ」形にして変動を作る。
        #    → 不人気帯の期待値は低いまま、逃げた馬だけが勝つので残差が＋に出る。
        leader = 9 + (r % 4)                      # 9〜12番人気の誰か
        for i in range(fs):
            pop = i + 1
            if pop == leader:
                st, place = '逃げ', 1
            else:
                st = '先行' if pop <= 4 else '差し' if pop <= 8 else '追込'
                place = pop if pop < leader else pop + 1
            con.execute('INSERT INTO horse_history VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (rid, '2025-01-05', '東京', '芝', 1600, st, place,
                         pop, i + 1, 34.0 + i * 0.1))
    con.commit()
    con.close()


@pytest.fixture
def base(tmp_path):
    os.makedirs(tmp_path / 'data')
    _make_db(str(tmp_path / 'data' / 'history.db'))
    return str(tmp_path)


def test_builds_and_writes_json(base):
    cells = build_course_ledger(base)
    assert '東京_芝' in cells
    out = json.load(open(os.path.join(base, 'data', 'course_ledger.json'),
                         encoding='utf-8'))
    assert out['cells'].keys() == cells.keys()
    assert '_source' in out and 'history.db' in out['_source']


def test_distance_band_is_split_out(base):
    cells = build_course_ledger(base)
    assert '東京_芝_1401-1800' in cells, '距離帯のセルが作られていない'


def test_style_residual_is_positive_for_the_favoured_style(base):
    """このフィクスチャでは逃げが必ず1着。残差は＋でなければならない。"""
    cells = build_course_ledger(base)
    st = cells['東京_芝']['style']
    assert st['逃']['resid_pt'] > 0, st
    assert st['追']['resid_pt'] < st['逃']['resid_pt']


def test_residual_controls_for_popularity():
    """人気どおりに走ったデータでは残差がほぼ0になること（交絡除去の検算）。

    残差が人気を統制できていなければ、人気順=着順のデータでも
    脚質や枠に見かけの差が出てしまう。
    """
    rows = []
    for r in range(300):
        for i in range(12):
            rows.append(dict(race_id=f'r{r}', date='2025-01-05', rc='東京', sf='芝',
                             dist=1600, rs='先行', place=i + 1, pop=i + 1,
                             horse_num=i + 1, agari3f=34.0, fs=12))
    d = _add_market_residual(pd.DataFrame(rows))
    assert abs(d.resid.mean()) < 1e-9, d.resid.mean()


def test_dist_band_boundaries():
    assert _dist_band(1400) == '~1400'
    assert _dist_band(1401) == '1401-1800'
    assert _dist_band(2200) == '1801-2200'
    assert _dist_band(2201) == '2201~'
    assert _dist_band(1800) == '1401-1800'
    assert _dist_band(1801) == '1801-2200'
    assert _dist_band(-1) is None


def test_style_map_covers_all_four():
    assert set(STYLE_MAP.values()) == {'逃', '先', '差', '追'}
