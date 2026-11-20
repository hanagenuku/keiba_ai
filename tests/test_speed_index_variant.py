"""スピード指数の馬場差（2026-11-18 の修正）を、本番と同じ形の SQLite で確かめる。"""
import os
import pickle
import sqlite3

import pytest

from src.features import speed_index as S


def _db(path, rows):
    """rows: (race_id, date, racecourse, distance, surface, finish_time, track_condition)。全部1着馬。"""
    c = sqlite3.connect(path)
    c.execute('CREATE TABLE race_history (race_id TEXT, date TEXT, racecourse TEXT, track_condition TEXT)')
    c.execute('CREATE TABLE horse_history (id INTEGER PRIMARY KEY, race_id TEXT, date TEXT, racecourse TEXT, '
              'distance INTEGER, surface TEXT, place INTEGER, finish_time REAL)')
    for rid, d, rc, dist, surf, ft, tc in rows:
        c.execute('INSERT INTO race_history VALUES (?,?,?,?)', (rid, d, rc, tc))
        c.execute('INSERT INTO horse_history (race_id,date,racecourse,distance,surface,place,finish_time) '
                  'VALUES (?,?,?,?,?,1,?)', (rid, d, rc, dist, surf, ft))
    c.commit()
    c.close()


def _base_rows():
    rows = []
    # 基準を作る日（芝1600=96.0 / ダ1200=72.0 / 芝2400=144.0）
    for i in range(6):
        d = f'2026-01-{10 + i:02d}'
        rows += [(f'b{i}a', d, '東京', 1600, '芝', 96.0, '良'), (f'b{i}b', d, '東京', 1200, 'ダート', 72.0, '良'),
                 (f'b{i}c', d, '東京', 2400, '芝', 144.0, '良'), (f'b{i}d', d, '東京', 1600, '芝', 96.0, '良')]
    # 対象日: 芝は1%速い・ダートは1%遅い
    d = '2026-11-01'
    rows += [('t1', d, '東京', 1600, '芝', 96.0 * 0.99, '良'), ('t2', d, '東京', 2400, '芝', 144.0 * 0.99, '良'),
             ('t3', d, '東京', 1200, 'ダート', 72.0 * 1.01, '良'), ('t4', d, '東京', 1200, 'ダート', 72.0 * 1.01, '良')]
    return rows


def test_turf_and_dirt_get_their_own_variant(tmp_path):
    p = str(tmp_path / 'history.db')
    _db(p, _base_rows())
    c = S.SpeedIndexCalculator(p)
    assert c.variant_ratio('20261101', '東京', '芝') == pytest.approx(-0.01, abs=1e-9)
    assert c.variant_ratio('2026-11-01', '東京', 'ダート') == pytest.approx(0.01, abs=1e-9)
    # 補正後は、芝もダートも「その日の平均的な勝ち時計」が基準と同じ（指数 0）になる
    assert c.calc_speed_figure(96.0 * 0.99, 1600, '芝', '良', '2026-11-01', '東京') == pytest.approx(0, abs=0.01)
    assert c.calc_speed_figure(72.0 * 1.01, 1200, 'ダート', '良', '2026-11-01', '東京') == pytest.approx(0, abs=0.01)


def test_variant_is_scaled_by_distance(tmp_path):
    p = str(tmp_path / 'history.db')
    _db(p, _base_rows())
    c = S.SpeedIndexCalculator(p)
    # 1%速い馬場は 1600m で 0.96秒・2400m で 1.44秒。どちらも指数 0 に揃う（秒で引くと揃わない）
    f16 = c.calc_speed_figure(96.0 - 0.96, 1600, '芝', '良', '2026-11-01', '東京')
    f24 = c.calc_speed_figure(144.0 - 1.44, 2400, '芝', '良', '2026-11-01', '東京')
    assert f16 == pytest.approx(0, abs=0.01) and f24 == pytest.approx(0, abs=0.01)


def test_falls_back_to_the_day_then_zero(tmp_path):
    p = str(tmp_path / 'history.db')
    rows = _base_rows() + [('x1', '2026-11-02', '東京', 1600, '芝', 96.0 * 0.98, '良'),
                           ('x2', '2026-11-02', '東京', 1600, '芝', 96.0 * 0.98, '良'),
                           ('x3', '2026-11-02', '東京', 1200, 'ダート', 72.0 * 0.98, '良')]
    _db(p, rows)
    c = S.SpeedIndexCalculator(p)
    assert c.variant_ratio('2026-11-02', '東京', 'ダート') == pytest.approx(-0.02, abs=1e-9)  # ダート1レース→合算
    assert c.variant_ratio('2026-11-03', '東京', '芝') == 0.0


def test_stale_or_old_format_cache_is_rebuilt(tmp_path):
    os.makedirs(tmp_path / 'data')
    p = str(tmp_path / 'data' / 'history.db')
    _db(p, _base_rows())
    cache = tmp_path / 'data' / 'speed_index_cache.pkl'

    old = S.SpeedIndexCalculator(p)
    old.version = 1                                 # 旧形式
    with open(cache, 'wb') as f:
        pickle.dump(old, f)
    got = S.load_speed_index_calculator(str(tmp_path))
    assert got.version == S.CACHE_VERSION and got.latest_date == '2026-11-01'

    # history.db に新しい日が増えたら、キャッシュは古いので作り直す
    c = sqlite3.connect(p)
    for rid in ('n1', 'n2'):
        c.execute('INSERT INTO race_history VALUES (?,?,?,?)', (rid, '2026-11-08', '東京', '良'))
        c.execute('INSERT INTO horse_history (race_id,date,racecourse,distance,surface,place,finish_time) '
                  "VALUES (?,'2026-11-08','東京',1600,'芝',1,95.0)", (rid,))
    c.commit()
    c.close()
    got = S.load_speed_index_calculator(str(tmp_path))
    assert got.latest_date == '2026-11-08'
    with open(cache, 'rb') as f:
        assert pickle.load(f).latest_date == '2026-11-08'   # 作り直した中身を保存している
