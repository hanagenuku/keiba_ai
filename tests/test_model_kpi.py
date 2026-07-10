"""市場ベースラインKPI（AI vs 市場 log-loss）のテスト"""
import json
import math
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_stats import calc_model_kpi, _save_kpi_weekly


def _setup_db(rows):
    """race_predictions テーブルにテストデータを投入したDBを返す。"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, race_id TEXT, horse_num INTEGER,
            win_prob REAL, tansho_odds REAL, actual_place INTEGER
        )
    ''')
    for r in rows:
        conn.execute(
            'INSERT INTO race_predictions (date, race_id, horse_num, win_prob, tansho_odds, actual_place) '
            'VALUES (?, ?, ?, ?, ?, ?)', r
        )
    conn.commit()
    return conn


def test_basic_kpi():
    """2頭レース1つで log-loss が正しく計算される。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.6, 2.0, 1),   # AI: 60%, market: 1/2=50%, winner
        ('2026-07-05', 'R001', 2, 0.4, 3.0, 2),   # AI: 40%, market: 1/3→normalized
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()

    assert kpi is not None
    assert kpi['total_races'] == 1
    assert kpi['total_horses'] == 2

    mkt_raw = [1/2.0, 1/3.0]
    mkt_total = sum(mkt_raw)
    mkt_p1 = mkt_raw[0] / mkt_total
    mkt_p2 = mkt_raw[1] / mkt_total

    eps = 1e-7
    ai_ll = -(math.log(0.6) + math.log(1 - 0.4)) / 2
    mkt_ll = -(math.log(mkt_p1) + math.log(1 - mkt_p2)) / 2

    assert abs(kpi['ai_logloss'] - round(ai_ll, 4)) < 0.001
    assert abs(kpi['mkt_logloss'] - round(mkt_ll, 4)) < 0.001
    assert abs(kpi['delta'] - round(ai_ll - mkt_ll, 4)) < 0.001


def test_no_data_returns_none():
    """actual_place がない場合 None を返す。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.5, 2.0, None),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()
    assert kpi is None


def test_zero_odds_excluded():
    """tansho_odds=0 の行は除外される。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.5, 0.0, 1),
        ('2026-07-05', 'R001', 2, 0.5, 3.0, 2),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()
    assert kpi is None  # 1頭だけではレースとして不完全


def test_weekly_breakdown():
    """日付ごとの週次ブレークダウンが出る。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.7, 2.0, 1),
        ('2026-07-05', 'R001', 2, 0.3, 4.0, 3),
        ('2026-07-06', 'R002', 1, 0.5, 3.0, 2),
        ('2026-07-06', 'R002', 2, 0.5, 2.5, 1),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()

    assert kpi is not None
    assert len(kpi['weekly']) == 2
    assert kpi['weekly'][0]['date'] == '2026-07-05'
    assert kpi['weekly'][1]['date'] == '2026-07-06'
    assert kpi['weekly'][0]['races'] == 1
    assert kpi['weekly'][1]['races'] == 1


def test_verdict():
    """AI が市場より良い場合 'AI優位' と判定。"""
    # AI が完璧に近い予測（勝ち馬に0.9）、市場は均等
    rows = [
        ('2026-07-05', 'R001', 1, 0.9, 5.0, 1),
        ('2026-07-05', 'R001', 2, 0.05, 5.0, 4),
        ('2026-07-05', 'R001', 3, 0.05, 5.0, 8),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()

    assert kpi['verdict'] == 'AI優位'
    assert kpi['delta'] < 0


def test_market_better_verdict():
    """市場が AI より良い場合 '市場優位' と判定。"""
    # 市場の1番人気（低オッズ）が勝ち、AI は分散
    rows = [
        ('2026-07-05', 'R001', 1, 0.2, 1.5, 1),  # market: high prob, AI: low
        ('2026-07-05', 'R001', 2, 0.4, 10.0, 5),
        ('2026-07-05', 'R001', 3, 0.4, 10.0, 3),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()

    assert kpi['verdict'] == '市場優位'
    assert kpi['delta'] > 0


def test_save_kpi_weekly_creates_file():
    """kpi_weekly.json が新規作成される。"""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'data'))
        kpi = {'weekly': [
            {'date': '2026-07-05', 'races': 5, 'horses': 50,
             'ai_logloss': 0.5, 'mkt_logloss': 0.4, 'delta': 0.1},
        ]}
        _save_kpi_weekly(kpi, d)

        path = os.path.join(d, 'data', 'kpi_weekly.json')
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]['date'] == '2026-07-05'


def test_save_kpi_weekly_updates_existing():
    """既存エントリを上書きし、新規エントリを追加する。"""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, 'data'))
        path = os.path.join(d, 'data', 'kpi_weekly.json')
        with open(path, 'w') as f:
            json.dump([
                {'date': '2026-07-05', 'races': 3, 'horses': 30,
                 'ai_logloss': 0.6, 'mkt_logloss': 0.4, 'delta': 0.2},
            ], f)

        kpi = {'weekly': [
            {'date': '2026-07-05', 'races': 5, 'horses': 50,
             'ai_logloss': 0.5, 'mkt_logloss': 0.4, 'delta': 0.1},
            {'date': '2026-07-06', 'races': 4, 'horses': 40,
             'ai_logloss': 0.45, 'mkt_logloss': 0.42, 'delta': 0.03},
        ]}
        _save_kpi_weekly(kpi, d)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]['races'] == 5  # updated
        assert data[1]['date'] == '2026-07-06'  # added


def test_single_horse_race_excluded():
    """1頭しかないレースは除外される。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.9, 1.2, 1),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()
    assert kpi is None


def test_multi_race_aggregation():
    """複数レースの平均 log-loss が算出される。"""
    rows = [
        ('2026-07-05', 'R001', 1, 0.6, 2.0, 1),
        ('2026-07-05', 'R001', 2, 0.4, 3.0, 2),
        ('2026-07-05', 'R002', 1, 0.3, 5.0, 3),
        ('2026-07-05', 'R002', 2, 0.7, 1.8, 1),
    ]
    conn = _setup_db(rows)
    kpi = calc_model_kpi(conn)
    conn.close()

    assert kpi is not None
    assert kpi['total_races'] == 2
    assert kpi['total_horses'] == 4


if __name__ == '__main__':
    test_basic_kpi()
    print('OK test_basic_kpi')
    test_no_data_returns_none()
    print('OK test_no_data_returns_none')
    test_zero_odds_excluded()
    print('OK test_zero_odds_excluded')
    test_weekly_breakdown()
    print('OK test_weekly_breakdown')
    test_verdict()
    print('OK test_verdict')
    test_market_better_verdict()
    print('OK test_market_better_verdict')
    test_save_kpi_weekly_creates_file()
    print('OK test_save_kpi_weekly_creates_file')
    test_save_kpi_weekly_updates_existing()
    print('OK test_save_kpi_weekly_updates_existing')
    test_single_horse_race_excluded()
    print('OK test_single_horse_race_excluded')
    test_multi_race_aggregation()
    print('OK test_multi_race_aggregation')
    print('\n✅ All tests passed')
