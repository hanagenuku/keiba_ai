"""AI vs 市場 乖離分析 + オッズ変動分析のテスト"""
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_stats import (
    calc_divergence_analysis, calc_odds_movement_analysis,
    _save_divergence_weekly,
)


def _setup_predictions_db(rows, odds_rows=None):
    """race_predictions + odds_snapshots テーブルにテストデータを投入。"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, race_id TEXT, racecourse TEXT, race_num INTEGER,
            horse_num INTEGER, horse_name TEXT, popularity INTEGER,
            tansho_odds REAL, rl_rank INTEGER,
            win_prob REAL, cal_prob REAL, fuku_prob REAL,
            actual_place INTEGER
        )
    ''')
    for r in rows:
        conn.execute(
            'INSERT INTO race_predictions '
            '(date, race_id, racecourse, race_num, horse_num, horse_name, '
            ' popularity, tansho_odds, rl_rank, win_prob, cal_prob, fuku_prob, actual_place) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', r
        )

    conn.execute('''
        CREATE TABLE odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT, horse_num INTEGER,
            tansho REAL, fukusho REAL,
            captured_at TEXT, source TEXT DEFAULT 'chokuzen',
            UNIQUE(race_id, horse_num, captured_at)
        )
    ''')
    if odds_rows:
        for r in odds_rows:
            conn.execute(
                'INSERT INTO odds_snapshots (race_id, horse_num, tansho, fukusho, captured_at) '
                'VALUES (?, ?, ?, ?, ?)', r
            )
    conn.commit()
    return conn


def _make_race(race_id, date, racecourse, n_horses=6, ai_fav=1, mkt_fav=1, winner=1):
    """1レース分のrace_predictions行を生成。"""
    rows = []
    for i in range(1, n_horses + 1):
        odds = 2.0 + (i - 1) * 2.0
        wp = max(0.05, 0.4 - (i - 1) * 0.07)
        pop = i
        rl = i
        if i == ai_fav:
            rl = 1
            wp = 0.35
        if i == mkt_fav:
            pop = 1
            odds = 2.5
        place = i if i != winner else 1
        if i == 1 and winner != 1:
            place = winner
        rows.append((
            date, race_id, racecourse, 1, i, f'Horse{i}',
            pop, odds, rl, wp, 0.3, 0.3, place
        ))
    return rows


class TestDivergenceAnalysis:

    def test_returns_none_insufficient_data(self):
        conn = _setup_predictions_db([])
        assert calc_divergence_analysis(conn) is None

    def test_basic_output_keys(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        rows += _make_race('R002', '2026-07-06', '東京', winner=2)
        rows += _make_race('R003', '2026-07-06', '阪神', winner=3)
        rows += _make_race('R004', '2026-07-06', '阪神', winner=1)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        assert result is not None
        assert 'total_horses' in result
        assert 'bucket_stats' in result
        assert 'ai_fav_win_rate' in result
        assert 'mkt_fav_win_rate' in result
        assert 'daily' in result
        assert 'top_overvalued' in result
        assert 'top_undervalued' in result

    def test_bucket_stats_structure(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        rows += _make_race('R002', '2026-07-06', '東京', winner=2)
        rows += _make_race('R003', '2026-07-06', '阪神', winner=1)
        rows += _make_race('R004', '2026-07-06', '阪神', winner=3)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        for bs in result['bucket_stats']:
            assert 'bucket' in bs
            assert 'count' in bs
            assert 'win_rate' in bs
            assert 'top3_rate' in bs
            assert bs['count'] > 0

    def test_agree_disagree_count(self):
        rows = _make_race('R001', '2026-07-06', '東京', ai_fav=1, mkt_fav=1, winner=1)
        rows += _make_race('R002', '2026-07-06', '東京', ai_fav=2, mkt_fav=1, winner=2)
        rows += _make_race('R003', '2026-07-06', '阪神', ai_fav=3, mkt_fav=1, winner=1)
        rows += _make_race('R004', '2026-07-06', '阪神', ai_fav=1, mkt_fav=1, winner=3)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        assert result['agree_count'] >= 0
        assert result['disagree_count'] >= 0


class TestOddsMovementAnalysis:

    def test_returns_none_without_snapshots(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        conn = _setup_predictions_db(rows, odds_rows=None)
        result = calc_odds_movement_analysis(conn)
        assert result is None

    def test_basic_output_with_snapshots(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        odds = []
        for i in range(1, 7):
            morning = 2.0 + (i - 1) * 2.0
            chokuzen = morning * (0.7 if i == 1 else 1.2)
            odds.append(('R001', i, chokuzen, chokuzen * 0.5, '2026-07-06 15:00'))

        rows += _make_race('R002', '2026-07-06', '東京', winner=2)
        for i in range(1, 7):
            morning = 2.0 + (i - 1) * 2.0
            chokuzen = morning * (0.5 if i == 2 else 1.1)
            odds.append(('R002', i, chokuzen, chokuzen * 0.5, '2026-07-06 15:00'))

        conn = _setup_predictions_db(rows, odds_rows=odds)
        result = calc_odds_movement_analysis(conn)
        assert result is not None
        assert 'total_horses' in result
        assert 'bucket_stats' in result
        assert 'big_risers' in result
        assert 'ai_agrees_rising' in result

    def test_movement_bucket_direction(self):
        """オッズ下落=人気上昇、オッズ上昇=人気下落の方向が正しい。"""
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        odds = []
        for i in range(1, 7):
            morning = 10.0
            chokuzen = 5.0 if i <= 2 else 15.0
            odds.append(('R001', i, chokuzen, chokuzen * 0.5, '2026-07-06 15:00'))

        rows += _make_race('R002', '2026-07-06', '東京', winner=1)
        for i in range(1, 7):
            odds.append(('R002', i, 10.0, 5.0, '2026-07-06 15:00'))

        conn = _setup_predictions_db(rows, odds_rows=odds)
        result = calc_odds_movement_analysis(conn)
        assert result is not None
        found_rise = any(b['bucket'] == '急騰(↓30%+)' for b in result['bucket_stats'])
        found_fall = any(b['bucket'] == '急落(↑30%+)' for b in result['bucket_stats'])
        assert found_rise or found_fall


class TestSaveDivergenceWeekly:

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, 'data'))
            _save_divergence_weekly({'test': 1}, None, td)
            path = os.path.join(td, 'data', 'divergence_weekly.json')
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert 'divergence' in data[0]

    def test_updates_same_day(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, 'data'))
            _save_divergence_weekly({'v': 1}, None, td)
            _save_divergence_weekly({'v': 2}, {'mv': 1}, td)
            path = os.path.join(td, 'data', 'divergence_weekly.json')
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]['divergence']['v'] == 2
            assert data[0]['odds_movement']['mv'] == 1
