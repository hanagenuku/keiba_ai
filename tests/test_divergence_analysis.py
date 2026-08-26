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


class TestRankMatrixAndRoi:
    """人気帯×RL帯マトリクス + 厳密単勝回収率 — 2026-07-27導入。

    「市場1番人気なのにRL5」「市場5番人気なのにRL1」のような順位の食い違いが
    実際の勝率・回収率にどう出るかをセル単位で蓄積する機能の検証。
    回収率は「バケット平均オッズ×勝率」の概算ではなく、勝ち馬の実オッズの
    合計から厳密に計算されることを、手計算で検証可能な小さいデータで確認する。
    """

    def test_bucket_stats_include_exact_tansho_roi(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        rows += _make_race('R002', '2026-07-06', '東京', winner=2)
        rows += _make_race('R003', '2026-07-06', '阪神', winner=1)
        rows += _make_race('R004', '2026-07-06', '阪神', winner=3)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        for bs in result['bucket_stats']:
            assert 'tansho_roi' in bs
            assert bs['tansho_roi'] >= 0

    def test_tansho_roi_exact_computation(self):
        """回収率が勝ち馬の実オッズから厳密計算されることを手計算で検証。

        4レース×6頭=24頭。_make_raceの構造では
        winner=1のレースは1番人気(odds2.5)が勝ち、winner=2のレースは
        2番馬(odds4.0)が勝つ。全マトリクスセルの回収率合計を検算する。
        """
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)   # 勝ち馬odds=2.5
        rows += _make_race('R002', '2026-07-06', '東京', winner=2)  # 勝ち馬odds=4.0
        rows += _make_race('R003', '2026-07-06', '阪神', winner=1)  # 勝ち馬odds=2.5
        rows += _make_race('R004', '2026-07-06', '阪神', winner=3)  # 勝ち馬odds=6.0
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)

        # 全セルの (roi × n) の合計 = 総払戻。総投資=24頭。
        # 総払戻 = 2.5 + 4.0 + 2.5 + 6.0 = 15.0 → 全体回収率 15/24 = 62.5%
        matrix = result['rank_matrix']
        assert matrix, "rank_matrixが空"
        total_n = sum(c['count'] for c in matrix)
        total_payout = sum(c['tansho_roi'] / 100 * c['count'] for c in matrix)
        assert total_n == 24
        assert abs(total_payout - 15.0) < 0.1, (
            f"総払戻が勝ち馬実オッズ合計(15.0)と一致しない: {total_payout}"
        )

    def test_rank_matrix_cell_structure_and_bands(self):
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        rows += _make_race('R002', '2026-07-06', '東京', winner=2)
        rows += _make_race('R003', '2026-07-06', '阪神', winner=1)
        rows += _make_race('R004', '2026-07-06', '阪神', winner=3)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        valid_bands = {'1', '2-3', '4-5', '6-9', '10+'}
        for cell in result['rank_matrix']:
            assert set(cell.keys()) == {
                'pop_band', 'rl_band', 'count', 'win_rate', 'top3_rate', 'tansho_roi'}
            assert cell['pop_band'] in valid_bands
            assert cell['rl_band'] in valid_bands
            assert cell['count'] > 0

    def test_rank_matrix_divergence_cell(self):
        """「市場5番人気×RL1」の馬が正しいセルに入ることを確認。

        ai_fav=5: 5番馬(人気5, odds10.0)がRL1になる → pop_band='4-5', rl_band='1'
        のセルが存在し、そのレースでこの馬が勝てば回収率にodds10.0が計上される。
        """
        rows = _make_race('R001', '2026-07-06', '東京', ai_fav=5, winner=5)
        rows += _make_race('R002', '2026-07-06', '東京', winner=1)
        rows += _make_race('R003', '2026-07-06', '阪神', winner=1)
        rows += _make_race('R004', '2026-07-06', '阪神', winner=2)
        conn = _setup_predictions_db(rows)
        result = calc_divergence_analysis(conn)
        cell = next((c for c in result['rank_matrix']
                     if c['pop_band'] == '4-5' and c['rl_band'] == '1'), None)
        assert cell is not None, "市場4-5人気×RL1のセルが存在しない"
        assert cell['count'] == 1
        assert cell['win_rate'] == 100.0
        # 5番馬のodds = 2.0 + (5-1)*2.0 = 10.0 → 1頭賭けて10.0倍払戻 = 1000%
        assert cell['tansho_roi'] == 1000.0


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

    def test_movement_bucket_aligns_with_app_badge_thresholds(self):
        """急騰/上昇バケットは index.html updateOddsAndEV() の急騰/上昇バッジ判定
        条件（%下落かつオッズ差の絶対値）と完全に一致させている。%だけ見れば
        同じ範囲でも、オッズ差の絶対値が小さくバッジが出ない馬は
        「相当(バッジ対象外)」に分類され、混同されないことを確認する
        （2026-07-22、集計側の閾値がアプリのバッジ条件とズレていた是正の回帰テスト）。
        """
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        # ⚠ 単勝オッズは最低1.0倍。1.0未満を置くと本番ではあり得ない値になり、
        #    集計側のガード(>= 1.0)に弾かれる（North Star #6: 本番と同じ値域で書く）
        odds = [
            # 馬1 morning=2.5(mkt_fav補正) → 1.6: -36%だがabs差0.9(<3.0) → 急騰相当
            ('R001', 1, 1.6, 0.8, '2026-07-06 15:00'),
            # 馬6 morning=12.0 → 6.0: -50%・abs差6.0(>=3.0) → 急騰(バッジ有)
            ('R001', 6, 6.0, 3.0, '2026-07-06 15:00'),
            # 馬3 morning=6.0 → 4.5: -25%だがabs差1.5(<2.0) → 上昇相当
            ('R001', 3, 4.5, 2.25, '2026-07-06 15:00'),
            # 馬4 morning=8.0 → 6.0: -25%・abs差2.0(>=2.0) → 上昇(バッジ有)
            ('R001', 4, 6.0, 3.0, '2026-07-06 15:00'),
            # 馬2,5 変化なし → 横ばい
            ('R001', 2, 4.0, 2.0, '2026-07-06 15:00'),
            ('R001', 5, 10.0, 5.0, '2026-07-06 15:00'),
        ]
        rows += _make_race('R002', '2026-07-06', '東京', winner=1)
        for i, m in {1: 2.5, 2: 4.0, 3: 6.0, 4: 8.0, 5: 10.0, 6: 12.0}.items():
            odds.append(('R002', i, m, m * 0.5, '2026-07-06 15:00'))  # 全馬変化なし（件数確保用）

        conn = _setup_predictions_db(rows, odds_rows=odds)
        result = calc_odds_movement_analysis(conn)
        assert result is not None
        buckets = {b['bucket']: b['count'] for b in result['bucket_stats']}
        assert buckets.get('急騰(↓30%+)') == 1
        assert buckets.get('急騰相当(バッジ対象外)') == 1
        assert buckets.get('上昇(↓15-30%)') == 1
        assert buckets.get('上昇相当(バッジ対象外)') == 1

    def test_impossible_odds_below_1x_are_excluded(self):
        """🔴 1.0倍未満は単勝としてあり得ない。集計に入れてはいけない。

        2026-06〜07（`_sanitize_win_odds` 導入前）に 0.1倍のゴミ値が
        race_predictions に43行混入しており、そのまま集計すると
        「変動 +76000%」のような行が big_fallers に出ていた（2026-08-26発覚）。
        """
        rows = _make_race('R001', '2026-07-06', '東京', winner=1)
        # 馬2 の朝オッズを 0.1倍のゴミ値に差し替える
        rows = [list(r) for r in rows]
        for r in rows:
            if r[4] == 2:
                r[7] = 0.1
        rows = [tuple(r) for r in rows]
        odds = [('R001', i, 5.0, 2.5, '2026-07-06 15:00') for i in range(1, 7)]
        rows += _make_race('R002', '2026-07-06', '東京', winner=1)
        odds += [('R002', i, 5.0, 2.5, '2026-07-06 15:00') for i in range(1, 7)]

        conn = _setup_predictions_db(rows, odds_rows=odds)
        result = calc_odds_movement_analysis(conn)
        assert result is not None
        # 12頭中、0.1倍の1頭が除外されて11頭
        assert result['total_horses'] == 11, (
            f"1.0倍未満が除外されていない: {result['total_horses']}")
        # 変動率が非現実的な行が出ていないこと
        for r in result.get('big_fallers', []) + result.get('big_risers', []):
            assert abs(r['change']) < 10000, f'非現実的な変動が残っている: {r}'


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
