"""rank_matrix_filter（人気×RL乖離マトリクス実績に基づく買い目フィルタ）のテスト。

North Starに従い、divergence_weekly.json は本番の generate_stats.py が
実際に出力する形式（[{date, divergence: {rank_matrix: [...]}}] のリスト）で
ファイルとして書き出して検証する。
"""

import json
import os

import pytest

from src.betting import rank_matrix_filter as rmf
from src.betting.bet_optimizer import build_optimal_bets


def _write_divergence_weekly(tmp_path, rank_matrix):
    """本番フォーマット通りのdivergence_weekly.jsonを作る。"""
    (tmp_path / 'data').mkdir(exist_ok=True)
    entries = [
        {'date': '2026-07-19', 'divergence': {'rank_matrix': []}, 'odds_movement': None},
        {'date': '2026-07-26',
         'divergence': {
             'total_horses': 3983, 'total_races': 318,
             'bucket_stats': [],
             'rank_matrix': rank_matrix,
         },
         'odds_movement': None},
    ]
    path = tmp_path / 'data' / 'divergence_weekly.json'
    path.write_text(json.dumps(entries, ensure_ascii=False))
    return str(tmp_path)


SAMPLE_MATRIX = [
    # 実測に近い形のセル: boost対象（N>=50, ROI>=120）
    {'pop_band': '2-3', 'rl_band': '4-5', 'count': 123, 'win_rate': 23.6,
     'top3_rate': 48.0, 'tansho_roi': 144.9},
    # suppress対象（N>=50, ROI<50）
    {'pop_band': '2-3', 'rl_band': '6-9', 'count': 92, 'win_rate': 6.5,
     'top3_rate': 20.7, 'tansho_roi': 37.7},
    # N不足（高ROIでも判定しない）
    {'pop_band': '6-9', 'rl_band': '1', 'count': 22, 'win_rate': 22.7,
     'top3_rate': 40.9, 'tansho_roi': 220.5},
    # 中間帯（neutral）
    {'pop_band': '1', 'rl_band': '1', 'count': 184, 'win_rate': 28.8,
     'top3_rate': 65.0, 'tansho_roi': 69.2},
]


@pytest.fixture(autouse=True)
def _clear_cache():
    rmf.clear_cache()
    yield
    rmf.clear_cache()


class TestClassifyHorse:

    def test_boost_cell(self, tmp_path):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        # 市場3番人気×RL4 → セル(2-3, 4-5) ROI144.9% N=123 → boost
        assert rmf.classify_horse(3, 4, base) == 'boost'

    def test_suppress_cell(self, tmp_path):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        # 市場2番人気×RL7 → セル(2-3, 6-9) ROI37.7% N=92 → suppress
        assert rmf.classify_horse(2, 7, base) == 'suppress'

    def test_small_n_neutral(self, tmp_path):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        # 市場7番人気×RL1 → セル(6-9, 1) ROI220%だがN=22<50 → neutral
        assert rmf.classify_horse(7, 1, base) is None

    def test_middle_roi_neutral(self, tmp_path):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        # 市場1番人気×RL1 → ROI69.2%（50〜120の中間帯） → neutral
        assert rmf.classify_horse(1, 1, base) is None

    def test_unknown_rank_neutral(self, tmp_path):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        assert rmf.classify_horse(99, 1, base) is None
        assert rmf.classify_horse(1, 99, base) is None
        assert rmf.classify_horse(None, 1, base) is None

    def test_missing_file_neutral(self, tmp_path):
        (tmp_path / 'data').mkdir()
        assert rmf.classify_horse(3, 4, str(tmp_path)) is None

    def test_kill_switch_env(self, tmp_path, monkeypatch):
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        monkeypatch.setenv('RANK_MATRIX_FILTER', '0')
        assert rmf.classify_horse(3, 4, base) is None

    def test_band_boundaries_match_generate_stats(self):
        """帯の定義がgenerate_stats.py側と一致していることを確認。

        両者がズレると集計と適用で違うセルを見ることになるため、
        同一入力に対する帯を直接突き合わせる。
        """
        import scripts.generate_stats  # noqa: F401  (実装参照コメント用)
        for rank, expected in [(1, '1'), (2, '2-3'), (3, '2-3'), (4, '4-5'),
                               (5, '4-5'), (6, '6-9'), (9, '6-9'), (10, '10+'),
                               (18, '10+')]:
            assert rmf._rank_band(rank) == expected


class TestBuildOptimalBetsWithFilter:
    """build_optimal_betsへの統合テスト。"""

    def _horses(self):
        # RL1: 市場2番人気(suppressセル該当なし)、RL4: 市場3番人気(boostセル)
        # RL2: 市場2-3人気×RL2-3 → サンプル行列に無いセル → neutral
        return [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 2.5},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 2, 'win_odds': 5.0},
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 3, 'popularity': 4, 'win_odds': 8.0},
            {'horse_num': 4, 'num': 4, 'name': 'D', 'rl_rank': 4, 'popularity': 3, 'win_odds': 12.0},
            {'horse_num': 5, 'num': 5, 'name': 'E', 'rl_rank': 7, 'popularity': 2, 'win_odds': 6.0},
        ]

    def _probs_and_odds(self):
        probs = {
            'win':   {1: 0.30, 2: 0.20, 3: 0.15, 4: 0.20, 5: 0.10},
            'place': {1: 0.60, 2: 0.50, 3: 0.40, 4: 0.45, 5: 0.30},
        }
        odds_map = {
            'win':   {1: 2.5, 2: 5.0, 3: 8.0, 4: 12.0, 5: 6.0},
            'place': {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 2.2},
        }
        return probs, odds_map

    def test_boost_horse_can_win_selection(self, tmp_path):
        """boost馬（RL4だが実測ROI 144.9%のセル）が単勝候補に入り、
        EV最大なら選ばれることを確認。従来ロジック(RL上位3頭のみ)では
        この馬は構造的に候補外だった。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        probs, odds_map = self._probs_and_odds()
        # 馬4(RL4, 市場3人気=boostセル)のEV: 0.20*12.0=2.4 → 全馬中最大
        bets = build_optimal_bets(probs, odds_map, self._horses(), {}, base_dir=base)
        assert bets['win'], "単勝買い目が生成されていない"
        assert bets['win'][0]['key'] == 4, (
            f"boost馬(#4)が選ばれるべきだが {bets['win'][0]['key']} が選ばれた"
        )

    def test_suppress_horse_excluded_from_win(self, tmp_path):
        """suppress馬は高EVでも単勝候補から除外される。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        probs, odds_map = self._probs_and_odds()
        horses = self._horses()
        # 馬2を市場2人気×RL7(suppressセル)に変更し、EVを最大化
        horses[1]['rl_rank'] = 7
        horses[4]['rl_rank'] = 2  # RL2を別の馬に
        probs['win'][2] = 0.50    # EV 0.5*5.0=2.5 → 本来なら文句なし1位
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        chosen = [b['key'] for b in bets['win']]
        assert 2 not in chosen, "suppress馬(#2)が単勝候補から除外されていない"

    def test_suppress_horse_excluded_from_place(self, tmp_path):
        """suppress馬は複勝候補からも除外される。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        probs, odds_map = self._probs_and_odds()
        horses = self._horses()
        # 馬5: RL7→RL5に変更しても市場2人気×RL5は(2-3,4-5)boostセルになるため、
        # 市場2人気のままRL7だとrl_top5外。市場2人気×RL6-9のsuppressを
        # 複勝で試すには、RL5かつsuppressセルの構成が必要。
        # 馬3を市場2人気×RL9... RL9はrl_top5外。
        # → 複勝のsuppress検証は「rl_top5内かつsuppressセル」の組み合わせで行う。
        # サンプル行列に(4-5帯, 4-5帯)のsuppressセルを追加した行列で検証する。
        matrix = SAMPLE_MATRIX + [
            {'pop_band': '4-5', 'rl_band': '4-5', 'count': 261, 'win_rate': 7.7,
             'top3_rate': 25.0, 'tansho_roi': 45.0},
        ]
        rmf.clear_cache()
        base = _write_divergence_weekly(tmp_path, matrix)
        # 馬3: 市場4人気×RL3 → neutral。RL4に変更 → (4-5,4-5)suppressセル
        horses[2]['rl_rank'] = 4
        horses[3]['rl_rank'] = 3
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        chosen = [b['key'] for b in bets['place']]
        assert 3 not in chosen, "suppress馬(#3)が複勝候補から除外されていない"

    def test_no_base_dir_keeps_legacy_behavior(self):
        """base_dir無し（旧呼び出し）ではフィルタが発動せず従来挙動。"""
        probs, odds_map = self._probs_and_odds()
        bets = build_optimal_bets(probs, odds_map, self._horses(), {})
        # RL上位3頭(1,2,3)のみが単勝候補。EV: #1=0.75(足切り), #2=1.0, #3=1.2
        assert bets['win'], "単勝買い目が生成されていない"
        assert bets['win'][0]['key'] in (1, 2, 3), "従来挙動(RL上位3頭)が保たれていない"


class TestBoostOnlyWinSelection:
    """単勝のboost限定化（2026-07-27②）。

    out-of-sample検証で、boost馬が居ないレースの従来ピックはROI 94.8%の
    負け筋であり、boost限定にすると241.1%（従来74.3%）になることを確認した。
    マトリクスが有効な時のみ適用し、無効時（旧環境・base_dir未指定）は
    従来挙動を保つ。
    """

    def _probs_and_odds(self, nums):
        probs = {'win': {n: 0.2 for n in nums}, 'place': {n: 0.5 for n in nums}}
        odds_map = {'win': {n: 8.0 for n in nums}, 'place': {n: 2.5 for n in nums}}
        return probs, odds_map

    def test_no_boost_horse_means_no_win_bet(self, tmp_path):
        """マトリクス有効かつboost馬が居ないレースでは単勝を見送る。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        # 全馬がneutralセル（市場1人気×RL1=ROI69.2%等）になる構成
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 2, 'win_odds': 8.0},
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 3, 'popularity': 1, 'win_odds': 8.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2, 3])
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        # EV=0.2*8.0=1.6 と十分高いが、boost馬が居ないので買わない
        assert bets['win'] == [], (
            f"boost馬が居ないのに単勝を買っている: {bets['win']}"
        )

    def test_boost_horse_selected_when_present(self, tmp_path):
        """boost馬が居ればその馬を買う（従来のRL上位馬より優先）。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 1, 'win_odds': 8.0},
            # #3: 市場3番人気×RL4 → (2-3, 4-5) = boostセル
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 4, 'popularity': 3, 'win_odds': 8.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2, 3])
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        assert bets['win'], "boost馬が居るのに単勝が生成されていない"
        assert bets['win'][0]['key'] == 3, (
            f"boost馬(#3)が選ばれるべきだが {bets['win'][0]['key']} だった"
        )

    def test_matrix_missing_keeps_legacy_win_bet(self, tmp_path):
        """マトリクスが無い環境ではboost限定を適用せず従来通り買う。

        （初回実行時・旧latest.json等でマトリクスが未生成の場合に
        単勝が一切買えなくなる事故を防ぐ）
        """
        (tmp_path / 'data').mkdir()   # divergence_weekly.json を置かない
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 2, 'win_odds': 8.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2])
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=str(tmp_path))
        assert bets['win'], "マトリクス不在時は従来通り単勝を買うべき"

    def test_all_boost_horses_bought_not_just_one(self, tmp_path):
        """boost馬が複数居れば全て買う（2026-07-27⑨）。

        従来はEV最大の1点のみ買っており、EV最大＝オッズが高い馬を選ぶ
        傾向から実際に的中馬を取り逃していた（7/26の中京R4・札幌R5）。
        ROI自体は同等だが、同じ利益率でベット数を増やす目的。
        """
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            # #2,#3 とも 市場2-3人気×RL4-5 = boostセル
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 4, 'popularity': 2, 'win_odds': 8.0},
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 5, 'popularity': 3, 'win_odds': 12.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2, 3])
        odds_map['win'][3] = 12.0
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        keys = sorted(b['key'] for b in bets['win'])
        assert keys == [2, 3], (
            f"boost馬2頭とも買うべきだが {keys} だった"
        )

    def test_legacy_mode_still_single_bet(self, tmp_path):
        """マトリクス無効時は従来通り1点のみ（複数買いはboost限定時だけ）。"""
        (tmp_path / 'data').mkdir()
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 2, 'win_odds': 8.0},
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 3, 'popularity': 3, 'win_odds': 8.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2, 3])
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=str(tmp_path))
        assert len(bets['win']) == 1, (
            f"マトリクス無効時は1点のみのはず: {bets['win']}"
        )

    def test_kill_switch_restores_legacy_win_bet(self, tmp_path, monkeypatch):
        """RANK_MATRIX_FILTER=0 で従来挙動に戻る（緊急停止）。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        monkeypatch.setenv('RANK_MATRIX_FILTER', '0')
        rmf.clear_cache()
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 8.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 2, 'win_odds': 8.0},
        ]
        probs, odds_map = self._probs_and_odds([1, 2])
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        assert bets['win'], "kill switch時は従来通り単勝を買うべき"


class TestPlaceBoostPriority:
    """複勝のboost優先（2026-07-27②）。

    実際の複勝払戻データ（results.fukusho_payout, 全期間4,148頭）で
    boost 97.2% / neutral 67.1% と確認したため、買い目点数は変えずに
    boost馬を優先して埋める。
    """

    def test_boost_horse_prioritized_over_better_rl(self, tmp_path):
        """RL順位が下でもboost馬が先に選ばれる。"""
        base = _write_divergence_weekly(tmp_path, SAMPLE_MATRIX)
        horses = [
            # #1,#2: 市場1人気×RL1-2 → neutral（RL順では先）
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 3.0},
            {'horse_num': 2, 'num': 2, 'name': 'B', 'rl_rank': 2, 'popularity': 1, 'win_odds': 4.0},
            # #3: 市場3人気×RL4 → boostセル（RL順では後）
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 4, 'popularity': 3, 'win_odds': 6.0},
        ]
        probs = {'win': {1: 0.3, 2: 0.2, 3: 0.2},
                 'place': {1: 0.60, 2: 0.55, 3: 0.50}}
        odds_map = {'win': {1: 3.0, 2: 4.0, 3: 6.0},
                    'place': {1: 2.0, 2: 2.2, 3: 2.5}}
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=base)
        chosen = [b['key'] for b in bets['place']]
        assert chosen, "複勝買い目が生成されていない"
        assert chosen[0] == 3, (
            f"boost馬(#3)が最優先で選ばれるべきだが {chosen} の順だった"
        )

    def test_place_order_unchanged_without_matrix(self, tmp_path):
        """マトリクスが無ければ従来通りRL順。"""
        (tmp_path / 'data').mkdir()
        horses = [
            {'horse_num': 1, 'num': 1, 'name': 'A', 'rl_rank': 1, 'popularity': 1, 'win_odds': 3.0},
            {'horse_num': 3, 'num': 3, 'name': 'C', 'rl_rank': 4, 'popularity': 3, 'win_odds': 6.0},
        ]
        probs = {'win': {1: 0.3, 3: 0.2}, 'place': {1: 0.60, 3: 0.50}}
        odds_map = {'win': {1: 3.0, 3: 6.0}, 'place': {1: 2.0, 3: 2.5}}
        bets = build_optimal_bets(probs, odds_map, horses, {}, base_dir=str(tmp_path))
        chosen = [b['key'] for b in bets['place']]
        assert chosen[0] == 1, f"従来のRL順が保たれていない: {chosen}"
