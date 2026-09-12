"""ワイド盤・複勝盤のパースと恒等式（dprime/CRITERIA.md Step 1〜2）の検証。

Step 0（到達）は 2026-09-12 の probe-wide-odds 実行で通過している:
  ワイド盤 = CNAME の pw15**5**ou…、「ワイドオッズ（馬番順）」で
  C(n,2) 組が範囲表記（例 '5.9-6.6'）で並ぶ。

⚠ 実機の生HTMLはまだ1レースぶんの断片しか見ていない（軸馬が caption に
   あるのか th にあるのかは未確認）。そのため:
   - パーサは見出しから軸馬を引き、取れない時だけ推測にフォールバックする
   - どちらで決めたかを必ず返す（推測が混ざったまま数字を出さないため）
   - 本テストは header / inferred の両方を固定する
   North Star #6 に従い、実機の構造が判明したらフィクスチャを実機側へ寄せること。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.betting.wide_book import (
    parse_odds_cell, parse_fukusho_book, parse_wide_book, check_books,
    implied_top3, identity_residual, book_sum, MIN_FIELD_SIZE,
)


def _fuku_html(rows):
    trs = ''.join(
        f'<tr><td>{(n + 1) // 2}</td><td>{n}</td><td>ウマ{n}</td>'
        f'<td>{tan}</td><td>{fk}</td><td>牡3</td></tr>'
        for n, tan, fk in rows)
    return ('<html><body><table>'
            '<tr><th>枠</th><th>馬番</th><th>馬名</th>'
            '<th>単勝</th><th>複勝(2着払い)</th><th>性齢</th></tr>'
            f'{trs}</table></body></html>')


def _wide_html(pairs, n, with_caption):
    """軸馬ごとに1テーブル。各行は [相手の馬番, オッズ範囲] しか持たない実機の形。"""
    parts = []
    for a in range(1, n + 1):
        rows = ''.join(f'<tr><td>{b}</td><td>{lo}-{hi}</td></tr>'
                       for (x, b), (lo, hi) in sorted(pairs.items()) if x == a)
        if not rows:
            continue
        cap = f'<caption>{a}番</caption>' if with_caption else ''
        parts.append(f'<table>{cap}{rows}</table>')
    return '<html><body>' + ''.join(parts) + '</body></html>'


def _flat_pairs(n, odds=10.0):
    return {(a, b): (odds, odds)
            for a in range(1, n + 1) for b in range(a + 1, n + 1)}


class TestParseOddsCell:
    def test_range(self):
        assert parse_odds_cell('5.9-6.6') == (5.9, 6.6)
        assert parse_odds_cell('98.6 〜 103.3') == (98.6, 103.3)

    def test_single_becomes_degenerate_range(self):
        # 単勝や馬連は単値。min==max として同じ形で扱えるようにする
        assert parse_odds_cell('16.7') == (16.7, 16.7)

    def test_junk(self):
        assert parse_odds_cell('') is None
        assert parse_odds_cell('馬名') is None
        assert parse_odds_cell('---') is None


class TestParseFukushoBook:
    def test_reads_by_header_not_by_position(self):
        html = _fuku_html([(1, '2.1', '1.1-1.3'), (2, '5.0', '1.8-2.4')])
        assert parse_fukusho_book(html) == {1: (1.1, 1.3), 2: (1.8, 2.4)}

    def test_column_shift_is_survived(self):
        # 2026-08-03③（結果ページの列がまるごとズレて1ヶ月気づかなかった）と
        # 同じ事故を、見出し引きで防げていることを固定する
        html = ('<html><body><table>'
                '<tr><th>馬番</th><th>馬名</th><th>複勝</th></tr>'
                '<tr><td>7</td><td>ウマ7</td><td>2.0-2.5</td></tr>'
                '</table></body></html>')
        assert parse_fukusho_book(html) == {7: (2.0, 2.5)}


class TestParseWideBook:
    def test_axis_from_caption(self):
        n = 8
        pairs = _flat_pairs(n)
        got, src = parse_wide_book(_wide_html(pairs, n, with_caption=True))
        assert got == pairs
        assert src['header'] == n - 1 and src['inferred'] == 0

    def test_axis_inferred_when_no_header(self):
        n = 8
        pairs = _flat_pairs(n)
        got, src = parse_wide_book(_wide_html(pairs, n, with_caption=False))
        assert got == pairs
        # 推測で決めたことが呼び出し側から見えること（黙って推測しない）
        assert src['inferred'] == n - 1 and src['header'] == 0

    def test_pair_key_is_order_independent(self):
        n = 8
        got, _ = parse_wide_book(_wide_html(_flat_pairs(n), n, with_caption=True))
        assert all(a < b for a, b in got)


class TestStep1BookHealth:
    def _books(self, n, fuku_o, wide_o):
        fuku = {i: (fuku_o, fuku_o) for i in range(1, n + 1)}
        pairs = _flat_pairs(n, wide_o)
        return fuku, pairs

    def test_healthy_books_pass(self):
        n = 12
        # Σ(1/o) が理論値ちょうどになるオッズを逆算して作る
        fuku_o = n / (3.0 / 0.80)
        wide_o = (n * (n - 1) / 2) / (3.0 / 0.775)
        fuku, pairs = self._books(n, fuku_o, wide_o)
        c = check_books(fuku, pairs, n)
        assert c['field_ok'] and c['pairs_complete'] and c['fuku_ok'] and c['wide_ok']

    def test_broken_board_is_caught(self):
        # 2026-08-15 C-2「全16頭が2.1倍」と同型。1頭ずつ見ても異常に見えないが
        # Σ で一発で分かる
        n = 12
        fuku, pairs = self._books(n, 2.1, 2.1)
        c = check_books(fuku, pairs, n)
        assert not c['fuku_ok'] and not c['wide_ok']

    def test_missing_pairs_is_caught(self):
        n = 12
        fuku_o = n / (3.0 / 0.80)
        fuku, pairs = self._books(n, fuku_o, 5.0)
        pairs.pop((1, 2))
        c = check_books(fuku, pairs, n)
        assert not c['pairs_complete']
        assert c['n_pairs'] == c['want_pairs'] - 1

    def test_seven_horse_race_is_out_of_scope(self):
        # 5〜7頭は複勝が2着払い。混ぜると実在しないズレを測ることになる
        n = 7
        fuku, pairs = self._books(n, 3.0, 5.0)
        assert not check_books(fuku, pairs, n)['field_ok']
        assert MIN_FIELD_SIZE == 8


class TestStep2Identity:
    def test_sum_q_is_exactly_three(self):
        """Σ_A q_A = 3 は代数的に自動成立する。ズレたら実装が壊れている。"""
        n = 10
        pairs = {(a, b): (float(a + b), float(a + b) + 1.0)
                 for a in range(1, n + 1) for b in range(a + 1, n + 1)}
        fuku = {i: (float(i) + 1.0, float(i) + 2.0) for i in range(1, n + 1)}
        for how in ('min', 'mid', 'max'):
            res = implied_top3(fuku, pairs, how=how)
            assert abs(res['sum_q'] - 3.0) < 1e-9, how
            assert abs(res['sum_p'] - 3.0) < 1e-9, how
            assert identity_residual(pairs, how=how) < 1e-9

    def test_consistent_market_gives_zero_divergence(self):
        """ワイドが『各馬の3着内確率の積に比例』なら乖離はゼロ近くになる。

        計測器がゼロを返せることの確認（対照群）。ここが非ゼロなら
        後で観測する乖離は全部実装の産物になる。
        """
        n = 12
        # 各馬に真の3着内確率を割り当て、合計3に正規化
        raw = [0.1 + 0.05 * i for i in range(n)]
        s = sum(raw)
        p = [x / s * 3.0 for x in raw]
        fuku = {i + 1: (1.0 / p[i], 1.0 / p[i]) for i in range(n)}
        # P(A,B) ∝ p_A * p_B（独立近似）。正規化は implied_top3 側がやる
        pairs = {}
        for a in range(n):
            for b in range(a + 1, n):
                v = p[a] * p[b]
                pairs[(a + 1, b + 1)] = (1.0 / v, 1.0 / v)
        res = implied_top3(fuku, pairs, how='mid')
        # 独立近似なので厳密ゼロにはならないが、桁として小さいこと
        assert res['max_abs_d'] < 0.30, res['max_abs_d']

    def test_divergence_is_detected_when_one_horse_is_mispriced(self):
        """ワイド側だけ1頭を割安にすると、その馬の d が正で最大になる。"""
        n = 12
        p = [3.0 / n] * n
        fuku = {i + 1: (1.0 / p[i], 1.0 / p[i]) for i in range(n)}
        pairs = {}
        for a in range(n):
            for b in range(a + 1, n):
                v = p[a] * p[b]
                if 0 in (a, b):        # 馬番1 だけワイド側で厚く買われている
                    v *= 2.0
                pairs[(a + 1, b + 1)] = (1.0 / v, 1.0 / v)
        res = implied_top3(fuku, pairs, how='mid')
        assert res['d'][1] == max(res['d'].values())
        assert res['d'][1] > 0
        assert abs(res['sum_q'] - 3.0) < 1e-9   # 検算は壊れていない

    def test_min_max_mid_are_reported_separately(self):
        """範囲表記の扱いが結論を作らないよう、3通りとも出せること。"""
        n = 8
        pairs = {(a, b): (10.0, 20.0)
                 for a in range(1, n + 1) for b in range(a + 1, n + 1)}
        fuku = {i: (2.0, 4.0) for i in range(1, n + 1)}
        vals = {how: book_sum(pairs, how) for how in ('min', 'mid', 'max')}
        assert vals['min'] > vals['mid'] > vals['max']


class TestProbeWiring:
    def test_probe_imports_and_calls_step12(self):
        import re
        src = open(os.path.join(os.path.dirname(__file__), '..',
                                'scripts', 'probe_wide_odds.py'),
                   encoding='utf-8').read()
        assert 'from src.betting.wide_book import' in src
        assert 'run_step12(sess' in src, 'Step 1/2 が main から呼ばれていない'
        # 到達しただけで「使える」と書かせない歯止めが残っていること
        assert 'Step 3' in src

    def test_probe_stays_read_only(self):
        yml = open(os.path.join(os.path.dirname(__file__), '..', '.github',
                                'workflows', 'probe-wide-odds.yml'),
                   encoding='utf-8').read()
        assert 'contents: read' in yml
        assert 'git add' not in yml and 'git push' not in yml
