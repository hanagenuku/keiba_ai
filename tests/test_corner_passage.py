"""コーナー通過順（隊列表記）のパースを固定する。

2026-07-22に「収集のみ・構造の解釈は後回し」として追加したが、
2026-08-17に実データを見たところ**パースが壊れていた**:

    corner_pass_3: '7,'     corner_pass_4: '(*7,'   ← 途中で切れている
    充足率 20.8%

原因は2つ:
  ① `soup.get_text(' ')` が要素間に空白を入れるのに、正規表現の文字クラスに
     空白を含めていなかった → 最初の区切りで切れる
  ② ラベルの後ろを貪欲に拾っていた → 次の見出し '4コーナー' の '4' が
     通過順に混ざる

North Star #6 に従い、**本番と同じ壊れ方（要素が分割され空白が入るHTML）**で
検証する。綺麗な1文字列のフィクスチャでは①を再現できない。
"""
import sys
import os

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.jra_scraper import (  # noqa: E402
    _extract_corner_passage, parse_corner_passage,
)

# 本番と同じ形：各要素がタグで分割されており get_text(' ') で空白が入る
SPLIT_HTML = '''<html><body><div>コーナー通過順位</div>
<div>3コーナー</div><div>(1,*5)</div><div>6,10</div><div>(2,9)</div>
<div>-</div><div>(3,4)8=7</div>
<div>4コーナー</div><div>1(5,6)</div><div>-</div><div>(9,10)</div>
<div>-</div><div>(2,3,4)=8=7</div>
<div>払戻金</div></body></html>'''

# 1〜4コーナーがすべて並ぶ形（長距離戦）
ALL_CORNERS_HTML = '''<html><body><p>コーナー通過順位
1コーナー 5,1,6,10,2 2コーナー 5,1,6,2,10
3コーナー (1,*5)6,10(2,9)-(3,4)8=7 4コーナー 1(5,6)-(9,10)-(2,3,4)=8=7
タイム</p></body></html>'''

EXPECTED_3 = '(1,*5)6,10(2,9)-(3,4)8=7'
EXPECTED_4 = '1(5,6)-(9,10)-(2,3,4)=8=7'


class TestExtractCornerPassage:
    def test_survives_split_elements(self):
        """要素が分割され空白が入っても最後まで拾えること（①の回帰テスト）。"""
        r = _extract_corner_passage(BeautifulSoup(SPLIT_HTML, 'lxml'))
        assert r['corner_pass_3'] == EXPECTED_3
        assert r['corner_pass_4'] == EXPECTED_4

    def test_does_not_swallow_next_label(self):
        """'4コーナー' の '4' を3コーナーの通過順に混ぜないこと（②の回帰）。"""
        r = _extract_corner_passage(BeautifulSoup(SPLIT_HTML, 'lxml'))
        assert not r['corner_pass_3'].endswith('4'), r['corner_pass_3']
        assert r['corner_pass_3'] == EXPECTED_3

    def test_handles_all_four_corners(self):
        """1・2コーナーがあっても3角・4角を正しく選ぶこと。"""
        r = _extract_corner_passage(BeautifulSoup(ALL_CORNERS_HTML, 'lxml'))
        assert r['corner_pass_3'] == EXPECTED_3
        assert r['corner_pass_4'] == EXPECTED_4

    def test_absent_section_returns_empty(self):
        r = _extract_corner_passage(
            BeautifulSoup('<html><body><p>払戻金 単勝 380円</p></body></html>', 'lxml'))
        assert r == {}

    def test_rejects_too_short_content(self):
        """見出しだけ拾って中身が空、という壊れ方を採用しないこと。

        修正前は '7,' のような断片がそのまま保存されていた。
        """
        html = ('<html><body><p>コーナー通過順位 3コーナー 7 '
                '4コーナー 払戻金</p></body></html>')
        r = _extract_corner_passage(BeautifulSoup(html, 'lxml'))
        assert 'corner_pass_3' not in r


class TestParseCornerPassage:
    def test_expands_to_rank_and_group(self):
        p = parse_corner_passage(EXPECTED_3)
        # (1,*5) が先頭で併走2頭
        assert p[1]['rank'] == 1 and p[5]['rank'] == 1
        assert p[1]['group_size'] == 2 and p[5]['group_size'] == 2
        # 6 は単独で2番手
        assert p[6]['rank'] == 2 and p[6]['group_size'] == 1
        # 7 が最後方
        assert p[7]['rank'] == max(v['rank'] for v in p.values())

    def test_covers_every_horse_once(self):
        p = parse_corner_passage(EXPECTED_4)
        assert sorted(p) == list(range(1, 11))

    def test_ranks_are_contiguous_from_one(self):
        p = parse_corner_passage(EXPECTED_4)
        ranks = sorted({v['rank'] for v in p.values()})
        assert ranks == list(range(1, len(ranks) + 1))

    def test_group_size_matches_bracket_size(self):
        p = parse_corner_passage('1(5,6)-(9,10)-(2,3,4)=8=7')
        assert p[2]['group_size'] == 3 and p[3]['group_size'] == 3
        assert p[1]['group_size'] == 1

    def test_asterisk_does_not_become_a_horse(self):
        """'*' は先頭表示であって馬番ではない。"""
        p = parse_corner_passage('(1,*5)6')
        assert sorted(p) == [1, 5, 6]

    def test_invalid_input_returns_empty(self):
        assert parse_corner_passage('') == {}
        assert parse_corner_passage(None) == {}

    def test_ignores_out_of_range_numbers(self):
        p = parse_corner_passage('(1,99)5')
        assert 99 not in p and 1 in p and 5 in p
