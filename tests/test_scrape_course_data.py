"""コースデータ収集ツール（scrape_course_data.py）の純粋ロジック部分のテスト。

fixture HTML は2026-07-22に中山競馬場の実ページ(Shift_JIS)を実機確認して
判明した実際の構造（div.block_unit + h3見出し、captionつきtable、
course_infoプロース文）を模したもの。ネットワークアクセスを伴う _get や
scrape_all_courses 全体は対象外（_write_csv・_clean_meters等の純粋ロジックのみ）。
"""
import csv
import os
import tempfile

from bs4 import BeautifulSoup

from src.tools.scrape_course_data import (
    _clean_meters,
    _first_value,
    _parse_data_table,
    _write_csv,
    extract_course_images,
    extract_course_info,
    extract_course_tables,
    DIRT_DISTANCES,
    TURF_DISTANCES,
)

NAKAYAMA_LIKE_HTML = """
<html><body>
<div id="course_list">
<ul class="block_list type1">
<li><div class="block_unit">
<h3>コース立体図（右回り）</h3>
<div class="content"><div class="inner">
<div class="img"><img src="img/pic_course_3d.jpg" alt=""></div>
</div></div></div></li>
<li><div class="block_unit">
<h3>コース平面図（右回り）</h3>
<div class="content"><div class="inner">
<div class="img"><img src="img/pic_course_heimenzu.gif" alt=""></div>
</div></div></div></li>
<li><div class="block_unit">
<h3>芝コース高低断面図（右・内回り）</h3>
<div class="content"><div class="inner">
<div class="img"><img src="img/pic_course_turf.gif" alt=""></div>
</div></div></div></li>
<li><div class="block_unit">
<h3>芝コース高低断面図（右・外回り）</h3>
<div class="content"><div class="inner">
<div class="img"><img src="img/pic_course_turf2.gif" alt=""></div>
</div></div></div></li>
<li><div class="block_unit">
<h3>ダートコース高低断面図（右回り）</h3>
<div class="content"><div class="inner">
<div class="img"><img src="img/pic_course_durt.gif" alt=""></div>
</div></div></div></li>
</ul>

<li><div class="block_unit">
<h3>芝コース</h3>
<div class="content"><div class="inner">
<table>
<caption>芝コース：コースデータ</caption>
<thead><tr><th>直線距離</th><th>高低差</th><th>発走距離</th></tr></thead>
<tbody><tr><td>310m</td><td>5.3m</td>
<td>1,200m（外）、1,600m（外） 1,800m（内）、2,000m（内）</td></tr></tbody>
</table>

<table>
<caption>芝コース：各コースデータ</caption>
<thead><tr><th>コース</th><th>一周距離</th><th>幅員</th></tr></thead>
<tbody>
<tr><td>A</td><td>1,667.1m(内回り) 1,839.7m(外回り)</td>
<td>20〜32m(内回り) 24〜32m(外回り)</td></tr>
</tbody>
</table>
</div></div></div></li>

<li><div class="block_unit">
<h3>ダートコース</h3>
<div class="content"><div class="inner">
<table>
<caption>ダートコース：コースデータ</caption>
<thead><tr><th>一周距離</th><th>幅員</th><th>直線距離</th><th>高低差</th><th>発走距離</th></tr></thead>
<tr><td>1,493m</td><td>20〜25m</td><td>308m</td><td>4.5m</td>
<td>1,000m、1,200m、1,700m、1,800m</td></tr>
</table>
</div></div></div></li>

<div class="course_info">
<p>コースは右回りで、ダートのレースは1200メートルのみが芝スタート。</p>
<p>残り180メートルから残り70メートル地点にかけて設けられている上り坂の高低差は2.2メートル、
最大勾配の2.24%も10場最大。</p>
</div>
</div>
</body></html>
"""

# 2026-07-22、阪神競馬場の実機HTMLで確認した構造。tableのcaptionが
# 「芝コース：コースデータ」のような文言ではなく「内回り」「外回り」のみ、
# ダートtableに至ってはcaption自体が存在しない。芝コースブロックにtableが
# 2つ（内回り用・外回り用）ある点も中山と異なる
HANSHIN_LIKE_HTML = """
<html><body>
<div id="course_list">
<li><div class="block_unit">
<h3>芝コース</h3>
<div class="content"><div class="inner">
<table class="basic">
<caption class="simple title-s"><div class="inner"><div class="main">内回り</div></div></caption>
<thead><tr><th>コース</th><th>一周距離</th><th>幅員</th><th>直線距離</th><th>高低差</th><th>発走距離</th></tr></thead>
<tbody><tr><td>A</td><td>1,689m</td><td>24〜28m</td><td>356.5m</td><td>1.9m</td>
<td>1,200m、1,400m、2,000m</td></tr></tbody>
</table>
<table class="basic">
<caption class="simple title-s"><div class="inner"><div class="main">外回り</div></div></caption>
<thead><tr><th>コース</th><th>一周距離</th><th>幅員</th><th>直線距離</th><th>高低差</th><th>発走距離</th></tr></thead>
<tbody><tr><td>A</td><td>2,089m</td><td>24〜29m</td><td>473.6m</td><td>2.4m</td>
<td>1,400m、1,600m、1,800m</td></tr></tbody>
</table>
</div></div></div></li>

<li><div class="block_unit">
<h3>ダートコース</h3>
<div class="content"><div class="inner">
<table class="basic">
<tr><th>一周距離</th><th>幅員</th><th>直線距離</th><th>高低差</th><th>発走距離</th></tr>
<tr><td>1,517.6m</td><td>22〜25m</td><td>352.7m</td><td>1.6m</td>
<td>1,200m、1,400m、1,800m、2,000m、2,600m</td></tr>
</table>
</div></div></div></li>

<div class="course_info">
<p>コースは右回り、ダートは1400メートル戦と2000メートル戦が芝スタート。</p>
</div>
</div>
</body></html>
"""


class TestExtractCourseImages:
    def setup_method(self):
        self.soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        self.url = 'https://www.jra.go.jp/facilities/race/nakayama/course/'

    def test_finds_five_images(self):
        results = extract_course_images(self.soup, self.url)
        assert len(results) == 5

    def test_classifies_3d(self):
        results = extract_course_images(self.soup, self.url)
        kinds = [k for k, _, _ in results]
        assert kinds.count('3d') == 1
        assert kinds.count('course') == 1
        assert kinds.count('turf_elevation') == 2  # 内回り/外回りで2枚
        assert kinds.count('dirt_elevation') == 1

    def test_resolves_relative_url(self):
        results = extract_course_images(self.soup, self.url)
        for _, img_url, _ in results:
            assert img_url.startswith('https://www.jra.go.jp/facilities/race/nakayama/course/img/')

    def test_no_block_unit_returns_empty(self):
        soup = BeautifulSoup('<html><body>no course data</body></html>', 'lxml')
        assert extract_course_images(soup, self.url) == []


class TestParseDataTable:
    def test_parses_caption_and_rows(self):
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        table = soup.find_all('table')[0]
        caption, rows = _parse_data_table(table)
        # NFKC正規化で全角コロン(：)は半角(:)になる
        assert caption == '芝コース:コースデータ'
        assert rows[0]['直線距離'] == '310m'
        assert rows[0]['高低差'] == '5.3m'

    def test_table_without_tbody_still_parses(self):
        # ダートコースtableは実機で<tbody>を持たない構造だった
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        table = soup.find_all('table')[2]
        caption, rows = _parse_data_table(table)
        assert caption == 'ダートコース:コースデータ'
        assert rows[0]['一周距離'] == '1,493m'


class TestExtractCourseTables:
    def test_groups_by_h3_heading_nakayama_style(self):
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        tables = extract_course_tables(soup)
        # 芝コースブロック内の2tableぶんの行がまとめて入る
        assert len(tables['turf']) == 2
        assert _first_value(tables['turf'], '直線距離') == '310m'
        assert _first_value(tables['turf'], '一周距離') == '1,667.1m(内回り) 1,839.7m(外回り)'
        assert tables['dirt'][0]['一周距離'] == '1,493m'

    def test_no_tables_returns_empty_lists(self):
        soup = BeautifulSoup('<html><body>no tables</body></html>', 'lxml')
        tables = extract_course_tables(soup)
        assert tables == {'turf': [], 'dirt': []}

    def test_groups_by_h3_heading_hanshin_style(self):
        """captionが「内回り」「外回り」のみ、ダートはcaption自体が無い実機構造でも
        親h3見出し（芝コース/ダートコース）で正しくグルーピングできることを確認。
        2026-07-22、阪神の実機HTMLでcaption文字列ベースの分類が全滅した
        バグの回帰テスト。"""
        soup = BeautifulSoup(HANSHIN_LIKE_HTML, 'lxml')
        tables = extract_course_tables(soup)
        assert len(tables['turf']) == 2  # 内回り・外回りの2table
        assert _first_value(tables['turf'], '直線距離') == '356.5m'
        assert _first_value(tables['turf'], '一周距離') == '1,689m'
        assert len(tables['dirt']) == 1
        assert tables['dirt'][0]['直線距離'] == '352.7m'


class TestFirstValue:
    def test_returns_first_match(self):
        rows = [{'高低差': '5.3m'}, {'高低差': '4.5m'}]
        assert _first_value(rows, '高低差') == '5.3m'

    def test_tries_multiple_labels(self):
        rows = [{'周長': '2083m'}]
        assert _first_value(rows, '一周距離', '周長') == '2083m'

    def test_no_match_returns_none(self):
        assert _first_value([{'a': '1'}], 'b') is None


class TestCleanMeters:
    def test_simple_value(self):
        assert _clean_meters('310m') == '310'

    def test_decimal_value(self):
        assert _clean_meters('5.3m') == '5.3'

    def test_comma_separated(self):
        assert _clean_meters('1,667.1m(内回り) 1,839.7m(外回り)') == '1667.1'

    def test_none_input(self):
        assert _clean_meters(None) is None

    def test_no_number_returns_none(self):
        assert _clean_meters('不明') is None


class TestExtractCourseInfo:
    def test_turn_direction(self):
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        info = extract_course_info(soup)
        assert info['turn'] == '右回り'

    def test_turf_start_dirt_distance(self):
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        info = extract_course_info(soup)
        assert info['turf_start_distances'] == ['1200']

    def test_hill_extraction(self):
        soup = BeautifulSoup(NAKAYAMA_LIKE_HTML, 'lxml')
        info = extract_course_info(soup)
        assert info['hill']['hill_start_desc'] == '残り180m'
        assert info['hill']['hill_end_desc'] == '残り70m'
        assert info['hill']['hill_direction'] == '上り'
        assert info['hill']['elevation_diff_m'] == '2.2'

    def test_missing_course_info_div(self):
        soup = BeautifulSoup('<html><body>no course_info here</body></html>', 'lxml')
        info = extract_course_info(soup)
        assert info['turn'] is None
        assert info['hill'] is None
        assert info['turf_start_distances'] == []

    def test_multiple_turf_start_distances(self):
        """阪神は「1400メートル戦と2000メートル戦が芝スタート」のように
        1文に複数距離が入る（2000mは既定のDIRT_DISTANCES候補リストに無い距離）。"""
        soup = BeautifulSoup(HANSHIN_LIKE_HTML, 'lxml')
        info = extract_course_info(soup)
        assert info['turf_start_distances'] == ['1400', '2000']


class TestWriteCsv:
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.csv')
            _write_csv(path, [{'a': '1', 'b': '2'}], ['a', 'b'])
            with open(path, encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            assert rows == [{'a': '1', 'b': '2'}]

    def test_empty_rows_writes_only_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'out.csv')
            _write_csv(path, [], ['a', 'b'])
            with open(path, encoding='utf-8') as f:
                content = f.read()
            assert 'a,b' in content


class TestDistanceLists:
    def test_turf_distances_no_duplicates(self):
        assert len(TURF_DISTANCES) == len(set(TURF_DISTANCES))

    def test_dirt_distances_no_duplicates(self):
        assert len(DIRT_DISTANCES) == len(set(DIRT_DISTANCES))
