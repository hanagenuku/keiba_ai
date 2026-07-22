"""コースデータ収集ツール（scrape_course_data.py）の純粋ロジック部分のテスト。

ネットワークアクセスを伴う _scrape_jra_venue / _scrape_umasiru_venue /
scrape_all_courses はこのテスト環境では検証できないため対象外。
正規表現抽出・画像分類・CSV書き出しのみを対象にする。
"""
import csv
import os
import tempfile

from bs4 import BeautifulSoup

from src.tools.scrape_course_data import (
    _classify_image,
    _extract_basic_data,
    _extract_corner_info,
    _write_csv,
    DIRT_DISTANCES,
    TURF_DISTANCES,
)


class TestExtractBasicData:
    def test_straight_length(self):
        text = '最終直線は525mで芝コースの中では長い部類に入ります。'
        out = _extract_basic_data(text)
        assert out['straight_length_m'] == '525'

    def test_elevation_diff(self):
        text = 'このコースの高低差は2.7mとなっています。'
        out = _extract_basic_data(text)
        assert out['elevation_diff_m'] == '2.7'

    def test_turf_lap(self):
        text = '芝コースの一周は2083mです。'
        out = _extract_basic_data(text)
        assert out['turf_lap_m'] == '2083'

    def test_width_with_comma(self):
        text = 'コース幅は 1,600m です。'  # 桁区切りカンマの除去確認
        out = _extract_basic_data(text)
        assert out['width_m'] is None or ',' not in (out['width_m'] or '')

    def test_turn_direction(self):
        text = 'このコースは右回りのコースです。'
        out = _extract_basic_data(text)
        assert out['turn'] == '右回り'

    def test_missing_fields_stay_none(self):
        text = '特に数値情報のないテキストです。'
        out = _extract_basic_data(text)
        assert out['straight_length_m'] is None
        assert out['elevation_diff_m'] is None
        assert out['turn'] is None


class TestExtractCornerInfo:
    def test_corner_count_detected(self):
        n, io = _extract_corner_info('このコースは4コーナー制です。')
        assert n == '4'

    def test_inner_outer_detected(self):
        n, io = _extract_corner_info('外回りコースで直線が長い。')
        assert io == '外回り'

    def test_no_match_returns_none(self):
        n, io = _extract_corner_info('コーナーに関する記述なし')
        assert n is None
        assert io is None


class TestClassifyImage:
    def test_classify_by_alt(self):
        html = '<img src="a.png" alt="コース平面図">'
        img = BeautifulSoup(html, 'lxml').find('img')
        assert _classify_image(img, None) == 'course'

    def test_classify_3d_by_title(self):
        html = '<img src="a.png" title="立体図">'
        img = BeautifulSoup(html, 'lxml').find('img')
        assert _classify_image(img, None) == '3d'

    def test_classify_turf_elevation_by_heading(self):
        html = '<img src="a.png">'
        img = BeautifulSoup(html, 'lxml').find('img')
        assert _classify_image(img, '芝コース高低断面図') == 'turf_elevation'

    def test_classify_dirt_elevation(self):
        html = '<img src="a.png" alt="ダート高低断面図">'
        img = BeautifulSoup(html, 'lxml').find('img')
        assert _classify_image(img, None) == 'dirt_elevation'

    def test_unclassifiable_returns_none(self):
        html = '<img src="a.png" alt="バナー広告">'
        img = BeautifulSoup(html, 'lxml').find('img')
        assert _classify_image(img, None) is None


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
