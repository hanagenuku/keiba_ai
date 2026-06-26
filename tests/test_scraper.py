import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scraper.parser import parse_header, get_class_from_racename, parse_hist
from src.scraper.calendar import get_base_from_calendar
from src.utils.config import KAISAI_CALENDAR
import src.scraper.jra_scraper as jra_scraper
from src.scraper.jra_scraper import find_r01_shutuba


def test_parse_header_basic():
    text = '2026年5月10日 東京 1600メートル（芝・右）1勝クラス'
    info = parse_header(text)
    assert info.get('date') == '2026-05-10'
    assert info.get('racecourse') == '東京'
    assert info.get('distance') == 1600
    assert info.get('surface') == '芝'


def test_parse_header_shogai():
    text = '2026年5月10日 中山 障害 3600メートル'
    info = parse_header(text)
    assert info.get('surface') == '障害'


def test_get_class_from_racename():
    assert get_class_from_racename('G1天皇賞') == 'G1'
    assert get_class_from_racename('G2産経大阪杯') == 'G2'
    assert get_class_from_racename('3勝クラス') == '3勝クラス'
    assert get_class_from_racename('未勝利戦') == '未勝利'
    assert get_class_from_racename('新馬戦') == '新馬'
    assert get_class_from_racename('1勝クラス') == '1勝クラス'
    # 「特別」はオープンとして扱われる（元コードの仕様）
    assert get_class_from_racename('3歳以上特別') == 'オープン'


def test_parse_hist_basic():
    text = '1着 16頭 1600ダ 良 0.3秒差'
    result = parse_hist(text)
    assert result is not None
    assert result['place'] == 1
    assert result['finishers'] == 16
    assert result['distance'] == 1600
    assert result['surface'] == 'ダート'


def test_get_base_from_calendar_found():
    # KAISAI_CALENDARに含まれる日付でテスト
    # 東京（05）の開催日
    days = KAISAI_CALENDAR.get('05', [{}])[0].get('days', [])
    if not days:
        return
    date_str = days[0]
    base = get_base_from_calendar('05', date_str)
    assert base is not None
    assert base.startswith('pw01dde01')


def test_get_base_from_calendar_not_found():
    base = get_base_from_calendar('05', '20200101')  # 存在しない日付
    assert base is None


class _FakeResp:
    def __init__(self, text):
        self.text = text
        self.encoding = 'shift_jis'


class _FakeSession:
    """R01のsuffixが target_suffix のときだけ<table>を返す擬似JRADB。
    それ以外はJRADBと同じく『パラメータエラー』を返す。"""
    def __init__(self, target_suffix):
        self.target = target_suffix

    def post(self, url, data=None, headers=None, timeout=None):
        cn = (data or {}).get('cname') or (data or {}).get('CNAME') or ''
        sx = cn.split('/')[-1].upper()
        if sx == f'{self.target:02X}':
            return _FakeResp('<html><body><table><tr><td>出馬表</td></tr></table></body></html>')
        return _FakeResp('パラメータエラー')


def test_find_r01_shutuba_scans_past_early_errors(monkeypatch):
    # 回帰防止: R01のsuffixが先頭3つより後（0x50=80）にあっても見つけられること。
    # 旧実装は3連続パラメータエラーで打ち切り、suffixが高いと常にNoneを返していた。
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    sess = _FakeSession(target_suffix=0x50)
    found = find_r01_shutuba('pw01dde010320260201', '20260627', sess)
    assert found == 0x50


def test_find_r01_shutuba_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    sess = _FakeSession(target_suffix=999)  # 256内に存在しない → None
    assert find_r01_shutuba('pw01dde010320260201', '20260627', sess) is None


if __name__ == '__main__':
    test_parse_header_basic()
    print('✅ test_parse_header_basic passed')
    test_parse_header_shogai()
    print('✅ test_parse_header_shogai passed')
    test_get_class_from_racename()
    print('✅ test_get_class_from_racename passed')
    test_parse_hist_basic()
    print('✅ test_parse_hist_basic passed')
    test_get_base_from_calendar_found()
    print('✅ test_get_base_from_calendar_found passed')
    test_get_base_from_calendar_not_found()
    print('✅ test_get_base_from_calendar_not_found passed')
