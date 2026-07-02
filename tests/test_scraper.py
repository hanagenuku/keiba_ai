import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bs4 import BeautifulSoup
from src.scraper.parser import parse_header, get_class_from_racename, parse_hist
from src.scraper.calendar import get_base_from_calendar
from src.utils.config import KAISAI_CALENDAR
import src.scraper.jra_scraper as jra_scraper
from src.scraper.jra_scraper import find_r01_shutuba, parse_result_soup


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
            return _FakeResp('<html><body><table><tr><td>馬名</td></tr></table></body></html>')
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


# JRA result table: 着順,枠番,馬番,馬名,性齢,斤量,騎手,タイム,着差,通過順,上がり,単勝,人気,馬体重,調教師
_RESULT_HTML = """
<html><body>
<table>
<tr>
  <th>レース情報</th>
</tr>
<tr>
  <td colspan="15">2023年1月7日 中山 1600メートル（芝・右）3歳以上1勝クラス 天候:晴 馬場:良</td>
</tr>
<tr>
  <td>1</td><td>3</td><td>5</td><td>テストウマ</td>
  <td>牡4</td><td>57.0</td><td>テスト騎手</td>
  <td>1:34.5</td><td></td><td>3-3-2-1</td><td>34.1</td>
  <td>3.5</td><td>2</td><td>516(+4)</td><td>テスト調教師</td>
</tr>
<tr>
  <td>2</td><td>1</td><td>1</td><td>ニバンウマ</td>
  <td>牝5</td><td>55.0</td><td>サブ騎手</td>
  <td>1:34.8</td><td>3/4</td><td>1-1-1-2</td><td>35.0</td>
  <td>1.8</td><td>1</td><td>480(-2)</td><td>サブ調教師</td>
</tr>
<tr>
  <td>3</td><td>5</td><td>9</td><td>サンバンウマ</td>
  <td>牡6</td><td>57.0</td><td>サード騎手</td>
  <td>1:35.0</td><td>1.1/4</td><td>5-5-4-3</td><td>33.8</td>
  <td>12.4</td><td>5</td><td>500(0)</td><td>サード調教師</td>
</tr>
</table>
</body></html>
"""


def test_parse_result_soup_win_odds():
    soup = BeautifulSoup(_RESULT_HTML, 'lxml')
    result = parse_result_soup(soup, '中山', 1, '20230107', '06')
    assert result is not None
    horses = result['finishers']
    assert len(horses) == 3
    # 単勝オッズ（texts[11]）が正しく取得されること
    assert horses[0]['win_odds'] == 3.5
    assert horses[1]['win_odds'] == 1.8
    assert horses[2]['win_odds'] == 12.4
    # 人気（texts[12]）が正しく取得されること
    assert horses[0]['popularity'] == 2
    assert horses[1]['popularity'] == 1
    assert horses[2]['popularity'] == 5
    # 調教師（texts[14]）が正しく取得されること
    assert horses[0]['trainer'] == 'テスト調教師'
    # 馬体重（texts[13]）が正しく取得されること
    assert horses[0]['body_weight'] == 516
    assert horses[0]['body_weight_diff'] == 4
    assert horses[1]['body_weight'] == 480
    assert horses[1]['body_weight_diff'] == -2


class _MultiHitSession:
    """複数のsuffixでtableを返す擬似JRADB（最小suffix採用の確認用）。"""
    def __init__(self, hit_suffixes):
        self.hits = {f'{s:02X}' for s in hit_suffixes}

    def post(self, url, data=None, headers=None, timeout=None):
        cn = (data or {}).get('cname') or (data or {}).get('CNAME') or ''
        sx = cn.split('/')[-1].upper()
        if sx in self.hits:
            return _FakeResp('<table><tr><td>馬名</td></tr></table>')
        return _FakeResp('パラメータエラー')


def test_find_r01_shutuba_returns_min_hit():
    # 0x10, 0x40, 0x80 がヒットするとき最小の 0x10 を返す（直列版と挙動一致）
    sess = _MultiHitSession([0x10, 0x40, 0x80])
    assert find_r01_shutuba('pw01dde010320260201', '20260627', sess) == 0x10


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
