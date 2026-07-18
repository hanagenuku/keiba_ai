import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bs4 import BeautifulSoup
from src.scraper.parser import parse_header, get_class_from_racename, parse_hist, parse_horse
from src.scraper.calendar import get_base_from_calendar
from src.utils.config import KAISAI_CALENDAR
import src.scraper.jra_scraper as jra_scraper
from src.scraper.jra_scraper import (
    find_r01_shutuba, parse_result_soup, apply_odds_to_races,
)


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


# ── apply_odds_to_races（専用オッズページの単勝を win_odds に反映） ──────────

def test_apply_odds_writes_tansho_to_win_odds():
    """market_odds_map の tansho を各馬の win_odds に書き戻す。"""
    races = [{'id': 'R1', 'horses': [
        {'num': 1, 'win_odds': 0.0},
        {'num': 2, 'win_odds': 0.0},
    ]}]
    mom = {'R1': {1: {'tansho': 2.4, 'fukusho': 1.3},
                  2: {'tansho': 9.1, 'fukusho': 2.0}}}
    n = apply_odds_to_races(races, mom)
    assert n == 2
    assert races[0]['horses'][0]['win_odds'] == 2.4
    assert races[0]['horses'][1]['win_odds'] == 9.1


def test_apply_odds_preserves_when_tansho_missing():
    """tansho が None / 0 の馬は既存 win_odds を保持する。"""
    races = [{'id': 'R1', 'horses': [
        {'num': 1, 'win_odds': 5.5},   # 専用ページに tansho なし
        {'num': 2, 'win_odds': 0.0},   # tansho=0
    ]}]
    mom = {'R1': {1: {'tansho': None, 'fukusho': 1.8},
                  2: {'tansho': 0.0, 'fukusho': None}}}
    n = apply_odds_to_races(races, mom)
    assert n == 0
    assert races[0]['horses'][0]['win_odds'] == 5.5
    assert races[0]['horses'][1]['win_odds'] == 0.0


def test_apply_odds_skips_race_without_map():
    """market_odds_map に無いレースは既存 win_odds を保持する。"""
    races = [{'id': 'R2', 'horses': [{'num': 1, 'win_odds': 3.3}]}]
    n = apply_odds_to_races(races, {})
    assert n == 0
    assert races[0]['horses'][0]['win_odds'] == 3.3


# ── 血統(父・母の父)スクレイピング（2026-07-17〜） ─────────────────────────

def test_parse_horse_extracts_pedigree_cname():
    """馬名リンクのhrefからCNAME(血統ページへの直リンク)を抽出できる。"""
    html = '''
    <tr>
      <td>1</td>
      <td><a href="/JRADB/accessU.html?CNAME=pw01dud002024103763/CB">テストウマ</a></td>
      <td>牡3</td>
      <td>57.0</td>
      <td><a href="#">調教師名</a></td>
      <td><a href="#">騎手名</a></td>
      <td>3.5</td>
    </tr>
    '''
    cells = BeautifulSoup(html, 'lxml').find('tr').find_all('td')
    h = parse_horse(cells, '東京', '芝')
    assert h is not None
    assert h['name'] == 'テストウマ'
    assert h['pedigree_cname'] == 'pw01dud002024103763/CB'


def test_parse_horse_pedigree_cname_none_without_href():
    """href が無い（通常のテキスト馬名など）場合は None のまま。"""
    html = '''
    <tr>
      <td>1</td>
      <td>テストウマ二</td>
      <td>牡3</td>
      <td>57.0</td>
    </tr>
    '''
    cells = BeautifulSoup(html, 'lxml').find('tr').find_all('td')
    h = parse_horse(cells, '東京', '芝')
    assert h is not None
    assert h.get('pedigree_cname') is None


_PEDIGREE_HTML = '''
<html><body>
<li class="data_col1">
<dl>
<dt>父</dt><dd>ステルヴィオ</dd>
<dt>母</dt><dd>オーミバンビーナ 産駒</dd>
<dt>母の父</dt><dd>ブラックタイド</dd>
<dt>母の母</dt><dd>ポットアカデミー 産駒</dd>
</dl>
</li>
</body></html>
'''


class _FakePedigreeSession:
    """accessU.htmlへのPOSTに対し、事前登録した cname → HTML を返す擬似JRADB。"""
    def __init__(self, html_by_cname):
        self.html_by_cname = html_by_cname
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        cn = (data or {}).get('cname') or (data or {}).get('CNAME') or ''
        self.calls.append(cn)
        html = self.html_by_cname.get(cn, '<html><body></body></html>')
        return _FakeResp(html)


def test_fetch_horse_pedigree_parses_sire_and_dam_sire():
    """<dt>/<dd>構造から父・母の父を取得し、母の"産駒"サフィックスは対象外にする。"""
    sess = _FakePedigreeSession({'pw01dud002024103763/CB': _PEDIGREE_HTML})
    result = jra_scraper.fetch_horse_pedigree(sess, 'pw01dud002024103763/CB')
    assert result == {'sire': 'ステルヴィオ', 'dam_sire': 'ブラックタイド'}


def test_fetch_horse_pedigree_missing_page_returns_empty():
    sess = _FakePedigreeSession({})
    result = jra_scraper.fetch_horse_pedigree(sess, 'pw01dud000000000000/00')
    assert result == {}


def test_fill_pedigree_skips_cached_horse(tmp_path):
    """history.dbに既に血統が記録済みの馬は再取得しない（ネットワークリクエストなし）。"""
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_01_01', 'racecourse': '東京', 'distance': 1600, 'surface': '芝',
        'finishers': [{'num': 1, 'name': 'キャッシュ済み馬', 'place': 3,
                       'sire': '既知の父', 'dam_sire': '既知の母父'}],
    }], db_path=str(hist_path))

    sess = _FakePedigreeSession({})  # 呼ばれたら空HTMLしか返せない＝取得ミスに気づける
    horses = [{'name': 'キャッシュ済み馬', 'pedigree_cname': 'pw01dud000000000000/00'}]
    jra_scraper._fill_pedigree(sess, horses, str(hist_path))

    assert horses[0]['sire'] == '既知の父'
    assert horses[0]['dam_sire'] == '既知の母父'
    assert sess.calls == []


def test_fill_pedigree_fetches_new_horse(tmp_path, monkeypatch):
    """history.dbに記録の無い新規馬は accessU.html から取得する。"""
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([], db_path=str(hist_path))  # スキーマ作成のみ

    sess = _FakePedigreeSession({'pw01dud002024103763/CB': _PEDIGREE_HTML})
    horses = [{'name': '新規馬', 'pedigree_cname': 'pw01dud002024103763/CB'}]
    jra_scraper._fill_pedigree(sess, horses, str(hist_path))

    assert horses[0]['sire'] == 'ステルヴィオ'
    assert horses[0]['dam_sire'] == 'ブラックタイド'
    assert sess.calls == ['pw01dud002024103763/CB']


def test_fill_pedigree_no_cname_skips_silently(tmp_path):
    """pedigree_cnameが取れなかった馬（href欠損等）はエラーにせずスキップする。"""
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([], db_path=str(hist_path))

    sess = _FakePedigreeSession({})
    horses = [{'name': 'CNAME無し馬', 'pedigree_cname': None}]
    jra_scraper._fill_pedigree(sess, horses, str(hist_path))

    assert 'sire' not in horses[0]
    assert sess.calls == []


def test_fill_pedigree_respects_budget(tmp_path, monkeypatch):
    """budgetの残数が尽きたら、それ以降の新規馬はリクエストせずスキップする。

    2026-07-18にworkflowが30分タイムアウトでキャンセルされ、その回の
    土曜結果・日曜予想が丸ごと保存されずに失われた事故の再発防止テスト
    （導入直後は全馬が"新規"扱いになり無制限だと数百リクエスト発生する）。
    """
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([], db_path=str(hist_path))

    sess = _FakePedigreeSession({
        'pw01dud0001/AA': _PEDIGREE_HTML,
        'pw01dud0002/BB': _PEDIGREE_HTML,
        'pw01dud0003/CC': _PEDIGREE_HTML,
    })
    horses = [
        {'name': '馬1', 'pedigree_cname': 'pw01dud0001/AA'},
        {'name': '馬2', 'pedigree_cname': 'pw01dud0002/BB'},
        {'name': '馬3', 'pedigree_cname': 'pw01dud0003/CC'},
    ]
    budget = {'remaining': 2}
    jra_scraper._fill_pedigree(sess, horses, str(hist_path), budget=budget)

    assert horses[0]['sire'] == 'ステルヴィオ'
    assert horses[1]['sire'] == 'ステルヴィオ'
    assert 'sire' not in horses[2]  # 上限到達でスキップ
    assert sess.calls == ['pw01dud0001/AA', 'pw01dud0002/BB']
    assert budget['remaining'] == 0


def test_fill_pedigree_budget_shared_across_calls(tmp_path, monkeypatch):
    """budgetは複数レース（複数回の_fill_pedigree呼び出し）にまたがって共有される。"""
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([], db_path=str(hist_path))

    sess = _FakePedigreeSession({
        'pw01dud0001/AA': _PEDIGREE_HTML,
        'pw01dud0002/BB': _PEDIGREE_HTML,
    })
    budget = {'remaining': 1}
    race1_horses = [{'name': 'レース1の馬', 'pedigree_cname': 'pw01dud0001/AA'}]
    race2_horses = [{'name': 'レース2の馬', 'pedigree_cname': 'pw01dud0002/BB'}]

    jra_scraper._fill_pedigree(sess, race1_horses, str(hist_path), budget=budget)
    jra_scraper._fill_pedigree(sess, race2_horses, str(hist_path), budget=budget)

    assert race1_horses[0]['sire'] == 'ステルヴィオ'
    assert 'sire' not in race2_horses[0]  # 前のレースで予算を使い切っている
    assert sess.calls == ['pw01dud0001/AA']


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
