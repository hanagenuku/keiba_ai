import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
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


def test_parse_header_surface_not_adjacent_to_distance_falls_back():
    """「メートル（芝・右）」のように芝ダ表記がメートルの直後に隣接していない
    実機ケース（2026-07-25調査、新潟R11で実際にparse失敗を起こした形）でも、
    _detect_surface()の多段判定にフォールバックして距離・surfaceを拾えることを
    確認する回帰テスト。修正前は本文中に芝/ダートの単独記載があっても
    surface='不明'・distance=0になり、そのままレース全体のparse失敗に繋がっていた。
    """
    text = ('2026年7月26日（日曜） 2回新潟2日 発走時刻： 18時00分 '
            '3歳以上1勝クラス [指定] 定量 コース： 1,000 メートル ダート 右回り')
    info = parse_header(text)
    assert info.get('distance') == 1000
    assert info.get('surface') == 'ダート'
    assert info.get('direction') == '右'


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


# ── _jradb_post 共通ラッパー（2026-07-20〜、重複コード削減のためのリファクタ） ──

def test_jradb_post_returns_response_on_success():
    sess = _FakeSession(target_suffix=0x50)
    resp = jra_scraper._jradb_post(sess, 'accessD.html', 'pw01dde010320260201015010326260627/50')
    assert resp is not None
    assert '<table>' in resp.text


def test_jradb_post_returns_none_on_parameter_error():
    sess = _FakeSession(target_suffix=0x50)
    resp = jra_scraper._jradb_post(sess, 'accessD.html', 'pw01dde01032026020101/FF')
    assert resp is None


def test_try_fetch_shutuba_success_and_parameter_error():
    """_jradb_post統一後も (resp, soup) / (None, None) の挙動が変わらないことを確認。"""
    sess = _FakeSession(target_suffix=0x50)
    resp, soup = jra_scraper._try_fetch_shutuba(
        sess, 'pw01dde010320260201', 1, '20260627', '50')
    assert resp is not None
    assert soup is not None
    assert soup.find_all('table')

    resp_none, soup_none = jra_scraper._try_fetch_shutuba(
        sess, 'pw01dde010320260201', 1, '20260627', 'FF')
    assert resp_none is None
    assert soup_none is None


def test_try_fetch_result_success_and_parameter_error():
    """_jradb_post統一後も soup / None の挙動が変わらないことを確認。"""
    sess = _FakeSession(target_suffix=0x50)
    soup = jra_scraper._try_fetch_result(sess, 'pw01dde010320260201', 1, '20260627', '50')
    assert soup is not None
    assert soup.find_all('table')

    soup_none = jra_scraper._try_fetch_result(sess, 'pw01dde010320260201', 1, '20260627', 'FF')
    assert soup_none is None


# JRA result table（2026-08-03、probe-result-columns.ymlで2026-08-02の実データを
# 取得し確定した列構成）: 着順,枠番,馬番,馬名,性齢,斤量,騎手,タイム,着差,通過順,
# 上がり,馬体重,調教師,人気（枠番は実データでも空欄。単勝オッズ列は存在しない）
#
# 旧構成（〜2026-06-28）は 着順,枠番,馬番,馬名,性齢,斤量,騎手,タイム,着差,通過順,
# 上がり,単勝,人気,馬体重,調教師 の15列だったが、2026-07-04頃にJRADB側で単勝オッズ
# 列が消え、人気/馬体重/調教師の並びも変わっていた（history.db実データの月次充足率
# 変化とprobe-result-columns.ymlの実HTML取得から特定。詳細はCLAUDE.md参照）。
_RESULT_HTML = """
<html><body>
<table>
<tr>
  <th>レース情報</th>
</tr>
<tr>
  <td colspan="14">2026年8月2日 新潟 1600メートル（芝・右）3歳以上1勝クラス 天候:晴 馬場:良</td>
</tr>
<tr>
  <td>1</td><td></td><td>7</td><td>テストウマ</td>
  <td>牡4</td><td>57.0</td><td>テスト騎手</td>
  <td>1:34.5</td><td></td><td>3 3 2 1</td><td>34.1</td>
  <td>516 (+4)</td><td>テスト調教師</td><td>2</td>
</tr>
<tr>
  <td>2</td><td></td><td>1</td><td>ニバンウマ</td>
  <td>牝5</td><td>55.0</td><td>サブ騎手</td>
  <td>1:34.8</td><td>3/4</td><td>1 1 1 2</td><td>35.0</td>
  <td>480 (-2)</td><td>サブ調教師</td><td>1</td>
</tr>
<tr>
  <td>3</td><td></td><td>9</td><td>サンバンウマ</td>
  <td>牡6</td><td>57.0</td><td>サード騎手</td>
  <td>1:35.0</td><td>1.1/4</td><td>5 5 4 3</td><td>33.8</td>
  <td>500 (0)</td><td>サード調教師</td><td>5</td>
</tr>
</table>
</body></html>
"""


def test_parse_result_soup_current_column_layout():
    """2026-08-03に実データで確定した現行の列構成（馬体重(11)/調教師(12)/人気(13)、
    単勝オッズ列なし）が正しくパースされることを確認する回帰テスト。
    旧構成（単勝(11)/人気(12)/馬体重(13)/調教師(14)）を前提にしたテストは
    実際のJRA側の列変更を検知できなかった（テストは通り続けたまま本番だけ
    壊れていた、North Star #6の実例）ため、実データ形状に更新した。
    """
    soup = BeautifulSoup(_RESULT_HTML, 'lxml')
    result = parse_result_soup(soup, '新潟', 1, '20260802', '04')
    assert result is not None
    horses = result['finishers']
    assert len(horses) == 3
    # 単勝オッズ列は現行構成に存在しないため常にNone
    assert horses[0]['win_odds'] is None
    # 人気（texts[13]）が正しく取得されること
    assert horses[0]['popularity'] == 2
    assert horses[1]['popularity'] == 1
    assert horses[2]['popularity'] == 5
    # 調教師（texts[12]）が正しく取得されること
    assert horses[0]['trainer'] == 'テスト調教師'
    # 馬体重（texts[11]）が正しく取得されること
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


def test_save_history_db_stores_trainer_affiliation(tmp_path):
    """horse_history.trainer_affiliation に「栗東」/「美浦」が保存・取得できる。"""
    import sqlite3
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_01_01', 'racecourse': '東京', 'distance': 1600, 'surface': '芝',
        'finishers': [
            {'num': 1, 'name': '栗東の馬', 'place': 1, 'trainer': '西村真幸',
             'trainer_affiliation': '栗東'},
            {'num': 2, 'name': '美浦の馬', 'place': 2, 'trainer': '秋本大介',
             'trainer_affiliation': '美浦'},
        ],
    }], db_path=str(hist_path))

    conn = sqlite3.connect(str(hist_path))
    rows = conn.execute(
        'SELECT horse_name, trainer_affiliation FROM horse_history ORDER BY horse_num'
    ).fetchall()
    conn.close()
    assert rows == [('栗東の馬', '栗東'), ('美浦の馬', '美浦')]


def test_save_history_db_stores_corner_passage(tmp_path):
    """race_history.corner_pass_3/4 に同着グルーピング表記が保存・取得できる。"""
    import sqlite3
    from src.utils.db import save_history_db
    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_01_01', 'racecourse': '東京', 'distance': 1600, 'surface': '芝',
        'corner_pass_3': '(1,*5)6,10(2,9)-(3,4)8=7',
        'corner_pass_4': '1(5,6)-(9,10)-(2,3,4)=8=7',
        'finishers': [{'num': 1, 'name': 'テスト馬', 'place': 1}],
    }], db_path=str(hist_path))

    conn = sqlite3.connect(str(hist_path))
    row = conn.execute(
        'SELECT corner_pass_3, corner_pass_4 FROM race_history WHERE race_id=?',
        ('20260101_01_01',),
    ).fetchone()
    conn.close()
    assert row == ('(1,*5)6,10(2,9)-(3,4)8=7', '1(5,6)-(9,10)-(2,3,4)=8=7')


# ── get_history_from_db（推論時の過去走取得）の racecourse/corner_all 欠落（2026-07-23〜） ──

def test_get_history_from_db_includes_racecourse_and_corner_all(tmp_path):
    """get_history_from_db()（推論時にcalc_all→calc_features_for_xgbが使うh['history']を
    構築する関数）が racecourse/corner_all を実際に返すことを確認する回帰テスト。

    この2列が欠けていると、calc_course_aptitude_features() 内の
    hrec.get('racecourse', '') が常に '' になり get_course_profile('', ...) が
    常に None を返すため、f_same_course_rate 等のコース適性特徴量群が推論時は
    常にデフォルト値に落ちる（学習時は build_training_data._get_history_before が
    racecourse を正しく渡すため、これは学習/推論パリティ違反になる）。
    本番のsave_history_db()で実際にDBへ書き込んだ上で読み出す（手打ちdictではない）。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'corner_all': '3-3-2-1'},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    assert len(hist) == 1
    assert hist[0]['racecourse'] == '中山'
    assert hist[0]['corner_all'] == '3-3-2-1'


def test_calc_course_aptitude_features_uses_inference_time_history(tmp_path):
    """get_history_from_db() が返す辞書（推論時と同じ形）を
    calc_course_aptitude_features() にそのまま渡した場合に、同一競馬場での
    経験が f_same_course_rate に反映されることを確認する統合テスト。
    修正前は racecourse が常に '' だったため、このアサーションは常に失敗していた
    （同一競馬場判定が一度も成立しなかった）。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.engine import calc_course_aptitude_features

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'corner_all': '3-3-2-1'},
        ],
    }], db_path=str(hist_path))

    root = os.path.join(os.path.dirname(__file__), '..')
    hist = get_history_from_db('テスト馬', str(hist_path))
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', hist, root)
    assert feats['f_course_coverage'] > 0  # 中山芝での経験走数がカウントされている


# ── get_history_from_db の finish_time/time_diff_sec/track_condition 欠落（2026-07-24） ──

def test_get_history_from_db_includes_finish_time_and_time_diff_sec(tmp_path):
    """get_history_from_db()が finish_time/time_diff_sec/track_condition を
    実際に返すことを確認する回帰テスト。

    time_diff_secとfinish_timeは完全に欠落しており、track_conditionは
    'condition'という別名でしか提供されていなかった。これらが欠けると推論時は
    calc_competitiveness()・f_finish_time_avg・f_time_diff_avg・
    speed_index（f_speed_fig_*）・f_heavy_track_rateが軒並みデフォルト値/NaNに
    落ちる。特にf_time_diff_avgは本番モデルの特徴量重要度で130特徴量中10位
    （1.76%）と上位のため影響が大きい。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1', 'track_condition': '重',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 2, 'agari3f': 34.5,
             'finish_time': 152.3, 'time_diff_sec': 0.5},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    assert len(hist) == 1
    assert hist[0]['finish_time'] == 152.3
    assert hist[0]['time_diff_sec'] == 0.5
    assert hist[0]['track_condition'] == '重'


def test_calc_features_for_xgb_uses_time_diff_from_inference_time_history(tmp_path):
    """get_history_from_db()の返り値をそのままcalc_features_for_xgbに渡した場合、
    f_time_diff_avg・f_finish_time_avgが実データを反映しNaNのままにならないことを
    確認する統合テスト。修正前はfinish_time/time_diff_secが常にNoneだったため、
    このアサーションは常に失敗していた（両方とも常にNaN）。
    """
    import math
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.engine import calc_features_for_xgb

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1', 'track_condition': '良',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 2, 'agari3f': 34.5,
             'finish_time': 152.3, 'time_diff_sec': 0.5},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    horse = {'name': 'テスト馬', 'horse_num': 1, 'running_style': '差し', 'history': hist}
    race = {
        'racecourse': '中山', 'surface': '芝', 'distance': 2500,
        'track_condition': '良', 'race_class': '2勝', 'first_3f': 35.0,
        'horses': [horse], 'date': '2026-02-01',
    }
    feats = calc_features_for_xgb(horse, race)
    assert not math.isnan(feats['f_time_diff_avg'])
    assert not math.isnan(feats['f_finish_time_avg'])


# ── get_history_from_db の agari_rank/num_finishers 欠落（2026-08-03） ──────

def test_get_history_from_db_includes_agari_rank_and_num_finishers(tmp_path):
    """get_history_from_db()が agari_rank/num_finishers を実際に返すことを
    確認する回帰テスト。

    agari_rank は元々SQLでSELECTされ agari3f_rank_pct の計算には使われていたが、
    返り値の辞書には一度も含まれていなかった。この結果、推論時は
    calc_agari_ability()（horse_type.py, f_agari_ability。Phase1・2026-05-25から
    本番稼働の距離適性特徴量）と calc_course_aptitude_features()の
    f_agari_rank_at_type の両方が hrec.get('agari_rank') -> None に落ち、
    常にデフォルト値（0.5）になっていた。学習側(_get_history_before)は
    agari_rank を正しく渡すため、学習/推論パリティ違反だった。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500,
        'surface': '芝', 'race_class': 'G1', 'num_finishers': 10,
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'agari_rank': 1},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    assert len(hist) == 1
    assert hist[0]['agari_rank'] == 1
    assert hist[0]['num_finishers'] == 10


def test_calc_agari_ability_uses_inference_time_history(tmp_path):
    """get_history_from_db()の返り値をそのまま calc_agari_ability()
    （horse_type.py, f_agari_ability の算出元）に渡した場合に、上がり3F1位
    （10頭立て）という好走実績が反映されることを確認する統合テスト。
    修正前は agari_rank が常にNoneだったため、このアサーションは常に失敗し
    デフォルトの0.5に固定されていた。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.horse_type import calc_agari_ability

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500,
        'surface': '芝', 'race_class': 'G1', 'num_finishers': 10,
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'agari_rank': 1},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    ability = calc_agari_ability(hist)
    assert ability == pytest.approx(1.0)


def test_calc_course_aptitude_features_uses_agari_rank_from_inference_time_history(tmp_path):
    """calc_course_aptitude_features()の f_agari_rank_at_type が、
    get_history_from_db()経由のagari_rankを実際に反映することを確認する
    統合テスト。修正前は常にデフォルト値0.5だった。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.engine import calc_course_aptitude_features

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500,
        'surface': '芝', 'race_class': 'G1', 'num_finishers': 10,
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'agari_rank': 1},
        ],
    }], db_path=str(hist_path))

    root = os.path.join(os.path.dirname(__file__), '..')
    hist = get_history_from_db('テスト馬', str(hist_path))
    feats = calc_course_aptitude_features('テスト馬', '中山', '芝', hist, root)
    assert feats['f_agari_rank_at_type'] == pytest.approx(0.1)


# ── corner_3 が常にNULLで死んでいた問題（2026-08-03発見） ──────────────────

class TestDeriveCorner3:
    """corner_3列は2026-06-25のcorner_all導入時に書き込みが停止し常にNULLに
    なった（docs/history_db_schema.md既知事項）。calc_features_for_xgb()の
    f_pos_avg_3等8特徴量とspeed_indexのcorner_pos引数はこの移行に追従せず
    今もcorner_3キーを読み続けており、corner_all移行以降ずっと
    running_styleベースの粗いフォールバックに落ちていた。
    """

    def test_derives_third_position_from_hyphenated_string(self):
        assert jra_scraper._derive_corner3('3-3-2-1') == 2

    def test_returns_none_when_fewer_than_three_positions(self):
        assert jra_scraper._derive_corner3('4-5') is None

    def test_returns_none_for_empty_or_none(self):
        assert jra_scraper._derive_corner3('') is None
        assert jra_scraper._derive_corner3(None) is None


def test_get_history_from_db_derives_corner_3_from_corner_all(tmp_path):
    """get_history_from_db()が、常にNULLのcorner_3列の代わりにcorner_allの
    3番目の値からcorner_3を導出して返すことを確認する回帰テスト。
    save_history_db()はcorner_3列に無条件でNoneを書き込む
    （finisher辞書にcorner_3を渡しても無視される）ため、修正前はここが
    常にNoneのままだった。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500,
        'surface': '芝', 'race_class': 'G1',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'corner_all': '3-3-2-1'},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    assert hist[0]['corner_3'] == 2


# ── 馬体重推移（f_weight_trend_avg / f_weight_last_diff、2026-07-24〜） ──────

def test_get_history_from_db_includes_body_weight(tmp_path):
    """get_history_from_db()が body_weight/body_weight_diff を実際に返すことを
    確認する回帰テスト。この2列は元々SELECT文に含まれておらず、engine.pyの
    どのf_*関数からも参照されていなかった（馬体重が完全に未活用だった）。
    """
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1',
        'finishers': [
            {'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
             'body_weight': 480, 'body_weight_diff': 6},
        ],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    assert len(hist) == 1
    assert hist[0]['body_weight'] == 480
    assert hist[0]['body_weight_diff'] == 6


def test_calc_features_for_xgb_computes_weight_trend_from_history(tmp_path):
    """get_history_from_db()の返り値をそのままcalc_features_for_xgbに渡した場合、
    f_weight_trend_avg・f_weight_last_diffが実データ(体重増減)を反映することを
    確認する統合テスト。過去2走の増減(+6, -2)から平均+2.0・直近値+6.0を期待する。
    """
    import math
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.engine import calc_features_for_xgb

    hist_path = tmp_path / 'history.db'
    save_history_db([
        {'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
         'race_class': 'G1',
         'finishers': [{'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5,
                        'body_weight': 480, 'body_weight_diff': 6}]},
        {'race_id': '20251201_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
         'race_class': 'G1',
         'finishers': [{'num': 5, 'name': 'テスト馬', 'place': 2, 'agari3f': 35.0,
                        'body_weight': 474, 'body_weight_diff': -2}]},
    ], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    horse = {'name': 'テスト馬', 'horse_num': 1, 'running_style': '差し', 'history': hist}
    race = {
        'racecourse': '中山', 'surface': '芝', 'distance': 2500,
        'track_condition': '良', 'race_class': '2勝', 'first_3f': 35.0,
        'horses': [horse], 'date': '2026-02-01',
    }
    feats = calc_features_for_xgb(horse, race)
    assert not math.isnan(feats['f_weight_trend_avg'])
    assert feats['f_weight_trend_avg'] == 2.0  # (+6 + -2) / 2
    assert feats['f_weight_last_diff'] == 6.0  # 直近走(2026-01-01)の増減


def test_calc_features_for_xgb_weight_trend_nan_when_no_data(tmp_path):
    """body_weight_diffが未記録の過去走しかない場合、f_weight_trend_avg等は
    0.0等に偽装せずNaN（欠損）として渡ることを確認する。
    """
    import math
    from src.utils.db import save_history_db
    from src.scraper.jra_scraper import get_history_from_db
    from src.features.engine import calc_features_for_xgb

    hist_path = tmp_path / 'history.db'
    save_history_db([{
        'race_id': '20260101_05_11', 'racecourse': '中山', 'distance': 2500, 'surface': '芝',
        'race_class': 'G1',
        'finishers': [{'num': 3, 'name': 'テスト馬', 'place': 1, 'agari3f': 34.5}],
    }], db_path=str(hist_path))

    hist = get_history_from_db('テスト馬', str(hist_path))
    horse = {'name': 'テスト馬', 'horse_num': 1, 'running_style': '差し', 'history': hist}
    race = {
        'racecourse': '中山', 'surface': '芝', 'distance': 2500,
        'track_condition': '良', 'race_class': '2勝', 'first_3f': 35.0,
        'horses': [horse], 'date': '2026-02-01',
    }
    feats = calc_features_for_xgb(horse, race)
    assert math.isnan(feats['f_weight_trend_avg'])
    assert math.isnan(feats['f_weight_last_diff'])


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


# ── 調教師所属（栗東/美浦）抽出（sp.jra.jp実機で確認した表記に基づく） ──────
def test_split_trainer_affiliation_ritto():
    name, affil = jra_scraper._split_trainer_affiliation('西村真幸(栗東)')
    assert name == '西村真幸'
    assert affil == '栗東'


def test_split_trainer_affiliation_miho():
    name, affil = jra_scraper._split_trainer_affiliation('秋本大介(美浦)')
    assert name == '秋本大介'
    assert affil == '美浦'


def test_split_trainer_affiliation_no_suffix_returns_none():
    """所属表記が無い場合は名前をそのまま返しaffiliationはNone（後方互換）"""
    name, affil = jra_scraper._split_trainer_affiliation('テスト調教師')
    assert name == 'テスト調教師'
    assert affil is None


def test_parse_result_soup_extracts_trainer_affiliation():
    """調教師欄「名前(栗東/美浦)」形式から所属を分離してtrainer_affiliationに格納する。
    trainerフィールド自体は名前のみになり既存の挙動を壊さない（後方互換）。"""
    html = _RESULT_HTML.replace('テスト調教師', '西村真幸(栗東)').replace(
        'サブ調教師', '秋本大介(美浦)')
    soup = BeautifulSoup(html, 'lxml')
    result = parse_result_soup(soup, '中山', 1, '20230107', '06')
    horses = result['finishers']
    assert horses[0]['trainer'] == '西村真幸'
    assert horses[0]['trainer_affiliation'] == '栗東'
    assert horses[1]['trainer'] == '秋本大介'
    assert horses[1]['trainer_affiliation'] == '美浦'


# ── ラップタイム見出し・払戻金の券種網羅（sp.jra.jp実機で確認したページ内容に基づく） ──
def _make_soup(text):
    return BeautifulSoup(f'<html><body><div>{text}</div></body></html>', 'lxml')


def test_extract_lap_times_matches_haron_time_heading():
    """実機(sp.jra.jp)で確認した「ハロンタイム」見出し表記に対応する。
    旧実装は「ラップタイム」表記のみ探索しており、この見出し違いにより
    first_3f/last_3fが長期未取得（docs/history_db_schema.md記載の0%）だった可能性が高い。"""
    soup = _make_soup(
        'タイム ハロンタイム 9.5 - 11.1 - 11.6 - 12.2 - 12.4 - 12.8 '
        '上り 4F 49.0 - 3F 37.4 '
        'コーナー通過順位 3コーナー (1,*5)6,10(2,9)-(3,4)8=7'
    )
    laps, first_3f, last_3f = jra_scraper._extract_lap_times(soup)
    assert laps == [9.5, 11.1, 11.6, 12.2, 12.4, 12.8]
    assert first_3f == round(9.5 + 11.1 + 11.6, 1)
    assert last_3f == round(12.2 + 12.4 + 12.8, 1)


def test_extract_lap_times_still_matches_old_heading():
    """旧「ラップタイム」表記でも従来どおり動作する（後方互換）"""
    soup = _make_soup('ラップタイム 12.5 - 10.9 - 11.4 - 11.8 - 12.0 - 12.3 ペース: M')
    laps, first_3f, last_3f = jra_scraper._extract_lap_times(soup)
    assert len(laps) == 6
    assert first_3f is not None and last_3f is not None


def test_extract_lap_times_no_heading_returns_empty():
    soup = _make_soup('見出しの無い本文のみ')
    laps, first_3f, last_3f = jra_scraper._extract_lap_times(soup)
    assert laps == [] and first_3f is None and last_3f is None


def test_parse_dividends_captures_all_ticket_types():
    """実機(sp.jra.jp)で確認した払戻金表を券種網羅で解析できる。
    db.pyのbet_type='馬単'決済は divs['umatan'] を参照するが、旧実装は
    umatan/wakuren/sanrentanを一切解析しておらず、馬単の的中払戻が常に0円に
    なる潜在バグがあった。"""
    soup = _make_soup(
        '単勝 1 110円 1番人気 '
        '複勝 1 110円 1番人気 6 180円 5番人気 10 110円 2番人気 '
        '枠連 1-6 880円 4番人気 '
        '馬連 1-6 920円 4番人気 '
        '馬単 1-6 960円 5番人気 '
        'ワイド 1-6 230円 4番人気 1-10 150円 1番人気 6-10 560円 7番人気 '
        '3連複 1-6-10 840円 3番人気 '
        '3連単 1-6-10 2,630円 8番人気'
    )
    divs = jra_scraper.parse_dividends(soup)
    assert divs['tansho'] == {'num': 1, 'payout': 110}
    assert len(divs['fukusho']) == 3
    assert divs['wakuren'] == {'nums': [1, 6], 'payout': 880}
    assert divs['umaren'] == {'nums': [1, 6], 'payout': 920}
    assert divs['umatan'] == {'nums': [1, 6], 'payout': 960}
    assert len(divs['wide']) == 3
    assert divs['sanrenpuku'] == {'nums': [1, 6, 10], 'payout': 840}
    assert divs['sanrentan'] == {'nums': [1, 6, 10], 'payout': 2630}


def test_parse_dividends_kanji_sanrenpuku_sanrentan_still_works():
    """旧「三連複」「三連単」表記（漢数字）でも従来どおり動作する（後方互換）"""
    soup = _make_soup('三連複 2-4-6 1,000円 1番人気 三連単 2-4-6 5,000円 1番人気')
    divs = jra_scraper.parse_dividends(soup)
    assert divs['sanrenpuku'] == {'nums': [2, 4, 6], 'payout': 1000}
    assert divs['sanrentan'] == {'nums': [2, 4, 6], 'payout': 5000}


# ── ペース判定（単語表記対応）・コーナー通過順位（グルーピング表記の収集） ──
def test_extract_weather_pace_matches_word_form():
    """実機(sp.jra.jp)の「ペース判定：ミドルペース」のような単語表記に対応する。
    旧実装は「ペース」直後のH/M/S 1文字のみ探索しており一致しなかった。"""
    full_text = 'タイム ハロンタイム 9.5 - 11.1 ペース判定： ミドルペース コーナー通過順位'
    weather, pace = jra_scraper._extract_weather_pace('天候:晴', full_text)
    assert weather == '晴'
    assert pace == 'M'


def test_extract_weather_pace_slow_and_high():
    _, pace_s = jra_scraper._extract_weather_pace('', 'ペース判定：スローペース')
    _, pace_h = jra_scraper._extract_weather_pace('', 'ペース判定：ハイペース')
    assert pace_s == 'S'
    assert pace_h == 'H'


def test_extract_weather_pace_still_matches_old_single_letter():
    """旧「ペース:M」のような1文字表記でも従来どおり動作する（後方互換）"""
    _, pace = jra_scraper._extract_weather_pace('', 'ペース: M')
    assert pace == 'M'


class TestPaceLabelDiagnostic:
    """ペース判定が取れないときの診断ログ（2026-07-28導入）。

    race_history.pace_label は5,411レース全てNULL。単語表記対応を入れた後
    （2026-07-25・07-26）も0件のままで、関数自体は上のテストどおり
    正しく拾えるため、取得元ページに文字列が無い可能性が高い。
    実HTMLを確認できないので決め打ち修正はせず、次回実行で
    どちらなのかを判別できる情報を残す。
    """

    def _run(self, capsys, text):
        jra_scraper._PACE_LABEL_DIAG_DONE = False
        jra_scraper.diag_pace_label_missing(text)
        return capsys.readouterr().out

    def test_reports_when_word_absent(self, capsys):
        """「ペース」の文字自体が無い＝取得元にその情報が無いケース。"""
        out = self._run(capsys, '天候:晴 ハロンタイム 12.1-11.2 上り 3F 34.5')
        assert 'ペース' in out and '存在しない' in out

    def test_reports_surrounding_text_when_word_present(self, capsys):
        """「ペース」はあるが表記が違うケース。周辺文字列を出して次の手がかりにする。"""
        out = self._run(capsys, 'レース情報 天候:晴 テンのペースは平均的 上り3F 34.5')
        assert 'テンのペースは平均的' in out

    def test_logs_only_once_per_run(self, capsys):
        """1レースごとに出すとログが溢れるため実行あたり1回に抑える。"""
        jra_scraper._PACE_LABEL_DIAG_DONE = False
        jra_scraper.diag_pace_label_missing('ペースあり')
        first = capsys.readouterr().out
        jra_scraper.diag_pace_label_missing('ペースあり')
        second = capsys.readouterr().out
        assert first.strip() and not second.strip()


class TestResultRowColumnDiagnostic:
    """結果ページの馬別行が popularity/body_weight/trainer を拾えないときの
    診断ログ（2026-08-03発見）。history.db実データで、この3列が2026-06-28
    までは正常に埋まっていたのに2026-07-04以降は突然0%になっていたと判明。
    jra_scraper.pyはこの間コミットされておらず、texts[0..10]相当
    （着順・馬番・馬名・性齢・斤量・騎手・タイム・着差・上がり）は現在も
    正常なため、実際のJRADB結果ページの列構成がtexts[11]以降のどこかで
    変わった可能性が高い。実HTMLを確認できないため決め打ち修正はせず、
    次回のワークフロー実行ログに実際の列内容を残すだけに留める。
    """

    def _run(self, capsys, texts):
        jra_scraper._RESULT_ROW_DIAG_DONE = False
        jra_scraper.diag_result_row_columns(texts)
        return capsys.readouterr().out

    def test_reports_actual_texts_array(self, capsys):
        texts = ['1', '2', '3', 'テスト馬', '牡3', '54.0', '騎手名',
                  '1:23.4', '', '', '34.5']
        out = self._run(capsys, texts)
        assert 'テスト馬' in out
        assert str(len(texts)) in out

    def test_logs_only_once_per_run(self, capsys):
        jra_scraper._RESULT_ROW_DIAG_DONE = False
        jra_scraper.diag_result_row_columns(['a'])
        first = capsys.readouterr().out
        jra_scraper.diag_result_row_columns(['b'])
        second = capsys.readouterr().out
        assert first.strip() and not second.strip()

    def test_parse_result_soup_fires_diagnostic_when_trailing_columns_missing(self, capsys):
        """popularity/body_weight/trainer相当の列(texts[11]以降)が無い行を
        parse_result_soup()に通すと診断ログが出ることを確認する統合テスト。
        着順〜上がり(texts[0..10]相当)しか無い行を渡し、例外にはならず
        （既存の len(texts)>N ガードで安全に空扱いに落ちる）診断だけが出ることを見る。
        """
        html = """
        <html><body><table>
        <tr><th>レース情報</th></tr>
        <tr><td colspan="11">2023年1月7日 中山 1600メートル（芝・右）3歳以上1勝クラス 天候:晴 馬場:良</td></tr>
        <tr>
          <td>1</td><td>3</td><td>5</td><td>テストウマ</td>
          <td>牡4</td><td>57.0</td><td>テスト騎手</td>
          <td>1:34.5</td><td></td><td>3-3-2-1</td><td>34.1</td>
        </tr>
        </table></body></html>
        """
        soup = BeautifulSoup(html, 'lxml')
        jra_scraper._RESULT_ROW_DIAG_DONE = False
        result = parse_result_soup(soup, '中山', 1, '20230107', '06')
        out = capsys.readouterr().out
        assert result is not None
        assert result['finishers'][0]['trainer'] == ''
        assert result['finishers'][0]['popularity'] == 99
        assert '結果ページ列診断' in out


def test_extract_corner_passage_captures_grouping_notation():
    """実機(sp.jra.jp)で確認した同着グルーピング表記を生テキストのまま抽出する。"""
    soup = _make_soup(
        'コーナー通過順位 3コーナー (1,*5)6,10(2,9)-(3,4)8=7 '
        '4コーナー 1(5,6)-(9,10)-(2,3,4)=8=7 払戻金 単勝 1 110円'
    )
    result = jra_scraper._extract_corner_passage(soup)
    assert result['corner_pass_3'] == '(1,*5)6,10(2,9)-(3,4)8=7'
    assert result['corner_pass_4'] == '1(5,6)-(9,10)-(2,3,4)=8=7'


def test_extract_corner_passage_no_section_returns_empty():
    soup = _make_soup('見出しの無い本文のみ')
    assert jra_scraper._extract_corner_passage(soup) == {}


def test_parse_result_soup_stores_corner_passage():
    """parse_result_soup() の info に corner_pass_3/4 が格納される。"""
    html = _RESULT_HTML.replace(
        '</table>',
        '<tr><td colspan="15">'
        'コーナー通過順位 3コーナー (1,*5)6,10(2,9)-(3,4)8=7 '
        '4コーナー 1(5,6)-(9,10)-(2,3,4)=8=7'
        '</td></tr></table>',
    )
    soup = BeautifulSoup(html, 'lxml')
    result = parse_result_soup(soup, '中山', 1, '20230107', '06')
    assert result['corner_pass_3'] == '(1,*5)6,10(2,9)-(3,4)8=7'
    assert result['corner_pass_4'] == '1(5,6)-(9,10)-(2,3,4)=8=7'


# ── fetch_races_on_date の取得失敗トラッキング（2026-07-25、データ品質可視化） ──
# 2026-07-25にparse_header()の芝ダ判定バグ（新潟R11）を修正したが、同種の
# parse失敗が将来別レースで再発してもコンソールログにしか残らず気づけなかった。
# fetch_races_on_date()が(all_races, failures)を返すよう変更し、failuresを
# latest.jsonのdata_quality.parse_failuresとして可視化できるようにする。

def test_fetch_races_on_date_tracks_genuine_parse_failures_only(monkeypatch):
    """ページ自体が存在しない（そのvenueがその日12レース未満）ケースは
    failuresに含めず、実際にページは取得できたのにparseに失敗した
    ケースのみを記録することを確認する。
    """
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(jra_scraper, 'get_kaisai_on_date',
                        lambda target_date, sess: {'pw01dde0103202607269901': target_date})
    monkeypatch.setattr(jra_scraper, 'find_r01_shutuba', lambda base, date_str, sess: 0x10)
    monkeypatch.setattr(jra_scraper, 'find_r01_odds', lambda odds_base, date_str, sess: None)
    monkeypatch.setattr(jra_scraper, '_fill_pedigree', lambda *a, **k: None)

    def fake_try_fetch_shutuba(sess, base, r, date_str, sx):
        if r == 1:
            return None, '<success-marker>'  # _parse_shutubaをモックするので中身は使わない
        if r == 11:
            html = ('<html><body><table><tr><td>'
                    '2026年7月26日 3歳以上1勝クラス'  # 芝ダ表記もメートル表記も無い
                    '</td></tr></table></body></html>')
            return None, BeautifulSoup(html, 'lxml')
        return None, None  # そのレース番号のページ自体が存在しない

    monkeypatch.setattr(jra_scraper, '_try_fetch_shutuba', fake_try_fetch_shutuba)

    def fake_parse_shutuba(soup, rc, r, date_str, pc, hist_db_path):
        if r == 1:
            return {'id': f'{date_str}_99_01', 'racecourse': rc, 'race_num': 1,
                    'race_name': 'テストR', 'distance': 1600, 'surface': '芝',
                    'horses': [{'num': 1, 'name': 'テストウマ'}]}
        return None

    monkeypatch.setattr(jra_scraper, '_parse_shutuba', fake_parse_shutuba)

    races, failures = jra_scraper.fetch_races_on_date(object(), '20260726', ':memory:')

    assert len(races) == 1
    assert races[0]['race_num'] == 1

    # r=11のみが「ページは取れたがparseに失敗した」ケースとして記録され、
    # ページ自体が無かった他のレース番号（2-10,12）はfailuresに含まれない
    assert len(failures) == 1
    assert failures[0]['race_num'] == 11
    assert failures[0]['reason'] == 'parse失敗'
    assert all(f['race_num'] != 2 for f in failures)


# ── find_r01_odds の失敗内訳ログ（2026-07-25、全venue原因不明で「未発見」に
#    なった障害の再発時に原因を特定できるようにするための診断ログ追加） ──

def test_find_r01_odds_finds_target_suffix():
    """既存の成功パス（該当suffixでtableを検出して返す）は変更しないことを確認。"""
    sess = _FakeSession(target_suffix=0x42)
    found = jra_scraper.find_r01_odds('pw151ouS30320260201', '20260627', sess)
    assert found == 0x42


def test_find_r01_odds_logs_breakdown_when_none_found(monkeypatch, capsys):
    """256件全て「パラメータエラー」だった場合、内訳をログに残しNoneを返す。"""
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    sess = _FakeSession(target_suffix=999)  # 256内に存在しない → 全てパラメータエラー
    found = jra_scraper.find_r01_odds('pw151ouS30320260201', '20260627', sess)
    assert found is None
    out = capsys.readouterr().out
    assert 'find_r01_odds' in out
    assert 'パラメータエラー=256' in out
    assert '例外=0' in out


def test_find_r01_odds_counts_exceptions_separately(monkeypatch, capsys):
    """通信例外はパラメータエラーと区別してカウントされることを確認。"""
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)

    class _RaisingSession:
        def post(self, url, data=None, headers=None, timeout=None):
            raise TimeoutError('boom')

    found = jra_scraper.find_r01_odds('pw151ouS30320260201', '20260627', _RaisingSession())
    assert found is None
    out = capsys.readouterr().out
    assert '例外=256' in out
    assert 'パラメータエラー=0' in out
    assert 'boom' in out


def test_fetch_races_on_date_obstacle_skip_not_recorded_as_failure(monkeypatch):
    """障害レースのスキップは意図した挙動のためfailuresに含めないことを確認する。"""
    monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(jra_scraper, 'get_kaisai_on_date',
                        lambda target_date, sess: {'pw01dde0103202607269901': target_date})
    monkeypatch.setattr(jra_scraper, 'find_r01_shutuba', lambda base, date_str, sess: 0x10)
    monkeypatch.setattr(jra_scraper, 'find_r01_odds', lambda odds_base, date_str, sess: None)
    monkeypatch.setattr(jra_scraper, '_fill_pedigree', lambda *a, **k: None)

    def fake_try_fetch_shutuba(sess, base, r, date_str, sx):
        if r == 1:
            html = ('<html><body><table><tr><td>'
                    '2026年7月26日 障害 3600メートル'
                    '</td></tr></table></body></html>')
            return None, BeautifulSoup(html, 'lxml')
        return None, None

    monkeypatch.setattr(jra_scraper, '_try_fetch_shutuba', fake_try_fetch_shutuba)
    monkeypatch.setattr(jra_scraper, '_parse_shutuba', lambda *a, **k: None)

    races, failures = jra_scraper.fetch_races_on_date(object(), '20260726', ':memory:')

    assert races == []
    assert failures == []


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


class TestFindR01ResultRetry:
    """R01結果スキャンの再試行（2026-08-02の日曜欠損への対応）。

    日曜結果取得で新潟・札幌の2会場がスキャン全滅し、予想35レースに対し
    結果が11レースしか取れなかった。同じ週の土曜は3会場とも成功していたので
    恒久的な不在ではなく一過性の失敗。1周で諦めるとその開催が永久に失われる。
    """

    class _Resp:
        def __init__(self, text):
            self.text = text
            self.encoding = None

    def _sess(self, behaviour):
        """behaviour(scan_no, suffix) -> 'param'|'ok'|'raise'|'notable'"""
        outer = self
        state = {'scans': 0}

        class S:
            def post(self, url, data=None, headers=None, timeout=None):
                sx = int(data['CNAME'][-2:], 16)
                if sx == 0:
                    state['scans'] += 1
                kind = behaviour(state['scans'], sx)
                if kind == 'raise':
                    raise ConnectionError('timeout')
                if kind == 'ok':
                    return outer._Resp('<table>着順 馬名 騎手 調教師</table>')
                if kind == 'notable':
                    return outer._Resp('なにもなし')
                return outer._Resp('パラメータエラー')

        return S(), state

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(jra_scraper.time, 'sleep', lambda *a, **k: None)

    def test_found_on_first_scan_does_not_retry(self):
        sess, st = self._sess(lambda scan, sx: 'ok' if sx == 0x6C else 'param')
        assert jra_scraper.find_r01_result('pw01sde10', '20260802', sess) == 0x6C
        assert st['scans'] == 1

    def test_all_param_error_gives_up_immediately(self):
        """開催のない会場で無駄に5分待たないこと。"""
        sess, st = self._sess(lambda scan, sx: 'param')
        assert jra_scraper.find_r01_result('pw01sde10', '20260802', sess) is None
        assert st['scans'] == 1

    def test_transient_failure_is_retried_and_succeeds(self):
        """本件の再現: 1周目が通信エラーで全滅 → 再試行で取得できる。"""
        def flaky(scan, sx):
            if scan == 1:
                return 'raise'
            return 'ok' if sx == 0x6C else 'param'
        sess, st = self._sess(flaky)
        assert jra_scraper.find_r01_result('pw01sde10', '20260802', sess) == 0x6C
        assert st['scans'] == 2

    def test_gives_up_after_attempts(self):
        sess, st = self._sess(lambda scan, sx: 'raise')
        assert jra_scraper.find_r01_result(
            'pw01sde10', '20260802', sess, attempts=3) is None
        assert st['scans'] == 3

    def test_signature_stays_backward_compatible(self):
        """既存の3引数呼び出し（fetch_results / rescrape_history）が壊れないこと。"""
        import inspect
        sig = inspect.signature(jra_scraper.find_r01_result)
        assert list(sig.parameters)[:3] == ['base', 'date', 'sess']
        assert sig.parameters['attempts'].default == 3


class TestRescrapeRepairsPopularityAndTrainer:
    """2026-07-04以降の列崩れで壊れた popularity / trainer を、
    再スクレイプ（save_history_db の再実行）で復旧できることの検証。

    修正前は UPDATE 節に popularity / trainer が無く、かつ popularity は
    取得失敗時に NULL ではなく 99（センチネル）で INSERT されるため
    COALESCE でも直せず、4,484行が永久に復旧不能だった（2026-08-05調査）。
    North Star #6 に従い、手打ちINSERTではなく本番と同じ save_history_db()
    で書き込んでから実際の sqlite3 で読み出して検証する。
    """

    def _broken_scrape(self):
        """列崩れ時の実際の姿: popularity=99(センチネル)・trainer空・体重なし"""
        return {
            'race_id': '20260704_01_01', 'racecourse': '福島',
            'distance': 1200, 'surface': '芝',
            'finishers': [
                {'num': 1, 'name': 'イチバンウマ', 'place': 1,
                 'popularity': 99, 'trainer': '', 'body_weight': None},
                {'num': 2, 'name': 'ニバンウマ', 'place': 2,
                 'popularity': 99, 'trainer': '', 'body_weight': None},
            ],
        }

    def _fixed_scrape(self):
        """列崩れ修正後(2026-08-03③)のパーサーで取り直した正しい値"""
        return {
            'race_id': '20260704_01_01', 'racecourse': '福島',
            'distance': 1200, 'surface': '芝',
            'finishers': [
                {'num': 1, 'name': 'イチバンウマ', 'place': 1,
                 'popularity': 3, 'trainer': '斎藤 誠', 'body_weight': 482},
                {'num': 2, 'name': 'ニバンウマ', 'place': 2,
                 'popularity': 1, 'trainer': '手塚 貴久', 'body_weight': 470},
            ],
        }

    def _read(self, hist_path):
        import sqlite3
        conn = sqlite3.connect(str(hist_path))
        rows = conn.execute(
            'SELECT horse_num, popularity, trainer, body_weight '
            'FROM horse_history ORDER BY horse_num').fetchall()
        conn.close()
        return rows

    def test_rescrape_repairs_broken_popularity_and_trainer(self, tmp_path):
        from src.utils.db import save_history_db
        hist_path = tmp_path / 'history.db'
        # ① 列崩れ状態で一度保存（本番で実際に起きた状態）
        save_history_db([self._broken_scrape()], db_path=str(hist_path))
        assert self._read(hist_path) == [(1, 99, '', None), (2, 99, '', None)]
        # ② 修正後のパーサーで再スクレイプ
        save_history_db([self._fixed_scrape()], db_path=str(hist_path))
        # ③ 復旧していること（修正前のコードではここで失敗した）
        assert self._read(hist_path) == [
            (1, 3, '斎藤 誠', 482),
            (2, 1, '手塚 貴久', 470),
        ]

    def test_rescrape_failure_does_not_destroy_good_popularity(self, tmp_path):
        """再スクレイプが失敗(99)しても、既に入っている正しい人気を壊さない。"""
        from src.utils.db import save_history_db
        hist_path = tmp_path / 'history.db'
        save_history_db([self._fixed_scrape()], db_path=str(hist_path))
        # 取得失敗した再スクレイプを流す
        save_history_db([self._broken_scrape()], db_path=str(hist_path))
        rows = self._read(hist_path)
        assert [r[1] for r in rows] == [3, 1], '99で正しい人気を上書きしてはいけない'
        assert [r[2] for r in rows] == ['斎藤 誠', '手塚 貴久'], '空文字で調教師を消してはいけない'

    def test_repaired_popularity_revives_f_popularity(self, tmp_path):
        """復旧した人気が f_popularity として実際に使える値になること
        （99のままだと engine 側で NaN に落ち base_margin が死ぬ）。"""
        import math
        from src.utils.db import save_history_db
        from src.features.engine import calc_features_for_xgb
        hist_path = tmp_path / 'history.db'
        save_history_db([self._broken_scrape()], db_path=str(hist_path))

        race = {'racecourse': '福島', 'distance': 1200, 'surface': '芝',
                'date': '2026-07-04', 'race_class': '1勝', 'horses': []}
        broken_h = {'name': 'イチバンウマ', 'horse_num': 1,
                    'popularity': 99, 'history': []}
        assert math.isnan(calc_features_for_xgb(broken_h, race)['f_popularity'])

        save_history_db([self._fixed_scrape()], db_path=str(hist_path))
        pop = self._read(hist_path)[0][1]
        fixed_h = {'name': 'イチバンウマ', 'horse_num': 1,
                   'popularity': pop, 'history': []}
        assert calc_features_for_xgb(fixed_h, race)['f_popularity'] == 3.0
