"""data/race_results.json（予想表に重ねる着順・配当）の検証。

North Star #6 に従い、手打ちdictではなく **実際に sqlite3 に INSERT した
history.db** から読み出して検証する。

🔴 固定したいこと:
  ・3着内でも `fukusho_payout` が NULL/0 の行が実データに約1.8%ある。
    これを「0円」として書かない（2026-08-18③: 欠損を0円として集計すると
    実力を過小評価する方向に静かにずれる、と実測済み）。
  ・`tansho_payout` は1着馬にしか入らない。2着以下の 0 を配当として書かない。
"""
import json
import os
import sqlite3

from scripts.generate_stats import save_race_results

_SCHEMA = """
CREATE TABLE race_history (race_id TEXT PRIMARY KEY, date TEXT, racecourse TEXT, race_num INTEGER);
CREATE TABLE horse_history (race_id TEXT, date TEXT, horse_num INTEGER, horse_name TEXT,
                            place INTEGER, tansho_payout INTEGER, fukusho_payout INTEGER);
"""

# (horse_num, place, tansho_payout, fukusho_payout)
_ROWS = [
    (7, 1, 1940, 490),
    (2, 2,    0, 200),
    (6, 3,    0,   0),   # 🔴 3着なのに複勝配当が欠損している実データの形
    (3, 4,    0,   0),
    (8, 99,   0,   0),   # 中止・除外のセンチネル
]


def _build(tmpdir, date='2026-09-06', race_id='20260906_01_01'):
    os.makedirs(os.path.join(tmpdir, 'data'), exist_ok=True)
    path = os.path.join(tmpdir, 'data', 'history.db')
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute('INSERT INTO race_history VALUES (?,?,?,?)', (race_id, date, '中山', 1))
    for num, place, tan, fuku in _ROWS:
        conn.execute('INSERT INTO horse_history VALUES (?,?,?,?,?,?,?)',
                     (race_id, date, num, f'馬{num}', place, tan, fuku))
    conn.commit()
    conn.close()
    return path


def test_writes_file_and_places(tmp_path):
    _build(str(tmp_path))
    out = save_race_results(str(tmp_path))
    assert out['dates'] == ['2026-09-06']
    r = out['races']['20260906_01_01']
    assert r['7']['place'] == 1
    assert r['2']['place'] == 2
    with open(tmp_path / 'data' / 'race_results.json', encoding='utf-8') as f:
        assert json.load(f)['races'] == out['races']


def test_missing_fukusho_is_null_not_zero(tmp_path):
    """🔴 3着なのに配当が取れていない行を「0円」と書かない。"""
    _build(str(tmp_path))
    r = save_race_results(str(tmp_path))['races']['20260906_01_01']
    assert r['6']['place'] == 3
    assert r['6']['fuku'] is None, '欠損を0円として書いている'


def test_tansho_only_for_winner(tmp_path):
    _build(str(tmp_path))
    r = save_race_results(str(tmp_path))['races']['20260906_01_01']
    assert r['7']['tan'] == 1940
    assert r['2']['tan'] is None, '2着に単勝配当0を書いている'
    assert r['3']['fuku'] is None, '4着に複勝配当を書いている'


def test_sentinel_place_excluded(tmp_path):
    """place=99（中止・除外）は着順として書かない。"""
    _build(str(tmp_path))
    r = save_race_results(str(tmp_path))['races']['20260906_01_01']
    assert '8' not in r


def test_limits_to_recent_days(tmp_path):
    path = _build(str(tmp_path))
    conn = sqlite3.connect(path)
    for i, d in enumerate(['2026-08-01', '2026-08-02', '2026-08-03', '2026-08-04']):
        rid = f'2026080{i+1}_01_01'
        conn.execute('INSERT INTO race_history VALUES (?,?,?,?)', (rid, d, '中山', 1))
        conn.execute('INSERT INTO horse_history VALUES (?,?,?,?,?,?,?)',
                     (rid, d, 1, '馬', 1, 100, 100))
    conn.commit()
    conn.close()
    out = save_race_results(str(tmp_path), n_days=2)
    assert out['dates'] == ['2026-09-06', '2026-08-04']
    assert len(out['races']) == 2


def test_no_db_is_not_an_error(tmp_path):
    """history.db が無くても例外を投げない（結果取得前は正常な状態）。"""
    os.makedirs(os.path.join(str(tmp_path), 'data'), exist_ok=True)
    out = save_race_results(str(tmp_path))
    assert out['races'] == {}
