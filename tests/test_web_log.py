"""`src/product/web_log.py`（web 見解の記録台帳）のテスト。

North Star #6: **本番と同じ形**で検証する。
- findings は `data/claude_web_findings.json` の実物から写した形
  （`going.state` が「収集時点で発表済みの馬場」で `expected` が自由文の想定、
   `horses` が出典つきの馬メモ）
- 着順・払戻は手打ち dict ではなく**実際に sqlite3 に INSERT した行**を読む
  （`sqlite3.Row` 由来の経路を通す）
"""

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.product import web_log  # noqa: E402

# 実物から写した形（2026-10-18 京都11R / 東京10R）
FINDINGS = {
    'schema_version': 1,
    'date': '20261018',
    'collected_at': '2026-10-18T09:05:18+09:00',
    'races': {
        '20261018_05_10': {
            'venue': '東京', 'race_num': 10,
            'sources': ['https://www.bloodline-trackbias.work/entry/2026/10/17/200000'],
            'going': {'surface': 'ダート', 'state': '良',
                      'expected': '良（含水率1%台で非常に乾燥・日曜はさらに乾く見込み）'},
            'summary': '東京ダートは含水率1%台で非常に乾燥。',
        },
        '20261018_08_11': {
            'venue': '京都', 'race_num': 11,
            'sources': ['https://www.bloodline-trackbias.work/entry/2026/10/17/200000'],
            'going': {'surface': '芝', 'state': '良', 'expected': '硬さは標準'},
            'summary': '京都芝は金曜昼時点で良。',
            'marks': {'6': '◎'},
            'horses': {
                '6': {'note': '栗東CW 最終 強め。', 'direction': 'up',
                      'sources': ['https://www.keibanomiryoku.com/article/x.html']},
                '9': {'note': '坂路 末強め。', 'direction': None,
                      'sources': ['https://www.keibanomiryoku.com/article/x.html']},
            },
        },
    },
}


@pytest.fixture()
def base(tmp_path):
    (tmp_path / 'data').mkdir()
    with open(tmp_path / 'data' / 'claude_web_findings.json', 'w',
              encoding='utf-8') as f:
        json.dump(FINDINGS, f, ensure_ascii=False)
    return str(tmp_path)


def _hist(tmp_path, place_9=None):
    """history.db を本番と同じ形で作る。place_9=None は「取消等で着順が無い」。"""
    p = tmp_path / 'data' / 'hist.db'
    c = sqlite3.connect(p)
    c.executescript("""
    CREATE TABLE race_history (race_id TEXT PRIMARY KEY, track_condition TEXT,
                               surface TEXT);
    CREATE TABLE horse_history (race_id TEXT, horse_num INTEGER, place INTEGER,
                                popularity INTEGER, horse_name TEXT);
    """)
    c.execute("INSERT INTO race_history VALUES ('20261018_05_10','良','ダート')")
    c.execute("INSERT INTO race_history VALUES ('20261018_08_11','稍重','芝')")
    c.execute("INSERT INTO horse_history VALUES ('20261018_08_11',6,2,1,'アアアア')")
    c.execute("INSERT INTO horse_history VALUES ('20261018_08_11',9,?,5,'イイイイ')",
              (place_9,))
    c.commit()
    return c


def _keiba(tmp_path, with_payouts=True):
    p = tmp_path / 'data' / 'keiba.db'
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE race_dividends (race_id TEXT, bet_type TEXT, '
              'combo TEXT, payout INTEGER)')
    if with_payouts:
        # 🔑 払戻表にある組だけが的中。着順から自分で判定しない
        c.execute("INSERT INTO race_dividends VALUES "
                  "('20261018_08_11','fukusho','6',210)")
    c.commit()
    return c


# ───────────────────────── 記録 ─────────────────────────

def test_source_host_strips_www():
    assert web_log.source_host('https://www.keibanomiryoku.com/a.html') == \
        'keibanomiryoku.com'
    assert web_log.source_host('https://db.netkeiba.com/x') == 'db.netkeiba.com'
    assert web_log.source_host('not a url') is None


def test_collect_splits_dimensions_and_keeps_per_entry_sources():
    es = web_log.collect_from_findings(FINDINGS)
    kinds = sorted({e['kind'] for e in es})
    assert kinds == ['going', 'horse', 'mark']
    going = [e for e in es if e['kind'] == 'going']
    assert len(going) == 2
    assert going[0]['hosts'] == ['bloodline-trackbias.work']
    # 🔑 馬メモは**その馬メモ自身の出典**を持つ（レースの出典で上書きしない）
    horse = [e for e in es if e['kind'] == 'horse']
    assert len(horse) == 2
    assert all(h['hosts'] == ['keibanomiryoku.com'] for h in horse)


def test_record_is_append_only(base):
    r1 = web_log.record(base)
    assert r1['added'] == 5 and r1['total'] == 5
    r2 = web_log.record(base)
    assert r2['added'] == 0 and r2['total'] == 5


def test_record_does_not_overwrite_an_existing_key(base):
    """当日の refresh で findings が差し替わっても朝の値を残す。"""
    web_log.record(base)
    changed = json.loads(json.dumps(FINDINGS))
    changed['races']['20261018_08_11']['horses']['6']['note'] = '差し替えた'
    web_log.record(base, changed)
    led = web_log.load(base)
    e = [x for x in led['entries']
         if x['kind'] == 'horse' and x['key'] == '6'][0]
    assert e['said']['note'] == '栗東CW 最終 強め。'


def test_load_refuses_an_unknown_schema_version(base):
    web_log.record(base)
    p = web_log.ledger_path(base)
    d = json.load(open(p, encoding='utf-8'))
    d['schema_version'] = 999
    json.dump(d, open(p, 'w', encoding='utf-8'))
    with pytest.raises(ValueError):
        web_log.load(base)


# ───────────────────────── going の採点 ─────────────────────────

def test_expected_state_is_extracted_only_when_named():
    assert web_log.expected_state('良（含水率1%台で乾燥）') == '良'
    assert web_log.expected_state('稍重まで渋るかも') == '稍重'
    # 🔴 実物の大半は状態名を名乗らない。無理に埋めない
    assert web_log.expected_state('硬さは標準') is None
    assert web_log.expected_state('乾燥（含水率2％台）') is None
    assert web_log.expected_state(None) is None


def test_going_separates_collection_state_from_the_prediction(base, tmp_path):
    """🔴 収集時点の馬場と確定馬場の一致を「予想が当たった」と書かない。

    実物の `state` は「金曜昼時点で良」＝既に発表されている値で、これを確定馬場と
    比べると馬場が動かなかったことしか測れない（実測 21/21 ＝ 100%）。
    予想として採点できるのは `expected` が状態名を名乗ったときだけ。
    """
    web_log.record(base)
    hist, keiba = _hist(tmp_path), _keiba(tmp_path)
    web_log.attach_results(base, hist, keiba)
    led = web_log.load(base)
    by = {(e['race_id'], e['kind']): e for e in led['entries']}

    # 東京10R: 収集時点 良 / 確定 良 / 想定も「良」と名乗っている → どちらも True
    r = by[('20261018_05_10', 'going')]['result']
    assert r['actual_track_condition'] == '良'
    assert r['collection_state_matches_actual'] is True
    assert r['said_expected_state'] == '良'
    assert r['expected_state_matches_actual'] is True

    # 京都11R: 収集時点 良 だが確定は 稍重 → 一致しない。想定は状態名を名乗らず採点不能
    r = by[('20261018_08_11', 'going')]['result']
    assert r['actual_track_condition'] == '稍重'
    assert r['collection_state_matches_actual'] is False
    assert r['said_expected_state'] is None
    assert r['expected_state_matches_actual'] is None


# ───────────────────────── 馬の結果 ─────────────────────────

def test_payout_comes_from_the_table_not_from_the_place(base, tmp_path):
    """🔑 3着内でも払戻行が無ければ `fukusho_payout` を作らない。

    少頭数のレースは複勝が2着までしか発売されない（2026-10-02 に実データで確認）。
    着順から払戻を推測すると、売られていない馬券を的中として記録する。
    """
    web_log.record(base)
    hist, keiba = _hist(tmp_path, place_9=3), _keiba(tmp_path)
    web_log.attach_results(base, hist, keiba)
    led = web_log.load(base)
    by = {(e['kind'], e['key']): e for e in led['entries']
          if e['race_id'] == '20261018_08_11'}
    r6 = by[('horse', '6')]['result']
    assert r6['place'] == 2 and r6['top3'] is True
    assert r6['fukusho_payout'] == 210
    r9 = by[('horse', '9')]['result']
    assert r9['place'] == 3 and r9['top3'] is True
    # 3着なのに払戻行が無い＝複勝が売られていない。推測で埋めない
    assert r9['fukusho_payout'] is None
    assert r9['fukusho_sold'] is False


def test_pending_when_payouts_are_missing(base, tmp_path):
    """🔴 払戻が1行も無いレースを空の結果で埋めない（全部外れとして残る）。"""
    web_log.record(base)
    hist = _hist(tmp_path, place_9=3)
    keiba = _keiba(tmp_path, with_payouts=False)
    out = web_log.attach_results(base, hist, keiba)
    led = web_log.load(base)
    horses = [e for e in led['entries'] if e['kind'] in ('horse', 'mark')]
    assert all(e['result'] is None for e in horses)
    assert out['pending'] == len(horses)
    # going は払戻に依存しないので確定馬場だけで埋まる
    goings = [e for e in led['entries'] if e['kind'] == 'going']
    assert all(e['result'] is not None for e in goings)


def test_pending_when_the_placing_is_missing(base, tmp_path):
    """着順が無い馬（取消等）も保留する。"""
    web_log.record(base)
    hist, keiba = _hist(tmp_path, place_9=None), _keiba(tmp_path)
    web_log.attach_results(base, hist, keiba)
    led = web_log.load(base)
    e9 = [e for e in led['entries']
          if e['kind'] == 'horse' and e['key'] == '9'][0]
    assert e9['result'] is None


def test_attach_is_idempotent(base, tmp_path):
    web_log.record(base)
    hist, keiba = _hist(tmp_path, place_9=3), _keiba(tmp_path)
    a = web_log.attach_results(base, hist, keiba)
    b = web_log.attach_results(base, hist, keiba)
    assert b['attached'] == 0
    assert b['already'] == a['attached']


# ───────────────────────── 集計 ─────────────────────────

def test_summary_groups_by_source_and_prints_no_recovery_rate(base, tmp_path):
    web_log.record(base)
    hist, keiba = _hist(tmp_path, place_9=3), _keiba(tmp_path)
    web_log.attach_results(base, hist, keiba)
    s = web_log.summarize(base)
    assert set(s['by_host']) == {'bloodline-trackbias.work', 'keibanomiryoku.com'}
    assert s['by_host']['keibanomiryoku.com']['horse']['top3_of'] == 2
    # 🔴 回収率を集計しない（2026-10-02 の決定。賭け金の仮定が要る）。
    #    数字として持っているキーに回収率が無いことを見る（注意文の本文は別）
    keys = set()
    for kinds in s['by_host'].values():
        for k in kinds.values():
            keys |= set(k)
    for bad in ('roi', 'roi_cut3', 'return', 'recovery'):
        assert bad not in keys
    assert '回収率は出していない' in s['caveat']
    assert '予想の精度ではない' in s['caveat']


# ───────────────────────── 予想に繋がっていないこと ─────────────────────────

def test_it_does_not_read_or_write_the_prediction(base):
    """`claude_web.py` の別枠設計をそのまま引き継ぐ（撤回された後付け層5件と違う形）。"""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src', 'product', 'web_log.py'),
        encoding='utf-8').read()
    body = src.split('"""', 2)[2]  # docstring の外だけを見る
    # 予想の数字を**読む形**（`.get('score')` 等）が無いことを見る。
    # 単語そのものの禁止だと自分の戻り値キー（'total' 等）に当たって無意味になる
    for field in ('score', 'total', 'win_prob', 'cal_prob', 'rl_rank',
                  'fit_rank', 'solo_rank', 'bets', 'gumbel_bets', 'rec', 'conf'):
        for pat in (f".get('{field}')", f'["{field}"]', f"['{field}']"):
            assert pat not in body, f'web_log が予想の {field} に触っている'
    # 本番の予想ファイルを読まない（読むのは findings と history/keiba だけ）
    assert 'latest.json' not in body


# ──────────── 序列（grade）＝「測れる形になっているか」の記録 ────────────
#
# 🔴 2026-10-23 に実データ97件で判明したこと: 情報源ごとに測れる形が違う。
#   `keibanomiryoku` は 1レースあたり中央値 15.5頭（ほぼ全馬）に**方向も序列も無い
#   メモ**を付けるので、3着内率は母集団の3着内率そのものになり当たり外れの概念が無い。
#   `umasiru` は note の文中に 評価 S/A/B/C を 26/27 持っていたが自由文なので
#   台帳から序列として読めなかった。基準は `websig/CRITERIA_RELIABILITY.md`。

def test_normalize_grade_takes_only_a_short_recorded_value():
    assert web_log.normalize_grade('A') == 'A'
    assert web_log.normalize_grade(' ★2 ') == '★2'
    assert web_log.normalize_grade(None) is None
    assert web_log.normalize_grade('') is None
    # 自由文が紛れ込んだら捨てる（8文字以内だけ通す）
    assert web_log.normalize_grade('栗東坂路で抜群の動き。評価A') is None


def test_collect_keeps_the_grade_the_source_gave(base):
    findings = json.loads(json.dumps(FINDINGS))
    findings['races']['20261018_08_11']['horses']['6']['grade'] = 'A'
    rows = web_log.collect_from_findings(findings)
    g = {r['key']: r['said'].get('grade') for r in rows if r['kind'] == 'horse'}
    assert g['6'] == 'A'
    # 情報源が序列を出していない馬は None のまま。こちらで埋めない
    assert g['9'] is None


def test_grade_is_never_parsed_out_of_the_free_text_note(base):
    """🔴 `note` から正規表現で取り出すと、情報源が付けた値とこちらが解釈した値が
    同じ列に混ざり後から区別できなくなる（2026-09-02 のパドック condition と同型）。"""
    findings = json.loads(json.dumps(FINDINGS))
    h = findings['races']['20261018_08_11']['horses']
    h['9']['note'] = '美浦ウッド 6F82秒4-上がり1F11秒8（強めの追い切り）。評価A'
    assert 'grade' not in h['9']
    rows = web_log.collect_from_findings(findings)
    said = next(r['said'] for r in rows if r['kind'] == 'horse' and r['key'] == '9')
    assert said['grade'] is None, 'note の「評価A」を grade に流用してはいけない'


def test_summary_reports_whether_the_source_is_measurable_at_all(base, tmp_path):
    """率より先に「測れる形か」を見るための3つ: 言及頭数・序列あり・方向あり。"""
    findings = json.loads(json.dumps(FINDINGS))
    findings['races']['20261018_08_11']['horses']['6']['grade'] = 'A'
    with open(os.path.join(base, 'data', 'claude_web_findings.json'), 'w',
              encoding='utf-8') as f:
        json.dump(findings, f, ensure_ascii=False)
    web_log.record(base)
    s = web_log.summarize(base)
    k = s['by_host']['keibanomiryoku.com']['horse']
    assert k['n'] == 2 and k['n_races'] == 1
    assert k['mentions_per_race'] == 2.0
    assert k['with_grade'] == 1          # #6 だけ序列あり
    assert k['with_direction'] == 1      # #9 は direction=None
    # 🔴 全馬型では率が信頼度にならないことを注意文に必ず残す
    assert '信頼度ではない' in s['caveat']
    assert 'mentions_per_race' in s['caveat']


def test_a_v1_ledger_is_read_without_touching_its_rows(base):
    """schema 1→2 は `said.grade` の追加だけ。既存の記録に手を入れずに読める。"""
    web_log.record(base)
    p = web_log.ledger_path(base)
    d = json.load(open(p, encoding='utf-8'))
    assert d['schema_version'] == web_log.SCHEMA_VERSION == 2
    # 本番の台帳（97件）と同じ v1 の形に戻し、grade を持たない行にする
    d['schema_version'] = 1
    for e in d['entries']:
        e['said'].pop('grade', None)
    json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    loaded = web_log.load(base)
    assert len(loaded['entries']) == len(d['entries'])
    s = web_log.summarize(base)
    assert s['by_host']['keibanomiryoku.com']['horse']['with_grade'] == 0
