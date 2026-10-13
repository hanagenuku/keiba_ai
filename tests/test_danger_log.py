"""「危険な人気馬」の警告とその後の着順を残す台帳のテスト。

🔴 この台帳が守らなければならないこと（崩れると記録が嘘になる）:
  1. 判定条件を再実装しない（`marks.danger_favorites` をそのまま使う）
  2. 一度記録した警告を上書きしない（朝に出した警告がそのまま残る）
  3. 結果がまだ無いレースを「外れ」として埋めない
  4. 3着内は**着順**で決め、複勝払戻は**払戻表に行があるか**で決める
     （少頭数で複勝が2着までしか発売されないレースが実在する）
  5. 発走後に条件を満たした警告を、予告と同じ扱いで集計しない
  6. 回収率を出さない

North Star #6 に従い、DB は手打ち dict ではなく**本番と同じ書き込み口**
（`src/utils/db.py::init_db` / `save_dividends_db` / `save_race_schedule`）を通した
実物の sqlite を使う。
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.product import danger_log, marks  # noqa: E402
from src.product import results as results_mod  # noqa: E402
from src.utils import db as db_mod  # noqa: E402

RID = '20261011_05_10'
DATE = '2026-10-11'


def _horse(n, pop, solo, rl=1, odds=4.7, name=None):
    return {'n': n, 'name': name or f'馬{n}', 'pop': pop, 'solo_rank': solo,
            'rl_rank': rl, 'odds': odds}


def _latest(races=None, generated_at='2026-10-11T08:52:41+09:00', coverage=1.0):
    """`data/latest.json` と同じ形（races は会場キーの dict）。"""
    if races is None:
        races = [{
            'race_id': RID, 'r': 10, 'name': 'テレビ静岡賞', 'conf': 53,
            'horses': [_horse(15, 1, 6), _horse(1, 2, 1), _horse(3, 3, 3)],
        }]
    return {'generated_at': generated_at, 'date': '10月11日(日)',
            'data_quality': {'odds_coverage': coverage},
            'races': {'東京': races}}


def _dbs(base_dir, placings, dividends, race_id=RID, post_time='15:10',
         date=DATE):
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    conn = sqlite3.connect(os.path.join(base_dir, 'data', 'history.db'))
    conn.execute('CREATE TABLE horse_history ('
                 'race_id TEXT, horse_num INTEGER, place INTEGER)')
    for num, place in placings.items():
        conn.execute('INSERT INTO horse_history (race_id,horse_num,place) '
                     'VALUES (?,?,?)', (race_id, num, place))
    conn.commit()
    conn.close()
    db_mod.init_db(base_dir)
    if dividends is not None:
        db_mod.save_dividends_db(
            [{'race_id': race_id, 'date': date, 'dividends': dividends}],
            base_dir=base_dir)
    if post_time:
        db_mod.save_race_schedule(
            [{'race_id': race_id, 'date': date, 'racecourse': '東京',
              'race_num': 10, 'post_time': post_time, 'n_horses': len(placings)}],
            base_dir=base_dir)


def _add_placings(base_dir, race_id, placings):
    """2レース目以降の着順を足す（history.db は `_dbs` が作ってある前提）。"""
    conn = sqlite3.connect(os.path.join(base_dir, 'data', 'history.db'))
    for num, place in placings.items():
        conn.execute('INSERT INTO horse_history (race_id,horse_num,place) '
                     'VALUES (?,?,?)', (race_id, num, place))
    conn.commit()
    conn.close()


def _attach(base_dir):
    return danger_log.attach_results(base_dir,
                                     results_mod.open_history(base_dir),
                                     results_mod.open_keiba(base_dir))


# ── 1. 条件の定義は1箇所だけ ───────────────────────────────────────────
def test_condition_is_not_reimplemented():
    """🔴 判定は marks.danger_favorites を呼ぶ。閾値をこちらに書き写さない。

    書き写すと「記事に出した警告」と「台帳の警告」が静かに食い違う。
    """
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'product',
                            'danger_log.py'), encoding='utf-8').read()
    body = src.split('"""', 2)[-1]  # docstring の説明文は対象外
    assert 'danger_favorites' in body
    for name in ('DANGER_MAX_POP', 'DANGER_MIN_SOLO'):
        assert name not in body, f'{name} を danger_log に書き写している'


def test_collect_matches_marks_definition(tmp_path):
    found = danger_log.collect_from_latest(_latest())
    assert [e['num'] for e in found] == [15]
    e = found[0]
    assert (e['pop'], e['solo_rank'], e['date']) == (1, 6, DATE)
    assert e['in_scope'] is True and e['verdict'] == 'skip'
    assert e['result'] is None


def test_horse_outside_the_condition_is_not_recorded():
    """2番人気や AI 上位は入れない（測っていない母集団に広げない）。"""
    races = [{'race_id': RID, 'r': 10, 'conf': 70,
              'horses': [_horse(1, 2, 9), _horse(2, 1, 5)]}]
    assert danger_log.collect_from_latest(_latest(races)) == []


def test_out_of_scope_race_is_recorded_with_a_flag():
    """範囲外（平場・conf 低い）も記録する。記事に出たかは in_scope で分かる。"""
    races = [{'race_id': '20261011_05_01', 'r': 1, 'conf': 53,
              'horses': [_horse(2, 1, 7, odds=2.4)]}]
    got = danger_log.collect_from_latest(_latest(races))
    assert len(got) == 1 and got[0]['in_scope'] is False


def test_date_comes_from_race_id_not_the_display_string():
    """`date` は `10月11日(日)` という表示用文字列なので使わない。"""
    got = danger_log.collect_from_latest(_latest())
    assert got[0]['date'] == '2026-10-11'


def test_race_without_a_usable_race_id_is_skipped():
    races = [{'race_id': None, 'r': 10, 'conf': 70, 'horses': [_horse(1, 1, 9)]}]
    assert danger_log.collect_from_latest(_latest(races)) == []


# ── 2. 上書きしない ───────────────────────────────────────────────────
def test_recording_twice_adds_nothing(tmp_path):
    d = str(tmp_path)
    assert danger_log.record(d, _latest())['added'] == 1
    r = danger_log.record(d, _latest())
    assert r['added'] == 0 and r['already'] == 1
    assert len(danger_log.load(d)['entries']) == 1


def test_later_refresh_does_not_overwrite_the_morning_warning(tmp_path):
    """🔴 当日 11:30 の refresh でオッズが動いても朝の警告はそのまま残す。

    記事に載せたのは朝の値なので、後の値で上書きすると記録が記事と食い違う。
    """
    d = str(tmp_path)
    danger_log.record(d, _latest())
    noon = _latest(generated_at='2026-10-11T11:30:00+09:00')
    noon['races']['東京'][0]['horses'][0]['odds'] = 9.9
    noon['races']['東京'][0]['conf'] = 71
    danger_log.record(d, noon)
    entries = danger_log.load(d)['entries']
    assert len(entries) == 1, '同じ馬の警告が2行になっている'
    e = entries[0]
    assert e['odds'] == 4.7 and e['conf'] == 53
    assert e['snapshot'] == '2026-10-11T08:52:41+09:00'


def test_a_new_horse_in_a_later_snapshot_is_added(tmp_path):
    """後の refresh で初めて条件を満たした馬は足す（朝に落ちたレースの回収）。"""
    d = str(tmp_path)
    danger_log.record(d, _latest())
    later = _latest()
    later['races']['東京'].append(
        {'race_id': '20261011_05_12', 'r': 12, 'conf': 49,
         'horses': [_horse(6, 1, 9, odds=2.9)]})
    assert danger_log.record(d, later)['added'] == 1
    assert len(danger_log.load(d)['entries']) == 2


def test_unknown_schema_version_is_refused(tmp_path):
    d = str(tmp_path)
    os.makedirs(os.path.join(d, 'data'))
    with open(danger_log.ledger_path(d), 'w', encoding='utf-8') as f:
        json.dump({'schema_version': 99, 'entries': []}, f)
    try:
        danger_log.load(d)
    except ValueError as e:
        assert 'schema_version' in str(e)
    else:
        raise AssertionError('版違いを黙って読んでいる（過去の警告が消える）')


# ── 3〜4. 結果の追記 ──────────────────────────────────────────────────
def test_place_and_payout_are_attached(tmp_path):
    d = str(tmp_path)
    danger_log.record(d, _latest())
    _dbs(d, {15: 1, 1: 2, 3: 3}, {'fukusho': [{'num': 15, 'payout': 170},
                                              {'num': 1, 'payout': 120},
                                              {'num': 3, 'payout': 110}]})
    r = _attach(d)
    assert (r['attached'], r['pending']) == (1, 0)
    res = danger_log.load(d)['entries'][0]['result']
    assert res['place'] == 1 and res['top3'] is True
    assert res['fukusho_payout'] == 170 and res['fukusho_sold'] is True
    assert res['n_finishers'] == 3


def test_pending_race_is_not_recorded_as_a_miss(tmp_path):
    """🔴 払戻が1行も無い＝まだ結果を取っていない。空の結果で埋めない。"""
    d = str(tmp_path)
    danger_log.record(d, _latest())
    _dbs(d, {15: 1}, None)
    r = _attach(d)
    assert (r['attached'], r['pending']) == (0, 1)
    assert danger_log.load(d)['entries'][0]['result'] is None


def test_payouts_without_placings_stay_pending(tmp_path):
    """着順が引けないレースは推測で埋めずに保留する（10/10 京都の実例）。"""
    d = str(tmp_path)
    danger_log.record(d, _latest())
    _dbs(d, {}, {'fukusho': [{'num': 2, 'payout': 110}]})
    assert _attach(d)['pending'] == 1
    assert danger_log.load(d)['entries'][0]['result'] is None


def test_attaching_twice_does_not_rewrite(tmp_path):
    d = str(tmp_path)
    danger_log.record(d, _latest())
    _dbs(d, {15: 1}, {'fukusho': [{'num': 15, 'payout': 170}]})
    _attach(d)
    first = danger_log.load(d)['entries'][0]['result']['attached_at']
    r = _attach(d)
    assert (r['attached'], r['already']) == (0, 1)
    assert danger_log.load(d)['entries'][0]['result']['attached_at'] == first


def test_third_place_without_a_fukusho_payout_is_still_top3(tmp_path):
    """🔴 少頭数で複勝が2着までのレース。3着内の事実を払戻の有無で消さない。

    実データの反例: 2026-09-27 中山9R は7頭立てで3着に複勝配当が無い。
    「払戻が無い＝3着内ではない」と書くと、ここで嘘の記録になる。
    """
    d = str(tmp_path)
    races = [{'race_id': RID, 'r': 10, 'conf': 70,
              'horses': [_horse(2, 1, 7)]}]
    danger_log.record(d, _latest(races))
    _dbs(d, {3: 1, 7: 2, 2: 3, 1: 4, 4: 5, 5: 6, 6: 7},
         {'fukusho': [{'num': 3, 'payout': 180}, {'num': 7, 'payout': 110}]})
    _attach(d)
    res = danger_log.load(d)['entries'][0]['result']
    assert res['place'] == 3 and res['top3'] is True
    assert res['fukusho_payout'] is None   # 発売されていないので払戻は無い
    assert res['fukusho_sold'] is True     # ただし複勝自体はこのレースにある


# ── 5. 発走後の記録を予告と混ぜない ───────────────────────────────────
def test_warning_recorded_after_the_start_is_flagged(tmp_path):
    """🔴 発走後に条件を満たした馬を「的中予告」として数えない。"""
    d = str(tmp_path)
    danger_log.record(d, _latest(generated_at='2026-10-11T16:00:00+09:00'))
    _dbs(d, {15: 1}, {'fukusho': [{'num': 15, 'payout': 170}]}, post_time='15:10')
    r = _attach(d)
    assert r['late'], '発走後の記録に印が付いていない'
    assert danger_log.load(d)['entries'][0]['result']['before_post'] is False
    s = danger_log.summarize(d)
    assert s['with_result'] == 0 and s['excluded_late'] == 1
    assert danger_log.summarize(d, include_late=True)['with_result'] == 1


def test_before_post_is_none_when_the_post_time_is_unknown(tmp_path):
    """発走時刻が無い開催（race_schedule は土日しか埋まらない）は不明のまま。"""
    d = str(tmp_path)
    danger_log.record(d, _latest())
    _dbs(d, {15: 1}, {'fukusho': [{'num': 15, 'payout': 170}]}, post_time=None)
    _attach(d)
    assert danger_log.load(d)['entries'][0]['result']['before_post'] is None
    assert danger_log.summarize(d)['with_result'] == 1  # 不明は除外しない


def test_before_post_handles_naive_and_missing_values():
    assert danger_log.before_post('2026-10-11T08:52:41', DATE, '15:10') is True
    assert danger_log.before_post(None, DATE, '15:10') is None
    assert danger_log.before_post('2026-10-11T08:52:41+09:00', DATE, None) is None
    assert danger_log.before_post('ごみ', DATE, '15:10') is None


# ── 6. 集計は事実だけ ─────────────────────────────────────────────────
def test_summary_counts_facts_only(tmp_path):
    d = str(tmp_path)
    races = [{'race_id': RID, 'r': 10, 'conf': 70, 'horses': [_horse(15, 1, 6)]},
             {'race_id': '20261011_05_01', 'r': 1, 'conf': 53,
              'horses': [_horse(2, 1, 7)]}]
    danger_log.record(d, _latest(races))
    _dbs(d, {15: 1}, {'fukusho': [{'num': 15, 'payout': 170}]})
    _add_placings(d, '20261011_05_01', {9: 1, 2: 8})
    db_mod.save_race_schedule(
        [{'race_id': '20261011_05_01', 'date': DATE, 'racecourse': '東京',
          'race_num': 1, 'post_time': '10:05', 'n_horses': 2}], base_dir=d)
    db_mod.save_dividends_db(
        [{'race_id': '20261011_05_01', 'date': DATE,
          'dividends': {'fukusho': [{'num': 9, 'payout': 150}]}}], base_dir=d)
    _attach(d)
    s = danger_log.summarize(d)
    assert (s['total'], s['with_result'], s['top3']) == (2, 2, 1)
    assert s['places'] == [1, 8]
    assert s['fukusho_payouts'] == [170]
    assert danger_log.summarize(d, scope_only=True)['total'] == 1
    # 🔴 回収率に当たるキーを作らない（賭け金の仮定が必要になる）
    for key in s:
        assert 'roi' not in key and 'return' not in key and 'recovery' not in key


def test_summary_does_not_print_a_recovery_rate():
    """集計の出力に回収率を足さないための歯止め。"""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'scripts',
                            'record_danger_favorites.py'), encoding='utf-8').read()
    assert '回収率' in src  # 「出さない」と書いてある
    assert '回収率:' not in src and 'ROI' not in src


# ── 本番の予想に触らない ──────────────────────────────────────────────
def test_it_does_not_touch_prediction_modules():
    """この層が本番の予想・買い目の経路を読まない（後付け層にしないため）。"""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'product',
                            'danger_log.py'), encoding='utf-8').read()
    for banned in ('src.features.engine', 'src.betting', 'src.models',
                   'calc_all', 'make_bets'):
        assert banned not in src, f'{banned} を読んでいる'


def test_marks_definition_is_still_the_measured_one():
    """測った母集団（市場1番人気 × 素のAI順位>=6）から広げない。"""
    assert (marks.DANGER_MAX_POP, marks.DANGER_MIN_SOLO) == (1, 6)


# ── ワークフローに入れても無害か / 実際に繋がっているか ──────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(BASE, 'scripts', 'record_danger_favorites.py')


def test_cli_is_noop_without_latest_json(tmp_path):
    """latest.json が無い環境でも失敗しない（非ブロッキングでも落とさない）。"""
    import subprocess
    r = subprocess.run([sys.executable, CLI, '--base-dir', str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'latest.json' in r.stderr


def test_cli_skips_attaching_without_the_dbs(tmp_path):
    """🔴 DB の実体が無いときに結果を書かない（全部外れとして残るのを防ぐ）。"""
    import subprocess
    r = subprocess.run([sys.executable, CLI, '--base-dir', str(tmp_path),
                        '--attach-results'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 'スキップ' in r.stderr
    assert not os.path.exists(danger_log.ledger_path(str(tmp_path)))


def test_cli_round_trip(tmp_path):
    """記録 → 結果追記 → 集計 が CLI でも通ること。"""
    import subprocess
    d = str(tmp_path)
    os.makedirs(os.path.join(d, 'data'))
    with open(os.path.join(d, 'data', 'latest.json'), 'w', encoding='utf-8') as f:
        json.dump(_latest(), f, ensure_ascii=False)
    _dbs(d, {15: 1, 1: 2, 3: 3}, {'fukusho': [{'num': 15, 'payout': 170}]})
    for args in ([], ['--attach-results']):
        assert subprocess.run([sys.executable, CLI, '--base-dir', d] + args,
                              capture_output=True, text=True).returncode == 0
    out = subprocess.run([sys.executable, CLI, '--base-dir', d, '--summary'],
                         capture_output=True, text=True)
    assert '3着内 1/1' in out.stdout
    assert '回収率' not in out.stdout   # 🔴 集計に回収率を出さない


def _wf(name):
    return open(os.path.join(BASE, '.github', 'workflows', name),
                encoding='utf-8').read()


def test_weekend_workflow_records_and_attaches():
    s = _wf('weekend.yml')
    assert 'record_danger_favorites.py --base-dir .' in s
    assert 'record_danger_favorites.py --base-dir . --attach-results' in s


def test_results_workflow_only_attaches():
    """🔴 夜（18:40）は記録しない。全レース発走済みで後出しの警告になる。"""
    s = _wf('sunday-results.yml')
    assert 'record_danger_favorites.py --base-dir . --attach-results' in s
    bare = [ln for ln in s.splitlines()
            if 'record_danger_favorites.py' in ln and '--attach-results' not in ln]
    assert not bare, f'結果取得のワークフローで記録している: {bare}'
