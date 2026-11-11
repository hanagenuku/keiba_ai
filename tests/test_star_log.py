"""☆（1番人気を上回る可能性）の台帳のテスト。

🔴 この台帳が守らなければならないこと（崩れると記録が嘘になる）:
  1. ☆の判定を再実装しない（`latest.json` の `race.star` をそのまま読む）
  2. 一度記録した☆を上書きしない（朝に出した☆がそのまま残る）
  3. 結果がまだ無いレースを「外れ」として埋めない
  4. ☆の主張は**着順比較**なので、1番人気の着順も一緒に残し `beat_p1` を立てる
  5. 当日の refresh で☆が動いたら、集計は**最初のスナップショットの1件だけ**
  6. 発走後に記録された☆を予告と同じ扱いで集計しない
  7. 回収率を出さない

North Star #6 に従い、DB は本番と同じ書き込み口（`src/utils/db.py`）を通した
実物の sqlite を使う。
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import star  # noqa: E402
from src.product import results as results_mod, star_log  # noqa: E402
from src.utils import db as db_mod  # noqa: E402

RID = '20261114_05_10'
DATE = '2026-11-14'


def _horse(n, pop, odds=7.2, rl=2, solo=1, name=None):
    return {'n': n, 'name': name or f'馬{n}', 'pop': pop, 'odds': odds,
            'rl_rank': rl, 'solo_rank': solo, 'star_prob': None}


def _race(star_num=7, prob=0.68, race_id=RID, horses=None):
    horses = horses or [_horse(7, 3), _horse(2, 1, odds=2.1, rl=1),
                        _horse(5, 2, odds=4.0)]
    return {
        'race_id': race_id, 'r': 10, 'name': 'テスト特別', 'conf': 61,
        'horses': horses,
        'star': ({'n': star_num, 'name': f'馬{star_num}', 'prob': prob,
                  'note': star.STAR_NOTE, 'tau': star.STAR_TAU}
                 if star_num else None),
    }


def _latest(races=None, updated_at='2026-11-14T08:55:00+09:00'):
    return {'updated_at': updated_at, 'date': '11月14日(土)',
            'races': {'東京': races if races is not None else [_race()]}}


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


def _attach(base_dir):
    return star_log.attach_results(base_dir,
                                   results_mod.open_history(base_dir),
                                   results_mod.open_keiba(base_dir))


# ── 1. 判定を再実装しない ───────────────────────────────────────────────
def test_threshold_is_not_copied_into_the_ledger():
    """🔴 閾値は star.py の1か所だけ。台帳に書き写さない。"""
    with open(os.path.join(os.path.dirname(__file__), '..', 'src', 'product',
                           'star_log.py'), encoding='utf-8') as f:
        src = f.read()
    assert '0.60' not in src and '0.6' not in src.replace('0.60', '')
    assert 'STAR_TAU' not in src.split('"""', 2)[-1], \
        'star_log が STAR_TAU を参照している（判定を持ってはいけない）'


def test_collect_reads_race_star_as_is(tmp_path):
    found = star_log.collect_from_latest(_latest())
    assert len(found) == 1
    e = found[0]
    assert (e['num'], e['prob'], e['tau']) == (7, 0.68, star.STAR_TAU)
    assert e['p1_num'] == 2, '1番人気の馬番を残していない'
    assert e['date'] == DATE and e['race_id'] == RID
    assert e['pop'] == 3 and e['odds'] == 7.2


def test_race_without_star_is_skipped():
    assert star_log.collect_from_latest(_latest([_race(star_num=None)])) == []


def test_date_comes_from_race_id():
    """`latest.json` の date は年の無い表示文字列。日付は race_id だけが信用できる。"""
    e = star_log.collect_from_latest(_latest())[0]
    assert e['date'] == '2026-11-14'


def test_race_without_a_usable_race_id_is_skipped():
    r = _race(race_id='x')
    assert star_log.collect_from_latest(_latest([r])) == []


def test_p1_num_is_none_when_favourite_is_ambiguous():
    """人気1位が2頭（オッズ全滅の馬番順フォールバック）なら比較相手を作らない。"""
    hs = [_horse(7, 3), _horse(2, 1), _horse(5, 1)]
    e = star_log.collect_from_latest(_latest([_race(horses=hs)]))[0]
    assert e['p1_num'] is None


# ── 2. 上書きしない ─────────────────────────────────────────────────────
def test_recording_twice_adds_nothing(tmp_path):
    d = str(tmp_path)
    assert star_log.record(d, _latest())['added'] == 1
    r = star_log.record(d, _latest())
    assert (r['added'], r['already']) == (0, 1)
    assert len(star_log.load(d)['entries']) == 1


def test_later_refresh_does_not_overwrite_the_morning_star(tmp_path):
    """🔴 昼に☆が別の馬へ動いても、朝の☆は残る（記事に載ったのは朝の分）。"""
    d = str(tmp_path)
    star_log.record(d, _latest())
    star_log.record(d, _latest([_race(star_num=5, prob=0.63)],
                               updated_at='2026-11-14T14:05:00+09:00'))
    ents = star_log.load(d)['entries']
    assert len(ents) == 2
    morning = [e for e in ents if e['num'] == 7][0]
    assert morning['prob'] == 0.68
    assert morning['snapshot'] == '2026-11-14T08:55:00+09:00'


def test_summarize_counts_only_the_first_star_per_race(tmp_path):
    """🔴 同じレースに2件あっても、集計は最初のスナップショットだけ。"""
    d = str(tmp_path)
    star_log.record(d, _latest())
    star_log.record(d, _latest([_race(star_num=5, prob=0.63)],
                               updated_at='2026-11-14T14:05:00+09:00'))
    _dbs(d, {7: 1, 2: 4, 5: 9}, {'fukusho': [{'num': 7, 'payout': 230},
                                             {'num': 2, 'payout': 130},
                                             {'num': 5, 'payout': 180}]})
    _attach(d)
    s = star_log.summarize(d)
    assert s['entries'] == 2
    assert s['races'] == 1, 'レース単位にまとめていない'
    assert s['with_result'] == 1
    assert s['beat_p1'] == 1


def test_unknown_schema_version_is_refused(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    p = star_log.ledger_path(d)
    with open(p, encoding='utf-8') as f:
        raw = f.read().replace('"schema_version": 1', '"schema_version": 99')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(raw)
    with pytest.raises(ValueError):
        star_log.load(d)


# ── 3・4. 結果の付け方 ──────────────────────────────────────────────────
def test_beat_p1_is_recorded(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 2, 2: 5, 5: 9}, {'fukusho': [{'num': 7, 'payout': 260},
                                             {'num': 2, 'payout': 140}]})
    r = _attach(d)
    assert (r['attached'], r['pending']) == (1, 0)
    res = star_log.load(d)['entries'][0]['result']
    assert res['place'] == 2 and res['top3'] is True
    assert res['p1_place'] == 5 and res['p1_top3'] is False
    assert res['beat_p1'] is True
    assert res['fukusho_payout'] == 260 and res['fukusho_sold'] is True


def test_beat_p1_false_when_favourite_finishes_ahead(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 6, 2: 1, 5: 3}, {'fukusho': [{'num': 2, 'payout': 120},
                                             {'num': 5, 'payout': 150}]})
    _attach(d)
    res = star_log.load(d)['entries'][0]['result']
    assert res['beat_p1'] is False
    assert res['top3'] is False and res['p1_top3'] is True
    # 🔴 3着内でないのに払戻が付いていない＝正常。外れとして扱う
    assert res['fukusho_payout'] is None and res['fukusho_sold'] is True


def test_pending_race_is_not_recorded_as_a_miss(tmp_path):
    """🔴 払戻が1行も無い＝まだ結果を取っていない。空の結果で埋めない。"""
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 1, 2: 2, 5: 3}, None)
    r = _attach(d)
    assert (r['attached'], r['pending']) == (0, 1)
    assert star_log.load(d)['entries'][0]['result'] is None


def test_payouts_without_placings_stay_pending(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {}, {'fukusho': [{'num': 7, 'payout': 230}]})
    r = _attach(d)
    assert (r['attached'], r['pending']) == (0, 1)


def test_attaching_twice_does_not_rewrite(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 1, 2: 2, 5: 3}, {'fukusho': [{'num': 7, 'payout': 230},
                                             {'num': 2, 'payout': 130},
                                             {'num': 5, 'payout': 150}]})
    _attach(d)
    first = star_log.load(d)['entries'][0]['result']['attached_at']
    r2 = _attach(d)
    assert (r2['attached'], r2['already']) == (0, 1)
    assert star_log.load(d)['entries'][0]['result']['attached_at'] == first


def test_third_place_without_a_fukusho_payout_is_still_top3(tmp_path):
    """少頭数で複勝が2着までしか発売されないレースがある（着順で決める）。"""
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 3, 2: 1, 5: 2}, {'fukusho': [{'num': 2, 'payout': 110},
                                             {'num': 5, 'payout': 130}]})
    _attach(d)
    res = star_log.load(d)['entries'][0]['result']
    assert res['top3'] is True
    assert res['fukusho_payout'] is None
    assert res['beat_p1'] is False


def test_beat_p1_is_none_when_favourite_unknown(tmp_path):
    d = str(tmp_path)
    hs = [_horse(7, 3), _horse(2, 1), _horse(5, 1)]
    star_log.record(d, _latest([_race(horses=hs)]))
    _dbs(d, {7: 1, 2: 2, 5: 3}, {'fukusho': [{'num': 7, 'payout': 230}]})
    _attach(d)
    res = star_log.load(d)['entries'][0]['result']
    assert res['beat_p1'] is None and res['p1_place'] is None


# ── 6. 発走後の記録 ─────────────────────────────────────────────────────
def test_star_recorded_after_the_start_is_flagged(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest(updated_at='2026-11-14T15:40:00+09:00'))
    _dbs(d, {7: 1, 2: 2, 5: 3}, {'fukusho': [{'num': 7, 'payout': 230}]},
         post_time='15:10')
    r = _attach(d)
    assert r['late'], '発走後の記録に印が付いていない'
    res = star_log.load(d)['entries'][0]['result']
    assert res['before_post'] is False
    # 集計からは除かれる
    assert star_log.summarize(d)['with_result'] == 0
    assert star_log.summarize(d, include_late=True)['with_result'] == 1


def test_before_post_is_none_when_the_post_time_is_unknown(tmp_path):
    d = str(tmp_path)
    star_log.record(d, _latest())
    _dbs(d, {7: 1, 2: 2, 5: 3}, {'fukusho': [{'num': 7, 'payout': 230}]},
         post_time=None)
    _attach(d)
    res = star_log.load(d)['entries'][0]['result']
    assert res['before_post'] is None
    assert star_log.summarize(d)['with_result'] == 1, \
        '不明を発走後として除いてはいけない'


# ── 7. 回収率を出さない ─────────────────────────────────────────────────
def test_summarize_has_no_roi():
    s = star_log.summarize('.')
    for k in s:
        assert 'roi' not in k and '回収' not in k
