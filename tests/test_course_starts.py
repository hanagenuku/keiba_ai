"""距離単位のコース物理DB data/course_starts.json の契約テスト。

🔑 ユーザー指示（2026-09-15 第2版 §19「禁止すること」）を構造で守る:
   禁止1 数字の推測（コース図を見て「たぶん350m」）は入れない
   禁止2 本線合流距離を初角距離として登録しない
   禁止5 欠損を0にしない → NULL + status で管理
"""
import json
import os

import pytest

from src.features.race_shape import (
    course_start_record,
    start_to_first_corner_m,
)

USABLE_LEVEL = {'A', 'C'}

# 資料が「本線合流までと混同するな」と注記している組（初角までは未確定）
MERGE_AMBIGUOUS = {('東京', '芝', 1800), ('東京', '芝', 2000), ('札幌', '芝', 1500)}

# 第2版が芝スタートと明記したダート（第1版は4つしか挙げていなかった）
DOCUMENTED_TURF_STARTS = {
    ('東京', 'ダート', 1600),
    ('中山', 'ダート', 1200),
    ('京都', 'ダート', 1400),
    ('阪神', 'ダート', 1400),
    ('阪神', 'ダート', 2000),
    ('中京', 'ダート', 1400),
    ('新潟', 'ダート', 1200),
    ('福島', 'ダート', 1150),
}


@pytest.fixture(scope='module')
def S():
    with open(os.path.join('data', 'course_starts.json'), encoding='utf-8') as f:
        return json.load(f)['starts']


def test_every_row_declares_a_status(S):
    """禁止5: 欠損を黙って0にしない。全行が status を宣言していること。"""
    valid = {'ok', 'needs_check', 'merge_ambiguous', 'none'}
    for k, e in S.items():
        assert e.get('start_to_first_corner_status') in valid, k


def test_number_only_when_status_is_ok(S):
    """数値があるのは status=ok の行だけ。逆も成り立つ。"""
    for k, e in S.items():
        has_num = e.get('start_to_first_corner_m') is not None
        assert has_num == (e['start_to_first_corner_status'] == 'ok'), k


def test_every_number_declares_a_level(S):
    """禁止1: 出典レベルの無い数値を入れない（推測の入り込む隙をなくす）。"""
    for k, e in S.items():
        if e.get('start_to_first_corner_m') is not None:
            assert e.get('start_to_first_corner_level') in USABLE_LEVEL, \
                f'{k}: level={e.get("start_to_first_corner_level")}'


def test_getter_refuses_everything_but_ok(S):
    """status が ok 以外の行は getter が None を返す（呼び出し側で埋められない）。"""
    for e in S.values():
        if e['start_to_first_corner_status'] == 'ok':
            continue
        assert start_to_first_corner_m(
            e['venue'], e['surface'], e['distance_m']) is None, e


def test_merge_distance_is_never_filed_as_first_corner(S):
    """禁止2: 本線合流までの距離を初角距離の列に入れない。

    資料が名指しした3組（東京芝1800/2000・札幌芝1500）は数値を持たない。
    """
    flagged = {(e['venue'], e['surface'], e['distance_m'])
               for e in S.values()
               if e['start_to_first_corner_status'] == 'merge_ambiguous'}
    assert flagged == MERGE_AMBIGUOUS
    for v, sf, d in MERGE_AMBIGUOUS:
        assert start_to_first_corner_m(v, sf, d) is None

    # 中京芝1600 の200mも「本線合流まで」なので別キーに入っている
    rec = course_start_record('中京', '芝', 1600)
    assert rec.get('start_to_main_course_merge_m') is not None


def test_straight_course_is_not_a_missing_value(S):
    """§16: 「欠損」と「コーナーが存在しない」を同じ扱いにしない。"""
    rec = course_start_record('新潟', '芝', 1000)
    assert rec['start_to_first_corner_status'] == 'none'
    assert rec['first_corner'] == 'NONE'
    assert rec['start_to_first_corner_m'] is None
    # 「存在しない」は1組だけ
    none_rows = [e for e in S.values()
                 if e['start_to_first_corner_status'] == 'none']
    assert len(none_rows) == 1


def test_documented_official_numbers_reach_the_getter():
    """資料が JRA公式と明記した数値が実際に届くこと（台帳に入れて終わりにしない）。"""
    assert start_to_first_corner_m('中山', '芝', 2000) == 404.9
    assert start_to_first_corner_m('中山', '芝', 1200) == 275.0
    assert start_to_first_corner_m('京都', '芝', 2000) == 309.0
    assert start_to_first_corner_m('中京', 'ダート', 1800) == 291.8
    assert start_to_first_corner_m('札幌', 'ダート', 1700) == 240.0


def test_turf_start_flag_is_only_the_documented_eight(S):
    """芝スタートは独立フラグ。資料が明記した8つ以外を推測で足さない。"""
    flagged = {(e['venue'], e['surface'], e['distance_m'])
               for e in S.values() if e['turf_start']}
    assert flagged == DOCUMENTED_TURF_STARTS, f'差分: {flagged ^ DOCUMENTED_TURF_STARTS}'


def test_turf_start_distance_only_where_documented(S):
    """芝区間の長さが分かっているのは東京ダ1600だけ。他は None のまま。"""
    withm = {(e['venue'], e['surface'], e['distance_m'])
             for e in S.values() if e.get('turf_start_distance_m') is not None}
    assert withm == {('東京', 'ダート', 1600)}
    assert course_start_record('東京', 'ダート', 1600)['turf_start_distance_m'] == 150.0


def test_dirt_has_no_inner_outer_variant(S):
    """ダートは内外回りの区別が無い。"""
    for k, e in S.items():
        if e['surface'] == 'ダート':
            assert e.get('course_variant') is None, k


def test_venue_constants_are_present_everywhere(S):
    """1周・直線・高低差・回りは会場×馬場で必ず入っていること。"""
    for k, e in S.items():
        for f in ('one_lap_distance_m', 'straight_distance_m',
                  'course_elevation_m', 'turn_direction'):
            assert e.get(f) is not None, f'{k}: {f} が無い'


def test_every_venue_surface_pair_is_covered(S):
    pairs = {(e['venue'], e['surface']) for e in S.values()}
    venues = {'札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'}
    assert pairs == {(v, s) for v in venues for s in ('芝', 'ダート')}


def test_getter_does_not_consult_the_old_venue_level_file():
    """出典は course_starts.json だけ。旧 course_physical.json を見ない。

    両方見ていた頃、第2版が merge_ambiguous に格下げした 東京芝2000 で
    旧ファイルの 100m（＝本線合流までの距離）が漏れていた（禁止2）。
    """
    with open(os.path.join('data', 'course_physical.json'), encoding='utf-8') as f:
        ph = json.load(f)['distances']
    with open(os.path.join('data', 'course_starts.json'), encoding='utf-8') as f:
        st = json.load(f)['starts']
    for k, e in ph.items():
        if e.get('start_to_first_corner_m') is None:
            continue
        row = st.get(k)
        if row is None or row['start_to_first_corner_status'] == 'ok':
            continue
        # 第2版が数値を持たないと判断した組は、旧ファイルに値があっても出さない
        assert start_to_first_corner_m(
            e_venue := row['venue'], row['surface'], row['distance_m']) is None, k
