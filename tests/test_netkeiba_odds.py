"""netkeiba の結果ページから実オッズを取り出すパーサの回帰テスト。

North Star #6 に従い、綺麗な自作HTMLではなく**実際に返ってきた構造**で検証する。
2026-08-24 のプローブで実測した見出しをそのまま使う:

  ['着順','枠番','馬番','馬名','性齢','斤量','騎手','タイム','着差',
   'ﾀｲﾑ指数…','ﾀｲﾑ指数M…','ｽﾀｰﾄ指数','追走指数','上がり指数','通過','上り',
   '単勝','人気','馬体重','調教ﾀｲﾑ','厩舎ｺﾒﾝﾄ','備考','調教師','馬主','賞金(万円)']

🔑 有料列（ﾀｲﾑ指数・ｽﾀｰﾄ指数・追走指数・上がり指数・調教ﾀｲﾑ・厩舎ｺﾒﾝﾄ）は
   `**` または空で返る。**そこには触らない**ことをテストで固定する。
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_netkeiba_odds import init_out_db, parse_result_table

_HEAD = ['着順', '枠番', '馬番', '馬名', '性齢', '斤量', '騎手', 'タイム', '着差',
         'ﾀｲﾑ指数タイム指数(通常)タイム指数マスター', 'ﾀｲﾑ指数M', 'ｽﾀｰﾄ指数',
         '追走指数', '上がり指数', '通過', '上り', '単勝', '人気', '馬体重',
         '調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ', '備考', '調教師', '馬主', '賞金(万円)']

# 2026-08-24 のプローブが実際に返した3行（2023年ダービー）
_ROWS = [
    ['1', '6', '12', 'タスティエーラ', '牡3', '57', 'レーン', '2:25.2', '',
     '**', '**', '**', '**', '**', '4-4-4-4', '33.5', '8.3', '4', '478(0)',
     '', '', '', '[東]堀宣行', 'キャロットファーム', '32,734.9'],
    ['2', '3', '5', 'ソールオリエンス', '牡3', '57', '横山武史', '2:25.2', 'クビ',
     '**', '**', '**', '**', '**', '6-6-6-6', '33.3', '1.8', '1', '460(-2)',
     '', '', '', '[東]手塚貴久', '社台レースホース', '12,781.4'],
    ['3', '6', '11', 'ハーツコンチェルト', '牡3', '57', '松山弘平', '2:25.2', 'ハナ',
     '**', '**', '**', '**', '**', '16-14-6-6', '33.4', '25.6', '6', '494(-4)',
     '', '', '', '[東]武井亮', 'グリーンファーム', '7,890.7'],
]


def _html(head=None, rows=None, extra_table=True):
    head = head or _HEAD
    rows = rows if rows is not None else _ROWS
    th = ''.join(f'<th>{h}</th>' for h in head)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    # 実ページは table が8個あり、結果表は最初とは限らない
    noise = '<table><tr><th>コース</th></tr><tr><td>芝2400m</td></tr></table>'
    return f'<html><body>{noise if extra_table else ""}' \
           f'<table><tr>{th}</tr>{body}</table></body></html>'


class TestParse:
    def test_extracts_win_odds_for_all_horses(self):
        rows = parse_result_table(_html())
        assert len(rows) == 3
        assert [r['win_odds'] for r in rows] == [8.3, 1.8, 25.6]
        assert [r['popularity'] for r in rows] == [4, 1, 6]
        assert [r['horse_num'] for r in rows] == [12, 5, 11]
        assert [r['place'] for r in rows] == [1, 2, 3]

    def test_body_weight_strips_the_diff(self):
        rows = parse_result_table(_html())
        assert [r['body_weight'] for r in rows] == [478, 460, 494]

    def test_finds_the_result_table_among_others(self):
        """結果表が最初の table とは限らない（実ページは table 8個）。"""
        assert len(parse_result_table(_html(extra_table=True))) == 3

    def test_columns_are_looked_up_by_header_not_position(self):
        """列順が変わっても壊れないこと。

        2026-08-03③ でJRA公式の結果ページの列順が変わり、popularity /
        body_weight / trainer が1ヶ月サイレントに壊れた。位置決め打ちにしない。
        """
        order = [16, 17, 2, 3, 18, 0]          # 単勝,人気,馬番,馬名,馬体重,着順
        head = [_HEAD[i] for i in order]
        rows = [[r[i] for i in order] for r in _ROWS]
        got = parse_result_table(_html(head, rows))
        assert len(got) == 3
        assert [r['win_odds'] for r in got] == [8.3, 1.8, 25.6]
        assert [r['horse_num'] for r in got] == [12, 5, 11]


class TestPaidColumnsAreNotTouched:
    def test_paid_columns_are_not_in_the_output(self):
        """有料列（`**`）を出力に含めないこと。無料の範囲だけ扱う。"""
        for r in parse_result_table(_html()):
            assert set(r) == {'horse_num', 'horse_name', 'win_odds',
                              'popularity', 'body_weight', 'place'}
            for v in r.values():
                assert v != '**'

    def test_paid_placeholder_never_becomes_a_number(self):
        """有料列が単勝の位置に来ても `**` を数値にしないこと。"""
        rows = [list(r) for r in _ROWS]
        for r in rows:
            r[16] = '**'                      # 単勝が有料扱いで返ってきた場合
        got = parse_result_table(_html(rows=rows))
        assert all(r['win_odds'] is None for r in got)


class TestNoResultTable:
    def test_returns_empty_when_no_result_table(self):
        assert parse_result_table('<html><body><p>なし</p></body></html>') == []

    def test_returns_empty_when_odds_column_missing(self):
        """単勝列が無いページ（＝欲しいものが無い）は空を返すこと。"""
        head = [h for h in _HEAD if h != '単勝']
        rows = [[c for i, c in enumerate(r) if i != 16] for r in _ROWS]
        assert parse_result_table(_html(head, rows)) == []


class TestOutputDbIsPrivate:
    def test_writes_only_under_data_private(self):
        """保存先が data/private 配下であること（公開リポジトリに出さない）。"""
        from scripts.fetch_netkeiba_odds import OUT_DB
        assert os.path.join('data', 'private') in OUT_DB

    def test_schema_is_resumable(self, tmp_path):
        """中断しても再開できるよう fetch_log を持つこと（North Star #4）。"""
        conn = init_out_db(str(tmp_path / 'nk.db'))
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'netkeiba_odds', 'fetch_log', 'day_map'} <= names
        conn.execute("INSERT INTO netkeiba_odds VALUES "
                     "('r1',1,'ウマ',8.3,4,478,1,'202301010101','now')")
        conn.execute("INSERT OR REPLACE INTO netkeiba_odds VALUES "
                     "('r1',1,'ウマ',8.4,4,478,1,'202301010101','now')")
        conn.commit()
        assert conn.execute('SELECT COUNT(*) FROM netkeiba_odds').fetchone()[0] == 1
        conn.close()


class TestResumeAcrossRuns:
    """🔴 2026-08-25 の事故の回帰テスト。

    ワークフローは `actions/download-artifact@v4` を素で使って「前回の続き」を
    復元しているつもりだったが、このアクションは **同じrunの中で upload された
    artifact しか見ない**。前回のrunのものを取るには run-id の指定が要る。
    そのため run#2 が「取得済み 0 / 残り 3,969」から再開し、run#1 の分を捨てて
    取り直していた（何度走らせても同じ556レースを取り続ける状態だった）。
    `continue-on-error: true` を付けていたので警告も埋もれていた。

    再開が壊れると「進んでいるように見えて1件も進まない」ので、静かに壊れる型。
    """

    @staticmethod
    def _wf():
        import yaml
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         '.github', 'workflows', 'fetch-netkeiba-odds.yml')
        with open(p, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _restore_step(self):
        for st in self._wf()['jobs']['fetch']['steps']:
            if '復元' in st.get('name', ''):
                return st
        pytest.fail('前回の成果物を復元するステップが無い（再開できない）')

    def test_restore_does_not_use_bare_download_artifact(self):
        """download-artifact を使うなら run-id 必須。素で使うと前回分を見ない。"""
        st = self._restore_step()
        uses = st.get('uses', '')
        if 'download-artifact' in uses:
            assert 'run-id' in (st.get('with') or {}), (
                'download-artifact は同じrunの artifact しか見ない。'
                '前回のrunから復元するには run-id が要る')

    def test_restore_targets_a_previous_successful_run(self):
        """直近の**成功run**の artifact を明示的に引きに行っていること。"""
        body = self._restore_step().get('run', '')
        assert 'status=success' in body
        assert 'archive_download_url' in body

    def test_restore_is_verified_out_loud(self):
        """復元できたかをログに出すこと（黙って0から始まるのを繰り返さない）。"""
        body = self._restore_step().get('run', '')
        assert 'fetch_log' in body, '復元後に取得済みレース数を表示していない'

    def test_reading_other_runs_artifacts_is_permitted(self):
        """actions: read が無いと前回runの artifact を引けない。書き込みは無いこと。"""
        perms = self._wf()['permissions']
        assert perms.get('actions') == 'read'
        assert 'write' not in str(perms.values()), '公開リポジトリへの書き込み権限を持たせない'
