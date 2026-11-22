"""新馬戦のレース名が落ちて「R05」になっていた不具合の回帰テスト（2026-09-11）。

本番 latest.json で 札幌R05（race_class='新馬'）が「R05」と表示され、
history.db 側は race_name が空文字になっていた。原因は parse_rname の
総称パターンに `新馬` が無かったこと。**同じ正規表現が jra_scraper にも
複製されていて両方同時に壊れていた**ので、複製をやめて共有した。

⚠ race_class は `_extract_class` が別経路で '新馬' を正しく取れていたため、
   モデルへの入力は壊れていない（表示名だけの不具合）。
"""
import re
import pytest
from src.scraper.parser import parse_rname


class TestShinbaRaceName:
    @pytest.mark.parametrize('header', [
        '2歳新馬 [指定] 定量 コース： 1,800 メートル 芝 右回り',
        '2026年8月30日 札幌 1800メートル（芝・右）2歳新馬 天候:晴 馬場:良',
    ])
    def test_shinba_name_is_not_rnn(self, header):
        """🔴 修正前はここが 'R05' になっていた（本番で実際に発生）。"""
        assert parse_rname(header, 5) == '2歳新馬'

    def test_bare_shinba_also_caught(self):
        """年齢が頭に付かない表記でも新馬は拾う。"""
        assert parse_rname('新馬 コース： 1,200 メートル ダート', 6) == '新馬'

    def test_shinba_does_not_shadow_a_real_race_name(self):
        """特別名がある場合はそちらが勝つ（新馬の追加で既存の名前を潰さない）。"""
        assert parse_rname('北辰特別 3歳以上1勝クラス コース： 1,700 メートル', 12) == '北辰特別'

    @pytest.mark.parametrize('header,want', [
        ('3歳以上1勝クラス [指定] 定量 コース： 1,000 メートル ダート 右回り', '3歳以上1勝クラス'),
        ('2歳未勝利 コース： 1,800 メートル 芝', '2歳未勝利'),
    ])
    def test_existing_classes_unchanged(self, header, want):
        assert parse_rname(header, 3) == want

    def test_fallback_shapes(self):
        """rn を渡せば 'R09'、省略すれば ''（結果ページ側の従来挙動）。"""
        assert parse_rname('何も無いヘッダ', 9) == 'R09'
        assert parse_rname('何も無いヘッダ') == ''


class TestNoDuplicatedRegex:
    """🔴 同じ正規表現を2箇所に置かないこと（片方だけ直る事故の再発防止）。"""

    def test_result_header_reuses_parse_rname(self):
        src = open('src/scraper/jra_scraper.py', encoding='utf-8').read()
        assert "info['race_name'] = parse_rname(header)" in src, \
            'jra_scraper が parse_rname を共有していない'

    def test_generic_class_regex_defined_once(self):
        """総称クラスの正規表現が src/ 全体で1箇所しか無いこと。"""
        import glob
        pat = re.compile(r'未勝利\|1勝クラス\|2勝クラス\|3勝クラス')
        hits = [f for f in glob.glob('src/**/*.py', recursive=True)
                if pat.search(open(f, encoding='utf-8').read())]
        assert hits == ['src/scraper/parser.py'], f'複製が残っている: {hits}'


# ── 2026-11-22: マイルチャンピオンシップが「R11」になっていた ──────────────
#
# 実測: 2026-09 以降の11Rで race_name が空なのは 20261122_08_11（マイルCS）だけ。
# エリザベス女王杯・天皇賞は「杯」「賞」で拾えていた。末尾の語の一覧に
# 「チャンピオンシップ」が無く、ヘッダの条件表記（サラ系3歳以上 オープン）は
# 間が空いていて総称にも一致せず、rn のフォールバックまで落ちていた。

class TestRaceNamesWithoutTheUsualSuffix:
    def test_mile_championship(self):
        h = ('マイルチャンピオンシップ（ＧⅠ） サラ系3歳以上 オープン（国際）(指定) '
             'コース： 1,600 メートル 芝')
        assert parse_rname(h, 11) == 'マイルチャンピオンシップ'

    def test_other_names_with_the_same_shape(self):
        """同じ形で落ちる名前をまとめて拾う（オークス・ダービー・マイル等）。"""
        for header, want in [
            ('優駿牝馬（オークス）（ＧⅠ） サラ系3歳 オープン コース： 2,400 メートル 芝',
             '優駿牝馬'),
            ('ヴィクトリアマイル（ＧⅠ） サラ系4歳以上 オープン コース： 1,600 メートル 芝',
             'ヴィクトリアマイル'),
            ('阪神ジュベナイルフィリーズ（ＧⅠ） サラ系2歳 オープン コース： 1,600 メートル 芝',
             '阪神ジュベナイルフィリーズ'),
            ('中山グランドジャンプ（Ｊ・ＧⅠ） サラ系4歳以上 オープン コース： 4,250 メートル',
             '中山グランドジャンプ'),
        ]:
            assert parse_rname(header, 11) == want

    def test_names_that_already_worked_do_not_change(self):
        """🔴 これまで拾えていた名前を壊さない。"""
        for header, want in [
            ('エリザベス女王杯（ＧⅠ） サラ系3歳以上 オープン コース： 2,200 メートル 芝',
             'エリザベス女王杯'),
            ('天皇賞（秋）（ＧⅠ） サラ系3歳以上 オープン コース： 2,000 メートル 芝',
             '天皇賞'),
            ('武蔵野ステークス（ＧⅢ） サラ系3歳以上 オープン コース： 1,600 メートル ダート',
             '武蔵野ステークス'),
            ('北辰特別 3歳以上1勝クラス コース： 1,700 メートル', '北辰特別'),
        ]:
            assert parse_rname(header, 11) == want

    def test_class_header_with_a_space_is_not_an_r_number(self):
        """特別名が無いレースでも「R11」に落とさない（条件表記の空白を許す）。"""
        h = 'サラ系3歳以上 1勝クラス コース： 1,800 メートル ダート'
        assert parse_rname(h, 7) == '3歳以上1勝クラス'

    def test_still_falls_back_when_there_is_nothing_to_read(self):
        """本当に何も読めないときだけ R番号にする（推測で名前を作らない）。"""
        assert parse_rname('コース： 1,600 メートル 芝', 11) == 'R11'
        assert parse_rname('コース： 1,600 メートル 芝') == ''
