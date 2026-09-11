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
