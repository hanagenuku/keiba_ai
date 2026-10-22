"""yosounohone パーサのテスト。

North Star #6: **本番と同じ形**で検証する。フィクスチャは 2023-10-28 東京 /
2026-10-18 京都 の実物から構造をそのまま写したもの（`id` の貼り付け残り・
`<span style>` の入れ子・`&nbsp;` の空段落・《短評》を含む）。

綺麗な自作HTMLだと `id` を読む実装でもテストが通ってしまう。
実物には `<h4 id="東京7R">…【京都1R】`・`<h5 id="ディアスポラ">…ピコジャック` が
あり、**`id` を信じると会場も馬名も静かに別物になる**。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper.yosounohone import (VENUE_CODE, list_month_articles,
                                     parse_article, parse_header_date_venue,
                                     parse_published_at, parse_url)

# 実物から写した構造。id は意図的に壊したまま（実物がそうなっている）
REAL_SHAPED = '''<html><head>
<title>【追い切り注目馬】【秋華賞】他　2026/10/18(日) 京都競馬　調教でも抜けちゃった - 競馬予想のホネ</title>
<meta property="article:published_time" content="2026-10-17T23:30:00Z" />
<script type="application/ld+json">{"datePublished":"2026-10-18T08:30:00+09:00"}</script>
</head><body>
<div class="entry-content hatenablog-entry">
<p><span style="color: #000000;">――まくら文</span></p>
<h3 id="調教評価20261018日-京都競馬"><span style="font-weight: bold;">調教評価　2026/10/18(日) 京都競馬</span></h3>
<blockquote><p><span>＜見方＞</span></p>
<p><span>・主に「最終迫い切り時計」評価を基準に、★の数で評価（5つ星まで）</span></p></blockquote>
<p>&nbsp;</p>
<h4 id="東京7R"><span style="color: #000000;">【京都1R】２歳未勝利</span></h4>
<h5 id="ディアスポラ"><span style="color: #000000;">ピコジャック</span></h5>
<p><span style="color: #000000;">最終迫い切り栗東坂路★★</span></p>
<p>&nbsp;</p>
<h4 id="京都4R">【京都4R】３歳以上１勝クラス</h4>
<h5 id="a"><span>ラブインアクション</span></h5>
<p><span>10/7栗東坂路★★★</span></p>
<p>&nbsp;</p>
<h5 id="b"><span>キンダープンシュ</span></h5>
<p><span>最終迫い切り栗東坂路★★</span></p>
<p>&nbsp;</p>
<h4 id="c">【京都11R】秋華賞 G1</h4>
<h5 id="d"><span>モンローウォーク</span></h5>
<p><span><strong>10/7栗東CW＆10/15栗東坂路★★</strong></span></p>
<p><span>《短評》時計的には何ら文句なし。</span></p>
<p>&nbsp;</p>
<h5 id="e"><span>スターが読めない馬</span></h5>
<p><span>最終迫い切り栗東坂路（評価対象外）</span></p>
<h4 id="f">【東京10R】別会場の見出し</h4>
<h5 id="g"><span>ヘンな馬</span></h5>
<p><span>最終迫い切り★★★★★</span></p>
</div></body></html>'''


def _parsed():
    return parse_article(REAL_SHAPED, '京都', '2026-10-18')


def test_url_and_body_agree():
    assert parse_url('https://www.yosounohone.com/entry/kyoto-2026-10-18') == (
        '京都', '2026-10-18')
    assert parse_header_date_venue(REAL_SHAPED) == ('京都', '2026-10-18')


def test_url_pattern_rejects_non_venue_articles():
    for u in ('https://www.yosounohone.com/entry/artemis-s-2023',
              'https://www.yosounohone.com/entry/kaiko-2024-7-13',
              'https://www.yosounohone.com/entry/2023/10/31/211813'):
        assert parse_url(u) is None


def test_sitemap_lists_only_venue_day_articles():
    xml = ('<urlset>'
           '<url><loc>https://www.yosounohone.com/entry/tokyo-2023-10-28</loc></url>'
           '<url><loc>https://www.yosounohone.com/entry/kyoto-2023-10-28</loc></url>'
           '<url><loc>https://www.yosounohone.com/entry/artemis-s-2023</loc></url>'
           '<url><loc>https://www.yosounohone.com/entry/kaiko-2023-10-8</loc></url>'
           '<url><loc>https://www.yosounohone.com/entry/2023/10/31/211813</loc></url>'
           '</urlset>')
    got = list_month_articles(xml)
    # 並び順は契約ではない（`list_articles` 側で日付順に並べ直す）
    assert {(v, d) for v, d, _ in got} == {('東京', '2023-10-28'),
                                           ('京都', '2023-10-28')}
    assert len(got) == 2


def test_published_at_prefers_og_and_ld_points_to_same_instant():
    """🔴 kayochin は JSON-LD が同じ時刻に違うオフセットを付けて9時間ずれていた。

    ここは両者が同一瞬間であることを**固定する**（カード §7 中止条件4）。
    """
    og, ld = parse_published_at(REAL_SHAPED)
    assert og is not None and ld is not None
    assert og == ld
    assert og.isoformat() == '2026-10-17T23:30:00+00:00'


def test_horse_name_comes_from_text_not_id():
    """`id="ディアスポラ"` に引きずられず、表示されている ピコジャック を取る。"""
    a = _parsed()
    r1 = [r for r in a['races'] if r['race_num'] == 1][0]
    assert [h['name'] for h in r1['horses']] == ['ピコジャック']


def test_race_head_comes_from_text_not_id():
    """`id="東京7R"` に引きずられず、表示されている 【京都1R】 を取る。"""
    a = _parsed()
    assert 1 in [r['race_num'] for r in a['races']]
    assert 7 not in [r['race_num'] for r in a['races']]


def test_other_venue_race_is_dropped_with_reason():
    """本文の h4 が別会場を名乗ったらそのレースを落とし、理由を残す。"""
    a = _parsed()
    assert 10 not in [r['race_num'] for r in a['races']]
    assert any('会場' in d and '東京' in d for d in a['dropped'])


def test_stars_and_training_split():
    a = _parsed()
    r4 = [r for r in a['races'] if r['race_num'] == 4][0]
    by = {h['name']: h for h in r4['horses']}
    assert by['ラブインアクション']['stars'] == 3
    assert by['ラブインアクション']['training'] == '10/7栗東坂路'
    assert by['キンダープンシュ']['stars'] == 2


def test_comment_is_not_mistaken_for_the_stars_line():
    a = _parsed()
    r11 = [r for r in a['races'] if r['race_num'] == 11][0]
    m = [h for h in r11['horses'] if h['name'] == 'モンローウォーク'][0]
    assert m['stars'] == 2
    assert m['training'] == '10/7栗東CW＆10/15栗東坂路'
    assert m['comment'].startswith('《短評》')


def test_horse_without_stars_is_dropped_not_zero():
    """★が読めない馬を 0 扱いにしない（「評価対象外」を最低評価と混同しない）。"""
    a = _parsed()
    r11 = [r for r in a['races'] if r['race_num'] == 11][0]
    assert 'スターが読めない馬' not in [h['name'] for h in r11['horses']]
    assert any('スターが読めない馬' in d for d in a['dropped'])


def test_all_stars_are_within_one_to_five():
    a = _parsed()
    for r in a['races']:
        for h in r['horses']:
            assert 1 <= h['stars'] <= 5


def test_venue_codes_match_history_db_convention():
    """race_id の場コード（YYYYMMDD_場コード_R）と揃っていること。"""
    assert VENUE_CODE['東京'] == '05'
    assert VENUE_CODE['京都'] == '08'
    assert len(VENUE_CODE) == 10
