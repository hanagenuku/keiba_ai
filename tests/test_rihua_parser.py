"""rihua-keiba の追い切り診断のパーサ。実機で確かめた構造を写した HTML で確かめる（North Star #6）。"""
from src.scraper.rihua import parse_article, race_label

HTML = """<html><head><title>【函館スプリントステークス２０２４】追い切り診断 | りふぁ競馬</title>
<meta property="article:published_time" content="2024-06-06T20:00:00+09:00"></head><body>
<div class="entry-content">
<h3>アサカラキング（斎藤新）　Ｂ</h3><p>短評</p>
<h3>サトノレーヴ（浜中俊）　Ａ</h3><p>短評</p>
<h3>シュバルツカイザー（大野拓弥）</h3><p>短評</p>
<h3>オタルエバー（角田大河）　Ｃ</h3>
<h2>【追い切りからのオススメ２頭】</h2>
<span class="sc_marker blue">※サトノレーヴ</span><span class="sc_marker blue">※オタルエバー</span>
<span class="sc_marker red">【最近の重賞レースの3連複🎯】</span>
</div></body></html>"""

TWO = """<html><head><title>【セントライト記念、ローズｓ２０２４】１週前追い切り診断</title></head><body>
<div class="entry-content"><h2>【セントライト記念２０２４】１週前追い切り診断</h2>
<h3>アーバンシック（Ｃ.ルメール）</h3><h3>エコロヴァルツ（岩田康誠）</h3>
<h2>【ローズステークス２０２４】１週前追い切り診断</h2><h3>クイーンズウォーク（川田将雅）</h3>
</div></body></html>"""


def test_grades_jockeys_and_recommendations():
    a = parse_article(HTML)
    assert a['published'] == '2024-06-06' and a['one_week'] is False
    [b] = a['blocks']
    assert b['race_label'] == '函館スプリントステークス'
    hs = {h['name']: h for h in b['horses']}
    assert hs['サトノレーヴ']['grade'] == 'A' and hs['サトノレーヴ']['grade_value'] == 3
    assert hs['アサカラキング']['jockey'] == '斎藤新'
    assert hs['シュバルツカイザー']['grade'] is None
    assert hs['サトノレーヴ']['recommended'] and hs['オタルエバー']['recommended']
    assert not hs['アサカラキング']['recommended']


def test_two_races_in_one_article_and_one_week_flag():
    a = parse_article(TWO)
    assert a['one_week'] is True
    assert [b['race_label'] for b in a['blocks']] == ['セントライト記念', 'ローズステークス']
    assert [h['name'] for h in a['blocks'][1]['horses']] == ['クイーンズウォーク']


def test_race_label_strips_year_and_round():
    assert race_label('【第１２回アルテミスステークス２０２３】追い切り診断') == 'アルテミスステークス'
