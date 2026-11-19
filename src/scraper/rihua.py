"""rihua-keiba.com の「追い切り診断」記事をパースする（2026-11-20）。

ユーザー指示「web 情報は広く集めて、検証して削っていく」の一つ目（`websig/SURVEY_2026-11-19.md`）。
robots は /wp-admin/ のみ Disallow。sitemap に 2022〜2026 の記事が残っている（重賞のみ）。

＝＝ 記事の構造（2026-11-20 の下見で確認） ＝＝

    <title>【函館スプリントステークス２０２４】追い切り診断 …</title>
    <h3>アサカラキング（斎藤新）　Ｂ</h3>     ← 最終の追い切り診断は 馬名（騎手）＋評価 A/B/C
    <h3>シュバルツカイザー（大野拓弥）</h3>    ← 評価が無い馬もいる
    <h2>【追い切りからのオススメ４頭】</h2>
    <span class="sc_marker blue">※ハーツコンチェルト</span>

- 「１週前追い切り診断」は評価記号が付かない（馬名だけ）。
- 1記事に2レース入ることがある（h2 の【…】で区切る）。
🔴 本文（短評）は取らない・保存しない。馬名・騎手・評価記号・オススメの有無だけ。
"""
import re
import unicodedata

SOURCE_NAME = 'rihua-keiba'
BASE_URL = 'https://rihua-keiba.com'
SITEMAP = f'{BASE_URL}/sitemap.xml'
GRADE_VALUE = {'S': 4, 'A': 3, 'B': 2, 'C': 1, 'D': 0, 'E': -1}
_H3 = re.compile(r'^([^\s(（【】※]{2,12})\s*(?:[(（]([^)）]{1,12})[)）])?\s*([SABCDE])?\s*$')


def nfkc(s):
    return unicodedata.normalize('NFKC', s or '').strip()


def race_label(s):
    """【第１２回アルテミスステークス２０２３】→ 'アルテミスステークス'。"""
    m = re.search(r'【([^】]+)】', nfkc(s))
    t = m.group(1) if m else nfkc(s)
    t = re.sub(r'第\d+回', '', t)
    t = re.sub(r'\d{4}', '', t)
    t = re.sub(r'(1週前|最終)?追い切り.*$', '', t)
    return t.strip(' 、,')


def is_workout_url(url):
    from urllib.parse import unquote
    return '追い切り' in unquote(url)


def parse_article(html):
    """記事を (レースごとの塊) のリストにする。

    返り値: {'title', 'published', 'one_week', 'blocks': [{'race_label', 'horses': [{'name','jockey','grade','grade_value','recommended'}]}]}
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    title = nfkc((soup.title.get_text() if soup.title else '') or '')
    pub = None
    m = soup.find('meta', attrs={'property': 'article:published_time'})
    if m and m.get('content'):
        pub = m['content'][:10]
    body = soup.find(class_='entry-content') or soup.find('article') or soup
    one_week = '1週前' in title
    blocks = []
    cur = {'race_label': race_label(title), 'horses': []}
    rec_mode = False
    for el in body.find_all(['h2', 'h3', 'span']):
        t = nfkc(el.get_text(' ', strip=True))
        if el.name == 'h2':
            if 'オススメ' in t or 'おすすめ' in t:
                rec_mode = True
                continue
            rec_mode = False
            if t.startswith('【') and ('追い切り' in t or '診断' in t) and cur['horses']:
                blocks.append(cur)
                cur = {'race_label': race_label(t), 'horses': []}
            elif t.startswith('【') and ('追い切り' in t or '診断' in t):
                cur['race_label'] = race_label(t) or cur['race_label']
            continue
        if el.name == 'h3':
            mm = _H3.match(t)
            if mm:
                g = mm.group(3)
                cur['horses'].append({'name': mm.group(1), 'jockey': mm.group(2), 'grade': g,
                                      'grade_value': GRADE_VALUE.get(g), 'recommended': False})
            continue
        if el.name == 'span' and rec_mode and 'sc_marker' in (el.get('class') or []) and t.startswith('※'):
            name = t.lstrip('※').strip()
            for b in blocks + [cur]:
                for h in b['horses']:
                    if h['name'] == name:
                        h['recommended'] = True
    if cur['horses']:
        blocks.append(cur)
    return {'title': title, 'published': pub, 'one_week': one_week, 'blocks': blocks}
