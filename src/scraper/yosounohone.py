"""yosounohone.com（競馬予想のホネ）の「追い切り注目馬」記事をパースする。

事前登録カード: `oikiri/CRITERIA.md`。**測るのは2023-2025**（2026 は 2026-10-21 の
logic2026 で読んでしまった）。

＝＝ なぜこの情報源か ＝＝

調教は 2026-08-16 に「丸ごと欠けている唯一の次元」と確定した領域で、
JRA公式には機械可読なタイムが存在しないことを4回検証して打ち切っている。
netkeiba の `oikiri.html` は無料部分が凡例だけ（2026-08-24）。
この情報源は **robots 上取得してよく、sitemap が 2019-02 まで遡れる**唯一の候補。

＝＝ 記事の構造（2023-10-28 と 2026-10-18 の実物で確認。同一） ＝＝

    <h3 id="...">調教評価　2023/10/28(土) 東京競馬</h3>
    <h4 id="...">【東京1R】2歳未勝利</h4>
    <h5 id="...">レッドレナート</h5>
    <p>最終追い切り美浦坂路★★★</p>
    <p>《短評》…</p>          ← 任意

1記事 ＝ **1会場 × 1開催日**。1レースに1〜2頭の「注目馬」だけに★1〜5が付く。
同じレースに印なしの馬が居るので**レース内の対照群が組める**
（jugem は 1頭/開催日 でレース内順位が常に1・z が 0 になり正規化が成立しなかった）。

🔴 **`id` 属性を使わないこと。** 実物に貼り付け残りがある:

    <h4 id="東京7R">  …中身は 【京都1R】
    <h5 id="ディアスポラ">  …中身は ピコジャック

`id` を信じると会場・馬名が静かに別物になる。**表示テキストだけを読む。**

🔑 **URL から読んだ (会場, 日付) と、本文の h3／title から読んだ (会場, 日付) を
突合して、一致しない記事は採用しない。** North Star #8 の「同じものを別経路で
作って値を突合する」がタダで手に入る（kayochin でも一覧と各レース再掲の突合を
同じ目的で入れた）。

🔴 **公開時刻は OG の `article:published_time` を正とする。**
kayochin は JSON-LD の `datePublished` が同じ時刻に違うオフセットを付けており
9時間ずれていた。ここは2026-10-22 の実測で**両者が同一瞬間を指す**ことを
確認しているが、ずれたら中止条件4で止める（カード §7）。
"""

import html as _html
import re

# 🔴 2026-10-22 に測って**不採用**（`oikiri/RESULTS.md`）。読む前に知っておくこと:
#   ①信号ライン4基準は**すべて通った**（このプロジェクトで初）。★が付いた馬の
#     3着内率は 33.61% vs 同レースの印なし 19.74% ＝ **+13.87pt（1.70倍）**で、
#     市場人気3分位の各分位でも +2.6〜6.6pt 残り、2期間（2023 / 2024-25）とも安定
#   ②🔴 **それでも複勝回収率は 85.02%**（3本抜き 83.42%）。複勝プール平均は
#     控除率20%で 80% なので +5.0pt しかなく、**100% には 15pt 足りない**
#   ③🔴 **市場に足すと複勝AUC が 0.8060 → 0.8059 と動かない**（★単独 0.5406）
#   ④機序: ★が付く 6,575頭のうち **3,744頭（57%）が市場人気の上位3分の1**に居る。
#     当たりやすい分だけ配当が縮む（North Star #9 のこれまでで最も純粋な実例）
#   🔴 **この情報源を定義・閾値・期間・結果変数を変えて測り直さない。**
#   ⚠ 「調教は効かない」ではない。測ったのは「筆者が選んだ注目馬1〜2頭への★」で、
#     **全馬の調教タイムは依然として一度も測っていない**
SOURCE_NAME = 'yosounohone'
BASE_URL = 'https://www.yosounohone.com'
SITEMAP_INDEX = f'{BASE_URL}/sitemap_index.xml'

# 会場のローマ字 → 日本語。URL の `/entry/{romaji}-YYYY-M-D` に使う
VENUE_ROMAJI = {
    'sapporo': '札幌', 'hakodate': '函館', 'fukushima': '福島',
    'niigata': '新潟', 'tokyo': '東京', 'nakayama': '中山',
    'chukyo': '中京', 'kyoto': '京都', 'hanshin': '阪神', 'kokura': '小倉',
}
# JRA の場コード（history.db の race_id は YYYYMMDD_場コード_R）
VENUE_CODE = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04', '東京': '05',
    '中山': '06', '中京': '07', '京都': '08', '阪神': '09', '小倉': '10',
}

_ARTICLE_URL_RE = re.compile(
    r'/entry/(' + '|'.join(VENUE_ROMAJI) + r')-(\d{4})-(\d{1,2})-(\d{1,2})/?$')

_PUBLISHED_OG_RE = re.compile(
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']')
_PUBLISHED_OG_RE2 = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']')
_PUBLISHED_LD_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')

# 本文の h3「調教評価　2023/10/28(土) 東京競馬」/ title「…　2023/10/28(土) 東京競馬　…」
_HEAD_DATE_VENUE_RE = re.compile(
    r'(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\s*[（(][^）)]*[）)]\s*'
    r'(' + '|'.join(VENUE_CODE) + r')\s*競馬')

# 【東京11R】アルテミスS G3
_RACE_HEAD_RE = re.compile(
    r'^[【\[]\s*(' + '|'.join(VENUE_CODE) + r')\s*(\d{1,2})\s*[Rr]\s*[】\]]\s*(.*)$')

_TAG_RE = re.compile(r'<[^>]+>')
_STAR_RE = re.compile(r'[★☆]+')


def article_url(venue_romaji, date):
    """date は 'YYYY-MM-DD'。⚠ 列挙は sitemap から行い、この関数で日付を推測しない。"""
    y, m, d = date.split('-')
    return f'{BASE_URL}/entry/{venue_romaji}-{int(y)}-{int(m)}-{int(d)}'


def parse_url(url):
    """記事URLから (会場日本語, 'YYYY-MM-DD') を返す。該当しなければ None。"""
    m = _ARTICLE_URL_RE.search(url)
    if not m:
        return None
    romaji, y, mo, d = m.groups()
    return VENUE_ROMAJI[romaji], f'{int(y):04d}-{int(mo):02d}-{int(d):02d}'


def list_month_articles(sitemap_xml):
    """月別 sitemap の XML から「会場×開催日」記事だけを (会場, 日付, URL) で返す。

    ⚠ 重賞単発の予想記事（`/entry/artemis-s-2023`）と回顧（`/entry/kaiko-*`）は
      対象外。重賞も会場×開催日の記事の中に 11R として入っているため取りこぼさない。
    """
    out = []
    seen = set()
    for loc in re.findall(r'<loc>([^<]+)</loc>', sitemap_xml):
        loc = _html.unescape(loc).strip()
        parsed = parse_url(loc)
        if parsed and loc not in seen:
            seen.add(loc)
            out.append((parsed[0], parsed[1], loc))
    return out


def _iso_instant(s):
    """ISO8601 を UTC の aware datetime にする。読めなければ None。"""
    from datetime import datetime, timezone
    if not s:
        return None
    t = s.strip().replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def parse_published_at(page_html):
    """(og_instant, ld_instant) を UTC aware datetime で返す。

    🔴 正は OG。LD は**中止条件4（両者が一致するか）のためだけ**に返す。
    """
    og = (_PUBLISHED_OG_RE.search(page_html) or
          _PUBLISHED_OG_RE2.search(page_html))
    ld = _PUBLISHED_LD_RE.search(page_html)
    return (_iso_instant(og.group(1) if og else None),
            _iso_instant(ld.group(1) if ld else None))


def _text(frag):
    return _html.unescape(_TAG_RE.sub('', frag)).replace(' ', ' ').strip()


def _entry_body(page_html):
    m = re.search(r'<div class="entry-content[^"]*"[^>]*>', page_html)
    return page_html[m.end():] if m else page_html


def parse_header_date_venue(page_html):
    """本文（h3）→ title の順に「YYYY/MM/DD(曜) {会場}競馬」を探す。

    URL と突合するための**別経路**。両方見つからなければ None。
    """
    body = _entry_body(page_html)
    for frag in re.findall(r'<h3[^>]*>(.*?)</h3>', body, re.S):
        m = _HEAD_DATE_VENUE_RE.search(_text(frag))
        if m:
            y, mo, d, v = m.groups()
            return v, f'{int(y):04d}-{int(mo):02d}-{int(d):02d}'
    t = re.search(r'<title>(.*?)</title>', page_html, re.S)
    if t:
        m = _HEAD_DATE_VENUE_RE.search(_text(t.group(1)))
        if m:
            y, mo, d, v = m.groups()
            return v, f'{int(y):04d}-{int(mo):02d}-{int(d):02d}'
    return None


def parse_article(page_html, venue, date):
    """1記事をパースして {'races': [...], 'dropped': [...]} を返す。

    `venue` / `date` は URL 由来の値。本文の h4 が別会場を名乗っていたら
    そのレースを落として理由を残す（`id` の貼り付け残りと同じ事故を本文側でも拾う）。
    """
    body = _entry_body(page_html)
    dropped = []
    races = []
    cur = None
    horse = None

    # h4 / h5 / p を**出現順**に走査する。構造を位置で決め打ちしない
    for m in re.finditer(r'<(h4|h5|p)\b[^>]*>(.*?)</\1>', body, re.S):
        tag, inner = m.group(1), m.group(2)
        txt = _text(inner)
        if tag == 'h4':
            horse = None
            cur = None
            hm = _RACE_HEAD_RE.match(txt)
            if not hm:
                continue
            v, rno, rname = hm.group(1), int(hm.group(2)), hm.group(3).strip()
            if v != venue:
                dropped.append(f'h4「{txt[:30]}」の会場 {v} が記事の {venue} と違う')
                continue
            if not (1 <= rno <= 12):
                dropped.append(f'h4「{txt[:30]}」のレース番号 {rno} が 1〜12 の外')
                continue
            cur = {'race_num': rno, 'race_name': rname, 'horses': []}
            races.append(cur)
        elif tag == 'h5':
            horse = None
            if cur is None or not txt:
                continue
            horse = {'name': txt, 'stars': None, 'training': None, 'comment': None}
            cur['horses'].append(horse)
        else:  # p
            if horse is None or not txt:
                continue
            if txt.startswith('《短評》') or txt.startswith('《短評'):
                if horse['comment'] is None:
                    horse['comment'] = txt
                continue
            if horse['stars'] is not None:
                continue
            sm = _STAR_RE.search(txt)
            if not sm:
                continue
            horse['stars'] = len(sm.group(0))
            horse['training'] = txt[:sm.start()].strip() or None

    # ★が取れなかった馬は採用しない（「印は付いたが評価が読めない」を0扱いにしない）
    for r in races:
        keep = []
        for h in r['horses']:
            if h['stars'] is None:
                dropped.append(f'{r["race_num"]}R {h["name"]} の★が読めない')
            else:
                keep.append(h)
        r['horses'] = keep
    races = [r for r in races if r['horses']]

    return {'venue': venue, 'date': date, 'races': races, 'dropped': dropped}
