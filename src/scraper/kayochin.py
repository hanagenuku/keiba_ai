"""kayochinkeiba.com の「能力指数」記事を、馬ごとの行に変換する。

なぜこれを作るか
----------------
direction ①（予測時点でまだ市場に入っていない情報）の候補として、
2026-09-23 に許可ドメイン11件を実地確認した結果、使えるのは3件だけだった。
そのうち **全レース・全馬に連続量が付いている唯一の情報源**がここ。
印（◎○▲）だけの情報源はレース内順位が作れないので、測定に耐えない。

🔑 過去記事が残っている（2026-09-24 に sitemap で確認）
   月別アーカイブ 2016-12〜2026-09（119ヶ月）。robots は /wp-admin/ のみ Disallow。
   history.db と重なる 2023-01〜2026-09 で **JRA記事 811本 / 開催日 399日**。
   2023-09 の記事が 2026-09 と**同一構造**であることを実HTMLで確認済み。
   → 「溜まるまで18ヶ月待つ」必要がない。**過去分で先に測れる。**

⚠ これは予想を後から書き換える層ではない。このプロジェクトで後付け補正層は
   例外なく撤回されている（市場補正レイヤー / error_tags週次補正 /
   rank_matrix_filter / AI xx%バッジ / ROI予測150%）。過去分で学習データを
   作れるので、**効くと分かってから学習済みの特徴量として入れる**。
   繋ぐ前に事前登録した基準を通すこと（パドック入力と同じ順序）。

⚠ 取得データの置き場は `data/private/`（.gitignore 済み）。
   netkeiba と同じ扱いにする。公開リポジトリに他所のデータを置かない。

🔴 公開時刻は OG の `article:published_time` だけを見る（2026-09-24 に実測）
   RSS pubDate                10:17:52 +0000   WordPress コアが出す
   OG article:published_time  10:17:52+00:00   ✅ RSS と完全一致（UTC）
   JSON-LD datePublished      10:17:52+09:00   🔴 同じ時刻に違うオフセット＝バグ
   JSON-LD を信じると **9時間ずれ**、「予測時点で利用可能だったか」の答えが
   逆になる。実測の公開は前日 19:17 JST で、本番の予想生成（19:06）の11分後。

🔑 検算がタダで手に入る（North Star #8「同じ値を別経路で作って突合する」）
   記事には同じ指数が2箇所に載っている。
     ① 冒頭の一覧テーブル（12レースぶん・25行）
     ② 各レースセクションの再掲（1レースぶん・3行）
   両者が一致しないレースは**採用しない**。パースのズレを黙って通さないため。
"""
import re
import unicodedata

from src.utils.config import PLACE_ENG

SOURCE_NAME = 'kayochinkeiba'
BASE_URL = 'https://kayochinkeiba.com'

# 丸数字 ①..⑳ → 1..20（馬番は最大18）
_CIRCLED = {chr(0x2460 + i): i + 1 for i in range(20)}

_PUBLISHED_RE = re.compile(
    r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']')
_RACE_LABEL_RE = re.compile(r'^(\d{1,2})R$')


def article_url(date, venue_romaji):
    """`2026-09-20`, `nakayama` → 記事URL。日付は YYMMDD に畳まれる。"""
    d = date.replace('-', '')
    if len(d) != 8:
        raise ValueError(f'date は YYYY-MM-DD か YYYYMMDD: {date!r}')
    return f'{BASE_URL}/{d[2:]}{venue_romaji}/'


def parse_published_at(html):
    """OG の `article:published_time` を返す（UTC の ISO 文字列）。

    ⚠ JSON-LD の `datePublished` は使わない（上記のとおりオフセットが誤り）。
    見つからなければ None。作り話の時刻を返さない。
    """
    m = _PUBLISHED_RE.search(html)
    return m.group(1) if m else None


def _circled_to_int(text):
    """'⑧' → 8。丸数字でなければ None（半角数字も受ける）。"""
    t = (text or '').strip()
    if len(t) == 1 and t in _CIRCLED:
        return _CIRCLED[t]
    t2 = unicodedata.normalize('NFKC', t)
    if t2.isdigit():
        n = int(t2)
        return n if 1 <= n <= 30 else None
    return None


def _table_rows(table):
    return [[c.get_text(strip=True) for c in tr.find_all(['td', 'th'])]
            for tr in table.find_all('tr')]


def _is_index_table(rows):
    """指数テーブルは 1行目が ['', '1', '2', '3', ...] で始まる。"""
    return bool(rows) and rows[0][:4] == ['', '1', '2', '3']


def _pairs_from_block(num_row, idx_row):
    """['1R','⑧','④',...] と ['指','91','70',...] → [(馬番, 指数), ...]

    空欄（出走頭数に満たないぶん）は落とす。壊れていれば None を返す。
    """
    if not num_row or not idx_row or idx_row[0] != '指':
        return None
    out = []
    for raw_n, raw_v in zip(num_row[1:], idx_row[1:]):
        if not raw_n and not raw_v:
            continue          # 頭数ぶんで打ち切られた末尾の空セル
        n = _circled_to_int(raw_n)
        v = unicodedata.normalize('NFKC', (raw_v or '').strip())
        if n is None or not v.isdigit():
            return None       # 片方だけ欠けている＝ズレている。黙って通さない
        out.append((n, int(v)))
    if not out:
        return None
    if len({n for n, _ in out}) != len(out):
        return None           # 馬番の重複＝列ズレ
    return out


def _blocks(rows):
    """['1R', ...] / ['指', ...] の2行1組を {レース番号: [(馬番, 指数)]} に。"""
    out = {}
    for i in range(1, len(rows) - 1):
        m = _RACE_LABEL_RE.match((rows[i] or [''])[0])
        if not m:
            continue
        pairs = _pairs_from_block(rows[i], rows[i + 1])
        if pairs is not None:
            out[int(m.group(1))] = pairs
    return out


def parse_index_tables(html):
    """記事HTMLから指数を取り出す。一覧と各レース再掲を突合し、一致分だけ返す。

    Returns
    -------
    (races, rejected)
      races    {race_num: [(horse_num, index_value), ...]}  指数の降順
      rejected [(race_num, 理由), ...]  一致しなかったレース
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    summary, repeats = {}, {}
    for table in soup.find_all('table'):
        rows = _table_rows(table)
        if not _is_index_table(rows):
            continue
        found = _blocks(rows)
        if len(found) > 1:
            summary.update(found)      # 12レースぶんの一覧
        else:
            repeats.update(found)      # 各レースの再掲

    races, rejected = {}, []
    for rn in sorted(set(summary) | set(repeats)):
        a, b = summary.get(rn), repeats.get(rn)
        if a is None or b is None:
            rejected.append((rn, '一覧と再掲のどちらかが無い'))
            continue
        if a != b:
            rejected.append((rn, '一覧と再掲が不一致'))
            continue
        races[rn] = a
    return races, rejected


def parse_article(html, date, venue_romaji):
    """1記事 → 馬ごとの行。history.db の race_id で結合できる形にする。

    Parameters
    ----------
    date : 'YYYY-MM-DD'（そのレース開催日）
    venue_romaji : 'nakayama' 等。`PLACE_ENG` のキー

    Returns
    -------
    (rows, meta)
      rows : [{'race_id','date','venue','race_num','horse_num',
               'index_value','index_rank','n_horses','source','published_at'}]
      meta : {'published_at','n_races','rejected'}
    """
    place = PLACE_ENG.get(venue_romaji)
    if place is None:
        raise ValueError(f'JRA の会場ではない: {venue_romaji!r}')

    published_at = parse_published_at(html)
    races, rejected = parse_index_tables(html)
    ymd = date.replace('-', '')

    rows = []
    for race_num, pairs in sorted(races.items()):
        race_id = f'{ymd}_{place}_{race_num:02d}'
        for rank, (horse_num, value) in enumerate(pairs, 1):
            rows.append({
                'race_id': race_id,
                'date': date,
                'venue': venue_romaji,
                'race_num': race_num,
                'horse_num': horse_num,
                'index_value': value,
                'index_rank': rank,
                'n_horses': len(pairs),
                'source': SOURCE_NAME,
                'published_at': published_at,
            })
    return rows, {'published_at': published_at,
                  'n_races': len(races),
                  'rejected': rejected}
