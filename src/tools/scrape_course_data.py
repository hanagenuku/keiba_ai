"""
JRA全10競馬場のコースデータ収集ツール（Colab専用・一括実行・一回限りのタスク）

【背景】
コース×距離別の特徴（ダートの芝スタート区間、坂の位置、コーナーのタイト度等）を
特徴量化するための土台として、JRA公式サイト（優先）+ umasiru.com（補完）から
コース平面図・立体図・高低断面図・コースデータを収集する。

このスクリプトは週次の本番パイプラインには組み込まない
（KEIBA_過去データ一括取得_v4.ipynb と同じ位置づけの、一回限りのデータ収集ツール）。

【北星ルールとの整合】
- 「決め打ちで実装しない」: 両サイトの実際のHTML構造をこの環境からは確認できないため、
  各データ項目は複数の候補パターンで探索し、見つからなければ例外を出さず空欄のまま
  ログに警告を出して次に進む。数値を推測・捏造することはしない。
- 画像はGitHubリポジトリには含めない（JRA/umasiru.comの著作物のため）。
  Google Drive等、リポジトリ外の output_dir に保存すること。

【出力】
{output_dir}/images/{venue_eng}_course.png            コース平面図
{output_dir}/images/{venue_eng}_3d.png                コース立体図
{output_dir}/images/{venue_eng}_turf_elevation.png    芝高低断面図
{output_dir}/images/{venue_eng}_dirt_elevation.png    ダート高低断面図
{output_dir}/images/{venue_eng}_raw.html              取得した生HTML（パーサー調整用）
{output_dir}/csv/course_basic.csv                     直線距離・高低差・一周距離・幅員・回り
{output_dir}/csv/distance_start.csv                   距離別スタート位置（可能な範囲）
{output_dir}/csv/start_to_corner.csv                  スタート〜1コーナー距離（空欄多数の想定）
{output_dir}/csv/elevation_features.csv               ゴール前坂の位置・高低差（空欄多数の想定）
{output_dir}/csv/corner_features.csv                  コーナー数・内外回り・タイト度分類
{output_dir}/scrape_log.txt                           競馬場ごとの取得結果ログ

⚠ start_to_corner.csv / elevation_features.csv は、テキストとして明記されていない
限り自動抽出できない（コース平面図・断面図を目視で読み取る必要がある）。
このスクリプト実行後、保存された images/ をアップロードしてもらえれば、
その画像を見て数値を書き起こす後続作業を別途行う想定。

【Colabでの使い方】
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.scrape_course_data import scrape_all_courses
    scrape_all_courses('/content/drive/MyDrive/keiba_ai/course_data')
"""
import csv
import os
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

from src.utils.config import HEADERS, JRA_BASE, PLACE_ENG

UMASIRU_CATEGORY_URL = 'https://umasiru.com/archives/category/racecourse'

TURF_DISTANCES = [1000, 1200, 1400, 1500, 1600, 1700, 1800, 2000, 2200, 2300,
                   2400, 2500, 2600, 3000, 3200, 3400, 3600, 4000]
DIRT_DISTANCES = [1000, 1150, 1200, 1300, 1400, 1600, 1700, 1800, 1900, 2100, 2400]

# 平面図/立体図/断面図を画像のalt・title・直前見出しテキストから分類するためのキーワード
IMAGE_KIND_PATTERNS = [
    ('course', re.compile(r'平面図|コース図|コースマップ')),
    ('3d', re.compile(r'立体図|3D|３Ｄ')),
    ('turf_elevation', re.compile(r'芝.*(高低|断面)|高低.*芝')),
    ('dirt_elevation', re.compile(r'ダート.*(高低|断面)|高低.*ダート')),
]

# コース基本データの候補パターン（表記ゆれに対応するため複数）
BASIC_DATA_PATTERNS = {
    'straight_length_m': [
        r'直線[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
        r'最終直線[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
    ],
    'elevation_diff_m': [
        r'高低差[^\d]{0,10}([\d.]+)\s*(?:m|メートル)',
    ],
    'turf_lap_m': [
        r'芝.{0,6}一周[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
        r'芝.{0,6}周長[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
    ],
    'dirt_lap_m': [
        r'ダート.{0,6}一周[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
        r'ダート.{0,6}周長[^\d]{0,10}([\d,]+(?:\.\d+)?)\s*(?:m|メートル)',
    ],
    'width_m': [
        r'幅員[^\d]{0,10}([\d.]+)\s*(?:m|メートル)',
        r'コース幅[^\d]{0,10}([\d.]+)\s*(?:m|メートル)',
    ],
}
TURN_PATTERN = re.compile(r'(右回り|左回り|右回|左回)')
CORNER_COUNT_PATTERN = re.compile(r'([234])\s*コーナー制')
INNER_OUTER_PATTERN = re.compile(r'(内回り|外回り)')


def _log(log_lines, msg):
    print(msg)
    log_lines.append(msg)


def _get(sess, url, log_lines, timeout=15):
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        _log(log_lines, f'  ⚠ 取得失敗 {url}: {e}')
        return None


def _classify_image(img_tag, heading_text_before):
    """img要素のalt/title・直前の見出しテキストから種類を推定する。"""
    hint = ' '.join(filter(None, [
        img_tag.get('alt', ''), img_tag.get('title', ''), heading_text_before or '',
    ]))
    hint = unicodedata.normalize('NFKC', hint)
    for kind, pattern in IMAGE_KIND_PATTERNS:
        if pattern.search(hint):
            return kind
    return None


def _download_image(sess, img_url, dest_path, log_lines):
    try:
        r = sess.get(img_url, timeout=20)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        _log(log_lines, f'  ⚠ 画像ダウンロード失敗 {img_url}: {e}')
        return False


def _extract_basic_data(full_text):
    """正規表現で直線距離等を探す。見つからなければキーごとにNoneのまま。"""
    out = {}
    for key, patterns in BASIC_DATA_PATTERNS.items():
        val = None
        for pat in patterns:
            m = re.search(pat, full_text)
            if m:
                val = m.group(1).replace(',', '')
                break
        out[key] = val
    m = TURN_PATTERN.search(full_text)
    out['turn'] = m.group(1) if m else None
    return out


def _extract_corner_info(full_text):
    m = CORNER_COUNT_PATTERN.search(full_text)
    n_corners = m.group(1) if m else None
    m2 = INNER_OUTER_PATTERN.search(full_text)
    inner_outer = m2.group(1) if m2 else None
    return n_corners, inner_outer


def _scrape_jra_venue(sess, venue_ja, venue_eng, images_dir, log_lines):
    """JRA公式のコースページから画像・基本データを取得する。"""
    url = f'{JRA_BASE}/facilities/race/{venue_eng}/course/'
    _log(log_lines, f'[{venue_ja}] JRA公式取得: {url}')
    r = _get(sess, url, log_lines)
    if r is None:
        return None

    raw_path = os.path.join(images_dir, f'{venue_eng}_raw.html')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(r.text)

    soup = BeautifulSoup(r.text, 'lxml')
    full_text = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))

    found_kinds = set()
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    for img in soup.find_all('img'):
        # 直前の見出し要素のテキストをヒントに使う（構造不明な場合の best-effort）
        heading_text = None
        for h in headings:
            if h.sourceline and img.sourceline and h.sourceline <= img.sourceline:
                heading_text = h.get_text(strip=True)
        kind = _classify_image(img, heading_text)
        if not kind or kind in found_kinds:
            continue
        src = img.get('src', '')
        if not src:
            continue
        img_url = src if src.startswith('http') else f'{JRA_BASE}{src}'
        dest = os.path.join(images_dir, f'{venue_eng}_{kind}.png')
        if _download_image(sess, img_url, dest, log_lines):
            found_kinds.add(kind)
            _log(log_lines, f'  ✅ 画像保存: {os.path.basename(dest)} (種類推定: {kind})')

    for expected in ('course', '3d', 'turf_elevation', 'dirt_elevation'):
        if expected not in found_kinds:
            _log(log_lines, f'  ⚠ 画像未検出: {expected}（手動確認が必要）')

    basic = _extract_basic_data(full_text)
    n_corners, inner_outer = _extract_corner_info(full_text)
    basic['n_corners'] = n_corners
    basic['inner_outer'] = inner_outer
    basic['source'] = url
    basic['full_text_len'] = len(full_text)
    return basic


def _find_umasiru_article_url(sess, venue_ja, log_lines):
    """umasiru.comのカテゴリ一覧から該当競馬場の記事URLをリンクテキストで探す。"""
    r = _get(sess, UMASIRU_CATEGORY_URL, log_lines)
    if r is None:
        return None
    soup = BeautifulSoup(r.text, 'lxml')
    for a in soup.find_all('a', href=True):
        text = unicodedata.normalize('NFKC', a.get_text(strip=True))
        if venue_ja in text and ('コース' in text or '競馬場' in text):
            href = a['href']
            return href if href.startswith('http') else f'https://umasiru.com{href}'
    _log(log_lines, f'  ⚠ umasiru.comで{venue_ja}の記事リンクが見つからない（一覧ページが複数ページに'
                     f'分かれている可能性。手動確認が必要）')
    return None


def _scrape_umasiru_venue(sess, venue_ja, venue_eng, images_dir, log_lines, existing_basic):
    """umasiru.comから、JRA公式で埋まらなかった項目を補完する。"""
    article_url = _find_umasiru_article_url(sess, venue_ja, log_lines)
    if article_url is None:
        return existing_basic
    _log(log_lines, f'[{venue_ja}] umasiru.com補完取得: {article_url}')
    r = _get(sess, article_url, log_lines)
    if r is None:
        return existing_basic

    raw_path = os.path.join(images_dir, f'{venue_eng}_umasiru_raw.html')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(r.text)

    soup = BeautifulSoup(r.text, 'lxml')
    full_text = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    supplement = _extract_basic_data(full_text)
    n_corners, inner_outer = _extract_corner_info(full_text)

    merged = dict(existing_basic) if existing_basic else {}
    for key, val in supplement.items():
        if not merged.get(key) and val:
            merged[key] = val
            merged.setdefault('_umasiru_filled', []).append(key)
    if not merged.get('n_corners') and n_corners:
        merged['n_corners'] = n_corners
    if not merged.get('inner_outer') and inner_outer:
        merged['inner_outer'] = inner_outer
    merged.setdefault('source_umasiru', article_url)
    return merged


def scrape_all_courses(output_dir, venues=None):
    """JRA全10競馬場のコースデータを収集する。

    Args:
        output_dir: 保存先ディレクトリ（Google Drive配下等。GitHubリポジトリ配下は
            画像の著作物配布になるため避けること）
        venues: 対象競馬場の日本語名リスト（Noneなら全10場）
    """
    images_dir = os.path.join(output_dir, 'images')
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    venues = venues or list(PLACE_ENG.keys())
    eng_to_ja = {}
    for eng, code in PLACE_ENG.items():
        from src.utils.config import PLACE_NAMES
        eng_to_ja[eng] = PLACE_NAMES[code]

    sess = requests.Session()
    sess.headers.update(HEADERS)

    log_lines = []
    basic_rows = []
    distance_start_rows = []
    start_to_corner_rows = []
    elevation_rows = []
    corner_rows = []

    for venue_eng in venues:
        venue_ja = eng_to_ja[venue_eng]
        _log(log_lines, f'\n━━━ {venue_ja} ({venue_eng}) ━━━')

        basic = _scrape_jra_venue(sess, venue_ja, venue_eng, images_dir, log_lines)
        basic = _scrape_umasiru_venue(sess, venue_ja, venue_eng, images_dir, log_lines, basic or {})

        if basic is None:
            _log(log_lines, f'  ❌ {venue_ja}: 両サイトとも取得失敗。行はスキップ')
            continue

        basic_rows.append({
            'venue': venue_ja,
            'venue_eng': venue_eng,
            'straight_length_m': basic.get('straight_length_m') or '',
            'elevation_diff_m': basic.get('elevation_diff_m') or '',
            'turf_lap_m': basic.get('turf_lap_m') or '',
            'dirt_lap_m': basic.get('dirt_lap_m') or '',
            'width_m': basic.get('width_m') or '',
            'turn': basic.get('turn') or '',
            'source': basic.get('source') or '',
        })

        corner_rows.append({
            'venue': venue_ja,
            'surface': '',  # 芝/ダートで別れる場合は手動で分割・補記
            'n_corners': basic.get('n_corners') or '',
            'inner_outer': basic.get('inner_outer') or '',
            'corner_shape_class': '',  # Very Tight〜Very Wide は目視分類が必要
            'source': basic.get('source') or basic.get('source_umasiru') or '',
        })

        # ⑥⑦⑧: 自動テキスト抽出では信頼できないため、対象距離ぶんの空行のみ用意する
        # （画像を見た後の手作業 or 別セッションでの目視読み取りで埋める前提）
        for dist in TURF_DISTANCES:
            distance_start_rows.append({
                'venue': venue_ja, 'surface': '芝', 'distance_m': dist,
                'start_position_desc': '', 'start_image_file': '',
                'source': basic.get('source') or '',
            })
            start_to_corner_rows.append({
                'course': venue_ja, 'surface': '芝', 'distance': dist,
                'start_to_corner_m': '', 'estimation_method': '', 'source': '',
            })
        for dist in DIRT_DISTANCES:
            distance_start_rows.append({
                'venue': venue_ja, 'surface': 'ダート', 'distance_m': dist,
                'start_position_desc': '', 'start_image_file': '',
                'source': basic.get('source') or '',
            })
            start_to_corner_rows.append({
                'course': venue_ja, 'surface': 'ダート', 'distance': dist,
                'start_to_corner_m': '', 'estimation_method': '', 'source': '',
            })

        for surface in ('芝', 'ダート'):
            elevation_rows.append({
                'venue': venue_ja, 'surface': surface,
                'hill_start_desc': '', 'hill_end_desc': '',
                'hill_direction': '', 'elevation_diff_m': '',
                'source': '',
            })

        time.sleep(1.0)

    _write_csv(os.path.join(csv_dir, 'course_basic.csv'), basic_rows,
               ['venue', 'venue_eng', 'straight_length_m', 'elevation_diff_m',
                'turf_lap_m', 'dirt_lap_m', 'width_m', 'turn', 'source'])
    _write_csv(os.path.join(csv_dir, 'distance_start.csv'), distance_start_rows,
               ['venue', 'surface', 'distance_m', 'start_position_desc',
                'start_image_file', 'source'])
    _write_csv(os.path.join(csv_dir, 'start_to_corner.csv'), start_to_corner_rows,
               ['course', 'surface', 'distance', 'start_to_corner_m',
                'estimation_method', 'source'])
    _write_csv(os.path.join(csv_dir, 'elevation_features.csv'), elevation_rows,
               ['venue', 'surface', 'hill_start_desc', 'hill_end_desc',
                'hill_direction', 'elevation_diff_m', 'source'])
    _write_csv(os.path.join(csv_dir, 'corner_features.csv'), corner_rows,
               ['venue', 'surface', 'n_corners', 'inner_outer',
                'corner_shape_class', 'source'])

    log_path = os.path.join(output_dir, 'scrape_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print(f'\n✅ 完了。images/ と csv/ を確認してください: {output_dir}')
    print('⚠ start_to_corner.csv / elevation_features.csv / corner_features.csv の'
          'corner_shape_class は空欄が多いはずです。保存された画像を見て埋める'
          '後続作業が必要です。')
    return {
        'basic_rows': len(basic_rows),
        'distance_start_rows': len(distance_start_rows),
        'log_path': log_path,
    }


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else './course_data'
    scrape_all_courses(out)
