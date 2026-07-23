"""
JRA全10競馬場のコースデータ収集ツール（Colab専用・一括実行・一回限りのタスク）

【背景】
コース×距離別の特徴（ダートの芝スタート区間、坂の位置、コーナーのタイト度等）を
特徴量化するための土台として、JRA公式サイトからコース平面図・立体図・高低断面図・
コースデータを収集する。

このスクリプトは週次の本番パイプラインには組み込まない
（KEIBA_過去データ一括取得_v4.ipynb と同じ位置づけの、一回限りのデータ収集ツール）。

【実機検証済みの構造（2026-07-22、中山競馬場の生HTMLで確認）】
`https://www.jra.go.jp/facilities/race/{venue_eng}/course/` は Shift_JIS エンコード。
- 画像: `<div class="block_unit"><h3>{見出し}</h3>...<div class="img"><img src="img/xxx.jpg"></div>`
  という構造の繰り返し。見出しは「コース立体図（右回り）」「コース平面図（右回り）」
  「芝コース高低断面図（右・内回り）」「芝コース高低断面図（右・外回り）」
  「ダートコース高低断面図（右回り）」等（内回り/外回りがあるコースは芝断面図が2枚）。
  img の src はページURL相対パス（`urljoin` が必須。ルート相対ではない）
- コースデータ: `<table><caption>芝コース：コースデータ</caption>...` のような
  captionつきtableが複数（芝コース基本データ/芝コース各コース(A/B/C)データ/
  ダートコースデータ）。thのラベルでtdの列を対応付けて読む
- コース紹介プロース文（`<div class="course_info">`）に、右回り/左回りや
  ゴール前坂の位置・高低差に加え、**「ダートのレースは1200メートルのみが芝スタート」**
  のような、ダートの芝スタート区間を明記した一文が含まれる（中山の実例で確認）。
  これは他の関数では拾えない、この一文でしか分からない情報

他9場のページも同じテンプレート（`course_common.css`）を使っている可能性が高いが、
内回り/外回りの有無や表記ゆれは venue ごとに異なりうるため、決め打ちにせず
キーワード・キャプション文字列で探索する設計は維持する。見つからない場合は
例外を出さず空欄のままログに警告する。

【北星ルールとの整合】
- 「決め打ちで実装しない」: 中山以外の9場は未検証のため、位置・インデックスではなく
  h3見出し・table caption・キーワードによるラベルベース探索を維持する
- 画像はGitHubリポジトリには含めない（JRA公式サイトの著作物のため）。
  Google Drive等、リポジトリ外の output_dir に保存すること

【出力】
{output_dir}/images/{venue_eng}_course.png              コース平面図
{output_dir}/images/{venue_eng}_3d.png                  コース立体図
{output_dir}/images/{venue_eng}_turf_elevation.png      芝高低断面図（1枚目。内回り等）
{output_dir}/images/{venue_eng}_turf_elevation_2.png    芝高低断面図2枚目（外回り等、あれば）
{output_dir}/images/{venue_eng}_dirt_elevation.png      ダート高低断面図
{output_dir}/images/{venue_eng}_raw.html                取得した生HTML（パーサー調整用）
{output_dir}/csv/course_basic.csv                       直線距離・高低差・一周距離・幅員・回り
{output_dir}/csv/distance_start.csv                     距離別の発走区分（内回り/外回り・芝スタート等）
{output_dir}/csv/start_to_corner.csv                    スタート〜1コーナー距離（空欄想定）
{output_dir}/csv/elevation_features.csv                 ゴール前坂の位置・高低差（プロース文から抽出）
{output_dir}/csv/corner_features.csv                    コーナー数・内外回り・タイト度分類（空欄想定）
{output_dir}/scrape_log.txt                             競馬場ごとの取得結果ログ

⚠ start_to_corner.csv / corner_features.csv の corner_shape_class は、テキストとして
明記されていない限り自動抽出できない（コース平面図を目視で読み取る必要がある）。

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
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.utils.config import HEADERS, JRA_BASE, PLACE_ENG, PLACE_NAMES

TURF_DISTANCES = [1000, 1200, 1400, 1500, 1600, 1700, 1800, 2000, 2200, 2300,
                   2400, 2500, 2600, 3000, 3200, 3400, 3600, 4000]
DIRT_DISTANCES = [1000, 1150, 1200, 1300, 1400, 1600, 1700, 1800, 1900, 2100, 2400]

# コース平面図/立体図/断面図の見出し(h3)を分類するためのキーワード
IMAGE_KIND_PATTERNS = [
    ('3d', re.compile(r'立体図')),
    ('course', re.compile(r'平面図|コース図|コースマップ')),
    ('turf_elevation', re.compile(r'芝.*(高低|断面)|高低.*芝')),
    ('dirt_elevation', re.compile(r'ダート.*(高低|断面)|高低.*ダート')),
]

TURN_PATTERN = re.compile(r'(右回り|左回り|右回|左回)')
# 「ダートのレースは1200メートルのみが芝スタート」のような一文から距離を抜く
TURF_START_SENTENCE_PATTERN = re.compile(r'[^。]*芝スタート[^。]*。')
DISTANCE_IN_TEXT_PATTERN = re.compile(r'([\d,]+)\s*メートル')
# 「残り180メートルから残り70メートル地点にかけて設けられている上り坂の高低差は2.2メートル」
HILL_PATTERN = re.compile(
    r'残り\s*([\d,]+)\s*メートルから残り\s*([\d,]+)\s*メートル地点にかけて設けられている'
    r'(上り|下り)坂の高低差は\s*([\d.]+)\s*メートル'
)


def _log(log_lines, msg):
    print(msg)
    log_lines.append(msg)


def _get(sess, url, log_lines, timeout=15):
    """JRA公式ページはShift_JISで配信されるため、レスポンスの文字コードを明示する
    （requestsのデフォルト判定に任せるとJRA公式ページは正しくデコードされず、
    後続の日本語正規表現・見出し検索が軒並み失敗する）。"""
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        if 'jra.go.jp' in url:
            r.encoding = 'shift_jis'
        return r
    except Exception as e:
        _log(log_lines, f'  ⚠ 取得失敗 {url}: {e}')
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


def _classify_heading(heading_text):
    hint = unicodedata.normalize('NFKC', heading_text or '')
    for kind, pattern in IMAGE_KIND_PATTERNS:
        if pattern.search(hint):
            return kind
    return None


def extract_course_images(soup, page_url):
    """div.block_unit の h3見出し + div.img img から画像URLを分類抽出する。

    Returns: list of (kind, absolute_img_url, heading_text)
    """
    results = []
    for block in soup.select('div.block_unit'):
        h3 = block.find('h3')
        img = block.select_one('div.img img')
        if h3 is None or img is None:
            continue
        src = img.get('src', '')
        if not src:
            continue
        heading_text = h3.get_text(strip=True)
        kind = _classify_heading(heading_text)
        if not kind:
            continue
        abs_url = urljoin(page_url, src)
        results.append((kind, abs_url, heading_text))
    return results


def _parse_data_table(table):
    """captionつきtableをth見出し→td値のdictのリスト（行ごと）に変換する。"""
    caption_tag = table.find('caption')
    caption = unicodedata.normalize('NFKC', caption_tag.get_text(strip=True)) if caption_tag else ''

    header_cells = table.find_all('th')
    headers = [unicodedata.normalize('NFKC', th.get_text(strip=True)) for th in header_cells]

    rows = []
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        values = [unicodedata.normalize('NFKC', td.get_text(' ', strip=True)) for td in tds]
        if len(values) != len(headers):
            continue
        rows.append(dict(zip(headers, values)))
    return caption, rows


def extract_course_tables(soup):
    """div.block_unit の h3見出し（「芝コース」「ダートコース」）でtableをグルーピングする。

    tableごとのcaption文字列（「芝コース：コースデータ」等）は場によって表記が
    バラバラ・またはcaption自体が存在しない場合がある（阪神は"内回り"/"外回り"の
    みのcaption、京都は直線距離を含むtableにcaptionが無い、等を実機HTMLで確認済み）。
    caption文字列に頼らず、より安定して存在する親h3見出しで分類することで、
    同じ「芝コース」ブロック内にある複数tableの値をまとめて拾えるようにする。

    Returns: dict with keys 'turf', 'dirt'（それぞれ _parse_data_table の rows の合計。
             見つからなければ空リスト）
    """
    out = {'turf': [], 'dirt': []}
    for block in soup.select('div.block_unit'):
        h3 = block.find('h3')
        if h3 is None:
            continue
        heading = unicodedata.normalize('NFKC', h3.get_text(strip=True))
        if heading == '芝コース':
            key = 'turf'
        elif heading == 'ダートコース':
            key = 'dirt'
        else:
            continue
        for table in block.find_all('table'):
            _, rows = _parse_data_table(table)
            out[key].extend(rows)
    return out


def _first_value(rows, *label_candidates):
    for row in rows:
        for label in label_candidates:
            if label in row and row[label]:
                return row[label]
    return None


def _clean_meters(text):
    """'310m' や '5.3m' のような文字列から数値部分だけを取り出す（先頭の値のみ）。"""
    if not text:
        return None
    m = re.search(r'([\d,]+(?:\.\d+)?)', text.replace(',', ''))
    return m.group(1) if m else None


def extract_course_info(soup):
    """course_info（コース紹介プロース文）から回り・坂・芝スタート情報を抽出する。

    Returns: dict with keys turn, hill (dict|None), turf_start_distances (list[str]),
             raw_text
    """
    div = soup.find('div', class_='course_info')
    if div is None:
        return {'turn': None, 'hill': None, 'turf_start_distances': [], 'raw_text': ''}

    text = unicodedata.normalize('NFKC', div.get_text(' ', strip=True))

    turn_m = TURN_PATTERN.search(text)
    turn = turn_m.group(1) if turn_m else None

    hill = None
    hill_m = HILL_PATTERN.search(text)
    if hill_m:
        hill = {
            'hill_start_desc': f'残り{hill_m.group(1)}m',
            'hill_end_desc': f'残り{hill_m.group(2)}m',
            'hill_direction': hill_m.group(3),
            'elevation_diff_m': hill_m.group(4),
        }

    turf_start_distances = []
    sentence_m = TURF_START_SENTENCE_PATTERN.search(text)
    if sentence_m:
        turf_start_distances = [
            d.replace(',', '') for d in DISTANCE_IN_TEXT_PATTERN.findall(sentence_m.group(0))
        ]

    return {
        'turn': turn, 'hill': hill,
        'turf_start_distances': turf_start_distances, 'raw_text': text,
    }


def _scrape_jra_venue(sess, venue_ja, venue_eng, images_dir, log_lines):
    """JRA公式のコースページから画像・コースデータ・コース紹介文を取得する。"""
    url = f'{JRA_BASE}/facilities/race/{venue_eng}/course/'
    _log(log_lines, f'[{venue_ja}] JRA公式取得: {url}')
    r = _get(sess, url, log_lines)
    if r is None:
        return None

    raw_path = os.path.join(images_dir, f'{venue_eng}_raw.html')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(r.text)

    soup = BeautifulSoup(r.text, 'lxml')

    kind_counts = {}
    for kind, img_url, heading_text in extract_course_images(soup, url):
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        suffix = '' if kind_counts[kind] == 1 else f'_{kind_counts[kind]}'
        dest = os.path.join(images_dir, f'{venue_eng}_{kind}{suffix}.png')
        if _download_image(sess, img_url, dest, log_lines):
            _log(log_lines, f'  ✅ 画像保存: {os.path.basename(dest)}'
                             f'（見出し「{heading_text}」→ 種類: {kind}）')

    for expected in ('course', '3d', 'turf_elevation', 'dirt_elevation'):
        if expected not in kind_counts:
            _log(log_lines, f'  ⚠ 画像未検出: {expected}（h3見出しの表記がこの場では'
                             f'異なる可能性。raw.htmlを確認してください）')

    tables = extract_course_tables(soup)
    if not any(tables.values()):
        _log(log_lines, '  ⚠ コースデータtableが見つからない（h3見出し「芝コース」'
                         '「ダートコース」がこの場では異なる可能性）')

    info = extract_course_info(soup)
    if info['hill'] is None:
        _log(log_lines, '  ⚠ ゴール前坂の位置・高低差をプロース文から抽出できず'
                         '（この場では言い回しが異なる可能性。目視確認が必要）')

    straight_length = _first_value(tables['turf'], '直線距離')
    elevation_diff = _first_value(tables['turf'], '高低差') or (
        info['hill']['elevation_diff_m'] if info['hill'] else None)
    turf_lap = _first_value(tables['turf'], '一周距離')
    width = _first_value(tables['turf'], '幅員')
    dirt_lap = _first_value(tables['dirt'], '一周距離')
    dirt_straight = _first_value(tables['dirt'], '直線距離')
    dirt_width = _first_value(tables['dirt'], '幅員')
    turf_start_desc = _first_value(tables['turf'], '発走距離')
    dirt_start_desc = _first_value(tables['dirt'], '発走距離')

    return {
        'source': url,
        'turn': info['turn'],
        'straight_length_m': _clean_meters(straight_length),
        'elevation_diff_m': _clean_meters(elevation_diff),
        'turf_lap_m': _clean_meters(turf_lap),
        'dirt_lap_m': _clean_meters(dirt_lap),
        'dirt_straight_length_m': _clean_meters(dirt_straight),
        'width_m': width or dirt_width,
        'hill': info['hill'],
        'turf_start_dirt_distances': info['turf_start_distances'],
        'turf_start_desc_raw': turf_start_desc,
        'dirt_start_desc_raw': dirt_start_desc,
    }


def scrape_all_courses(output_dir, venues=None):
    """JRA全10競馬場のコースデータを収集する。

    Args:
        output_dir: 保存先ディレクトリ（Google Drive配下等。GitHubリポジトリ配下は
            画像の著作物配布になるため避けること）
        venues: 対象競馬場の英語スラッグリスト（Noneなら全10場。PLACE_ENGのキー）
    """
    images_dir = os.path.join(output_dir, 'images')
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    venues = venues or list(PLACE_ENG.keys())
    eng_to_ja = {eng: PLACE_NAMES[code] for eng, code in PLACE_ENG.items()}

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
        if basic is None:
            _log(log_lines, f'  ❌ {venue_ja}: 取得失敗。行はスキップ')
            time.sleep(1.0)
            continue

        basic_rows.append({
            'venue': venue_ja, 'venue_eng': venue_eng,
            'straight_length_m': basic.get('straight_length_m') or '',
            'elevation_diff_m': basic.get('elevation_diff_m') or '',
            'turf_lap_m': basic.get('turf_lap_m') or '',
            'dirt_lap_m': basic.get('dirt_lap_m') or '',
            'width_m': basic.get('width_m') or '',
            'turn': basic.get('turn') or '',
            'source': basic.get('source') or '',
        })

        corner_rows.append({
            'venue': venue_ja, 'surface': '',
            'n_corners': '', 'inner_outer': '',
            'corner_shape_class': '',  # Very Tight〜Very Wide は目視分類が必要
            'source': basic.get('source') or '',
        })

        turf_start_dirt = set(basic.get('turf_start_dirt_distances') or [])
        # プロース文で芝スタートと確認された距離が既定の候補リストに無い場合も
        # 行が失われないよう、候補リストとの和集合を取る（例: 阪神の2000mダート）
        dirt_distances_for_venue = sorted(
            set(DIRT_DISTANCES) | {int(d) for d in turf_start_dirt if d.isdigit()}
        )
        for dist in TURF_DISTANCES:
            distance_start_rows.append({
                'venue': venue_ja, 'surface': '芝', 'distance_m': dist,
                'start_position_desc': basic.get('turf_start_desc_raw') or '',
                'start_image_file': '', 'source': basic.get('source') or '',
            })
        for dist in dirt_distances_for_venue:
            distance_start_rows.append({
                'venue': venue_ja, 'surface': 'ダート', 'distance_m': dist,
                'start_position_desc': (
                    '芝スタート' if str(dist) in turf_start_dirt else ''
                ),
                'start_image_file': '', 'source': basic.get('source') or '',
            })

        for dist in TURF_DISTANCES + dirt_distances_for_venue:
            surface = '芝' if dist in TURF_DISTANCES else 'ダート'
            start_to_corner_rows.append({
                'course': venue_ja, 'surface': surface, 'distance': dist,
                'start_to_corner_m': '', 'estimation_method': '', 'source': '',
            })

        hill = basic.get('hill')
        for surface in ('芝', 'ダート'):
            elevation_rows.append({
                'venue': venue_ja, 'surface': surface,
                'hill_start_desc': (hill or {}).get('hill_start_desc', ''),
                'hill_end_desc': (hill or {}).get('hill_end_desc', ''),
                'hill_direction': (hill or {}).get('hill_direction', ''),
                'elevation_diff_m': (hill or {}).get('elevation_diff_m', ''),
                'source': basic.get('source') or '' if hill else '',
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
    print('⚠ start_to_corner.csv / corner_features.csv の corner_shape_class は'
          '空欄のはずです。保存された画像を見て埋める後続作業が必要です。')
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
