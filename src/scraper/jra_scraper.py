import re
import time
import sqlite3
import statistics
import unicodedata
from collections import defaultdict
from bs4 import BeautifulSoup

from src.utils.config import JRA_BASE, HEADERS, PLACE_NAMES
from src.scraper.calendar import get_base_from_calendar, get_kaisai_on_date
from src.scraper.parser import (
    parse_header, parse_rname, parse_hist, parse_horse,
    get_class_from_racename, _detect_surface,
)


def calc_suffix(r01, r):
    if r <= 9:
        return f'{(r01 + (r - 1) * 181) % 256:02X}'
    elif r == 10:
        return f'{(r01 + 8 * 181 + 245) % 256:02X}'
    else:
        return f'{(r01 + 8 * 181 + 245 + (r - 10) * 181) % 256:02X}'


def find_r01_shutuba(base, date, sess):
    """R01出走表のsuffixを探索する。障害レースも含めて最初に見つかったR01のsuffixを返す。

    suffixは0x00〜0xFFを総当たりする（オッズ側 find_r01_odds と同じ全走査方式）。
    JRADBは間違ったsuffixに対してパラメータエラーを返すが、R01の正しいsuffixが
    0〜255のどこにあるかは不定。途中で打ち切ると suffix が大きいときに取りこぼし、
    出馬表が常に取得できなくなるため、早期breakせず最後まで走査する。
    障害レースのフィルタリングは呼び出し側（_parse_shutuba）で行う。
    """
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                          data={'cname': cn, 'CNAME': cn}, headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception:
            continue
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


def find_r01_result(base, date, sess):
    """R01結果ページのsuffixを探索する。障害レースも含めて最初に見つかったR01のsuffixを返す。

    suffixは0x00〜0xFFを総当たりする（find_r01_shutuba / find_r01_odds と同じ全走査方式）。
    早期打ち切りは suffix が大きいとき取りこぼすため行わない。
    障害レースのフィルタリングは parse_result_soup 内で行う（Noneを返す）。
    """
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                          data={'CNAME': cn}, headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception:
            continue
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


def _try_fetch_shutuba(sess, base, r, date_str, sx):
    """指定suffixで出走表ページを取得。(resp, soup) を返す。パラメータエラーの場合はNone, None。"""
    cn = f'{base}{r:02d}{date_str}/{sx}'
    resp = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                     data={'cname': cn, 'CNAME': cn},
                     headers=HEADERS, timeout=15)
    resp.encoding = 'shift_jis'
    if 'パラメータエラー' in resp.text:
        return None, None
    soup = BeautifulSoup(resp.text, 'lxml')
    if not soup.find_all('table'):
        return None, None
    return resp, soup


# 単勝・複勝オッズページ(accessO.html)のCNAME prefix。
# 出馬表は 'pw01dde01' だが、オッズページは 'pw151ouS3' を使用する（実機検証済み）。
ODDS_PREFIX = 'pw151ouS3'


def _to_odds_base(base):
    """出馬表用base(pw01dde01...)をオッズページ用base(pw151ouS3...)に変換する。"""
    return re.sub(r'^pw01dde01', ODDS_PREFIX, base)


def find_r01_odds(odds_base, date_str, sess):
    """単勝・複勝オッズページ(accessO.html)のR01 suffixを探索する。

    CNAMEは「レース番号(01) + 日付 + Z + / + suffix」の形式（実機検証済み）。
    suffixを0x00〜0xFFで総当たりし、テーブルが取得できた値を返す。
    """
    for s in range(256):
        cn = f'{odds_base}01{date_str}Z/{s:02X}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessO.html',
                          data={'cname': cn, 'CNAME': cn}, headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception:
            continue
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


def fetch_odds_for_race(sess, odds_base, race_num, date_str, sx):
    """指定レースの単勝・複勝オッズを取得する。

    オッズページ(accessO.html / pw151ouS3系CNAME)のテーブルを解析する。
    「枠」列はrowspanで複数馬にまたがるため、同枠2頭目以降の行には
    枠セルが無く、セル数が1つ少なくなる（セル数9/10で列位置を切り替え）。

    Returns:
        {horse_num: {'tansho': float|None, 'fukusho': float|None}}
    """
    cn = f'{odds_base}{race_num:02d}{date_str}Z/{sx}'
    try:
        resp = sess.post(f'{JRA_BASE}/JRADB/accessO.html',
                         data={'cname': cn, 'CNAME': cn},
                         headers=HEADERS, timeout=15)
        resp.encoding = 'shift_jis'
        if 'パラメータエラー' in resp.text:
            return {}
        soup = BeautifulSoup(resp.text, 'lxml')
    except Exception:
        return {}

    odds_map = {}
    for table in soup.find_all('table'):
        for tr in table.find_all('tr'):
            cells = [unicodedata.normalize('NFKC', c.get_text(strip=True))
                     for c in tr.find_all(['td', 'th'])]
            if len(cells) not in (9, 10):
                continue
            # 10セル: 枠 馬番 馬名 ... / 9セル: (枠省略) 馬番 馬名 ...
            offset = 1 if len(cells) == 10 else 0
            horse_cell = cells[offset]
            if not re.match(r'^\d{1,2}$', horse_cell):
                continue
            horse_num = int(horse_cell)
            if not (1 <= horse_num <= 18):
                continue

            tansho = None
            fukusho = None
            for cell in cells[offset + 1:]:
                # 複勝オッズ: "X.X - Y.Y" 形式の範囲表示 → 中央値を採用
                fm = re.match(r'^(\d{1,4}\.\d)\s*[-~〜]\s*(\d{1,4}\.\d)$', cell)
                if fm:
                    fukusho = round((float(fm.group(1)) + float(fm.group(2))) / 2, 1)
                    continue
                # 単勝オッズ: "X.X" 単独表示（複勝より先に出現する想定）
                tm = re.match(r'^(\d{1,4}\.\d)$', cell)
                if tm and tansho is None:
                    tansho = float(tm.group(1))

            if tansho is not None or fukusho is not None:
                odds_map[horse_num] = {'tansho': tansho, 'fukusho': fukusho}

    return odds_map


def fetch_odds_map(sess, races):
    """races（fetch_races_on_dateの戻り値）の各レースについて
    単勝・複勝オッズを取得し、to_app_json の market_odds_map 形式で返す。

    開催（_odds_cn['base']）ごとにR01のsuffixを1回だけ探索し、
    各レースのsuffixは calc_suffix で算出する。

    Args:
        sess  : requests.Session
        races : fetch_races_on_date が返すレース辞書のリスト
                （各要素に _odds_cn キーが必要）

    Returns:
        {race_id: {horse_num: {'tansho': float|None, 'fukusho': float|None}}}
        取得失敗したレースは空dict（market_odds_map[race_id] = {}）。
    """
    market_odds_map = {}
    r01_cache = {}
    for race in races:
        cn = race.get('_odds_cn')
        if not cn:
            continue
        odds_base = _to_odds_base(cn['base'])
        if odds_base not in r01_cache:
            r01_cache[odds_base] = find_r01_odds(odds_base, cn['date_str'], sess)
        r01 = r01_cache[odds_base]
        if r01 is None:
            market_odds_map[race['id']] = {}
            continue
        sx = calc_suffix(r01, cn['race_num'])
        odds_map = fetch_odds_for_race(sess, odds_base, cn['race_num'], cn['date_str'], sx)
        market_odds_map[race['id']] = odds_map
        time.sleep(0.5)
    return market_odds_map


def fetch_races_on_date(sess, target_date, hist_db_path):
    """指定日の全レース出走表を取得"""
    print(f'📡 {target_date} 出走表取得中...')
    all_races = []
    links = get_kaisai_on_date(target_date, sess)
    for base, date_str in links.items():
        pc = re.search(r'pw01dde01(\d{2})', base)
        pc = pc.group(1) if pc else '00'
        rc = PLACE_NAMES.get(pc, '?')
        print(f'\n🏟 {rc}  suffix探索...', end=' ', flush=True)
        r01 = find_r01_shutuba(base, date_str, sess)
        if r01 is None:
            print('❌')
            continue
        print(f'✅ {r01:02X}')
        # オッズページ専用のr01を探索（suffixがシャトウバと異なる）
        odds_base = _to_odds_base(base)
        odds_r01 = find_r01_odds(odds_base, date_str, sess)
        print(f'  オッズR01: {odds_r01:02X}' if odds_r01 is not None else '  オッズR01: 未発見')
        for r in range(1, 13):
            sx = calc_suffix(r01, r)
            _, soup = _try_fetch_shutuba(sess, base, r, date_str, sx)

            # R10以降：suffixが合わない場合は単純式(r-1)*181でも試みる
            if soup is None and r >= 10:
                sx_simple = f'{(r01 + (r - 1) * 181) % 256:02X}'
                if sx_simple != sx:
                    _, soup2 = _try_fetch_shutuba(sess, base, r, date_str, sx_simple)
                    if soup2 is not None:
                        soup = soup2
                        sx = sx_simple

            if soup is None:
                print(f'  R{r:02d}: suffix={sx} → パラメータエラー/ページなし')
                continue

            race = _parse_shutuba(soup, rc, r, date_str, pc, hist_db_path)
            if not race:
                # 原因を特定するため詳細ログを出力
                try:
                    tables = soup.find_all('table')
                    if tables:
                        header_text = tables[0].get_text(' ', strip=True)
                        info_tmp = parse_header(header_text)
                        if info_tmp.get('surface') == '障害':
                            print(f'  R{r:02d}: 障害レース → スキップ')
                        else:
                            expected = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
                            got_date = info_tmp.get('date', '?')
                            if got_date and got_date != expected:
                                print(f'  R{r:02d}: 日付不一致 expected={expected} got={got_date} (suffix={sx})')
                            else:
                                print(f'  R{r:02d}: parse失敗 (馬なし or 例外) suffix={sx}')
                    else:
                        print(f'  R{r:02d}: テーブルなし suffix={sx}')
                except Exception:
                    print(f'  R{r:02d}: ログ取得中に例外 suffix={sx}')
                time.sleep(0.3)
                continue

            # オッズ取得用のCNAME情報を保持（fetch_odds_for_race で使用）
            race['_odds_cn'] = {'base': base, 'date_str': date_str, 'sx': sx, 'race_num': r, 'odds_r01': odds_r01}

            all_races.append(race)
            print(f'  R{r:02d}: {race.get("race_name", "")} '
                  f'{race.get("num_horses", 0)}頭 '
                  f'{race.get("distance", 0)}m{race.get("surface", "")}')
            time.sleep(0.8)
    print(f'\n📋 出走表取得完了: {len(all_races)}レース')
    return all_races


def _parse_shutuba(soup, racecourse, race_num, date, place_code, hist_db_path):
    try:
        tables = soup.find_all('table')
        if not tables:
            return None
        header_text = tables[0].get_text(' ', strip=True)
        info = parse_header(header_text)
        if info.get('surface') in ('障害', '不明'):
            return None
        # suffixズレ検知: ページの日付が指定日と合わない場合はスキップ
        expected_date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
        if info.get('date') and info['date'] != expected_date:
            return None
        info['race_num'] = race_num
        info['racecourse'] = racecourse
        info['race_name'] = parse_rname(header_text, race_num)
        info['id'] = f"{date}_{place_code}_{race_num:02d}"
        surf = info.get('surface', '芝')
        horses = []
        for row in tables[0].find_all('tr'):
            cells = row.find_all('td')
            h = parse_horse(cells, racecourse, surf)
            if not h:
                continue
            hist = get_history_from_db(h['name'], hist_db_path)
            h['history'] = hist
            h['running_style'] = _infer_running_style(h['name'], hist, h.get('post_position'))
            horses.append(h)
        if not horses:
            return None
        info['horses'] = horses
        info['num_horses'] = len(horses)
        # 脚質カウント（calc_pace_distribution が使う）
        info['escape_count'] = sum(1 for h in horses if h.get('running_style') == '逃げ')
        info['front_count']  = sum(1 for h in horses if h.get('running_style') == '先行')
        return info
    except Exception:
        return None


def _infer_running_style(horse_name, hist, post_position=None):
    if not hist:
        # 枠番を脚質の弱いプロキシとして使用（内枠=先行傾向）
        if post_position is not None and post_position <= 3:
            return '先行'
        return '差し'
    # 履歴に running_style が記録されていればそれを多数決で使う
    from collections import Counter
    styles = [h.get('running_style') for h in hist
              if h.get('running_style') and h.get('running_style') != '']
    if styles:
        return Counter(styles).most_common(1)[0][0]
    # corner_3 フォールバック
    corner_3_list = [h.get('corner_3') for h in hist if h.get('corner_3') is not None]
    if not corner_3_list:
        return '差し'
    avg = sum(corner_3_list) / len(corner_3_list)
    if avg <= 1.5:  return '逃げ'
    if avg <= 3.0:  return '先行'
    if avg <= 6.0:  return '差し'
    return '追込'


def get_history_from_db(horse_name, hist_db_path, limit=5):
    """history.dbから馬の直近N走を取得"""
    try:
        conn = sqlite3.connect(hist_db_path)

        rows = conn.execute("""
            SELECT h.race_id, h.date, h.distance, h.surface,
                   h.place, h.agari3f, h.running_style,
                   h.corner_3, r.first_3f, h.horse_num,
                   COALESCE(r.race_class, '1勝クラス') as race_class,
                   COALESCE(r.track_condition, '良') as track_condition,
                   COALESCE(h.margin, -1.0) as margin_stored,
                   COALESCE(h.agari_rank, -1) as agari_rank_stored,
                   COALESCE(r.num_finishers, 0) as num_finishers,
                       COALESCE(r.race_name, '') as race_name
            FROM horse_history h
            LEFT JOIN race_history r ON h.race_id = r.race_id
            WHERE h.horse_name = ?
            ORDER BY h.date DESC, h.race_id DESC
            LIMIT ?
        """, (horse_name, limit)).fetchall()

        if not rows and len(horse_name) >= 4:
            rows = conn.execute("""
                SELECT h.race_id, h.date, h.distance, h.surface,
                       h.place, h.agari3f, h.running_style,
                       h.corner_3, r.first_3f, h.horse_num,
                       COALESCE(r.race_class, '1勝クラス') as race_class,
                       COALESCE(r.track_condition, '良') as track_condition,
                       COALESCE(h.margin, -1.0) as margin_stored,
                       COALESCE(h.agari_rank, -1) as agari_rank_stored,
                       COALESCE(r.num_finishers, 0) as num_finishers,
                       COALESCE(r.race_name, '') as race_name
                FROM horse_history h
                LEFT JOIN race_history r ON h.race_id = r.race_id
                WHERE h.horse_name LIKE ?
                ORDER BY h.date DESC, h.race_id DESC
                LIMIT ?
            """, (horse_name[:5] + '%', limit)).fetchall()

        if not rows:
            conn.close()
            return []

        results = []
        for row in rows:
            (race_id, date, distance, surface, place, agari3f,
             running_style_hist, corner_3, first_3f_val, horse_num_val,
             race_class, track_condition, margin_stored,
             agari_rank_stored, num_finishers, race_name) = row

            if margin_stored >= 0:
                margin = margin_stored
            else:
                winner = conn.execute(
                    "SELECT agari3f FROM horse_history WHERE race_id=? AND place=1",
                    (race_id,),
                ).fetchone()
                if winner and winner[0] and agari3f and place > 1:
                    margin = max(0.0, round((agari3f - winner[0]) * 0.3, 2))
                else:
                    margin = 0.0

            if agari_rank_stored > 0:
                fn = num_finishers if num_finishers > 1 else max(agari_rank_stored, 1)
                agari3f_rank_pct = (agari_rank_stored - 1) / max(fn - 1, 1)
            elif agari3f:
                all_agari = conn.execute(
                    "SELECT agari3f FROM horse_history WHERE race_id=? AND agari3f IS NOT NULL",
                    (race_id,),
                ).fetchall()
                all_vals = sorted([x[0] for x in all_agari])
                if all_vals:
                    rank = sum(1 for v in all_vals if v < agari3f)
                    agari3f_rank_pct = rank / max(len(all_vals) - 1, 1)
                else:
                    agari3f_rank_pct = 0.5
            else:
                agari3f_rank_pct = 0.5

            if num_finishers > 0:
                finishers_count = num_finishers
            else:
                finishers_count = conn.execute(
                    "SELECT COUNT(*) FROM horse_history WHERE race_id=?", (race_id,)
                ).fetchone()[0]

            results.append({
                "place": place,
                "finishers": max(finishers_count, 1),
                "distance": distance,
                "surface": surface,
                "class": race_class,
                "margin": margin,
                "agari3f_rank_pct": round(agari3f_rank_pct, 3),
                "condition": track_condition,
                "date": date,
                "last_3f": agari3f,
                "first_3f": first_3f_val,
                "corner_3": corner_3,
                "race_id": race_id,
                "running_style": running_style_hist,
                "race_name": race_name,
            })
        conn.close()
        return results
    except Exception:
        return []



# ── 結果取得 ────────────────────────────────────────────────

def parse_dividends(soup):
    text = soup.get_text(' ', strip=True)
    divs = {}
    m = re.search(r'単勝\s+(\d+)\s+([\d,]+)\s*円', text)
    if m:
        divs['tansho'] = {'num': int(m.group(1)), 'payout': int(m.group(2).replace(',', ''))}
    idx = text.find('複勝')
    if idx >= 0:
        fm = re.findall(r'(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if fm:
            divs['fukusho'] = [{'num': int(f[0]), 'payout': int(f[1].replace(',', ''))} for f in fm[:3]]
    idx = text.find('馬連')
    if idx >= 0:
        um = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if um:
            divs['umaren'] = {'nums': [int(um[0][0]), int(um[0][1])],
                              'payout': int(um[0][2].replace(',', ''))}
    idx = text.find('ワイド')
    if idx >= 0:
        wm = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 300])
        if wm:
            divs['wide'] = [{'nums': [int(w[0]), int(w[1])], 'payout': int(w[2].replace(',', ''))} for w in wm[:3]]
    idx = text.find('三連複')
    if idx >= 0:
        sm = re.findall(r'(\d+)-(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if sm:
            divs['sanrenpuku'] = {'nums': [int(sm[0][0]), int(sm[0][1]), int(sm[0][2])],
                                  'payout': int(sm[0][3].replace(',', ''))}
    return divs



def _extract_class(header_text):
    """ヘッダ全文からクラスを堅実に判定する（race_name パースより信頼できる）。"""
    t = unicodedata.normalize('NFKC', header_text or '')
    if re.search(r'\(\s*G\s*3\s*\)|\(GIII\)', t): return 'G3'
    if re.search(r'\(\s*G\s*2\s*\)|\(GII\)',  t): return 'G2'
    if re.search(r'\(\s*G\s*1\s*\)|\(GI\)',   t): return 'G1'
    if re.search(r'\(\s*L\s*\)', t):              return 'L'
    if '3勝クラス' in t:  return '3勝'
    if '2勝クラス' in t:  return '2勝'
    if '1勝クラス' in t:  return '1勝'
    if '未勝利' in t:    return '未勝利'
    if '新馬' in t:      return '新馬'
    if 'オープン' in t:  return 'OP'
    return ''


def _parse_finish_time(text):
    """タイム文字列を秒に変換。'1:34.5' / '1.34.5' / '59.8' 等に対応。"""
    if not text: return 0.0
    t = str(text).strip().replace(' ', '')
    m = re.match(r'^(\d+)[:\.](\d{1,2})\.(\d)$', t)
    if m: return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10
    m = re.match(r'^(\d+(?:\.\d+)?)$', t)
    if m: return float(m.group(1))
    return 0.0


def _parse_margin(text):
    """着差テキストを数値（馬身）に変換する。"""
    if not text or text in ('---', '-', ''):
        return 0.0
    named = {'ハナ': 0.1, 'クビ': 0.2, 'アタマ': 0.3, '大差': 10.0}
    if text in named:
        return named[text]
    m = re.match(r'^(\d+)\s+(\d+)/(\d+)$', text.strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.match(r'^(\d+)/(\d+)$', text.strip())
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r'^(\d+(?:\.\d+)?)$', text.strip())
    if m:
        return float(m.group(1))
    return 0.0


def _extract_body_weight(texts, start_idx=10):
    """テキスト列から馬体重(増減)を抽出。
    フォーマット: '516(+4)' / '516(-2)' / '516' / '計不'
    返り値: (body_weight: int|None, body_weight_diff: int|None)
    """
    for t in texts[start_idx:]:
        s = t.strip()
        m = re.match(r'^(\d{3,4})\s*[\(（]\s*([+-]?\d{1,3})\s*[\)）]', s)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.match(r'^(\d{3,4})\s*$', s)
        if m:
            v = int(m.group(1))
            if 300 <= v <= 700:  # 馬体重の妥当範囲
                return v, None
    return None, None


def _extract_sex_age(texts, start_idx=4, end_idx=6):
    """性齢欄から性別と年齢を抽出。'牡3' / '牝4' / 'セ5' / '騙4' """
    for t in texts[start_idx:end_idx]:
        m = re.match(r'^([牡牝騸セ騙])\s*(\d+)', t.strip())
        if m:
            sex = m.group(1)
            if sex == '騙':
                sex = 'セ'
            return sex, int(m.group(2))
    return '', None


def _extract_weight_load(texts, start_idx=4, end_idx=7):
    """斤量を抽出。'57.0' / '54' / '57.5' 等。"""
    for t in texts[start_idx:end_idx]:
        s = t.strip()
        m = re.match(r'^(\d{2}(?:\.\d)?)$', s)
        if m:
            v = float(m.group(1))
            if 45.0 <= v <= 65.0:
                return v
    return None


def _extract_win_odds(texts, start_idx=10):
    """単勝オッズを抽出。row末尾付近の "NN.N" 形式の数字。
    タイム('1:34.5')/上がり3F(NN.N同形式だが既に取得済)等と区別が難しいので、
    indexが大きい後ろの方から探す。
    """
    for t in reversed(texts[start_idx:]):
        s = t.strip()
        m = re.match(r'^(\d{1,4}\.\d)$', s)
        if m:
            v = float(m.group(1))
            if 1.0 <= v <= 9999.9:
                return v
    return None


def _extract_weather_pace(header_text):
    """ヘッダから天候とペース判定を抽出。"""
    t = unicodedata.normalize('NFKC', header_text or '')
    weather = None
    wm = re.search(r'天候[\s:：]*([晴曇雨雪]+小?雨?)', t)
    if wm:
        weather = wm.group(1)
    else:
        for w in ['小雨', '小雪', '晴', '曇', '雨', '雪']:
            if w in t:
                weather = w
                break
    pace = None
    pm = re.search(r'ペース[\s:：]*([HMS])', t)
    if pm:
        pace = pm.group(1)
    return weather, pace


def _extract_lap_times(soup):
    """結果ページからラップタイム（ハロンごとの区間タイム）を抽出する。

    JRA結果ページは「ラップタイム」見出しに続けて
    "12.5 - 10.9 - 11.4 - 11.8 - ..." の形式で各200m区間タイムを掲載する。

    Returns:
        (lap_times: list[float], first_3f: float|None, last_3f: float|None)
        抽出できない場合は ([], None, None)。
    """
    text = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    idx = text.find('ラップタイム')
    if idx < 0:
        return [], None, None
    # 見出し以降・「ペース」または200文字までを対象に区間タイムを収集
    segment = text[idx:idx + 300]
    end = segment.find('ペース')
    if end > 0:
        segment = segment[:end]
    laps = [float(m) for m in re.findall(r'(\d{1,2}\.\d)', segment)]
    # ラップは概ね 9.0〜15.0 秒/200m の範囲。範囲外（誤検出）は除外
    laps = [v for v in laps if 8.0 <= v <= 16.0]
    if len(laps) < 3:
        return [], None, None
    first_3f = round(sum(laps[:3]), 1)
    last_3f = round(sum(laps[-3:]), 1)
    return laps, first_3f, last_3f


def parse_result_soup(soup, racecourse, race_num, date, place_code):
    try:
        tables = soup.find_all('table')
        header = tables[0].get_text(' ', strip=True)
        date_norm = f'{date[:4]}-{date[4:6]}-{date[6:8]}' if len(date) == 8 else date
        info = {
            'racecourse': racecourse,
            'race_num': race_num,
            'race_id': f'{date}_{place_code}_{race_num:02d}',
            'id':      f'{date}_{place_code}_{race_num:02d}',
            'date':    date_norm,
        }
        dm = re.search(r'([\d,]+)\s*[メ]ートル\s*[（(]\s*([芝ダ])', header)
        info['distance'] = int(dm.group(1).replace(',', '')) if dm else 0
        # surface: 堅実な多段判定（サイレントなフォールバック廃止）
        surf = _detect_surface(header)
        if surf in ('芝', 'ダート'):
            info['surface'] = surf
        elif surf == '障害':
            return None  # 障害は履歴対象外
        else:
            # 最終手段: 距離regex由来
            info['surface'] = '芝' if dm and dm.group(2) == '芝' else ('ダート' if dm and dm.group(2) == 'ダ' else None)
            if info['surface'] is None:
                return None  # 判定不能なら静かに捨てる（誤判定混入を避ける）
        c = header.replace('本賞金', '').replace('付加賞', '')
        sp = re.search(r'([぀-鿿゠-ヿa-zA-Z0-9]+(?:賞|杯|記念|特別|ステークス|カップ|トロフィー))', c)
        gen = re.search(r'(\d歳(?:以上)?(?:未勝利|1勝クラス|2勝クラス|3勝クラス|オープン))', header)
        info['race_name'] = (
            sp.group(1).strip()
            if sp and sp.group(1) not in ('本賞', '付加賞') and len(sp.group(1)) >= 3
            else gen.group(1).strip() if gen else ''
        )
        tc_m = re.search(r'(良|稍重|重|不良)', header)
        info['track_condition'] = tc_m.group(1) if tc_m else '良'
        info['race_class'] = _extract_class(header)
        # 天候・ペース判定（race-level）
        weather, pace = _extract_weather_pace(header)
        info['weather'] = weather
        info['pace_label'] = pace
        # ラップタイム（区間タイム）と前半/後半3F
        laps, first_3f, last_3f = _extract_lap_times(soup)
        info['lap_times'] = '-'.join(f'{v:.1f}' for v in laps) if laps else ''
        info['first_3f'] = first_3f
        info['last_3f'] = last_3f
        finishers = []
        for row in tables[0].find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 10:
                continue
            texts = [c.get_text(' ', strip=True) for c in cells]
            pm = re.match(r'^(\d+)$', texts[0].strip())
            if not pm:
                continue
            place = int(pm.group(1))
            # 枠番（texts[1]）
            br_m = re.match(r'^(\d+)$', texts[1].strip()) if len(texts) > 1 else None
            bracket = int(br_m.group(1)) if br_m else None
            num_m = re.match(r'^(\d+)$', texts[2].strip())
            num = int(num_m.group(1)) if num_m else 0
            name_m = re.match(
                r'^([゠-ヿA-Za-z][゠-ヿA-Za-z0-9・]{1,20})',
                texts[3].strip(),
            )
            name = name_m.group(1).strip() if name_m else texts[3].strip()[:10]
            # 性齢（texts[4]近辺）
            sex, age = _extract_sex_age(texts, start_idx=4, end_idx=6)
            # 斤量（texts[5]近辺）
            weight_load = _extract_weight_load(texts, start_idx=4, end_idx=7)
            # 通過順（既存ロジック：脚質推定用 + 全文保存）
            corner_all_text = texts[9] if len(texts) > 9 else ''
            pos_nums = re.findall(r'\d+', corner_all_text)
            if pos_nums:
                positions = [int(n) for n in pos_nums[:4]]
                first = positions[0]
                avg = sum(positions) / len(positions)
                style = '逃げ' if first == 1 else '先行' if avg <= 3 else '差し' if avg <= 7 else '追込'
            else:
                style = '差し'
            corner_all = '-'.join(pos_nums[:4]) if pos_nums else ''
            agari_m = re.search(r'(\d{2}\.\d)', texts[10]) if len(texts) > 10 else None
            agari = float(agari_m.group(1)) if agari_m else 0.0
            pop_m = re.match(r'^(\d+)$', texts[13].strip()) if len(texts) > 13 else None
            jockey = texts[6].strip() if len(texts) > 6 else ''
            trainer = texts[12].strip() if len(texts) > 12 else ''
            margin_txt = texts[8].strip() if len(texts) > 8 else ''
            finish_time = _parse_finish_time(texts[7].strip() if len(texts) > 7 else '')
            # 馬体重（texts[11]近辺、'516(+4)' 形式）
            body_weight, body_weight_diff = _extract_body_weight(texts, start_idx=10)
            # 単勝オッズ（row末尾付近の小数）
            win_odds = _extract_win_odds(texts, start_idx=13)
            finishers.append({
                'place': place, 'num': num, 'name': name,
                'running_style': style, 'post_position': num,
                'agari3f': agari,
                'popularity': int(pop_m.group(1)) if pop_m else 99,
                'jockey': jockey, 'trainer': trainer,
                'distance': info['distance'], 'surface': info['surface'],
                'margin': _parse_margin(margin_txt),
                'chakusa_text': margin_txt,
                'finish_time': finish_time,
                # 新フィールド
                'bracket': bracket,
                'sex': sex, 'age': age,
                'weight_load': weight_load,
                'body_weight': body_weight,
                'body_weight_diff': body_weight_diff,
                'corner_all': corner_all,
                'win_odds': win_odds,
            })
        divs = parse_dividends(soup)
        if not finishers:
            return None
        valid = sorted(
            [(i, h['agari3f']) for i, h in enumerate(finishers) if h['agari3f'] > 0],
            key=lambda x: x[1],
        )
        for rank, (i, _) in enumerate(valid):
            finishers[i]['agari_rank'] = rank + 1
        for h in finishers:
            if 'agari_rank' not in h:
                h['agari_rank'] = 99
        tan_payout = divs.get('tansho', {}).get('payout', 0)
        fuku_list = divs.get('fukusho', [])
        for h in finishers:
            h['tansho_payout'] = tan_payout if h['place'] == 1 else 0
            h['fukusho_payout'] = next(
                (f['payout'] for f in fuku_list if f['num'] == h['num']), 0)
        # 着差秒：勝ち馬との実タイム差
        winner = next((h for h in finishers if h.get('place') == 1), None)
        wt = winner['finish_time'] if winner and winner.get('finish_time', 0) > 0 else 0
        for h in finishers:
            ft = h.get('finish_time', 0)
            h['time_diff_sec'] = round(ft - wt, 2) if (wt > 0 and ft > 0) else None

        info['num_finishers'] = len(finishers)
        info['finishers'] = finishers
        info['dividends'] = divs
        return info
    except Exception:
        return None



def fetch_results(sess, target_date, calendar=None):
    """指定日の全レース結果を取得"""
    from src.scraper.calendar import get_kaisai_on_date
    print(f'📡 {target_date} 結果取得中...')
    all_results = []

    # Step1: 結果一覧(pw01sli00/AF)から sde_base を取得
    bases = {}
    try:
        r0 = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                       data={'CNAME': 'pw01sli00/AF'}, timeout=15)
        r0.encoding = 'shift_jis'
        soup0 = BeautifulSoup(r0.text, 'lxml')
        for tag in soup0.find_all(onclick=True):
            oc = tag.get('onclick', '')
            m = re.search(r'pw01srl\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/(\w{2})', oc)
            if not m:
                continue
            pc_m, year, kai, nichi, date = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            if date != target_date:
                continue
            base = f'pw01sde10{pc_m}{year}{kai}{nichi}'
            if base not in bases:
                bases[base] = target_date
                print(f"  📋 {PLACE_NAMES.get(pc_m, '?')} → {base}")
    except Exception as e:
        print(f'  ⚠ 結果一覧取得失敗: {e}')

    # フォールバック: 出走表一覧から変換
    if not bases:
        shutuba_bases = get_kaisai_on_date(target_date, sess)
        for shutuba_base in shutuba_bases:
            result_base = shutuba_base.replace('pw01dde01', 'pw01sde10')
            bases[result_base] = target_date
            pc_m = re.search(r'pw01sde10(\d{2})', result_base)
            pc_m = pc_m.group(1) if pc_m else '?'
            print(f"  📋(FB) {PLACE_NAMES.get(pc_m, '?')} → {result_base}")

    if not bases:
        print(f'  ❌ {target_date}の開催情報が見つかりません')
        return all_results

    for base_result, _ in bases.items():
        pc = re.search(r'pw01sde10(\d{2})', base_result)
        pc = pc.group(1) if pc else '00'
        rc = PLACE_NAMES.get(pc, '?')
        print(f'\n🏟 {rc}  suffix探索...', end=' ', flush=True)
        r01 = find_r01_result(base_result, target_date, sess)
        if r01 is None:
            print('❌')
            continue
        print(f'✅ {r01:02X}')
        for r in range(1, 13):
            sx = calc_suffix(r01, r)
            cn = f'{base_result}{r:02d}{target_date}/{sx}'
            resp = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                             data={'cname': cn, 'CNAME': cn}, timeout=15)
            resp.encoding = 'shift_jis'
            if 'パラメータエラー' in resp.text:
                continue
            soup = BeautifulSoup(resp.text, 'lxml')
            if not soup.find_all('table'):
                continue
            result = parse_result_soup(soup, rc, r, target_date, pc)
            if not result:
                continue
            all_results.append(result)
            top3 = result['finishers'][:3]
            t3 = ' '.join(
                f"{h['place']}着#{h['num']}{h['name'][:4]}({h['running_style']})"
                for h in top3
            )
            print(f'  R{r:02d}: {result.get("race_name", "")} {t3}')
            time.sleep(0.8)
    print(f'\n📋 結果取得完了: {len(all_results)}レース')
    return all_results


# ── バイアス分析 ─────────────────────────────────────────────

AGARI_BASE = {
    ('芝', 'sp'): 34.2, ('芝', 'mi'): 34.6, ('芝', 'md'): 35.0, ('芝', 'lo'): 35.5,
    ('ダート', 'sp'): 37.0, ('ダート', 'mi'): 37.5, ('ダート', 'md'): 38.0, ('ダート', 'lo'): 38.5,
}


def _dist_zone(d):
    d = int(d)
    if d <= 1400: return 'sp'
    if d <= 1800: return 'mi'
    if d <= 2200: return 'md'
    return 'lo'


def analyze_bias(results):
    bias_by_course = {}
    for rc in {r['racecourse'] for r in results}:
        rc_res = [r for r in results if r['racecourse'] == rc]
        io_scores = []
        for r in rc_res:
            fin = r['finishers']
            if len(fin) < 3:
                continue
            num_h = max(h['post_position'] for h in fin)
            avg_all = (num_h + 1) / 2
            avg_top3 = statistics.mean([h['post_position'] for h in fin[:3]])
            io_scores.append((avg_all - avg_top3) / max(num_h / 4, 1))
        inner_outer = max(-3, min(3, statistics.mean(io_scores) * 2)) if io_scores else 0
        style_cnt = defaultdict(int)
        total = 0
        for r in rc_res:
            for h in r['finishers'][:3]:
                style_cnt[h['running_style']] += 1
                total += 1
        front = (style_cnt['逃げ'] + style_cnt['先行']) / max(total, 1)
        pace_bias = max(-3, min(3, (front - 0.45) * 6))
        speed_devs = []
        for r in rc_res:
            fin = r['finishers']
            if not fin:
                continue
            winner = fin[0]
            agari = winner.get('agari3f', 0)
            if agari < 30:
                continue
            dist = winner.get('distance', r.get('distance', 2000))
            surf = winner.get('surface', r.get('surface', '芝'))
            zone = _dist_zone(dist)
            base_val = AGARI_BASE.get((surf, zone), 35.0)
            speed_devs.append(max(-2, min(2, (base_val - agari) / 0.8)))
        track_speed = round(statistics.mean(speed_devs), 2) if speed_devs else 0
        parts = []
        if abs(inner_outer) >= 1.0:
            parts.append('内有利' if inner_outer > 0 else '外有利')
        if abs(pace_bias) >= 1.0:
            parts.append('先行有利' if pace_bias > 0 else '差し・追込有利')
        if abs(track_speed) >= 0.5:
            parts.append('時計速め' if track_speed > 0 else '時計遅め')
        bias_by_course[rc] = {
            'inner_outer': round(inner_outer, 2),
            'pace_bias': round(pace_bias, 2),
            'track_speed': round(track_speed, 2),
            'summary': '・'.join(parts) if parts else 'フラット',
            'style_dist': dict(style_cnt),
            'race_count': len(rc_res),
        }
    return bias_by_course
