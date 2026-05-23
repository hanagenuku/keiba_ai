import re
import time
import sqlite3
import statistics
from collections import defaultdict
from bs4 import BeautifulSoup

from src.utils.config import JRA_BASE, HEADERS, PLACE_NAMES
from src.scraper.calendar import get_base_from_calendar, get_kaisai_on_date
from src.scraper.parser import (
    parse_header, parse_rname, parse_hist, parse_horse,
    get_class_from_racename,
)


def calc_suffix(r01, r):
    if r <= 9:
        return f'{(r01 + (r - 1) * 181) % 256:02X}'
    elif r == 10:
        return f'{(r01 + 8 * 181 + 245) % 256:02X}'
    else:
        return f'{(r01 + 8 * 181 + 245 + (r - 10) * 181) % 256:02X}'


def find_r01_shutuba(base, date, sess):
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        r = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                      data={'cname': cn, 'CNAME': cn}, timeout=10)
        r.encoding = 'shift_jis'
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


def find_r01_result(base, date, sess):
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        r = sess.post(f'{JRA_BASE}/JRADB/accessS.html', data={'CNAME': cn}, timeout=10)
        r.encoding = 'shift_jis'
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


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
        for r in range(1, 13):
            sx = calc_suffix(r01, r)
            cn = f'{base}{r:02d}{date_str}/{sx}'
            resp = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                             data={'cname': cn, 'CNAME': cn},
                             headers=HEADERS, timeout=15)
            resp.encoding = 'shift_jis'
            if 'パラメータエラー' in resp.text:
                continue
            soup = BeautifulSoup(resp.text, 'lxml')
            if not soup.find_all('table'):
                continue
            race = _parse_shutuba(soup, rc, r, date_str, pc, hist_db_path)
            if not race:
                continue
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
        if info.get('surface') == '障害':
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

        rows = conn.execute('''
            SELECT h.race_id, h.date, h.distance, h.surface,
                   h.place, h.agari3f, h.running_style,
                   h.corner_3, r.first_3f, h.horse_num
            FROM horse_history h
            LEFT JOIN race_history r ON h.race_id = r.race_id
            WHERE h.horse_name = ?
            ORDER BY h.date DESC, h.race_id DESC
            LIMIT ?
        ''', (horse_name, limit)).fetchall()

        if not rows and len(horse_name) >= 4:
            rows = conn.execute('''
                SELECT h.race_id, h.date, h.distance, h.surface,
                       h.place, h.agari3f, h.running_style,
                       h.corner_3, r.first_3f, h.horse_num
                FROM horse_history h
                LEFT JOIN race_history r ON h.race_id = r.race_id
                WHERE h.horse_name LIKE ?
                ORDER BY h.date DESC, h.race_id DESC
                LIMIT ?
            ''', (horse_name[:5] + '%', limit)).fetchall()

        if not rows:
            conn.close()
            return []

        results = []
        for row in rows:
            race_id, date, distance, surface, place, agari3f, \
                running_style_hist, corner_3, first_3f_val, horse_num_val = row
            finishers = conn.execute(
                'SELECT COUNT(*) FROM horse_history WHERE race_id=?', (race_id,)
            ).fetchone()[0]
            winner = conn.execute(
                'SELECT agari3f FROM horse_history WHERE race_id=? AND place=1',
                (race_id,),
            ).fetchone()
            if winner and winner[0] and agari3f and place > 1:
                margin = max(0.0, round((agari3f - winner[0]) * 0.3, 2))
            else:
                margin = 0.0
            if agari3f:
                all_agari = conn.execute(
                    'SELECT agari3f FROM horse_history WHERE race_id=? AND agari3f IS NOT NULL',
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
            results.append({
                'place': place,
                'finishers': max(finishers, 1),
                'distance': distance,
                'surface': surface,
                'class': '1勝クラス',
                'margin': margin,
                'agari3f_rank_pct': round(agari3f_rank_pct, 3),
                'condition': '良',
                'date': date,
                'last_3f': agari3f,
                'first_3f': first_3f_val,
                'corner_3': corner_3,
                'race_id': race_id,
                'running_style': running_style_hist,
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
    idx = text.find('ワイド')
    if idx >= 0:
        wm = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 300])
        if wm:
            divs['wide'] = [{'nums': [int(w[0]), int(w[1])], 'payout': int(w[2].replace(',', ''))} for w in wm[:3]]
    return divs


def parse_result_soup(soup, racecourse, race_num, date, place_code):
    try:
        tables = soup.find_all('table')
        header = tables[0].get_text(' ', strip=True)
        info = {
            'racecourse': racecourse,
            'race_num': race_num,
            'race_id': f'{date}_{place_code}_{race_num:02d}',
        }
        dm = re.search(r'([\d,]+)\s*メートル\s*[（(]\s*([芝ダ])', header)
        info['distance'] = int(dm.group(1).replace(',', '')) if dm else 2000
        info['surface'] = '芝' if dm and dm.group(2) == '芝' else 'ダート'
        c = header.replace('本賞金', '').replace('付加賞', '')
        sp = re.search(r'([぀-鿿゠-ヿa-zA-Z0-9]+(?:賞|杯|記念|特別|ステークス|カップ|トロフィー))', c)
        gen = re.search(r'(\d歳(?:以上)?(?:未勝利|1勝クラス|2勝クラス|3勝クラス|オープン))', header)
        info['race_name'] = (
            sp.group(1).strip()
            if sp and sp.group(1) not in ('本賞', '付加賞') and len(sp.group(1)) >= 3
            else gen.group(1).strip() if gen else ''
        )
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
            num_m = re.match(r'^(\d+)$', texts[2].strip())
            num = int(num_m.group(1)) if num_m else 0
            name_m = re.match(
                r'^([゠-ヿ一-鿿A-Za-z][゠-ヿ一-鿿A-Za-z0-9・]{1,20})',
                texts[3].strip(),
            )
            name = name_m.group(1).strip() if name_m else texts[3].strip()[:10]
            pos_nums = re.findall(r'\d+', texts[9] if len(texts) > 9 else '')
            if pos_nums:
                positions = [int(n) for n in pos_nums[:4]]
                first = positions[0]
                avg = sum(positions) / len(positions)
                style = '逃げ' if first == 1 else '先行' if avg <= 3 else '差し' if avg <= 7 else '追込'
            else:
                style = '差し'
            agari_m = re.search(r'(\d{2}\.\d)', texts[10]) if len(texts) > 10 else None
            agari = float(agari_m.group(1)) if agari_m else 0.0
            pop_m = re.match(r'^(\d+)$', texts[13].strip()) if len(texts) > 13 else None
            jockey = texts[6].strip() if len(texts) > 6 else ''
            trainer = texts[12].strip() if len(texts) > 12 else ''
            finishers.append({
                'place': place, 'num': num, 'name': name,
                'running_style': style, 'post_position': num,
                'agari3f': agari,
                'popularity': int(pop_m.group(1)) if pop_m else 99,
                'jockey': jockey, 'trainer': trainer,
                'distance': info['distance'], 'surface': info['surface'],
            })
        divs = parse_dividends(soup)
        if not finishers:
            return None
        info['finishers'] = finishers
        info['dividends'] = divs
        return info
    except Exception:
        return None


def fetch_results(sess, target_date, calendar=None):
    """指定日の全レース結果を取得"""
    from src.scraper.calendar import get_base_from_calendar
    from src.utils.config import KAISAI_CALENDAR
    cal = calendar if calendar is not None else KAISAI_CALENDAR
    print(f'📡 {target_date} 結果取得中...')
    all_results = []
    for pc in cal:
        base_shutuba = get_base_from_calendar(pc, target_date, cal)
        if not base_shutuba:
            continue
        base_result = base_shutuba.replace('pw01dde01', 'pw01sde10')
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
            resp = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                             data={'CNAME': cn}, headers=HEADERS, timeout=15)
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
