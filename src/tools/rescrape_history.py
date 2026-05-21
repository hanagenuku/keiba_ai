"""
history.db 再スクレイピングツール。
8頭打ち切りで欠けている過去レースの馬データを補完する。

使い方（Colabセルに貼る）:
    from src.tools.rescrape_history import run_rescrape
    run_rescrape(BASE_DIR, sess)

オプション:
    # 過去の開催カレンダーを手動指定する場合
    hist_cal = {
        "05": [{"kai": "01", "days": ["20250412", "20250413", ...]}],
        ...
    }
    run_rescrape(BASE_DIR, sess, custom_calendar=hist_cal)
"""
import re
import time
import sqlite3
from collections import defaultdict

from bs4 import BeautifulSoup

from src.utils.config import JRA_BASE, PLACE_NAMES
from src.scraper.calendar import get_kaisai_on_date, get_base_from_calendar
from src.scraper.jra_scraper import find_r01_result, calc_suffix, parse_result_soup
from src.utils.db import save_history_db, get_history_db_path


# ── ヘルパー ────────────────────────────────────────────────

def get_dates_to_rescrape(base_dir):
    """race_history のユニークな (date_str, place_code) ペアを返す（昇順）"""
    db_path = get_history_db_path(base_dir)
    conn = sqlite3.connect(db_path)
    rows = conn.execute('SELECT DISTINCT race_id FROM race_history').fetchall()
    conn.close()
    date_places = set()
    for (race_id,) in rows:
        parts = race_id.split('_')
        if len(parts) >= 2:
            date_places.add((parts[0], parts[1]))
    return sorted(date_places)


def max_horses_for_date_place(base_dir, date_str, place_code):
    """その日×競馬場で horse_history に記録済みの最大頭数を返す"""
    db_path = get_history_db_path(base_dir)
    conn = sqlite3.connect(db_path)
    pattern = f'{date_str}_{place_code}_%'
    row = conn.execute('''
        SELECT MAX(cnt) FROM (
            SELECT race_id, COUNT(*) AS cnt
            FROM horse_history
            WHERE race_id LIKE ?
            GROUP BY race_id
        )
    ''', (pattern,)).fetchone()
    conn.close()
    return row[0] if row and row[0] else 0


def _probe_base(base, date_str, sess, tries=8):
    """
    ベースURLが有効かどうかを少数プローブで確認する。
    有効なら r01 値を返し、無効なら None を返す。
    tries で最初に試す r01 の個数を指定（通常のfind_r01_resultは256回）。
    """
    import math
    step = max(1, 256 // tries)
    candidates = list(range(0, 256, step))
    for s in candidates:
        cn = f'{base}01{date_str}/{s:02X}'
        r = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                      data={'CNAME': cn}, timeout=10)
        r.encoding = 'shift_jis'
        if 'パラメータエラー' not in r.text and BeautifulSoup(r.text, 'lxml').find_all('table'):
            return s
        time.sleep(0.05)
    return None


def _find_base_bruteforce(date_str, place_code, sess, max_kai=6, max_nichi=10):
    """
    kai × nichi の組み合わせをブルートフォースでベースURLを探す。
    最初に8点プローブで弾き、ヒット候補のみ全探索する。
    """
    year = date_str[:4]
    for kai in range(1, max_kai + 1):
        for nichi in range(1, max_nichi + 1):
            base = f'pw01sde10{place_code}{year}{kai:02d}{nichi:02d}'
            # 8点プローブで存在確認
            quick = _probe_base(base, date_str, sess, tries=8)
            if quick is not None:
                print(f'      ✅ brute-force: kai={kai:02d} nichi={nichi:02d} r01={quick:02X}')
                return base, quick
    return None, None


def find_base_and_r01(date_str, place_code, sess, custom_calendar=None, brute_force=False):
    """
    指定日×競馬場のベースURL(pw01sde10形式)とr01を返す。
    1) custom_calendar
    2) JRA自動取得(get_kaisai_on_date)
    3) brute_force=True の場合のみ kai/nichi ブルートフォース
    """
    # 1) custom_calendar
    if custom_calendar:
        base_shutuba = get_base_from_calendar(place_code, date_str, custom_calendar)
        if base_shutuba:
            base_result = base_shutuba.replace('pw01dde01', 'pw01sde10')
            r01 = find_r01_result(base_result, date_str, sess)
            if r01 is not None:
                return base_result, r01

    # 2) JRA自動取得
    cal_to_use = custom_calendar if custom_calendar else {}
    links = get_kaisai_on_date(date_str, sess, calendar=cal_to_use)
    for base_shutuba in links:
        m = re.search(r'pw01dde01(\d{2})', base_shutuba)
        if m and m.group(1) == place_code:
            base_result = base_shutuba.replace('pw01dde01', 'pw01sde10')
            r01 = find_r01_result(base_result, date_str, sess)
            if r01 is not None:
                return base_result, r01

    # 3) ブルートフォース（オプション）
    if brute_force:
        print(f'    ⚡ brute-force開始: {PLACE_NAMES.get(place_code, "?")} {date_str}')
        base_result, r01 = _find_base_bruteforce(date_str, place_code, sess)
        if base_result and r01 is not None:
            return base_result, r01

    return None, None


def _scrape_one_place_day(base_result, r01, date_str, place_code, sess):
    """ベースURLとr01が判明しているレース一式を取得"""
    rc = PLACE_NAMES.get(place_code, place_code)
    results = []
    for r in range(1, 13):
        sx = calc_suffix(r01, r)
        cn = f'{base_result}{r:02d}{date_str}/{sx}'
        resp = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                         data={'CNAME': cn}, timeout=15)
        resp.encoding = 'shift_jis'
        if 'パラメータエラー' in resp.text:
            continue
        soup = BeautifulSoup(resp.text, 'lxml')
        if not soup.find_all('table'):
            continue
        result = parse_result_soup(soup, rc, r, date_str, place_code)
        if not result:
            continue
        n = len(result['finishers'])
        top3 = ' '.join(
            f"{h['place']}着#{h['num']}{h['name'][:4]}"
            for h in result['finishers'][:3]
        )
        print(f'    R{r:02d}: {n}頭  {top3}')
        results.append(result)
        time.sleep(0.8)
    return results


# ── メイン関数 ─────────────────────────────────────────────

def run_rescrape(
    BASE_DIR,
    sess,
    custom_calendar=None,
    skip_if_ge=10,
    brute_force=False,
    limit=None,
):
    """
    history.db の8頭打ち切りを解消するため、過去レースを再スクレイピングする。

    Args:
        BASE_DIR        : Colabのベースディレクトリ（例: '/content/drive/MyDrive/keiba_ai'）
        sess            : requests.Session（ログイン済み）
        custom_calendar : KAISAI_CALENDAR形式の過去開催辞書（省略可）
        skip_if_ge      : その日×場で既にこの頭数以上ならスキップ（デフォルト10）
        brute_force     : 自動取得に失敗した場合にkai/nichiBFを試みる（デフォルトFalse）
        limit           : テスト用・最大処理件数制限（省略時は全件）

    Returns:
        dict: {'done': [...], 'failed': [...], 'skipped': [...]}
    """
    date_places = get_dates_to_rescrape(BASE_DIR)
    print(f'📋 対象: {len(date_places)} 日×競馬場ペア')

    done, failed, skipped = [], [], []
    processed = 0

    for date_str, place_code in date_places:
        if limit is not None and processed >= limit:
            print(f'\n⏹ limit={limit} 件処理済み → 終了')
            break

        key = f'{date_str}_{place_code}'
        rc = PLACE_NAMES.get(place_code, place_code)

        # スキップ判定
        max_h = max_horses_for_date_place(BASE_DIR, date_str, place_code)
        if max_h >= skip_if_ge:
            skipped.append(key)
            continue

        print(f'\n🏟 {rc} {date_str}  (現在の最大頭数: {max_h})')
        base_result, r01 = find_base_and_r01(
            date_str, place_code, sess, custom_calendar, brute_force
        )

        if base_result is None:
            print(f'  ❌ ベースURL取得失敗')
            failed.append(key)
            processed += 1
            continue

        print(f'  🔗 {base_result}  r01={r01:02X}')
        results = _scrape_one_place_day(base_result, r01, date_str, place_code, sess)

        if results:
            save_history_db(results, base_dir=BASE_DIR)
            done.append(key)
        else:
            print(f'  ⚠ レース取得0件')
            failed.append(key)

        processed += 1

    print(f'\n📊 完了: {len(done)} / スキップ: {len(skipped)} / 失敗: {len(failed)}')
    if failed:
        print(f'  失敗リスト (先頭10件): {failed[:10]}{"..." if len(failed) > 10 else ""}')

    return {'done': done, 'failed': failed, 'skipped': skipped}


# ── ユーティリティ ──────────────────────────────────────────

def show_rescrape_summary(BASE_DIR):
    """
    現在の horse_history の頭数分布を表示する。
    再スクレイピングが必要なレースがどれくらいあるかを確認できる。
    """
    db_path = get_history_db_path(BASE_DIR)
    conn = sqlite3.connect(db_path)
    rows = conn.execute('''
        SELECT race_id, COUNT(*) AS cnt
        FROM horse_history
        GROUP BY race_id
        ORDER BY cnt
    ''').fetchall()
    conn.close()

    from collections import Counter
    dist = Counter(cnt for _, cnt in rows)
    total = len(rows)
    need = sum(cnt for cnt_horses, cnt in dist.items() if cnt_horses < 10)

    print(f'【horse_history 頭数分布】 総レース数: {total}')
    for n in sorted(dist):
        bar = '█' * min(dist[n] // 5, 40)
        pct = dist[n] / total * 100
        marker = ' ←要補完' if n < 10 else ''
        print(f'  {n:2d}頭: {dist[n]:4d}件 ({pct:5.1f}%) {bar}{marker}')
    print(f'\n補完が必要なレース（10頭未満）: {need}件 / {total}件 ({need/total*100:.1f}%)')
