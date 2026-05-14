import re
from bs4 import BeautifulSoup
from src.utils.config import JRA_BASE, PLACE_NAMES, KAISAI_CALENDAR as _DEFAULT_CALENDAR


def get_base_from_calendar(place_code, date_str, calendar=None):
    cal = calendar if calendar is not None else _DEFAULT_CALENDAR
    for entry in cal.get(place_code, []):
        if date_str in entry['days']:
            nichi = entry['days'].index(date_str) + 1
            return f"pw01dde01{place_code}{date_str[:4]}{entry['kai']}{nichi:02d}"
    return None


def get_kaisai_on_date(date_str, sess, calendar=None):
    """指定日の開催情報を取得（カレンダー＋thisweek補完）"""
    cal = calendar if calendar is not None else _DEFAULT_CALENDAR
    links = {}
    for pc in cal:
        base = get_base_from_calendar(pc, date_str, cal)
        if base:
            links[base] = date_str
            print(f"  📅 {PLACE_NAMES.get(pc, '?')} → {base}")

    thisweek_pcs = set()
    try:
        resp = sess.get(f'{JRA_BASE}/keiba/thisweek/', timeout=15)
        resp.encoding = 'shift_jis'
        for a in BeautifulSoup(resp.text, 'lxml').find_all('a', href=True):
            href = a['href']
            if 'pw01dde01' not in href:
                continue
            m = re.search(r'pw01dde01(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})', href)
            if not m:
                continue
            pc_tw = m.group(1)
            date_tw = m.group(5)
            if date_tw != date_str:
                continue
            thisweek_pcs.add(pc_tw)
            base_tw = f'pw01dde01{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}'
            if base_tw not in links:
                links[base_tw] = date_str
                print(f"  🌐 thisweek補完: {PLACE_NAMES.get(pc_tw, '?')} → {base_tw}")
        if thisweek_pcs:
            links = {b: d for b, d in links.items()
                     if re.search(r'pw01dde01(\d{2})', b) and
                     re.search(r'pw01dde01(\d{2})', b).group(1) in thisweek_pcs}
            for pc in cal:
                if pc not in thisweek_pcs:
                    base_cal = get_base_from_calendar(pc, date_str, cal)
                    if base_cal:
                        print(f"  ⏭ {PLACE_NAMES.get(pc, '?')} カレンダーにあるがthisweekになし → スキップ")
    except Exception as e:
        print(f"  ⚠ thisweek取得失敗: {e}")

    if not links:
        try:
            print(f"  ⚠ {date_str}はカレンダー・thisweek両方になし → 結果一覧から取得")
            resp2 = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                              data={'CNAME': 'pw01sli00/AF'}, timeout=15)
            resp2.encoding = 'shift_jis'
            for tag in BeautifulSoup(resp2.text, 'lxml').find_all(onclick=True):
                oc = tag['onclick']
                m = re.search(r'pw01srl10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/(\w{2})', oc)
                if m and m.group(5) == date_str:
                    pc_r = m.group(1)
                    base_r = f'pw01dde01{pc_r}{m.group(2)}{m.group(3)}{m.group(4)}'
                    if base_r not in links:
                        links[base_r] = date_str
                        print(f"  📋 結果一覧補完: {PLACE_NAMES.get(pc_r, '?')} → {base_r}")
        except Exception as e:
            print(f"  ⚠ 結果一覧取得失敗: {e}")

    if not links:
        print(f"  ❌ {date_str}の開催情報が見つかりません")
    return links
