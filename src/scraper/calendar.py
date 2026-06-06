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
    """指定日の開催情報を取得（出走表一覧ページから正確なkai/nichiを取得）"""
    links = {}

    # ① 出走表一覧(pw01dli00)から pw01drl00 形式でkai/nichiを取得（最も信頼性が高い）
    try:
        r = sess.post(f'{JRA_BASE}/JRADB/accessD.html',
                      data={'cname': 'pw01dli00/F3', 'CNAME': 'pw01dli00/F3'}, timeout=15)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'lxml')
        for tag in soup.find_all(onclick=True):
            oc = tag.get('onclick', '')
            m = re.search(r'pw01drl00(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})', oc)
            if not m:
                continue
            pc, year, kai, nichi, date = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            if date != date_str:
                continue
            base = f'pw01dde01{pc}{year}{kai}{nichi}'
            if base not in links:
                links[base] = date_str
                print(f"  📅 {PLACE_NAMES.get(pc, '?')} → {base}")
    except Exception as e:
        print(f"  ⚠ 出走表一覧取得失敗: {e}")

    # ② thisweek補完（dli00で取得できなかった会場を補完）
    if not links:
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
                if m.group(5) != date_str:
                    continue
                base_tw = f'pw01dde01{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}'
                if base_tw not in links:
                    links[base_tw] = date_str
                    print(f"  🌐 thisweek補完: {PLACE_NAMES.get(m.group(1), '?')} → {base_tw}")
        except Exception as e:
            print(f"  ⚠ thisweek取得失敗: {e}")

    if not links:
        print(f"  ❌ {date_str}の開催情報が見つかりません")
    return links


def get_kaisai_on_date_result(date_str, sess):
    """指定日の結果取得用 base を取得（結果一覧ページから pw01sde10 形式で返す）"""
    links = {}

    # 結果一覧(pw01sli00)から pw01srl10 の onclick を解析して正確な kai/nichi を取得
    try:
        r = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                      data={'CNAME': 'pw01sli00/AF'}, timeout=15)
        r.encoding = 'shift_jis'
        soup = BeautifulSoup(r.text, 'lxml')
        for tag in soup.find_all(onclick=True):
            oc = tag.get('onclick', '')
            m = re.search(r'pw01srl10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/(\w{2})', oc)
            if not m:
                continue
            pc, year, kai, nichi, date = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            if date != date_str:
                continue
            base = f'pw01sde10{pc}{year}{kai}{nichi}'
            if base not in links:
                links[base] = date_str
                print(f"  📋 {PLACE_NAMES.get(pc, '?')} → {base}")
    except Exception as e:
        print(f"  ⚠ 結果一覧取得失敗: {e}")

    if not links:
        print(f"  ❌ {date_str}の結果開催情報が見つかりません")
    return links
