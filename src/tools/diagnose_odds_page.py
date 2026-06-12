"""
オッズページ(accessO.html)構造調査スクリプト。

fetch_odds_for_race / find_r01_odds は pw151ouS3 系CNAMEで実機検証済み。
このスクリプトは実際のJRAページのCNAME・HTML構造を調査するための
診断専用ツール。Colab（JRAアクセス可能な環境）で実行する。

【実行方法（Colab）】
    import sys; sys.path.insert(0, BASE_DIR)
    from src.tools.diagnose_odds_page import diagnose_odds_for_race
    # races は fetch_races_on_date の戻り値（_odds_cn を持つ）
    diagnose_odds_for_race(sess, races[0])
"""
import re
import unicodedata
from bs4 import BeautifulSoup

from src.utils.config import JRA_BASE, HEADERS


def diagnose_odds_for_race(sess, race, max_suffix_scan=40):
    """1レース分のオッズページ取得を試行し、結果を詳細に表示する。

    Args:
        sess : requests.Session
        race : fetch_races_on_date が返すレース辞書（_odds_cn 必須）
        max_suffix_scan: 推測CNAME prefixでの総当たり探索の最大件数

    手順:
        1. _odds_cn['base'] の pw01dde01 を様々な候補prefixに置換して
           同じ suffix で accessO.html を試行
        2. すべて失敗した場合、各候補prefixで suffix 0x00-0x{max} を
           総当たりして 'パラメータエラー' でないレスポンスを探す
        3. 成功したレスポンスは <table> の構造（行・セル内容）をそのまま表示
    """
    cn_info = race.get('_odds_cn')
    if not cn_info:
        print('❌ このレースには _odds_cn がありません')
        return

    base, date_str, sx, race_num = (
        cn_info['base'], cn_info['date_str'], cn_info['sx'], cn_info['race_num']
    )
    print(f'━━━ {race.get("racecourse","?")} R{race_num} ({date_str}) ━━━')
    print(f'出馬表 base: {base}')
    print(f'出馬表 suffix: {sx}')

    # 候補prefix（JRA CNAME規則の推測パターン）
    m = re.search(r'^(.*?)(\d{2})(\d{8}/[0-9A-F]{2})$', f'{base}{race_num:02d}{date_str}/{sx}')
    candidates = []
    if 'pw01dde01' in base:
        for repl in ['pw01oxw1', 'pw15oxw1', 'pw01odt01', 'pw01tyo01', 'pw01ovz1']:
            candidates.append(base.replace('pw01dde01', repl))

    print(f'\n[1] 同一suffixでの候補prefix試行 ({len(candidates)}件)')
    found = False
    for odds_base in candidates:
        cn = f'{odds_base}{race_num:02d}{date_str}/{sx}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessO.html',
                          data={'cname': cn, 'CNAME': cn}, headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception as e:
            print(f'  {odds_base}: 例外 {e}')
            continue

        is_err = 'パラメータエラー' in r.text
        tables = BeautifulSoup(r.text, 'lxml').find_all('table') if not is_err else []
        print(f'  {odds_base}: {"パラメータエラー" if is_err else f"テーブル{len(tables)}個"}')
        if not is_err and tables:
            found = True
            _dump_tables(r.text, odds_base, cn)
            break

    if found:
        return

    print(f'\n[2] suffix総当たり探索 (0x00-{max_suffix_scan-1:02X}) × 候補prefix')
    for odds_base in candidates:
        for s in range(max_suffix_scan):
            cn = f'{odds_base}{race_num:02d}{date_str}/{s:02X}'
            try:
                r = sess.post(f'{JRA_BASE}/JRADB/accessO.html',
                              data={'cname': cn, 'CNAME': cn}, headers=HEADERS, timeout=10)
                r.encoding = 'shift_jis'
            except Exception:
                continue
            if 'パラメータエラー' not in r.text:
                tables = BeautifulSoup(r.text, 'lxml').find_all('table')
                if tables:
                    print(f'  ✅ 発見: {odds_base} suffix={s:02X}')
                    _dump_tables(r.text, odds_base, cn)
                    return

    print('\n❌ すべての候補で失敗。CNAME prefix規則が異なる可能性があります。')
    print('   ヒント: ブレラウザでJRAサイトの「オッズ」タブを開き、')
    print('   開発者ツール(Network)で accessO.html へのPOSTリクエストの')
    print('   cname/CNAME パラメータを確認してください。')


def _dump_tables(html_text, odds_base, cn, max_rows=25):
    """取得したHTMLのテーブル構造を表示（パーサー調整用）。"""
    print(f'\n--- 取得成功: {odds_base} ---')
    print(f'CNAME: {cn}')
    soup = BeautifulSoup(html_text, 'lxml')
    tables = soup.find_all('table')
    print(f'テーブル数: {len(tables)}')
    for ti, table in enumerate(tables):
        rows = table.find_all('tr')
        print(f'\n[テーブル{ti}] 行数={len(rows)}')
        for ri, tr in enumerate(rows[:max_rows]):
            cells = [unicodedata.normalize('NFKC', c.get_text(strip=True))
                     for c in tr.find_all(['td', 'th'])]
            print(f'  行{ri}: {cells}')
