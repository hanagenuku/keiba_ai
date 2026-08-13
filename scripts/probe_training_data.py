"""調教（追い切り）データが JRA公式から無料で取れるかを実データで確認する調査用スクリプト。

背景
----
2026-08-03のサマリで「次の一手」として合意した2案のうちの1つが
**「市場が見ていない情報を入れる（調教・厩舎談話・パドック）」**。
ただしユーザーは時間を割けないため、自動取得できるものに限る。

このスクリプトは**何も実装しない**。実際のページ構造を1回だけログに出す。
決め打ちで parser を書かないための材料集め（2026-08-03③で結果ページの
列構成を実データで確定させたのと同じやり方）。

⚠ 読み取り専用。DB・モデル・latest.json には一切書き込まない。

調べること
----------
1. robots.txt が www.jra.go.jp のスクレイピングをどう扱っているか
   （docs/history_db_schema.md に「要ユーザー確認」として長く残っていた宿題）
2. /keiba/thisweek/ の構造（今週の注目レースの一覧）
3. training.html が実在するか・何が載っているか
   - 出走全頭か、一部か
   - タイムはHTMLのtableか画像か
   - 出典表記（「競馬ブック提供」等）
4. 過去の開催分に遡れるか（学習データが作れるかの分かれ目）

使い方
------
    python scripts/probe_training_data.py                # 今週分を調べる
    python scripts/probe_training_data.py 2025/1228_1    # 特定ページを直接
"""
import re
import sys
import time

sys.path.insert(0, '.')

from bs4 import BeautifulSoup                            # noqa: E402

from scripts._session import create_session              # noqa: E402

BASE = 'https://www.jra.go.jp'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
SLEEP = 1.0          # JRA側に負荷をかけない
MAX_PAGES = 8        # ⚠ North Star #4: 新規リクエスト元には必ず上限を設ける


def _get(sess, url):
    """1ページ取得。JRA公式は Shift_JIS 配信なので明示指定する。"""
    try:
        r = sess.get(url, headers=HEADERS, timeout=20)
    except Exception as e:
        print(f'  ❌ 通信失敗 {url} : {type(e).__name__}: {e}')
        return None
    if r.status_code != 200:
        print(f'  ❌ HTTP {r.status_code} {url}')
        return None
    r.encoding = 'shift_jis'
    time.sleep(SLEEP)
    return r.text


def probe_robots(sess):
    print('=' * 70)
    print('① robots.txt（長く「要ユーザー確認」だった宿題）')
    print('=' * 70)
    txt = _get(sess, f'{BASE}/robots.txt')
    if txt is None:
        print('  取得できず')
        return
    print(f'  --- {BASE}/robots.txt 全文 ---')
    for line in txt.splitlines():
        print(f'  | {line}')
    print('  --- ここまで ---')
    # 我々が実際に叩いているパスが Disallow に該当するか機械的に見る
    dis = re.findall(r'(?im)^\s*Disallow:\s*(\S+)', txt)
    ours = ['/JRADB/accessD.html', '/JRADB/accessS.html', '/JRADB/accessO.html',
            '/JRADB/accessU.html', '/keiba/thisweek/']
    print('\n  我々が叩くパスの判定:')
    for p in ours:
        hit = [d for d in dis if d != '/' and p.startswith(d)]
        blanket = [d for d in dis if d == '/']
        mark = '🔴 Disallow該当' if (hit or blanket) else '✅ 明示的な禁止なし'
        print(f'    {p:<28} {mark}' + (f'  ({hit or blanket})' if (hit or blanket) else ''))


def probe_thisweek_index(sess):
    print()
    print('=' * 70)
    print('② /keiba/thisweek/ の構造（注目レースの一覧）')
    print('=' * 70)
    html = _get(sess, f'{BASE}/keiba/thisweek/')
    if html is None:
        return []
    soup = BeautifulSoup(html, 'lxml')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'thisweek' in href:
            links.append((a.get_text(strip=True)[:24], href))
    print(f'  thisweek 配下へのリンク {len(links)}件（先頭20件）:')
    for t, h in links[:20]:
        print(f'    {t:<26} {h}')
    # training.html を持つレースディレクトリを推定
    dirs = sorted({re.sub(r'[^/]*$', '', h) for _, h in links if re.search(r'/\d{4}/\d{4}_\d+/', h)})
    print(f'\n  レースディレクトリ候補 {len(dirs)}件:')
    for d in dirs[:10]:
        print(f'    {d}')
    return dirs


def probe_training_page(sess, path):
    print()
    print('=' * 70)
    print(f'③ training.html の中身: {path}')
    print('=' * 70)
    url = path if path.startswith('http') else f'{BASE}{path}'
    html = _get(sess, url)
    if html is None:
        print('  → このページは存在しない（または取得不可）')
        return
    soup = BeautifulSoup(html, 'lxml')

    title = soup.find('title')
    print(f'  <title>: {title.get_text(strip=True) if title else "(なし)"}')

    tables = soup.find_all('table')
    print(f'  <table> の数: {len(tables)}')
    for i, tb in enumerate(tables[:4]):
        cap = tb.find('caption')
        rows = tb.find_all('tr')
        print(f'\n  --- table[{i}] caption={cap.get_text(strip=True) if cap else "(なし)"} '
              f'行数={len(rows)} ---')
        for tr in rows[:6]:
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
            print(f'    {cells}')

    # タイムらしき文字列が本文にあるか（画像だけなら特徴量化できない）
    text = soup.get_text(' ', strip=True)
    times = re.findall(r'\d{1,2}[-\.]\d{1,2}(?:[-\.]\d{1,2})+', text)
    print(f'\n  タイムらしき文字列: {len(times)}件  例: {times[:8]}')
    for kw in ['坂路', 'ウッド', 'ポリトラック', '併走', '一杯', '強め', '馬なり',
               '競馬ブック', '提供', '自動計測']:
        print(f'    「{kw}」の出現: {text.count(kw)}回')

    imgs = soup.find_all('img')
    print(f'  <img> の数: {len(imgs)}（タイムが画像なら特徴量化できない）')

    print(f'\n  本文の先頭400字:\n  {text[:400]}')


def main():
    sess = create_session()
    probe_robots(sess)

    if len(sys.argv) > 1:
        probe_training_page(sess, f'/keiba/thisweek/{sys.argv[1]}/training.html')
        return

    dirs = probe_thisweek_index(sess)
    if not dirs:
        print('\n  ⚠ thisweek のリンクが拾えなかった。'
              '既知の実在URLで直接確認する。')
        dirs = ['/keiba/thisweek/2026/0118_2/']
    for d in dirs[:MAX_PAGES]:
        probe_training_page(sess, d.rstrip('/') + '/training.html')

    print()
    print('=' * 70)
    print('判断材料まとめ')
    print('=' * 70)
    print('  ・training.html が「出走全頭」なら特徴量化の価値あり')
    print('  ・「重賞の一部の馬だけ」ならカバレッジが薄すぎて学習に使えない')
    print('  ・タイムが画像なら自動取得は不可能')
    print('  ・出典が外部（競馬ブック等）なら二次利用の可否を要確認')


if __name__ == '__main__':
    main()
