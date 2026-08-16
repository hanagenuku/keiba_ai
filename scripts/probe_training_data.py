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

2026-08-13の1回目で確定したこと
--------------------------------
- robots.txt は `User-agent: * / Disallow:`（空）。禁止パスは1つも無い
- `thisweek/.../training.html` は**動画ページ**。table 0個・タイム文字列0件で
  調教タイムのテキストは存在しない → この経路は死んだ

2回目で調べること
-----------------
A. JRA公式内に調教タイムのテキストが本当に無いか、機械的に棚卸しする
   - sitemap.xml の有無
   - 実際の出馬表ページから全 access*.html エンドポイントを抽出
     （血統ページ accessU.html はまさにこの方法で見つかった）
   - サイト内リンクに「調教」を含むものがあるか
B. netkeiba の robots.txt と利用規約（**読むだけ**。取得はしない）
   スクレイピング禁止なら、そこで検討終了になる

使い方
------
    python scripts/probe_training_data.py                # 全部
    python scripts/probe_training_data.py 2025/1228_1    # training.htmlを直接
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
# ⚠ 出馬表ページは開催が終わると消える。日付をハードコードすると
#   「開催情報が見つかりません」で空振りする（2026-08-15を固定していて実際に空振りした）。
#   当日から数日ぶん試して、実在する開催日を自動で拾う。
PROBE_DAYS_BACK = 9      # 当日→過去9日ぶん試す（土日開催を必ず1つは踏む）


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


def _find_live_kaisai(sess):
    """実在する開催日と base を自動で探す。当日から遡って最初に見つかったもの。"""
    from datetime import datetime, timedelta, timezone
    from src.scraper.calendar import get_kaisai_on_date
    jst = timezone(timedelta(hours=9))
    today = datetime.now(jst)
    for d in range(PROBE_DAYS_BACK):
        ds = (today - timedelta(days=d)).strftime('%Y%m%d')
        links = get_kaisai_on_date(ds, sess)
        if links:
            return ds, list(links.keys())[0]
    return None, None


def probe_jra_endpoints(sess, date_str=None):
    """実際の出馬表ページから、リンクされている全エンドポイントを棚卸しする。

    血統ページ(accessU.html)は「馬名リンクのhrefにCNAMEが埋まっていた」ことで
    見つかった（2026-07-17②）。同じやり方で調教ページが在るかを確かめる。
    """
    print()
    print('=' * 70)
    print('④ 実際の出馬表ページのリンク棚卸し')
    print('=' * 70)
    from src.scraper.calendar import get_kaisai_on_date
    from src.scraper.jra_scraper import _try_fetch_shutuba, find_r01_shutuba

    # ⚠ get_kaisai_on_date は {base: 日付} を返す。**キーがbase**。
    #   2026-08-13の1回目は values() を取って base='20260815' になり、
    #   スキャンが空振りして「入口が無い」と誤読しかけた。
    date_str, base = _find_live_kaisai(sess)
    if base is None:
        print(f'  ❌ 直近{PROBE_DAYS_BACK}日で実在する開催日が見つからない')
        return
    print(f'  開催日 = {date_str}')
    print(f'  base = {base}')
    r01 = find_r01_shutuba(base, date_str, sess)
    if r01 is None:
        print('  ❌ R01のsuffixが見つからない')
        return
    _, soup = _try_fetch_shutuba(sess, base, 1, date_str, f'{r01:02X}')
    if soup is None:
        print('  ❌ 出馬表ページが取れない')
        return

    hrefs = [a.get('href', '') for a in soup.find_all('a', href=True)]
    eps = sorted({m for h in hrefs for m in re.findall(r'(access[A-Z]+\.html)', h)})
    print(f'  リンク総数: {len(hrefs)}')
    print(f'  access*.html エンドポイント: {eps}')

    forms = [f.get('action', '') for f in soup.find_all('form')]
    fe = sorted({m for f in forms for m in re.findall(r'(access[A-Z]+\.html)', f)})
    print(f'  <form action> のエンドポイント: {fe or "(なし)"}')

    known = {'accessD.html', 'accessS.html', 'accessO.html', 'accessU.html'}
    unknown = [e for e in set(eps) | set(fe) if e not in known]
    print(f'  🔎 未知のエンドポイント: {unknown or "(なし＝新しい入口は無い)"}')

    text = soup.get_text(' ', strip=True)
    print(f'  ページ本文の「調教」出現: {text.count("調教")}回 '
          f'（調教師名で出るので、これだけでは判断できない）')
    chokyo_links = [(a.get_text(strip=True)[:20], a["href"])
                    for a in soup.find_all('a', href=True)
                    if '調教' in a.get_text() and '調教師' not in a.get_text()]
    print(f'  「調教」を含むリンク（調教師を除く）: {chokyo_links or "(なし)"}')


def probe_jra_sitemap(sess):
    print()
    print('=' * 70)
    print('⑤ sitemap.xml と サイト内の調教ページ')
    print('=' * 70)
    for path in ['/sitemap.xml', '/keiba/']:
        body = _get(sess, f'{BASE}{path}')
        if body is None:
            print(f'  {path}: 取得できず')
            continue
        if path.endswith('.xml'):
            urls = re.findall(r'<loc>([^<]+)</loc>', body)
            print(f'  {path}: URL {len(urls)}件')
            hit = [u for u in urls if 'chokyo' in u or 'training' in u or '調教' in u]
            print(f'    調教らしきURL: {hit[:10] or "(なし)"}')
        else:
            soup = BeautifulSoup(body, 'lxml')
            hit = [(a.get_text(strip=True)[:24], a['href'])
                   for a in soup.find_all('a', href=True)
                   if ('調教' in a.get_text() and '調教師' not in a.get_text())
                   or 'training' in a['href'] or 'chokyo' in a['href']]
            print(f'  {path}: 調教らしきリンク {len(hit)}件')
            for t, h in hit[:12]:
                print(f'    {t:<26} {h}')


def probe_netkeiba_terms(sess):
    """netkeiba の robots.txt と利用規約を**読むだけ**。データ取得はしない。

    ⚠ 規約でスクレイピングが禁じられていれば、そこで検討終了。
    先に規約を読むのが順序（North Star: 決め打ちで実装しない）。
    """
    print()
    print('=' * 70)
    print('⑥ netkeiba の robots.txt と利用規約（読むだけ・データは取らない）')
    print('=' * 70)
    import requests
    ua = {'User-Agent': 'Mozilla/5.0'}

    try:
        r = requests.get('https://www.netkeiba.com/robots.txt', headers=ua, timeout=20)
        r.encoding = r.apparent_encoding
        print('  --- robots.txt 全文 ---')
        for line in r.text.splitlines()[:60]:
            print(f'  | {line}')
        print('  --- ここまで ---')
    except Exception as e:
        print(f'  ❌ robots.txt 取得失敗: {type(e).__name__}: {e}')
    time.sleep(SLEEP)

    # ⚠ ?pid=agreement は404ページだった（本文に「ページが見つかりません」）。
    #   トップのフッターから正しい規約URLが判明したので、それを最優先で読む。
    for url in ['https://www.netkeiba.com/info/kiyaku.html',
                'https://regist.netkeiba.com/account/?pid=agreement']:
        try:
            r = requests.get(url, headers=ua, timeout=20)
            r.encoding = r.apparent_encoding
            if r.status_code != 200:
                print(f'  ❌ HTTP {r.status_code} {url}')
                continue
            txt = BeautifulSoup(r.text, 'lxml').get_text(' ', strip=True)
            print(f'\n  --- {url} 本文長 {len(txt)}字 ---')
            # ⚠ 短い本文はエラーページやJS描画の可能性。中身を必ず目視する。
            #   キーワード0回を「禁止条項が無い」と読むのは誤り。
            print(f'    本文の先頭600字: {txt[:600]!r}')
            for kw in ['スクレイピング', 'クローラ', 'ロボット', '自動', '収集',
                       '複製', '転載', '二次利用', '禁止', 'プログラム']:
                n = txt.count(kw)
                print(f'    「{kw}」: {n}回')
            # 該当条項の前後を抜き出す
            for kw in ['スクレイピング', 'クローラ', '自動', '収集', '複製', '転載']:
                for m in re.finditer(kw, txt):
                    s0, e0 = max(0, m.start() - 90), min(len(txt), m.end() + 90)
                    print(f'    [{kw}] …{txt[s0:e0]}…')
                    break
        except Exception as e:
            print(f'  ❌ {url} 取得失敗: {type(e).__name__}: {e}')
        time.sleep(SLEEP)

    # 規約ページの正しいURLを、トップページのリンクから探す
    try:
        r = requests.get('https://www.netkeiba.com/', headers=ua, timeout=20)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, 'lxml')
        cand = [(a.get_text(strip=True)[:20], a['href'])
                for a in soup.find_all('a', href=True)
                if any(k in a.get_text() for k in ['規約', '約款', '免責', 'ご利用'])
                or any(k in a['href'] for k in ['agreement', 'terms', 'kiyaku', 'rule'])]
        print(f'\n  トップページの規約らしきリンク {len(cand)}件:')
        for t, h in cand[:15]:
            print(f'    {t:<22} {h}')
    except Exception as e:
        print(f'  ❌ トップページ取得失敗: {type(e).__name__}: {e}')


def probe_koukai_chokyo(sess):
    """/keiba/ にあった唯一の「調教」リンク /event/index5.html の正体を確認する。

    名前から「公開調教（見学イベント）の告知」と推測できるが、推測で切り捨てない。
    """
    print()
    print('=' * 70)
    print('⑦ /keiba/ にあった「公開調教」リンクの正体')
    print('=' * 70)
    body = _get(sess, f'{BASE}/event/index5.html')
    if body is None:
        print('  取得できず')
        return
    soup = BeautifulSoup(body, 'lxml')
    t = soup.find('title')
    txt = soup.get_text(' ', strip=True)
    print(f'  <title>: {t.get_text(strip=True) if t else "(なし)"}')
    print(f'  <table>: {len(soup.find_all("table"))}個')
    times = re.findall(r'\d{1,2}[-\.]\d{1,2}(?:[-\.]\d{1,2})+', txt)
    print(f'  タイムらしき文字列: {len(times)}件')
    for kw in ['坂路', 'ウッド', '併走', '一杯', '馬なり', '見学', 'イベント', '入場']:
        print(f'    「{kw}」: {txt.count(kw)}回')
    print(f'  本文の先頭300字: {txt[:300]}')


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

    probe_jra_sitemap(sess)
    probe_koukai_chokyo(sess)
    probe_jra_endpoints(sess)
    probe_netkeiba_terms(sess)


if __name__ == '__main__':
    main()
