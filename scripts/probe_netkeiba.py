#!/usr/bin/env python3
"""netkeiba に「追い切り（調教）タイム」が機械で読める形で存在するかを実データで確認する。

⚠ 完全な読み取り専用。DB・モデル・latest.json・リポジトリに一切書き込まない。
   結果はジョブログにのみ出力する。

なぜプローブから始めるか
------------------------
2026-08-16 のJRA公式調査の最大の教訓は「**空振りと『探した結果ゼロ』は別物**」。
あのときは `training.html` が動画ページだと確定するまで実HTMLを4回確認している。
パイプラインを書く前に、まず「タイムがテキストとして存在するか」だけを見る。

規約・robots.txt について
------------------------
netkeiba の規約（2026-08-16に本文10,597字を実際に読んで記録）は取得行為自体を
名指しで禁じていないが、入手データの用途を**私的利用に限定**している。
ユーザーの明示的な判断（2026-08-24「グレーのまま進める」）に基づき調査する。
robots.txt は毎回取得し、**Disallow に該当したらそこで中止**する。
"""
import argparse
import datetime as dt
import re
import sys
import time
import urllib.robotparser as rp

import requests
from bs4 import BeautifulSoup

UA = 'keiba_ai-research/1.0 (personal use; contact via github.com/hanagenuku/keiba_ai)'
SLEEP = 2.0                      # 1リクエストあたりの間隔（秒）
MAX_REQUESTS = 10                # 総リクエスト数の上限（North Star #4）
TIME_RE = re.compile(r'\b\d{1,2}\.\d\b')          # 追い切りタイム（例 52.3 / 13.1）
_count = 0


def _get(sess, url, label):
    """1リクエスト。上限を超えたら None を返して打ち切る。"""
    global _count
    if _count >= MAX_REQUESTS:
        print(f'  ⚠ リクエスト上限 {MAX_REQUESTS} に到達。{label} をスキップ')
        return None
    _count += 1
    try:
        r = sess.get(url, timeout=20)
    except Exception as e:
        print(f'  ❌ {label}: 通信失敗 {type(e).__name__}: {e}')
        return None
    print(f'  [{_count}/{MAX_REQUESTS}] {label}: HTTP {r.status_code}  {len(r.content):,} bytes')
    time.sleep(SLEEP)
    if r.status_code != 200:
        return None
    r.encoding = r.apparent_encoding or 'euc-jp'
    return r.text


def check_robots(sess, base):
    """robots.txt を取得して中身を出す。取得できなければ「不在」として続行。"""
    print(f'\n■ robots.txt ({base})')
    txt = _get(sess, f'{base}/robots.txt', 'robots.txt')
    if txt is None:
        print('  → 取得できず（404 or 到達不可）。許可も禁止も明示されていない')
        return None
    body = txt.strip()
    print('  --- 本文 ---')
    for line in body.splitlines()[:40]:
        print(f'    {line}')
    if len(body.splitlines()) > 40:
        print(f'    …（残り {len(body.splitlines())-40} 行）')
    p = rp.RobotFileParser()
    p.parse(body.splitlines())
    return p


def probe_oikiri(sess, race_id):
    """追い切りページに「テキストとしてのタイム」があるかを見る。"""
    url = f'https://race.netkeiba.com/race/oikiri.html?race_id={race_id}'
    print(f'\n■ 追い切りページ  race_id={race_id}')
    html = _get(sess, url, 'oikiri.html')
    if html is None:
        return
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(' ', strip=True)
    tables = soup.find_all('table')
    times = TIME_RE.findall(text)
    print(f'  <title>       : {(soup.title.get_text(strip=True) if soup.title else "?")[:70]}')
    print(f'  table         : {len(tables)}個')
    print(f'  img           : {len(soup.find_all("img"))}個')
    print(f'  本文の長さ      : {len(text):,}字')
    print(f'  タイムらしき数値 : {len(times)}件  例: {times[:8]}')
    for kw in ['坂路', 'ウッド', '併走', '馬なり', '一杯', 'CW', 'DW', '追切', '評価']:
        print(f'    「{kw}」        : {text.count(kw)}回')
    # 🔑 JRA公式の training.html は table 0個・img 48個の「動画ページ」だった。
    #    同じ形なら netkeiba も同じ結論になる。
    if tables and len(times) >= 20:
        print('  ✅ テキストとしてタイムが存在する可能性が高い（要・列構造の確認）')
        for i, t in enumerate(tables[:3]):
            rows = t.find_all('tr')
            print(f'    table[{i}]: {len(rows)}行')
            for r in rows[:3]:
                cells = [c.get_text(strip=True) for c in r.find_all(['th', 'td'])]
                if cells:
                    print(f'      {cells[:12]}')
    else:
        print(f'  ⚠ 初期HTMLにタイムが無い（HTML {len(html):,}B に対し本文 {len(text):,}字）')
        # 🔑 本文が短いときは全文を出す。「無料では見出しだけ」なのか
        #    「JSで後から入る」のかは、実際の文面を読まないと区別できない。
        #    有料会員限定なら、購読せずに取りに行くのは規約の曖昧さの話ではなく
        #    課金の回避になる。ユーザーの判断の範囲を超えるので必ず確認する。
        print('  --- 表示されている本文（全文） ---')
        for line in [text[i:i + 110] for i in range(0, min(len(text), 3300), 110)]:
            print(f'    {line}')
        for kw in ['会員', '有料', 'プレミアム', 'ログイン', '登録', '無料', 'ウマい馬券']:
            c = text.count(kw)
            if c:
                print(f'    🔎 「{kw}」が本文に {c}回')
        if len(html) > len(text) * 5:
            print('    → JSで描画される作りとみられる。'
                  'race_list.html が race_list_sub.html で解決したのと同じ形。')
            _hunt_data_url(sess, html, race_id)
        else:
            print('    → JRA公式 training.html と同じ「そもそも無い」型の可能性')


def _hunt_data_url(sess, shell_html, race_id):
    """シェルHTMLから実データの取得先を**抽出**して追う（推測で当てない）。"""
    cands = []
    for pat in (r'["\'](/[^"\'\s]*?_sub\.html[^"\'\s]*)["\']',
                r'["\'](https?://[^"\'\s]*?_sub\.html[^"\'\s]*)["\']',
                r'url\s*:\s*["\']([^"\'\s]+)["\']'):
        cands += re.findall(pat, shell_html)
    cands = [c for c in dict.fromkeys(cands) if 'sub' in c or 'ajax' in c.lower()]
    print(f'    シェルHTMLから抽出した取得先候補: {len(cands)}件')
    for c in cands[:6]:
        print(f'      {c}')
    # 抽出できたものを優先。無ければ race_list で実証済みの命名規則を1つだけ試す。
    urls = []
    for c in cands[:3]:
        u = c if c.startswith('http') else f'https://race.netkeiba.com{c}'
        if 'race_id' not in u:
            u += ('&' if '?' in u else '?') + f'race_id={race_id}'
        urls.append(u)
    urls.append(f'https://race.netkeiba.com/race/oikiri_sub.html?race_id={race_id}')
    for u in dict.fromkeys(urls):
        html = _get(sess, u, u.split('/')[-1][:40])
        if html is None:
            continue
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(' ', strip=True)
        times = TIME_RE.findall(text)
        print(f'      → table {len(soup.find_all("table"))}個 / 本文 {len(text):,}字 / '
              f'タイム {len(times)}件  例: {times[:8]}')
        if len(times) >= 20:
            print('      ✅ ここに追い切りタイムがある')
            for t in soup.find_all('table')[:1]:
                for r in t.find_all('tr')[:4]:
                    cells = [c.get_text(strip=True) for c in r.find_all(['th', 'td'])]
                    if cells:
                        print(f'        {cells[:12]}')
            return u
    print('    ❌ 抽出した候補のどれにもタイムが無かった')
    return None


def _dump(html, label, n=1200):
    """中身を実際に目で見る。

    🔑 2026-08-16の教訓: キーワードが0件のときは、まず「本当にその文書を
       読めているか」を疑う（`?pid=agreement` が878字・キーワード全0で
       返ってきて、正体は404ページだった）。「0件」を「該当なし」と読まない。
    """
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(' ', strip=True)
    print(f'  --- {label} の中身 ---')
    print(f'    <title>  : {(soup.title.get_text(strip=True) if soup.title else "?")[:70]}')
    print(f'    本文の長さ : {len(text):,}字   script: {len(soup.find_all("script"))}個   '
          f'table: {len(soup.find_all("table"))}個   a: {len(soup.find_all("a"))}個')
    print(f'    本文の冒頭 : {text[:200]}')
    # JSで描画される作りかどうかの手掛かり
    for kw in ['race_id', 'kaisai', 'RaceList', 'ajax', 'json']:
        print(f'      「{kw}」 : {html.count(kw)}回')
    print(f'    生HTMLの冒頭 {n}字:')
    print('      ' + html[:n].replace('\n', ' ')[:n])


def find_race_id(sess, date_str):
    """開催日のレース一覧から race_id を1つ拾う。

    netkeiba の `race_list.html` はJSで描画される作りなので、実データを持つ
    フラグメント `race_list_sub.html` も試す。
    """
    print(f'\n■ レース一覧から race_id を探す  {date_str}')
    for name, url in [
        ('race_list_sub.html',
         f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}'),
        ('race_list.html',
         f'https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}'),
        ('db.netkeiba race_list',
         f'https://db.netkeiba.com/race/list/{date_str}/'),
    ]:
        html = _get(sess, url, name)
        if html is None:
            continue
        ids = re.findall(r'race_id=(\d{12})', html)
        ids += re.findall(r'/race/(\d{12})', html)      # db.netkeiba 形式
        ids = sorted(set(ids))
        print(f'  {name}: race_id {len(ids)}件  例: {ids[:3]}')
        if ids:
            return ids[0]
        _dump(html, name)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--race-id', default='', help='直接見たい race_id（12桁）')
    ap.add_argument('--date', default='', help='race_id を探す開催日 YYYYMMDD')
    a = ap.parse_args()

    print('=' * 66)
    print('netkeiba 追い切りデータ プローブ（読み取り専用・保存なし）')
    print(f'  User-Agent   : {UA}')
    print(f'  リクエスト上限 : {MAX_REQUESTS}件 / 間隔 {SLEEP}秒')
    print('=' * 66)

    sess = requests.Session()
    sess.headers.update({'User-Agent': UA})

    robots = check_robots(sess, 'https://race.netkeiba.com')
    target = 'https://race.netkeiba.com/race/oikiri.html'
    if robots is not None and not robots.can_fetch(UA, target):
        print(f'\n🛑 robots.txt が {target} を Disallow している。ここで中止する。')
        return 0

    rid = a.race_id.strip()
    if not rid:
        d = a.date.strip() or dt.date.today().strftime('%Y%m%d')
        rid = find_race_id(sess, d)
    if not rid:
        print('\n⚠ race_id が得られなかった（開催日でない可能性）。')
        print('   --race-id か --date を指定して再実行すること。')
        print('   ⚠ 「空振り」であって「該当なし」ではない（2026-08-16の教訓）。')
        return 0

    probe_oikiri(sess, rid)
    print(f'\n総リクエスト数: {_count}件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
