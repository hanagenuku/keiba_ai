#!/usr/bin/env python3
"""過去日のレース結果ページへの到達経路を実地で切り分ける調査スクリプト。

## なぜ要るか

2026-08-02 の日曜結果取得で新潟・札幌の R01 suffix 探索が全滅し、24レースを
落とした。R02〜R12 の suffix は `calc_suffix()` が R01 から導出するため、
**R01 が見つからないとその会場が丸ごと落ちる**（単一障害点）。

一方、JRA の結果ページは GET でも開け、2023〜2024年のものが現在も生きている:
    https://www.jra.go.jp/JRADB/accessS.html?CNAME=pw01sde1010202401030420240120/D9
CNAME の構造は本スクレイパーが組み立てているものと同一。つまり
**データは消えておらず、取り方の問題**である可能性が高い。

本番コードは POST のみを使っている。GET や sp.jra.jp が使えるなら、
取りこぼしの回収手段になる。推測で実装を変える前にここで実測する。

## 使い方（GitHub Actions の probe.yml から実行。ローカルからは JRA に届かない）

    TARGET_DATE=20260802 PLACES=04,01 python scripts/probe_result_access.py

出力するもの:
  - 各会場の base（回次・日次）を、本番と同じ経路で導出できるか
  - R01 を POST / GET / sp.jra.jp の3経路でスキャンした結果
  - 失敗の内訳（パラメータエラー / テーブルなし / 内容なし / 例外）

⚠ 読み取り専用。DB もモデルも一切書き換えない。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bs4 import BeautifulSoup

from scripts._session import create_session
from src.scraper.jra_scraper import HEADERS, JRA_BASE, calc_suffix
from src.scraper.calendar import get_kaisai_on_date, get_base_from_calendar

PLACE_NAMES = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
               '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}

RESULT_KEYWORDS = ('騎手', '馬名', '着順', '調教師')


def _classify(text):
    """応答を分類する。'hit' なら結果ページ。"""
    if 'パラメータエラー' in text:
        return 'param_error'
    soup = BeautifulSoup(text, 'lxml')
    if not soup.find_all('table'):
        return 'no_table'
    if not any(k in text for k in RESULT_KEYWORDS):
        return 'no_content'
    return 'hit'


def _post(sess, cname):
    r = sess.post(f'{JRA_BASE}/JRADB/accessS.html',
                  data={'cname': cname, 'CNAME': cname},
                  headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    return r.text


def _get(sess, cname):
    r = sess.get(f'{JRA_BASE}/JRADB/accessS.html',
                 params={'CNAME': cname}, headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    return r.text


def _get_sp(sess, cname):
    r = sess.get('https://sp.jra.jp/JRADB/accessS.html',
                 params={'CNAME': cname}, headers=HEADERS, timeout=15)
    r.encoding = 'shift_jis'
    return r.text


METHODS = [('POST (本番と同じ)', _post), ('GET', _get), ('GET sp.jra.jp', _get_sp)]


def scan(sess, fetch, base, date, limit=256):
    """R01 の suffix を1周探す。(見つかったsuffix, 内訳) を返す。"""
    diag = {'param_error': 0, 'no_table': 0, 'no_content': 0, 'exc': 0,
            'last_exc': None, 'sample': None}
    for s in range(limit):
        cname = f'{base}01{date}/{s:02X}'
        try:
            text = fetch(sess, cname)
        except Exception as e:
            diag['exc'] += 1
            diag['last_exc'] = f'{type(e).__name__}: {e}'
            time.sleep(0.02)
            continue
        kind = _classify(text)
        if kind == 'hit':
            return f'{s:02X}', diag
        diag[kind] += 1
        if kind != 'param_error' and diag['sample'] is None:
            diag['sample'] = text[:160].replace('\n', ' ')
        if kind != 'param_error':
            time.sleep(0.02)
    return None, diag


def main():
    date = (os.environ.get('TARGET_DATE') or '').strip()
    if not (len(date) == 8 and date.isdigit()):
        raise SystemExit('TARGET_DATE=YYYYMMDD を指定してください')
    places = [p.strip() for p in
              (os.environ.get('PLACES') or '').split(',') if p.strip()]

    sess = create_session()

    print(f'🔎 対象日: {date}')
    print('─' * 62)

    # ① 本番と同じ経路で base（回次・日次）を導出できるか
    print('① base の導出（本番は出馬表一覧ページから逆算している）')
    try:
        kaisai = get_kaisai_on_date(date, sess)
        print(f'   get_kaisai_on_date → {kaisai if kaisai else "（空）"}')
    except Exception as e:
        kaisai = {}
        print(f'   ⚠ get_kaisai_on_date が例外: {type(e).__name__}: {e}')

    bases = {}
    for pc in (places or sorted(PLACE_NAMES)):
        cal_base = get_base_from_calendar(pc, date)
        if cal_base:
            # 出馬表用(pw01dde01…)を結果用(pw01sde10…)に読み替える
            bases[pc] = 'pw01sde10' + cal_base[len('pw01dde01'):]
        print(f'   {PLACE_NAMES.get(pc, pc)}({pc}): カレンダー由来 base = '
              f'{bases.get(pc) or "取得できず"}')
    print()

    # ② 経路ごとに R01 を探す
    print('② R01 suffix 探索（経路ごと・見つかった時点で次の会場へ）')
    for pc, base in bases.items():
        name = PLACE_NAMES.get(pc, pc)
        print(f'\n  ── {name}({pc})  base={base} ──')
        found = False
        for label, fetch in METHODS:
            t0 = time.time()
            sx, diag = scan(sess, fetch, base, date)
            dt = time.time() - t0
            if sx:
                url = f'{JRA_BASE}/JRADB/accessS.html?CNAME={base}01{date}/{sx}'
                print(f'    ✅ {label}: suffix={sx}  ({dt:.0f}秒)')
                print(f'       R01 {url}')
                print(f'       R12 suffix（導出）= {calc_suffix(int(sx, 16), 12)}')
                found = True
                break
            print(f'    ❌ {label}: 未発見 ({dt:.0f}秒) '
                  f'param={diag["param_error"]} no_table={diag["no_table"]} '
                  f'no_content={diag["no_content"]} exc={diag["exc"]}')
            if diag['last_exc']:
                print(f'       直近の例外: {diag["last_exc"]}')
            if diag['sample']:
                print(f'       非パラメータエラー応答の例: {diag["sample"]!r}')
        if not found:
            print(f'    → {name} はどの経路でも取得できなかった')

    print('\n' + '─' * 62)
    print('※ 読み取り専用。DB・モデルには一切書き込んでいない。')


if __name__ == '__main__':
    main()
