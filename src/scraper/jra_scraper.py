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


def _jradb_post(sess, page, cname, timeout=15):
    """JRADB(accessX.html)への共通POSTラッパー。

    出馬表・結果・血統ページで共通の「cname/CNAME 両キー送信 → shift_jis →
    パラメータエラー判定」パターンを一箇所にまとめる。通信エラー等の例外は
    そのまま呼び出し元に伝播させる（呼び出し元ごとに再試行方針が異なるため、
    ここで握りつぶさない）。

    Returns
    -------
    requests.Response または None（'パラメータエラー'応答の場合）
    """
    resp = sess.post(f'{JRA_BASE}/JRADB/{page}',
                      data={'cname': cname, 'CNAME': cname},
                      headers=HEADERS, timeout=timeout)
    resp.encoding = 'shift_jis'
    if 'パラメータエラー' in resp.text:
        return None
    return resp


def _scan_r01_shutuba(base, date, sess):
    """R01出走表のsuffixを1周スキャンする。(suffix, 失敗内訳) を返す。

    JRADBの出走表はsuffix=BFのような高い値にある場合があるため、
    3連続エラーで打ち切らず256全体をスキャンする。
    パラメータエラー応答はスリープなしでスキップし高速化。
    テーブル+競馬固有コンテンツ（騎手/馬名/馬体重）を確認して偽陽性を除外。
    障害レースのフィルタリングは呼び出し側（_parse_shutuba）で行う。
    """
    diag = {'param_error': 0, 'no_table': 0, 'no_content': 0, 'exc': 0,
            'last_exc': None}
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        try:
            r = _jradb_post(sess, 'accessD.html', cn, timeout=10)
        except Exception as e:
            diag['exc'] += 1
            diag['last_exc'] = f'{type(e).__name__}: {e}'
            time.sleep(0.02)
            continue
        if r is None:
            diag['param_error'] += 1
            continue  # パラメータエラー：sleepなしで高速スキップ
        text = r.text
        soup = BeautifulSoup(text, 'lxml')
        if not soup.find_all('table'):
            diag['no_table'] += 1
            time.sleep(0.02)
            continue
        # 出走表固有コンテンツを確認（偽陽性排除）
        if not any(kw in text for kw in ('騎手', '馬名', '馬体重', '調教師')):
            diag['no_content'] += 1
            print(f'  [scan] suffix {s:02X}: テーブルあるが競馬コンテンツなし → {text[:80]!r}')
            time.sleep(0.02)
            continue
        print(f'  [scan] suffix {s:02X}: ✓ 競馬コンテンツ確認 → {text[:60]!r}')
        return s, diag
    return None, diag


def find_r01_shutuba(base, date, sess, attempts=3):
    """R01出走表のsuffixを探索する（0x00〜0xFF を順次スキャン）。

    ⚠ 1周失敗しただけで諦めると、その開催の**予想が丸ごと消える**。
    2026-08-09の日曜朝refreshでは新潟だけこのスキャンが全滅し、
    前夜は同じ新潟がsuffix=04で取れていたにもかかわらず、アプリの予想が
    中京・札幌の23レースだけになった（新潟11レースが消滅）。
    同じ失敗モードは結果ページ側で2026-08-02に起きており、そちらは
    find_r01_result に再試行を入れて解決済みだったが、出走表側は
    未対応のまま残っていた。

    そこで結果ページ側と同じ方針を適用する。一過性の兆候（通信例外・
    テーブルなし等）がある場合に限り再試行し、256件すべてが綺麗に
    「パラメータエラー」なら、そのページは本当に存在しないので即座に
    諦める（開催のない会場で待たされない）。
    """
    for attempt in range(1, attempts + 1):
        s, diag = _scan_r01_shutuba(base, date, sess)
        if s is not None:
            return s
        transient = diag['exc'] or diag['no_table'] or diag['no_content']
        if not transient:
            print(f'  ⚠ R01出走表 未発見: 256件すべてパラメータエラー'
                  f'（このページは存在しない）')
            return None
        if attempt < attempts:
            wait = 5 * attempt
            print(f'  ⚠ R01出走表 未発見（例外{diag["exc"]}件 / テーブルなし'
                  f'{diag["no_table"]}件 / 内容なし{diag["no_content"]}件'
                  + (f' / 直近の例外: {diag["last_exc"]}' if diag['last_exc'] else '')
                  + f'）→ {wait}秒後に再試行 ({attempt}/{attempts - 1})')
            time.sleep(wait)
        else:
            print(f'  ❌ R01出走表 {attempts}回試行しても未発見'
                  f'（例外{diag["exc"]}件 / テーブルなし{diag["no_table"]}件 / '
                  f'内容なし{diag["no_content"]}件）'
                  + (f' 直近の例外: {diag["last_exc"]}' if diag['last_exc'] else ''))
    return None


def _scan_r01_result(base, date, sess):
    """R01結果ページのsuffixを1周スキャンする。(suffix, 失敗内訳) を返す。"""
    diag = {'param_error': 0, 'no_table': 0, 'no_content': 0, 'exc': 0,
            'last_exc': None}
    for s in range(256):
        cn = f'{base}01{date}/{s:02X}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessS.html', data={'CNAME': cn},
                          headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception as e:
            diag['exc'] += 1
            diag['last_exc'] = f'{type(e).__name__}: {e}'
            time.sleep(0.02)
            continue
        text = r.text
        if 'パラメータエラー' in text:
            diag['param_error'] += 1
            continue
        soup = BeautifulSoup(text, 'lxml')
        if not soup.find_all('table'):
            diag['no_table'] += 1
            time.sleep(0.02)
            continue
        if not any(kw in text for kw in ('騎手', '馬名', '着順', '調教師')):
            diag['no_content'] += 1
            print(f'  [scan-result] suffix {s:02X}: テーブルあるが結果コンテンツなし → {text[:80]!r}')
            time.sleep(0.02)
            continue
        return s, diag
    return None, diag


def find_r01_result(base, date, sess, attempts=3):
    """R01結果ページのsuffixを探索する（0x00〜0xFF を順次スキャン）。

    障害レースのフィルタリングは parse_result_soup 内で行う（Noneを返す）。

    ⚠ 1周失敗しただけで諦めると、その開催のレースが**丸ごと永久に失われる**。
    2026-08-02の日曜結果取得では新潟・札幌の2会場でこのスキャンが全滅し、
    予想35レースに対し結果が11レースしか取れなかった（同じ週の土曜は3会場
    とも成功しているので、恒久的な不在ではなく一過性の失敗）。

    そこで一過性の兆候（通信例外・テーブルなし等）がある場合に限り再試行する。
    256件すべてが綺麗に「パラメータエラー」なら、そのページは本当に存在しない
    ので再試行しても無駄なため即座に諦める（開催のない会場で5分待たない）。
    """
    for attempt in range(1, attempts + 1):
        s, diag = _scan_r01_result(base, date, sess)
        if s is not None:
            return s
        transient = diag['exc'] or diag['no_table'] or diag['no_content']
        if not transient:
            print(f'  ⚠ R01結果 未発見: 256件すべてパラメータエラー'
                  f'（このページは存在しない）')
            return None
        if attempt < attempts:
            wait = 5 * attempt
            print(f'  ⚠ R01結果 未発見（例外{diag["exc"]}件 / テーブルなし'
                  f'{diag["no_table"]}件 / 内容なし{diag["no_content"]}件'
                  + (f' / 直近の例外: {diag["last_exc"]}' if diag['last_exc'] else '')
                  + f'）→ {wait}秒後に再試行 ({attempt}/{attempts - 1})')
            time.sleep(wait)
        else:
            print(f'  ❌ R01結果 {attempts}回試行しても未発見'
                  f'（例外{diag["exc"]}件 / テーブルなし{diag["no_table"]}件 / '
                  f'内容なし{diag["no_content"]}件）'
                  + (f' 直近の例外: {diag["last_exc"]}' if diag['last_exc'] else ''))
    return None


def _try_fetch_shutuba(sess, base, r, date_str, sx):
    """指定suffixで出走表ページを取得。(resp, soup) を返す。パラメータエラーの場合はNone, None。

    ⚠ 対になる `_try_fetch_result` は通信例外を握って None を返すのに、
    こちらは素通ししていた（2026-08-09の監査で発覚）。近傍±60スキャンで
    1回でも通信が切れると、そのvenueの残りレースが丸ごと失われる。
    結果側と同じく「取れなければ None」に揃える。
    """
    cn = f'{base}{r:02d}{date_str}/{sx}'
    try:
        resp = _jradb_post(sess, 'accessD.html', cn, timeout=15)
    except Exception:
        return None, None
    if resp is None:
        return None, None
    soup = BeautifulSoup(resp.text, 'lxml')
    if not soup.find_all('table'):
        return None, None
    return resp, soup


def fetch_horse_pedigree(sess, cname):
    """血統情報ページ(accessU.html)から父・母の父を取得する。

    出馬表の馬名リンクに埋め込まれた CNAME（例: 'pw01dud002024103763/CB'）を使う。
    ページは <dt>父</dt><dd>馬名</dd> のような定義リスト構造。
    母・母の母は "○○ 産駒" という表記（繁殖牝馬自体のページではないため）になるが、
    父・母の父はそのまま種牡馬名が入る。

    Returns
    -------
    dict: {'sire': str, 'dam_sire': str}（取得できなかった項目は含まない）
    """
    resp = _jradb_post(sess, 'accessU.html', cname, timeout=15)
    if resp is None:
        return {}
    soup = BeautifulSoup(resp.text, 'lxml')

    result = {}
    label_to_key = {'父': 'sire', '母の父': 'dam_sire'}
    for dt in soup.find_all('dt'):
        label = dt.get_text(strip=True)
        key = label_to_key.get(label)
        if not key:
            continue
        dd = dt.find_next_sibling('dd')
        if not dd:
            continue
        value = re.sub(r'\s*産駒\s*$', '', dd.get_text(' ', strip=True))
        if value:
            result[key] = value
    return result


# 1回のワークフロー実行（出馬表取得 or 結果取得の全レース分）あたりの
# 血統新規取得の上限。導入直後は history.db に sire が一切無いため全馬が
# "新規"扱いになり、無制限だと数百件の追加リクエストでCIの30分タイムアウトに
# 達してしまう（2026-07-18 に実際に発生・全データ喪失）。上限に達した馬は
# 静かにスキップし、次回の実行で改めて拾われる（数週間かけて段階的に埋まる）。
PEDIGREE_FETCH_BUDGET_DEFAULT = 60


def _fill_pedigree(sess, horses, hist_db_path, budget=None):
    """出走馬の血統(父・母の父)を補完する。

    history.db に既に記録済みの馬（過去に一度でも取得済み）は再取得しない
    （血統は不変データのためキャッシュとして扱える）。未記録の新規馬のみ
    accessU.html へ追加リクエストする。1頭の失敗が他馬・レース全体を
    止めないよう、例外は個別に握りつぶす。

    Parameters
    ----------
    budget : dict または None
        {'remaining': int} 形式の共有カウンタ。呼び出し元が複数レース分を
        ループしながら同じ dict を使い回すことで、実行全体での新規取得数に
        上限を設ける。上限到達後は残りの馬を静かにスキップする（次回の
        実行で改めて拾われる）。None なら無制限（テスト・単発呼び出し用）。
    """
    conn = sqlite3.connect(hist_db_path)
    try:
        for h in horses:
            cname = h.get('pedigree_cname')
            row = None
            try:
                row = conn.execute(
                    "SELECT sire, dam_sire FROM horse_history "
                    "WHERE horse_name=? AND sire IS NOT NULL AND sire != '' "
                    "ORDER BY date DESC LIMIT 1", (h.get('name', ''),)
                ).fetchone()
            except sqlite3.OperationalError:
                pass  # 未マイグレーションの旧DB（sire列なし）
            if row and row[0]:
                h['sire'] = row[0]
                h['dam_sire'] = row[1] or ''
                continue
            if not cname:
                continue
            if budget is not None:
                if budget.get('remaining', 0) <= 0:
                    continue  # 今回の実行では上限到達。次回の実行で拾う
                budget['remaining'] -= 1
            try:
                ped = fetch_horse_pedigree(sess, cname)
                if ped.get('sire'):
                    h['sire'] = ped['sire']
                if ped.get('dam_sire'):
                    h['dam_sire'] = ped['dam_sire']
            except Exception as e:
                print(f'  ⚠ 血統取得失敗 ({h.get("name", "?")}): {e}')
            time.sleep(0.3)
    finally:
        conn.close()


def _try_fetch_result(sess, base, r, date_str, sx):
    """指定suffixで結果ページを取得。soup を返す。失敗時はNone。"""
    cn = f'{base}{r:02d}{date_str}/{sx}'
    try:
        resp = _jradb_post(sess, 'accessD.html', cn, timeout=15)
    except Exception:
        return None
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, 'lxml')
    if not soup.find_all('table'):
        return None
    return soup


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

    256件全て不一致で終わった場合、原因の内訳（パラメータエラー/テーブルなし/
    例外）をログに残す。2026-07-25にこの関数が全venueで原因不明のまま
    「未発見」になる障害が発生した際、例外を無条件に握りつぶす実装のため
    実際に何が起きていたのか（JRA側の遅延か別の問題か）を後から特定できな
    かった反省による（詳細はCLAUDE.md参照）。
    """
    n_param_error = 0
    n_no_table = 0
    n_exception = 0
    last_exception = None
    for s in range(256):
        cn = f'{odds_base}01{date_str}Z/{s:02X}'
        try:
            r = sess.post(f'{JRA_BASE}/JRADB/accessO.html',
                          data={'cname': cn, 'CNAME': cn}, headers=HEADERS, timeout=10)
            r.encoding = 'shift_jis'
        except Exception as e:
            n_exception += 1
            last_exception = e
            continue
        text = r.text
        if 'パラメータエラー' in text:
            n_param_error += 1
            continue
        if BeautifulSoup(text, 'lxml').find_all('table'):
            return s
        n_no_table += 1
        time.sleep(0.02)
    detail = f'パラメータエラー={n_param_error} テーブルなし={n_no_table} 例外={n_exception}'
    if last_exception is not None:
        detail += f' (直近の例外: {last_exception!r})'
    print(f'  ⚠ find_r01_odds: 256件全て不一致 [{detail}]')
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


def apply_odds_to_races(races, market_odds_map):
    """market_odds_map の単勝オッズを各馬の win_odds に書き戻す。

    出馬表ページ(_parse_shutuba)にはオッズが載らない（特に前日=金曜）ため、
    専用オッズページ(fetch_odds_map)で取得した値を各馬に反映する。
    これを呼ばないと win_odds=0.0 のまま予想が走り、popularity 導出・
    バリュー表示・EV買い目がすべて空になる。

    market_odds_map に無い / tansho が None・0 の馬は既存 win_odds を保持する。
    単勝として成立しない値（1.0倍未満）は取得失敗時のゴミ値とみなして
    反映しない（2026-07-26に0.1〜0.2倍が45頭混入し、オッズ昇順で導出する
    popularity が壊れた事故への対応。詳細は CLAUDE.md 2026-07-27⑦）。

    Args:
        races           : fetch_races_on_date が返すレースリスト
        market_odds_map : {race_id: {horse_num: {'tansho', 'fukusho'}}}

    Returns:
        int: win_odds を更新した馬の数
    """
    from src.features.engine import MIN_VALID_WIN_ODDS

    updated = 0
    invalid = 0
    for race in races:
        om = market_odds_map.get(race.get('id')) or {}
        if not om:
            continue
        for h in race.get('horses', []):
            num = h.get('num') or h.get('horse_num')
            if num is None:
                continue
            info = om.get(num) or om.get(int(num))
            tansho = info.get('tansho') if isinstance(info, dict) else None
            if tansho and tansho >= MIN_VALID_WIN_ODDS:
                h['win_odds'] = float(tansho)
                updated += 1
            elif tansho and tansho > 0:
                invalid += 1
    if invalid:
        print(f'⚠ [オッズ異常] 単勝1.0倍未満の値を{invalid}頭ぶん破棄'
              f'（オッズ取得の部分失敗の可能性）')
    return updated


def fetch_races_on_date(sess, target_date, hist_db_path):
    """指定日の全レース出走表を取得

    Returns:
        (all_races, failures) のタプル。
        failures: [{'racecourse': str, 'race_num': int, 'reason': str}, ...]
        ページ自体が取得できた（=そのレースは実在する）のにパースに失敗した
        もののみを記録する。障害レースのスキップ、および該当venueがその日
        12レース未満で該当レース番号のページ自体が存在しないケース
        （suffix探索を尽くしても soup が None のまま）は意図した挙動のため
        含めない。
    """
    print(f'📡 {target_date} 出走表取得中...')
    all_races = []
    failures = []
    pedigree_budget = {'remaining': PEDIGREE_FETCH_BUDGET_DEFAULT}
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

            # 全レース共通：計算式が外れた場合に近傍±60をスキャン
            if soup is None:
                base_s = int(sx, 16)
                found_delta = None
                for delta in range(1, 61):
                    for sign, cand in [(+delta, (base_s + delta) % 256),
                                       (-delta, (base_s - delta) % 256)]:
                        sx_c = f'{cand:02X}'
                        _, soup_c = _try_fetch_shutuba(sess, base, r, date_str, sx_c)
                        if soup_c is not None:
                            soup = soup_c
                            sx = sx_c
                            found_delta = sign
                            break
                    if soup is not None:
                        break
                if found_delta is not None:
                    print(f'  R{r:02d}: suffix補正 {found_delta:+d} → {sx}')

            if soup is None:
                # 開催venueがその日12レース未満しかない場合も含まれるため
                # （fetch_results()の同種分岐と同じ扱い）、failuresには含めない。
                print(f'  R{r:02d}: suffix={sx} → パラメータエラー/ページなし')
                continue

            race = _parse_shutuba(soup, rc, r, date_str, pc, hist_db_path)
            if not race:
                # 原因を特定するため詳細ログを出力（障害レースのスキップは
                # 意図した挙動のためfailuresには記録しない）
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
                                failures.append({'racecourse': rc, 'race_num': r, 'reason': '日付不一致'})
                            else:
                                print(f'  R{r:02d}: parse失敗 (馬なし or 例外) suffix={sx}')
                                failures.append({'racecourse': rc, 'race_num': r, 'reason': 'parse失敗'})
                    else:
                        print(f'  R{r:02d}: テーブルなし suffix={sx}')
                        failures.append({'racecourse': rc, 'race_num': r, 'reason': 'テーブルなし'})
                except Exception:
                    print(f'  R{r:02d}: ログ取得中に例外 suffix={sx}')
                    failures.append({'racecourse': rc, 'race_num': r, 'reason': '例外'})
                time.sleep(0.3)
                continue

            # 血統(父・母の父)を補完する。history.dbに未記録の新規馬のみ追加リクエストする
            # （実行全体でPEDIGREE_FETCH_BUDGET_DEFAULT件までに制限。超過分は次回に持ち越し）。
            _fill_pedigree(sess, race['horses'], hist_db_path, budget=pedigree_budget)

            # オッズ取得用のCNAME情報を保持（fetch_odds_for_race で使用）
            race['_odds_cn'] = {'base': base, 'date_str': date_str, 'sx': sx, 'race_num': r, 'odds_r01': odds_r01}

            all_races.append(race)
            print(f'  R{r:02d}: {race.get("race_name", "")} '
                  f'{race.get("num_horses", 0)}頭 '
                  f'{race.get("distance", 0)}m{race.get("surface", "")}')
            time.sleep(0.8)
    print(f'\n📋 出走表取得完了: {len(all_races)}レース'
          + (f'（取得失敗 {len(failures)}件）' if failures else ''))
    return all_races, failures


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


def _derive_corner3(corner_all):
    """corner_all（'3-3-2-1'形式のハイフン区切り通過順）から3コーナー通過順位を導出する。

    2026-08-03発見: horse_history.corner_3 列は2026-06-25のcorner_all導入時に
    書き込みが停止し「常にNULL固定」になった（docs/history_db_schema.md に
    明記済みの既知事項）が、calc_features_for_xgb() 側の f_pos_avg_3 等
    8特徴量・speed_index の corner_pos 引数はこの移行に追従せず今も
    corner_3 キーを読み続けており、corner_all移行以降ずっとフォールバック
    値（running_styleベースの粗い代用）に落ちていた。corner_allは3コーナー
    以降を4値持つとは限らない（2〜4値）ため、3番目の値が無ければNoneのまま。
    """
    if not corner_all:
        return None
    parts = corner_all.split('-')
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except (TypeError, ValueError):
        return None


# 過去走を何走ぶん見るか。学習(build_training_data._get_history_before)と
# 推論(get_history_from_db)は必ず同じ値でなければならない。
# 🔴 2026-08-27まで 学習=10 / 推論=5 と食い違っており、134特徴量のうち78個が
#    別の値になっていた（6走以上ある馬＝直近窓の約52%が該当）。
#    実測すると AUC への影響は +0.0002 と小さかったが、Brier/LogLoss は
#    3窓とも10走側が良く、cal_prob の較正がわずかに悪化していた。
#    片方だけ変えられないよう、両側がこの定数を参照する形にしてある。
HISTORY_LIMIT = 10


def get_history_from_db(horse_name, hist_db_path, limit=HISTORY_LIMIT):
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
                   COALESCE(r.race_name, '') as race_name,
                   COALESCE(h.popularity, 0) as popularity,
                   COALESCE(h.racecourse, '') as racecourse,
                   COALESCE(h.corner_all, '') as corner_all,
                   h.finish_time, h.time_diff_sec,
                   h.body_weight, h.body_weight_diff
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
                       COALESCE(r.race_name, '') as race_name,
                       COALESCE(h.popularity, 0) as popularity,
                       COALESCE(h.racecourse, '') as racecourse,
                       COALESCE(h.corner_all, '') as corner_all,
                       h.finish_time, h.time_diff_sec,
                       h.body_weight, h.body_weight_diff
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
             agari_rank_stored, num_finishers, race_name, popularity,
             racecourse, corner_all, finish_time_val, time_diff_sec_val,
             body_weight_val, body_weight_diff_val) = row

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
                # num_finishers は finishers の別名。学習側(_get_history_before)が
                # 両キー名を提供する設計に合わせる。calc_course_aptitude_features の
                # f_agari_rank_at_type が num_finishers/agari_rank を参照する。
                "num_finishers": max(finishers_count, 1),
                "distance": distance,
                "surface": surface,
                "class": race_class,
                "margin": margin,
                # agari_rank: 学習側と同じく、未記録(-1)ならNoneで欠損を明示する
                # （2026-08-03発見: この列自体は既にSELECTされ agari3f_rank_pct の
                # 計算に使われていたが、出力辞書には含まれておらず
                # calc_course_aptitude_features の f_agari_rank_at_type が
                # 推論時は常に hrec.get('agari_rank') -> None に落ち、
                # 学習時の実データ分布と乖離していた）
                "agari_rank": agari_rank_stored if agari_rank_stored > 0 else None,
                "agari3f_rank_pct": round(agari3f_rank_pct, 3),
                "condition": track_condition,
                # track_condition は f_heavy_track_rate / speed_index が参照するキー名。
                # 'condition' と同じ値を別名でも持たせ、学習側(_get_history_before)の
                # 両キー提供パターンに合わせる。
                "track_condition": track_condition,
                "date": date,
                "last_3f": agari3f,
                "first_3f": first_3f_val,
                # corner_3 は列自体が常にNULL（docs/history_db_schema.md既知事項）。
                # corner_all（4値までのハイフン区切り通過順）の3番目から導出する。
                "corner_3": corner_3 if corner_3 is not None else _derive_corner3(corner_all),
                "race_id": race_id,
                "running_style": running_style_hist,
                "race_name": race_name,
                "popularity": popularity,
                "racecourse": racecourse,
                "corner_all": corner_all,
                "finish_time": finish_time_val,
                "time_diff_sec": time_diff_sec_val,
                "body_weight": body_weight_val,
                "body_weight_diff": body_weight_diff_val,
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
    idx = text.find('枠連')
    if idx >= 0:
        km = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if km:
            divs['wakuren'] = {'nums': [int(km[0][0]), int(km[0][1])],
                               'payout': int(km[0][2].replace(',', ''))}
    idx = text.find('馬連')
    if idx >= 0:
        um = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if um:
            divs['umaren'] = {'nums': [int(um[0][0]), int(um[0][1])],
                              'payout': int(um[0][2].replace(',', ''))}
    # 馬単は着順ありの組（1着→2着）。db.py の bet_type='馬単' 決済がこのキーを参照する
    idx = text.find('馬単')
    if idx >= 0:
        utm = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if utm:
            divs['umatan'] = {'nums': [int(utm[0][0]), int(utm[0][1])],
                              'payout': int(utm[0][2].replace(',', ''))}
    idx = text.find('ワイド')
    if idx >= 0:
        wm = re.findall(r'(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 300])
        if wm:
            divs['wide'] = [{'nums': [int(w[0]), int(w[1])], 'payout': int(w[2].replace(',', ''))} for w in wm[:3]]
    # 「三連複」(旧表記)と「3連複」(現行ページの数字表記)の両方に対応
    idx = text.find('三連複')
    if idx < 0:
        idx = text.find('3連複')
    if idx >= 0:
        sm = re.findall(r'(\d+)-(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if sm:
            divs['sanrenpuku'] = {'nums': [int(sm[0][0]), int(sm[0][1]), int(sm[0][2])],
                                  'payout': int(sm[0][3].replace(',', ''))}
    idx = text.find('三連単')
    if idx < 0:
        idx = text.find('3連単')
    if idx >= 0:
        stm = re.findall(r'(\d+)-(\d+)-(\d+)\s+([\d,]+)\s*円', text[idx:idx + 200])
        if stm:
            divs['sanrentan'] = {'nums': [int(stm[0][0]), int(stm[0][1]), int(stm[0][2])],
                                 'payout': int(stm[0][3].replace(',', ''))}
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


_RESULT_ROW_DIAG_DONE = False


def diag_result_row_columns(texts):
    """結果ページの馬別行が想定した列数・列位置と一致しない場合に、
    実際の texts 配列を1回だけ診断ログに残す（2026-08-03発見）。

    2026-08-03の調査で、history.db の popularity/body_weight/trainer が
    2026-06-28まで正常に埋まっていたのに 2026-07-04以降は突然100%欠損に
    転じていたと判明した（win_odds は元々0%充足で無関係・既知）。
    jra_scraper.py はこの間コミットされておらず、texts[0..10]相当
    （着順・枠番・馬番・馬名・性齢・斤量・騎手・タイム・着差・上がり）は
    今も正常に取れているため、コード側ではなく実際のJRADB結果ページの
    列構成が texts[11]（単勝）以降のどこかで変わった可能性が高い。

    この環境からは www.jra.go.jp への到達がブロックされており実HTMLを
    確認できないため、find_r01_odds/diag_pace_label_missing と同じ方針で
    決め打ち修正はせず、次回のワークフロー実行（GitHub Actions、
    ネットワーク到達可）のログに実際の列内容を残すだけに留める。
    """
    global _RESULT_ROW_DIAG_DONE
    if _RESULT_ROW_DIAG_DONE:
        return
    _RESULT_ROW_DIAG_DONE = True
    print(f'⚠ 結果ページ列診断: texts配列(len={len(texts)}) = {texts!r}')


_TRAINER_AFFIL_RE = re.compile(r'^(.+?)[\(（](栗東|美浦)[\)）]$')


def _split_trainer_affiliation(trainer_text):
    """調教師欄の「西村真幸(栗東)」/「秋本大介(美浦)」形式から所属を分離する。

    実機（sp.jra.jp）の結果ページで確認済みの表記。所属表記が無い場合
    （通期表示ではなく名前のみ等）は affiliation=None を返す後方互換設計。
    列位置の想定がズレて trainer_text 自体が空/別データの場合も、
    パターン不一致でNoneを返すだけで例外は出さない。

    Returns:
        (trainer_name: str, affiliation: '栗東'|'美浦'|None)
    """
    m = _TRAINER_AFFIL_RE.match(trainer_text.strip())
    if m:
        return m.group(1).strip(), m.group(2)
    return trainer_text.strip(), None


_PACE_LABEL_DIAG_DONE = False


def diag_pace_label_missing(full_text):
    """ペース判定が取れなかった原因を1回だけ診断ログに残す。

    race_history.pace_label は 5,411レースすべてで NULL（充足0%）。
    2026-07-22に単語表記（スロー/ミドル/ハイ）対応を入れたが、その後に
    取得した 2026-07-25・07-26 も 0件のままだった。関数自体は
    「ペース判定：ミドルペース」等を正しく拾えることを単体で確認済みなので、
    正規表現ではなく **取得元ページにその文字列が無い** 可能性が高い
    （7/22の修正根拠は sp.jra.jp のスクリーンショットだが、
      スクレイパーの実際の取得元は www.jra.go.jp の JRADB ページ）。

    実HTMLをこの環境から確認できないため決め打ちの修正はせず、
    次回のワークフロー実行でページに何が載っているかを判別できる
    情報だけを残す（find_r01_odds と同じ方針）。

    なお pace_label が無くても展開分類は機能する
    （train_pace_model._classify_pace が first_3f から導出。充足88.9%）。
    """
    global _PACE_LABEL_DIAG_DONE
    if _PACE_LABEL_DIAG_DONE:
        return
    _PACE_LABEL_DIAG_DONE = True
    t = unicodedata.normalize('NFKC', full_text or '')
    if 'ペース' not in t:
        print('⚠ ペース判定: ページに「ペース」の文字自体が存在しない '
              '（取得元にこの情報が無い可能性が高い）')
        return
    i = t.find('ペース')
    print(f'⚠ ペース判定: 「ペース」は存在するが既知の表記に一致せず。'
          f'周辺: ...{t[max(0, i - 40):i + 60]}...')


def _extract_weather_pace(header_text, full_text=None):
    """ヘッダから天候を、ページ全体からペース判定を抽出。

    天候はヘッダ表（レース情報の1行目、例:「天候:晴」）から取得できるが、
    ペース判定は実機（sp.jra.jp）ではヘッダとは別のセクション（タイム欄近辺）に
    「ペース判定：ミドルペース」のような**単語表記**で載っている。旧実装は
    「ペース」直後のH/M/S 1文字のみを探索しており、この単語表記には一致せず、
    ペース判定が長期未取得だった可能性が高い。header_text・full_text の両方を
    対象に、1文字表記(H/M/S)と単語表記(スロー/ミドル/ハイ)の両方を探索する。
    """
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

    pace_text = unicodedata.normalize('NFKC', full_text) if full_text else t
    pace = None
    pm = re.search(r'ペース[\s:：]*([HMS])\b', pace_text)
    if pm:
        pace = pm.group(1)
    else:
        pm2 = re.search(r'ペース(?:判定)?[\s:：]*(スロー|ミドル|ハイ)', pace_text)
        if pm2:
            pace = {'スロー': 'S', 'ミドル': 'M', 'ハイ': 'H'}[pm2.group(1)]
    return weather, pace


def _extract_lap_times(soup):
    """結果ページからラップタイム（ハロンごとの区間タイム）を抽出する。

    JRA結果ページ（sp.jra.jp実機で確認済み）は「タイム」欄の中に「ハロンタイム」
    見出しで各区間タイムを、「上り」見出しで4F/3F上りタイムを掲載する
    （例: "9.5 - 11.1 - 11.6 - 12.2 - 12.4 - 12.8" / "4F 49.0 - 3F 37.4"）。
    旧実装は「ラップタイム」表記のみを探しており、この見出し違いにより
    first_3f/last_3fが長期間未取得（0%）だった可能性が高いため、両表記に対応する。

    Returns:
        (lap_times: list[float], first_3f: float|None, last_3f: float|None)
        抽出できない場合は ([], None, None)。
    """
    text = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    idx = text.find('ラップタイム')
    if idx < 0:
        idx = text.find('ハロンタイム')
    if idx < 0:
        return [], None, None
    # 見出し以降・次見出し（ペース/コーナー通過順位/払戻金）または300文字までを対象に区間タイムを収集
    segment = text[idx:idx + 300]
    end_candidates = [e for e in (segment.find('ペース'),
                                   segment.find('コーナー通過順位'),
                                   segment.find('払戻金')) if e > 0]
    if end_candidates:
        segment = segment[:min(end_candidates)]
    laps = [float(m) for m in re.findall(r'(\d{1,2}\.\d)', segment)]
    # ラップは概ね 9.0〜15.0 秒/200m の範囲。範囲外（誤検出）は除外
    laps = [v for v in laps if 8.0 <= v <= 16.0]
    if len(laps) < 3:
        return [], None, None
    first_3f = round(sum(laps[:3]), 1)
    last_3f = round(sum(laps[-3:]), 1)
    return laps, first_3f, last_3f


def _extract_corner_passage(soup):
    """「コーナー通過順位」セクションから3角・4角の通過順を生テキストのまま抽出する。

    実機（sp.jra.jp）で確認した表記例:
        3コーナー: (1,*5)6,10(2,9)-(3,4)8=7
        4コーナー: 1(5,6)-(9,10)-(2,3,4)=8=7
    括弧は併走（横に並んでいる）、"-"は差、"="は大きく離れていることを表す。
    ここでは生テキストのまま保存し、構造の展開は parse_corner_passage() が行う。

    Returns:
        dict: {'corner_pass_3': str|None, 'corner_pass_4': str|None}
        セクション自体が見つからない場合は空dict。
    """
    text = unicodedata.normalize('NFKC', soup.get_text(' ', strip=True))
    idx = text.find('コーナー通過順位')
    if idx < 0:
        return {}
    segment = text[idx:idx + 500]
    end_candidates = [e for e in (segment.find('払戻金'), segment.find('タイム')) if e > 0]
    if end_candidates:
        segment = segment[:min(end_candidates)]

    # 🔴 先にラベルで区切ってから中身を取る。
    #    ラベルの後ろを貪欲に拾うと、次の見出し '4コーナー' の '4' まで
    #    通過順に混ざる。また soup.get_text(' ') は要素間に空白を入れるため、
    #    文字クラスに空白を含めないと最初の区切りで切れる
    #    （実データで '(1,*5)6,10(2,9)-(3,4)8=7' が '(*7,' になっていた）。
    LABELS = [('1コーナー', None), ('2コーナー', None),
              ('3コーナー', 'corner_pass_3'), ('4コーナー', 'corner_pass_4'),
              ('最終コーナー', 'corner_pass_4')]
    marks = sorted(
        ((m.start(), m.end(), key)
         for label, key in LABELS
         for m in re.finditer(re.escape(label), segment)),
        key=lambda t: t[0])

    result = {}
    for i, (_s, e, key) in enumerate(marks):
        if key is None or key in result:
            continue
        body = segment[e: marks[i + 1][0] if i + 1 < len(marks) else len(segment)]
        raw = re.sub(r'[^0-9,()（）*=\-]', '', body).strip(',-=')
        # 通過順の表記なら馬番が3つ以上並ぶ。満たさないものは採用しない
        # （見出しだけ拾って中身が空、という壊れ方を防ぐ）
        if len(re.findall(r'\d+', raw)) >= 3:
            result[key] = raw
    return result


def parse_corner_passage(text):
    """コーナー通過順の表記を、馬番→隊列内の位置に展開する。

    表記例: '(1,*5)6,10(2,9)-(3,4)8=7'
      括弧 = 併走（横に並んでいる）、'-' = 差がある、'=' = 大きく離れている
      '*' = 先頭表示

    Returns:
        dict: {馬番: {'rank': 先頭からの順位(1始まり),
                      'group_size': 同じ括弧内の頭数（1なら単独）}}
        解釈できなければ空dict。

    「同じ3番手でも単独か馬群の中か」を区別するための最小限の展開に留める。
    走行ルート（内・外）は表記に含まれないため、ここでは分からない。
    """
    if not text:
        return {}
    out, rank = {}, 0
    for m in re.finditer(r'\(([^)]*)\)|(\d+)', text):
        nums = re.findall(r'\d+', m.group(1) if m.group(1) else m.group(2))
        if not nums:
            continue
        rank += 1
        for n in nums:
            v = int(n)
            if 1 <= v <= 18 and v not in out:
                out[v] = {'rank': rank, 'group_size': len(nums)}
    return out


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
        _full_text = soup.get_text(' ', strip=True)
        weather, pace = _extract_weather_pace(header, _full_text)
        info['weather'] = weather
        info['pace_label'] = pace
        if not pace:
            diag_pace_label_missing(_full_text)
        # ラップタイム（区間タイム）と前半/後半3F
        laps, first_3f, last_3f = _extract_lap_times(soup)
        info['lap_times'] = '-'.join(f'{v:.1f}' for v in laps) if laps else ''
        info['first_3f'] = first_3f
        info['last_3f'] = last_3f
        # コーナー通過順位（同着グルーピング表記込みの生テキスト、収集のみ）
        corner_pass = _extract_corner_passage(soup)
        info['corner_pass_3'] = corner_pass.get('corner_pass_3')
        info['corner_pass_4'] = corner_pass.get('corner_pass_4')
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
            # 馬名リンクは血統情報ページ(accessU.html?CNAME=...)への直リンクになっている
            pedigree_cname = None
            if len(cells) > 3:
                name_a = cells[3].find('a')
                if name_a:
                    cn_m = re.search(r'CNAME=([^&]+)', name_a.get('href', ''))
                    if cn_m:
                        pedigree_cname = cn_m.group(1)
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
            # 列順: ...上がり(10), 馬体重(11), 調教師(12), 人気(13)。
            # 2026-08-03発見: 2026-07-04頃にJRADB結果ページの列構成が変わり、
            # 単勝オッズ列が消滅した上で人気/馬体重/調教師の並びが
            # 「単勝(11)人気(12)馬体重(13)調教師(14)」→「馬体重(11)調教師(12)人気(13)」
            # に変わっていた（旧構成はtexts末尾が15列、新構成は14列）。
            # probe-result-columns.yml で 2026-08-02 の実データを取得し
            # texts配列を直接確認して特定した
            # （例: ['1','','7','ノドゥス','牡2','55.0','三浦 皇成','1:34.3','',
            #        '4 5','33.5','482 (+4)','斎藤 誠','3']）。
            # 調教師欄に「(栗東)」等の所属表記は付いておらず、trainer_affiliationは
            # 別の情報源が必要（_split_trainer_affiliationは名前のみ渡されれば
            # affiliation=Noneを返す後方互換設計なので、そのまま安全に動作する）。
            pop_m = re.match(r'^(\d+)$', texts[13].strip()) if len(texts) > 13 else None
            jockey = texts[6].strip() if len(texts) > 6 else ''
            trainer_raw = texts[12].strip() if len(texts) > 12 else ''
            trainer, trainer_affiliation = _split_trainer_affiliation(trainer_raw)
            margin_txt = texts[8].strip() if len(texts) > 8 else ''
            finish_time = _parse_finish_time(texts[7].strip() if len(texts) > 7 else '')
            # 馬体重（texts[11]、'482 (+4)' 形式）
            body_weight, body_weight_diff = _extract_body_weight(texts, start_idx=11)
            # 単勝オッズ列は新構成では存在しない（2026-08-03確認）。
            # 呼び出しは残すが常にNoneが返る想定（popularityで代替済み、
            # engine.py に既存の明示コメントあり）
            win_odds = _extract_win_odds(texts, start_idx=11)
            if not pop_m or not trainer_raw or body_weight is None:
                diag_result_row_columns(texts)
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
                'pedigree_cname': pedigree_cname,
                'trainer_affiliation': trainer_affiliation,
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



def fetch_results(sess, target_date, calendar=None, hist_db_path=None):
    """指定日の全レース結果を取得。

    hist_db_path を渡すと、確定した出走馬の血統(父・母の父)を補完して
    history.db に永続化できる状態にする（history.dbに未記録の新規馬のみ
    追加リクエストする）。省略時は血統補完をスキップする（後方互換）。
    """
    from src.scraper.calendar import get_kaisai_on_date
    print(f'📡 {target_date} 結果取得中...')
    all_results = []
    pedigree_budget = {'remaining': PEDIGREE_FETCH_BUDGET_DEFAULT}

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
            soup = _try_fetch_result(sess, base_result, r, target_date, sx)

            if soup is None and r >= 10:
                sx_simple = f'{(r01 + (r - 1) * 181) % 256:02X}'
                if sx_simple != sx:
                    soup2 = _try_fetch_result(sess, base_result, r, target_date, sx_simple)
                    if soup2 is not None:
                        soup = soup2
                        sx = sx_simple

            if soup is None:
                base_s = int(sx, 16)
                found_delta = None
                for delta in range(1, 61):
                    for _sign, cand in [(+delta, (base_s + delta) % 256),
                                        (-delta, (base_s - delta) % 256)]:
                        sx_c = f'{cand:02X}'
                        soup_c = _try_fetch_result(sess, base_result, r, target_date, sx_c)
                        if soup_c is not None:
                            soup = soup_c
                            sx = sx_c
                            found_delta = _sign
                            break
                    if soup is not None:
                        break
                if found_delta is not None:
                    print(f'  R{r:02d}: 結果suffix補正 {found_delta:+d} → {sx}')

            # ⚠ 出走表側(fetch_races_on_date)は同じ分岐で必ず理由をログに出すが、
            #   結果側はここが無言の continue だった。2026-08-09の日曜結果で
            #   札幌R10(富良野特別)・R11(UHB賞)が**何のログも残さず消えた**
            #   （再実行したら取れたので一過性の失敗）。片側だけ対策されていた
            #   典型例なので、結果側にも同じ粒度のログを入れる。
            if soup is None:
                print(f'  R{r:02d}: suffix={sx} → パラメータエラー/ページなし')
                continue
            result = parse_result_soup(soup, rc, r, target_date, pc)
            if not result:
                # 障害レースは parse_result_soup が意図的に None を返す（正常）。
                # それ以外は取りこぼしなので区別できるようにする。
                try:
                    tbls = soup.find_all('table')
                    head = tbls[0].get_text(' ', strip=True) if tbls else ''
                    if '障害' in head:
                        print(f'  R{r:02d}: 障害レース → スキップ')
                    else:
                        print(f'  R{r:02d}: 結果parse失敗 (着順なし or 例外) suffix={sx}')
                except Exception:
                    print(f'  R{r:02d}: 結果parse失敗（ログ取得中に例外） suffix={sx}')
                continue
            if hist_db_path:
                _fill_pedigree(sess, result['finishers'], hist_db_path, budget=pedigree_budget)
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
