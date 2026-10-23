"""朝に web から集めた見解と、その後の結果を**情報源ごとに**台帳に残す層。

＝＝ なぜ必要か（これが無いと「どこの情報が有効か」は永久に測れない）＝＝

2026-09-25 に作った `data/source_registry.json` は「**測った結果**を機械可読にして
採用を縛る」側はできていて、`src/utils/source_registry.py` が
①事前登録カード ②結果ファイル ③実測 ΔAUC ④列名 の4つを要求し、
欠ければ採用を無効化する。

🔴 しかし **実運用の当たり外れを情報源ごとに積む層が1つも無かった。**
`data/claude_web_findings.json` は開催日ごとに**上書き**され（過去分は git 履歴に
5コミットだけ）、結果と突合するコードは1行も存在しなかった。
だから今「どこの情報が有効か」を測ろうとすると **N=0 から始まる**。

この台帳は `data/web_findings_log.json` に**追記**し、夜の結果取得で事実を足す。

＝＝ 何を事実として足せるか ＝＝

| 次元 | 突合先 | 備考 |
|---|---|---|
| `going`（馬場・想定） | `race_history.track_condition` | 🔑 **馬券の結果を待たずに採点できる**。市場とも無関係 |
| 馬メモ・印 | `horse_history.place` / `race_dividends` | 着順・3着内・複勝払戻（100円あたり）という事実だけ |

＝＝ 設計上の約束（`danger_log.py` と同じ4つ）＝＝

🔴 **判定を再実装しない。** 読むのは `data/claude_web_findings.json` の中身そのまま
   （`src/features/claude_web.py` が検証して書いた形）。条件が2箇所に分かれると、
   記事に載せた見解と台帳の見解が静かに食い違う。

🔴 **一度記録したものを上書きしない**（`(race_id, kind, key)` で先着順）。
   当日 11:30/14:00 の refresh で findings が差し替わっても朝の値が残る。

🔴 **結果が無いレースを空で埋めない。** 払戻が1行も無ければ保留
   （空で埋めると全部外れとして記録される）。着順だけ無い場合も保留。

🔴 **回収率を出さない。** 2026-10-02 の決定どおり、保存するのは着順・3着内か・
   複勝払戻という事実だけ。集計は読む側でやる。

⚠ **これは測定の器であって成績ではない。** 週2日・10R/11R だけなので
   N は年間100件程度しか増えない。**これ単独では答えが出ない**（North Star #7）。
   価値は「手作業なしで残る」ことだけ。

⚠ **予想には一切繋がない。** この層は本番の数字を読まないし書かない
   （`claude_web.py` の別枠設計をそのまま引き継ぐ）。

＝＝ 🔴 情報源ごとに「測れる形」が違う（2026-10-23 に実データ97件で判明）＝＝

「どの情報源が信頼できるか」は**共通の的中率1本では測れない**。実測した内訳:

| 情報源 | 1レースあたり言及頭数 | 序列 | 同レース対照群 |
|---|---|---|---|
| `keibanomiryoku` | 中央値 **15.5頭**（ほぼ全馬） | 自由文のみ（評価記号 2/51） | **無い** |
| `umasiru` | 中央値 **13.5頭**（ほぼ全馬） | note に 評価 S/A/B/C が 26/27 | グレード帯の**間**に取る |
| `yosounohone` | **3頭**（選別型） | ★（note の文中） | 同レースの印なし |
| `bloodline-trackbias` | going のみ | — | 確定馬場との突合 |

🔴 **全馬型（ほぼ全馬にメモが付く情報源）の「3着内率」は信頼度ではない。**
全馬に言及すれば率は母集団の3着内率（約20%）そのものになる。実測 10/50 = 20.0% は
まさにそれで、**当たり外れの概念が無い**。昨日 `yosounohone` の★が測れたのは
「同レースに印なしの馬が居た」からで（`oikiri/RESULTS.md`）、全馬型にはその
対照群が**構造的に存在しない**（jugem を「1レース1頭なのでレース内順位が常に1・
z が 0」で閉じたのと同型の、向きが逆の問題）。

🔑 **だから測れるのは「レース内で変わる序列」を持つ情報源だけ。**
全馬型でも序列（評価 S/A/B/C）があれば帯の間に差を取れる。序列が無ければ測れない。
判定の基準は `websig/CRITERIA_RELIABILITY.md` に事前登録した。

⚠ **序列は `said.grade` に構造化して記録する。`note` の自由文から正規表現で
後から取り出さない**（記録された値とこちらが解釈した値が同じ列に混ざり、後から
区別できなくなる。2026-09-02 のパドック `condition` キーと同型）。
既存97件は `grade=None` のまま残す（遡って埋めない）。
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

JST = timezone(timedelta(hours=9))

SCHEMA_VERSION = 2
LEDGER_FILENAME = os.path.join('data', 'web_findings_log.json')
FINDINGS_FILENAME = os.path.join('data', 'claude_web_findings.json')

RESULT_SOURCE = ('history.db:race_history.track_condition（確定馬場） / '
                 'history.db:horse_history.place（着順） / '
                 'keiba.db:race_dividends（複勝払戻） / '
                 'keiba.db:race_schedule（発走時刻）')

KINDS = ('going', 'horse', 'mark')

# JRA の馬場状態。`expected`（自由文の想定）からこの語を拾えたときだけ
# 「予想が当たったか」を書く。
TRACK_STATES = ('不良', '稍重', '重', '良')

#: 情報源が付けた序列（評価 S/A/B/C、★2 など）の最大長。これより長ければ
#: 自由文が紛れ込んだと見なして捨てる
GRADE_MAX_LEN = 8


def normalize_grade(value):
    """情報源が付けた序列を短い文字列に正規化する。

    🔴 `note` の自由文から正規表現で取り出さない。**情報源が構造化して渡した
    ものだけ**を受ける。後付け抽出をすると、同じ `grade` 列に「情報源が付けた値」と
    「こちらが解釈した値」が混ざり、後から区別できなくなる。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > GRADE_MAX_LEN:
        return None
    return text


def ledger_path(base_dir):
    return os.path.join(base_dir or '.', LEDGER_FILENAME)


def empty_ledger():
    return {'schema_version': SCHEMA_VERSION, 'updated_at': None, 'entries': []}


#: 読める schema_version。**1 → 2 は `said.grade` の追加だけ**なので、
#: 既存エントリに手を入れずそのまま読める（`said.get('grade')` が None になる）。
#: 🔴 遡って `grade` を埋めない。note の自由文から後付けで入れた値と、情報源が
#: 付けた値が同じ列に混ざると後から区別できない。
READABLE_SCHEMA_VERSIONS = (1, 2)


def load(base_dir):
    path = ledger_path(base_dir)
    if not os.path.exists(path):
        return empty_ledger()
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    ver = data.get('schema_version')
    if ver not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f'{LEDGER_FILENAME} の schema_version={ver} は未対応'
            f'（読めるのは {READABLE_SCHEMA_VERSIONS}）。作り直すと過去の記録が'
            '消えるので、移行コードを書いてから読むこと')
    data.setdefault('entries', [])
    return data


def save(base_dir, ledger, now=None):
    ledger['schema_version'] = SCHEMA_VERSION
    ledger['updated_at'] = (now or datetime.now(JST)).isoformat()
    ledger['entries'] = sorted(ledger.get('entries') or [],
                               key=lambda e: (str(e.get('date')),
                                              str(e.get('race_id')),
                                              str(e.get('kind')),
                                              str(e.get('key'))))
    path = ledger_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, ensure_ascii=False, indent=1)
            f.write('\n')
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _key(entry):
    return (str(entry.get('race_id')), str(entry.get('kind')), str(entry.get('key')))


def source_host(url):
    """出典 URL のホストを返す。これが「どこの情報か」の単位。"""
    try:
        h = (urlparse(str(url)).hostname or '').lower()
    except ValueError:
        return None
    return h[4:] if h.startswith('www.') else (h or None)


def _hosts(sources):
    out = []
    for s in sources or []:
        h = source_host(s)
        if h and h not in out:
            out.append(h)
    return out


def expected_state(text):
    """`expected`（自由文）から JRA の馬場状態を1語だけ拾う。拾えなければ None。

    🔴 **`state` と `expected` は別物。** 実物を見ると
      `state` は「金曜昼時点で良」＝**収集時点で既に発表されている馬場**で、
      `expected` が「土曜はさらに乾燥する見込み」という**予想**のほう。
      `state` を確定馬場と比べると「馬場が動かなかった」ことしか測れず、
      実測で 21/21 と 100% になる（＝予想の精度ではない）。
      予想として採点できるのは `expected` が状態名を名乗ったときだけで、
      実際は「乾燥（含水率2％台）」「やや硬め」のように名乗らないことが多い。
    """
    t = str(text or '')
    for st in TRACK_STATES:
        if st in t:
            return st
    return None


def _race_date(race_id):
    rid = str(race_id or '')
    if len(rid) >= 8 and rid[:8].isdigit():
        return f'{rid[0:4]}-{rid[4:6]}-{rid[6:8]}'
    return None


def load_findings_raw(base_dir):
    """`data/claude_web_findings.json` を素のまま読む。

    ⚠ `claude_web.load_findings()` は対象日と違えば None を返す（古い見解が
      今日の画面に出る事故を防ぐため）。台帳は**その日の記録を残すのが目的**なので、
      ファイル自身が名乗る `date` を使う。
    """
    path = os.path.join(base_dir or '.', FINDINGS_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get('races'), dict):
        return None
    return data


def collect_from_findings(findings):
    """findings から台帳エントリ（結果は未記入）を組み立てる。

    1レースから出るのは最大で「going 1件 ＋ 馬メモ n件 ＋ 印 m件」。
    """
    if not findings:
        return []
    collected_at = findings.get('collected_at')
    out = []
    for race_id, e in sorted((findings.get('races') or {}).items()):
        if not isinstance(e, dict):
            continue
        date = _race_date(race_id)
        if date is None:
            continue
        base = {'date': date, 'race_id': race_id,
                'venue': e.get('venue'), 'race_num': e.get('race_num'),
                'collected_at': collected_at,
                'race_sources': list(e.get('sources') or [])}

        going = e.get('going') if isinstance(e.get('going'), dict) else None
        if going and (going.get('state') or going.get('expected')):
            out.append(dict(base, kind='going',
                            key=str(going.get('surface') or '?'),
                            said={'surface': going.get('surface'),
                                  # 収集時点で発表されていた馬場（予想ではない）
                                  'state': going.get('state'),
                                  # 予想（自由文）
                                  'expected': going.get('expected'),
                                  'expected_state': expected_state(
                                      going.get('expected')),
                                  'summary': e.get('summary')},
                            sources=list(e.get('sources') or []),
                            hosts=_hosts(e.get('sources')),
                            result=None))

        for num, v in sorted((e.get('horses') or {}).items(),
                             key=lambda kv: str(kv[0])):
            if not isinstance(v, dict) or not str(v.get('note') or '').strip():
                continue
            srcs = list(v.get('sources') or [])
            out.append(dict(base, kind='horse', key=str(num),
                            said={'note': str(v.get('note')).strip(),
                                  'direction': v.get('direction'),
                                  # 情報源が付けた序列。無ければ None のまま
                                  'grade': normalize_grade(v.get('grade'))},
                            sources=srcs, hosts=_hosts(srcs), result=None))

        for num, mk in sorted((e.get('marks') or {}).items(),
                              key=lambda kv: str(kv[0])):
            if not mk:
                continue
            out.append(dict(base, kind='mark', key=str(num),
                            said={'mark': mk},
                            sources=list(e.get('sources') or []),
                            hosts=_hosts(e.get('sources')), result=None))
    return out


def record(base_dir, findings=None, now=None):
    """findings を台帳に追記する。既にあるキーは**触らない**。"""
    if findings is None:
        findings = load_findings_raw(base_dir)
    ledger = load(base_dir)
    have = {_key(x) for x in ledger['entries']}
    added = [e for e in collect_from_findings(findings) if _key(e) not in have]
    if added:
        ledger['entries'].extend(added)
        save(base_dir, ledger, now=now)
    return {'added': len(added), 'total': len(ledger['entries'])}


# ───────────────────────── 結果の追記 ─────────────────────────

def _track_condition(hist_conn, race_id):
    if hist_conn is None:
        return None
    row = hist_conn.execute(
        'SELECT track_condition, surface FROM race_history WHERE race_id = ?',
        (race_id,)).fetchone()
    if not row:
        return None
    return {'track_condition': row[0], 'surface': row[1]}


def _horse_fact(hist_conn, race_id, num):
    if hist_conn is None:
        return None
    row = hist_conn.execute(
        'SELECT place, popularity, horse_name FROM horse_history '
        'WHERE race_id = ? AND horse_num = ?', (race_id, int(num))).fetchone()
    if not row or row[0] is None:
        return None
    return {'place': int(row[0]),
            'popularity': int(row[1]) if row[1] and 0 < row[1] < 99 else None,
            'horse_name': row[2]}


def build_result(entry, hist_conn, keiba_conn, payouts, now=None):
    """1エントリぶんの結果を組み立てる。確定していなければ None（保留）。"""
    from . import results as _results

    rid = entry.get('race_id')
    now = now or datetime.now(JST)
    common = {'checked_at': now.isoformat(), 'source': RESULT_SOURCE}

    if entry.get('kind') == 'going':
        tc = _track_condition(hist_conn, rid)
        if not tc or not tc.get('track_condition'):
            return None
        said = entry.get('said') or {}
        at_collect = said.get('state')
        exp = said.get('expected_state') or expected_state(said.get('expected'))
        actual = tc['track_condition']
        return dict(
            common,
            actual_track_condition=actual,
            actual_surface=tc['surface'],
            # 🔴 2つを混ぜない。前者は「馬場が動かなかったか」で**予想の精度ではない**
            said_state_at_collection=at_collect,
            collection_state_matches_actual=(
                (at_collect == actual) if at_collect else None),
            said_expected_state=exp,
            expected_state_matches_actual=((exp == actual) if exp else None),
        )

    if payouts is None:
        # 🔴 払戻が1行も無い＝まだ結果を取っていない。空で埋めない
        return None
    fact = _horse_fact(hist_conn, rid, entry.get('key'))
    if fact is None:
        return None
    place = fact['place']
    pay = payouts.get(('fukusho', str(entry.get('key'))))
    if pay is None:
        pay = payouts.get(('fukusho', int(entry.get('key'))))
    return dict(common, place=place, top3=bool(1 <= place <= 3),
                popularity=fact['popularity'], horse_name=fact['horse_name'],
                # 🔑 的中と払戻は払戻表を正とする（着順から自分で判定しない。
                #    少頭数のレースは複勝が2着までしか売られない。2026-10-02）
                fukusho_payout=(int(pay) if pay is not None else None),
                fukusho_sold=bool(pay is not None or not (1 <= place <= 3)))


def attach_results(base_dir, hist_conn, keiba_conn, now=None, dry_run=False):
    """確定した事実を台帳に追記する。何度呼んでも二重処理しない。"""
    from . import results as _results

    now = now or datetime.now(JST)
    ledger = load(base_dir)
    out = {'attached': 0, 'already': 0, 'pending': 0, 'pending_races': []}
    pay_cache = {}
    changed = False

    for entry in ledger['entries']:
        if entry.get('result') is not None:
            out['already'] += 1
            continue
        rid = entry.get('race_id')
        if rid not in pay_cache:
            pay_cache[rid] = _results.load_payouts(keiba_conn, rid)
        res = build_result(entry, hist_conn, keiba_conn, pay_cache[rid], now=now)
        if res is None:
            out['pending'] += 1
            if rid not in out['pending_races']:
                out['pending_races'].append(rid)
            continue
        if not dry_run:
            entry['result'] = res
            changed = True
        out['attached'] += 1

    if changed and not dry_run:
        save(base_dir, ledger, now=now)
    return out


# ───────────────────────── 集計（事実だけ） ─────────────────────────

def summarize(base_dir):
    """情報源 × 次元ごとの事実を返す。🔴 回収率は出さない。

    🔴 **率をそのまま信頼度として読まないこと。** 1レースのほぼ全馬にメモが付く
    情報源（`mentions_per_race` の中央値が出走頭数に近い）では、3着内率は母集団の
    3着内率そのもので当たり外れの概念が無い。信頼度を測れる形になっているかは
    `with_grade` / `with_direction` / `mentions_per_race` の3つで見る
    （基準は `websig/CRITERIA_RELIABILITY.md`）。
    """
    ledger = load(base_dir)
    entries = ledger.get('entries') or []
    by_host = {}
    for e in entries:
        done = e.get('result') is not None
        said = e.get('said') or {}
        for h in (e.get('hosts') or ['(出典なし)']):
            d = by_host.setdefault(h, {})
            k = d.setdefault(e.get('kind'), {
                'n': 0, 'with_result': 0, 'top3': 0, 'top3_of': 0,
                'stable': 0, 'stable_of': 0,
                'expected_match': 0, 'expected_of': 0,
                # 🔑 測れる形になっているかの進捗メーター（効果ではない）
                'with_grade': 0, 'with_direction': 0, 'n_races': 0,
                'mentions_per_race': None, '_per_race': {}})
            k['n'] += 1
            if e.get('kind') != 'going':
                if said.get('grade'):
                    k['with_grade'] += 1
                if said.get('direction') in ('up', 'down'):
                    k['with_direction'] += 1
            rid = e.get('race_id')
            if rid:
                k['_per_race'][rid] = k['_per_race'].get(rid, 0) + 1
            if not done:
                continue
            k['with_result'] += 1
            r = e['result']
            if e.get('kind') == 'going':
                # 収集時点の馬場 == 確定馬場（＝馬場が動かなかった）。予想の精度ではない
                if r.get('collection_state_matches_actual') is not None:
                    k['stable_of'] += 1
                    k['stable'] += 1 if r['collection_state_matches_actual'] else 0
                # 予想が状態名を名乗ったときだけ採点できる
                if r.get('expected_state_matches_actual') is not None:
                    k['expected_of'] += 1
                    k['expected_match'] += 1 if r['expected_state_matches_actual'] else 0
            else:
                if r.get('top3') is not None:
                    k['top3_of'] += 1
                    k['top3'] += 1 if r['top3'] else 0
    for kinds in by_host.values():
        for k in kinds.values():
            per = k.pop('_per_race')
            k['n_races'] = len(per)
            k['mentions_per_race'] = _median(sorted(per.values())) if per else None
    return {
        'n_entries': len(entries),
        'n_with_result': sum(1 for e in entries if e.get('result') is not None),
        'dates': sorted({str(e.get('date')) for e in entries}),
        'by_host': by_host,
        # 🔴 N が小さいので率を読まないこと（North Star #7）。
        'caveat': ('週2日・10R/11R だけなので N は年間100件程度しか増えない。'
                   'これ単独では「どの情報源が有効か」の答えにならない。'
                   '回収率は出していない（事実だけ）。'
                   '🔴 going の stable は「収集時点の馬場と確定馬場が同じだったか」で、'
                   '**予想の精度ではない**（予想は expected 側で、'
                   '状態名を名乗らないことが多い）。'
                   '🔴 1レースのほぼ全馬にメモが付く情報源では 3着内率は母集団の'
                   '3着内率そのもので、**信頼度ではない**'
                   '（mentions_per_race を必ず並べて読む）'),
    }


def _median(values):
    """ソート済みの数列の中央値。偶数個なら2つの平均。"""
    if not values:
        return None
    n = len(values)
    mid = n // 2
    if n % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0
