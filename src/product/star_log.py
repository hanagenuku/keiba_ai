"""☆（市場1番人気を着順で上回る可能性）と、その後の着順を台帳に残す層。

＝＝ なぜ必要か ＝＝

☆は 2026-11-11 から本番に出ているが、**予想時点の確率を残す仕組みが無いと
「☆は当たっているのか」に後から答えられない**。`latest.json` は上書きされるので、
朝に何番に☆を付けて何%と言ったかは残らない。`danger_log.py` と同じ理由・同じ形。

＝＝ 設計上の約束（`danger_log.py` と同じ4つ）＝＝

🔴 **判定条件を再実装しない。** ☆の決定は `app_json._build_star()` が
   `latest.json` に書いた `race.star` を**そのまま読む**。閾値 `STAR_TAU` も
   `src/features/star.py` の1か所だけ（ここには書かない）。

🔴 **本番の予想を1バイトも変えない。** 読むのは `data/latest.json` と
   `data/history.db`（着順）と `data/keiba.db`（払戻・発走時刻）だけ。

🔴 **一度記録した☆は上書きしない**（`(race_id, 馬番)` で先着順）。
   当日 11:30/14:00 の refresh でオッズが動いて1番人気が入れ替わると☆も動くが、
   **朝に出した☆はそのまま残す**（記事に載せたものが残らなければ意味がない）。
   ⚠ そのため同じレースに複数の☆が並ぶことがある。集計は
   **レースごとに最初のスナップショットの☆だけ**を数える（`summarize`）。

🔴 **回収率は出さない。** 残すのは着順・3着内か・複勝払戻という事実だけ。
   ☆の主張は着順比較（1番人気を上回るか）なので、1番人気の着順も一緒に残す。

⚠ **これは測定の器であって、成績ではない。** N が小さいうちの偏りを読まない
   （North Star #7）。**台帳が溜まるまで「☆の成績」を名乗らない**
   （`biz/S4_HITRATE_AXIS.md` §5。提案: 最低15開催日）。
"""
import json
import os
import tempfile
from datetime import datetime

from src.product import results as _results
from src.product.danger_log import JST, _post_time, _race_date, before_post

LEDGER_FILENAME = os.path.join('data', 'star_log.json')
SCHEMA_VERSION = 1
RESULT_SOURCE = 'history.db+keiba.db'


def ledger_path(base_dir):
    return os.path.join(base_dir or '.', LEDGER_FILENAME)


def empty_ledger():
    return {'schema_version': SCHEMA_VERSION, 'updated_at': None, 'entries': []}


def load(base_dir):
    """台帳を読む。無ければ空。版が違えば例外（黙って作り直さない）。"""
    path = ledger_path(base_dir)
    if not os.path.exists(path):
        return empty_ledger()
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    ver = data.get('schema_version')
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f'{LEDGER_FILENAME} の schema_version={ver} は未対応'
            f'（期待 {SCHEMA_VERSION}）。作り直すと過去の☆が消えるので、'
            '移行コードを書いてから読むこと')
    data.setdefault('entries', [])
    return data


def save(base_dir, ledger, now=None):
    """台帳を書く。途中で落ちても壊れないよう一時ファイル経由で置き換える。"""
    ledger['schema_version'] = SCHEMA_VERSION
    ledger['updated_at'] = (now or datetime.now(JST)).isoformat()
    ledger['entries'] = sorted(ledger.get('entries') or [],
                               key=lambda e: (str(e.get('date')),
                                              str(e.get('race_id')),
                                              int(e.get('num') or 0)))
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
    return (str(entry.get('race_id')), int(entry.get('num') or 0))


def _p1_num(race):
    """そのレースの市場1番人気の馬番（`latest.json` の `pop`）。無ければ None。"""
    hit = [h for h in (race.get('horses') or []) if (h.get('pop') or 0) == 1]
    return hit[0].get('n') if len(hit) == 1 else None


def collect_from_latest(data, now=None):
    """`latest.json` から☆の台帳エントリを組み立てて返す（結果は未記入）。

    ☆の決定は読まない・作らない。`race.star` を**そのまま**使う。
    """
    now = now or datetime.now(JST)
    snapshot = data.get('updated_at') or data.get('generated_at') or now.isoformat()
    out = []
    for venue, races in (data.get('races') or {}).items():
        for race in races or []:
            st = race.get('star') or None
            if not st or not st.get('n'):
                continue
            date = _race_date(race)
            if not date:
                continue
            num = int(st['n'])
            h = next((x for x in (race.get('horses') or [])
                      if x.get('n') == num), {})
            out.append({
                'date': date,
                'race_id': race.get('race_id'),
                'venue': venue,
                'race_num': race.get('r'),
                'num': num,
                'name': st.get('name') or h.get('name'),
                # 予想時点の較正済み確率と、そのときの閾値（後から τ を変えても残る）
                'prob': st.get('prob'),
                'tau': st.get('tau'),
                'pop': h.get('pop'),
                'odds': h.get('odds'),
                'rl_rank': h.get('rl_rank'),
                'solo_rank': h.get('solo_rank'),
                # ☆の主張は「1番人気を上回る」なので比較相手も残す
                'p1_num': _p1_num(race),
                'snapshot': snapshot,
                'result': None,
            })
    return out


def record(base_dir, data, now=None, backfill_source=None):
    """☆を台帳へ追記する。既にある `(race_id, 馬番)` は**上書きしない**。

    Returns:
        dict: {'added', 'already', 'total', 'entries'}（entries は今回追記した分）
    """
    now = now or datetime.now(JST)
    ledger = load(base_dir)
    have = {_key(e) for e in ledger['entries']}
    found = collect_from_latest(data, now=now)
    fresh = []
    for entry in found:
        if _key(entry) in have:
            continue
        if backfill_source:
            entry['backfilled'] = str(backfill_source)
        ledger['entries'].append(entry)
        have.add(_key(entry))
        fresh.append(entry)
    if fresh:
        save(base_dir, ledger, now=now)
    return {'added': len(fresh), 'already': len(found) - len(fresh),
            'total': len(ledger['entries']), 'entries': fresh}


def build_result(entry, placings, payouts, post_time, now=None):
    """1件ぶんの結果。着順が取れないときは None（保留）。

    ☆の主張は着順比較なので、**1番人気の着順も一緒に残す**。
    複勝払戻の扱いは `danger_log.build_result` と同じ規約
    （少頭数で複勝が2着までのレースがあるので `fukusho_sold` も残す）。
    """
    num = int(entry['num'])
    place = placings.get(num)
    if place is None:
        return None
    fukusho = {c: p for (bt, c), p in (payouts or {}).items() if bt == 'fukusho'}
    p1n = entry.get('p1_num')
    p1_place = placings.get(int(p1n)) if p1n else None
    return {
        'place': int(place),
        'top3': int(place) <= 3,
        'n_finishers': len(placings),
        'fukusho_payout': fukusho.get(str(num)),
        'fukusho_sold': bool(fukusho),
        # 1番人気の側（比較相手）
        'p1_place': int(p1_place) if p1_place is not None else None,
        'p1_top3': (int(p1_place) <= 3) if p1_place is not None else None,
        'p1_fukusho_payout': fukusho.get(str(p1n)) if p1n else None,
        # ☆の主張そのもの: 1番人気を着順で上回ったか
        'beat_p1': (int(place) < int(p1_place)) if p1_place is not None else None,
        'post_time': post_time,
        'before_post': before_post(entry.get('snapshot'), entry.get('date'),
                                   post_time),
        'attached_at': (now or datetime.now(JST)).isoformat(),
        'source': RESULT_SOURCE,
    }


def attach_results(base_dir, hist_conn, keiba_conn, now=None, dry_run=False):
    """確定した着順・複勝払戻を追記する。何度呼んでも二重処理しない。"""
    now = now or datetime.now(JST)
    ledger = load(base_dir)
    out = {'attached': 0, 'already': 0, 'pending': 0,
           'pending_races': [], 'late': []}
    placings_cache, payouts_cache = {}, {}
    changed = False

    for entry in ledger['entries']:
        if entry.get('result') is not None:
            out['already'] += 1
            continue
        rid = entry.get('race_id')
        if rid not in payouts_cache:
            payouts_cache[rid] = _results.load_payouts(keiba_conn, rid)
            placings_cache[rid] = _results.load_placings(hist_conn, rid)
        if payouts_cache[rid] is None:
            # 🔴 払戻が1行も無い＝まだ結果を取っていない。空の結果で埋めない
            out['pending'] += 1
            out['pending_races'].append(rid)
            continue
        result = build_result(entry, placings_cache[rid], payouts_cache[rid],
                              _post_time(keiba_conn, rid), now=now)
        if result is None:
            out['pending'] += 1
            out['pending_races'].append(rid)
            continue
        if result['before_post'] is False:
            out['late'].append(f"{entry.get('date')} {rid} #{entry['num']}")
        if not dry_run:
            entry['result'] = result
            changed = True
        out['attached'] += 1

    if changed and not dry_run:
        save(base_dir, ledger, now=now)
    return out


def first_per_race(entries):
    """レースごとに**最初のスナップショットの☆だけ**を返す。

    🔴 当日の refresh で☆が動くと同じレースに複数のエントリが残る（上書き禁止）。
    集計では朝に出した1件だけを数える（記事に載ったのがそれだから）。
    """
    best = {}
    for e in entries:
        rid = str(e.get('race_id'))
        cur = best.get(rid)
        if cur is None or str(e.get('snapshot') or '') < str(cur.get('snapshot') or ''):
            best[rid] = e
    return list(best.values())


def summarize(base_dir, include_late=False):
    """台帳の集計（事実だけ。回収率は出さない）。

    ⚠ **N が小さいうちは読まない。** 「☆の成績」を名乗るのは台帳が溜まってから
    （`biz/S4_HITRATE_AXIS.md` §5。提案: 最低15開催日）。
    """
    ledger = load(base_dir)
    rows = first_per_race(ledger['entries'])
    done = [e for e in rows if e.get('result')]
    if not include_late:
        done = [e for e in done if e['result'].get('before_post') is not False]
    n = len(done)
    beat = [e for e in done if e['result'].get('beat_p1') is True]
    top3 = [e for e in done if e['result'].get('top3')]
    p1top3 = [e for e in done if e['result'].get('p1_top3')]
    days = sorted({str(e.get('date')) for e in rows})
    return {
        'entries': len(ledger['entries']),
        'races': len(rows),
        'with_result': n,
        'pending': len(rows) - len([e for e in rows if e.get('result')]),
        'days': len(days),
        'date_from': days[0] if days else None,
        'date_to': days[-1] if days else None,
        # ☆の主張そのもの
        'beat_p1': len(beat),
        'beat_p1_rate': round(len(beat) / n * 100, 2) if n else None,
        # 参考（判定には使わない）
        'star_top3': len(top3),
        'star_top3_rate': round(len(top3) / n * 100, 2) if n else None,
        'p1_top3': len(p1top3),
        'p1_top3_rate': round(len(p1top3) / n * 100, 2) if n else None,
        'late_excluded': len([e for e in rows if e.get('result')
                              and e['result'].get('before_post') is False]),
    }
