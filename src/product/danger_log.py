"""「危険な人気馬」の警告と、その後の着順を台帳に残す層。

＝＝ なぜ必要か ＝＝

商品の2本柱の片方が「危険な人気馬」（市場1番人気のうち素のAIが下位＝複勝率
68.5%→51.4%・N=242・walk-forward 5窓・2026-08-29）。ところが **その警告が当たったか
外れたかを残す仕組みが1つも無かった**。2026-10-10〜12 の初運用で「4戦4敗（4頭すべて
3着内）」と報告したのは、そのとき私が `latest.json` の git 履歴と `race_results.json` を
手で突き合わせて作った数字で、**次の開催日には同じ手作業をやり直さないと残らない**。

`prediction_archive/`（`src/product/results.py`）は同じ情報を持つが、ユーザー判断
（2026-10-05）で `--publish` を使わない＝**実データ0件**なので、そこには溜まらない。
この台帳はそれとは独立に `data/danger_favorites.json` へ追記する。

＝＝ 設計上の約束 ＝＝

🔴 **警告の条件を再実装しない。** 判定は `marks.danger_favorites()` を**そのまま呼ぶ**。
   条件（市場1番人気 × 素のAI順位>=6）が2箇所に分かれると、片方だけ動いて
   「記事に出した警告」と「台帳の警告」が静かに食い違う。

🔴 **本番の予想を1バイトも変えない。** 読むのは `data/latest.json` と
   `data/history.db`（着順）と `data/keiba.db`（払戻・発走時刻）だけ。

🔴 **一度記録した警告は上書きしない**（`(race_id, 馬番)` で先着順）。
   当日 11:30/14:00 の refresh でオッズが動いて人気が入れ替わっても、
   朝に出した警告はそのまま残す（記事に載せたものが記録として残らなければ意味がない）。

🔴 **回収率は出さない。** 保存するのは着順・3着内か・複勝払戻（100円あたり）という
   事実だけ。理由は `results.py` の冒頭と同じ（記事の金額は換算例なので、回収率には
   賭け金の仮定が必要になり、仮定混じりの数字を「結果」として残せない）。

🔴 **後出しの警告を事実として混ぜない。** refresh は当日何度も走るので、
   既に発走したレースの馬が後から条件を満たすことがある。結果追記のときに
   `race_schedule.post_time` と記録時刻を比べ、`before_post` を必ず立てる
   （不明なら None）。集計では `before_post is False` を除くこと。

⚠ **これは測定の器であって、成績ではない。** N が小さいうちの偏りを読まない
   （North Star #7）。「4戦4敗」も N=4 で何の証拠にもならない。
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from . import marks as _marks
from . import results as _results
from . import scope as _scope
from . import tickets as _tickets

JST = timezone(timedelta(hours=9))

SCHEMA_VERSION = 1
LEDGER_FILENAME = os.path.join('data', 'danger_favorites.json')

RESULT_SOURCE = ('history.db:horse_history.place（着順） / '
                 'keiba.db:race_dividends（複勝払戻） / '
                 'keiba.db:race_schedule（発走時刻）')


def ledger_path(base_dir):
    return os.path.join(base_dir or '.', LEDGER_FILENAME)


def empty_ledger():
    return {'schema_version': SCHEMA_VERSION, 'updated_at': None, 'entries': []}


def load(base_dir):
    """台帳を読む。無ければ空の台帳。版が違えば例外（黙って作り直さない）。"""
    path = ledger_path(base_dir)
    if not os.path.exists(path):
        return empty_ledger()
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    ver = data.get('schema_version')
    if ver != SCHEMA_VERSION:
        raise ValueError(
            f'{LEDGER_FILENAME} の schema_version={ver} は未対応'
            f'（期待 {SCHEMA_VERSION}）。作り直すと過去の警告が消えるので、'
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


def _race_date(race):
    """race_id の先頭8桁から 'YYYY-MM-DD'。取れなければ None。

    🔴 `latest.json` の `date` は `10月12日(月)` という表示用の文字列（年が無い）で、
       `generated_at` は前夜のことがある。日付は race_id だけが信用できる
       （2026-10-05 に `build_product_article._target_date` で同じ結論）。
    """
    rid = str(race.get('race_id') or '')
    if len(rid) >= 8 and rid[:8].isdigit():
        return f'{rid[0:4]}-{rid[4:6]}-{rid[6:8]}'
    return None


def collect_from_latest(data, now=None):
    """`latest.json` の中身から、警告の台帳エントリを組み立てて返す（結果は未記入）。

    判定は `marks.danger_favorites()`（記事と同じ関数）。
    ⚠ **対象範囲（9R〜11R＋自信のある平場）では絞らない。** 記事に出るのは
      範囲内だけだが、測定したい母集団は「市場1番人気×素のAI下位」そのもので、
      2026-08-29 の実測（N=242）も全レースで測っている。範囲で絞ると N が
      1/3 になるので、`in_scope` を立てて**両方読めるように**しておく。
    """
    now = now or datetime.now(JST)
    snapshot = data.get('generated_at')
    coverage = (data.get('data_quality') or {}).get('odds_coverage')
    races = data.get('races') or {}
    pairs = ([(v, r) for v, rs in races.items() for r in rs]
             if isinstance(races, dict)
             else [((r or {}).get('venue') or '', r) for r in races])

    out = []
    for venue, race in pairs:
        race = race or {}
        horses = race.get('horses') or []
        if not horses:
            continue
        date_str = _race_date(race)
        if date_str is None:
            # 日付が決まらないものは記録しない（いつの警告か言えない記録は無価値）
            continue
        in_scope, why = _scope.in_scope(race)
        for d in _marks.danger_favorites(horses):
            if d.get('num') is None:
                continue
            out.append({
                'date': date_str,
                'race_id': race.get('race_id'),
                'venue': venue,
                'race_num': race.get('r'),
                'race_name': race.get('name'),
                'num': int(d['num']),
                'name': d.get('name'),
                'pop': d.get('pop'),
                'solo_rank': d.get('solo_rank'),
                'rl_rank': d.get('rl_rank'),
                'odds': d.get('odds'),
                'conf': race.get('conf'),
                'verdict': _tickets.verdict(race.get('conf')),
                'in_scope': in_scope,
                'scope_reason': why,
                'snapshot': snapshot,
                'odds_coverage': coverage,
                'recorded_at': now.isoformat(),
                'result': None,
            })
    return out


def record(base_dir, data, now=None, backfill_source=None):
    """`latest.json` の警告を台帳に追記する。先着順で、既存は上書きしない。

    Returns:
        dict: {'added': n, 'already': n, 'entries': [追記したもの]}
    """
    ledger = load(base_dir)
    seen = {_key(e) for e in ledger['entries']}
    added = []
    for entry in collect_from_latest(data, now=now):
        if _key(entry) in seen:
            # 🔴 上書きしない。朝に出した警告をそのまま残す
            continue
        if backfill_source:
            entry['backfilled'] = True
            entry['backfill_source'] = backfill_source
        ledger['entries'].append(entry)
        seen.add(_key(entry))
        added.append(entry)
    if added:
        save(base_dir, ledger, now=now)
    return {'added': len(added), 'already': len(seen) - len(added),
            'entries': added}


def _post_time(keiba_conn, race_id):
    if keiba_conn is None:
        return None
    try:
        row = keiba_conn.execute(
            'SELECT post_time FROM race_schedule WHERE race_id = ?',
            (race_id,)).fetchone()
    except Exception:
        # race_schedule が無い環境（古い keiba.db）。不明として扱う
        return None
    return row[0] if row and row[0] else None


def before_post(snapshot_iso, date_str, post_time):
    """その警告が発走前に記録されたものか。判断できなければ None。

    `post_time` は `race_schedule` の 'HH:MM'（JST）。この表は
    `collect_odds.py` が土日に書き、`merge_odds_ts.py` が夜に取り込むので、
    **朝の記録時点では埋まっていないことがある**。だから記録時に弾くのではなく、
    結果追記のときに後から判定する。
    """
    if not snapshot_iso or not date_str or not post_time:
        return None
    try:
        snap = datetime.fromisoformat(str(snapshot_iso))
        hh, mm = str(post_time).split(':')[:2]
        post = datetime.fromisoformat(f'{date_str}T00:00:00+09:00').replace(
            hour=int(hh), minute=int(mm))
    except (TypeError, ValueError):
        return None
    if snap.tzinfo is None:
        snap = snap.replace(tzinfo=JST)
    return snap < post


def build_result(entry, placings, payouts, post_time, now=None):
    """1頭ぶんの結果を組み立てる。着順が取れないときは None（保留）。

    🔑 **`top3` は着順（`history.db`）から決める。** 商品の主張が
       「複勝率 68.5%→51.4%」＝3着内率なので、見るべきは着順そのもの。
    🔑 一方 **複勝払戻は払戻表に行があるかどうかで決める**（`results.py` と同じ規約）。
       少頭数で複勝が2着までしか発売されないレースが実在するため、
       「3着内なのに複勝払戻が無い」は正常な状態。だから `fukusho_payout is None` を
       「外れ」と読まないよう、`fukusho_sold`（そのレースで複勝が発売されたか）も残す。
    """
    num = int(entry['num'])
    place = placings.get(num)
    if place is None:
        return None
    fukusho = {c: p for (bt, c), p in (payouts or {}).items() if bt == 'fukusho'}
    payout = fukusho.get(str(num))
    return {
        'place': int(place),
        'top3': int(place) <= 3,
        'n_finishers': len(placings),
        'fukusho_payout': payout,
        'fukusho_sold': bool(fukusho),
        'post_time': post_time,
        'before_post': before_post(entry.get('snapshot'), entry.get('date'),
                                   post_time),
        'attached_at': (now or datetime.now(JST)).isoformat(),
        'source': RESULT_SOURCE,
    }


def attach_results(base_dir, hist_conn, keiba_conn, now=None, dry_run=False):
    """確定した着順・複勝払戻を台帳に追記する。何度呼んでも二重処理しない。

    Returns:
        dict: {'attached', 'already', 'pending', 'pending_races', 'late'}
    """
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
        payouts = payouts_cache[rid]
        if payouts is None:
            # 🔴 払戻が1行も無い＝まだ結果を取っていない。空の結果で埋めない
            out['pending'] += 1
            out['pending_races'].append(rid)
            continue
        result = build_result(entry, placings_cache[rid], payouts,
                              _post_time(keiba_conn, rid), now=now)
        if result is None:
            # 払戻はあるのに着順が無い（history.db にそのレースが無い等）。
            # 推測で埋めずに保留する（10/10 京都 R2/R4/R7/R12 が実例）
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


def summarize(base_dir, scope_only=False, include_late=False):
    """台帳の集計（事実だけ）。

    🔴 回収率は出さない。出すのは件数と3着内の数、複勝払戻の一覧まで。
       賭け金の仮定を混ぜた数字をここで作らない（`results.py` 冒頭と同じ理由）。
    """
    ledger = load(base_dir)
    entries = ledger.get('entries') or []
    if scope_only:
        entries = [e for e in entries if e.get('in_scope')]
    done, late, pending = [], [], []
    for e in entries:
        r = e.get('result')
        if r is None:
            pending.append(e)
        elif r.get('before_post') is False and not include_late:
            late.append(e)
        else:
            done.append(e)
    top3 = [e for e in done if (e['result'] or {}).get('top3')]
    payouts = sorted(p for p in ((e['result'] or {}).get('fukusho_payout')
                                 for e in top3) if p)
    # 🔴 ◎ 自身かどうかで分ける（2026-10-15）。実測で 11件のうち9件が
    #    `rl_rank == 1` ＝ その記事が軸に推している馬だった。
    #    `danger_favorites()` は `rl_rank` を見ないので構造的にこうなる。
    #    毎回この内訳が見えないと、同じことに次も気づけない。
    axis = [e for e in done if e.get('rl_rank') == 1]
    other = [e for e in done if e.get('rl_rank') != 1]
    _t3 = lambda g: sum(1 for e in g if (e['result'] or {}).get('top3'))
    return {
        'total': len(entries),
        'with_result': len(done),
        'pending': len(pending),
        'excluded_late': len(late),
        'top3': len(top3),
        'places': sorted((e['result'] or {}).get('place') for e in done),
        'fukusho_payouts': payouts,
        'dates': sorted({str(e.get('date')) for e in entries}),
        'axis_n': len(axis),
        'axis_top3': _t3(axis),
        'other_n': len(other),
        'other_top3': _t3(other),
    }
