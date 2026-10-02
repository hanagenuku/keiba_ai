"""公開済みの予想に、確定した結果を追記する（§12 の「結果」）。

🔴 **記録であって評価ではない。回収率はここで出さない。**
   記事は金額を「換算例（推奨額ではありません）」としか書いていない（§10 資金配分は
   各自調整）。したがって回収率を出すには賭け金の仮定を外から与えるしかなく、
   仮定を混ぜた数字を「結果」として永久保存すると後から前提を変えられない
   （アーカイブは上書き禁止）。保存するのは **着順と払戻（100円あたり）という事実だけ**。

🔴 **的中と払戻は `keiba.db` の `race_dividends` を正とする。着順から自分で判定しない。**
   実データで確認した反例がある: 2026-09-27 中山9R は7頭立てで、複勝は2着までしか
   発売されていない（3着の馬番2 に複勝配当は無い）。一方**ワイドは同じレースでも
   3着内の2頭組で成立している**（2-3 / 2-7 / 3-7）。「頭数が少なければ複勝は2着まで」
   のような規則を手で書くと、ワイドまで同じ規則で潰して誤判定する。
   **払戻表に行があるかどうか**が唯一の正しい判定。
   ⚠ `horse_history.fukusho_payout` は使わない。3着内なのに 0/NULL の行が 1.73%
   （36,428行中629行）あり、その一部はこの「少頭数で複勝が無い」ケースそのもので、
   スクレイプ漏れと区別できない。

🔑 着順は `history.db` の `horse_history.place` から取る（払戻とは**別経路**）。
   両者を突き合わせる検算を1つ入れてある（ワイド3組の和集合 == 3着内）。
   同じ数字を2経路で作って突合する、という North Star #8 の手順をそのまま適用した。
"""

import os
import sqlite3

from . import archive

# 記事の券種表記 → race_dividends の bet_type
_BET_TYPE_MAP = {'複勝': 'fukusho', 'ワイド': 'wide'}


def open_history(base_dir):
    """history.db を読み取り専用で開く。実体が無ければ None。"""
    return _open(os.path.join(base_dir, 'data', 'history.db'))


def open_keiba(base_dir):
    """keiba.db を読み取り専用で開く。実体が無ければ None。"""
    return _open(os.path.join(base_dir, 'data', 'keiba.db'))


def _open(path):
    # LFS ポインタ（133バイトのテキスト）だと sqlite3 が開けない。
    # 握りつぶすと「結果が無い」と「DBが無い」が混ざるので、呼び出し側で
    # None を明示的に扱う（commentary.py の n_past_runs と同じ扱い）。
    if not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        conn.execute('SELECT 1').fetchone()
        return conn
    except sqlite3.DatabaseError:
        return None


def combo_key(nums):
    """race_dividends の combo 表記（昇順ハイフン連結）に正規化する。

    複勝・ワイドはどちらも順不同なので昇順でよい
    （`src/utils/db.py::_dividend_rows` と同じ規約）。
    """
    return '-'.join(str(int(n)) for n in sorted(int(x) for x in nums))


def load_placings(hist_conn, race_id):
    """{馬番: 着順} を返す。行が無ければ空dict。"""
    if hist_conn is None:
        return {}
    rows = hist_conn.execute(
        'SELECT horse_num, place FROM horse_history WHERE race_id = ?',
        (race_id,)).fetchall()
    return {int(n): int(p) for n, p in rows
            if n is not None and p is not None}


def load_payouts(keiba_conn, race_id):
    """{(bet_type, combo): 払戻} を返す。そのレースの払戻が未取得なら None。

    🔑 空dict ではなく None を返す。「払戻を取得済みで、この組は外れ」と
    「まだ結果が取れていない」は別物で、後者で結果を追記すると全部外れとして
    記録されてしまう。
    """
    if keiba_conn is None:
        return None
    rows = keiba_conn.execute(
        'SELECT bet_type, combo, payout FROM race_dividends WHERE race_id = ?',
        (race_id,)).fetchall()
    if not rows:
        return None
    return {(bt, combo): payout for bt, combo, payout in rows}


def _top3(placings):
    return [n for n, p in sorted(placings.items(), key=lambda kv: kv[1])
            if p <= 3]


def _cross_check(placings, payouts):
    """ワイド3組の和集合が3着内と一致するか（着順と払戻の別経路突合）。"""
    wide = [k[1] for k in payouts if k[0] == 'wide']
    if not wide or not placings:
        return 'unavailable'
    union = set()
    for combo in wide:
        union |= {int(x) for x in combo.split('-')}
    return 'ok' if union == set(_top3(placings)) else 'mismatch'


def build_result(snap, placings, payouts):
    """保存する `result` を組み立てる。回収率は含めない（事実だけ）。"""
    top3 = _top3(placings)
    bets = []
    for b in snap.get('bets') or []:
        bt = _BET_TYPE_MAP.get(b.get('type'))
        for combo in b.get('combos') or []:
            entry = {'type': b.get('type'), 'combo': [int(x) for x in combo]}
            if bt is None:
                # 券種が増えたら黙って「外れ」にせず、判定しないと明示する
                entry.update(hit=None, payout_per_100=None, payout_known=False,
                             note='この券種は結果判定に未対応')
            else:
                payout = payouts.get((bt, combo_key(combo)), '__miss__')
                hit = payout != '__miss__'
                entry.update(
                    hit=hit,
                    payout_per_100=payout if hit else None,
                    payout_known=bool(hit and payout is not None),
                )
            bets.append(entry)

    marks = [{'mark': r.get('mark'), 'num': r.get('num'),
              'place': placings.get(r.get('num'))}
             for r in (snap.get('ranks') or []) if r.get('mark')]
    dangers = [{'num': d.get('num'), 'place': placings.get(d.get('num'))}
               for d in (snap.get('danger_favorites') or [])]

    return {
        'source': 'history.db:horse_history.place（着順） / '
                  'keiba.db:race_dividends（的中・払戻）',
        'n_finishers': len(placings),
        'top3': top3,
        'placings': {str(n): p for n, p in sorted(placings.items())},
        'marks': marks,
        'danger_favorites': dangers,
        'bets': bets,
        'cross_check': _cross_check(placings, payouts),
    }


def attach_for_date(base_dir, date_str, hist_conn, keiba_conn, dry_run=False):
    """その日の公開済み予想に結果を追記する。何度呼んでも二重処理しない。

    Returns:
        dict: {'attached': n, 'already': n, 'pending': n, 'mismatch': [race_id]}
    """
    out = {'attached': 0, 'already': 0, 'pending': 0, 'mismatch': []}
    for race_id in archive.list_published(base_dir, date_str):
        snap = archive.load_snapshot(base_dir, date_str, race_id)
        if snap is None:
            continue
        if snap.get('result') is not None:
            out['already'] += 1
            continue
        payouts = load_payouts(keiba_conn, race_id)
        if payouts is None:
            # 結果がまだ取れていない。次回の実行に持ち越す（空のまま埋めない）
            out['pending'] += 1
            continue
        placings = load_placings(hist_conn, race_id)
        result = build_result(snap, placings, payouts)
        if result['cross_check'] == 'mismatch':
            out['mismatch'].append(race_id)
        if not dry_run:
            archive.attach_result(base_dir, date_str, race_id, result)
        out['attached'] += 1
    return out


def published_dates(base_dir):
    d = os.path.join(base_dir, archive.ARCHIVE_DIRNAME)
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d)
                  if os.path.isdir(os.path.join(d, x)))
