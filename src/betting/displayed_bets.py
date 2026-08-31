"""アプリ画面に実際に出した買い目（gumbel_bets = 軸1頭ベース）をDBに残す。

なぜ必要か（2026-08-31 の棚卸しで判明した穴）:
    `bets` テーブルに保存されるのは旧 `make_bets()` の出力で、
    **画面に出ている買い目とは別物**。実例（2026-08-30 新潟R3）:

        画面 : 単勝1点 + ワイド3点 + 馬連3点 + 三連複11点 = 18点 ¥1,800
        DB   : 複勝1点 ¥500

    さらに `bets` / `bet_simulation` は `select_quality_races` を通った
    推奨レース（1日6R）だけが対象で、残り約29Rの買い目はどこにも残らない。
    そのため「全レースの買い目と結果」を後から集計する手段が
    `data/latest.json` のgit履歴を掘ることしか無かった（24日/802レース）。
    latest.json は同じ日に何度も上書きされるので、どの世代が朝の予想かも
    特定できない。

    ここで保存するのは *表示したそのもの* で、成績評価の一次資料になる。
    買い目ロジックを差し替えたら、この記録も自動的に新しい方に追従する。

券種名は `race_dividends` と同じ表記（tansho/wide/umaren/sanrenpuku）で
持つ。組番も昇順ハイフン連結に正規化するので、決済は素直なJOINで済む。
"""

from __future__ import annotations


def _amount(raw):
    """'¥1,100' → 1100。取れなければ 0。"""
    try:
        return int(str(raw or '0').replace('¥', '').replace(',', '').strip() or 0)
    except ValueError:
        return 0


def _combo(nums):
    return '-'.join(str(n) for n in sorted(int(x) for x in nums))


def extract_tickets(gumbel_bets):
    """`_format_axis_bets` の出力を (bet_type, combo, amount) の点に分解する。

    1行が複数点を表す場合（ワイド3点・三連複11点など）は金額を等分する。
    画面の 'amt' はその行の合計額なので、点あたりに割り戻さないと
    投資額を過大に数えることになる。
    """
    out = []
    for b in gumbel_bets or []:
        tag = b.get('tag')
        amt = _amount(b.get('amt'))
        tickets = []

        if tag == 'tan':
            head = str(b.get('horse', '')).split()
            n = head[0].lstrip('#') if head else ''
            if n.isdigit():
                tickets = [('tansho', _combo([n]))]

        elif tag == 'fuku':
            head = str(b.get('horse', '')).split()
            n = head[0].lstrip('#') if head else ''
            if n.isdigit():
                tickets = [('fukusho', _combo([n]))]

        elif tag in ('wide', 'umaren'):
            axis = b.get('axis')
            mates = [m.get('n') for m in (b.get('mates') or []) if m.get('n')]
            if axis and mates:
                bt = 'wide' if tag == 'wide' else 'umaren'
                tickets = [(bt, _combo([axis, m])) for m in mates]

        elif tag == 'sanfuku':
            for cb in (b.get('combos') or []):
                try:
                    tickets.append(('sanrenpuku', _combo(cb.split('-'))))
                except ValueError:
                    continue

        if not tickets:
            continue
        per = amt / len(tickets)
        for bt, cb in tickets:
            out.append((bt, cb, per))
    return out


def rows_from_app_json(app_json, snapshot='initial'):
    """to_app_json の戻り値から displayed_bets の行を作る。

    表示用JSONを唯一の入力にしているので、画面に出ていないものは
    絶対に記録されないし、逆に画面に出たものは必ず記録される。
    """
    raw = app_json.get('races') or {}
    if isinstance(raw, dict):
        # races は {競馬場名: [レース...]} 形式。競馬場はキー側にしか無い
        races = [(venue, r) for venue, v in raw.items() for r in v]
    else:
        races = [(r.get('racecourse', ''), r) for r in raw]

    rows = []
    for venue, r in races:
        rid = r.get('race_id')
        if not rid:
            continue
        d = f'{rid[:4]}-{rid[4:6]}-{rid[6:8]}'
        for bt, combo, amt in extract_tickets(r.get('gumbel_bets')):
            rows.append({
                'date': d,
                'race_id': rid,
                'racecourse': venue,
                'race_num': r.get('r', 0),
                'is_recommended': 1 if r.get('rec') else 0,
                'snapshot': snapshot,
                'bet_type': bt,
                'combo': combo,
                'amount': round(amt, 1),
            })
    return rows
