"""商品として買い目を出す対象レースを決める。

ユーザー指示（2026-09-29）:
    「買い目は、各競馬場9R〜11R(OP以上）と自信のある平場レース」

🔴 これは**商品の対象範囲**の規則であって、買い目の選び方ではない。
   買い目そのもの（券種・配分・見送り）は `tickets.py` が決め、
   その閾値は本番の既存値（`ev_filter.MIN_AXIS_FUKU_PROB` / `rec`）に合わせてある。
   ここで新しい閾値を発明しない。

🔴 「OP以上」は `data/latest.json` から判定できない。
   レース側のキーは `name` / `dist` / `num_horses` / `conf` / `rec` などで、
   **クラス（新馬・未勝利・1勝〜3勝・OP・重賞）を表す列が1つも無い**
   （2026-09-29 に実データで確認）。レース名から推測すると
   「特別名がついている＝OP以上」という誤った判定になる
   （実例: 2026-09-27 中山9R サフラン賞は2歳1勝クラス、阪神9R 兵庫特別も平場の特別）。
   よって**指示された 9R〜11R という範囲をそのまま使い、クラスでは絞らない**。
   クラスで絞る必要が出たら、先に出馬表側でクラスを取得すること。

「自信のある平場」の定義は `tickets.CONF_STRONG`（=66）を流用する。
根拠は発明ではなく既存値との一致で、本番の `rec=True`（推奨レース）の境界そのもの。
2026-09-27 の実データ23レースで `conf >= 66` と `rec=True` は食い違い0。
"""

from src.product import tickets as _tk

# 各競馬場でクラスに関係なく対象にするレース番号（ユーザー指示の範囲）
MAIN_RACE_NUMS = (9, 10, 11)

# 平場（上記以外）を対象にする最低のレース信頼度。
# 新しい閾値ではなく tickets.CONF_STRONG（= 本番の rec の境界）を流用する。
FLAT_MIN_CONF = _tk.CONF_STRONG


def is_main_race(race):
    """9R〜11R か。"""
    try:
        return int(race.get('r')) in MAIN_RACE_NUMS
    except (TypeError, ValueError):
        return False


def in_scope(race):
    """(対象か, 理由) を返す。

    理由は記事やログにそのまま出せる日本語。
    """
    if is_main_race(race):
        return True, f"{race.get('r')}R（各競馬場9R〜11Rは対象）"

    conf = race.get('conf')
    if conf is None:
        return False, '平場でレース信頼度が無い'
    if conf >= FLAT_MIN_CONF:
        return True, f'平場だがレース信頼度{conf}（{FLAT_MIN_CONF}以上）'
    return False, f'平場でレース信頼度{conf}（{FLAT_MIN_CONF}未満）'


def select_races(races):
    """対象レースだけを返す。並び順は入力のまま。"""
    return [r for r in races if in_scope(r)[0]]
