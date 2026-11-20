"""商品として記事にする対象レースと、どちらの記事に載せるかを決める。

🟢 **2026-11-04 にユーザーが改正した。**
    「有料記事は10レースから11レースという縛りを無くします。純粋にAIが推奨する
      レースのみにします。推奨レースの中でも自信のあるレースをS、次点をAとして
      表示。10レース11レースの推奨漏れ分に関しては、無料予想として記事にします」
    → 確認したうえでの確定（同日）:
      「9〜11の内推奨レースがあればそれは有料。推奨が無ければ無料。
        その他平場の推奨レースは有料。推奨外は記事の必要なし」

    つまり:
      有料 = 推奨（conf >= FLAT_MIN_CONF）。**レース番号は問わない**
      無料 = 9〜11R のうち推奨でないもの
      記事にしない = 9〜11R 以外で推奨でないもの

    S/A の境界は `tickets.CONF_GRADE_S`（=72）。これは `confidence_stars()` が
    元から持っていた★5の境界の再利用で、**新しい閾値は1つも作っていない**。

🔑 **`in_scope()` の意味は改正前後で変わっていない。**
    改正前 = 9〜11R または conf>=66
    改正後 = 有料(conf>=66) ∪ 無料(9〜11R かつ conf<66) = 9〜11R または conf>=66
    ＝ **集合として同一**。だから `danger_log` が残してきた `in_scope` の記録は
    今後の記録とそのまま比べられる（意味がずれたら過去の台帳が静かに壊れる）。
    変わったのは「どちらの記事に載るか」だけで、それは `article_kind()` が返す。

--- 以下は改正前からの経緯（範囲を決めた根拠なので残す）---

ユーザー指示（2026-09-29）:
    「買い目は、各競馬場9R〜11R(OP以上）と自信のある平場レース」

🔴 これは**商品の対象範囲**の規則であって、買い目の選び方ではない。
   買い目そのもの（券種・配分・見送り）は `tickets.py` が決める。
   ここで新しい閾値を発明しない（`tickets.py` の値をそのまま使う）。

🔴 「OP以上」は `data/latest.json` から判定できない。
   レース側のキーは `name` / `dist` / `num_horses` / `conf` / `rec` などで、
   **クラス（新馬・未勝利・1勝〜3勝・OP・重賞）を表す列が1つも無い**
   （2026-09-29 に実データで確認）。レース名から推測すると
   「特別名がついている＝OP以上」という誤った判定になる
   （実例: 2026-09-27 中山9R サフラン賞は2歳1勝クラス、阪神9R 兵庫特別も平場の特別）。
   よって**指示された 9R〜11R という範囲をそのまま使い、クラスでは絞らない**。
   クラスで絞る必要が出たら、先に出馬表側でクラスを取得すること。

「自信のある平場」の定義は `tickets.CONF_STRONG`（=66）＝**固定の閾値**。

🟢 **2026-10-06 ユーザー決定: 「固定の数値でよい。上位6本という選び方はしない」。**
   本番の `rec`（その日の上位6本）には追従しない。同日の
   「今までやっていたものと商品にするものは別物と考えてよい」に従う。
   `rec` は当日の出走レース数で対象が増減するが、商品は毎週同じ基準で
   同じ範囲を出す必要がある。

🔴 **2026-10-05 訂正**（同じ誤りを書き戻さないため）。当初ここに「本番の
   `rec=True` の境界そのもの。2026-09-27 の実データ23レースで食い違い0」と書いたが、
   **その日だけの偶然だった**（10/03 は食い違い4件・10/04 は3件）。`rec` は閾値ではなく
   「軸の3着内確率>=0.55 を満たすレースのうち**その日の上位6本**」なので、
   そもそも固定の conf 値とは一致しない。
   よって 66 は**こちらが選んだ閾値**であり、**結果を見てから動かさない**。
"""

from src.product import tickets as _tk

# 各競馬場でクラスに関係なく対象にするレース番号（ユーザー指示の範囲）
MAIN_RACE_NUMS = (9, 10, 11)

# 平場（上記以外）を対象にする最低のレース信頼度。
# 🔴 tickets.CONF_STRONG（=66）と同じ**固定値**。本番の `rec` には追従しない
#    （2026-10-06 ユーザー決定）。ここで別の閾値を作らない。
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
    """対象レースだけを返す（有料・無料の両方）。並び順は入力のまま。"""
    return [r for r in races if in_scope(r)[0]]


#: `article_kind()` が返す記事の種類
PAID = 'paid'
FREE = 'free'


# 🟢 2026-11-21 ユーザー決定: 「最初の数ヶ月は、全て無料で良い。各競馬場の平場2レースずつ、
#    重賞は、11レース＋自信の有る他の重賞」（前日の「無料は上位6本くらいが良い」を受けて）。
#    → 無料 = 各競馬場の 11R ＋ 11R 以外でレース信頼度が高い順に2本。有料は当面なし。
#    ⚠ 「自信の有る他の重賞」は未実装: latest.json にクラス（重賞かどうか）の列が無い
#       （このファイル冒頭の 2026-09-29 の注記）。出馬表でクラスを取れるようにしてから足す。
FREE_MAIN_NUMS = (11,)
FREE_FLAT_PER_VENUE = 2
ALL_FREE = True


def race_key(race, venue=None):
    """日ごとの振り分け表のキー。race_id があればそれ、無ければ 会場+レース番号。"""
    rid = race.get('race_id') or race.get('id')
    if rid:
        return str(rid)
    return f"{venue if venue is not None else race.get('_venue', '')}{race.get('r')}"


def _r(race):
    try:
        return int(race.get('r'))
    except (TypeError, ValueError):
        return None


def day_selection(races):
    """その日の全レースから {race_key: (記事の種類, 理由)} を作る（2026-11-21 の規則）。

    races: latest.json の `races`（会場名 → レースの list）か、`_venue` 付きのレースの list。
    各会場で 11R を無料に、11R 以外は `conf` が高い順に2本を無料にする（同点はレース番号の若い順）。
    `conf` が無いレースは選ばない。それ以外は記事にしない (None)。
    """
    if isinstance(races, dict):
        pairs = [(v, r) for v, rs in races.items() for r in (rs or [])]
    else:
        pairs = [(r.get('_venue', ''), r) for r in (races or [])]
    by_venue = {}
    for v, r in pairs:
        by_venue.setdefault(v, []).append(r)
    out = {}
    for v, rs in by_venue.items():
        flats = []
        for r in rs:
            k = race_key(r, v)
            if _r(r) in FREE_MAIN_NUMS:
                out[k] = (FREE, f"{r.get('r')}R（各競馬場の11Rは無料で出す）")
                continue
            out[k] = (None, f"平場で各場の上位{FREE_FLAT_PER_VENUE}本に入らない（レース信頼度{r.get('conf')}）")
            if r.get('conf') is not None:
                flats.append((-float(r['conf']), _r(r) or 99, k, r))
        for rank, (_c, _n, k, r) in enumerate(sorted(flats)[:FREE_FLAT_PER_VENUE], 1):
            out[k] = (FREE, f"平場で{v}の信頼度{rank}位（レース信頼度{r.get('conf')}）")
    return out


def article_kind(race, day=None):
    """(記事の種類, 理由) を返す。どちらの記事にも載せないなら (None, 理由)。

    2026-11-04 のユーザー改正:
        有料 = 推奨（conf >= FLAT_MIN_CONF）。レース番号は問わない
        無料 = 9〜11R のうち推奨でないもの
        それ以外 = 記事にしない

    🔴 ここで新しい閾値を作らない。推奨の境界は `tickets.CONF_STRONG` 一つだけ。

    🟢 2026-11-21 から本番の記事は `day=day_selection(その日の全レース)` を渡す
       （各場 11R ＋ 平場の上位2本・全部無料）。`day` を渡さない呼び方は
       2026-11-04〜11-20 の規則のままで、成績ページの過去の区分にだけ使う。
    """
    if day is not None:
        return day.get(race_key(race), (None, 'その日の振り分け表に無い'))
    conf = race.get('conf')
    if conf is not None and conf >= FLAT_MIN_CONF:
        return PAID, f'推奨（レース信頼度{conf}・{FLAT_MIN_CONF}以上）'
    if is_main_race(race):
        shown = '無し' if conf is None else conf
        return FREE, f'{race.get("r")}Rだが推奨外（レース信頼度{shown}）'
    if conf is None:
        return None, '平場でレース信頼度が無い'
    return None, f'平場で推奨外（レース信頼度{conf}・{FLAT_MIN_CONF}未満）'
