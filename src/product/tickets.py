"""買い目と資金配分。**固定ルール**（後から最適化しない）。

根拠は `biz/PHASE4_DESIGN.md` §3・§4。要点だけ再掲する。

🟢 **2026-10-15 にユーザーが構成を変更した。**
「読者は連系の少しおおきな当たりを好む。固めと中穴がバランスよく入る馬券を選ぶ。
さらに全券種を記載する」という指示を受け、複勝＋ワイドの2券種から
**単勝・複勝・馬連・ワイド・三連複・三連単（流し／2頭軸／マルチ／フォーメーション）
を全部並べる**形に変えた。

🔴 **着手前に伝えた懸念をここに残す（数字が良くなる変更ではない）。**
- 控除率は券種ごとに違い、**広げるほど不利になる**
  （単勝・複勝 20% / 馬連・ワイド 22.5% / 三連複 25% / 三連単 27.5%）。
- 2026-08-03 に396レースで買い方を総当たりした実測は
  馬連 70.2% / ワイド 71.4% / 三連複は最高配当1本を除くと 58.9% で、
  **複勝（最良 93.4%・N=3,560）より低い**。
- 2026-08-24 の馬連検証では **AIで相手を選ぶより素直に人気順で選ぶほうが良かった**
  （60.5% vs 71.4%）。
- 2026-09-19/21 の本番実績でも「軸は当たるのに相手を広げると落ちる」形が2日とも出た
  （土 軸単勝 96.5% → ワイド 44.2% / 馬連 53.6% / 三連複 43.1%。
  月 120.0% → 58.3% / 79.3% / 32.0%。⚠ N=12〜23 なので証拠にはならない）。
→ **それでもユーザーの判断で広げる。** 記事には上の事実をそのまま書く
  （`REASON_*` を参照）。回収率が上がるとは書かない。

🔴 複勝を配分の中心に置く理由は「数字が高いから」ではない。2026-08-18③ で高自信レースの
券種を総当たりしたとき、**上位3本の配当を抜いても崩れないのは複勝だけ**
（複勝 −1.4pt / ワイド −8.5pt）。ワイドは控除率 22.5% で複勝 20% より不利。
単勝は RL1 が 86.0% で 1番人気 86.1% と区別がつかない。

🔴 傾斜配分で期待値は動かない（総期待値 = 各EVの加重平均。全EV<1ならどう配分しても<1）。
配分を示すのは分散を抑えるためであって、「配分で勝てる」とは書かない。

🟢 **2026-10-26 にユーザーが「全部の買い目は推奨しない」と決めた。**
「固くいきたい人は単複のみでよい。大きく勝ちたい人は連系を買う。2種程度を推奨する」
という指示を受け、**9行を1つの塊として出すのをやめ、2つのコースに分けた**。

- **堅実コース** = 単勝・複勝
- **勝負コース** = 連系（馬連・ワイド・三連複・三連単）から**グループごとに本線1行だけ**

🔴 **コースは「選ぶもの」で「両方買うもの」ではない。** 記事にもそう書く
   （`COURSE_NOTE`）。全券種にまたがる配分は出さない（出すと結局全部買う形になる）。

🔴 **残りの3行（三連複(2頭軸)・三連単マルチ・三連単フォーメーション）は
   コースに入れず「同じ券種の別の組み方」として参考に置く。**
   これは好みの問題ではなく**算数の問題**で、実データで確かめてある
   （相手5頭のとき）:
   - 三連複(2頭軸) 4点は 三連複(流し) 10点の**完全な部分集合**
   - 三連単フォーメーション 6点は 三連単(流し) 20点の**完全な部分集合**
   - 三連単マルチ 24点のうち 8点が 三連単(流し) と**重複**
   → 並べて全部買うと**同じ当たり組を二重に買う**。合計点数も回収率も
     その分だけ嘘になる。コースには各グループ1行しか入れない。

⚠ **どちらのコースも「勝てる」ことを示していない。** コース分けは
   「全部買う前提をやめる」ための構成の変更であって、回収率を上げる変更ではない。
   どの券種も実測は100%未満（`REASON_*` 参照）。
⚠ **コースの中身を成績を見てから選び直さない。** 2026-10-25 の実績（13レース）は
   N が小さすぎて券種の取捨の根拠にならない（North Star #7）。
   いまの割り当ては**構造**だけで決めてある——単複は点数が少なく3着までで当たる側、
   連系は配当が大きい側、という定義上の区別。

🔴 **中穴の定義を商品側で発明しない。** 固め／中穴の境界は本番の
`src/betting/axis_bets.py`（2026-08-08 導入・実測でオッズ帯を比べて決めた値）を
そのまま import する。7〜30倍を使う根拠もそこに実測表がある。
"""

from itertools import combinations

from src.betting.axis_bets import ANA_ODDS_MAX, ANA_ODDS_MIN

from . import marks as _marks

# 勝負度の境界。conf は race_confidence()（軸の3着内確率 × 100）。
#
# 🔴 CONF_LIGHT は**本番で既に使われている閾値**（`ev_filter.MIN_AXIS_FUKU_PROB`）に
#    合わせてある。CONF_STRONG は**商品側で選んだ固定値**で、2026-10-06 に
#    ユーザーが「固定の数値でよい」と決めた。
#    どちらも**結果を見てから動かさない**（それをやると後から最適化したのと同じになる）。
#
#  CONF_STRONG = 66 : 🟢 **商品側で選んだ固定の閾値**（本番 `rec` の境界ではない）。
#                     2026-10-06 にユーザーが「**固定の数値でよい。上位6本という
#                     選び方はしない**」と決定した。以後この値を動かさない。
#                     🔑 **商品は本番の `rec` に追従しない。** 同日のユーザー判断
#                     「今までやっていたものと商品にするものは別物と考えてよい」に従う。
#                     `rec` はその日の出走レース数で対象が増減するが、商品は毎週
#                     同じ基準で同じ範囲を出す必要がある。
#                     ⚠ **2026-10-05 訂正の経緯**（同じ誤りを書き戻さないため）:
#                     当初「2026-09-27 の23レースで rec と conf>=66 が1件も食い違わない」
#                     ことを根拠に「本番由来の閾値」と書いたが、**その日だけの偶然だった**。
#                     10/03 は食い違い4件（rec=True の conf が 62/62/63/64）、
#                     10/04 は3件（55/63/63）。
#                     機序: `rec` は閾値ではなく「軸の3着内確率>=0.55 を満たすレースのうち
#                     **その日の上位 max_races=6 本**」（`ev_filter.select_quality_races`）
#                     なので、固定の conf 値とは一致しようがない。
#                     **結果を見てから動かさないこと。**
#  CONF_LIGHT  = 55 : `ev_filter.MIN_AXIS_FUKU_PROB = 0.55` と同じ。
#                     本番が「そもそも買い目を出すか」を判定している閾値で、
#                     根拠は 2026-08-08②（軸の複勝確率が下位半分の帯はROIが一貫して悪い）
CONF_STRONG = 66   # これ以上: 全券種（三連単まで）
CONF_LIGHT = 55    # これ以上: 単勝・複勝・馬連・ワイド・三連複（三連単は出さない）
# CONF_LIGHT 未満: 見送り

# 1レースに並べる買い目の行数の上限。
# 🟢 2026-10-15 にユーザーが「全券種を記載する」と決めたので 2 → 9 に変えた。
#    内訳は 単勝・複勝・馬連・ワイド・三連複・三連複(2頭軸)・
#    三連単・三連単マルチ・三連単フォーメーション の9行。
# ⚠ これは「9券種を買え」ではない。読者が選べるように**並べる**上限。
# 🟢 2026-10-26: 並べる上限は変えていないが、**推奨の単位はコースに変わった**。
#    9行のうちコースに入るのは最大6行で、残り3行は参考（上の冒頭を読む）。
#    点数と100円均等の総額は**コースごとに**出す（§10 各自調整）。
MAX_BET_TYPES = 9

# 単勝オッズの下限。JRAの単勝は 1.0 倍未満が存在しないので、それ未満は「未取得」
# （2026-07-27⑧ でエンジン側にも同じ下限を入れてある）。
# 🔴 有料商品に 0.0倍 を載せない。土曜夜の日曜予想は毎週 odds_coverage=0.0 になる
MIN_VALID_ODDS = 1.0

# ── 相手（ヒモ）の構成 ────────────────────────────────────────────────────
# 🟢 2026-10-15 ユーザー指示「固めと中穴がバランスよく入る馬券を選ぶ」。
#
# 固め = 単勝 ANA_ODDS_MIN(7.0) 倍未満 ／ 中穴 = 7.0〜30.0倍。
# 🔴 境界は本番 `axis_bets` から import している（商品側で発明しない）。
#    そこに実測表がある: ワイド1点の回収率／高配当1本抜きで
#    3〜7倍 71.8%/67.9% ・ **7〜30倍 102.5%/87.4%** ・ 15〜50倍 57.9%/51.5%。
#    ⚠ 102.5% は1本抜きで87.4%まで落ちるので「7〜30倍なら勝てる」ではない。
#    言えるのは「**30倍超の大穴よりはっきり良い帯である**」まで。
# 🔴 30倍超（大穴）は相手に入れない。上の実測で明確に悪い側。
#
# 相手の総数はユーザー例（①－②③④⑤⑥ ＝ 相手5頭）に合わせた。
# ⚠ その内訳（固め3＋中穴2）は**こちらが決めた比率**で、実測の裏付けは無い。
#    「バランスよく」を機械的に表せる最小の形として置いている。
PARTNERS_STRONG = 5
PARTNERS_LIGHT = 3
SOLID_SHARE_STRONG = 3
ANA_SHARE_STRONG = 2
SOLID_SHARE_LIGHT = 2
ANA_SHARE_LIGHT = 1

# 配分（比率）。複勝を最も厚く、連系はそれぞれ押さえ。
# 🔴 これは**指示ではなく目安**。§10 は「資金配分は、オッズと資金に応じて各自調整 とする」
#    と定めており、PHASE4_DESIGN.md §4 も「比率で示す・金額は換算例」としている。
#    比率を断定形で出すと、期待値を動かせない操作（下記）を効くもののように見せてしまう。
# ⚠ 連系を厚くしていないのは、控除率が高い側へ資金を寄せる根拠が無いため。
#    読者が連系を厚くしたいなら各自調整してよい（そう書いてある）。
ALLOC_STRONG = {'複勝': 0.25, 'ワイド': 0.15, '馬連': 0.15,
                '三連複': 0.20, '三連単': 0.15, '単勝': 0.10}
ALLOC_LIGHT = {'複勝': 0.40, 'ワイド': 0.20, '馬連': 0.15,
               '三連複': 0.15, '単勝': 0.10}

# 金額は「比率を円に直したときの換算例」であって推奨額ではない。
AMOUNT_EXAMPLES = (1000, 3000, 5000)

# 配分に必ず添える注記。記事・アーカイブの両方へこの文のまま載せる。
# 🔴 2026-08-18③: 総期待値 = 各EVの加重平均。全EV<1ならどう配分しても<1。
#    配分は分散を変えるだけで期待値を1ミリも動かさない（算数で決着している）。
ALLOCATION_NOTE = (
    '配分は目安です。資金配分で期待値は動きません'
    '（全体の期待値は各買い目の期待値の加重平均になるため、'
    'どう振り分けても平均より上には行きません）。'
    '比率を示しているのは当たり外れの振れ幅を抑えるためで、'
    'オッズと手持ち資金に応じて各自調整してください。'
)

# レースごとに載せる短縮版。理由の全文（上）は記事末尾の注意事項に1回だけ出す。
# 🔴 同じ段落を1記事に7回以上くり返すと読み飛ばされる。短縮版でも
#    「目安であること」と「各自調整」は落とさない（そこが §10 の要件）。
ALLOCATION_NOTE_SHORT = (
    '配分は目安です。オッズと手持ち資金に応じて各自調整してください。'
)


# ── 推奨コース ───────────────────────────────────────────────────────────
# 🟢 2026-10-26 ユーザー指示「全部の買い目は推奨しない。固くいきたい人は単複のみで
#    よい。大きく勝ちたい人は連系を買う。2種程度を推奨する」。
#
# 🔴 どの券種をどちらに入れるかは**構造だけ**で決めてある。成績では決めていない。
#    - 堅実 = 単勝・複勝。点数が最小で、複勝は3着までで当たる（定義）。控除率も20%で最小。
#    - 勝負 = 連系。相手を絡める以上、当たりにくく配当は大きい（定義）。
#    ⚠ 実測ではどの券種も100%未満で、コース間の優劣は測っていない。
#    ⚠ 2026-10-25 の13レースのような小さい N を見て中身を入れ替えない（North Star #7）。
#
# 🔴 コースに入れるのは**グループごとに本線1行だけ**。同じグループの別の行
#    （三連複(2頭軸)・三連単マルチ・三連単フォーメーション）は重複するので
#    コースに入れない（冒頭の算数を参照）。
COURSES = (
    {
        'key': 'steady',
        'name': '堅実コース',
        'lead': '固く当てにいきたい人向け',
        'groups': ('単勝', '複勝'),
    },
    {
        'key': 'aggressive',
        'name': '勝負コース',
        'lead': '連系で大きく取りにいきたい人向け',
        'groups': ('馬連', 'ワイド', '三連複', '三連単'),
    },
)

# コースの扱いを必ず添える。記事・アーカイブの両方にこの文のまま載せる。
# 🔴 「両方買う」と読まれたら元の全部買う形に戻ってしまうので、ここは落とさない。
COURSE_NOTE = (
    'コースは**どちらか一方を選ぶ**ためのものです。両方を足して買う前提では'
    'ありません。どちらのコースも、回収率が100%を超えることを確認できては'
    'いません（券種ごとの実測は下に書いたとおり、いずれも100%未満です）。'
)

# コースに入らなかった行の説明。
# 🔴 ここは好みではなく算数。相手5頭のときの実測値をそのまま書く。
ALTERNATIVE_NOTE = (
    '以下は**同じ券種を点数を絞って買う場合の組み方**です。'
    '上のコースと重ねて買うものではありません。'
    '三連複(2頭軸)の買い目は三連複(流し)に完全に含まれ、'
    '三連単フォーメーションの買い目は三連単(流し)に完全に含まれます。'
    '三連単マルチも三連単(流し)と一部が重なります。'
    '重ねて買うと同じ当たりを二重に買うことになります。'
)

REASON_COURSES = (
    '買い目は**2つのコースに分けて**出しています。'
    '全部を並べて「どれも買ってください」という形にはしません。'
    '**堅実コース（単勝・複勝）**は点数が最小で控除率も20%と最も低く、'
    '複勝は3着までで当たります。'
    '**勝負コース（連系）**は相手を絡めるぶん当たりにくく、そのかわり配当が'
    '大きくなります。'
    'この振り分けは券種の作りから決めたもので、'
    '**どちらのほうが回収率が高いかを測って決めたものではありません。**'
)


# 券種を選んだ理由（§11「この予想なら、なぜこの券種なのか」まで説明する）。
#
# 🔴 出どころは全部「過去データで買い方を総当たりした測定」で、こちらの成績ではない。
#    どの数字も100%を超えていない。券種を選べば控除率を超えられる、とは書かない。
REASON_SPREAD = (
    '券種を広げているのは、連系で少し大きな当たりを取りたいという読者の好みに'
    '合わせた構成です。**広げたから当たりやすくなる、という意味ではありません。**'
    '控除率は券種ごとに違い、単勝・複勝20%、馬連・ワイド22.5%、三連複25%、'
    '三連単27.5%と、点数を広げるほど不利な側へ寄ります。'
)
# 🔴 ここに 102.5% という生の数字は書かない。高配当1本で作られた数字を
#    有料記事の見出しに置くと、こちら自身が禁じている誇大表示になる
#    （North Star・biz/PHASE4_DESIGN.md §20）。帯の比較には**高配当1本を抜いた後**の
#    数字（すべて100%未満）を使い、順序だけを示す。
REASON_BALANCE = (
    f'相手は**固め（単勝{ANA_ODDS_MIN:.0f}倍未満）と中穴（{ANA_ODDS_MIN:.0f}〜'
    f'{ANA_ODDS_MAX:.0f}倍）を混ぜて**選んでいます。'
    f'{ANA_ODDS_MAX:.0f}倍を超える大穴は相手に入れません。'
    f'過去データでオッズ帯ごとにワイド1点の回収率を比べ、'
    f'高配当1本を除いた後の数字で並べると'
    f'3〜7倍 67.9%、7〜30倍 87.4%、15〜50倍 51.5% で、'
    f'7〜30倍の帯がいちばん落ち込みませんでした（どれも100%未満です）。'
    f'言えるのは「大穴まで広げるよりはこの帯のほうが安定している」までです。'
)
REASON_FUKUSHO = (
    '複勝に配分を最も厚く置いているのは、過去データで買い方を総当たりしたときに'
    '**上位の高配当を3本抜いても順位が崩れなかったのが複勝だけ**だったためです'
    '（3本抜きでの下落は複勝 −1.4pt に対しワイド −8.5pt）。'
    '控除率も20%で、扱っている券種のなかで最も低いです。'
)
REASON_WIDE = (
    'ワイドは押さえです。控除率が22.5%で複勝より不利なぶん、'
    '◎から固め1頭・中穴1頭への2点に絞っています。'
)
REASON_QUINELLA = (
    '馬連は◎から相手全頭への流しです。同じ過去データでの実測は70.2%でした。'
    '⚠ さらに、2026-08-24 に馬連の買い方を比べたときは'
    '**AIで相手を選ぶより素直に人気順で選ぶほうが良い**という結果でした'
    '（60.5% vs 71.4%）。相手の選び方そのものに優位は確認できていません。'
)
REASON_TRIO = (
    'コースに入れている三連複は◎からの流し1本です。'
    '◎＋相手筆頭の2頭軸は「点数を絞る場合の組み方」として参考に置いています'
    '（2頭軸の買い目は流しに完全に含まれるので、重ねて買うものではありません）。'
    '控除率は25%です。実測で6点流しが一度だけ収支トントンに見えたことがありますが、'
    '**万馬券1本（¥97,560）を除くと58.9%**まで落ち、期間を前後に割っても'
    '崩れました。高配当1本に乗っていただけの数字として扱っています。'
)
REASON_TRIFECTA = (
    'コースに入れている三連単は1着◎固定の流し1本だけです。'
    '2頭軸マルチとフォーメーションは、点数を絞る／着順の自由度を上げる場合の'
    '組み方として参考に置いています（流しと買い目が重なります）。'
    '控除率は27.5%で、扱っている券種のなかで最も高くなります。'
    '三連単そのものの回収率はこちらで測っていません'
    '（実配当が揃っているのが2026-06以降しかなく、大規模な検証ができていません）。'
    '**測っていないものを「良い」とは書けない**ので、点数と組み方だけを出します。'
)
REASON_WIN = (
    '単勝は◎1点です。◎（AI＋市場の1位）の単勝回収率は86.0%でしたが、'
    'これは素直に1番人気を買った86.1%と**区別がつきません**。'
    '単勝を載せているのは券種を全部並べるためで、ここに優位は確認できていません。'
)
REASON_LIGHT_ONLY = (
    '今回は三連単を出していません。このレースはレース信頼度が'
    '最上位の帯に届いていないため、点数の多い券種を増やしていません。'
)
REASON_NARROW_FIELD = (
    '⚠ 相手に取れる頭数が少ないため、点数の多い券種は組めた分だけになっています。'
)
REASON_CAVEAT = (
    'なお、ここに挙げた回収率はいずれも100%を下回っています。'
    '券種の選び方で控除率を超えられるとは考えていません。'
    '複勝に配分を厚くしているのは「勝てるから」ではなく'
    '「測った数字が偶然の高配当に左右されにくいから」です。'
    'また、本番の実績では「軸は当たるのに相手を広げると落ちる」形も出ています'
    '（2026-09-19 軸の単勝96.5%に対しワイド44.2%・馬連53.6%・三連複43.1%。'
    '⚠ 12〜23レースしかないので、こちらも証拠にはなりません）。'
)


def bet_type_reason(tickets):
    """この買い目でその券種になった理由を文のリストで返す（§11）。

    見送りなら空リスト（買っていないものの理由は書かない）。
    """
    if tickets.get('verdict') == 'skip':
        return []
    kinds = {b.get('group') or b['type'] for b in tickets.get('bets', [])}
    if not kinds:
        return []
    out = [REASON_COURSES, REASON_SPREAD, REASON_BALANCE, REASON_FUKUSHO]
    if 'ワイド' in kinds:
        out.append(REASON_WIDE)
    if '馬連' in kinds:
        out.append(REASON_QUINELLA)
    if '三連複' in kinds:
        out.append(REASON_TRIO)
    if '三連単' in kinds:
        out.append(REASON_TRIFECTA)
    else:
        out.append(REASON_LIGHT_ONLY)
    if '単勝' in kinds:
        out.append(REASON_WIN)
    if tickets.get('narrow_field'):
        out.append(REASON_NARROW_FIELD)
    out.append(REASON_CAVEAT)
    return out


def usable_odds(horse):
    """使えるオッズ（1.0倍以上）を返す。無ければ None。"""
    for key in ('odds', 'tansho_odds'):
        v = horse.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f >= MIN_VALID_ODDS:
            return f
    return None


def confidence_stars(conf):
    """conf → 勝負度の星。conf が無ければ None（作り話をしない）。"""
    if conf is None:
        return None
    if conf >= 72:
        return 5
    if conf >= CONF_STRONG:
        return 4
    if conf >= CONF_LIGHT:
        return 3
    if conf >= 46:
        return 2
    return 1


def verdict(conf):
    """買うか見送るか。conf が無ければ見送り（不明なら勝負しない）。"""
    if conf is None:
        return 'skip'
    if conf >= CONF_STRONG:
        return 'strong'
    if conf >= CONF_LIGHT:
        return 'light'
    return 'skip'


def allocate(total, ratios, unit=100):
    """比率 → 100円単位の金額。合計は必ず total に一致する。

    🔴 各券種を独立に四捨五入すると、券種が増えたときに
       **比率の大小が金額で逆転する**（1,000円・6券種で複勝100円 < ワイド200円 に
       なった）。最大剰余法で配ることで順序を保つ。
    """
    if not ratios:
        return {}
    units = int(total // unit)
    keys = sorted(ratios, key=lambda k: -ratios[k])
    if units < len(keys):
        # 全券種に1単位も配れない。比率の大きい順に配れる分だけ配る
        return {k: unit for k in keys[:units]} if units else {}
    base = {}
    rem = []
    for k in keys:
        exact = total * ratios[k] / unit
        floor = max(1, int(exact))
        base[k] = floor
        rem.append((exact - int(exact), ratios[k], k))
    left = units - sum(base.values())
    rem.sort(key=lambda t: (-t[0], -t[1], t[2]))
    i = 0
    while left > 0 and rem:
        base[rem[i % len(rem)][2]] += 1
        left -= 1
        i += 1
    while left < 0:
        # 配りすぎ（最低1単位の下駄で起きうる）。比率の小さい側から削る
        for _, _, k in reversed(rem):
            if left == 0:
                break
            if base[k] > 1:
                base[k] -= 1
                left += 1
        else:
            break
    return {k: base[k] * unit for k in keys}


def _partner_candidates(horses, axis_num):
    """相手候補を (rl_rank, 馬番, 単勝オッズ) の昇順で返す。

    オッズが取れていない馬・rl_rank が不明な馬は相手にしない
    （記事に「—倍」の馬を連系の相手として載せないため）。
    """
    out = []
    for h in horses:
        n = h.get('n')
        if n is None or n == axis_num:
            continue
        o = usable_odds(h)
        if o is None:
            continue
        r = h.get('rl_rank')
        if not isinstance(r, int) or not (0 < r < 99):
            continue
        out.append((r, int(n), o))
    out.sort()
    return out


def split_partners(horses, axis_num, n_solid, n_ana):
    """相手を「固め」「中穴」に分けて選ぶ。

    返り値は (相手全体, 固め, 中穴)。相手全体は rl_rank 昇順。
    片方の枠が埋まらなければもう片方で埋め、総数（n_solid + n_ana）を保つ。
    """
    cands = _partner_candidates(horses, axis_num)
    total = n_solid + n_ana
    solid = [n for _, n, o in cands if o < ANA_ODDS_MIN]
    ana = [n for _, n, o in cands if ANA_ODDS_MIN <= o <= ANA_ODDS_MAX]
    got_solid = solid[:n_solid]
    got_ana = ana[:n_ana]
    short = total - len(got_solid) - len(got_ana)
    if short > 0:
        got_solid += solid[len(got_solid):len(got_solid) + short]
    short = total - len(got_solid) - len(got_ana)
    if short > 0:
        got_ana += ana[len(got_ana):len(got_ana) + short]
    order = {n: i for i, (_, n, _o) in enumerate(cands)}
    partners = sorted(got_solid + got_ana, key=lambda n: order.get(n, 99))
    return partners, got_solid, got_ana


def _join(nums):
    """馬番を馬券の書き方どおり**昇順**で並べる。

    🔑 選ぶ順（rl_rank）と書く順は分ける。どの馬を2頭目の軸にするか・
       フォーメーションの2着列に誰を置くかは rl_rank で決めるが、
       **並び自体に意味が無い列は馬番の昇順で書く**（11 － 1・7・5・10・2 は読めない）。
    """
    return '・'.join(str(n) for n in sorted(nums))


def _bet(type_, group, axis, combos, line, note=''):
    return {
        'type': type_, 'group': group, 'axis': axis,
        'combos': [list(c) for c in combos], 'points': len(combos),
        'line': line, 'note': note,
    }


def build_bet_lines(axis_num, partners, solid, ana, allow_trifecta):
    """全券種の買い目を並べて返す。買える形にならない券種は出さない。"""
    bets = [
        _bet('単勝', '単勝', axis_num, [[axis_num]], f'単勝　{axis_num}',
             '◎1点'),
        _bet('複勝', '複勝', axis_num, [[axis_num]], f'複勝　{axis_num}',
             '本線'),
    ]
    if not partners:
        return bets

    # 馬連: ◎から相手全頭
    q = [[axis_num, p] for p in partners]
    bets.append(_bet('馬連', '馬連', axis_num, q,
                     f'馬連　{axis_num} － {_join(partners)}（{len(q)}点）',
                     '◎から相手全頭'))

    # ワイド: 固め筆頭 ＋ 中穴筆頭（どちらか無ければ相手の上位で埋める）
    wide_nums = [n for n in (solid[0] if solid else None,
                             ana[0] if ana else None) if n is not None]
    if len(wide_nums) < 2:
        for n in partners:
            if n not in wide_nums:
                wide_nums.append(n)
            if len(wide_nums) == 2:
                break
    wide_nums = sorted(wide_nums, key=lambda n: partners.index(n))
    w = [[axis_num, p] for p in wide_nums]
    bets.append(_bet('ワイド', 'ワイド', axis_num, w,
                     f'ワイド　{axis_num} － {_join(wide_nums)}（{len(w)}点）',
                     '固め1頭＋中穴1頭'))

    # 三連複: ◎からの流し
    if len(partners) >= 2:
        t = [[axis_num, a, b] for a, b in combinations(partners, 2)]
        bets.append(_bet('三連複', '三連複', axis_num, t,
                         f'三連複　{axis_num} － {_join(partners)}'
                         f'（{len(t)}点）', '◎1頭軸流し'))
    # 三連複: ◎＋相手筆頭の2頭軸
    if len(partners) >= 3:
        sub, rest = partners[0], partners[1:]
        t2 = [[axis_num, sub, r] for r in rest]
        bets.append(_bet('三連複(2頭軸)', '三連複', axis_num, t2,
                         f'三連複(2頭軸)　{axis_num} － {sub} － {_join(rest)}'
                         f'（{len(t2)}点）', '◎と相手筆頭の2頭軸'))

    if not allow_trifecta:
        return bets

    # 三連単: 1着◎固定の流し
    if len(partners) >= 2:
        f = [[axis_num, a, b] for a in partners for b in partners if a != b]
        bets.append(_bet('三連単', '三連単', axis_num, f,
                         f'三連単　{axis_num} → {_join(partners)}'
                         f'（{len(f)}点）', '◎1着固定'))
    # 三連単マルチ: ◎と相手筆頭の2頭軸（着順を問わない）
    if len(partners) >= 2:
        sub, rest = partners[0], partners[1:]
        m = []
        for r in rest:
            trio = (axis_num, sub, r)
            m += [[a, b, c] for a in trio for b in trio for c in trio
                  if len({a, b, c}) == 3]
        bets.append(_bet('三連単マルチ', '三連単', axis_num, m,
                         f'三連単マルチ　{axis_num}・{sub} － {_join(rest)}'
                         f'（{len(m)}点）', '2頭軸マルチ'))
    # 三連単フォーメーション: 1着◎ / 2着 相手上位2頭 / 3着 残り
    if len(partners) >= 3:
        second, third = partners[:2], partners[2:]
        fm = [[axis_num, a, b] for a in second for b in third if a != b]
        if fm:
            bets.append(_bet('三連単フォーメーション', '三連単', axis_num, fm,
                             f'三連単フォーメーション　{axis_num} → '
                             f'{_join(second)} → {_join(third)}'
                             f'（{len(fm)}点）', '1着◎固定'))
    return bets


def build_courses(bets, base_ratios):
    """並べた買い目を2つのコースに振り分けて返す。

    返り値は (courses, alternatives)。

    - courses: `COURSES` の順。各グループの**本線1行だけ**が入る。
      行が1つも無いコースは返さない（空のコースを記事に出さない）。
    - alternatives: コースに入らなかった行（同じ券種の絞り込み版）。

    🔴 コースをまたいだ配分は作らない。コースは選ぶものなので、
       配分は**コースの中**だけで意味を持つ。
    """
    used = set()
    courses = []
    for spec in COURSES:
        lines = []
        for g in spec['groups']:
            for b in bets:
                if (b.get('group') or b['type']) != g:
                    continue
                lines.append(b)
                used.add(id(b))
                break          # 同じグループからは本線1行だけ
        if not lines:
            continue
        ratios = {g: base_ratios[g] for g in spec['groups']
                  if g in base_ratios
                  and any((b.get('group') or b['type']) == g for b in lines)}
        total = sum(ratios.values()) or 1.0
        ratios = {k: round(v / total, 2) for k, v in ratios.items()}
        pts = sum(b['points'] for b in lines)
        courses.append({
            'key': spec['key'],
            'name': spec['name'],
            'lead': spec['lead'],
            'lines': lines,
            'types': [b['type'] for b in lines],
            'points': pts,
            'flat_unit_total': pts * 100,
            'allocation_ratio': ratios,
            'allocation_examples': {
                amt: allocate(amt, ratios) for amt in AMOUNT_EXAMPLES
            } if len(ratios) >= 2 else {},
        })
    alternatives = [b for b in bets if id(b) not in used]
    return courses, alternatives


def build_tickets(race, horses):
    """買い目・配分・見送り判断を返す。

    入力は変更しない。並べる行数は必ず MAX_BET_TYPES 以内。

    🟢 2026-10-15 ユーザー指示で全券種（単勝・複勝・馬連・ワイド・三連複・
       三連単）を並べる形にした。相手は固めと中穴を混ぜて選ぶ。
       ⚠ これは読者の好みに合わせた構成で、回収率が上がる根拠は無い
       （モジュール冒頭の懸念を参照）。

    🔑 ◎ 自身が「危険な人気馬」に当たった場合は勝負度を1段下げる。
       これが §10「買わない」も予想として販売する、の実体。
       根拠: 市場1番人気のうち素のAIが6位以下の馬は3着内率 68.5%→51.4%
       （B期間6,077レースでも 63.6%→48.6% で再現。⚠ 半分近くは来るので
       「危険」と断定しない。全文 `danger/RESULTS.md`）
       （N=242・73開催日・walk-forward）。◎がそこに当たっているなら、
       同じレースを強気に買う理由が無い。
    """
    conf = race.get('conf')
    v = verdict(conf)
    axis = _marks.axis_horse(horses)

    # ◎ が危険な人気馬か
    axis_is_danger = False
    if axis is not None:
        danger_nums = {d['num'] for d in _marks.danger_favorites(horses)}
        axis_is_danger = axis.get('n') in danger_nums

    result = {
        'conf': conf,
        'stars': confidence_stars(conf),
        'verdict': v,
        'bets': [],
        'courses': [],
        'alternatives': [],
        'course_note': None,
        'alternative_note': None,
        'partners': [],
        'solid_partners': [],
        'ana_partners': [],
        'narrow_field': False,
        'total_points': 0,
        'flat_unit_total': 0,
        'allocation_ratio': {},
        'allocation_examples': {},
        'allocation_note': None,
        'bet_type_reasons': [],
        'skip_reason': None,
    }

    if axis is None:
        result['verdict'] = 'skip'
        result['skip_reason'] = '軸（AI+市場1位）が特定できないため見送り'
        return result

    # 🔴 2026-10-06 訂正。ここの理由を「妙味（＝期待値）の判断ができない」と
    #    書いていたが**誤り**だった（ユーザー指摘）。`EV = AI確率 × オッズ` で買い目を
    #    選ぶのは 2026-07-05 / 07-30 / 08-31 に**3回否定済み**（EV>=1.0 は回収57.9%で、
    #    買わない側 EV<1.0 の 75.0% より悪い）。商品は EV で選んでいないので、
    #    EV を理由に見送るのは自分の記録と矛盾する。**この文面を書き戻さないこと。**
    #
    #    見送り自体は残す。オッズが要る本当の理由は EV ではなく次の3つ:
    #    (1) 印そのものがオッズに乗っている。本番は残差学習で
    #        `予測 = base_margin(市場) + AI補正`。オッズが無いとフラットアンカーに
    #        落ち、5窓すべてで AUC が下がる（健全 0.78〜0.80 → フラット 0.76〜0.79。
    #        2026-08-29 FB）。⚠ ただし複勝回収率は 82.6% vs 83.0% で区別がつかないので、
    #        「当てる精度が落ちる」までしか言えない（「儲けが落ちる」とは書かない）。
    #    (2) 「危険な人気馬」が1頭も出せない。`marks.danger_favorites` は市場人気を
    #        必須にしているため（商品の2本柱の片方・68.5%→51.4%）。
    #    (3) 記事の数字が全部「—」になる（表のオッズ列・◎の見出し・市場人気）。
    #        2026-09-27 は refresh 遅延で odds_coverage=0.0 のまま推定オッズで組まれ、
    #        阪神R3 単勝#2 が est 10.0倍 / 実際 1.3倍だった。有料で売る物に出せない。
    #    🔴 2026-10-15 追記: 連系を入れたので (3) はさらに重い。相手のオッズが
    #        分からないと「固め」「中穴」の区別そのものが成立しない。
    if usable_odds(axis) is None:
        result['verdict'] = 'skip'
        result['skip_reason'] = (
            'オッズが取得できていないため見送り。'
            'このAIは市場評価を予想の土台に使っているため、'
            'オッズが無い状態では印の精度が落ちます。'
            '固め・中穴の区別もオッズが無いと付けられません'
        )
        return result

    # ◎ が危険な人気馬なら1段下げる（strong→light、light→skip）
    downgraded = False
    if axis_is_danger and v != 'skip':
        v = 'light' if v == 'strong' else 'skip'
        downgraded = True
    result['verdict'] = v
    result['axis_is_danger_favorite'] = axis_is_danger

    if v == 'skip':
        if downgraded:
            result['skip_reason'] = (
                '◎は市場1番人気だが、市場を見ないAIの評価は下位。'
                'この形に当たった市場1番人気は2つの期間の検証で3着内率が'
                '68.5%→51.4% / 63.6%→48.6% まで落ちている'
                '（⚠ 落ち幅は本物だが、半分近くは3着内に来る）。'
                'レース信頼度も高くないため今回は見送り'
            )
        else:
            result['skip_reason'] = (
                f'レース信頼度が低い（{conf}）ため見送り。'
                f'軸の3着内確率が下位の帯では回収率が一貫して悪い'
                if conf is not None else
                'レース信頼度が算出できなかったため見送り'
            )
        return result

    if downgraded:
        result['downgrade_reason'] = (
            '◎は市場1番人気だが、市場を見ないAIの評価は下位。'
            'この形は過去の検証で3着内率が落ちているため、'
            '相手を広げず点数を絞った'
        )

    axis_num = axis.get('n')
    if v == 'strong':
        n_solid, n_ana = SOLID_SHARE_STRONG, ANA_SHARE_STRONG
        want = PARTNERS_STRONG
    else:
        n_solid, n_ana = SOLID_SHARE_LIGHT, ANA_SHARE_LIGHT
        want = PARTNERS_LIGHT

    partners, solid, ana = split_partners(horses, axis_num, n_solid, n_ana)
    result['partners'] = partners
    result['solid_partners'] = solid
    result['ana_partners'] = ana
    result['narrow_field'] = len(partners) < want

    bets = build_bet_lines(axis_num, partners, solid, ana,
                           allow_trifecta=(v == 'strong'))
    result['bets'] = bets[:MAX_BET_TYPES]
    # ⚠ 並べた行をそのまま足した点数。**記事には出さない**（2026-10-26）。
    #    三連複(2頭軸)は三連複(流し)の部分集合、三連単フォーメーションは
    #    三連単(流し)の部分集合なので、この合計は同じ当たり組を二重に数えている。
    #    読者に出すのは**コースごとの点数**（courses[].points）。
    result['total_points'] = sum(b['points'] for b in result['bets'])
    result['flat_unit_total'] = result['total_points'] * 100

    base = dict(ALLOC_STRONG) if v == 'strong' else dict(ALLOC_LIGHT)
    courses, alternatives = build_courses(result['bets'], base)
    result['courses'] = courses
    result['alternatives'] = alternatives
    # コースが1つしか組めないとき（相手が1頭も取れない等）に
    # 「どちらか一方を選ぶ」と書くと意味が通らない。
    result['course_note'] = COURSE_NOTE if len(courses) >= 2 else None
    result['alternative_note'] = ALTERNATIVE_NOTE if alternatives else None
    # 🔴 2026-10-26: 全券種にまたがる配分は作らない。コースは選ぶものなので、
    #    またいだ比率を出すと結局「全部買う」形に戻る。配分は各コースが持つ。
    #    キー自体は残す（アーカイブの _FROZEN_KEYS と既存スナップショットの互換）。
    result['allocation_ratio'] = {}
    result['allocation_examples'] = {}
    result['allocation_note'] = ALLOCATION_NOTE
    result['bet_type_reasons'] = bet_type_reason(result)
    return result


def bet_type_count(tickets):
    """並べた買い目の行数（券種の数え方は group ではなく表示行）。"""
    return len({b['type'] for b in tickets.get('bets', [])})


def bet_group_count(tickets):
    """券種グループの数（三連複と三連複(2頭軸)は1つと数える）。"""
    return len({b.get('group') or b['type'] for b in tickets.get('bets', [])})
