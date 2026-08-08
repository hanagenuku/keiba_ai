"""軸1頭ベースの買い目生成（2026-08-08 導入）

## なぜ作ったか

従来の 📊EV買い目（`bet_optimizer.build_optimal_bets`）は
「AI確率 ÷ 市場確率」が最大の組み合わせを選ぶ設計だった。
これは **AIと市場が最も食い違う買い目を選ぶ操作**であり、AIが市場より
正確でない限り「自分の誤りを最大化して買う」ことになる。

実測（本番が実際に出した予想401レース × race_dividends の実配当）:

| 買い方 | 点数 | 的中率 | 回収率 |
|---|---|---|---|
| 従来のEV方式（馬連） | 1,166 | **2.7%** | 68.7% |
| 軸(RL1) 単勝 | 386 | 24.6% | 79.5% |
| 軸 × 中穴 ワイド1点 | 343 | **15.7%** | 102.2% |

従来方式は**試した中で的中率が最下位**で、回収率でも上回っていなかった
（どの軸でも他に負けている ＝ 支配されている状態）。

## 構成（ユーザー指定・2026-08-08）

    ① 単勝    軸(RL1)                          1点
    ② ワイド  軸 × 中穴（3着内確率順）             最大3点
    ③ 馬連    軸 × 中穴                          合成オッズ >= 3.0 を保つ範囲
    ④ 三連複  A - 中穴2頭 - (RL2..6 ＋ 中穴)      最大24点

中穴 = 単勝 7〜30倍 のうち、3着内確率(`cal_prob`)が高い順。

⚠ **三連複の2列目に上位人気(RL2,RL3)を置くと崩れる**ことを実測した:

| 三連複の組み方 | 回収率 | 高配当1本抜き |
|---|---|---|
| A - **RL2,RL3** - RL2..10 | 56.3% | 52.2% |
| A - **中穴2** - RL2..6＋中穴 | 157.8% | 107.0% |

11通り試したうちの向きとしてはこれだけが一貫していたため2列目は中穴とする。

## ⚠ 回収率は控除率を超えていない（採用の根拠にしないこと）

上記の100%超は**高配当1〜2本に依存**しており、頑健性検査を通っていない:

    A - 中穴2 - RL2..6＋中穴 : 回収率157.8%
      高配当2本を抜くと      : 61.6%
      前半 266.9% → 後半      : 49.3%
      プラスだった開催日      : 4/12日   日別中央値 63.7%
      95%信頼区間(日ブロック) : [56.7%, 307.3%]

これは 2026-07-27 の買い目フィルタ（本番76点で182.8% → 2,000レースで71.3%）と
同じ壊れ方。**この構成を採用する理由は「儲かるから」ではなく
「従来方式より的中率が6倍良く、買う理由が見えるから」**である。
表示には必ず「検証中」と明記すること。
"""

# ── 中穴の定義 ───────────────────────────────────────────────────────────
# 実測したオッズ帯（ワイド1点の回収率 / 高配当1本抜き）:
#   3〜7倍   71.8% / 67.9%     7〜15倍  99.2% / 82.6%
#   7〜30倍 102.5% / 87.4%    15〜50倍  57.9% / 51.5%
# 7〜30倍と10〜30倍がほぼ同値で、7〜30倍のほうが的中率が高い（15.8% vs 12.4%）
ANA_ODDS_MIN = 7.0
ANA_ODDS_MAX = 30.0

WIDE_MAX_POINTS = 3          # ユーザー指定: 3点まで（1点でも良い）
TRIO_MAX_POINTS = 24         # ユーザー指定: 最大24点くらいまで
TRIO_SECOND_LEG = 2          # 三連複2列目（中穴）の頭数
# 3列目に入れるRL順位の上限。ユーザー提案の A-BC-BCDEFGHIJ（3列目9頭）に対応。
# 実測（396レース）で狭い版と広い版を比べると:
#   RL2..6（平均7点）  : 回収157.8% / 高配当1本抜き107.0% / 3本抜き54.9% / ＋4-12日
#   RL2..9（平均13点） : 回収121.2% / 高配当1本抜き 93.4% / 3本抜き61.9% / ＋5-12日
# 見出しの回収率は狭い版が高いが、**より深い頑健性検査（3本抜き・プラスの日数）
# では広い版が上**。見出しの数字は高配当1本に依存しており信用できないため、
# 広い版を採る。ユーザー提案の形とも一致する。
TRIO_THIRD_LEG_RL_MAX = 9

# 馬連は「合成オッズがこれを下回らない範囲で何点でも」（ユーザー指定）。
# 合成オッズ = 1 / Σ(1/各オッズ)。当たっても投資割れする買い方（取りガミ）を防ぐ。
QUINELLA_MIN_SYNTHETIC_ODDS = 3.0

UNIT = 100


def _num(h):
    """馬番を取り出す（calc_all は 'num'、特徴量DFは 'horse_num'）。"""
    v = h.get('horse_num')
    if v is None:
        v = h.get('num')
    return int(v) if v is not None else None


def _fuku_prob(h):
    """3着内確率。calc_all のキャリブレーション済み cal_prob を優先する。"""
    for k in ('cal_prob', 'fuku_prob', 'top3_prob'):
        v = h.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _win_odds(h):
    try:
        v = float(h.get('win_odds') or 0)
    except (TypeError, ValueError):
        return 0.0
    return v


def synthetic_odds(odds_list):
    """合成オッズ = 1 / Σ(1/各オッズ)。

    10倍・15倍・20倍の3点なら 1/(0.1+0.0667+0.05) = 4.6倍。
    ここに4倍を足すと 2.4倍まで落ちるので、閾値3.0なら足さない。
    """
    inv = sum(1.0 / o for o in odds_list if o and o > 0)
    return (1.0 / inv) if inv > 0 else 0.0


def select_axis(scored):
    """軸 = RL1（能力・脚質・展開はXGBの129特徴量に既に織り込まれている）。"""
    for h in scored:
        if h.get('rl_rank') == 1:
            return h
    return None


def select_anas(scored, axis_num, n,
                lo=ANA_ODDS_MIN, hi=ANA_ODDS_MAX):
    """中穴を3着内確率の高い順に n 頭返す。

    「2,3着まで入線の確率が高い中穴馬」というユーザーの指定をそのまま実装する。
    """
    cand = [h for h in scored
            if _num(h) != axis_num and lo <= _win_odds(h) < hi]
    cand.sort(key=_fuku_prob, reverse=True)
    return cand[:n]


def _entry(tag, label, key, prob, odds, amount=UNIT):
    return {'tag': tag, 'label': label, 'key': key,
            'prob': prob, 'odds': odds, 'amount': amount,
            'ev': round(prob * odds, 2) if odds else 0.0}


def build_axis_bets(scored, probs, odds_map,
                    wide_points=WIDE_MAX_POINTS,
                    trio_max_points=TRIO_MAX_POINTS):
    """軸1頭ベースの買い目を組む。

    Parameters
    ----------
    scored   : calc_all の出力（rl_rank / win_odds / cal_prob を持つ）
    probs    : calc_ticket_probabilities の出力（win/place/wide/quinella/trio）
    odds_map : estimate_payouts_from_win_odds の出力（推定配当）

    Returns
    -------
    dict: {'win': [...], 'wide': [...], 'quinella': [...], 'trio': [...],
           'summary': {...}}
    空のレース（軸が取れない・中穴がいない等）では該当券種を空で返す。
    """
    out = {'win': [], 'wide': [], 'quinella': [], 'trio': [], 'summary': {}}
    axis = select_axis(scored)
    if not axis:
        return out
    a = _num(axis)
    if a is None:
        return out

    def pair_key(x, y):
        return tuple(sorted((int(x), int(y))))

    # ── ① 単勝: 軸1点 ──────────────────────────────────────────────────
    w_odds = _win_odds(axis)
    w_prob = (probs.get('win', {}) or {}).get(a)
    if w_odds > 0:
        out['win'].append(_entry('tan', '単勝', a,
                                 round(float(w_prob), 4) if w_prob else 0.0,
                                 round(w_odds, 1)))

    # ── ② ワイド: 軸 × 中穴（3着内確率順） ───────────────────────────────
    anas = select_anas(scored, a, wide_points)
    for h in anas:
        k = pair_key(a, _num(h))
        p = (probs.get('wide', {}) or {}).get(k)
        o = (odds_map.get('wide', {}) or {}).get(k)
        if p and o:
            out['wide'].append(_entry('wide', 'ワイド', k, round(float(p), 4),
                                      round(float(o), 1)))

    # ── ③ 馬連: 合成オッズ >= 3.0 を保つ範囲で追加 ──────────────────────
    # 中穴を確率の高い順に並べ、足しても合成オッズが閾値を割らない間だけ買う。
    q_cands = []
    for h in select_anas(scored, a, n=len(scored)):
        k = pair_key(a, _num(h))
        p = (probs.get('quinella', {}) or {}).get(k)
        o = (odds_map.get('quinella', {}) or {}).get(k)
        if p and o:
            q_cands.append((float(p), float(o), k))
    q_cands.sort(reverse=True)          # 当たりやすい＝オッズの低い順から
    picked = []
    for p, o, k in q_cands:
        trial = [x[1] for x in picked] + [o]
        if synthetic_odds(trial) < QUINELLA_MIN_SYNTHETIC_ODDS:
            # これ以上足すと取りガミ側に寄るのでここで打ち切る
            break
        picked.append((p, o, k))
    for p, o, k in picked:
        out['quinella'].append(_entry('umaren', '馬連', k, round(p, 4), round(o, 1)))
    if picked:
        out['summary']['quinella_syn_odds'] = round(
            synthetic_odds([x[1] for x in picked]), 1)

    # ── ④ 三連複: A - 中穴2 - (RL2..6 ＋ 中穴) ─────────────────────────
    g2 = [_num(h) for h in select_anas(scored, a, TRIO_SECOND_LEG)]
    if len(g2) >= 2:
        g3 = [_num(h) for h in sorted(
                  (h for h in scored if h.get('rl_rank')),
                  key=lambda x: x['rl_rank'])
              if _num(h) != a and h['rl_rank'] <= TRIO_THIRD_LEG_RL_MAX]
        for x in g2:
            if x not in g3:
                g3.append(x)
        seen, combos = set(), []
        for x in g2:
            for y in g3:
                if x == y:
                    continue
                k = pair_key(x, y)
                if k in seen:
                    continue
                seen.add(k)
                t = tuple(sorted((a, *k)))
                p = (probs.get('trio', {}) or {}).get(t)
                o = (odds_map.get('trio', {}) or {}).get(t)
                combos.append((float(p or 0), float(o or 0), t))
        combos.sort(reverse=True)       # 当たりやすい順に上から採る
        for p, o, t in combos[:trio_max_points]:
            if p > 0 and o > 0:
                out['trio'].append(_entry('sanfuku', '三連複', t,
                                          round(p, 5), round(o, 1)))
        if out['trio']:
            odds_l = [b['odds'] for b in out['trio']]
            # 実際に採用された点だけから列を復元する。
            # 点数上限で切ると 2列目/3列目に一度も使われない馬が残るため、
            # 宣言した g2/g3 をそのまま出すと画面と買い目が食い違う。
            used = [set(b['key']) - {a} for b in out['trio']]
            leg2 = sorted(x for x in g2 if any(x in u for u in used))
            leg3 = sorted({n for u in used for n in u})
            out['summary'].update({
                'trio_legs':     [[a], leg2, leg3],
                'trio_points':   len(out['trio']),
                'trio_syn_odds': round(synthetic_odds(odds_l), 1),
                'payout_min':    int(min(odds_l) * UNIT),
                'payout_max':    int(max(odds_l) * UNIT),
            })

    pts = sum(len(out[k]) for k in ('win', 'wide', 'quinella', 'trio'))
    out['summary']['total_points'] = pts
    out['summary']['total_amount'] = pts * UNIT
    out['summary']['axis'] = a
    return out


def make_axis_bets(scored, race, base_dir=None, market_odds_map=None,
                   n_sims=3000):
    """軸1頭ベースの買い目を、シミュレーションと推定配当込みで組む。

    `make_bets_v2` と同じ入力で呼べるようにしてある。
    """
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
    from src.betting.payout_estimator import estimate_payouts_from_win_odds
    from src.betting.bet_optimizer import _load_gumbel_rating_temperature

    horse_nums = [_num(h) for h in scored]
    T = _load_gumbel_rating_temperature(base_dir) if base_dir else 2.5
    ratings = [float(h.get('rating', 0.0) or 0.0) / T for h in scored]
    orders = simulate_race(ratings, n_sims=n_sims)
    probs = calc_ticket_probabilities(orders, horse_nums)

    real = (market_odds_map or {}).get(race.get('id', ''), {})
    if real:
        win_odds = {n: i['tansho'] for n, i in real.items()
                    if i.get('tansho', 0) > 0}
    else:
        win_odds = {_num(h): _win_odds(h) for h in scored if _win_odds(h) > 0}
    odds_map = estimate_payouts_from_win_odds(win_odds, n_sims=min(n_sims, 10000))

    return build_axis_bets(scored, probs, odds_map), probs, odds_map
