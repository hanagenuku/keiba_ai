"""展開（隊列・ペース）の予測に使う特徴量を、学習と推論で**同じ関数**から作る。

なぜ作り直したか
----------------
既存の `train_pace_model.py` は19特徴量でペースを3分類していたが、
**学習時と推論時で入力が別物**だった（2026-09-14 に実測）:

    学習時  running_style = そのレースの実際の脚質 / agari3f = そのレースの実測値
    推論時  running_style = 過去走からの推定       / agari3f = 定数 36.0, 1.5

推定脚質と実際の脚質の一致率は **39.8%**。19特徴量のうち6つがこの経路にあり、
3分類の精度は 学習時入力 52.80% に対し **推論時入力では 50.45%** だった。

さらに「逃げ／先行／差し／追込」の4分類ラベル自体が表現として悪い。
前半3F を実数で予測して測ると（学習2025 / 検定2026・完全OOS・2,082レース）:

    条件だけ（距離・表面・馬場・頭数・クラス）  RMSE 1.5730s
    + 既存の脚質カウント                   RMSE 1.5439s  (-0.0290s)
    + 本モジュールの連続量                  RMSE 1.5071s  (-0.0659s)
    + そのレースの**実際の脚質**（反則・天井）   RMSE 1.5447s  (-0.0283s)

**事後の正解ラベルを知っているより、本モジュールのほうが 2.3倍良い。**
展開が説明する分散は R² 0.2891 → 0.3474（展開由来 2.6% → 5.8%）。

設計
----
学習も推論も `summarize_horse_shape()` → `race_shape_features()` の
**同じ経路**を通す。入力の出どころ（history.db か 出馬表＋履歴か）だけが違う。
これはこのプロジェクトで繰り返し起きた学習/推論パリティ違反を、
構造的に起こせなくするための作りである。
"""
import math

# 「ハナを切った経験がある」とみなす正規化位置のしきい値
CAN_LEAD_POS = 0.10
# 「先行型」とみなす正規化位置のしきい値
FRONT_POS = 0.30
DEFAULT_POS = 0.5          # 履歴が無い馬の位置（レース中央）
DEFAULT_AGARI = 36.0

HORSE_KEYS = ('p_led', 'p_mean', 'p_min', 'p_last', 'p_std', 'p_ag', 'p_n')
RACE_FEATURE_COLS = [
    'P_max', 'P_2nd', 'P_3rd', 'P_ent', 'P_hhi', 'P_gap', 'P_top2',
    'n_can_lead', 'n_front', 'pos_mean', 'pos_min', 'pos_std',
    'ag_mean', 'ag_min', 'pop_of_leader', 'n_horses',
]


def first_corner_position(corner_all, field_size):
    """corner_all('3-3-2-1') の**最初の**通過順を 0(先頭)〜1(最後方) に正規化する。

    ⚠ 4地点そろうのは 42.9% で、残りは3-4角のみ（＝「最初の」が3角）。
       どちらでも「その馬が前にいたか」を表すので同じ列として扱う。
    """
    if not corner_all or not field_size or field_size <= 1:
        return None
    head = str(corner_all).split('-')[0].strip()
    try:
        pos = float(head)
    except (TypeError, ValueError):
        return None
    if pos <= 0 or pos > field_size:
        return None
    return (pos - 1) / (field_size - 1)


def summarize_horse_shape(past_runs):
    """1頭ぶんの過去走から展開サマリを作る。

    past_runs は**そのレースより前**の走りだけを渡すこと（呼び出し側の責任）。
    各要素は corner_all / field_size(or finishers) / agari3f を持つ dict。
    """
    positions, agari, led = [], [], []
    for r in past_runs or []:
        fs = r.get('field_size') or r.get('finishers') or r.get('num_finishers')
        p = first_corner_position(r.get('corner_all'), fs)
        if p is not None:
            positions.append(p)
            led.append(1.0 if p == 0.0 else 0.0)
        a = r.get('agari3f')
        try:
            a = float(a)
            if a > 0:
                agari.append(a)
        except (TypeError, ValueError):
            pass

    n = len(positions)
    if n == 0:
        return {'p_led': None, 'p_mean': None, 'p_min': None, 'p_last': None,
                'p_std': None, 'p_n': 0,
                'p_ag': (sum(agari) / len(agari)) if agari else None}
    mean = sum(positions) / n
    var = sum((x - mean) ** 2 for x in positions) / (n - 1) if n > 1 else 0.0
    return {
        'p_led': sum(led) / n,
        'p_mean': mean,
        'p_min': min(positions),
        'p_last': positions[-1],
        'p_std': math.sqrt(var),
        'p_n': n,
        'p_ag': (sum(agari) / len(agari)) if agari else None,
    }


def race_shape_features(summaries, lead_probs, popularities=None):
    """レース単位の「隊列争い」特徴量。

    summaries   : summarize_horse_shape() の出力リスト（出走馬ぶん）
    lead_probs  : 各馬が先頭に立つ確率（正規化前でよい。ここで合計1に直す）
    popularities: 各馬の人気（無ければ None）

    🔑 「逃げ馬が何頭いるか」を数える代わりに、**P(先頭) の分布の形**で表す。
       単騎逃げ（P_gap 大）とやり合い（P_ent 大）が別の値になる。
    """
    n = len(summaries)
    if n == 0:
        return {c: None for c in RACE_FEATURE_COLS}

    tot = sum(p for p in lead_probs if p) or 1.0
    P = sorted((p or 0.0) / tot for p in lead_probs)[::-1]
    ent = -sum(p * math.log(p + 1e-12) for p in P)
    hhi = sum(p * p for p in P)

    pos = [s['p_mean'] for s in summaries if s.get('p_mean') is not None]
    pmin = [s['p_min'] for s in summaries if s.get('p_min') is not None]
    ag = [s['p_ag'] for s in summaries if s.get('p_ag') is not None]
    pos_mean = sum(pos) / len(pos) if pos else DEFAULT_POS
    pos_var = (sum((x - pos_mean) ** 2 for x in pos) / (len(pos) - 1)) if len(pos) > 1 else 0.0

    pop_leader = None
    if popularities:
        i = max(range(n), key=lambda k: lead_probs[k] or 0.0)
        p = popularities[i]
        if p and 0 < p < 99:
            pop_leader = float(p)

    return {
        'P_max': P[0],
        'P_2nd': P[1] if n > 1 else 0.0,
        'P_3rd': P[2] if n > 2 else 0.0,
        'P_ent': ent,
        'P_hhi': hhi,
        'P_gap': P[0] - (P[1] if n > 1 else 0.0),
        'P_top2': P[0] + (P[1] if n > 1 else 0.0),
        'n_can_lead': sum(1 for x in pmin if x < CAN_LEAD_POS),
        'n_front': sum(1 for x in pos if x < FRONT_POS),
        'pos_mean': pos_mean,
        'pos_min': min(pos) if pos else DEFAULT_POS,
        'pos_std': math.sqrt(pos_var),
        'ag_mean': sum(ag) / len(ag) if ag else DEFAULT_AGARI,
        'ag_min': min(ag) if ag else DEFAULT_AGARI,
        'pop_of_leader': pop_leader,
        'n_horses': n,
    }
