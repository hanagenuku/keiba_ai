"""展開（隊列・ペース）の予測に使う特徴量を、学習と推論で**同じ関数**から作る。

なぜ作り直したか
----------------
既存の `train_pace_model.py` は19特徴量でペースを3分類していたが、
**学習時と推論時で入力が別物**だった（2026-09-14 に実測）:

    学習時  running_style = そのレースの実際の脚質 / agari3f = そのレースの実測値
    推論時  running_style = 過去走からの推定       / agari3f = 定数 36.0, 1.5

推定脚質と実際の脚質の一致率は **39.8%**。19特徴量のうち6つがこの経路にあり、
3分類の精度は 学習時入力 52.80% に対し **推論時入力では 50.45%** だった。

さらに「逃げ／先行／差し／追込」の4分類ラベル自体が表現として悪い（ラベルが
距離帯×表面のパーセンタイルなので、距離そのものが最重要特徴量になってしまう）。
そこで**前半600m を実数で予測する**形に作り直した。

🔴 その過程で2つ目の defect を踏んだ。目的変数にしていた `race_history.first_3f`
は「記録された先頭3区間の和」で、**端数区間を記録する回としない回があるため
同じ距離で二峰**になっていた（下の early_pace_seconds を参照）。
揃えた目的変数で測り直すと、レース構成（P(先頭)の分布・過去の通過位置・
過去の上がり）を足すと**2窓とも悪化**した:

    条件だけ（距離・表面・クラス・頭数）   窓A 0.8272s / 窓B 0.8325s
    + P(先頭)由来                     +0.0145 / +0.0077
    + 過去の通過位置                    +0.0121 / +0.0167
    + 過去の上がり                     +0.0130 / +0.0051
    + 全部                          +0.0151 / +0.0210

⚠ 初版のこの docstring には「作り直した連続量は反則版（そのレースの実際の脚質を
  知っている）の 2.3倍良い」と書いてあったが、**壊れた目的変数の産物だったので
  取り消した**。詳細は tenkai/RESULTS.md。

→ ペースは**レース条件だけ**で決める。副産物として学習時と推論時の入力が
  構造的に同一になり、上のパリティ違反が起こせなくなる。
  本モジュールの馬ごとの量（P(先頭)・隊列争い）は**ペース予測には使わず**、
  「逃げそうな馬は距離が持つか／他の馬にどう影響するか」を本体モデルで
  測るための入力として残す。

設計
----
学習も推論も `summarize_horse_shape()` → `race_shape_features()` の
**同じ経路**を通す。入力の出どころ（history.db か 出馬表＋履歴か）だけが違う。
これはこのプロジェクトで繰り返し起きた学習/推論パリティ違反を、
構造的に起こせなくするための作りである。
"""
import math

# JRAのラップタイムは「スタート→最初の200m標識」が端数区間になる。
# 1700m なら 100m + 200m×8。この端数を無視して先頭3区間を足すと、
# 同じ距離でも記録の仕方で 5秒以上ずれる（2026-09-14 実測）。
LAP_SEGMENT_M = 200.0
EARLY_SECTION_M = 600.0


def parse_lap_times(lap_times):
    """'7.1-11.1-12.4-...' を float のリストにする。壊れていれば空リスト。"""
    if not lap_times:
        return []
    out = []
    for tok in str(lap_times).split('-'):
        try:
            v = float(tok)
        except (TypeError, ValueError):
            return []
        if v <= 0:
            return []
        out.append(v)
    return out


def early_pace_seconds(lap_times, distance, n_seg=3):
    """前半3ハロン（600m）の所要秒。**端数区間の有無を吸収して揃える。**

    🔴 既存の race_history.first_3f は「記録された先頭3区間の和」で、
       同じ距離でも記録の仕方で別の量になっていた（2026-09-14 実測）:

           ダート1700m  n=9  lap 7.1-11.1-12.4-…  合計108.3s  先頭100mを記録
                       n=8  lap 11.5-12.1-13.1-… 合計100.1s  先頭100mを記録しない
           first_3f     端数側 30.05s (248R) / フル側 35.53s (101R)   差 5.48s

       前半3F の std は 1.85s なのでこの差は致命的。さらに端数記録の割合は
       2025-07 の 25.4% から 2026-08 の 0% へ変わっており、学習期と検定期で
       別の量になる。実際、first_3f を目的変数にすると検定期の RMSE が
       2.2590s（平均を答える 1.5288s より悪い）まで劣化した。

    ここでは区間数から端数の有無を判定し、**どちらも「最初の200m標識からの
    3区間」**に揃える。1700m ならどちらも 100m→700m の区間になる。
    """
    laps = parse_lap_times(lap_times)
    if not laps or not distance or distance <= 0:
        return None
    rem = int(distance) % int(LAP_SEGMENT_M)
    n_with_partial = int(distance) // int(LAP_SEGMENT_M) + (1 if rem else 0)
    # 端数区間が記録されているのは、区間数が端数込みの本数と一致するときだけ
    start = 1 if (rem and len(laps) == n_with_partial) else 0
    seg = laps[start:start + n_seg]
    if len(seg) < n_seg:
        return None
    return round(sum(seg), 3)




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


# ─────────────────────────────────────────────────────────────────────
# ペース（前半600m）モデルの入出力。**学習も推論もここを通す。**
#
# 🔑 旧モデルのパリティ違反（学習=実際の脚質 / 推論=推定脚質・一致率39.8%）を
#    構造的に起こせなくするため、入力はレース条件だけに限り、
#    その組み立てをこの1関数に閉じ込める。
# ─────────────────────────────────────────────────────────────────────
PACE_INPUT_COLS = ['dist', 'surface_num', 'cls', 'n_horses']

DIST_ZONES = [(0, 1400, '~1400'), (1401, 1800, '1401-1800'),
              (1801, 2200, '1801-2200'), (2201, 9999, '2201~')]

# クラスは序数。カテゴリ番号にするとDBと出馬表で採番がずれるので使わない。
# （engine.calc_features_for_xgb の f_class_level と同じ対応表）
CLASS_LEVEL = {'新馬': 1, '未勝利': 2, '1勝': 3, '1勝クラス': 3, '2勝': 4,
               '2勝クラス': 4, '3勝': 5, '3勝クラス': 5, 'OP': 6, 'オープン': 6,
               'L': 7, 'G3': 8, 'G2': 9, 'G1': 10}
DEFAULT_CLASS_LEVEL = 3


def dist_zone(distance):
    d = int(distance or 0)
    for lo, hi, label in DIST_ZONES:
        if lo <= d <= hi:
            return label
    return '2201~'


def class_level(race_class):
    return float(CLASS_LEVEL.get((race_class or '').strip(), DEFAULT_CLASS_LEVEL))


def pace_model_inputs(distance, surface, race_class, n_horses):
    """ペースモデルへ渡す1行。学習・推論ともこの関数で作る。"""
    return {
        'dist': float(distance or 1600),
        'surface_num': 1.0 if surface == '芝' else 0.0,
        'cls': class_level(race_class),
        'n_horses': float(n_horses or 0) or 14.0,
    }


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pace_probs_from_seconds(pred_seconds, sigma, thresholds):
    """予測した前半600m（秒）を high/mid/slow の確率に直す。

    前半が**速い**＝秒数が小さい＝ハイペース。
    thresholds は [33%点, 67%点]（学習期の同じ距離帯×表面で算出）。
    ハードな3値ではなく正規分布で確率にするので、閾値ぎりぎりでも極端にならない。
    """
    if pred_seconds is None or not thresholds or len(thresholds) < 2:
        return None
    s = max(float(sigma or 0.0), 0.05)
    q33, q67 = float(thresholds[0]), float(thresholds[1])
    p_high = _norm_cdf((q33 - pred_seconds) / s)
    p_slow = 1.0 - _norm_cdf((q67 - pred_seconds) / s)
    p_mid = max(0.0, 1.0 - p_high - p_slow)
    tot = p_high + p_mid + p_slow or 1.0
    return {'high': round(p_high / tot, 3),
            'mid': round(p_mid / tot, 3),
            'slow': round(p_slow / tot, 3)}
