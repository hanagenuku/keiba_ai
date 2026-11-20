"""☆マーク（市場1番人気を着順で上回る可能性が比較的高い馬）の推論。

## 何を出すか

各レースで「**予測時点の市場1番人気（P1）を着順で上回る確率**」を馬ごとに出し、
その最大値が `STAR_TAU` 以上なら、その1頭に ☆ を付ける。

🔴 **☆は表示専用。** ◎○▲△・`rl_rank`・買い目・確率・推奨の範囲（`article_kind`）を
1つも変えない。買い目（`axis_bets.py`）にも入れない。

🔴 **☆は買い推奨ではない。** 記事・アプリに出すときは必ず併記する:
> ☆は着順で1番人気を上回る可能性についての表示です。過去データでは、この表示が
> 出た馬は1番人気より3着内に入りやすい傾向がありました。一方で**回収率は改善して
> いません**（配当が小さくなるため）。買い目の根拠ではありません。
> **☆の本数は週によって変わります。**

## 根拠（事前登録と実測）

- 事前登録: `hitrate/CRITERIA_C_RERUN.md` / `hitrate/CRITERIA_STAR.md`
- 実測: `hitrate/RESULTS_C_RERUN.md` / `hitrate/RESULTS_STAR.md`
- 未読の 2024年で測った S-1（τ=0.60）の成績:
    実測「P1を上回った率」 0.6413 ／ 複勝率 56.07% vs P1 35.04%（**+21.03pt**）
    複勝回収 78.82% vs 77.87%（+0.95pt）／ 3本抜き +12.50pt ／ 119レース（4.02%）
  2025年・2026年でも「上回った率」は 0.6407 / 0.6294 でほぼ動かない。
- 🔴 事前登録の基準④（頻度の安定）は**落ちた**（3.61% → 18.72% と5倍動く）。
  2026-11-11 にユーザーが「頻度が変わることを明示するなら進める」と判断したので、
  ④を**商品側の条件（本数が変わる旨の明示）に置き換えて**実装している。
  他の6基準は、測定時点で未読だった 2024年で合格している。
- 🔴 ◎を差し替える案は `hitrate/RESULTS_C.md` で**不採用**（複勝回収率が改善しない）。

## 動かしてはいけない値

`STAR_TAU = 0.60` は事前登録した閾値。**結果を見てから動かさない。**
τ=0.55 も5基準に合格しているが、実測 58.8% は「比較的高い」と書けないので採らない
（`hitrate/RESULTS_C_RERUN.md` §5）。
"""
import json
import os

import numpy as np

# 🔴 事前登録した閾値。動かさない（`hitrate/CRITERIA_STAR.md` §1）
STAR_TAU = 0.60
# 欠損の埋め方。学習（train_star.py）と推論で**同じ値**でなければならない
STAR_FILL = 5.0

_MODEL = None
_ISO = None
_COLS = None
_DIR = None

# 記事・アプリに出すときの併記文（省略不可。`biz/S4_HITRATE_AXIS.md`）
# 🔴 2026-11-18 ユーザー指示「(気になる一頭) ぐらいのニュアンスでいい」で書き換えた。
#    旧文の「1番人気より3着内に入りやすい傾向がありました」は、評価で使った1番人気が
#    作り物（実際の1番人気と一致 27〜30%）で、実際の1番人気に対しては裏付けが無かった
#    （biz/RESULTS_HONMEI_STAR.md §2）。成績・確率の主張は書かない。
STAR_NOTE_V1 = (
    '☆は、AIが気になる一頭として挙げた馬です。'
    '当たりやすさや回収率の裏付けがある印ではなく、回収率は改善していません。'
    '買い目の根拠ではありません。☆の本数は週によって変わります。'
)
# 🟢 2026-11-20: v2 の測定（hitrate/RESULTS_STAR_V2.md・2025-01〜2026-11 の4窓）で言えることだけを書く。
#    数字は区間の下端に寄せて丸めた「約」（先着 +5.41pt [3.60, 7.15]・3着内 +3.44pt [1.64, 5.20]）。
STAR_NOTE_V2 = (
    '☆は、AIが気になる一頭として挙げた馬です。'
    '過去の検証（2025年1月〜2026年11月）では、☆の馬は同じ人気の馬と比べて、'
    '1番人気に先着した割合が約4ポイント、3着内率が約2ポイント高めでした。'
    'ただし回収率は改善していません。買い目の根拠ではありません。☆の本数は週によって変わります。'
)

# 2026-11-20 STAR-NARROW（hitrate/RESULTS_STAR_NARROW.md）: 候補を4〜9番人気に限った☆。
# 書けるのは合格した「先着」だけ（3着内は報告のみの統計量）。数字は対照群Aの平均を引いて切り下げた値。
STAR_NOTE = (
    '☆は、AIが4〜9番人気の中から気になる一頭として挙げた馬です。'
    '過去の検証（2024年3月〜2026年11月）では、☆の馬は同じ人気の馬と比べて、'
    '1番人気に先着した割合が約3ポイント高めでした。'
    'ただし回収率は改善していません。買い目の根拠ではありません。☆の本数は週によって変わります。'
)


def star_note(meta):
    """成果物に合った併記文（省略不可）。"""
    if int(meta.get('version', 1)) < 2:
        return STAR_NOTE_V1
    return STAR_NOTE if meta.get('pop_lo') is not None else STAR_NOTE_V2


def _num(v, col=''):
    """特徴量1つを float にする。欠損は STAR_FILL。web 由来の列（f_ws_*）だけは NaN のまま。

    🔴 学習側（`matrix_from_frame`）と同じ扱いにすること（本番の `web_signals.fill_missing` と同じ方針）。
    """
    web = str(col).startswith('f_ws_')
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float('nan') if web else STAR_FILL
    if f != f:
        return float('nan') if web else STAR_FILL
    return f


def to_matrix(feat_dicts, cols):
    """特徴量 dict のリスト → (n, len(cols)) の行列。

    🔴 **学習と推論はこの関数だけを使う**（列順・欠損の埋め方を1か所に閉じる）。
    `src/tools/train_star.py` もこれを import している。
    """
    return np.array([[_num(d.get(c), c) for c in cols] for d in feat_dicts],
                    dtype=np.float32)


def matrix_from_frame(df, cols):
    """学習側（DataFrame）用の行列化。`to_matrix` と**同じ値**を返さなければならない。

    🔴 推論側は `to_matrix`（dict のリスト）、学習側はこちら（DataFrame）を使う。
    経路が2つあるので、**同値であることを `tests/test_star_parity.py` が固定している**
    （North Star #3: 学習/推論パリティ。崩れると片方だけ静かに 5.0 になる）。
    """
    import pandas as pd
    m = df.reindex(columns=list(cols)).apply(pd.to_numeric, errors='coerce')
    keep = [c for c in m.columns if str(c).startswith('f_ws_')]
    rest = [c for c in m.columns if c not in keep]
    m[rest] = m[rest].fillna(STAR_FILL)
    return m.to_numpy(dtype=np.float32)


def p1_index(pops):
    """予測時点の市場1番人気の添字。1頭に定まらなければ None。

    🔴 学習時は「ドリフト注入後の人気順位」、推論時は「当日オッズから作った人気順位」。
    どちらも「そのレースで市場が最も支持している1頭」という同じ役回りで、
    `base_margin` と同じ扱い（North Star #1 で許されている市場の使い方）。
    """
    hit = [i for i, p in enumerate(pops) if p is not None and int(p or 0) == 1]
    return hit[0] if len(hit) == 1 else None


def build_delta(X, i1):
    """P1 を基準にした差分。☆モデルの特徴量はこれだけ（オッズ・人気は入れない）。"""
    return (X - X[i1]).astype(np.float32)


def load_star_model(base_dir):
    """`data/star_model.ubj` / `star_calibrator.pkl` / `star_feature_cols.json` を読む。

    1つでも無ければ (None, None, None) を返す（☆を出さない）。
    """
    global _MODEL, _ISO, _COLS, _DIR
    if _DIR == base_dir and _COLS is not None:
        return _MODEL, _ISO, _COLS
    mp = os.path.join(base_dir, 'data', 'star_model.ubj')
    cp = os.path.join(base_dir, 'data', 'star_calibrator.pkl')
    fp = os.path.join(base_dir, 'data', 'star_feature_cols.json')
    if not (os.path.exists(mp) and os.path.exists(cp) and os.path.exists(fp)):
        _DIR, _MODEL, _ISO, _COLS = base_dir, None, None, None
        return None, None, None
    try:
        import pickle

        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(mp)
        with open(cp, 'rb') as f:
            iso = pickle.load(f)
        with open(fp, encoding='utf-8') as f:
            cols = list(json.load(f)['feature_cols'])
    except Exception as e:                       # pragma: no cover
        print(f'  ⚠ ☆モデルの読み込みに失敗したので☆を出しません: {e}')
        _DIR, _MODEL, _ISO, _COLS = base_dir, None, None, None
        return None, None, None
    _DIR, _MODEL, _ISO, _COLS = base_dir, bst, iso, cols
    return bst, iso, cols


def star_probs(feat_dicts, pops, base_dir):
    """各馬の「P1 を着順で上回る較正済み確率」。P1 自身と算出不能な馬は None。

    返り値は feat_dicts と同じ長さのリスト。
    """
    n = len(feat_dicts)
    bst, iso, cols = load_star_model(base_dir)
    if bst is None or n < 2:
        return [None] * n
    i1 = p1_index(pops)
    if i1 is None:
        return [None] * n
    import xgboost as xgb
    X = to_matrix(feat_dicts, cols)
    d = build_delta(X, i1)
    others = [i for i in range(n) if i != i1]
    # 🔴 木の本数を版の既定に任せない（`tests/test_predict_tree_count.py` と同じ方針）
    raw = bst.predict(xgb.DMatrix(d[others], feature_names=list(cols)),
                      iteration_range=(0, bst.num_boosted_rounds()))
    cal = (iso['iso1'] if isinstance(iso, dict) else iso).predict(raw)
    out = [None] * n
    for k, i in enumerate(others):
        out[i] = float(cal[k])
    return out


def pick_star(nums, probs, tau=STAR_TAU):
    """☆を付ける1頭。該当なしなら (None, None)。

    同値は馬番の小さい方（意味を持たない中立な順序）。
    """
    cand = [(p, nm) for nm, p in zip(nums, probs) if p is not None and p >= tau]
    if not cand:
        return None, None
    p, nm = max(cand, key=lambda x: (x[0], -x[1]))
    return nm, round(float(p), 4)


# ── v2（2026-11-20・hitrate/CRITERIA_STAR_V2.md）────────────────────────────
# ☆ = S1（実際の1番人気に先着する較正済み確率）が τ1 以上の馬。いなければ
#     S2（本番モデルの3着内確率 − 市場だけの3着内確率・どちらも較正済み）が τ2 以上の馬。
# τ1・τ2 は学習の内側ホールドアウトで「1番人気以外の最大値の85パーセンタイル」（star_feature_cols.json）。
# v1 の成果物（較正器が1つ・`star_tau`）しか無いときは従来どおり S1 >= STAR_TAU だけで選ぶ。

def load_star_meta(base_dir):
    """star_feature_cols.json の中身（無ければ {}）。"""
    fp = os.path.join(base_dir, 'data', 'star_feature_cols.json')
    try:
        with open(fp, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def star_s2(raw_margins, base_margins, pops, base_dir):
    """各馬の S2（3着内の上積み）。1番人気・算出不能な馬・v1 の成果物のときは None。"""
    n = len(raw_margins)
    bst, cal, cols = load_star_model(base_dir)
    if not isinstance(cal, dict) or 'cm' not in cal:
        return [None] * n
    i1 = p1_index(pops)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    out = [None] * n
    for i, (r, b) in enumerate(zip(raw_margins, base_margins)):
        if i == i1 or r is None or b is None:
            continue
        pm = float(cal['cm'].predict([sig(float(r))])[0])
        pk = float(cal['ck'].predict([sig(float(b))])[0])
        out[i] = pm - pk
    return out


def pick_star_v2(nums, s1, s2, base_dir, pops=None):
    """v2 の ☆。返り値は (馬番, 種類 'S1'/'S2', S1 の値)。該当なしなら (None, None, None)。

    成果物に `pop_lo`/`pop_hi` があれば（STAR-NARROW・2026-11-20）候補をその人気の馬に限る。
    人気が分からない馬は候補にしない。
    """
    meta = load_star_meta(base_dir)
    if int(meta.get('version', 1)) < 2:
        num, p = pick_star(nums, s1)
        return (num, 'S1', p) if num is not None else (None, None, None)
    t1, t2 = float(meta['tau1']), float(meta['tau2'])
    if meta.get('pop_lo') is not None:
        lo, hi = int(meta['pop_lo']), int(meta['pop_hi'])
        pops = list(pops) if pops is not None else [None] * len(nums)
        ok = [pp is not None and lo <= int(pp) <= hi for pp in pops]
        s1 = [v if o else None for v, o in zip(s1, ok)]
        s2 = [v if o else None for v, o in zip(s2 or [None] * len(nums), ok)]
    c1 = [(p, -nm, nm) for nm, p in zip(nums, s1) if p is not None]
    if c1:
        p, _, nm = max(c1)
        if p >= t1:
            return nm, 'S1', round(float(p), 4)
    c2 = [(q, -nm, nm) for nm, q in zip(nums, s2 or [None] * len(nums)) if q is not None]
    if c2:
        q, _, nm = max(c2)
        if q >= t2:
            p = dict(zip(nums, s1)).get(nm)
            return nm, 'S2', (round(float(p), 4) if p is not None else None)
    return None, None, None
