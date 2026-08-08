"""Plackett-Luce レーティング（2026-08-08 導入）

## 何をするか

1レースの着順**全体**を1つの観測として扱い、各馬の潜在能力 θ を推定する。

    P(着順 π) = Π_k  exp(θ_π(k)) / Σ_{j>=k} exp(θ_π(j))

「Aに勝ったBに勝ったC」の連鎖から、**直接対戦していない馬同士も比較できる
1つの尺度**に載せる。

## なぜ要ったか（診断で特定した伸びしろ）

オラクル法（答えを覗く反則の計算で天井を測る）で、市場ゼロAIの弱点を特定した:

    市場ゼロAI（126特徴量）                 AUC 0.7706
    その馬の「真の複勝率」を知る（反則）       AUC 0.8740  ← 1つの数だけで
    ＋126特徴量                            AUC 0.9356
    （参考）市場の人気だけ                    AUC 0.8001

**126個の特徴量は「その馬がどれくらい強いか」という一点にすら負けていた。**
既存の `f_rl` / `f_speed_fig_avg` は過去走スコアの重み付き平均で、
**相手の強さを考慮していない**（強い相手への2着と弱い相手への2着が同じ扱い）。

なお、他の候補は同じ診断で先に潰してある:

| 案 | 実測 | 判定 |
|---|---|---|
| 学習データを増やす | 1年0.7651→2年0.7711→3年0.7710→3.5年0.7706 | 2年で飽和 |
| 騎手情報を厚くする | 各騎手の真の複勝率を完璧に知っても **-0.0002** | 既に天井 |
| 馬体重の欠損を埋める | 充足行だけで評価しても改善なし | ボトルネックでない |

## 実測の効果：小さいが一貫してプラス（5窓すべて）

本番パイプラインで作った学習データで walk-forward 測定（3シード平均）:

| 検証窓 | 本番構成 PLなし→あり | 市場ゼロAI PLなし→あり |
|---|---|---|
| 2025-09〜10 | 0.8142 → 0.8153 (+0.0011) | 0.7759 → 0.7795 (+0.0036) |
| 2026-02〜03 | 0.8224 → 0.8231 (+0.0007) | 0.7941 → 0.7981 (+0.0039) |
| 2026-05 | 0.7683 → 0.7757 (**+0.0074**) | 0.7859 → 0.7932 (**+0.0073**) |
| 2026-06 | 0.7989 → 0.7992 (+0.0003) | 0.7701 → 0.7730 (+0.0029) |
| 2026-07〜08 | 0.8034 → 0.8037 (+0.0003) | 0.7697 → 0.7751 (+0.0054) |
| **平均** | **+0.0020**（悪化 0/5窓） | **+0.0046**（悪化 0/5窓） |

**10窓すべてでプラス、符号が一度も反転しない。** シード揺らぎ(±0.0003〜0.0005)の
4〜6倍で、5窓連続同符号が偶然に出る確率は 1/32。小さいがノイズではない。

⚠ **市場ゼロAI(+0.0046)のほうが本番構成(+0.0020)より効く。**
市場（オッズ）が既に馬の実力をかなり織り込んでいるため、市場アンカーを
土台にすると θ が持ち込む情報の半分以上が重複する。

### なぜ効きが小さいのか — 既存特徴量との重複

| 既存特徴量 | θ との相関 |
|---|---|
| `f_recent`（近走成績） | **+0.682** |
| `f_agari_ability`（末脚） | **+0.664** |
| `f_speed_fig_avg` | +0.441 |
| `f_cl_rank` | -0.339 |

θ 単体は明確に強い（θ帯5分位で複勝率 7.4% → 38.7%、**5.2倍差**）。
しかし126特徴量と併用すると、新しく足せる分がほとんど残っていない。

診断で見えた天井（その馬の「真の複勝率」を知れば AUC 0.8740）に届かない理由は
2つ考えられ、**今のデータでは区別できない**:
1. 検証期の馬は**出走回数の中央値が7回**。7走からの推定が粗い → 溜まれば効く
2. 「真の複勝率」には未来（成長・仕上がり）が含まれ、過去走からは原理的に不可能

θ は週次で更新され続けるので、数ヶ月後に測り直せば 1 か 2 かが分かる。

## 🔴 一度「+0.037・回収率115%」と誤報告した。その原因（重要）

検証用に別途書いたスクリプトが θ を作る際、SQLが

    ORDER BY date, race_id, place      ← place = 着順

で、順位を `(-th).argsort().argsort()` で付けていた。**θが同値の馬
（初出走馬など）ではタイの並び順がそのまま配列順＝着順**になり、
`rl_f_pl_rating_rank` が実際の着順を漏らしていた。

| θが同値の行（22,207行・全体の14%）で | 順位 vs 実着順の相関 |
|---|---|
| 検証スクリプト版 | **+0.8678** ← 着順そのもの |
| 本番パイプライン版（このモジュール） | +0.1694（正常） |

⚠ **本モジュールの実装は最初から正しい**。`sorted(range(k), key=lambda i: -th[i])`
は呼び出し側の並び（`build_training_data` は `ORDER BY horse_num`）で
タイブレークするため着順は漏れない。壊れていたのは検証コードだけ。

### 🔴 プラセボ対照はリーク検出に使えない（この失敗の教訓）

当時「レース内でθを馬にランダムに振り直しても改善が出ないから、
構造的リークは無い」と結論したが**誤り**。プラセボは「馬固有の情報」を
壊す操作なので、**本物のシグナルもリークも同じように消える**。

同様に、当時実施した他の検査（初出走馬のθが厳密に0／同じ馬が同じ日に
複数走ってもθ不変／θと着順の相関 -0.357）は**すべて θ を見ており、
順位のタイブレークを一度も見ていなかった**。

**リーク検証は「同じ特徴量を別実装で作って値を突合する」のが確実。**
実際、本番パイプラインで作り直して初めて発覚した。

## モデルへの入り方（記憶に頼らない）

`train_xgb` は除外リスト以外の数値列を全部拾うため、
**`build_training_data()` を先に走らせれば PL 特徴量は自動的にモデルに入る**
（実測: 129 → 134特徴量）。「次回忘れずに追加する」という運用は不要。

唯一壊れる経路は「誰かが `_EXCLUDE_COLS` / `_MARKET_FEAT_COLS` に足す」ことで、
そこは `tests/test_pl_rating.py::TestPLFeaturesReachTheModel` で固定してある。

⚠ **PLのためだけに再学習はしない**（+0.002 に見合わない）。次回の再学習で
自然に入る。⚠ `train_xgb()` を単独実行すると古いCSVを読む点に注意
（2026-08-07③の記録参照）。

⚠ ハイパーパラメータ（LR / DECAY_DAYS / PRIOR_SHRINK）は**一度も振っていない**。
振る場合は多重比較に注意し、必ず本番パイプラインで測ること。

## 学習/推論パリティの担保

**読み取りコードを1本にする。** 学習も推論も
`calc_features_for_xgb` → `_PL_RATINGS` を読む、で同一。
違うのは**書き込み側**だけ:

- 推論: `data/pl_rating.pkl`（週次バッチが作った最新θ）をロード
- 学習: `build_training_data` が日付順に走りながら `advance_day()` で再現

これにより「片方だけ静かにデフォルト値化する」というこのプロジェクトで
繰り返し起きた事故（f_post / racecourse / agari_rank …）を構造的に防ぐ。
"""
import math
import os
import pickle
from collections import defaultdict

# ハイパーパラメータ。⚠ 一度も振っていない（探索は多重比較になるため固定）。
LR = 0.30            # 学習率。Eloの K-factor に相当
EXP_DAMP = 0.05      # 経験を積むほど動かなくする係数
PRIOR_SHRINK = 0.02  # 全体平均への引き戻し
DECAY_DAYS = 400.0   # 時間減衰の時定数（長期休養で確からしさが落ちる）
MIN_FIELD = 3        # これ未満の頭数では更新しない

FEATURE_COLS = [
    'f_pl_rating',          # θ（絶対値）
    'f_pl_rating_n',        # 出走回数＝推定の信頼度
    'rl_f_pl_rating',       # レース内平均との差
    'rl_f_pl_rating_z',     # 同 z値
    'rl_f_pl_rating_rank',  # レース内のθ順位（1=最上位）
]

RATING_FILE = 'pl_rating.pkl'


class PLRatings:
    """馬ごとの θ を保持し、レース結果で更新する。

    ⚠ `feature_for` は**更新前**の θ を返す。`update` は必ず
    その日の全レースの特徴量を作り終えてから呼ぶこと。
    """

    def __init__(self, theta=None, nrun=None, last_date=None):
        self.theta = defaultdict(float, theta or {})
        self.nrun = defaultdict(int, nrun or {})
        self.last_date = last_date
        self._pending = []          # その日の更新を溜める

    # ── 読み取り（学習・推論で共通） ──────────────────────────────────
    def get(self, name):
        """(θ, 出走回数)。未知の馬は (0.0, 0) ＝ 初出走と同じ扱い。"""
        return float(self.theta.get(name, 0.0)), int(self.nrun.get(name, 0))

    def field_features(self, names):
        """出走馬名リスト → 各馬の特徴量dictのリスト（順序は入力どおり）。"""
        th = [self.get(n)[0] for n in names]
        ns = [self.get(n)[1] for n in names]
        k = len(th)
        if k == 0:
            return []
        mean_th = sum(th) / k
        if k > 1:
            var = sum((t - mean_th) ** 2 for t in th) / k
            std_th = math.sqrt(var)
        else:
            std_th = 0.0
        order = sorted(range(k), key=lambda i: -th[i])
        rank = [0] * k
        for r, i in enumerate(order, 1):
            rank[i] = r
        out = []
        for i in range(k):
            d = th[i] - mean_th
            out.append({
                'f_pl_rating':         round(th[i], 5),
                'f_pl_rating_n':       ns[i],
                'rl_f_pl_rating':      round(d, 5),
                'rl_f_pl_rating_z':    round(d / std_th, 5) if std_th > 1e-6 else 0.0,
                'rl_f_pl_rating_rank': rank[i],
            })
        return out

    # ── 更新（学習時の再現・週次バッチで使う） ──────────────────────
    def queue_result(self, names_in_finish_order):
        """1レースの結果を、その日の更新キューに積む。

        names_in_finish_order: 1着から順の馬名（取消・中止は除く）。
        """
        if len(names_in_finish_order) >= MIN_FIELD:
            self._pending.append(list(names_in_finish_order))

    def advance_day(self, day):
        """溜めた結果で θ を更新し、前回からの日数ぶん減衰させる。

        day: 'YYYY-MM-DD'。その日の全レースの特徴量を作り終えてから呼ぶ。
        """
        for names in self._pending:
            self._apply(names)
        self._pending = []
        if self.last_date and DECAY_DAYS:
            dt = _days_between(self.last_date, day)
            if dt > 0:
                f = math.exp(-dt / DECAY_DAYS)
                for n in list(self.theta):
                    self.theta[n] *= f
        self.last_date = day

    def _apply(self, names):
        """Plackett-Luce の対数尤度の勾配で1レース分だけ動かす。

        dLL/dθ_i = Σ_{k<=rank_i}  [ 1{i が k 位} - exp(θ_i)/Σ_{j>=k} exp(θ_j) ]
        """
        k = len(names)
        th = [self.theta.get(n, 0.0) for n in names]
        m = max(th)
        e = [math.exp(t - m) for t in th]
        # 下から累積して Σ_{j>=k} exp(θ_j)
        tail = [0.0] * k
        acc = 0.0
        for i in range(k - 1, -1, -1):
            acc += e[i]
            tail[i] = acc
        grad = [0.0] * k
        for pos in range(k - 1):
            t = tail[pos]
            if t <= 0:
                continue
            grad[pos] += 1.0
            for j in range(pos, k):
                grad[j] -= e[j] / t
        for i, n in enumerate(names):
            self.nrun[n] = self.nrun.get(n, 0) + 1
            step = LR / (1.0 + EXP_DAMP * self.nrun[n])
            v = self.theta.get(n, 0.0) + step * grad[i]
            self.theta[n] = v - PRIOR_SHRINK * v

    # ── 永続化 ─────────────────────────────────────────────────────
    def to_dict(self):
        return {'theta': dict(self.theta), 'nrun': dict(self.nrun),
                'last_date': self.last_date}


def _days_between(a, b):
    from datetime import date
    def p(s):
        s = str(s)[:10].replace('/', '-')
        y, m, d = s.split('-')
        return date(int(y), int(m), int(d))
    try:
        return (p(b) - p(a)).days
    except Exception:
        return 0


def save_ratings(ratings, base_dir):
    path = os.path.join(base_dir, 'data', RATING_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(ratings.to_dict(), f)
    return path


def load_ratings(base_dir, today=None):
    """保存済み θ をロードする。

    today を渡すと、最終更新日から今日までのぶん減衰させる
    （学習時の日次減衰と同じ扱いにするため）。
    見つからなければ None を返す（呼び出し側は既定値で動かすこと）。
    """
    path = os.path.join(base_dir, 'data', RATING_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
    except Exception:
        return None
    r = PLRatings(d.get('theta'), d.get('nrun'), d.get('last_date'))
    if today and r.last_date and DECAY_DAYS:
        dt = _days_between(r.last_date, today)
        if dt > 0:
            f_ = math.exp(-dt / DECAY_DAYS)
            for n in list(r.theta):
                r.theta[n] *= f_
    return r


def build_from_history(hist_db_path, progress=None):
    """history.db を日付順に走って θ を構築する（週次バッチ用）。

    ⚠ その日の全レースを読んでから更新する（当日の結果を当日の特徴量に
    使わない）。学習時の再現と完全に同じ手順。
    """
    import sqlite3
    conn = sqlite3.connect(hist_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT race_id, date, horse_name, place
        FROM horse_history
        WHERE horse_name != '' AND date IS NOT NULL AND date != ''
        ORDER BY date, race_id, place
    """).fetchall()
    conn.close()

    r = PLRatings()
    cur_day = None
    by_race = defaultdict(list)
    n_races = 0
    for row in rows:
        d = str(row['date'])[:10]
        if cur_day is not None and d != cur_day:
            for rid, fin in by_race.items():
                fin.sort(key=lambda x: x[0])
                r.queue_result([nm for _, nm in fin])
                n_races += 1
            by_race = defaultdict(list)
            r.advance_day(cur_day)
            if progress and n_races % 2000 < 20:
                progress(n_races)
        cur_day = d
        p = row['place']
        if p and 0 < int(p) < 99:
            by_race[row['race_id']].append((int(p), row['horse_name']))
    if cur_day is not None:
        for rid, fin in by_race.items():
            fin.sort(key=lambda x: x[0])
            r.queue_result([nm for _, nm in fin])
            n_races += 1
        r.advance_day(cur_day)
    return r, n_races
