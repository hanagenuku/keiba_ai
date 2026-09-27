"""コース・馬場適性の順位（画面の参考列専用）。

ここで使う5列は 2026-09-18 の Gate 0 で「能力と独立した適性軸」として
**事前登録して通過した組み合わせそのもの**（`betting/gate0.py` の `T_COLS`）。
新しい定義を作ったのではなく、研究スクリプトから本番の表示側へ移植したもの。

🔴 実測で分かっていること（この列を過大評価しないために必ず読む）

- Gate 0（2026-09-18）: 能力 `f_pl_rating` との縮退は ρ̄ = +0.4931 で合格線0.80を
  下回ったが、**列ごとの差が大きい**。`f_course_fit_score` だけ ρ 0.7273 /
  条件感応 τ 0.9114 で5列中最悪。この列は
  `5*(stamina_demand*f_stamina_score + speed_demand*f_speed_score)` で、
  素の能力水準の掛け算なので「適性の顔をした能力」。
- Step 1（2026-09-22）: この T の3着内率への主効果は A1 +8.88pt / A2 +9.68pt だが、
  能力×市場で統制すると **+1.68 / +3.20pt** ＝ 効果の **67〜81% が市場評価の言い換え**。
  回収率は 9セル×2期間×2券種 = **36マスすべて100%未満**（最良は複勝81.9%で
  複勝プール平均80%と同水準）。ユーザー判断で研究は Step 1 で閉じている。

→ **買い目・軸・レース選択・集計には使わない。画面の参考列だけ。**
  「適性が高いから買う」と読める表示にしないこと（North Star #9: 的中率と回収率は別物）。

⚠ `f_course_fukusho` / `f_dist_fukusho` / `f_same_course_rate` / `f_same_turn_rate` /
  `f_uphill_match` は**入れない**。これらは「適性」ではなく**条件付き能力**
  （過去走を今回条件で絞り込んだ3着内率）で、2026-09-18 に3つの独立した裏付けで
  確認済み（`HORSE_ABILITY_DESIGN.md` §9）。
"""

# (列名, 符号の扱い)。'absneg' は「0から離れるほど不利」＝ -|x| を使う。
# betting/gate0.py の T_COLS と**同一**。片方だけ変えると研究と表示が食い違う。
APT_COLS = (
    ('f_style_course_fit',    'pos'),
    ('f_dist_vs_optimal',     'absneg'),
    ('f_speed_x_shortening',  'pos'),
    ('f_stamina_x_extension', 'pos'),
    ('f_course_fit_score',    'pos'),
)


def _signed(vals, mode):
    return [-abs(v) for v in vals] if mode == 'absneg' else list(vals)


def _zrace(vals):
    """レース内 z 化。std=0 なら None（その列はそのレースでは使わない）。

    母標準偏差（ddof=0）。`betting/gate0.py::_zrace` と同じ計算。
    """
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = var ** 0.5
    if sd < 1e-12:
        return None
    return [(v - m) / sd for v in vals]


def aptitude_scores(xfeats_list):
    """採用5列のレース内 z 値の等重み平均を返す。

    Returns:
        (scores, used_cols) — scores は各馬の float のリスト。
        使える列が1つも無ければ ``(None, [])``。
    """
    n = len(xfeats_list)
    if n < 2:
        return None, []
    parts, used = [], []
    for col, mode in APT_COLS:
        raw = [xf.get(col) for xf in xfeats_list]
        if any(v is None for v in raw):
            continue
        try:
            fv = [float(v) for v in raw]
        except (TypeError, ValueError):
            continue
        if not all(v == v and abs(v) != float('inf') for v in fv):  # NaN/inf を弾く
            continue
        z = _zrace(_signed(fv, mode))
        if z is not None:
            parts.append(z)
            used.append(col)
    if not parts:
        return None, []
    return [sum(p[i] for p in parts) / len(parts) for i in range(n)], used


def calc_fit_ranks(nums, xfeats_list):
    """馬番 → コース・馬場適性順位（1が最も向く）。計算できなければ ``{}``。

    同値は同順位（competition ranking）。研究側（Gate 0）は対照群の都合で
    **平均順位**を使っているが、こちらは画面表示なので直感的な同順位にしてある。
    順位の値そのものを研究と突き合わせないこと。
    """
    scores, used = aptitude_scores(xfeats_list)
    if scores is None or len(nums) != len(scores):
        return {}
    order = sorted(range(len(nums)), key=lambda i: scores[i], reverse=True)
    ranks, prev_score, prev_rank = {}, None, 0
    for pos, i in enumerate(order, 1):
        if prev_score is not None and abs(scores[i] - prev_score) < 1e-12:
            ranks[nums[i]] = prev_rank
        else:
            ranks[nums[i]] = pos
            prev_rank, prev_score = pos, scores[i]
    return ranks
