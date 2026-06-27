"""
市場補正レイヤー

XGBoostの予測（cal_prob）を、市場人気との整合性で補正する。
「AIが高評価だが市場が極端に低評価」の馬を抑制する。

設計思想:
  AIと市場が両方評価する馬 → 信頼（補正なし）
  AIだけが評価する不人気馬 → 抑制（補正で下げる）

6/27データ: AIが市場と異なる本命を出した25Rで市場が6倍正確だった。
この「独自判断の暴走」を抑えるのが目的。
"""
import os

MARKET_CORRECTION_ENABLED = os.environ.get('MARKET_CORRECTION', 'true').lower() == 'true'

# RL順位帯 × 人気帯 の補正係数
# 6/27データに基づく暫定値。データ蓄積で correction_table.json による自動更新に移行予定
CORRECTION_FACTORS = {
    # (RL帯, 人気帯): 補正係数
    ('top', 'popular'):   1.0,   # RL上位 × 1-3番人気 → 信頼（そのまま）
    ('top', 'mid'):       0.85,  # RL上位 × 4-6番人気 → やや抑制
    ('top', 'low'):       0.55,  # RL上位 × 7-9番人気 → 抑制
    ('top', 'unpopular'): 0.30,  # RL上位 × 10番人気以上 → 大幅抑制
    ('mid', 'popular'):   1.15,  # RL中位 × 1-3番人気 → やや強調（市場が支持）
    ('mid', 'mid'):       1.0,
    ('mid', 'low'):       0.9,
    ('mid', 'unpopular'): 0.7,
    ('low', 'popular'):   1.2,   # RL下位 × 1-3番人気 → 強調（AIの見落とし補正）
    ('low', 'mid'):       1.0,
    ('low', 'low'):       1.0,
    ('low', 'unpopular'): 1.0,
}


def classify_rl_band(rl_rank):
    if rl_rank <= 3:
        return 'top'
    elif rl_rank <= 6:
        return 'mid'
    else:
        return 'low'


def classify_pop_band(popularity):
    if popularity <= 3:
        return 'popular'
    elif popularity <= 6:
        return 'mid'
    elif popularity <= 9:
        return 'low'
    else:
        return 'unpopular'


def apply_market_correction(horses):
    """
    全馬の total と cal_prob に市場補正を適用する。

    Parameters
    ----------
    horses : list of dict
        各馬に total, cal_prob, rl_rank（暫定）, popularity が必要。
        rl_rank は cal_prob 降順の暫定順位を事前に付与しておくこと。

    Returns
    -------
    horses : list of dict
        cal_prob_raw（補正前）, rl_rank_raw（補正前順位）,
        correction_factor, correction_applied を追加した辞書リスト。
        total と cal_prob は補正後の値に更新される。
    """
    if not MARKET_CORRECTION_ENABLED:
        for h in horses:
            h['cal_prob_raw'] = h.get('cal_prob', 0.0)
            h['correction_applied'] = False
            h['correction_factor'] = 1.0
            h.setdefault('rl_rank_raw', h.get('rl_rank', 99))
        return horses

    # 補正前のRL順位を保存（表示用）
    for h in horses:
        h['rl_rank_raw'] = h.get('rl_rank', 99)

    orig_total_sum = sum(h.get('total', 0) for h in horses)

    # 各馬に補正を適用
    for h in horses:
        raw_cal = h.get('cal_prob', 0.0)
        h['cal_prob_raw'] = raw_cal

        rl_band  = classify_rl_band(h.get('rl_rank', 99))
        pop_band = classify_pop_band(h.get('popularity', 99))
        factor   = CORRECTION_FACTORS.get((rl_band, pop_band), 1.0)

        h['cal_prob']          = raw_cal * factor
        h['total']             = h.get('total', 0) * factor
        h['correction_factor'] = factor
        h['correction_applied'] = (factor != 1.0)

    # total を正規化（合計を補正前の値に戻す。softmax のスケールを維持）
    new_total_sum = sum(h.get('total', 0) for h in horses)
    if new_total_sum > 0.01:
        for h in horses:
            h['total'] = round(h['total'] * orig_total_sum / new_total_sum, 2)

    # cal_prob を正規化（合計を 3.0 に。Harville の入力スケールを保持）
    total_cal = sum(h['cal_prob'] for h in horses)
    if total_cal > 0:
        target = min(3.0, len(horses))
        for h in horses:
            h['cal_prob'] = h['cal_prob'] * target / total_cal

    return horses
