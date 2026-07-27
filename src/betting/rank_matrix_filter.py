"""人気×RL乖離マトリクスの実績に基づく買い目フィルタ（2026-07-27導入）。

`scripts/generate_stats.py` が週次蓄積する `data/divergence_weekly.json` の
`rank_matrix`（人気帯×RL帯ごとの実測単勝回収率）を読み、セル実績に応じて
買い目候補を調整する:

- suppress: 十分なN（>= MIN_N）で実測回収率が SUPPRESS_ROI 未満のセル。
  単勝候補・複勝候補から除外する（例: 市場2-3人気×RL6-9 = 実測37.7%）
- boost: 十分なNで実測回収率が BOOST_ROI 以上のセル。
  単勝候補の対象をRL上位3頭からRL上位5頭のboostセル該当馬まで広げる
  （例: 市場2-3人気×RL4-5 = 実測144.9%。RL4-5の馬は従来の単勝選択
  ロジック(RL上位3頭のみ)では構造的に候補にすら入らなかった）

これは**モデル変更ではなく意思決定層のフィルタ**。モデルの予測値
（win_prob/rl_rank等）には一切触れない。「モデル変更は一旦停止」の方針とも
矛盾しない。

⚠ 既知の限界:
- rank_matrixは全期間累積のため、旧・市場コピー型モデル期（〜2026-07-14）の
  データが混ざっている。残差モデル期のデータが蓄積するほど純化される
- 回収率は単勝ベースの実測のみ。複勝には suppress のみ適用し boost は
  適用しない（複勝回収率は未測定のため）
- 環境変数 RANK_MATRIX_FILTER=0 で全体を無効化できる（緊急時のkill switch）
"""

import json
import os

# セル実績に基づく判定閾値。
# MIN_N=50: これ未満のセルは数回の的中で回収率が大きく振れるため判定しない
# SUPPRESS_ROI=50: 実測回収率50%未満 = 賭け金の半分以上を失っている実績
# BOOST_ROI=120: 実測回収率120%以上 = 控除率(約20%)を大きく超える実績
MIN_N = 50
SUPPRESS_ROI = 50.0
BOOST_ROI = 120.0

_MATRIX_CACHE = {}


def _rank_band(rank):
    """generate_stats.py の帯定義と完全に一致させること。"""
    if rank == 1:
        return '1'
    if rank <= 3:
        return '2-3'
    if rank <= 5:
        return '4-5'
    if rank <= 9:
        return '6-9'
    return '10+'


def load_rank_matrix(base_dir):
    """divergence_weekly.json の最新エントリから rank_matrix を読む（キャッシュ付き）。

    ファイルが無い・形式が違う・rank_matrix未生成（旧フォーマット）の場合は
    空dictを返し、フィルタは全馬 neutral（従来挙動と同一）になる。
    """
    if base_dir in _MATRIX_CACHE:
        return _MATRIX_CACHE[base_dir]

    cells = {}
    try:
        path = os.path.join(base_dir, 'data', 'divergence_weekly.json')
        with open(path, encoding='utf-8') as f:
            entries = json.load(f)
        if entries:
            latest = entries[-1].get('divergence') or {}
            for c in latest.get('rank_matrix', []):
                cells[(c['pop_band'], c['rl_band'])] = c
    except Exception:
        cells = {}

    _MATRIX_CACHE[base_dir] = cells
    return cells


def clear_cache():
    """テスト・再ロード用。"""
    _MATRIX_CACHE.clear()


def _filter_enabled():
    return os.environ.get('RANK_MATRIX_FILTER', '1') != '0'


def classify_horse(popularity, rl_rank, base_dir):
    """1頭のマトリクスセル実績から 'suppress' / 'boost' / None を返す。

    popularity/rl_rank が不明（None/99以上）の馬、セルのNが不足している馬、
    実績が中間帯の馬は None（従来挙動のまま）。
    """
    if not _filter_enabled():
        return None
    if popularity is None or rl_rank is None:
        return None
    if popularity >= 99 or rl_rank >= 99:
        return None

    cells = load_rank_matrix(base_dir)
    cell = cells.get((_rank_band(popularity), _rank_band(rl_rank)))
    if not cell or cell.get('count', 0) < MIN_N:
        return None

    roi = cell.get('tansho_roi')
    if roi is None:
        return None
    if roi < SUPPRESS_ROI:
        return 'suppress'
    if roi >= BOOST_ROI:
        return 'boost'
    return None


def is_matrix_active(base_dir):
    """マトリクスが実際にロードできているか（＝フィルタが機能する状態か）。

    `build_flag_map` の結果が空でも「マトリクスは有るがboost/suppress該当馬が
    そのレースに居なかった」のか「マトリクス自体が無い（旧環境・初回実行）」のか
    を区別する必要があるため、別途この判定を提供する。
    単勝の「boost馬がいなければ買わない」判断はマトリクスが有効な時のみ
    適用し、無効時は従来挙動を保つ。
    """
    if base_dir is None or not _filter_enabled():
        return False
    return bool(load_rank_matrix(base_dir))


def build_flag_map(horses, base_dir):
    """calc_all() 出力の馬リストから {horse_num: 'suppress'|'boost'} を作る。

    neutral の馬はマップに含めない（呼び出し側は .get(num) が None なら従来挙動）。
    base_dir が None の場合は空dict（フィルタ無効と同じ）。
    """
    if base_dir is None:
        return {}
    flags = {}
    for h in horses:
        num = h.get('horse_num') or h.get('num')
        if num is None:
            continue
        flag = classify_horse(h.get('popularity'), h.get('rl_rank'), base_dir)
        if flag:
            flags[num] = flag
    return flags
