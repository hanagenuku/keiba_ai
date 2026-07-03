"""
3モデル比較フレームワーク（複勝 vs pairwise vs ndcg）

horse_features.csv を使って Val 期間の各レースを評価し、
ROI・的中率・Calibration（ECE）を券種別に比較する。

使い方（Colab）:
    from src.tools.compare_models import compare_all_models, print_comparison
    val_periods = [
        ('2026-06-01', '2026-06-15'),
        ('2026-03-01', '2026-03-31'),
        ('2025-09-01', '2025-09-30'),
    ]
    metrics = compare_all_models(BASE_DIR, val_periods)
    print_comparison(metrics)
"""

import os
import json
import pickle
import numpy as np


_EXCLUDE_COLS = {'race_id', 'date', 'horse_name', 'horse_num', 'place',
                 'is_fukusho', 'date_obj'}
_BET_TYPES    = ['win', 'place', 'quinella', 'trio']


# ── モデルロード ─────────────────────────────────────────────────────────

def _load_models(base_dir):
    """
    3モデルとその温度・特徴量リストを返す。
    見つからないモデルはスキップ。

    Returns
    -------
    dict: {
        model_name: {
            'model': object,
            'is_logistic': bool,
            'feat_cols': list,
            'T': float,
        }
    }
    """
    import xgboost as xgb

    data_dir = os.path.join(base_dir, 'data')
    temp_path = os.path.join(data_dir, 'rating_temperature.json')

    temperatures = {}
    if os.path.exists(temp_path):
        with open(temp_path) as f:
            temperatures = json.load(f).get('calibration', {})
    else:
        print('⚠ rating_temperature.json なし → T=1.0 で全モデル実行')

    def _load_feat_cols(path, df_cols):
        if path and os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
            return info.get('feature_cols', [])
        return [c for c in df_cols if c not in _EXCLUDE_COLS]

    defs = {
        'A_fukusho': {
            'path':      os.path.join(data_dir, 'xgb_fukusho_model.pkl'),
            'is_logistic': True,
            'cols_path': os.path.join(data_dir, 'xgb_feature_cols.json'),
            'temp_key':  'fukusho',
        },
        'B1_pairwise': {
            'path':      os.path.join(data_dir, 'xgb_ranking_pairwise.pkl'),
            'is_logistic': False,
            'cols_path': os.path.join(data_dir, 'xgb_ranking_feature_cols.json'),
            'temp_key':  'ranking_pairwise',
        },
        'B2_ndcg': {
            'path':      os.path.join(data_dir, 'xgb_ranking_ndcg.pkl'),
            'is_logistic': False,
            'cols_path': os.path.join(data_dir, 'xgb_ranking_feature_cols.json'),
            'temp_key':  'ranking_ndcg',
        },
    }

    loaded = {}
    for name, d in defs.items():
        if not os.path.exists(d['path']):
            print(f'  ⚠ {name}: ファイルなし → スキップ')
            continue
        with open(d['path'], 'rb') as f:
            model = pickle.load(f)
        T = temperatures.get(d['temp_key'], {}).get('T', 1.0)
        loaded[name] = {
            'model':       model,
            'is_logistic': d['is_logistic'],
            'cols_path':   d['cols_path'],
            'T':           T,
        }
        print(f'  ✅ {name}: T={T}')

    return loaded


def _predict_ratings(model_info, feat_df, feat_cols):
    """
    モデル種別に応じた能力値を返す（Gumbel入力スケール、温度補正前）。

    - XGBClassifier (is_logistic=True): prob → logit → 能力値
    - xgb.Booster   (is_logistic=False): predict → 能力値
    """
    import xgboost as xgb
    import pandas as pd

    model       = model_info['model']
    is_logistic = model_info['is_logistic']
    T           = model_info['T']

    # モデルの feature_names を優先（JSON とモデル pkl の不一致を吸収）
    try:
        booster = model.get_booster() if is_logistic else model
        if booster.feature_names:
            feat_cols = list(booster.feature_names)
    except Exception:
        pass

    # CSV にない特徴量は 5.0 で補完
    aligned = pd.DataFrame(index=feat_df.index)
    for c in feat_cols:
        aligned[c] = feat_df[c] if c in feat_df.columns else 5.0
    X = aligned.fillna(5.0)

    if is_logistic:
        prob = model.predict_proba(X)[:, 1]
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        ratings = np.log(prob / (1 - prob))
    else:
        dmat = xgb.DMatrix(X.values, feature_names=feat_cols)
        ratings = model.predict(dmat)

    return ratings / T   # 温度補正済み能力値


# ── メイン比較 ────────────────────────────────────────────────────────────

def _init_metrics():
    return {bt: {'hits': 0, 'n': 0, 'roi_sum': 0.0,
                 'probs': [], 'hit_flags': []} for bt in _BET_TYPES}


def _update_metrics(metrics_bt, hit, est_payout, pred_prob):
    """
    1レース分の結果を指標に追記する。

    hit         : bool（その買い目が当たったか）
    est_payout  : 推定配当 float | None
    pred_prob   : モデルの予測確率（ECE計算用）
    """
    metrics_bt['n'] += 1
    if hit:
        metrics_bt['hits'] += 1
        if est_payout:
            metrics_bt['roi_sum'] += est_payout - 1.0  # 1点買いのネット収益
        else:
            metrics_bt['roi_sum'] += 0.0
    else:
        metrics_bt['roi_sum'] -= 1.0   # 外れ → -100%

    metrics_bt['probs'].append(float(pred_prob))
    metrics_bt['hit_flags'].append(1 if hit else 0)


def _finalize_metrics(metrics_bt):
    n = metrics_bt['n']
    if n == 0:
        return {'hit_rate': 0.0, 'roi': 0.0, 'ece': 0.0, 'n': 0}

    from src.betting.rating_calibration import calc_ece
    probs = np.array(metrics_bt['probs'])
    hits  = np.array(metrics_bt['hit_flags'])
    hit_rate = metrics_bt['hits'] / n
    roi      = metrics_bt['roi_sum'] / n        # 1点あたりの平均収益
    ece      = calc_ece(probs, hits) if len(probs) > 0 else 0.0
    return {'hit_rate': hit_rate, 'roi': roi, 'ece': ece, 'n': n}


def compare_all_models(base_dir, val_periods, n_sims=10000):
    """
    複数 Val 期間で 3 モデルを比較する。

    Parameters
    ----------
    base_dir    : プロジェクトルート
    val_periods : [(start, end), ...]  複数月を指定して過学習リスクを下げる
    n_sims      : Gumbel シミュレーション回数（デフォルト 10k: 速度重視）

    Returns
    -------
    dict: {model_name: {bet_type: {hit_rate, roi, ece, n}}}
    """
    import pandas as pd
    from src.betting.race_simulator import simulate_race, calc_ticket_probabilities
    from src.betting.payout_estimator import estimate_payouts_from_win_odds, get_top_payout

    data_dir = os.path.join(base_dir, 'data')
    csv_path = os.path.join(data_dir, 'horse_features.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'horse_features.csv が見つかりません: {csv_path}')

    print('モデルロード中...')
    models = _load_models(base_dir)
    if not models:
        raise RuntimeError('比較するモデルがありません。先にモデルを学習してください。')

    # 特徴量列は各モデルの cols_path から取得
    _feat_cols_cache = {}
    df_all = pd.read_csv(csv_path)
    for name, info in models.items():
        cp = info.get('cols_path', '')
        if cp and os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            _feat_cols_cache[name] = d.get('feature_cols', [])
        else:
            _feat_cols_cache[name] = [c for c in df_all.columns
                                       if c not in _EXCLUDE_COLS
                                       and df_all[c].dtype in
                                       ('float64', 'int64', 'float32', 'int32')]

    df_all['date_obj'] = pd.to_datetime(
        df_all['date'].astype(str).str.replace('-', '', regex=False).str[:8],
        format='%Y%m%d', errors='coerce'
    )
    df_all = df_all.dropna(subset=['date_obj'])

    # 指標バッファ初期化
    all_metrics_raw = {name: _init_metrics() for name in models}
    total_races = 0

    for (start, end) in val_periods:
        val_df = df_all[(df_all['date_obj'] >= pd.Timestamp(start)) &
                        (df_all['date_obj'] <= pd.Timestamp(end))].copy()
        if len(val_df) == 0:
            print(f'  ⚠ {start}〜{end}: データなし')
            continue

        n_races = val_df['race_id'].nunique()
        print(f'\n📅 {start}〜{end}: {n_races} レース')
        total_races += n_races

        val_df = val_df.sort_values(['race_id', 'horse_num'])

        for race_id, grp in val_df.groupby('race_id', sort=False):
            valid = grp[grp['place'].fillna(99) < 99].copy()
            if len(valid) < 5:
                continue

            # 実際の結果
            winner_rows = valid[valid['place'] == 1]
            second_rows = valid[valid['place'] == 2]
            third_rows  = valid[valid['place'] == 3]
            if winner_rows.empty:
                continue

            actual_1st = int(winner_rows.iloc[0]['horse_num'])
            actual_2nd = int(second_rows.iloc[0]['horse_num']) if not second_rows.empty else -1
            actual_3rd = int(third_rows.iloc[0]['horse_num'])  if not third_rows.empty  else -1
            actual_top3 = {actual_1st, actual_2nd, actual_3rd} - {-1}

            nums = valid['horse_num'].astype(int).tolist()

            # 払戻推定（単勝オッズ → 市場確率ベース）
            win_odds_map = {}
            for _, row in valid.iterrows():
                wo = row.get('f_win_odds') or row.get('win_odds')
                if wo and float(wo) > 0:
                    win_odds_map[int(row['horse_num'])] = float(wo)
            # f_win_odds がなければ popularity から粗い推定
            if not win_odds_map:
                for _, row in valid.iterrows():
                    pop = row.get('f_popularity') or row.get('popularity') or 10
                    win_odds_map[int(row['horse_num'])] = max(1.1, float(pop) * 1.5)

            try:
                payouts = estimate_payouts_from_win_odds(win_odds_map, n_sims=n_sims)
            except Exception:
                payouts = {bt: {} for bt in _BET_TYPES}

            # 各モデルの予測と照合
            for name, model_info in models.items():
                feat_cols = _feat_cols_cache[name]
                try:
                    ratings = _predict_ratings(model_info, valid, feat_cols)
                except Exception as e:
                    if not getattr(_predict_ratings, f'_warned_{name}', False):
                        print(f'  ⚠ {name}: predict失敗 → {type(e).__name__}: {e}')
                        setattr(_predict_ratings, f'_warned_{name}', True)
                    continue

                orders = simulate_race(ratings, n_sims=n_sims)
                probs  = calc_ticket_probabilities(orders, nums)

                # 単勝
                top_win, est_win = get_top_payout(payouts, 'win', probs)
                hit_win = (top_win == actual_1st)
                p_win   = probs['win'].get(top_win, 0.0)
                _update_metrics(all_metrics_raw[name]['win'], hit_win, est_win, p_win)

                # 複勝
                top_place, est_place = get_top_payout(payouts, 'place', probs)
                hit_place = (top_place in actual_top3)
                p_place   = probs['place'].get(top_place, 0.0)
                _update_metrics(all_metrics_raw[name]['place'], hit_place, est_place, p_place)

                # 馬連
                top_quin, est_quin = get_top_payout(payouts, 'quinella', probs)
                if top_quin is not None:
                    hit_quin = (set(top_quin) == {actual_1st, actual_2nd})
                    p_quin   = probs['quinella'].get(top_quin, 0.0)
                    _update_metrics(all_metrics_raw[name]['quinella'], hit_quin, est_quin, p_quin)

                # 三連複
                top_trio, est_trio = get_top_payout(payouts, 'trio', probs)
                if top_trio is not None:
                    hit_trio = (set(top_trio) == actual_top3)
                    p_trio   = probs['trio'].get(top_trio, 0.0)
                    _update_metrics(all_metrics_raw[name]['trio'], hit_trio, est_trio, p_trio)

    # 集計
    final = {}
    for name in models:
        final[name] = {bt: _finalize_metrics(all_metrics_raw[name][bt])
                       for bt in _BET_TYPES}

    print(f'\n✅ 比較完了: {total_races} レース')
    return final


# ── 比較レポート ──────────────────────────────────────────────────────────

def print_comparison(metrics):
    """
    ROI・的中率・ECE を券種別に縦横表示し、
    券種ごとの最良モデルを判定して表示する。
    """
    if not metrics:
        print('比較データがありません。')
        return

    model_names = list(metrics.keys())
    bet_labels  = {'win': '単勝', 'place': '複勝', 'quinella': '馬連', 'trio': '三連複'}

    print('\n' + '=' * 72)
    print('  3 モデル比較レポート')
    print('=' * 72)

    best_by_bet = {}

    for bt, label in bet_labels.items():
        print(f'\n【{label} ({bt})】')
        print(f'  {"モデル":<18} {"N":>5} {"的中率":>8} {"推定ROI":>10} {"ECE":>8}')
        print('  ' + '-' * 50)

        best_name = None
        best_roi  = -999.0

        for name in model_names:
            m = metrics[name].get(bt, {})
            n        = m.get('n', 0)
            hit_rate = m.get('hit_rate', 0.0)
            roi      = m.get('roi', 0.0)
            ece      = m.get('ece', 0.0)
            tag      = ''
            if roi > best_roi:
                best_roi  = roi
                best_name = name

            print(f'  {name:<18} {n:>5} {hit_rate:>7.1%} {roi:>+9.1%} {ece:>7.3f}{tag}')

        best_by_bet[bt] = best_name
        print(f'  → 推定ROI最良: {best_name}')

    print('\n' + '=' * 72)
    print('  【総合判定】')
    print('=' * 72)
    for bt, label in bet_labels.items():
        winner = best_by_bet.get(bt, '?')
        print(f'  {label:<6}: {winner}')

    # 全券種でのスコア集計
    model_wins = {n: 0 for n in model_names}
    for winner in best_by_bet.values():
        if winner in model_wins:
            model_wins[winner] += 1

    print('\n  券種別勝利数:')
    for name, w in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
        bar = '★' * w
        print(f'    {name:<18}: {w}券種  {bar}')

    print()
    max_wins = max(model_wins.values()) if model_wins else 0
    top_models = [n for n, w in model_wins.items() if w == max_wins]
    if len(top_models) == 1:
        print(f'  ✅ 推奨: {top_models[0]}  （{max_wins}/{len(bet_labels)} 券種で最良）')
    else:
        print(f'  △ 差なし: {", ".join(top_models)} が同数（データ追加後に再評価推奨）')

    print('=' * 72)
    return best_by_bet


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else '/content/drive/MyDrive/keiba_ai'
    periods = [('2026-06-01', '2026-06-15'), ('2026-03-01', '2026-03-31')]
    m = compare_all_models(base, periods)
    print_comparison(m)
