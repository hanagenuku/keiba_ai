"""
レース後エラータグ自動分類 + 週次補正係数

毎週のレース結果から「AIがなぜ外したか」を12種のタグで自動分類し、
venue × surface × distance帯 × 馬場状態 の条件別に蓄積する。

2段階で活用:
  1. 翌週から即反映: 蓄積データから条件別補正係数を自動計算 → 予測スコアを補正
  2. XGB再学習時: 蓄積タグを特徴量化 → モデル自体の判断力向上

蓄積先: data/error_tags_weekly.json
"""

import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

MIN_SAMPLES_FOR_CORRECTION = 20

# 補正係数の上下限
CORRECTION_MIN = 0.70
CORRECTION_MAX = 1.40

TAG_DEFINITIONS = [
    'pace_miss',         # ペース予測と実際が不一致
    'escape_win',        # 逃げ馬が勝利（AIが逃げ馬を過小評価）
    'position_bias',     # 内/外枠が偏って好走
    'style_miss',        # AI低評価の脚質が好走
    'class_miss',        # 昇級/降級馬がAI予想外に好走
    'form_miss',         # 休み明け/連戦馬がAI予想外に好走
    'dist_short_win',    # 距離短縮馬が好走
    'dist_ext_win',      # 距離延長馬が好走
    'heavy_upset',       # 重/不良で人気薄が好走
    'mare_upset',        # 牝馬がAI予想外に好走
    'young_upset',       # 3歳馬が古馬戦で好走
    'jockey_switch_win', # 乗り替わりで好走
]


def _dist_band(d):
    """距離を帯に分類"""
    if not d:
        return 'unknown'
    if d <= 1400:
        return 'short'
    if d <= 1800:
        return 'mile'
    if d <= 2200:
        return 'middle'
    return 'long'


def _condition_key(venue, surface, distance, track_condition):
    """条件キーを生成（蓄積・参照用）"""
    band = _dist_band(distance)
    cond = (track_condition or 'unknown').replace('　', '')
    return f"{venue}_{surface}_{band}_{cond}"


def classify_race_tags(race_result, predictions, history_db_path=None):
    """
    1レースのエラータグを分類する。

    Parameters
    ----------
    race_result : dict
        レース結果。キー: race_id, racecourse, surface, distance,
        track_condition, first_3f, horses (list of dicts with
        horse_num, place, horse_name, running_style, popularity,
        sex, age, jockey, history)
    predictions : dict  {horse_num: {'rl_rank': int, 'win_prob': float, ...}}
        AIの予測（race_predictions から）
    history_db_path : str or None
        history.db のパス（前走情報の参照用。Noneなら前走依存タグをスキップ）

    Returns
    -------
    dict: {
        'race_id': str,
        'venue': str,
        'surface': str,
        'distance': int,
        'track_condition': str,
        'condition_key': str,
        'tags': list[str],
        'details': dict,
    }
    """
    tags = []
    details = {}

    horses = race_result.get('horses', [])
    if not horses:
        return None

    venue = race_result.get('racecourse', '')
    surface = race_result.get('surface', '')
    distance = race_result.get('distance', 0)
    track_cond = race_result.get('track_condition', '')

    top3_horses = [h for h in horses if h.get('place') and h['place'] <= 3]
    winner = next((h for h in horses if h.get('place') == 1), None)

    if not top3_horses or not winner:
        return None

    # AI上位3頭
    ai_top3_nums = set()
    for num, pred in predictions.items():
        if pred.get('rl_rank') and pred['rl_rank'] <= 3:
            ai_top3_nums.add(num)

    # ── escape_win: 逃げ馬が勝って、AIがその馬を低評価していた ──
    winner_style = winner.get('running_style', '')
    winner_num = winner.get('horse_num')
    if winner_style == '逃げ' and winner_num not in ai_top3_nums:
        tags.append('escape_win')
        details['escape_win'] = {'winner_num': winner_num}

    # ── position_bias: 好走馬が内枠/外枠に偏っている ──
    n_horses = len(horses)
    if n_horses >= 8:
        top3_positions = [h.get('horse_num', 99) for h in top3_horses]
        inner_count = sum(1 for p in top3_positions if p <= n_horses * 0.33)
        outer_count = sum(1 for p in top3_positions if p >= n_horses * 0.67)
        if inner_count >= 2:
            tags.append('position_bias')
            details['position_bias'] = {'bias': 'inner', 'inner_count': inner_count}
        elif outer_count >= 2:
            tags.append('position_bias')
            details['position_bias'] = {'bias': 'outer', 'outer_count': outer_count}

    # ── style_miss: AI低評価(RL>5)の脚質が好走 ──
    for h in top3_horses:
        hnum = h.get('horse_num')
        pred = predictions.get(hnum, {})
        rl = pred.get('rl_rank', 99)
        if rl > 5:
            style = h.get('running_style', '')
            if style:
                tags.append('style_miss')
                details['style_miss'] = {
                    'horse_num': hnum, 'style': style, 'rl_rank': rl,
                }
                break

    # ── form_miss: 休み明け(間隔3ヶ月以上)の馬がAI低評価で好走 ──
    for h in top3_horses:
        hnum = h.get('horse_num')
        pred = predictions.get(hnum, {})
        rl = pred.get('rl_rank', 99)
        if rl > 5:
            hist = h.get('history', [])
            if hist:
                last_date_str = hist[0].get('date', '')
                race_date_str = race_result.get('date', '')
                if last_date_str and race_date_str:
                    try:
                        ld = _parse_date(last_date_str)
                        rd = _parse_date(race_date_str)
                        if ld and rd and (rd - ld).days >= 90:
                            tags.append('form_miss')
                            details['form_miss'] = {
                                'horse_num': hnum,
                                'gap_days': (rd - ld).days,
                                'type': 'layoff',
                            }
                            break
                    except (ValueError, TypeError):
                        pass

    # ── dist_short_win / dist_ext_win: 距離変更馬が好走 ──
    for h in top3_horses:
        hist = h.get('history', [])
        if hist and distance:
            prev_dist = hist[0].get('distance')
            if prev_dist and prev_dist > 0:
                diff = distance - prev_dist
                hnum = h.get('horse_num')
                pred = predictions.get(hnum, {})
                rl = pred.get('rl_rank', 99)
                if diff <= -200 and rl > 3:
                    if 'dist_short_win' not in tags:
                        tags.append('dist_short_win')
                        details['dist_short_win'] = {
                            'horse_num': hnum, 'prev': prev_dist,
                            'current': distance, 'diff': diff,
                        }
                elif diff >= 200 and rl > 3:
                    if 'dist_ext_win' not in tags:
                        tags.append('dist_ext_win')
                        details['dist_ext_win'] = {
                            'horse_num': hnum, 'prev': prev_dist,
                            'current': distance, 'diff': diff,
                        }

    # ── heavy_upset: 重/不良で人気薄(10番人気以上)が好走 ──
    if track_cond in ('重', '不良'):
        for h in top3_horses:
            pop = h.get('popularity', 0)
            if pop and pop >= 10:
                tags.append('heavy_upset')
                details['heavy_upset'] = {
                    'horse_num': h.get('horse_num'),
                    'popularity': pop,
                }
                break

    # ── mare_upset: 牝馬がAI低評価で好走 ──
    for h in top3_horses:
        sex = h.get('sex', '')
        hnum = h.get('horse_num')
        pred = predictions.get(hnum, {})
        rl = pred.get('rl_rank', 99)
        if sex == '牝' and rl > 5:
            tags.append('mare_upset')
            details['mare_upset'] = {'horse_num': hnum, 'rl_rank': rl}
            break

    # ── young_upset: 3歳馬がAI低評価で好走（古馬混合戦） ──
    ages = [h.get('age', 0) for h in horses if h.get('age')]
    has_old = any(a >= 4 for a in ages)
    if has_old:
        for h in top3_horses:
            age = h.get('age', 0)
            hnum = h.get('horse_num')
            pred = predictions.get(hnum, {})
            rl = pred.get('rl_rank', 99)
            if age == 3 and rl > 5:
                tags.append('young_upset')
                details['young_upset'] = {'horse_num': hnum, 'rl_rank': rl}
                break

    # ── jockey_switch_win: 乗り替わりで好走（前走と騎手が違う） ──
    for h in top3_horses:
        jockey = h.get('jockey', '')
        hist = h.get('history', [])
        hnum = h.get('horse_num')
        pred = predictions.get(hnum, {})
        rl = pred.get('rl_rank', 99)
        if hist and jockey and rl > 3:
            prev_jockey = hist[0].get('jockey', '')
            if prev_jockey and prev_jockey != jockey:
                tags.append('jockey_switch_win')
                details['jockey_switch_win'] = {
                    'horse_num': hnum,
                    'new_jockey': jockey,
                    'prev_jockey': prev_jockey,
                }
                break

    # ── class_miss: 昇級馬がAI低評価で好走 ──
    race_class = race_result.get('race_class', '')
    for h in top3_horses:
        hist = h.get('history', [])
        hnum = h.get('horse_num')
        pred = predictions.get(hnum, {})
        rl = pred.get('rl_rank', 99)
        if hist and rl > 5:
            prev_class = hist[0].get('race_class', '')
            if prev_class and race_class:
                curr_level = _class_level(race_class)
                prev_level = _class_level(prev_class)
                if curr_level > prev_level:
                    tags.append('class_miss')
                    details['class_miss'] = {
                        'horse_num': hnum,
                        'prev_class': prev_class,
                        'current_class': race_class,
                        'type': 'promotion',
                    }
                    break

    # ── pace_miss: 予想ペースと実際が乖離 ──
    predicted_pace = race_result.get('predicted_pace')
    first_3f = race_result.get('first_3f')
    if predicted_pace and first_3f and distance:
        norm_3f = first_3f / (distance / 1000.0) if distance > 0 else first_3f
        if surface == 'ダート':
            actual_pace = 'slow' if norm_3f > 37.5 else ('high' if norm_3f < 35.5 else 'mid')
        else:
            actual_pace = 'slow' if norm_3f > 36.5 else ('high' if norm_3f < 34.5 else 'mid')
        if predicted_pace != actual_pace:
            tags.append('pace_miss')
            details['pace_miss'] = {
                'predicted': predicted_pace,
                'actual': actual_pace,
                'first_3f': first_3f,
            }

    if not tags:
        return None

    cond_key = _condition_key(venue, surface, distance, track_cond)
    return {
        'race_id': race_result.get('race_id', ''),
        'date': race_result.get('date', ''),
        'venue': venue,
        'surface': surface,
        'distance': distance,
        'track_condition': track_cond,
        'condition_key': cond_key,
        'tags': list(set(tags)),
        'details': details,
    }


def _parse_date(s):
    """日付文字列をdatetimeに変換"""
    s = str(s).replace('-', '').strip()[:8]
    if len(s) == 8:
        try:
            return datetime.strptime(s, '%Y%m%d')
        except ValueError:
            return None
    return None


def _class_level(class_str):
    """クラス文字列をレベル数値に変換"""
    if not class_str:
        return 0
    if any(x in class_str for x in ['新馬']):
        return 1
    if any(x in class_str for x in ['未勝利']):
        return 2
    if any(x in class_str for x in ['1勝', '500万']):
        return 3
    if any(x in class_str for x in ['2勝', '1000万']):
        return 4
    if any(x in class_str for x in ['3勝', '1600万']):
        return 5
    if any(x in class_str for x in ['OP', 'オープン', 'リステッド']):
        return 6
    if any(x in class_str for x in ['G3', 'G３']):
        return 7
    if any(x in class_str for x in ['G2', 'G２']):
        return 8
    if any(x in class_str for x in ['G1', 'G１']):
        return 9
    return 0


def load_error_tags(base_dir):
    """data/error_tags_weekly.json を読み込む"""
    path = os.path.join(base_dir, 'data', 'error_tags_weekly.json')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {'updated_at': None, 'entries': [], 'corrections': {}}


def save_error_tags(data, base_dir):
    """data/error_tags_weekly.json に保存"""
    path = os.path.join(base_dir, 'data', 'error_tags_weekly.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def accumulate_tags(base_dir, new_entries):
    """
    新しいエラータグエントリを蓄積ファイルに追加し、補正係数を再計算する。

    Parameters
    ----------
    base_dir : str
    new_entries : list[dict]
        classify_race_tags() の戻り値のリスト（Noneは除外済み）
    """
    if not new_entries:
        return

    data = load_error_tags(base_dir)
    entries = data.get('entries', [])

    existing_race_ids = {e['race_id'] for e in entries}
    added = 0
    for entry in new_entries:
        if entry['race_id'] not in existing_race_ids:
            entries.append(entry)
            existing_race_ids.add(entry['race_id'])
            added += 1

    data['entries'] = entries
    data['updated_at'] = datetime.now(JST).strftime('%Y-%m-%d %H:%M')
    data['corrections'] = _calc_corrections(entries)

    save_error_tags(data, base_dir)
    print(f'📊 エラータグ蓄積: +{added}件（累計 {len(entries)}件）')
    if data['corrections']:
        n_active = sum(1 for v in data['corrections'].values()
                       if v.get('factor', 1.0) != 1.0)
        print(f'   補正有効: {n_active}条件 / {len(data["corrections"])}条件')


def _calc_corrections(entries):
    """
    蓄積されたエラータグから条件別の補正係数を計算する。

    条件キー × タグ種 ごとに出現頻度を集計し、
    頻度が高い条件では該当パターンの馬のスコアを引き上げる。

    Returns
    -------
    dict: {condition_key: {'factor': float, 'n': int, 'top_tags': list, ...}}
    """
    if not entries:
        return {}

    # 条件キーごとのタグ出現回数
    cond_tags = defaultdict(lambda: defaultdict(int))
    cond_total = defaultdict(int)

    for entry in entries:
        ckey = entry.get('condition_key', '')
        if not ckey:
            continue
        cond_total[ckey] += 1
        for tag in entry.get('tags', []):
            cond_tags[ckey][tag] += 1

    # 全体でのタグ出現率（ベースライン）
    total_races = len(entries)
    global_tag_rate = {}
    for tag in TAG_DEFINITIONS:
        count = sum(cond_tags[ck].get(tag, 0) for ck in cond_tags)
        global_tag_rate[tag] = count / total_races if total_races > 0 else 0

    corrections = {}
    for ckey, n in cond_total.items():
        if n < MIN_SAMPLES_FOR_CORRECTION:
            continue

        tag_counts = cond_tags[ckey]
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]

        # 条件別のタグ発生率 vs 全体のベースラインで補正係数を算出
        adjustment = 0.0
        active_tags = []
        for tag, count in top_tags:
            local_rate = count / n
            base_rate = global_tag_rate.get(tag, 0)
            if base_rate > 0 and local_rate > base_rate * 1.3:
                excess = (local_rate - base_rate) / base_rate
                adjustment += min(excess * 0.1, 0.15)
                active_tags.append({
                    'tag': tag, 'count': count,
                    'local_rate': round(local_rate, 3),
                    'base_rate': round(base_rate, 3),
                })

        factor = 1.0 + min(adjustment, CORRECTION_MAX - 1.0)
        factor = max(CORRECTION_MIN, min(CORRECTION_MAX, factor))

        corrections[ckey] = {
            'factor': round(factor, 4),
            'n': n,
            'top_tags': [{'tag': t, 'count': c} for t, c in top_tags],
            'active_adjustments': active_tags,
        }

    return corrections


def get_correction_factor(base_dir, venue, surface, distance, track_condition,
                          horse=None):
    """
    蓄積されたエラータグに基づく補正係数を返す。

    特定の条件（venue×surface×distance帯×馬場）での
    AIの予測傾向を補正するための係数。

    馬個別のタグマッチングも行う:
    - dist_short_win が多い条件 + 距離短縮馬 → さらに上乗せ
    - escape_win が多い条件 + 逃げ馬 → さらに上乗せ

    Parameters
    ----------
    base_dir : str
    venue, surface : str
    distance : int
    track_condition : str
    horse : dict or None
        馬情報（running_style, history, sex, age等）。個別補正に使用

    Returns
    -------
    float: 補正係数（1.0 = 補正なし）
    """
    data = load_error_tags(base_dir)
    corrections = data.get('corrections', {})
    if not corrections:
        return 1.0

    ckey = _condition_key(venue, surface, distance, track_condition)
    entry = corrections.get(ckey)
    if not entry:
        return 1.0

    base_factor = entry.get('factor', 1.0)

    if horse is None:
        return base_factor

    # 馬個別のタグマッチング（該当パターンの馬をさらに引き上げ）
    horse_bonus = 0.0
    active_tags = {a['tag'] for a in entry.get('active_adjustments', [])}

    if 'dist_short_win' in active_tags:
        hist = horse.get('history', [])
        if hist and distance:
            prev_dist = hist[0].get('distance')
            if prev_dist and prev_dist > distance:
                horse_bonus += 0.05

    if 'dist_ext_win' in active_tags:
        hist = horse.get('history', [])
        if hist and distance:
            prev_dist = hist[0].get('distance')
            if prev_dist and prev_dist < distance:
                horse_bonus += 0.05

    if 'escape_win' in active_tags:
        if horse.get('running_style') == '逃げ':
            horse_bonus += 0.05

    if 'heavy_upset' in active_tags:
        pop = horse.get('popularity', 0)
        if pop and pop >= 8:
            horse_bonus += 0.05

    if 'mare_upset' in active_tags:
        if horse.get('sex') == '牝':
            horse_bonus += 0.03

    if 'young_upset' in active_tags:
        if horse.get('age') == 3:
            horse_bonus += 0.03

    if 'jockey_switch_win' in active_tags:
        hist = horse.get('history', [])
        jockey = horse.get('jockey', '')
        if hist and jockey:
            prev_jockey = hist[0].get('jockey', '')
            if prev_jockey and prev_jockey != jockey:
                horse_bonus += 0.03

    total = base_factor + horse_bonus
    return max(CORRECTION_MIN, min(CORRECTION_MAX, total))


def calc_error_tag_features(horse, base_dir, venue, surface, distance,
                            track_condition):
    """
    エラータグ蓄積データからXGB用の特徴量を生成する。

    XGB再学習時に追加特徴量として使用。条件別のタグ発生率を数値化する。

    Returns
    -------
    dict: {'f_et_escape_rate': float, 'f_et_dist_short_rate': float, ...}
    """
    data = load_error_tags(base_dir)
    corrections = data.get('corrections', {})
    ckey = _condition_key(venue, surface, distance, track_condition)
    entry = corrections.get(ckey, {})

    features = {}
    n = entry.get('n', 0)
    if n < MIN_SAMPLES_FOR_CORRECTION:
        for tag in TAG_DEFINITIONS:
            features[f'f_et_{tag}_rate'] = 0.0
        features['f_et_correction'] = 1.0
        return features

    tag_counts = {t['tag']: t['count'] for t in entry.get('top_tags', [])}
    for tag in TAG_DEFINITIONS:
        count = tag_counts.get(tag, 0)
        features[f'f_et_{tag}_rate'] = round(count / n, 4) if n > 0 else 0.0

    features['f_et_correction'] = entry.get('factor', 1.0)
    return features


def process_weekly_error_tags(base_dir, db_path=None, target_date=None):
    """
    週次処理: race_predictions + history.db からエラータグを分類・蓄積する。

    sunday_results.py から呼ばれる想定。

    Parameters
    ----------
    base_dir : str
    db_path : str or None  keiba.db のパス
    target_date : str or None  対象日（YYYY-MM-DD）。Noneなら当日
    """
    if db_path is None:
        db_path = os.path.join(base_dir, 'data', 'keiba.db')
    hist_db_path = os.path.join(base_dir, 'data', 'history.db')

    if not os.path.exists(db_path):
        print('⚠ keiba.db が見つかりません。エラータグ処理をスキップ。')
        return

    if target_date is None:
        target_date = datetime.now(JST).strftime('%Y-%m-%d')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 対象日の予測データを取得（actual_place が設定済み = 結果照合済み）
    rows = conn.execute('''
        SELECT race_id, racecourse, race_num, horse_num, horse_name,
               popularity, tansho_odds, rl_rank, win_prob, cal_prob,
               actual_place
        FROM race_predictions
        WHERE date = ? AND actual_place IS NOT NULL
    ''', (target_date,)).fetchall()
    conn.close()

    if not rows:
        print(f'⚠ {target_date} のエラータグ対象データなし。')
        return

    # race_id ごとにグループ化
    races = defaultdict(list)
    for r in rows:
        races[r['race_id']].append(dict(r))

    # history.db からレース情報・馬の前走情報を取得
    hist_conn = None
    if os.path.exists(hist_db_path):
        hist_conn = sqlite3.connect(hist_db_path)
        hist_conn.row_factory = sqlite3.Row

    new_entries = []
    for race_id, pred_horses in races.items():
        race_result = _build_race_result(race_id, pred_horses, hist_conn)
        if not race_result:
            continue

        predictions = {}
        for ph in pred_horses:
            predictions[ph['horse_num']] = {
                'rl_rank': ph['rl_rank'],
                'win_prob': ph['win_prob'],
            }

        entry = classify_race_tags(race_result, predictions, hist_db_path)
        if entry:
            new_entries.append(entry)

    if hist_conn:
        hist_conn.close()

    accumulate_tags(base_dir, new_entries)


def _build_race_result(race_id, pred_horses, hist_conn):
    """race_predictions + history.db からレース結果dictを構築する"""
    if not hist_conn:
        return None

    race_row = hist_conn.execute(
        'SELECT * FROM race_history WHERE race_id = ?', (race_id,)
    ).fetchone()
    if not race_row:
        return None

    horse_rows = hist_conn.execute(
        'SELECT * FROM horse_history WHERE race_id = ? ORDER BY place',
        (race_id,)
    ).fetchall()
    if not horse_rows:
        return None

    # 前走情報を取得（各馬の直近レース）
    horses_with_history = []
    pred_map = {ph['horse_num']: ph for ph in pred_horses}

    for hr in horse_rows:
        h = dict(hr)
        horse_name = h.get('horse_name', '')

        prev_rows = hist_conn.execute('''
            SELECT hh.distance, hh.jockey, hh.race_id, rh.race_class
            FROM horse_history hh
            LEFT JOIN race_history rh ON hh.race_id = rh.race_id
            WHERE hh.horse_name = ? AND hh.race_id != ?
            ORDER BY hh.date DESC LIMIT 1
        ''', (horse_name, race_id)).fetchall()

        history = []
        for pr in prev_rows:
            history.append({
                'distance': pr['distance'],
                'jockey': pr['jockey'],
                'race_class': pr['race_class'],
                'date': '',
            })
        h['history'] = history

        ph = pred_map.get(h.get('horse_num'))
        if ph:
            h['popularity'] = h.get('popularity') or ph.get('popularity')

        horses_with_history.append(h)

    return {
        'race_id': race_id,
        'date': race_row['date'],
        'racecourse': race_row['racecourse'],
        'surface': race_row['surface'],
        'distance': race_row['distance'],
        'track_condition': race_row['track_condition'],
        'race_class': race_row['race_class'] or '',
        'first_3f': race_row['first_3f'],
        'horses': horses_with_history,
    }
