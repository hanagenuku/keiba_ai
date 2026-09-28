"""history.db から「見解に書いてよい事実」だけを抽出する。

🔴 推測で埋めない。データが無い項目は **キーを作らない**（None を入れて後段で
「不明」と書かせるのではなく、そもそも存在させない）。呼び出し側は
`if 'dist_change' in facts` の形で分岐する。

🔴 リーク防止: 対象レースの日付 **より前** の走りだけを使う。history.db には
対象レース自身の結果が入っていることがある（結果取得後に生成し直した場合）。
"""

import os
import re
import sqlite3
from datetime import datetime

# 見解に使う過去走の本数。5走を超えると市場も同じ履歴を見ており差が出ない
# （2026-08-18: その馬を見た回数の上乗せは3〜5走でピークを打ち10走以降は負に転じる）
MAX_HISTORY = 5

# 「傾向」として書いてよい最低の走数。1〜2走で「末脚は確か」「道悪は得意」と
# 書くのは North Star #7（小さいNの好成績を信じない）に反する
MIN_RUNS_FOR_TREND = 3

# body_weight は充足率 6.5% しかないので事実として扱わない（biz/PHASE4_DESIGN.md §5）
_UNUSABLE_COLUMNS = ('body_weight', 'body_weight_diff')

_DIST_RE = re.compile(r'(\d+)\s*m\s*(芝|ダート|ダ)?')


def parse_dist(dist_text):
    """'1800mダート' -> (1800, 'ダート')。読めなければ (None, None)。"""
    if not dist_text:
        return None, None
    m = _DIST_RE.search(str(dist_text))
    if not m:
        return None, None
    surface = m.group(2)
    if surface == 'ダ':
        surface = 'ダート'
    return int(m.group(1)), surface


def _fetch_past_runs(conn, horse_name, before_date, limit=MAX_HISTORY):
    """対象日より前の走りを新しい順に返す。"""
    if not horse_name or not before_date:
        return []
    sql = """
        SELECT h.date, h.racecourse, h.surface, h.distance, h.place, h.field_size,
               h.popularity, h.class_grade, h.agari3f, h.agari_rank,
               h.running_style, h.weight_load, h.jockey, h.time_diff_sec,
               r.track_condition
          FROM horse_history h
          LEFT JOIN race_history r ON h.race_id = r.race_id
         WHERE h.horse_name = ?
           AND h.date < ?
         ORDER BY h.date DESC
         LIMIT ?
    """
    try:
        rows = conn.execute(sql, (horse_name, before_date, limit)).fetchall()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]


def _days_between(newer, older):
    try:
        a = datetime.strptime(str(newer)[:10], '%Y-%m-%d')
        b = datetime.strptime(str(older)[:10], '%Y-%m-%d')
    except (ValueError, TypeError):
        return None
    return (a - b).days


def _is_placed(run):
    """3着内か。place が無ければ判定しない。"""
    p = run.get('place')
    if not isinstance(p, int) or p <= 0:
        return None
    return p <= 3


def build_facts(horse, race, target_date, conn):
    """1頭ぶんの「書いてよい事実」を返す。

    存在しない事実はキーを作らない。すべて history.db と latest.json の実値のみ。
    """
    facts = {}
    today_dist, today_surface = parse_dist(race.get('dist'))
    if today_dist:
        facts['today_dist'] = today_dist
    if today_surface:
        facts['today_surface'] = today_surface

    runs = _fetch_past_runs(conn, horse.get('name'), target_date)
    facts['n_past_runs'] = len(runs)
    if not runs:
        # 新馬・初出走。書ける過去の事実はゼロ
        return facts

    facts['past_runs'] = runs
    last = runs[0]

    # --- 距離の変化 -------------------------------------------------------
    if today_dist and isinstance(last.get('distance'), int) and last['distance'] > 0:
        delta = today_dist - last['distance']
        if delta != 0:
            facts['dist_change'] = {
                'from': last['distance'], 'to': today_dist, 'delta': delta,
                'direction': '延長' if delta > 0 else '短縮',
            }

    # --- コースの変化 -----------------------------------------------------
    # ⚠ 過去1走しか無い馬は当コース初になるのが当たり前で、書くと全馬同じ文になる。
    #    2走以上ある馬に限る
    if race.get('_venue') and last.get('racecourse') and len(runs) >= 2:
        if race['_venue'] != last['racecourse']:
            facts['venue_change'] = {'from': last['racecourse'], 'to': race['_venue']}

    # --- 芝ダ替わり -------------------------------------------------------
    if today_surface and last.get('surface') and today_surface != last['surface']:
        facts['surface_change'] = {'from': last['surface'], 'to': today_surface}

    # --- 近走の連続3着内 --------------------------------------------------
    streak = 0
    for r in runs:
        placed = _is_placed(r)
        if placed is None or not placed:
            break
        streak += 1
    if streak >= 2:
        facts['placed_streak'] = streak

    # --- 末脚（上がり順位が上位だった走り） --------------------------------
    # ⚠ 1〜2走で「末脚は確か」と書くのは North Star #7（小さいNを信じない）に反する。
    #    3走以上ある馬だけ事実として扱う
    good_agari = [r for r in runs
                  if isinstance(r.get('agari_rank'), int) and 1 <= r['agari_rank'] <= 3]
    if good_agari and len(runs) >= MIN_RUNS_FOR_TREND:
        facts['agari_top3'] = {
            'count': len(good_agari), 'of': len(runs),
            'best_rank': min(r['agari_rank'] for r in good_agari),
        }

    # --- 前走で着外だが末脚は上位（「着外だが末脚は評価」） -----------------
    lp = _is_placed(last)
    if lp is False and isinstance(last.get('agari_rank'), int) and last['agari_rank'] <= 3:
        facts['last_lost_but_good_agari'] = {
            'place': last['place'], 'agari_rank': last['agari_rank'],
        }

    # --- 脚質（直近で最も多いもの） ---------------------------------------
    styles = [r['running_style'] for r in runs if r.get('running_style')]
    if styles:
        facts['style'] = max(set(styles), key=styles.count)

    # --- 道悪実績 ---------------------------------------------------------
    heavy = [r for r in runs if r.get('track_condition') in ('稍重', '重', '不良')]
    if len(heavy) >= MIN_RUNS_FOR_TREND:
        hit = [r for r in heavy if _is_placed(r)]
        facts['heavy_record'] = {'runs': len(heavy), 'placed': len(hit)}

    # --- クラスの変化 -----------------------------------------------------
    if last.get('class_grade') and race.get('name'):
        facts['last_class'] = last['class_grade']

    # --- 市場の見立てを超えた走り -----------------------------------------
    beat = [r for r in runs
            if isinstance(r.get('place'), int) and isinstance(r.get('popularity'), int)
            and 0 < r['place'] < r['popularity']]
    if beat and len(runs) >= MIN_RUNS_FOR_TREND:
        facts['beat_market'] = {'count': len(beat), 'of': len(runs)}

    # --- 騎手替わり -------------------------------------------------------
    if horse.get('jockey') and last.get('jockey') and horse['jockey'] != last['jockey']:
        facts['jockey_change'] = {'from': last['jockey'], 'to': horse['jockey']}

    # --- 休養間隔 ---------------------------------------------------------
    gap = _days_between(target_date, last.get('date'))
    if gap is not None and gap >= 0:
        facts['days_since_last'] = gap

    # --- 前走の着差 -------------------------------------------------------
    if isinstance(last.get('time_diff_sec'), (int, float)):
        facts['last_time_diff'] = float(last['time_diff_sec'])
    if isinstance(last.get('place'), int) and last['place'] > 0:
        facts['last_place'] = last['place']

    return facts


def open_history(base_dir):
    """history.db を読み取り専用で開く。無ければ None。"""
    path = os.path.join(base_dir, 'data', 'history.db')
    if not os.path.exists(path):
        return None
    # LFS ポインタ（実体が無い）を掴まない
    try:
        with open(path, 'rb') as f:
            head = f.read(16)
        if head.startswith(b'version https:'):
            return None
    except OSError:
        return None
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn
