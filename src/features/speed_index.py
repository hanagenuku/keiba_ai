"""
スピード指数（Speed Figure）計算モジュール。

【設計方針】
- 基準タイム: (distance, surface, track_condition) ごとの1着馬 finish_time 中央値
- Track Variant: 同日×同競馬場の全レースで「基準タイムからのズレ」の中央値
- スピード指数: (基準タイム - (実タイム - Track Variant)) / 基準タイム * 1000
  → 高いほど速い（= 強い）
- 上がり3Fスピード指数: (レース最速上がり - 自身の上がり) / レース最速上がり * 1000
  → 高いほど上がりが速い

【キャッシュ】
- data/speed_index_cache.pkl に保存。init_engine() から自動ロード。

【利用方法】
    calc = SpeedIndexCalculator(hist_db_path)
    fig  = calc.calc_speed_figure(finish_time, distance, surface, track_condition, date, racecourse)
    agari_fig = calc.calc_agari_speed_figure(agari3f, race_last_3f)
"""
import os
import pickle
import sqlite3
import statistics


class SpeedIndexCalculator:
    """基準タイムと Track Variant を事前計算してスピード指数を提供するクラス。"""

    def __init__(self, db_path):
        self.base_times = {}      # (dist, surf, tc) → median winner time
        self.track_variants = {}  # (date, racecourse) → median deviation (秒)
        self._build(db_path)

    # ── 構築 ──────────────────────────────────────────────────

    def _build(self, db_path):
        if not os.path.exists(db_path):
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # ① 全1着馬のタイムを取得（finish_time > 0 のみ）
        rows = conn.execute("""
            SELECT h.finish_time, h.distance, h.surface, h.date, h.racecourse,
                   COALESCE(r.track_condition, '良') AS track_condition
            FROM horse_history h
            LEFT JOIN race_history r ON h.race_id = r.race_id
            WHERE h.place = 1
              AND h.finish_time IS NOT NULL
              AND h.finish_time > 0
              AND h.distance IS NOT NULL
        """).fetchall()
        conn.close()

        # ── Step 1: 基準タイム ────────────────────────────────
        from collections import defaultdict
        winner_times = defaultdict(list)
        all_by_dist_surf = defaultdict(list)  # (dist, surf) → list of finish_time (for fallback)

        for r in rows:
            dist = int(r['distance'] or 0)
            surf = r['surface'] or '芝'
            tc   = r['track_condition'] or '良'
            ft   = float(r['finish_time'])
            if dist <= 0 or ft <= 0:
                continue
            winner_times[(dist, surf, tc)].append(ft)
            if tc == '良':
                all_by_dist_surf[(dist, surf)].append(ft)

        for key, times in winner_times.items():
            if len(times) >= 1:
                self.base_times[key] = statistics.median(times)

        # サンプル不足（n<5）の組み合わせは「良」ベースから馬場補正で推定
        TC_ADJUST_SEC = {'良': 0.0, '稍重': 0.5, '重': 1.2, '不良': 2.0}
        for (dist, surf, tc), times in winner_times.items():
            if len(times) < 5 and tc != '良':
                base_key = (dist, surf, '良')
                if base_key in self.base_times:
                    self.base_times[(dist, surf, tc)] = (
                        self.base_times[base_key] + TC_ADJUST_SEC.get(tc, 0.0)
                    )

        # ── Step 2: Track Variant ──────────────────────────────
        day_deviations = defaultdict(list)

        for r in rows:
            dist = int(r['distance'] or 0)
            surf = r['surface'] or '芝'
            tc   = r['track_condition'] or '良'
            ft   = float(r['finish_time'])
            date = str(r['date'] or '')[:10]  # YYYY-MM-DD か YYYYMMDD どちらでも上位8桁
            rc   = r['racecourse'] or ''
            if dist <= 0 or ft <= 0 or not date or not rc:
                continue
            base = self.base_times.get((dist, surf, tc))
            if base is None:
                continue
            deviation = ft - base  # 負=速い、正=遅い
            day_deviations[(date, rc)].append(deviation)

        for key, devs in day_deviations.items():
            if len(devs) >= 2:
                self.track_variants[key] = statistics.median(devs)

    # ── 計算 ──────────────────────────────────────────────────

    def calc_speed_figure(self, finish_time, distance, surface,
                          track_condition, date, racecourse):
        """1頭のスピード指数を計算。finish_time=None/0 のときは None を返す。

        スピード指数 = (基準タイム - 調整済タイム) / 基準タイム * 1000
        調整済タイム = finish_time - track_variant（その日の馬場速度補正）
        """
        if not finish_time or float(finish_time) <= 0:
            return None

        ft   = float(finish_time)
        dist = int(distance or 0)
        surf = surface or '芝'
        tc   = track_condition or '良'

        # 日付文字列正規化
        date_str = str(date or '').replace('-', '')[:8]
        if len(date_str) == 8:
            date_norm = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
        else:
            date_norm = str(date or '')[:10]

        base = self.base_times.get((dist, surf, tc))
        if base is None:
            # 「良」でフォールバック
            base = self.base_times.get((dist, surf, '良'))
        if base is None or base <= 0:
            return None

        tv = self.track_variants.get((date_norm, racecourse or ''), 0.0)
        adjusted = ft - tv
        fig = (base - adjusted) / base * 1000
        return round(fig, 2)

    def calc_agari_speed_figure(self, agari3f, race_last_3f):
        """上がり3Fスピード指数を計算。

        agari_speed_figure = (race_last_3f - horse_agari3f) / race_last_3f * 1000
        → 正 = 上がりが速い
        """
        if not agari3f or not race_last_3f:
            return None
        a = float(agari3f)
        r = float(race_last_3f)
        if r <= 0 or a <= 0:
            return None
        return round((r - a) / r * 1000, 2)

    def __repr__(self):
        return (f'SpeedIndexCalculator('
                f'base_times={len(self.base_times)}, '
                f'track_variants={len(self.track_variants)})')


# ── キャッシュ I/O ────────────────────────────────────────────

def load_speed_index_calculator(base_dir):
    """speed_index_cache.pkl を読み込む。なければ history.db から構築して保存。"""
    cache_path = os.path.join(base_dir, 'data', 'speed_index_cache.pkl')
    db_path    = os.path.join(base_dir, 'data', 'history.db')

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    if not os.path.exists(db_path):
        return None

    calc = SpeedIndexCalculator(db_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(calc, f)
    return calc


def rebuild_speed_index_cache(base_dir):
    """キャッシュを強制再構築して保存。horse_features.csv 再生成前に呼ぶ。"""
    cache_path = os.path.join(base_dir, 'data', 'speed_index_cache.pkl')
    db_path    = os.path.join(base_dir, 'data', 'history.db')

    calc = SpeedIndexCalculator(db_path)
    with open(cache_path, 'wb') as f:
        pickle.dump(calc, f)
    print(f'✅ speed_index_cache.pkl 再構築完了: {calc}')
    return calc
