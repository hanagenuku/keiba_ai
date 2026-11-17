"""
スピード指数（Speed Figure）計算モジュール。

【設計方針】
- 基準タイム: (distance, surface, track_condition) ごとの1着馬 finish_time 中央値
- Track Variant: 同日×同競馬場×芝ダートの勝ち馬で「基準タイムからのずれの比率」の中央値（2026-11-18〜）
- スピード指数: (基準タイム - (実タイム - 比率×基準タイム)) / 基準タイム * 1000
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


#: キャッシュの形式。変えたら上げる（古い形式のキャッシュは読み込み時に作り直す）。
#: v2（2026-11-18）: 馬場差を「日付×競馬場×芝ダート」ごとに、基準タイムに対する比率で出す。
CACHE_VERSION = 2

#: 馬場差を出すのに要る、同じ日・同じ競馬場・同じ芝ダートの勝ち馬の数。
#: これ未満なら「日付×競馬場」（芝ダート合算）の値に戻し、それも無ければ 0。
MIN_RACES_PER_SURFACE = 2


def _norm_date(raw):
    d = str(raw or '').replace('-', '')[:8]
    return f'{d[:4]}-{d[4:6]}-{d[6:8]}' if len(d) == 8 else str(raw or '')[:10]


class SpeedIndexCalculator:
    """基準タイムと馬場差（Track Variant）を事前計算してスピード指数を提供するクラス。

    🔴 2026-11-18 に2点を直した（ユーザー指示「効果があろうがなかろうがなおしてください」）:

    1. **芝とダートを分ける。** 以前は「日付×競馬場」で1つの値だったので、
       同じ日に芝が高速・ダートが時計のかかる馬場でも、両方を同じ値で補正していた。
       いまは「日付×競馬場×芝ダート」ごと。数が足りなければ合算の値に戻す。
    2. **距離で割る。** 以前は秒のずれをそのまま中央値にしていたので、
       1200m の1秒と 2400m の1秒が同じ重さで混ざっていた。
       いまは「基準タイムに対するずれの比率」の中央値を馬場差にし、
       補正は その比率 × そのレースの基準タイム（秒）で行う。
    """

    def __init__(self, db_path):
        self.version = CACHE_VERSION
        self.base_times = {}         # (dist, surf, tc) → 勝ち馬タイムの中央値（秒）
        self.track_variants = {}     # (date, racecourse, surface) → ずれの比率の中央値
        self.track_variants_all = {} # (date, racecourse) → 同上（芝ダート合算・予備）
        self.latest_date = ''        # 馬場差を出した最後の日（キャッシュの古さの判定に使う）
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
        for r in rows:
            dist = int(r['distance'] or 0)
            surf = r['surface'] or '芝'
            tc   = r['track_condition'] or '良'
            ft   = float(r['finish_time'])
            if dist <= 0 or ft <= 0:
                continue
            winner_times[(dist, surf, tc)].append(ft)

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

        # ── Step 2: 馬場差（比率）────────────────────────────
        by_surface = defaultdict(list)
        by_day = defaultdict(list)
        for r in rows:
            dist = int(r['distance'] or 0)
            surf = r['surface'] or '芝'
            tc   = r['track_condition'] or '良'
            ft   = float(r['finish_time'])
            date = _norm_date(r['date'])
            rc   = r['racecourse'] or ''
            if dist <= 0 or ft <= 0 or len(date) != 10 or not rc:
                continue
            base = self.base_times.get((dist, surf, tc))
            if not base:
                continue
            ratio = (ft - base) / base   # 負=速い、正=遅い
            by_surface[(date, rc, surf)].append(ratio)
            by_day[(date, rc)].append(ratio)

        for key, v in by_surface.items():
            if len(v) >= MIN_RACES_PER_SURFACE:
                self.track_variants[key] = statistics.median(v)
        for key, v in by_day.items():
            if len(v) >= MIN_RACES_PER_SURFACE:
                self.track_variants_all[key] = statistics.median(v)
        if by_day:
            self.latest_date = max(k[0] for k in by_day)

    # ── 計算 ──────────────────────────────────────────────────

    def variant_ratio(self, date, racecourse, surface):
        """その日・その競馬場・その芝ダートの馬場差（比率）。無ければ合算、それも無ければ 0。"""
        d = _norm_date(date)
        rc = racecourse or ''
        v = self.track_variants.get((d, rc, surface or '芝'))
        if v is None:
            v = self.track_variants_all.get((d, rc), 0.0)
        return v

    def calc_speed_figure(self, finish_time, distance, surface,
                          track_condition, date, racecourse):
        """1頭のスピード指数を計算。finish_time=None/0 のときは None を返す。

        スピード指数 = (基準タイム - 調整済タイム) / 基準タイム * 1000
        調整済タイム = finish_time - 馬場差の比率 × 基準タイム
        """
        if not finish_time or float(finish_time) <= 0:
            return None

        ft   = float(finish_time)
        dist = int(distance or 0)
        surf = surface or '芝'
        tc   = track_condition or '良'

        base = self.base_times.get((dist, surf, tc))
        if base is None:
            # 「良」でフォールバック
            base = self.base_times.get((dist, surf, '良'))
        if base is None or base <= 0:
            return None

        adjusted = ft - self.variant_ratio(date, racecourse, surf) * base
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
        return (f'SpeedIndexCalculator(v{getattr(self, "version", 1)}, '
                f'base_times={len(self.base_times)}, '
                f'track_variants={len(self.track_variants)}, '
                f'latest={getattr(self, "latest_date", "")})')


# ── キャッシュ I/O ────────────────────────────────────────────

def _history_latest_date(db_path):
    try:
        conn = sqlite3.connect(db_path)
        d = conn.execute('SELECT MAX(date) FROM horse_history WHERE place = 1 '
                         'AND finish_time > 0').fetchone()[0]
        conn.close()
        return _norm_date(d) if d else ''
    except Exception:
        return ''


def cache_is_usable(calc, db_path):
    """キャッシュがいまの形式で、history.db の最後の日まで馬場差を持っているか。

    🔴 2026-11-18 に見つかった不具合: 以前は「キャッシュがあれば中身を問わず使う」作りで、
       どのワークフローも作り直していなかった。リポジトリのキャッシュは 2026-07-26 で止まっていて、
       それ以降の約4か月の過去走は**馬場差の補正が 0（無補正）**になっていた（学習データも本番も）。
    """
    if getattr(calc, 'version', 1) != CACHE_VERSION:
        return False
    latest = _history_latest_date(db_path)
    return not latest or getattr(calc, 'latest_date', '') >= latest


def load_speed_index_calculator(base_dir):
    """speed_index_cache.pkl を読み込む。古い形式・古い中身なら history.db から作り直して保存する。"""
    cache_path = os.path.join(base_dir, 'data', 'speed_index_cache.pkl')
    db_path    = os.path.join(base_dir, 'data', 'history.db')

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                calc = pickle.load(f)
            if not os.path.exists(db_path) or cache_is_usable(calc, db_path):
                return calc
            print('  🔨 speed_index_cache.pkl が古いので作り直す')
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
