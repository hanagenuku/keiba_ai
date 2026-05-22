"""
トラックバイアス分析。
当日の結果データから内外・ペース・時計偏差を計算する。
"""
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from src.features.engine import dist_zone_label

_AGARI_BASE = {
    ('芝', '短距離'): 34.0, ('芝', 'マイル'): 34.5,
    ('芝', '中距離'): 35.0, ('芝', '長距離'): 35.5,
    ('ダート', '短距離'): 35.0, ('ダート', 'マイル'): 35.5,
    ('ダート', '中距離'): 36.0, ('ダート', '長距離'): 36.5,
}


def analyze_bias(results):
    """レース結果リストから競馬場別バイアスを分析する。

    Args:
        results: fetch_results が返すレース結果リスト

    Returns:
        {競馬場名: {inner_outer, pace_bias, track_speed, summary, ...}} の辞書
    """
    bias_by_course = {}
    for rc in {r['racecourse'] for r in results}:
        rc_res = [r for r in results if r['racecourse'] == rc]

        # 内外バイアス（上位3頭の馬番平均 vs 全体中央値）
        io_scores = []
        for r in rc_res:
            fin = r.get('finishers', [])
            if len(fin) < 3:
                continue
            num_h = max((h.get('post_position', h.get('num', 8)) for h in fin), default=8)
            avg_all = (num_h + 1) / 2
            avg_top3 = statistics.mean(
                [h.get('post_position', h.get('num', 8)) for h in fin[:3]]
            )
            io_scores.append((avg_all - avg_top3) / max(num_h / 4, 1))
        inner_outer = (
            max(-3.0, min(3.0, statistics.mean(io_scores) * 2)) if io_scores else 0.0
        )

        # ペース・脚質バイアス（上位3頭の脚質分布）
        style_cnt = defaultdict(int)
        total = 0
        for r in rc_res:
            for h in r.get('finishers', [])[:3]:
                style_cnt[h.get('running_style', '差し')] += 1
                total += 1
        front = (style_cnt['逃げ'] + style_cnt['先行']) / max(total, 1)
        pace_bias = max(-3.0, min(3.0, (front - 0.45) * 6))

        # 時計偏差（勝ち馬の上がり3F vs 標準値）
        speed_devs = []
        for r in rc_res:
            fin = r.get('finishers', [])
            if not fin:
                continue
            winner = fin[0]
            agari = winner.get('agari3f', 0) or 0
            if agari < 30:
                continue
            dist = winner.get('distance', r.get('distance', 2000))
            surf = winner.get('surface', r.get('surface', '芝'))
            zone = dist_zone_label(dist)
            base_val = _AGARI_BASE.get((surf, zone), 35.0)
            speed_devs.append(max(-2.0, min(2.0, (base_val - agari) / 0.8)))
        track_speed = round(statistics.mean(speed_devs), 2) if speed_devs else 0.0

        parts = []
        if abs(inner_outer) >= 1.0:
            parts.append('内有利' if inner_outer > 0 else '外有利')
        if abs(pace_bias) >= 1.0:
            parts.append('先行有利' if pace_bias > 0 else '差し・追込有利')
        if abs(track_speed) >= 0.5:
            parts.append('時計速め' if track_speed > 0 else '時計遅め')

        bias_by_course[rc] = {
            'inner_outer': round(inner_outer, 2),
            'pace_bias':   round(pace_bias, 2),
            'track_speed': track_speed,
            'summary':     '・'.join(parts) if parts else 'フラット',
            'style_dist':  dict(style_cnt),
            'race_count':  len(rc_res),
        }
    return bias_by_course


def build_avg_bias(bias_by_course, prev_bias=None):
    """競馬場別バイアスから全体平均バイアス辞書を生成する。

    Args:
        bias_by_course : analyze_bias の返り値
        prev_bias      : 前週バイアス（bias_by_course が空の場合のフォールバック）

    Returns:
        avg_bias 辞書 {inner_outer, pace_bias, track_speed, summary, by_course, date}
    """
    _empty = {
        'inner_outer': 0, 'pace_bias': 0, 'track_speed': 0,
        'summary': 'フラット', 'by_course': {}, 'date': '',
    }
    if not bias_by_course:
        return prev_bias or _empty

    try:
        avg_io = statistics.mean([b['inner_outer'] for b in bias_by_course.values()])
        avg_pb = statistics.mean([b['pace_bias']   for b in bias_by_course.values()])
        avg_ts = statistics.mean([b['track_speed'] for b in bias_by_course.values()])
    except statistics.StatisticsError:
        return prev_bias or {**_empty, 'by_course': bias_by_course}

    parts = []
    if abs(avg_io) >= 1.0: parts.append('内有利' if avg_io > 0 else '外有利')
    if abs(avg_pb) >= 1.0: parts.append('先行有利' if avg_pb > 0 else '差し・追込有利')
    if abs(avg_ts) >= 0.5: parts.append('時計速め' if avg_ts > 0 else '時計遅め')

    jst = timezone(timedelta(hours=9))
    return {
        'inner_outer': round(avg_io, 2),
        'pace_bias':   round(avg_pb, 2),
        'track_speed': round(avg_ts, 2),
        'summary':     '・'.join(parts) if parts else 'フラット',
        'by_course':   bias_by_course,
        'date':        datetime.now(jst).strftime('%Y-%m-%d'),
    }
