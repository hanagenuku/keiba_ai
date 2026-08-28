"""オッズ時系列の収集（設計書 Phase 0 → B「リアルタイム収集」の実体）。

■ なぜ長時間ジョブ + 内部ループなのか
GitHub Actions の cron は5分粒度で、しかも実際には5〜20分遅延する。
「発走30分前」のような時点を揃えたいのに、cron のジッタがそのまま
サンプリング誤差になる。1本のジョブを数時間走らせ、内部で時刻を見て
poll する方が正確で、公開リポジトリなので Actions 分数は無料。

■ 設計書に無かったが必須の6点（2026-08-27に洗い出し）
  ① 発走時刻   … 「発走何分前か」が無いとレース間で揃えられない。
                  parse_header は取っていたがDBに保存されていなかった
  ② 複勝の範囲 … (min+max)/2 に潰すと市場の不確実性が消える
  ③ 取消・除外 … 1頭消えると全馬のオッズが動く。記録しないと
                  「情報を持った資金流入」と誤読する
  ④ 完全性     … 一部の馬しか取れなかった回で市場シェアを正規化すると
                  静かに誤った値になる
  ⑤ 粒度       … 粗く集めたものは後で細かくできない。取れる限り細かく集める
  ⑥ 最終オッズ … 結果ページからオッズ列は消滅済み（2026-08-03③）。
                  発走直前のスナップショットが唯一の最終オッズになる

■ リーク対策
captured_at と minutes_to_post を必ず記録する。学習時は
「T分前モデルなら minutes_to_post >= T の行だけ」で切れば時点が固定される。
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._session import create_session                    # noqa: E402
from src.scraper.jra_scraper import (                          # noqa: E402
    get_kaisai_on_date, find_r01_odds, fetch_odds_for_race,
    fetch_races_on_date,
)
from src.utils.db import init_db, save_odds_snapshots, save_race_schedule  # noqa: E402

JST = timezone(timedelta(hours=9))

# North Star #4: 新しいリクエスト元には必ず件数上限を設ける。
MAX_REQUESTS_DEFAULT = 4000
# 発走の何分前から poll を始めるか / 何分前で打ち切るか（締切は発走1〜2分前）
WINDOW_START_MIN = 75.0
WINDOW_END_MIN = 0.0
# 1周の最小間隔（秒）。⑤より、取れる限り細かく。
INTERVAL_SEC_DEFAULT = 120


def _post_dt(date_str, post_time):
    """'20260830' + '15:00' → JSTのdatetime。取れなければ None。"""
    if not post_time or ':' not in str(post_time):
        return None
    try:
        hh, mm = str(post_time).split(':')[:2]
        d = datetime.strptime(date_str, '%Y%m%d')
        return datetime(d.year, d.month, d.day, int(hh), int(mm), tzinfo=JST)
    except Exception:
        return None


def build_schedule(sess, date_str, base_dir):
    """出馬表から発走時刻と頭数を取り、race_schedule に保存して返す。

    ⚠ これを最初にやらないと「発走何分前か」が永久に復元できない。
    """
    races, failures = fetch_races_on_date(sess, date_str)
    rows = []
    for r in races:
        rows.append({
            'race_id': r.get('id'),
            'date': r.get('date') or date_str,
            'racecourse': r.get('racecourse', ''),
            'race_num': r.get('race_num'),
            'post_time': r.get('start_time'),
            'n_horses': len(r.get('horses') or []),
        })
    saved = save_race_schedule(rows, base_dir=base_dir)
    n_with_time = sum(1 for x in rows if x['post_time'])
    print(f'📅 出馬表 {len(rows)}レース（発走時刻あり {n_with_time}）'
          f' / 新規保存 {saved} / parse失敗 {len(failures)}')
    if rows and n_with_time == 0:
        print('🔴 発走時刻が1件も取れていない。このまま集めても'
              '「発走何分前か」が出せず、時点別の分析ができない。')
    return rows


def collect(base_dir, date_str=None, max_minutes=330,
            interval_sec=INTERVAL_SEC_DEFAULT,
            max_requests=MAX_REQUESTS_DEFAULT, dry_run=False):
    init_db(base_dir)
    sess = create_session()
    now = datetime.now(JST)
    date_str = date_str or now.strftime('%Y%m%d')

    kaisai = get_kaisai_on_date(sess, date_str)
    if not kaisai:
        print(f'❌ {date_str} の開催情報が見つかりません（開催日ではない可能性）')
        return {'races': 0, 'snapshots': 0, 'requests': 0}
    print(f'開催: {kaisai}')

    sched = {r['race_id']: r for r in build_schedule(sess, date_str, base_dir)}
    if not sched:
        print('❌ 出馬表が取れないので収集しない')
        return {'races': 0, 'snapshots': 0, 'requests': 0}

    # R01 の suffix は会場ごとに1回だけ探索する（他レースは計算で出る）
    r01 = {}
    for base, venue in kaisai.items():
        sfx = find_r01_odds(sess, base, date_str)
        if sfx is None:
            print(f'  ⚠ {venue}: オッズR01が見つからない（未発売の可能性）')
        else:
            print(f'  ✅ {venue}: R01 suffix={sfx}')
        r01[base] = sfx

    deadline = now + timedelta(minutes=max_minutes)
    req = 0
    total_saved = 0
    rounds = 0
    seen_horses = {}          # race_id -> これまでに一度でも見た馬番の集合
    unmapped = set()          # 収集できなかったレース（1回だけ警告する）

    while datetime.now(JST) < deadline and req < max_requests:
        rounds += 1
        tick = datetime.now(JST)
        due = []
        for rid, meta in sched.items():
            pdt = _post_dt(date_str, meta.get('post_time'))
            if pdt is None:
                continue
            mtp = (pdt - tick).total_seconds() / 60.0
            if WINDOW_END_MIN <= mtp <= WINDOW_START_MIN:
                due.append((rid, meta, mtp))
        if not due:
            # まだ早い / 全部終わった
            nxt = [(_post_dt(date_str, m.get('post_time')), r)
                   for r, m in sched.items() if _post_dt(date_str, m.get('post_time'))]
            future = [p for p, _ in nxt if p and (p - tick).total_seconds() / 60.0 > WINDOW_START_MIN]
            if not future:
                print('■ 収集対象のレースが無くなったので終了')
                break
            time.sleep(min(interval_sec, 60))
            continue

        for rid, meta, mtp in due:
            if req >= max_requests:
                print(f'⚠ リクエスト上限 {max_requests} に到達'); break
            # race_id は 'YYYYMMDD_場_R'。odds_base は kaisai のキー側にある
            odds_base = None
            for b, v in kaisai.items():
                if v == meta.get('racecourse'):
                    odds_base = b
                    break
            if odds_base is None or r01.get(odds_base) is None:
                # 🔴 黙って飛ばすと「1件も集まらなかった」ことに気づけない。
                #    空振りと該当なしは別物（2026-08-16の教訓）。
                if rid not in unmapped:
                    unmapped.add(rid)
                    why = ('会場名が開催情報と一致しない' if odds_base is None
                           else 'R01のsuffixが見つからなかった')
                    print(f'  ⚠ {rid} ({meta.get("racecourse")}) を収集できない: {why}')
                continue
            try:
                omap = fetch_odds_for_race(sess, odds_base, int(meta['race_num']),
                                           date_str, r01[odds_base])
                req += 1
            except Exception as e:                       # noqa: BLE001
                print(f'  ⚠ {rid} 取得失敗: {e}')
                req += 1
                continue
            if not omap:
                continue
            expected = meta.get('n_horses') or 0
            prev = seen_horses.setdefault(rid, set())
            prev |= set(omap.keys())
            captured_at = tick.strftime('%Y-%m-%d %H:%M:%S')
            rows = []
            for num in sorted(prev):
                o = omap.get(num)
                rows.append({
                    'race_id': rid, 'horse_num': num,
                    'tansho': (o or {}).get('tansho'),
                    'fukusho': (o or {}).get('fukusho'),
                    'fukusho_min': (o or {}).get('fukusho_min'),
                    'fukusho_max': (o or {}).get('fukusho_max'),
                    'captured_at': captured_at,
                    'minutes_to_post': round(mtp, 2),
                    'n_captured': len(omap),
                    'n_expected': expected,
                    # 一度見えた馬が今回消えた = 取消・除外の可能性
                    'is_scratched': o is None,
                    'source': 'auto',
                })
            if dry_run:
                total_saved += len(rows)
            else:
                total_saved += save_odds_snapshots(rows, base_dir=base_dir)

        print(f'  [{tick.strftime("%H:%M:%S")}] 周回{rounds} '
              f'対象{len(due)}R 保存累計{total_saved} リクエスト{req}')
        time.sleep(interval_sec)

    print(f'\n■ 収集終了  周回{rounds} / スナップショット{total_saved}行 / '
          f'リクエスト{req}件')
    if unmapped:
        print(f'⚠ 収集できなかったレース {len(unmapped)}件: {sorted(unmapped)[:5]}')
    if total_saved == 0:
        print('🔴 1行も保存できていない。オッズ未発売か、会場名の対応が取れていない。'
              'このまま放置すると何週間も空振りし続けるので原因を確認すること。')
    return {'races': len(sched), 'snapshots': total_saved, 'requests': req}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default='.')
    ap.add_argument('--date', default=None, help='YYYYMMDD（省略時は当日）')
    ap.add_argument('--max-minutes', type=int, default=330)
    ap.add_argument('--interval-sec', type=int, default=INTERVAL_SEC_DEFAULT)
    ap.add_argument('--max-requests', type=int, default=MAX_REQUESTS_DEFAULT)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    collect(a.base_dir, a.date, a.max_minutes, a.interval_sec,
            a.max_requests, a.dry_run)


if __name__ == '__main__':
    main()
