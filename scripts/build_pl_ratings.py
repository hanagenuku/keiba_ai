#!/usr/bin/env python3
"""history.db から Plackett-Luce レーティングを構築して data/pl_rating.pkl に保存する。

週次（日曜結果取得後）に実行する。推論時は init_engine がこのファイルを読む。
学習データ生成（build_training_data）はこのファイルを使わず、日付順に自前で
再現するため、両者は必ず同じ手順・同じ値になる（pl_rating.py 冒頭を参照）。

    python scripts/build_pl_ratings.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from src.features.pl_rating import build_from_history, save_ratings

    db = os.path.join(ROOT, 'data', 'history.db')
    if not os.path.exists(db):
        print(f'history.db が見つかりません: {db}')
        return 1

    print('Plackett-Luce レーティングを構築中...')
    ratings, n_races = build_from_history(
        db, progress=lambda n: print(f'  {n:,}レース...', flush=True))
    path = save_ratings(ratings, ROOT)

    th = list(ratings.theta.values())
    print(f'\n✅ 保存: {path}')
    print(f'   {len(th):,}頭 / {n_races:,}レース / 最終更新 {ratings.last_date}')
    print(f'   θ 範囲 {min(th):+.3f} 〜 {max(th):+.3f}')
    top = sorted(ratings.theta.items(), key=lambda kv: -kv[1])[:5]
    print('   上位: ' + ' / '.join(f'{n}({v:+.2f})' for n, v in top))
    return 0


if __name__ == '__main__':
    sys.exit(main())
