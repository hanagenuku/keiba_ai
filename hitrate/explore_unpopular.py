#!/usr/bin/env python3
"""【探索的・判定しない】不人気馬が複勝圏（3着内）に入った形の記述統計。

🔴 **これは事前登録カードのある検証ではない。** 仮説を探すための記述統計であり、
   ここに出た数字を「根拠」として買い目・印・モデルに入れてはいけない。
   使うなら `CRITERIA*.md` を書いて別に測る（検証の決まった手順）。

🔴 **出力は2つに厳密に分ける。**
   A: レース後にしか分からない量（実際のペース・実際の脚質・通過順・上がり順位）
      → 記述はできるが**予想には使えない**（North Star #8）
   B: レース開始前に分かる量（過去走からの脚質傾向・枠・距離・馬場・クラス・斤量・間隔）
      → 予想に使える候補。ただし既に否定された案が多い（§撤回表）

使い方:
    cd <隔離ワークスペース>
    python3 hitrate/explore_unpopular.py
"""
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)

POP_MIN = 6          # 「不人気」の定義（確定人気6位以下）
POP_MIN2 = 10        # 副（10位以下）
MIN_N = 200          # これ未満のセルは表示しない（小さいNの好成績を信じない）


def load(db):
    con = sqlite3.connect(db)
    h = pd.read_sql_query(
        """SELECT race_id, date, racecourse, distance, surface, place, horse_num,
                  horse_name, running_style, agari3f, agari_rank, popularity,
                  fukusho_payout, class_grade, weight_load, sex, age,
                  body_weight_diff, corner_all, margin, time_diff_sec
           FROM horse_history WHERE place IS NOT NULL AND place > 0""", con)
    r = pd.read_sql_query(
        """SELECT race_id, first_3f, last_3f, track_condition, weather,
                  num_finishers, race_class
           FROM race_history""", con)
    con.close()
    df = h.merge(r, on='race_id', how='left', suffixes=('', '_r'))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'popularity']).sort_values(['date', 'race_id'])
    df['top3'] = (df['place'] <= 3).astype(int)
    df['win'] = (df['place'] == 1).astype(int)
    # 複勝回収率（1点100円）。払戻が無い的中行は評価不能として落とす
    df['_fuku_ok'] = ~((df['top3'] == 1) & (df['fukusho_payout'].fillna(0) <= 0))
    return df


def rate_table(df, by, label, min_n=MIN_N, extra=None):
    g = df[df['_fuku_ok']].groupby(by)
    out = pd.DataFrame({
        'N': g.size(),
        '3着内率%': g['top3'].mean() * 100,
        '勝率%': g['win'].mean() * 100,
        '複勝回収%': g.apply(lambda x: np.where(x['top3'] == 1,
                                               x['fukusho_payout'].fillna(0), 0).sum()
                             / (len(x) * 100) * 100),
    })
    out = out[out['N'] >= min_n].sort_values('3着内率%', ascending=False)
    base = df[df['_fuku_ok']]['top3'].mean() * 100
    out['基準比pt'] = out['3着内率%'] - base
    print(f'\n── {label}（母集団の3着内率 {base:.2f}%・N>={min_n} のセルだけ）')
    print(out.round(2).to_string())
    return out


def pace_bucket(df):
    """前半3F を (馬場, 距離) 内で標準化して Slow / Mid / Fast に3分割する。

    🔴 これは**レース後の情報**。予想には使えない。
    `pace_label` は充足0%（一度も取れていない）ので first_3f から作る。
    """
    d = df.dropna(subset=['first_3f']).copy()
    key = d['surface'].astype(str) + '_' + d['distance'].astype(str)
    z = d.groupby(key)['first_3f'].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    # 前半が遅い＝スロー
    d['pace3'] = pd.cut(z, [-np.inf, -0.43, 0.43, np.inf],
                        labels=['Slow(前半遅)', 'Mid', 'Fast(前半速)'])
    return d


def past_style(df):
    """過去走の running_style の最頻値（直近5走）＝**レース前に分かる**脚質傾向。

    `engine.py::_estimate_style` と同じ考え方（直近5走の最頻値）。
    🔴 `horse_history.running_style` はそのレースの実際の脚質なので、
       必ず「その馬のそれ以前の走り」だけから作る（リーク防止）。
    """
    d = df.sort_values(['horse_name', 'date']).copy()
    styles = []
    hist = {}
    for hn, st in zip(d['horse_name'].values, d['running_style'].values):
        past = hist.get(hn, [])
        if len(past) >= 2:
            s = pd.Series(past[-5:])
            styles.append(s.mode().iloc[0])
        else:
            styles.append(None)
        past.append(st)
        hist[hn] = past
    d['past_style'] = styles
    return d


def main():
    base = os.getcwd()
    df = load(os.path.join(base, 'data', 'history.db'))
    print('=' * 100)
    print('【探索的・判定しない】不人気馬が複勝圏に入った形の記述統計')
    print('=' * 100)
    print(f'全体: {len(df):,}行 / {df.race_id.nunique():,}レース / '
          f'{df.date.min().date()} 〜 {df.date.max().date()}')

    for pm in (POP_MIN, POP_MIN2):
        u = df[df['popularity'] >= pm]
        ok = u[u['_fuku_ok']]
        print(f'\n人気{pm}位以下: {len(u):,}頭  3着内 {ok["top3"].mean()*100:.2f}%  '
              f'勝率 {ok["win"].mean()*100:.2f}%  '
              f'複勝回収 '
              f'{np.where(ok["top3"]==1, ok["fukusho_payout"].fillna(0), 0).sum()/(len(ok)*100)*100:.2f}%')

    u = df[df['popularity'] >= POP_MIN].copy()

    print('\n' + '=' * 100)
    print('■ A: レース後にしか分からない量（🔴 記述のみ。予想には使えない）')
    print('=' * 100)

    rate_table(u, 'running_style', 'A-1 そのレースの実際の脚質 × 不人気馬の3着内率')

    up = pace_bucket(u)
    print(f'\n（前半3F があるのは {len(up):,}/{len(u):,}頭 = '
          f'{len(up)/len(u)*100:.1f}%。first_3f 充足 47.6%）')
    rate_table(up, 'pace3', 'A-2 ペース（前半3Fの標準化）× 不人気馬の3着内率')
    rate_table(up, ['pace3', 'running_style'],
               'A-3 ペース × 実際の脚質（ユーザー仮説「スローペース逃げ切り」の検算）',
               min_n=100)

    u['agari_band'] = pd.cut(u['agari_rank'], [0, 1, 3, 5, 99],
                             labels=['上がり1位', '2〜3位', '4〜5位', '6位以下'])
    rate_table(u, 'agari_band', 'A-4 上がり3F順位 × 不人気馬の3着内率')

    # 通過順（corner_all の最後＝4角）を頭数で正規化
    def last_corner(s):
        try:
            xs = [int(x) for x in str(s).replace('*', '').split('-') if x.strip().isdigit()]
            return xs[-1] if xs else np.nan
        except Exception:
            return np.nan
    u['c4'] = u['corner_all'].map(last_corner)
    u['c4_rel'] = u['c4'] / u['num_finishers'].replace(0, np.nan)
    u['c4_band'] = pd.cut(u['c4_rel'], [0, .2, .4, .6, .8, 1.0],
                          labels=['先頭2割', '2〜4割', '4〜6割', '6〜8割', '後方2割'])
    rate_table(u, 'c4_band', 'A-5 4角の位置（頭数で正規化）× 不人気馬の3着内率')

    print('\n' + '=' * 100)
    print('■ B: レース開始前に分かる量（予想に使える候補）')
    print('=' * 100)

    ups = past_style(u)
    ups = ups.dropna(subset=['past_style'])
    print(f'\n（過去2走以上ある馬: {len(ups):,}/{len(u):,}頭 = {len(ups)/len(u)*100:.1f}%）')
    rate_table(ups, 'past_style',
               'B-1 過去5走の脚質最頻値（🔑 A-1 の「レース前に分かる版」）')

    upsp = pace_bucket(ups)
    rate_table(upsp, ['pace3', 'past_style'],
               'B-2 ペース（レース後） × 過去の脚質傾向（レース前）'
               '— 🔴 ペース側がレース後なのでこの表も予想には使えない', min_n=100)

    u['wk'] = pd.cut(u['horse_num'] / u['num_finishers'].replace(0, np.nan),
                     [0, .25, .5, .75, 1.0], labels=['内1/4', '内寄り', '外寄り', '外1/4'])
    rate_table(u, 'wk', 'B-3 馬番の相対位置 × 不人気馬の3着内率')
    rate_table(u, 'track_condition', 'B-4 馬場状態')
    rate_table(u, 'surface', 'B-5 芝/ダート')
    u['dist_band'] = pd.cut(u['distance'], [0, 1300, 1700, 2100, 9999],
                            labels=['〜1300', '1400〜1700', '1800〜2100', '2200〜'])
    rate_table(u, 'dist_band', 'B-6 距離帯')
    rate_table(u, 'class_grade', 'B-7 クラス', min_n=300)
    rate_table(u, 'racecourse', 'B-8 競馬場', min_n=300)

    # 斤量の相対（レース内偏差）
    u['wl_rel'] = u['weight_load'] - u.groupby('race_id')['weight_load'].transform('mean')
    u['wl_band'] = pd.cut(u['wl_rel'], [-99, -2, -0.5, 0.5, 2, 99],
                          labels=['−2kg超軽', '軽め', '平均', '重め', '+2kg超重'])
    rate_table(u, 'wl_band', 'B-9 斤量のレース内相対')

    # 間隔（前走からの日数）
    d2 = u.sort_values(['horse_name', 'date'])
    d2['prev'] = d2.groupby('horse_name')['date'].shift(1)
    d2['gap'] = (d2['date'] - d2['prev']).dt.days
    d2['gap_band'] = pd.cut(d2['gap'], [0, 14, 28, 56, 112, 9999],
                            labels=['中1〜2週', '3〜4週', '5〜8週', '9〜16週', '17週〜'])
    rate_table(d2, 'gap_band', 'B-10 前走からの間隔')

    # 馬体重増減（当日発表＝レース前に分かる）
    u['bw_band'] = pd.cut(u['body_weight_diff'], [-99, -8, -2, 2, 8, 99],
                          labels=['−8kg超', '−3〜−8', '±2', '+3〜+8', '+8kg超'])
    rate_table(u, 'bw_band', 'B-11 馬体重増減（充足 56.1%）')

    rate_table(u, ['past_style'] if False else 'sex', 'B-12 性別')

    print('\n' + '=' * 100)
    print('■ 🔴 読むときの注意（これを飛ばして使わない）')
    print('=' * 100)
    print("""  1. これは**記述統計**で、事前登録も対照群も無い。セルを選べば何でも出る。
  2. A の表（実際の脚質・ペース・通過順・上がり順位）は**レース後の情報**。
     「スローペースで逃げた不人気馬は来やすい」は事実として言えても、
     レース前に「この馬が逃げる」「今日はスローになる」は分からない。
     - `pace_label` は充足 **0%**（一度も取れていない）
     - `first_3f` は **47.6%** しか無く、しかもレース後
     - 展開シミュレーションは測って否定済み（2026-08-17: 再現したのは f_post 1軸だけ）
     - 前半3F予測は目的変数が壊れていて取り消し（2026-09-14）
  3. B の表で「不人気馬の中から選ぶ」形は **2,318レースで否定済み**（2026-09-01）。
     「先行×実力上位×4〜9.9倍の単勝」も否定（2026-08-24）。
     「4〜9.9倍の1着馬に共通の特徴」は246セルで100%超が帰無並み（2026-08-24）。
  4. 複勝回収率を併記したのは、3着内率が上がっても回収率は別物だから（North Star #9）。
     控除率があるので 100% 近い数字が出ても「買える」ことにはならない。
  5. ここから何か作るなら `CRITERIA*.md` を書いて**別に1回だけ**測る。""")


if __name__ == '__main__':
    main()
