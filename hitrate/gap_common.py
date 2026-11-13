"""AI単と市場のずれの測定で共通に使う部品（レース前に分かる値だけで作る追加変数）。"""
import os
import sqlite3

import numpy as np
import pandas as pd

CLS = {'新馬': 0, '未勝利': 0, '1勝': 1, '2勝': 2, '3勝': 3, 'OP': 4, 'オープン': 4}


def z_in_race(df, col):
    g = df.groupby('race_id')[col]
    return (df[col] - g.transform('mean')) / g.transform('std').replace(0, np.nan)


def history_extras(base):
    """horse_history から (race_id, horse_num) ごとの追加変数 x_* を作る。すべて「その日より前」の値。"""
    con = sqlite3.connect(os.path.join(base, 'data', 'history.db'))
    hh = pd.read_sql('select race_id,date,horse_num,horse_name,place,popularity,jockey,trainer,'
                     'class_grade,time_diff_sec,agari_rank,weight_load from horse_history', con)
    hh['date'] = hh['date'].astype(str).str[:10]
    hh = hh.sort_values(['date', 'race_id', 'horse_num']).reset_index(drop=True)
    hh['top3'] = hh['place'].between(1, 3).astype(float)
    hh['win'] = (hh['place'] == 1).astype(float)
    hh['pop'] = hh['popularity'].where(hh['popularity'].between(1, 18))
    hh['cls'] = hh['class_grade'].map(CLS)

    def prior_by_day(key, prefix):
        d = hh.assign(k=1.0, hp=hh['pop'].notna().astype(float), ps=hh['pop'].fillna(0))
        day = d.groupby(key + ['date'], as_index=False)[['k', 'top3', 'hp', 'ps']].sum().sort_values(key + ['date'])
        cs = day.groupby(key)[['k', 'top3', 'hp', 'ps']].cumsum() - day[['k', 'top3', 'hp', 'ps']]
        day[prefix + '_n'] = cs['k']
        day[prefix + '_rate'] = (cs['top3'] + 20 * 0.21) / (cs['k'] + 20)
        day[prefix + '_pop'] = (cs['ps'] + 20 * 7.5) / (cs['hp'] + 20)
        return day[key + ['date', prefix + '_n', prefix + '_rate', prefix + '_pop']]

    hh = hh.merge(prior_by_day(['jockey'], 'x_jk'), on=['jockey', 'date'], how='left')
    hh = hh.merge(prior_by_day(['trainer'], 'x_tr'), on=['trainer', 'date'], how='left')
    hh = hh.merge(prior_by_day(['jockey', 'trainer'], 'x_jt'), on=['jockey', 'trainer', 'date'], how='left')
    g = hh.groupby('horse_name', sort=False)
    hh['x_n_prior'] = g.cumcount()
    for src, dst in [('jockey', '_last_jk'), ('x_jk_rate', '_last_jk_rate'), ('time_diff_sec', 'x_last_td'),
                     ('agari_rank', 'x_last_agari_rank'), ('win', 'x_last_win')]:
        hh[dst] = g[src].shift(1)
    hh['x_jk_same'] = (hh['jockey'] == hh['_last_jk']).astype(float).where(hh['_last_jk'].notna())
    hh['x_jk_rate_chg'] = hh['x_jk_rate'] - hh['_last_jk_rate']
    hh['x_career_top3'] = (g['top3'].cumsum() - hh['top3']) / hh['x_n_prior'].replace(0, np.nan)
    hh['x_career_wins'] = g['win'].cumsum() - hh['win']
    hh['x_best_pop'] = g['pop'].transform(lambda s: s.shift(1).expanding().min())
    hh['x_career_pop'] = g['pop'].transform(lambda s: s.shift(1).expanding().mean())
    hh['_c3'] = hh['cls'].where(hh['top3'] == 1)
    hh['x_best_cls_top3'] = g['_c3'].transform(lambda s: s.shift(1).expanding().max())
    hh['x_cls_vs_best'] = hh['cls'] - hh['x_best_cls_top3']
    hh['x_weight_load'] = hh['weight_load']
    extra = [c for c in hh.columns if c.startswith('x_')]
    return hh[['race_id', 'horse_num'] + extra], extra


def load_frame(base):
    """学習データを読み、人気が有効な5頭以上のレースに絞って base_margin を付ける。"""
    from src.tools.train_xgb import _popularity_to_base_margin
    df = pd.read_csv(os.path.join(base, 'data', 'horse_features.csv'))
    df['date'] = df['date'].astype(str).str[:10]
    df['_n'] = df.groupby('race_id')['horse_num'].transform('count')
    df = df[(df['f_popularity'] >= 1) & (df['f_popularity'] <= 18) & (df['_n'] >= 5)].copy()
    df['bm'] = _popularity_to_base_margin(df['f_popularity'], df['_n'])
    return df.sort_values(['race_id', 'horse_num']).reset_index(drop=True)
