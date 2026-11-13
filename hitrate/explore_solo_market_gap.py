"""AI単（市場を見ない評価）と市場の評価がずれるとき、何がそのずれを説明するか（記述統計）。

問い（2026-11-14 ユーザー）: 市場が正しいとしたとき、AI単に抜けている情報は何か。

やること
  1. 2024-08-31 までで2つのモデルを作る（本番と同じ設定・全木で予測）
       R = 残差学習（本番と同型）。AI単 = raw_margin - base_margin（アプリの「AI単」列と同じ定義）
       F = 市場を一切使わないモデル（base_margin なし）。「特徴量だけで作れる評価」
  2. 2025-01-01 以降（学習に使っていない）で、レース内の
       gap = z(市場) - z(AI単)   （＋＝市場のほうが高く評価）
     を作る。
  3. gap を説明できる量を OOF R^2（レース単位5分割）で測る
       A = モデルに既に入っている特徴量だけ
       B = DB にあるがモデルに入っていない変数（騎手の乗り替わり・騎手/厩舎の人気を集める度合い・
           通算の実績・前走の着差 など。すべてレース前に分かる値だけで作る）
       残り = DB に無い情報 ＋ 市場のノイズ
  4. 「市場が正しい」は本当か（3着内を z(市場) と z(AI単) で同時に説明して係数を見る）

これは記述統計で、採否の判定ではない。モデル・買い目・表示は何も変えない。
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

BASE = sys.argv[1] if len(sys.argv) > 1 else '.'
sys.path.insert(0, BASE)
from src.tools.train_xgb import _popularity_to_base_margin, _EXCLUDE_COLS

TRAIN_END, VAL_START, VAL_END, OOS_START = '2024-08-31', '2024-09-01', '2024-11-24', '2025-01-01'
SEED = 42
CLS = {'新馬': 0, '未勝利': 0, '1勝': 1, '2勝': 2, '3勝': 3, 'OP': 4, 'オープン': 4}
out = {}


def z_in_race(df, col):
    g = df.groupby('race_id')[col]
    return (df[col] - g.transform('mean')) / g.transform('std').replace(0, np.nan)


# ── 1. データ ────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'data', 'horse_features.csv'))
df['date'] = df['date'].astype(str).str[:10]
feat = [c for c in df.columns if c not in _EXCLUDE_COLS | {'f_popularity'}
        and df[c].dtype.kind in 'fi']
df['_n'] = df.groupby('race_id')['horse_num'].transform('count')
df = df[(df['f_popularity'] >= 1) & (df['f_popularity'] <= 18) & (df['_n'] >= 5)].copy()
df['bm'] = _popularity_to_base_margin(df['f_popularity'], df['_n'])

con = sqlite3.connect(os.path.join(BASE, 'data', 'history.db'))
hh = pd.read_sql('select race_id,date,horse_num,horse_name,place,popularity,jockey,trainer,'
                 'class_grade,time_diff_sec,agari_rank,weight_load from horse_history', con)
hh['date'] = hh['date'].astype(str).str[:10]
hh = hh.sort_values(['date', 'race_id', 'horse_num']).reset_index(drop=True)
hh['top3'] = hh['place'].between(1, 3).astype(float)
hh['win'] = (hh['place'] == 1).astype(float)
hh['pop'] = hh['popularity'].where(hh['popularity'].between(1, 18))
hh['cls'] = hh['class_grade'].map(CLS)


def prior_by_day(key, prefix):
    """key ごとに「その日より前」の 出走数・3着内率（収縮つき）・平均人気 を付ける。"""
    d = hh.assign(k=1.0, hp=hh['pop'].notna().astype(float), ps=hh['pop'].fillna(0))
    day = d.groupby(key + ['date'], as_index=False)[['k', 'top3', 'hp', 'ps']].sum()
    day = day.sort_values(key + ['date'])
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
                 ('agari_rank', 'x_last_agari_rank'), ('win', 'x_last_win'), ('cls', '_last_cls')]:
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
EXTRA = [c for c in hh.columns if c.startswith('x_')]
df = df.merge(hh[['race_id', 'horse_num'] + EXTRA + ['jockey']], on=['race_id', 'horse_num'], how='left')
for c in list(EXTRA):
    df['z' + c] = z_in_race(df, c)
EXTRA_ALL = EXTRA + ['z' + c for c in EXTRA]

# ── 2. モデル ────────────────────────────────────────────────────────
tr = df[df['date'] <= TRAIN_END]; va = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)]
oos = df[df['date'] >= OOS_START].copy()
pos = tr['is_fukusho'].mean()
P = dict(max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
         alpha=0.1, reg_lambda=1.0, scale_pos_weight=round((1 - pos) / pos, 2),
         objective='binary:logistic', eval_metric='logloss', seed=SEED, nthread=8)


def fit(residual):
    dt = xgb.DMatrix(tr[feat].fillna(5.0), label=tr['is_fukusho'])
    dv = xgb.DMatrix(va[feat].fillna(5.0), label=va['is_fukusho'])
    do = xgb.DMatrix(oos[feat].fillna(5.0))
    if residual:
        dt.set_base_margin(tr['bm'].values); dv.set_base_margin(va['bm'].values); do.set_base_margin(oos['bm'].values)
    b = xgb.train(P, dt, 500, evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=False)
    m = b.predict(do, output_margin=True)          # 既定＝全木（本番と同じ）
    return (m - oos['bm'].values) if residual else m, b.num_boosted_rounds()


oos['s_R'], nR = fit(True)
oos['s_F'], nF = fit(False)
oos['z_m'] = z_in_race(oos, 'bm')
for k in ('R', 'F'):
    oos['z_' + k] = z_in_race(oos, 's_' + k)
    oos['rank_' + k] = oos.groupby('race_id')['s_' + k].rank(ascending=False, method='first')
    oos['gap_' + k] = oos['z_m'] - oos['z_' + k]
    oos['rg_' + k] = oos['rank_' + k] - oos['f_popularity']      # ＋＝市場のほうが上位に見ている
oos = oos.dropna(subset=['z_m', 'z_R', 'z_F']).reset_index(drop=True)
out['n'] = dict(train=len(tr), oos_horses=len(oos), oos_races=int(oos['race_id'].nunique()),
                oos_from=oos['date'].min(), oos_to=oos['date'].max(), trees_R=nR, trees_F=nF, n_feat=len(feat))

from sklearn.metrics import roc_auc_score
y = oos['is_fukusho'].values
out['auc'] = {k: round(roc_auc_score(y, oos[c]), 4) for k, c in
              [('市場のみ', 'bm'), ('AI単R', 's_R'), ('AI単F', 's_F')]}
out['auc']['市場+R（本番同型）'] = round(roc_auc_score(y, oos['bm'] + oos['s_R']), 4)

# ── 3. ずれの大きさと「市場が正しい」の確認 ─────────────────────────
for k in ('R', 'F'):
    o = {}
    o['corr_z'] = round(float(np.corrcoef(oos['z_m'], oos['z_' + k])[0, 1]), 3)
    top = oos[oos['rank_' + k] == 1]
    o['AI単1位が1番人気'] = round(float((top['f_popularity'] == 1).mean()), 3)
    o['|順位差|>=4 の馬の割合'] = round(float((oos['rg_' + k].abs() >= 4).mean()), 3)
    lr = LogisticRegression(C=1e6).fit(oos[['z_m', 'z_' + k]], y)
    o['3着内ロジット係数'] = dict(z市場=round(float(lr.coef_[0][0]), 3), zAI単=round(float(lr.coef_[0][1]), 3))
    rows = []
    for lab, lo, hi in [('市場が4以上 上', 4, 99), ('市場が2〜3 上', 2, 3), ('ほぼ同じ', -1, 1),
                        ('AI単が2〜3 上', -3, -2), ('AI単が4以上 上', -99, -4)]:
        s = oos[oos['rg_' + k].between(lo, hi)]
        # 同じ人気の馬全体の3着内率（＝市場だけから期待される値）／同じAI単順位の馬全体の3着内率
        em = s['f_popularity'].map(oos.groupby('f_popularity')['is_fukusho'].mean()).mean()
        ea = s['rank_' + k].map(oos.groupby('rank_' + k)['is_fukusho'].mean()).mean()
        rows.append(dict(帯=lab, 頭数=len(s), 実測3着内=round(float(s['is_fukusho'].mean()), 4),
                         市場どおりなら=round(float(em), 4), AI単どおりなら=round(float(ea), 4)))
    o['帯別'] = rows
    out['gap_' + k] = o

# ── 4. ずれを説明できる量（OOF R^2） ────────────────────────────────
RP = dict(max_depth=6, eta=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
          objective='reg:squarederror', seed=SEED, nthread=8)


def oof_r2(cols, target, rounds=300):
    X = oos[cols]; t = oos[target].values; pred = np.zeros(len(oos))
    for a, b in GroupKFold(5).split(X, t, oos['race_id']):
        m = xgb.train(RP, xgb.DMatrix(X.iloc[a], label=t[a]), rounds)
        pred[b] = m.predict(xgb.DMatrix(X.iloc[b]))
    return round(1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum(), 4), pred


for k in ('R', 'F'):
    t = 'gap_' + k
    rA, _ = oof_r2(feat, t); rB, _ = oof_r2(EXTRA_ALL, t); rAB, pAB = oof_r2(feat + EXTRA_ALL, t)
    out['r2_' + k] = dict(既存特徴量だけ=rA, 追加候補だけ=rB, 両方=rAB,
                          追加候補の上乗せ=round(rAB - rA, 4), 説明できない残り=round(1 - rAB, 4))
    oos['resid_' + k] = oos[t] - pAB
    oos['expl_' + k] = pAB
    # 1変数ずつ: レース内 z と gap の順位相関（＋＝その値が高い馬ほど市場のほうが高評価）
    cand = {}
    for c in ['z' + e for e in EXTRA] + [f for f in feat if f.startswith(('rl_', 'cl_')) and f.endswith('_vs_field')] \
            + ['f_pop_last', 'f_pop_avg', 'f_career_runs', 'f_is_debut', 'f_days_since_last', 'f_age', 'f_class_jump',
               'f_jockey_rate', 'f_pl_rating', 'f_last1_rank', 'f_beat_market_rate', 'f_speed_fig_max']:
        v = oos[[c, t]].dropna()
        if len(v) > 5000:
            cand[c] = round(float(v[c].rank().corr(v[t].rank())), 3)
    out['spearman_' + k] = dict(sorted(cand.items(), key=lambda kv: -abs(kv[1]))[:22])

# ── 5. 具体的な場面での平均ずれ（R＝アプリのAI単。順位差の平均。＋＝市場が上） ─────────
def cell(mask, k='R'):
    s = oos[mask]
    return dict(頭数=int(len(s)), 平均順位差=round(float(s['rg_' + k].mean()), 2),
                市場が4以上上の割合=round(float((s['rg_' + k] >= 4).mean()), 3),
                AI単が4以上上の割合=round(float((s['rg_' + k] <= -4).mean()), 3))


jk_hi = oos['x_jk_pop'] <= oos['x_jk_pop'].quantile(0.10)       # 人気を集める騎手（上位1割）
jk_lo = oos['x_jk_pop'] >= oos['x_jk_pop'].quantile(0.90)
tr_hi = oos['x_tr_pop'] <= oos['x_tr_pop'].quantile(0.10)
sc = {}
for k in ('R', 'F'):
    sc[k] = {
        '全体': cell(oos['rg_R'].notna(), k),
        '初出走（過去走なし）': cell(oos['x_n_prior'] == 0, k),
        '過去1〜2走': cell(oos['x_n_prior'].between(1, 2), k),
        '過去3走以上': cell(oos['x_n_prior'] >= 3, k),
        '人気を集める騎手（上位1割）': cell(jk_hi, k),
        '人気を集めない騎手（下位1割）': cell(jk_lo, k),
        '人気騎手へ乗り替わり': cell(jk_hi & (oos['x_jk_same'] == 0), k),
        '人気騎手から他へ乗り替わり': cell((~jk_hi) & (oos['x_jk_same'] == 0) & (oos['x_jk_rate_chg'] < -0.08), k),
        '人気を集める厩舎（上位1割）': cell(tr_hi, k),
        '前走1着（昇級初戦が多い）': cell(oos['x_last_win'] == 1, k),
        '前走が0.3秒差以内の負け': cell((oos['x_last_win'] == 0) & (oos['x_last_td'] <= 0.3), k),
        '前走が2秒以上の大敗': cell(oos['x_last_td'] >= 2.0, k),
        '過去に1番人気になったことがある': cell(oos['x_best_pop'] == 1, k),
        '過去に5番人気以内が一度も無い': cell(oos['x_best_pop'] > 5, k),
        '格上で3着内の実績あり（今回は下のクラス）': cell(oos['x_cls_vs_best'] < 0, k),
        '2歳': cell(oos['f_age'] == 2, k),
    }
out['scenes'] = sc
# 大きなずれ（|順位差|>=4）のうち、過去走が2走以下の馬が占める割合
big = oos['rg_R'].abs() >= 4
out['big_gap_share'] = dict(全馬に占める過去2走以下=round(float((oos['x_n_prior'] <= 2).mean()), 3),
                            大きなずれに占める過去2走以下=round(float((oos.loc[big, 'x_n_prior'] <= 2).mean()), 3))
# 説明できなかった残りは結果を当てるか（市場を固定して残差を足す）
for k in ('R', 'F'):
    lr = LogisticRegression(C=1e6).fit(oos[['z_m', 'resid_' + k]], y)
    out['resid_coef_' + k] = round(float(lr.coef_[0][1]), 3)
    # AI単を固定して「説明できたずれ」「説明できなかったずれ」を足す。＋なら市場側の情報が結果を当てている
    lr = LogisticRegression(C=1e6).fit(oos[['z_' + k, 'expl_' + k, 'resid_' + k]], y)
    out['given_ai_' + k] = dict(zAI単=round(float(lr.coef_[0][0]), 3), 説明できたずれ=round(float(lr.coef_[0][1]), 3),
                                説明できなかったずれ=round(float(lr.coef_[0][2]), 3))
    out['auc_steps_' + k] = dict(
        AI単だけ=round(roc_auc_score(y, oos['z_' + k]), 4),
        AI単と説明できたずれ=round(roc_auc_score(y, oos['z_' + k] + oos['expl_' + k]), 4),
        市場=round(roc_auc_score(y, oos['z_m']), 4))

json.dump(out, open(os.path.join(BASE, 'hitrate', 'solo_market_gap_result.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
