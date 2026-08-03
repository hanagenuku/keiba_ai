#!/usr/bin/env python3
"""「RL順位と市場人気の差」の大規模検証ノートブックを生成する。

## 何を検証するか

本番の実記録（400レース）で見えた
    4-6番人気 × 差(人気−RL)+3以上 → 単勝回収率 165.7% (N=82, 13的中)
が本物かどうかを、約2,000レース規模で確かめる。

⚠ この165.7%は **3つの異なるモデル**（v4/v5/v6）が出した予想の寄せ集めで、
  大半は今はもう使っていないモデルによるもの。現行モデルで測り直す必要がある。

## 事前登録した採用基準（見てから変えない）

  ① 単勝回収率 ≥ 105%
  ② N ≥ 300
  ③ 期間を前後半に分けても両方100%超
  ④ 開催日ブロックブートストラップの95%CI下限 > 100%
  ⑤ 対照群（同人気帯・差が小さい馬）と信頼区間が重ならない

1つでも欠けたら「見つからなかった」と結論する。

## 併せて検証する仮説（本番82頭で示唆されたもの・N=18なので未確定）

  差の主因が「スピード能力」なら買い / 「過去人気推移」なら切る

## 既存ノートとの違い

KEIBA_買い目フィルタ検証_v1.ipynb は人気×RL帯のマトリクスを作るものだった。
本ノートは「差」そのものと、その**中身（SHAP分解）**に焦点を当てる。
学習の枠組み（2025年末までで学習→2026年で検証）は同じものを流用する。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(ROOT, 'KEIBA_買い目フィルタ検証_v1.ipynb')
OUT_NB = os.path.join(ROOT, 'KEIBA_RL市場差_検証_v1.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.splitlines(True)}


HEADER = '''# RL順位と市場人気の「差」の大規模検証

本番の実記録（400レース）で見えた

> **4-6番人気 × 差(人気−RL)が+3以上 → 単勝回収率 165.7%（N=82・13的中・11開催日中7日プラス）**

が本物かどうかを、約2,000レース規模で確かめる。

⚠ **165.7%は3つの異なるモデル（v4/v5/v6）の寄せ集め**で、大半は今は使っていない
モデルが出した順位。現行モデルで測り直さないと採用判断はできない。

## 事前登録した採用基準（結果を見てから変えないこと）

| # | 基準 |
|---|---|
| ① | 単勝回収率 ≥ 105% |
| ② | N ≥ 300 |
| ③ | 期間を前後半に分けても両方100%超 |
| ④ | 開催日ブロックブートストラップの95%CI下限 > 100% |
| ⑤ | 対照群と信頼区間が重ならない |

**1つでも欠けたら「見つからなかった」と結論する。**

## 併せて検証する仮説

本番82頭の内訳では、差の主因別に
スピード能力 302.8%(N=18) / 騎手 186.9%(N=16) / **過去人気推移 10.8%(N=13)**
と分かれた。ただしNが小さく未確定。ここで検証する。

## 実行方法

セル1から順に実行するだけ。セル3の学習に10〜20分かかる。
'''

CELL_GAP = '''# == 差(gap)の算出 + SHAP分解 =============================================
# rl_rank はレース内の raw_margin 降順。
# ⚠ 本番の rl_rank は total（相対ブレンド・ペースボーナス・エラータグ補正込み）
#    の順位だが、相対ブレンドはレース内の一次変換なので順位を変えない。
#    順位を動かすのはペースボーナスとエラータグ補正だけで、影響は小さい。
#    ここでは raw_margin 順位で近似する（この近似の限界は最後に報告する）。
import numpy as np, pandas as pd
from src.features.shap_explain import FEATURE_CATEGORY_MAP

t0 = time.time()
m['gap'] = m['popularity'] - m['rl_rank']          # +ならAIが市場より強気

# ── 差をカテゴリに分解 ────────────────────────────────────────────
contribs = booster.predict(dev, pred_contribs=True)      # (n, 特徴量数+1)
cat_of = {c: FEATURE_CATEGORY_MAP.get(c, 'その他') for c in feat_cols}
cats = sorted(set(cat_of.values()))
tmp = pd.DataFrame(0.0, index=eval_df.index, columns=cats)
for i, c in enumerate(feat_cols):
    tmp[cat_of[c]] += contribs[:, i]
tmp['race_id'] = eval_df['race_id'].values
tmp['horse_num'] = pd.to_numeric(eval_df['horse_num'], errors='coerce').values
m = m.merge(tmp, on=['race_id', 'horse_num'], how='left')

m['主因'] = m[cats].idxmax(axis=1)
print(f'差の分解 完了  {len(m):,}頭 / {m.race_id.nunique():,}レース  '
      f'({time.time()-t0:.0f}秒)')
print(f'\\n差の分布: 中央値 {m.gap.median():.0f} / '
      f'+3以上 {(m.gap>=3).sum():,}頭 / 4-6番人気かつ+3以上 '
      f'{((m.popularity.between(4,6))&(m.gap>=3)).sum():,}頭')

# ── ★ 先に「基準②(N>=300)に届く見込みがあるか」を確認する ──────────
# 本番の実記録では 400レースで82頭（1レースあたり0.21頭）出ていた。
# ここでの出現率がそれより大幅に低い場合、レース数を増やしてもNは足りない。
_n_tgt = int(((m.popularity.between(4, 6)) & (m.gap >= 3)).sum())
_rate = _n_tgt / max(m.race_id.nunique(), 1)
print(f'\\n1レースあたりの該当頭数: {_rate:.3f}頭  （本番実記録では 0.21頭）')
if _n_tgt < 300:
    print(f'⚠ 該当が {_n_tgt}頭しかなく、事前登録した基準②(N>=300)には届かない。')
    print('  → この時点で「採用しない」はほぼ確定。ただし基準を後から緩めないこと。')
    print('  　 セル6以降は「向きだけでも見ておく」ための参考として実行する。')
'''

CELL_MAIN = '''# == ★ 本命の検証: 4-6番人気 × 差+3以上 ==================================
import numpy as np
rng = np.random.default_rng(0)
m['day'] = m['race_id'].str[:8]      # ⚠ date列は 'YYYY-MM-DD' なので race_id から取る

def blockboot(s, n=4000):
    """開催日ブロックブートストラップ。同日レースは条件が相関するため必須。

    開催日が3日以下だとブロックが足りず区間推定が成立しない
    （2026-07-27に59点を独立サンプル扱いして過信した失敗の再発防止）。
    """
    days = s.day.unique()
    if len(s) == 0 or len(days) < 4:
        return float('nan'), float('nan')
    by = {k: v.payout.values for k, v in s.groupby('day')}
    out = [np.concatenate([by[k] for k in rng.choice(days, len(days), replace=True)]).mean()
           for _ in range(n)]
    return np.percentile(out, [2.5, 97.5])


def _gt(a, b):
    """NaN（区間推定できず）は「基準を満たした」と誤判定しないようFalseにする。"""
    return bool(a == a and b == b and a > b)

tgt = m[(m.popularity.between(4, 6)) & (m.gap >= 3)]
ctl = m[(m.popularity.between(4, 6)) & (m.gap < 3)]

print('=' * 64)
print(f'対象: 4-6番人気 × 差+3以上')
print(f'  N={len(tgt):,}  勝率{100*tgt.won.mean():.1f}%  回収率{tgt.payout.mean():.1f}%  '
      f'開催日{tgt.day.nunique()}日  的中{int(tgt.won.sum())}本')
print(f'対照群: 4-6番人気 × 差+2以下')
print(f'  N={len(ctl):,}  勝率{100*ctl.won.mean():.1f}%  回収率{ctl.payout.mean():.1f}%')
print('=' * 64)

res = {}
res['①回収率>=105%'] = (tgt.payout.mean() >= 105, f'{tgt.payout.mean():.1f}%')
res['②N>=300'] = (len(tgt) >= 300, f'{len(tgt)}')

days = sorted(tgt.day.unique()); mid = days[len(days)//2]
a, b = tgt[tgt.day <= mid], tgt[tgt.day > mid]
res['③前後半とも100%超'] = (a.payout.mean() > 100 and b.payout.mean() > 100,
                       f'前半{a.payout.mean():.1f}% / 後半{b.payout.mean():.1f}%')

lo, hi = blockboot(tgt)
res['④CI下限>100%'] = (_gt(lo, 100), f'95%CI [{lo:.1f}, {hi:.1f}]')
clo, chi = blockboot(ctl)
res['⑤対照群とCI非重複'] = (_gt(lo, chi), f'対照群CI [{clo:.1f}, {chi:.1f}]')

print('\\n【事前登録した採用基準の判定】')
for k, (ok, detail) in res.items():
    print(f'  {"✅" if ok else "❌"} {k:20s} {detail}')
verdict = all(v[0] for v in res.values())
print('\\n' + ('★ 全基準クリア → 採用を検討してよい' if verdict
              else '→ 基準未達。採用しない（「見つからなかった」が結論）'))

print('\\n【開催日ごとの成績】（1〜2日に偏っていないか）')
g = tgt.groupby('day').agg(N=('payout','size'), 的中=('won','sum'), 回収率=('payout','mean'))
print(g.round(1).to_string())
print(f'  プラスだった日: {(g.回収率>100).sum()}/{len(g)}日')

w = tgt[tgt.won].sort_values('payout', ascending=False)
if len(w) >= 3:
    print(f'\\n【高配当依存の確認】最高配当を除くと '
          f'{tgt[~tgt.index.isin(w.index[:1])].payout.mean():.1f}% / '
          f'上位3本を除くと {tgt[~tgt.index.isin(w.index[:3])].payout.mean():.1f}%')
'''

CELL_CAT = '''# == 差の「中身」で絞り込めるか ===========================================
# 本番82頭では スピード能力302.8% / 過去人気推移10.8% と分かれた（N小・未確定）。
print('【差の主因ごとの成績】4-6番人気 × 差+3以上 の中で\\n')
t = tgt.groupby('主因').agg(N=('payout','size'), 的中=('won','sum'),
                          勝率=('won','mean'), 回収率=('payout','mean'))
t['勝率'] = (t['勝率']*100).round(1); t['回収率'] = t['回収率'].round(1)
t['的中'] = t['的中'].astype(int)
_shown = t[t.N >= 30].sort_values('回収率', ascending=False)
print(_shown.to_string() if len(_shown)
      else f'  主因別に30頭以上あるグループなし（対象が{len(tgt)}頭しかない）')

print('\\n\\n【的中馬 vs 不的中馬 のカテゴリ寄与】')
if tgt.won.sum() >= 5:
    cmp = pd.DataFrame({'的中馬': tgt[tgt.won][cats].mean(),
                        '不的中馬': tgt[~tgt.won][cats].mean()})
    cmp['差'] = cmp['的中馬'] - cmp['不的中馬']
    print(cmp.sort_values('差', ascending=False).round(4).to_string())
else:
    print(f'  的中が{int(tgt.won.sum())}本しかなく比較できない')

print('\\n\\n【仮説の検証】主因で絞ると基準を満たすか')
_cands = t[t.N >= 100].index
for name in _cands:
    s = tgt[tgt.主因 == name]
    lo2, hi2 = blockboot(s)
    ok = (s.payout.mean() >= 105 and len(s) >= 300 and _gt(lo2, 100))
    print(f'  {"✅" if ok else "❌"} 主因={name:10s} N={len(s):4d} '
          f'回収率{s.payout.mean():6.1f}%  95%CI [{lo2:6.1f}, {hi2:6.1f}]')
if not len(_cands):
    print('  100頭以上ある主因なし（絞り込みの検証はできない）')
print('\\n⚠ 主因で切ると必ずNが減る。②N>=300 を満たせるかが分かれ目。')
'''

CELL_ROBUST = '''# == 頑健性: 差の閾値・人気帯を変えても成り立つか =========================
print('【差の閾値を変える】4-6番人気\\n')
for th in (1, 2, 3, 4, 5):
    s = m[(m.popularity.between(4, 6)) & (m.gap >= th)]
    if len(s) < 30: continue
    lo3, hi3 = blockboot(s)
    print(f'  差>=+{th}: N={len(s):5,d} 勝率{100*s.won.mean():5.1f}% '
          f'回収率{s.payout.mean():6.1f}%  CI [{lo3:6.1f}, {hi3:6.1f}]')

print('\\n【人気帯を変える】差+3以上\\n')
for lo_p, hi_p, lab in [(1,3,'1-3番人気'), (4,6,'4-6番人気'),
                        (7,9,'7-9番人気'), (10,18,'10番人気以上')]:
    s = m[(m.popularity.between(lo_p, hi_p)) & (m.gap >= 3)]
    if len(s) < 30: continue
    lo4, hi4 = blockboot(s)
    print(f'  {lab:12s}: N={len(s):5,d} 勝率{100*s.won.mean():5.1f}% '
          f'回収率{s.payout.mean():6.1f}%  CI [{lo4:6.1f}, {hi4:6.1f}]')

print('\\n\\n【この検証の限界】')
print('  ・人気は history.db の**確定人気**。本番は朝の人気を使うため条件が違う')
print('    （実測で平均1.5位ずれ、1番人気が入れ替わるレースが34.9%ある）')
print('  ・rl_rank は raw_margin 順位で近似（ペースボーナス・エラータグ補正を除く）')
print('  ・検証期のモデルは2025年末までで学習したもの。本番モデルとは別物')
print('  → 「効果の向き」は信頼できるが、「回収率の絶対値」はそのまま本番に移らない')
'''


def main():
    src = json.load(open(SRC_NB, encoding='utf-8'))
    # セル0-8（セットアップ〜rl_rank算出）までを流用し、以降を差し替える
    cells = [md(HEADER)] + src['cells'][1:9] + [
        md('## セル5: 差(gap)の算出と SHAP 分解'), code(CELL_GAP),
        md('## セル6: ★ 本命の検証（事前登録した基準で判定）'), code(CELL_MAIN),
        md('## セル7: 差の「中身」で絞り込めるか'), code(CELL_CAT),
        md('## セル8: 頑健性チェックと限界'), code(CELL_ROBUST),
    ]
    nb = {'cells': cells, 'metadata': src.get('metadata', {}),
          'nbformat': 4, 'nbformat_minor': 0}
    json.dump(nb, open(OUT_NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'生成: {OUT_NB}  ({len(cells)}セル)')


if __name__ == '__main__':
    main()
