# keiba_ai 設計指針書（ロードマップ）

> **このファイルは「ブレない指針」です。**  
> 毎セッション開始時に必ず参照し、実装の方向性を確認すること。  
> 最終更新: 2026-05-25

> **この設計図は、綺麗に整理するためではなく、事故を構造で防ぐためにある。**
> 「気をつける」は仕組みではない。破られたら分かる形（ルール・テスト・CI）にして初めて意味を持つ。
> 迷ったときは「このドキュメントに書いてあるか」を確認し、書いていなければ実装せずまず相談する。

---

## 1. 根本思想

### 予想の目的
**「強い馬」ではなく「その条件で走る馬」を見つける。**  
市場オッズを追いかけるのではなく、独自の適性評価で期待値のある馬を発見する。

### システムの目指す姿
参照モデル（優秀な予想家）の思想を機械化する：

| 要素 | 思想 |
|------|------|
| 適性ゲーム | 距離・コース・上がり・ペース耐性で好走馬は固定化される |
| 再現性重視 | 毎年繰り返されるパターンを信頼。単発データは信用しない |
| 着順より内容 | 負けても上がり最速なら評価。勝っても内容が悪ければ信用しない |
| レース構造 | 「どんな馬が来るレースか」の構造を先に理解する |
| 上がり最重視 | 東京芝では上がり最速経験が決定的 |
| ローテーション | 桜花賞組・フローラS組・忘れな草賞組で求められる能力が異なる |
| 完成度 | G1経験・重賞実績・OP実績を体系的に評価 |
| 市場は結果 | 市場オッズを特徴量に入れない。比較は予想後に行う |

---

## 2. RL/CL フレームワーク（核心設計）

### 2つの評価軸

```
RL（レースレベル）= 「この馬はこのクラスで速く走れるか」= 絶対能力軸
CL（コースレベル）= 「この馬はこの条件に合っているか」= 適性軸
```

アプリの表示列はこの2軸でランク付けする。

### RL を構成する要素
| 要素 | 重要度 | 現状 | 目標 |
|------|--------|------|------|
| 上がり3F実績（クラス比較） | ★★★★★ | ほぼゼロ | f_rl に統合 |
| テン3F・ペース適性 | ★★★★☆ | f_pace で部分対応 | f_rl に統合 |
| 前走メンバーレベル | ★★★★☆ | ゼロ | Phase 3 |
| クラス別スピード指数 | ★★★★★ | ゼロ | Phase 2 |
| G1・重賞経験（完成度） | ★★★★☆ | 断片的 | Phase 2 |

### CL を構成する要素
| 要素 | 重要度 | 現状 | 目標 |
|------|--------|------|------|
| 距離適性 | ★★★★☆ | f_distance で対応 | 精度向上 |
| コース適性 | ★★★★☆ | f_distance で部分対応 | 精度向上 |
| 枠順バイアス | ★★★☆☆ | f_post, f_bias で対応 | 維持 |
| 騎手適性 | ★★★☆☆ | f_jockey で対応 | 維持 |
| 血統適性 | ★★☆☆☆ | f_blood で部分対応 | 血統DB拡充 |
| ローテーション | ★★★★☆ | ほぼゼロ | Phase 3 |

---

## 3. 現在のシステムの問題点（2026-05-25時点）

### 致命的問題：データ層の貧困

`history.db` の現行スキーマ：
```sql
race_history:  race_id, date, racecourse, distance, surface, first_3f
horse_history: id, race_id, date, racecourse, horse_name, horse_num,
               place, running_style, agari3f, jockey, trainer,
               corner_3, distance, surface
```

**ない列：** `race_class`, `race_name`, `prize`, `num_finishers`,
`track_condition`, `margin`, `last_3f`, `agari_rank`

この欠如により：
- f_recent の `speed_bonus` がほぼ常に 0 にフォールバック
- G1での3着も条件戦での3着も同じ評価
- 上がり順位が計算できない
- レース格が不明

### 構造的問題：RL/CL の混在

現在の9特徴量は1つの加重和に混在。RL・CL という意味のある2軸で分離されていない。

### 具体的バグ事例（オークス2026）
- 市場152倍の単騎逃げ馬がAI1位（勝率15.8%）になった
- 原因：逃げ1頭のみ → ペナルティなし、長距離スロー予測 → 逃げに+2ボーナス
- 根本：G1適性・重賞経験ゼロを検知できるデータがない

---

## 4. ロードマップ

### Phase 0：RL/CL 分離表示（即時実装可能）

**目標：** 形式を正しくする。精度は現状のまま。

#### 実装内容

**engine.py に追加：**
```python
def calc_rl_cl_ranks(scored_horses):
    """RL/CLの生スコアを計算し、ランクを付与する"""
    RL_FEATURES = ['recent', 'pace']
    CL_FEATURES = ['distance', 'post', 'bias', 'jockey', 'blood']
    
    for h in scored_horses:
        sc = h['scores']
        rl_w = {k: _W.get(k, 0) for k in RL_FEATURES}
        cl_w = {k: _W.get(k, 0) for k in CL_FEATURES}
        rl_sum = sum(rl_w.values()) or 1
        cl_sum = sum(cl_w.values()) or 1
        h['rl_raw'] = sum(sc.get(k, 5.0) * rl_w[k] for k in RL_FEATURES) / rl_sum
        h['cl_raw'] = sum(sc.get(k, 5.0) * cl_w[k] for k in CL_FEATURES) / cl_sum
    
    sorted_rl = sorted(scored_horses, key=lambda h: h['rl_raw'], reverse=True)
    sorted_cl = sorted(scored_horses, key=lambda h: h['cl_raw'], reverse=True)
    for i, h in enumerate(sorted_rl):
        h['rl_rank'] = i + 1
    for i, h in enumerate(sorted_cl):
        h['cl_rank'] = i + 1
    return scored_horses
```

**calc_all() の末尾で呼び出す：**
```python
out = sorted(out, key=lambda x: x['total'], reverse=True)
calc_rl_cl_ranks(out)  # RL/CLランク付与
return out
```

**app_json.py で出力：**
```python
'rl_rank': h.get('rl_rank', 99),
'cl_rank': h.get('cl_rank', 99),
```

**完了条件：** アプリのRL列・CL列に意味のある数値が表示される

---

### Phase 1：DB スキーマ拡張（2〜3週間）

**目標：** RLを本物にするためのデータ基盤を作る。

#### 1-A: スキーマ追加

```sql
-- race_history の拡張
ALTER TABLE race_history ADD COLUMN race_class TEXT;
-- 'G1','G2','G3','L','OP','3勝','2勝','1勝','未勝利','新馬'
ALTER TABLE race_history ADD COLUMN race_name TEXT;
ALTER TABLE race_history ADD COLUMN num_finishers INTEGER;
ALTER TABLE race_history ADD COLUMN track_condition TEXT;
-- '良','稍重','重','不良'

-- horse_history の拡張
ALTER TABLE horse_history ADD COLUMN margin REAL;
-- 勝ち馬とのタイム差（秒）。1着は0。
ALTER TABLE horse_history ADD COLUMN last_3f REAL;
-- 上がり3Fタイム（秒）
ALTER TABLE horse_history ADD COLUMN agari_rank INTEGER;
-- レース内の上がり順位（1=最速）
ALTER TABLE horse_history ADD COLUMN prize INTEGER;
-- 獲得賞金（万円）
```

#### 1-B: スクレイパー修正

`parser.py` および `jra_scraper.py` で以下を取得するよう修正：
- レース名（race_name）
- クラス（race_class）: レース名から自動判定
- 上がり3F（last_3f）: 既存の agari3f と同義だが horse_history にも保存
- 上がり順位（agari_rank）: 同レース内の相対順位
- 着差（margin）: 勝ち馬との秒差
- 出走頭数（num_finishers）
- 馬場状態（track_condition）

#### 1-C: race_class 自動判定ロジック

```python
def infer_race_class(race_name, prize_total=None):
    """レース名と賞金からクラスを推定する"""
    name = race_name or ''
    if any(k in name for k in ['天皇賞','有馬記念','ジャパンC','宝塚記念',
                                '桜花賞','皐月賞','ダービー','オークス',
                                'スプリンターズS','マイルCS','菊花賞','エリザベス女王杯',
                                'フェブラリーS','高松宮記念','秋華賞','阪神JF','朝日杯']):
        return 'G1'
    if 'G2' in name or any(k in name for k in ['中山記念','京都記念','阪神大賞典']):
        return 'G2'
    if 'G3' in name:
        return 'G3'
    if 'L' in name or 'リステッド' in name:
        return 'L'
    if 'オープン' in name or 'OP' in name:
        return 'OP'
    # 賞金で推定
    if prize_total:
        if prize_total >= 5000: return 'OP'
        if prize_total >= 2000: return '3勝'
        if prize_total >= 1000: return '2勝'
        if prize_total >= 500:  return '1勝'
    if '未勝利' in name: return '未勝利'
    if '新馬' in name:   return '新馬'
    return '1勝'  # デフォルト
```

**完了条件：** 新規スクレイピングデータに race_class, last_3f, agari_rank, margin が含まれる

---

### Phase 2：RL 本格実装（1ヶ月後）

**目標：** スピード指数ベースの本物のRLスコア。

#### 2-A: クラス別基準上がりタイム定義

```python
# 東京芝・標準良馬場での各クラス基準上がり3Fタイム（秒）
# ※コース・距離・馬場によって補正が必要
CLASS_BASE_AGARI = {
    'G1':   33.5,
    'G2':   33.8,
    'G3':   34.0,
    'L':    34.2,
    'OP':   34.3,
    '3勝':  34.5,
    '2勝':  34.8,
    '1勝':  35.2,
    '未勝利':35.8,
    '新馬': 36.0,
}

TRACK_CONDITION_ADJUST = {'良': 0.0, '稍重': +0.3, '重': +0.6, '不良': +1.0}
```

#### 2-B: f_rl 関数

```python
def f_rl(h, race):
    """スピード指数・上がり実績ベースのRLスコア (0-10)"""
    hist = h.get('history', [])
    if not hist:
        return 5.0
    
    scores = []
    for i, r in enumerate(hist[:5]):
        last_3f   = r.get('last_3f') or r.get('agari3f') or 0
        race_class = r.get('race_class', '1勝')
        agari_rank = r.get('agari_rank', 9)
        num_fin    = max(r.get('num_finishers', 16), 8)
        track_cond = r.get('track_condition', '良')
        
        # ベースとなる上がり基準（馬場補正済み）
        base = CLASS_BASE_AGARI.get(race_class, 35.0)
        base += TRACK_CONDITION_ADJUST.get(track_cond, 0)
        
        # スピード指数（基準より速いほど高い）
        if last_3f > 0:
            speed_idx = (base - last_3f) * 10 + 50
        else:
            speed_idx = 50
        
        # 上がり順位ボーナス（最速ほど大きい）
        agari_pct = (num_fin - agari_rank) / max(num_fin - 1, 1)
        agari_bonus = agari_pct * 15  # 最速で+15、最遅で0
        
        # クラス格ボーナス（高クラスで走れたこと自体を評価）
        class_mult = {'G1':1.5,'G2':1.3,'G3':1.2,'L':1.1,'OP':1.05}.get(race_class, 1.0)
        
        raw = (speed_idx + agari_bonus) * class_mult
        weight = 0.75 ** i  # 直近ほど重み大
        scores.append((raw, weight))
    
    if not scores:
        return 5.0
    
    total_w = sum(w for _, w in scores)
    weighted_avg = sum(s * w for s, w in scores) / total_w
    
    # 40〜90の範囲を 0〜10 にスケール
    return max(0, min(10, (weighted_avg - 40) / 5))
```

#### 2-C: f_maturity（完成度スコア）

```python
def f_maturity(h, race):
    """G1・重賞・OP経験による完成度スコア (0-10)"""
    hist = h.get('history', [])
    
    class_points = {'G1':5, 'G2':3, 'G3':2, 'L':1.5, 'OP':1}
    place_mult   = {1:2.0, 2:1.5, 3:1.2}
    
    total = 0
    for r in hist:
        rc = r.get('race_class', '')
        pt = class_points.get(rc, 0)
        if pt > 0:
            p = r.get('place', 9)
            total += pt * place_mult.get(p, 1.0 if p <= 5 else 0.5)
    
    # 0〜15点を 0〜10 にスケール
    return min(10, total / 1.5)
```

#### 2-D: f_recent の再設計（着順より内容）

```python
def calc_race_content_score(r):
    """1走分の内容スコア（着順ではなく中身を評価）"""
    place      = r.get('place', 10)
    finishers  = max(r.get('num_finishers', 16), 2)
    margin     = r.get('margin', 0.0)
    agari_rank = r.get('agari_rank', finishers)
    race_class = r.get('race_class', '1勝')
    
    # 1. 相対着順スコア
    pos_score = max(0, 10 * (1 - (place - 1) / max(finishers - 1, 1)))
    
    # 2. 接戦ボーナス（0.3秒以内の負けは評価下げない）
    if place > 1 and 0 < margin <= 0.3:
        pos_score = min(10, pos_score + 1.5)
    
    # 3. 上がり順位ボーナス（最重要）
    agari_pct = (finishers - agari_rank) / max(finishers - 1, 1)
    agari_bonus = agari_pct * 3.0  # 最速で+3.0
    
    # 4. クラス格係数（高いクラスでの実績を重視）
    class_mult = {
        'G1':2.0,'G2':1.6,'G3':1.4,'L':1.2,'OP':1.1,
        '3勝':1.0,'2勝':0.85,'1勝':0.7,'未勝利':0.5,'新馬':0.5
    }.get(race_class, 1.0)
    
    return min(10, (pos_score + agari_bonus) * class_mult * 0.45 + 2.0)
```

#### 2-E: 重みの再調整（Phase 2 完了後にチューニング）

```python
# Phase 2 目標重み（チューニング前の初期値）
_W_TARGET_PHASE2 = {
    'rl':       0.35,  # 新規（f_rlに置き換え）
    'distance': 0.20,
    'pace':     0.15,  # RLに一部吸収されて減少
    'maturity': 0.10,  # 新規（f_maturity）
    'trainer':  0.08,  # 大幅減少（データ外要因）
    'jockey':   0.04,
    'blood':    0.03,
    'post':     0.03,
    'bias':     0.02,
}
# ※ 実際の重みは tune_weights.py で history.db から最適化する
```

**完了条件：** オークスクラスのG1で「G1経験馬 > 条件戦馬」の評価が出る

---

### Phase 3：ローテーション・メンバーレベル（継続）

**目標：** 「前走のメンバーが強かった」を自動検知する。

#### 3-A: 前走メンバーレベルの算出

```python
def calc_prev_member_level(horse_name, prev_race_id, history_db_path):
    """前走のメンバーレベルを算出（同レース出走馬のRL平均）"""
    import sqlite3
    conn = sqlite3.connect(history_db_path)
    
    # 同じrace_idで出走した全馬のagari3fと着順を取得
    rows = conn.execute(
        'SELECT horse_name, place, agari3f FROM horse_history WHERE race_id=?',
        (prev_race_id,)
    ).fetchall()
    conn.close()
    
    if len(rows) < 5:
        return 5.0  # データ不足
    
    # 上位馬の上がり平均をレベル指標とする
    top3_agari = [r[2] for r in sorted(rows, key=lambda x: x[1])[:3] if r[2] > 0]
    if not top3_agari:
        return 5.0
    
    avg_top3 = sum(top3_agari) / len(top3_agari)
    # 33秒台=高レベル(10)、36秒台=低レベル(3)
    return max(3, min(10, (36.5 - avg_top3) * 2.5 + 3))
```

#### 3-B: 主要重賞ローテーションテーブル

```python
PREP_RACE_PROFILES = {
    'オークス': {
        '桜花賞':        {'level': 5, 'dist_match': 0.3},  # マイル→2400m
        'フローラS':     {'level': 4, 'dist_match': 0.9},
        '忘れな草賞':    {'level': 3, 'dist_match': 0.95},
        'スイートピーS': {'level': 3, 'dist_match': 0.8},
    },
    '日本ダービー': {
        '皐月賞':    {'level': 5, 'dist_match': 0.6},
        '青葉賞':    {'level': 4, 'dist_match': 1.0},
        '京都新聞杯':{'level': 3, 'dist_match': 0.9},
    },
    # 主要G1のみ随時追加
}
```

**注：** 「今年の忘れな草賞が特別強い」という年次変動は、Phase 3-A の「前走メンバーレベル自動算出」が蓄積された後に初めて検知可能になる。**最低1シーズンのデータ蓄積が必要。**

---

## 5. 進捗トラッキング

| Phase | 状態 | 完了条件 |
|-------|------|---------|
| Phase 0: RL/CL 分離表示 | ✅ 完了 | アプリにRL/CLランクが表示される |
| Phase 1: DB スキーマ拡張 | ✅ 完了 | race_class, agari_rank 等が蓄積される |
| Phase 2: RL 本格実装 | ✅ 完了 | G1でG1経験馬が正当評価される |
| Phase 3: ローテーション | ✅ 完了 | 前走メンバーレベルが自動算出される |

### 即時修正（Phase 0 前に適用済み）
- [x] GAS廃止→GitHub Pages直接配信（2026-05-24）
- [x] Softmax temperature: 0.8 → 1.3（ノイズ増幅を抑制）
- [ ] 逃げ×長距離スロー補正: +2 → 0（要実装）
- [ ] 長距離初挑戦ペナルティ（要実装）

---

## 6. やってはいけないこと

| NG | 理由 |
|----|------|
| 市場オッズを特徴量に追加 | 大衆と同じ予想になるだけ。エッジが消える |
| パラメータの感覚的な調整 | 必ず tune_weights.py で history.db から最適化する |
| データなしの特徴量追加 | DBスキーマ拡張なしにf_rlを作ってもデフォルト値にフォールバックするだけ |
| ローテーションテーブルの過剰整備 | データが溜まる前に手動テーブルに頼りすぎない |
| 特定レース用の調整 | 1レースの問題を直すパラメータは別のレースで悪化する |
| スクレイピングに新規リクエスト元を追加する際、件数上限(budget)を設けない | 導入直後は対象が全件"新規"扱いになり、CIのタイムアウトでその回のデータが丸ごと失われる（2026-07-18に実際に発生） |
| 学習/推論の特徴量パリティを確認せずに特徴量を追加する | 学習時と予測時で計算経路が違うと、片方だけ空文字/デフォルト値になり気づかれないまま精度が落ちる（f_bloodのdam_sire希釈バグで実際に発生） |
| 実験用の一時ファイル（モデル・ノートブック）をdata/直下やリポジトリ直下にコミットしたままにする | 本番コードが「存在すれば優先ロード」する設計だと、消し忘れが本番モデルをサイレントに無効化する（xgb_ensemble_model.pklで実際に発生） |

---

## 7. データフロー全体像

```
[JRAサイト]
    ↓ スクレイピング（jra_scraper.py + parser.py）
    ↓ ← Phase 1 でここに race_class, agari_rank, margin 等を追加
[history.db]
    ├── race_history（レース情報）
    └── horse_history（馬別成績）
    ↓ init_engine() で辞書化
[特徴量エンジン（engine.py）]
    ├── f_rl（Phase 2 で追加）     → RL軸
    ├── f_pace（現行）             → RL軸（補助）
    ├── f_distance（現行）         → CL軸
    ├── f_maturity（Phase 2 追加） → RL軸（補助）
    ├── f_jockey（現行）           → CL軸
    ├── f_blood（現行）            → CL軸
    ├── f_post（現行）             → CL軸
    └── f_bias（現行）             → CL軸
    ↓ calc_all() → calc_rl_cl_ranks()
[予想結果（scored horses）]
    ├── rl_rank（RL順位）
    ├── cl_rank（CL順位）
    ├── total（最終スコア）
    └── pn（勝率確率）
    ↓ to_app_json() → push_to_github()
[GitHub Pages: data/latest.json]
    ↓
[スマホアプリ（index.html）]
    └── RL列・CL列・勝率・連対率・複勝率 表示
```

---

## 8. 参考：現行ファイル構成

```
src/
  features/engine.py      ← 特徴量計算の核心。このファイルが設計図の中心
  models/predict.py       ← softmax_probs, calibrate_and_renormalize
  betting/make_bets.py    ← calc_ev, calc_kelly, make_bets
  betting/ev_filter.py    ← ability_first_loose（EV×確率でレース厳選）
  betting/app_json.py     ← to_app_json（アプリ用JSON生成）
  utils/db.py             ← save_history_db（結果→history.db蓄積）
  utils/config.py         ← VENUE_PACE_TENDENCY 等の定数
  scraper/parser.py       ← HTML解析
  scraper/jra_scraper.py  ← JRAスクレイピング
data/
  history.db              ← 学習データ（Phase 1 でスキーマ拡張）
  optimal_weights.json    ← チューニング済み重み（Phase 2 後に再チューニング）
  latest.json             ← アプリ配信用JSON（GitHub Pages経由）
index.html                ← スマホアプリ本体
DESIGN.md                 ← このファイル（設計指針書）
CLAUDE.md                 ← セッション引き継ぎ文書
```
