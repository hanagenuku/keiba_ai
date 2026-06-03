# keiba_ai プロジェクト 引き継ぎドキュメント

## プロジェクト概要
JRA競馬AI予想システム。Google Colab + Google Drive で運用。

## リポジトリ
- GitHub: `hanagenuku/keiba_ai`
- 運用ブランチ: `main`（すべての変更はmainに直接push。Colabの強制アップデートセルもmainから取得）

## Colabノートブック構成
| ファイル | 用途 |
|----------|------|
| `KEIBA_土日_v5_ROI.ipynb` | 土曜朝（出馬表取得・予想）、土曜夜・日曜夜（結果取得・照合） |
| `KEIBA_金曜_v5_最新.ipynb` | 金曜夜（翌週レース確認・準備） |
| `KEIBA_チューニング_v1.ipynb` | 月1〜2回：重みチューニング＋キャリブレーション |
| `KEIBA_過去データ一括取得_v4.ipynb` | 過去データ一括取得専用（GitHubには未push・Drive管理） |

> ⚠ `KEIBA_過去データ一括取得_v4.ipynb` はGitHubに含まれていない。Driveのみで管理。

## Google Drive パス
`/content/drive/MyDrive/keiba_ai/`

## データ・モデル構造
```
data/
  history.db      # 学習データ（horse_history: 67,843件 / race_history: 4,893件）
  keiba.db        # 予想・ベット結果（bets, bet_simulation, results）
  optimal_weights.json  # チューニング済み重み（※Phase2-3後に再チューニング必要）
  calibrator.pkl
  horse_dist_dict.pkl / horse_course_dict.pkl / horse_venue_dist_dict.pkl
  post_zone_bias.pkl
  month_suffix_map.json
```

## 最新の重み（2026-05-22チューニング ※旧キーのまま・再チューニング必要）
```
distance:0.2667  pace:0.2666  trainer:0.2541  recent:0.1625
jockey:0.0100  blood:0.0100  post:0.0100  bias:0.0100  weight:0.0100
```
Acc@1: 19.6%  ECE: 0.0270
※Phase 2-3 実装後に rl/maturity/rotation が追加されたため再チューニング必須。

## 強制アップデートセル（チューニングノートのセル1とセル2の間に挿入）
```python
import urllib.request, os, sys
BASE_URL = 'https://raw.githubusercontent.com/hanagenuku/keiba_ai/main'
files = [
    'src/tools/__init__.py', 'src/tools/tune_weights.py',
    'src/tools/calibrate.py', 'src/tools/analyze_divergence.py',
    'src/tools/rescrape_history.py', 'src/tools/build_training_data.py',
    'src/tools/train_xgb.py', 'src/tools/calibrate_xgb.py',
    'src/features/engine.py',
    'src/utils/config.py', 'src/utils/db.py', 'src/utils/model_registry.py',
    'src/scraper/parser.py', 'src/scraper/jra_scraper.py',
    'src/models/__init__.py', 'src/models/calibration.py', 'src/models/predict.py',
    'src/betting/__init__.py', 'src/betting/make_bets.py',
    'src/betting/ev_filter.py', 'src/betting/app_json.py',
]
for rel in files:
    dest = f'{BASE_DIR}/{rel}'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(f'{BASE_URL}/{rel}', dest)
    print(f'OK {rel}')
for key in list(sys.modules.keys()):
    if 'src' in key:
        del sys.modules[key]
print('done')
```

## 主要ファイルと役割
| ファイル | 役割 |
|----------|------|
| `src/features/engine.py` | 特徴量エンジン。f_rl/f_maturity/f_rotation/f_pace等。Phase 2-3 実装済み |
| `src/models/predict.py` | softmax_probs, calibrate_and_renormalize |
| `src/betting/make_bets.py` | calc_ev, calc_kelly, make_bets |
| `src/betting/ev_filter.py` | ability_first_loose（EV×pnフィルタ） |
| `src/betting/app_json.py` | to_app_json（アプリ用JSON） |
| `src/utils/model_registry.py` | save_version, rollback |
| `src/scraper/jra_scraper.py` | JRAスクレイピング。Phase 1-3 対応済み |
| `src/tools/tune_weights.py` | 重みチューニング。Phase 2-3 の新キー(rl/maturity/rotation)対応済み |
| `src/utils/db.py` | save_history_db。Phase 1 スキーマ拡張済み |

## セッション履歴

### 2026-05-25：DESIGN.md 全Phase実装（Phase 0〜3 完了）

**Phase 0: RL/CL 分離表示**（engine.py + app_json.py）
- `calc_rl_cl_ranks()` 追加。`calc_all()` の戻り値に `rl_rank`/`cl_rank` 付与

**Phase 1: DB スキーマ拡張**（jra_scraper.py + db.py）
- `parse_result_soup()` に track_condition/race_class/margin/agari_rank/num_finishers/払戻金を追加
- `_parse_margin()` ヘルパー追加（ハナ=0.1、クビ=0.2 等）
- `get_history_from_db()` で新カラムを取得（COALESCE でNULL時フォールバック）。race_name も追加
- `save_history_db()` スキーマ拡張（9カラム追加）＋ ALTER TABLE マイグレーション

**Phase 2: RL 本格実装**（engine.py）
- `CLASS_BASE_AGARI`/`TRACK_CONDITION_ADJUST` 定数
- `calc_race_content_score()` — 着順・接戦ボーナス・上がり順位・クラス格係数
- `f_rl()` — スピード指数×クラス格の RL スコア (0-10)
- `f_maturity()` — G1/重賞/OP 経験の完成度スコア (0-10)
- `f_recent()` を calc_race_content_score ベースに再設計
- `_W` デフォルト: rl:0.35, distance:0.20, pace:0.15, maturity:0.10 ...

**Phase 3: ローテーション・メンバーレベル**（engine.py + jra_scraper.py）
- `PREP_RACE_PROFILES` — 主要G1の前哨戦テーブル
- `calc_prev_member_level()` — 前走メンバーレベル算出（DB参照）
- `f_rotation()` — メンバーレベル×ローテーション適合スコア

**即時修正**（engine.py）
- `PACE_STYLE_SCORE['長距離']['slow']['逃げ']`: +2 → 0
- `f_dist_v2()` に長距離初挑戦ペナルティ（-1.0）追加

**チューニング対応**（tune_weights.py）
- WEIGHT_KEYS に rl/maturity/rotation 追加
- エンジンから f_rl/f_maturity/f_rotation をインポートし sc 辞書に追加

### 2026-06-03：XGBoost再学習準備（engine.py + train_xgb.py + KEIBA_XGB_retrain.ipynb）
- `calc_features_for_xgb` に8個の新特徴量追加（f_sex, f_age, f_track_cond, f_heavy_track_rate, f_class_level, f_class_jump, f_finish_time_avg, f_time_diff_avg）
- `add_relative_features` に4列の相対化追加（cl_f_heavy_track, cl_f_weight_load, rl_f_finish_time, rl_f_time_diff）
- `train_xgb.py` ハイパーパラメータ更新（n_estimators=500, min_child_weight=10, early_stopping_rounds=50）
- `KEIBA_XGB_retrain.ipynb` 作成（セル1〜6: 学習データ生成→再学習→キャリブレーション→統合テスト）

### 2026-06-03：Stage3 全レース再スクレイプ完了
- KEIBA_Stage3_rescrape.ipynb を作成・実行（v5 URL構造を使用）
- race_history: 4,893件 / horse_history: 67,843件
- surface/track_condition/race_class/weather/weight_load/sex/age/corner_all/finish_time を補完
- bracket/win_odds/body_weight は列マッピングのズレにより未取得（要修正）
- ランタイム切れ後の再開ロジック組み込み済み（開催日×競馬場単位でスキップ）

### 2026-05-25：KEIBA_Stage3_rescrape.ipynb 作成・src/ バグ修正
- parser.py / jra_scraper.py の surface フォールバックを `'不明'` に統一
- `_parse_shutuba()` が `surface='不明'` のレースをスキップするよう修正

### 2026-05-22：history.db 8頭打ち切り補完・重みチューニング
- horse_history: 34,086件 → 62,835件（全頭取得に改善）
- ECE: 0.0726 → 0.0270

### 2026-05-19〜20：バグ修正・モジュール分離リファクタリング
- escape_count/front_count バグ修正
- 騎手・調教師・年齢・斤量の全馬定数バグ修正
- src/models/, src/betting/, src/tools/ 各種モジュール分離

## 残っている課題
| 課題 | 深刻度 | 備考 |
|------|--------|------|
| optimal_weights.json が旧キー（rl/maturity/rotation なし） | 高 | チューニング再実行が必要 |
| 過去データノートのセル7（pkl再生成）未実行 | 中 | チューニング前に必ず実行すること |
| horse_history.body_weight が 6.5% しか埋まっていない | 中 | Stage3の列順ズレが原因の可能性。tx[13]の内容を確認して修正が必要 |
| horse_history.bracket が 0% | 中 | tx[1]が枠番でない可能性（horse_numと混在？）。列マッピング要確認 |
| horse_history.win_odds が 0% | 中 | tx[11]が単勝オッズでない可能性。列マッピング要確認 |
| f_rotation のローテ照合は1シーズン後から有効 | 低 | データ蓄積待ち |
| 騎手DBが少数件 | 中 | save_history_dbで週次蓄積→自然解消 |

## 毎週の運用フロー
1. **金曜夜**: KEIBA_金曜ノートブック実行
2. **土曜朝**: 土日ノートブック実行（出馬表取得・予想）
3. **土曜夜**: 土日ノートブック実行（土曜結果・save_history_db・照合）
4. **日曜夜**: 土日ノートブック実行（日曜結果・save_history_db・照合）
5. **月1〜2回**: チューニングノートブック実行
6. **チューニング後**: save_version(BASE_DIR, ...) でバージョン保存

## 現在の作業状況（セッション引き継ぎ用）

### 最終更新: 2026-06-03

---

## ⚠️ 重要：設計指針書（必ず読むこと）

**`DESIGN.md`（このリポジトリのルート）を必ず参照すること。**
DESIGN.md の Phase 0〜3 はすべて実装完了（2026-05-25）。

### 次にやること（優先順）
1. **KEIBA_XGB_retrain.ipynb 実行**（XGBoost再学習）← 実装完了・実行待ち
   - セル1〜6を順番に実行（所要15〜30分）
   - 合格基準: AUC≥0.77 / Brier≤0.150 / cal_prob合計2.8〜3.2
2. **Stage3 列マッピング修正**（bracket/win_odds/body_weight が 0〜6.5%）← 要調査
   - JRA結果ページの実際の列順を確認し、`parse_result_page()` の tx インデックスを修正
   - 修正後に Stage3 を再実行（再開ロジックあり・完了済み開催日は自動スキップ）
3. **チューニングノート実行**（rl/maturity/rotation を含む重み最適化）→ optimal_weights.json 更新
4. **週末の実運用**で動作確認・ROI計測

### Stage3 再スクレイプ 完了状況（2026-06-03）
```
race_history （4,893件）
  surface: 100%  track_condition: 94.6%  race_class: 94.7%
  weather: 94.5%  num_finishers: 95.0%  race_name: 100%

horse_history （67,843件）
  surface: 100%  weight_load: 95.2%  sex/age: 90.8%
  corner_all: 94.5%  finish_time: 95.2%
  ❌ body_weight: 6.5%  bracket: 0%  win_odds: 0%  ← 要修正
```

### セッション開始時の確認事項
- PATをユーザーから取得（毎セッション必要）
- すべての変更は **main ブランチ**に直接push
- **ローカルgitリポジトリは使用しない**
- GitHubに**ないファイル**はユーザーに確認してから作業する

---

## git操作（PAT使用）
```bash
PAT="<ユーザーから取得>"
git remote set-url origin "https://${PAT}@github.com/hanagenuku/keiba_ai.git"
git push -u origin <branch-name>
git remote set-url origin "https://github.com/hanagenuku/keiba_ai.git"
```
