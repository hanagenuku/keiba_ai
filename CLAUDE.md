# keiba_ai プロジェクト 引き継ぎドキュメント

> ⚠ **作業前に `docs/KEIBA-AI_引き継ぎ書_追補_2026-06-28.md` も必ず参照すること。**
> モデル状況・週次運用フロー・Colab手順・既知制限の詳細が記載されている。

## 🌟 North Star（絶対に守るルール・最初にこれだけ読む）

このセクションはこのファイルが肥大化しても劣化させないための最重要ルールの抜粋。
詳細な経緯は各ルール末尾の日付から本文の該当セッションを参照。

1. **市場オッズ（win_odds等）を特徴量に追加しない**（DESIGN.md「6. やってはいけないこと」）。
   精度は上がって見えても市場コピー化してエッジが消える（2026-07-06〜09で実際に発生・後述）
2. **コード変更は作業ブランチ→PR→CI確認→mainマージ**。main直pushはしない
   （GitHub Actionsが自動生成するdata/配下のコミットのみ例外）
3. **データなしの特徴量追加をしない**。追加する場合は学習時と推論時で同じ経路・同じ値が
   入ることを確認する（学習/推論パリティ）。パリティが崩れると片方だけ静かにデフォルト値化する
4. **スクレイピングに新規リクエスト元を追加する際は、件数上限(budget)を必ず設ける**。
   無制限だと導入直後に全件が"新規"扱いになりCIタイムアウトでその回のデータが丸ごと失われる
   （2026-07-18で実際に発生）
5. **実験用の一時ファイル・ノートブックを本番コードパス（data/直下・リポジトリ直下）に
   コミットしたままにしない**。「存在すれば優先ロード」する設計だと消し忘れが本番モデルを
   サイレントに無効化する（2026-07-16で実際に発生）
6. **テストは「本番と同じデータ型・同じフォーマット」で書く**。DBの行を扱うコードは
   手打ちdictではなく実際に`sqlite3.Row`として取り出したもので、スクレイパーは
   綺麗な自作HTML文字列ではなく実機のHTML構造を模したfixtureで、モデルファイルを
   読むコードは実際にpickle/xgboost UBJ等その形式で保存したファイルで検証する。
   「本物に似ているが微妙に違う代用品」でテストするとテストは通り続けるが本番だけ壊れる、
   という事故がこの1ヶ月で複数回発生した（`sqlite3.Row.get()`未対応・Shift_JIS未指定・
   GitHub/Drive間history.dbスキーマずれ等、いずれもテストは通っていたのに本番で発覚。
   2026-07-22③〜07-23④で繰り返し確認・2026-07-23⑥セッションでNorth Star格上げ）
7. 迷ったら**DESIGN.mdの「やってはいけないこと」表**を確認する。書いていなければ
   実装せずまず確認・相談する

### ⚠ 過去に試して撤回した判断（同じ提案を繰り返さないために）

| 時期 | 試したこと→分かったこと | 現在の状態 |
|------|------------------------|-----------|
| 2026-06-27〜07-14 | 市場補正レイヤー(後付け抑制)→市場特徴量を直接追加→**AIが市場のコピーになった**（f_popularity重要度24.6%）→残差学習(市場からのズレのみ学習)に転換 | 市場補正レイヤーは完全廃止。残差学習が本番稼働中 |
| 2026-07-10 | pairwiseモデルを試した | T=5.0で使い物にならず完全削除 |
| 2026-07-02〜 | dual_model（単勝はB2_ndcg、他はA_fukusho）を実装 | 本番パス(feat_dfなし)では使われず凍結。Colab実験用に残存のみ |
| 2026-06-08〜2026-07-16 | 券種選択モデル(bet_selector_model)を学習・改良 | 分類精度がベースライン並みで実用に耐えないと判定。ロード処理も削除済み |
| 2026-07-05〜06 | ev_direct(Val列 = pn×odds)を買い目選択の根拠にしようとした | 識別力なし（EV>=1.3でも勝率≒baseline）と判明。粗いフィルタ以上の用途では使わない |
| 2026-07-06 | shadow_bets（成績記録）は結果取得時にcalc_all()を事後再実行していた | 最終オッズが特徴量に混入するリークと判明。朝予想スナップショット参照に修正済み |
| 2026-07-16 | （気づかず放置）xgb_ensemble_model.pklという実験の消し忘れファイル | 残差学習モデルをサイレントに無効化していた重大バグと判明・修正済み |
| 2026-07-17 | （気づかず放置）f_blood()が母父(dam_sire)を常に汎用値とブレンドしていた | 父側の実データを常に30%希釈していたバグと判明・修正済み |
| 2026-07-18 | 血統スクレイピングをbudget上限なしで実装 | 導入直後に全馬が"新規"扱いになりCIタイムアウトでその回のデータ喪失。budget機構を追加して修正済み |

---

## プロジェクト概要
JRA競馬AI予想システム。Google Colab + Google Drive で運用。

## リポジトリ
- GitHub: `hanagenuku/keiba_ai`
- 本番ブランチ: `main`（Colabの強制アップデートセル・GAS・各ワークフローは `main` から取得）
- **コード変更の運用フロー（2026-06-23 変更）**:
  作業ブランチ → **Pull Request 作成 → CI(テスト)確認 → main へマージ**。
  `main` への直接pushは原則しない（コードレビュー・CIを必ず通すため）。
  ただし GitHub Actions（金曜/週末/日曜ワークフロー）が自動生成する
  **データコミット**（latest.json / *.db / stats等）は従来どおり bot が `main` へ直接pushする。

## Colabノートブック構成
| ファイル | 用途 |
|----------|------|
| `KEIBA_土日_v5_ROI.ipynb` | 土曜夜（土曜結果取得＋日曜予想）、日曜夜（日曜結果取得・照合）※GitHub Actionsが主体、Colabはチューニング用 |
| `KEIBA_金曜_v5_最新.ipynb` | 金曜夜（翌週レース確認・準備） |
| `KEIBA_チューニング_v1.ipynb` | 月1〜2回：重みチューニング＋キャリブレーション |
| `KEIBA_XGB_retrain_v5.ipynb` | XGB再学習＋残差学習モデル本番投入（セル1〜10を順に実行） |
| `KEIBA_過去データ一括取得_v4.ipynb` | 過去データ一括取得専用（GitHubには未push・Drive管理） |

> ⚠ `KEIBA_過去データ一括取得_v4.ipynb` はGitHubに含まれていない。Driveのみで管理。

## Google Drive パス
`/content/drive/MyDrive/keiba_ai/`

## データ・モデル構造

> ⚠ `history.db`（race_history/horse_history）のカラム定義・意味・充足率は
> `docs/history_db_schema.md`（スキーマ契約書）を参照。カラムが存在することと
> 実際にデータが埋まっていることは別問題（例: `bracket`列は存在するが実データ0%）。
> 新しい特徴量を追加する前に必ず確認すること。

```
data/
  history.db      # 学習データ（race_history: 11,153件以上 / horse_history: 対応する出走数）
  keiba.db        # 予想・ベット結果（bets, bet_simulation, results）
  optimal_weights.json  # チューニング済み重み（※Phase2-3後に再チューニング必要）
  calibrator.pkl
  horse_dist_dict.pkl / horse_course_dict.pkl / horse_venue_dist_dict.pkl
  post_zone_bias.pkl
  month_suffix_map.json
```

## 最新の重み（rl/maturity/rotation 含む新キーで再チューニング済み）
```
jockey:0.2943  distance:0.2552  pace:0.2003  trainer:0.1702
rl:0.01  maturity:0.01  rotation:0.01  recent:0.01
blood:0.01  post:0.01  bias:0.01  weight:0.01
```
※ Phase 2-3 の新キー(rl/maturity/rotation)を含めて再チューニング済み。
※ ただし rl/maturity がほぼ無効化（0.01）されている点は要確認（後述「重みの妥当性確認」）。

## 強制アップデートセル（チューニングノートのセル1とセル2の間に挿入）
```python
import urllib.request, os, sys
BASE_URL = 'https://raw.githubusercontent.com/hanagenuku/keiba_ai/main'
files = [
    'src/tools/__init__.py', 'src/tools/tune_weights.py',
    'src/tools/calibrate.py', 'src/tools/analyze_divergence.py',
    'src/tools/rescrape_history.py', 'src/tools/build_training_data.py',
    'src/tools/train_xgb.py', 'src/tools/calibrate_xgb.py',
    'src/tools/generate_style_advantage.py',
    'src/tools/train_pace_model.py',
    'src/features/engine.py', 'src/features/speed_index.py', 'src/features/horse_type.py',
    'src/features/error_tags.py', 'src/features/shap_explain.py',
    'src/utils/config.py', 'src/utils/db.py', 'src/utils/model_registry.py',
    'src/scraper/parser.py', 'src/scraper/jra_scraper.py',
    'src/models/__init__.py', 'src/models/calibration.py', 'src/models/predict.py',
    'src/betting/__init__.py', 'src/betting/make_bets.py',
    'src/betting/ev_filter.py', 'src/betting/app_json.py',
    'src/betting/rank_matrix_filter.py',
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
| `src/features/shap_explain.py` | ability_marginをカテゴリ別SHAP寄与度に分解する説明可能性レイヤー |
| `src/models/predict.py` | softmax_probs, calibrate_and_renormalize |
| `src/betting/make_bets.py` | calc_ev, calc_kelly, make_bets |
| `src/betting/ev_filter.py` | ability_first_loose（EV×pnフィルタ） |
| `src/betting/app_json.py` | to_app_json（アプリ用JSON） |
| `src/betting/rank_matrix_filter.py` | 人気×RL乖離マトリクス実績に基づく買い目フィルタ（boost/suppress判定） |
| `src/utils/model_registry.py` | save_version, rollback |
| `src/scraper/jra_scraper.py` | JRAスクレイピング。Phase 1-3 対応済み |
| `src/tools/tune_weights.py` | 重みチューニング。Phase 2-3 の新キー(rl/maturity/rotation)対応済み |
| `src/utils/db.py` | save_history_db。Phase 1 スキーマ拡張済み |

## 市場ベースラインKPI（2026-07-10 導入）

### 概要
AIモデルと市場（オッズ）の予測精度を log-loss で比較する唯一のKPI。
`generate_stats.py` が毎週のワークフロー実行時に自動算出し、stats.json に出力。

### 計算方法
- AI確率: `win_prob`（softmax出力、レース内合計1）
- 市場確率: `1/tansho_odds` をレース内で正規化（合計1）
- 正解: `actual_place == 1`
- log-loss: `-mean( y*log(p) + (1-y)*log(1-p) )`
- **delta = AI log-loss - 市場 log-loss**（負ならAI優位）

### 出力先
- `stats.json` の `model_kpi` セクション（全体 + 日別ブレークダウン）
- `data/kpi_weekly.json`（累積週次トレンド）

### 判定基準
| delta | verdict | 意味 |
|-------|---------|------|
| < -0.001 | AI優位 | AIの予測が市場より正確 |
| > 0.001 | 市場優位 | 市場の予測がAIより正確 |
| それ以外 | 同等 | 差なし |

### 目標
delta を負にする（AI < 市場）ことが全ての改善の指標。
delta が正の間は、AIが市場に劣っている＝馬券で長期プラスにならない。

---

## セッション履歴

### 2026-07-13：乖離分析蓄積システム + オッズ変動×結果分析

#### 概要
AI予測と市場オッズの乖離を定量化し、結果との相関を週次蓄積する仕組みを追加。
直前オッズ変動（急騰・急落）と結果の因果関係分析も統合。

#### 実装内容
- `scripts/generate_stats.py`:
  - `calc_divergence_analysis()`: AI確率/市場確率の比率を6バケットに分類し、勝率・3着内率を集計
    - 本命一致/不一致時の成績比較、過大/過小評価馬ランキング
  - `calc_odds_movement_analysis()`: 朝→直前オッズ変動を5段階に分類（急騰/上昇/横ばい/下降/急落）
    - AI評価との一致/不一致別の成績、大変動馬リスト
  - `_save_divergence_weekly()`: `data/divergence_weekly.json`に週次蓄積（同日上書き）
  - `generate_stats()`に統合: stats.jsonに`divergence_analysis`・`odds_movement`セクション追加
- `tests/test_divergence_analysis.py`: 9テスト新規

#### 出力先
- `stats.json` の `divergence_analysis` / `odds_movement` セクション
- `data/divergence_weekly.json`（累積週次トレンド）

#### 日曜ワークフローでの自動蓄積
`generate_stats()`は既にsunday-results.ymlから呼ばれるため、追加設定不要で自動蓄積開始。

---

### 2026-07-12：展開予測モデル強化（19特徴量化）

#### 概要
従来の8特徴量ペース分類器を19特徴量に拡張。レース展開予想の精度向上を目指す。

#### 新特徴量（11個追加）
| カテゴリ | 特徴量 | 意味 |
|----------|--------|------|
| 枠順×脚質 | escape_avg_pos | 逃げ馬の平均馬番（内枠→ハナ取りやすい） |
| 枠順×脚質 | escape_outer_ratio | 逃げ馬のうち外枠(>60%)にいる割合 |
| ペース耐性 | escape_avg_pop | 逃げ馬の平均人気（人気=実力→ペース耐性高） |
| コース特性 | straight_length | 直線長（course_profiles.json） |
| コース特性 | straight_class | 直線分類(1-4) |
| コース特性 | has_uphill | 坂の有無 |
| コース特性 | n_corners | コーナー数（距離から推定） |
| 騎手傾向 | jockey_pace_median | 逃げ騎手の正規化前半3F中央値 |
| 騎手傾向 | jockey_escape_pct | 全騎手の平均逃げ率 |
| 馬場 | condition_num | 馬場状態(良0/稍重1/重2/不良3) |

#### 実装内容
- `src/tools/train_pace_model.py` 新規作成
  - `_classify_pace()`: first_3f→ペース3分類（距離正規化・表面別閾値）
  - `_build_jockey_pace_stats()`: 騎手ごとの逃げ時ペースメイク統計
  - `_build_features()`: 19特徴量構築
  - `train_pace_model()`: XGBClassifier学習パイプライン
  - 保存: `pace_model.pkl`（LabelEncoder添付）、`jockey_pace_stats.json`
- `src/features/engine.py` 更新
  - `_JOCKEY_PACE_STATS` グローバル追加、`init_engine()` で自動ロード
  - `_build_pace_features_for_inference()` 新関数: 推論時に19特徴量を構築
  - `calc_pace_distribution()`: 新モデル（`_pace_feature_cols`属性あり）なら19特徴量、旧モデルなら8特徴量で後方互換
  - `_pace_label_encoder` からクラス順序を取得（ハードコード排除）
- `tests/test_pace_model.py` 新規18テスト

#### Colabでの再学習手順
```python
from src.tools.train_pace_model import train_pace_model
result = train_pace_model(BASE_DIR)
# → data/pace_model.pkl + data/jockey_pace_stats.json が生成
# → 次回 init_engine() で自動ロード
```

#### 安全性
- 旧モデル（8特徴量）は自動退避（pace_model_old.pkl）
- 旧モデルフォーマットでも `calc_pace_distribution()` が後方互換で動作
- `jockey_pace_stats.json` 未生成でもデフォルト値で推論可能

---

### 2026-07-10：大掃除完了 + 市場ベースラインKPI導入

#### Phase A: 大掃除（PR #46 マージ済み）
- pairwise モデル完全削除（rating_calibration/train_ranking_model/compare_models/.gitattributes）
- value_gap 計算ロジック撤去（ev_filter.py、常時0.0を返す後方互換）
- dual_model 凍結（bet_optimizer.py の feat_df パスを削除、dual_model.py は Colab 用に残存）

#### Phase B: 市場ベースラインKPI（PR #47 マージ済み）
- `scripts/generate_stats.py` に `calc_model_kpi()` 追加
  - race_predictions の win_prob（AI）と tansho_odds（市場）から log-loss を算出
  - stats.json に `model_kpi` セクションを出力
- `_save_kpi_weekly()` で `data/kpi_weekly.json` に累積追記
- `tests/test_model_kpi.py` 新規10テスト

---

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

### 2026-06-05：スピード指数（Speed Figure）実装
- `src/features/speed_index.py` 新規作成（SpeedIndexCalculator + load/rebuild キャッシュ）
- 基準タイム: (distance, surface, track_condition) ごとの1着馬 finish_time 中央値
- Track Variant: 同日×同競馬場の全レースで基準タイムからのズレの中央値
- `engine.py` に特徴量4個追加: f_speed_fig_last / f_speed_fig_avg / f_speed_fig_max / f_speed_fig_trend
- `engine.py` の `add_relative_features` に相対特徴量1個追加: rl_f_speed_fig_avg
- `init_engine` で speed_index_cache.pkl を自動ロード（なければ history.db から構築）
- 強制アップデートセルに `src/features/speed_index.py` を追加

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

> 2026-07-21セッションで各課題の初出時期〜現在までの経緯を追跡し、実際の
> コードでの使われ方を確認した上で深刻度を再評価した（詳細は本セクション末尾の
> セッション履歴参照）。

| 課題 | 深刻度 | 備考 |
|------|--------|------|
| Gumbel rating温度(`DEFAULT_GUMBEL_RATING_T=2.5`)が残差モデルのスケール修正後に未検証 | **高（新規）** | 2026-07-26③で発見・記録。2重sigmoidバグ修正により`rating`（Gumbelシミュレーション入力）のスケールが本来の（より広い）log-odds幅に戻ったが、T=2.5は2026-07-06に旧（非残差）モデルのスケールを基準にフォワードデータで最適化された値で、修正後の残差モデルに対して再検証されていない。次回フォワードデータ蓄積後、Gumbel買い目のlog-loss/ECEを確認しT再校正を検討すること |
| 残差学習モデルのbase_marginが学習時(確定人気)と推論時(発売直後の薄いオッズ由来人気)で情報の成熟度が違う | **高** | 2026-07-24⑤で発見。ユーザーからの「前日の薄いオッズを土台にするのは腑に落ちない」という指摘がきっかけ。市場KPI・乖離分析で観測されている「AIより市場の方が正確」という結果の一部は、AI手法自体の欠陥ではなく比較の土台（AIは金曜夜の薄い人気、比較対象の結果は最終確定オッズ）が不公平である可能性がある。次回モデル見直し時の最有力調査候補。詳細は本セクション末尾のセッション履歴参照 |
| horse_history.bracket が 0% | 低（旧:中） | db.pyへの書き込みのみでどのf_*関数からも読まれておらず、現状のモデル・特徴量に影響なし。将来bracketベースの特徴量を新設する時まで対応不要 |
| horse_history.win_odds が 0% | 低（旧:中） | 2026-07-06に代替済み。engine.pyに「win_odds0%のためpopularity(99.2%充足)を使う」との明示コメントあり。実質解消済み |
| horse_history.body_weight が 6.5% しか埋まっていない | 低（旧:中） | build_training_data.pyがSQLで取得するのみでどのf_*関数にも渡されておらず未使用。将来馬体重ベースの特徴量を新設する時まで対応不要 |
| optimal_weights.json で rl/maturity がほぼ無効化（0.01） | 低（旧:中） | 本番でXGBが正常動作している間、ルールベース重み`_W`はcalc_all()で一切使われない（use_xgb時はtotal=raw_prob*10）。`_W`が使われるのはXGB推論が例外を吐いた時のフォールバックのみのため、重みの検証価値は低い |
| B2_ndcg（dual_model）を残差学習版で再学習 | 低（旧:中） | app_json.pyの本番呼び出しはfeat_dfを渡しておらずdual_model自体が発動しない。本番未使用のモデルの改良は優先度低 |
| 過去データノートのセル7（pkl再生成）未実行 | 低（旧:中） | CLAUDE.md内でこの1箇所以外に言及なし。horse_dist_dict等は`_build_and_save_stats()`によりチューニング実行のたびに自動再生成されており、既に自然解消している可能性が高い（Colab側の実行ログで最終確認が必要） |
| f_rotation のローテ照合は1シーズン後から有効 | 低 | データ蓄積待ち |
| 騎手DBが少数件 | 低 | save_history_dbで週次蓄積→自然解消 |

## 毎週の運用フロー
1. **金曜夜〜土曜朝**: 「金曜予想」ボタン → 土曜レースの予想生成（friday-predict.yml）
2. **土曜夜**: 「土曜結果＋日曜予想」ボタン → 土曜レース結果取得＋日曜レースの予想生成（weekend.yml）
3. **日曜夜**: 「日曜結果」ボタン → 日曜レース結果取得・save_history_db・照合（sunday-results.yml）
4. **月1〜2回**: チューニングノートブック実行
5. **チューニング後**: save_version(BASE_DIR, ...) でバージョン保存

## 市場補正レイヤー（2026-06-27 導入）

### 概要
XGBoostの予測（cal_prob）を市場人気で補正する層。
「AIが高評価だが市場が極端に低評価」の馬を抑制する。

### 発動条件
`MARKET_CORRECTION_ENABLED = True`（環境変数 `MARKET_CORRECTION` で制御。デフォルトON）

### 補正の内容
- RL上位 × 不人気(10番人気以上) → cal_prob × 0.30（大幅抑制）
- RL上位 × 人気(1-3番人気) → cal_prob × 1.0（信頼そのまま）
- RL下位 × 人気(1-3番人気) → cal_prob × 1.2（強調）
（詳細は `src/features/market_correction.py` の `CORRECTION_FACTORS`）

### 実装の仕組み
- `engine.py` の `calc_all()` 内で、softmax 前に `apply_market_correction()` を呼ぶ
- `total`（softmax入力）と `cal_prob`（表示用）の両方に同じ補正係数を乗算
- `total` は合計保存で正規化、`cal_prob` は合計3.0で正規化
- 補正前の値: `cal_prob_raw`（馬辞書）、`rl_rank_raw`（馬辞書）で参照可能

### アプリでの表示
- `🔧 市場補正 ON` バッジがバイアスバーの下に常時表示（忘れ防止）
- 補正で順位が変わった馬はRL欄に「旧位↑新位」「旧位↓新位」と表示
- 補正で本命が変わったレースは「🔧 補正により本命変更: #旧→#新」の注記が出る

### 導入理由
6/27の32レースで AI RL1の3着内率46.9% vs 市場1番人気75%。
AIが市場と異なる本命を出した25Rで市場が6倍正確だったため、暫定補正を導入。

### 今後の方針（暫定対応）
- これは手動の補正係数（6/27データ基準）。完璧でなくていい、明らかな暴走を抑える
- データ4週間分蓄積後に `correction_table.json` による自動更新へ移行予定
- `CORRECTION_FACTORS` の調整はフォワードテストの結果を見て随時更新

---

## 現在の作業状況（セッション引き継ぎ用）

### 最終更新: 2026-07-27③（人気×RL乖離マトリクス実績に基づく買い目フィルタを実装）

---

### 2026-07-27③：人気×RL乖離マトリクス実績に基づく買い目フィルタを実装

#### 背景
②で追加したマトリクスの**実データによる集計を、このセッション内で実施できた**
（後述「本環境からkeiba.dbを読む方法」）。その結果に基づき、実績のあるセルへ
買い目を寄せ、実績の悪いセルを切るフィルタを実装した。

#### 🔑 本環境からLFS管理のkeiba.db実データを読む方法（重要・今後も使う）
これまで「本環境ではkeiba.dbがLFSポインタのため分析できない」と繰り返し記録して
きたが、**`media.githubusercontent.com/media/{owner}/{repo}/{branch}/{path}` から
LFS実体を直接HTTP取得できる**ことが判明した（git-lfsコマンド不要）：
```bash
curl -sL -o /tmp/keiba.db \
  "https://media.githubusercontent.com/media/hanagenuku/keiba_ai/main/data/keiba.db"
```
これにより、以降のセッションでは**ユーザーにColab実行を依頼せずとも、その場で
本番の実データを使った分析・検証が可能**。今後「LFSだから読めない」と諦める前に
必ずこの方法を試すこと。

#### 実データによる集計結果（累積332レース、うち残差モデル期137レース）
まず②で概算した「AI強気乖離ゾーンのROI 124%」は、**厳密計算では77.3%**だった
（概算は「バケット平均オッズ×勝率」で、実際に勝つ馬がバケット内の低オッズ側に
偏るぶん上振れする——②で警告した通りの結果）。全6バケットとも100%未満。

さらに重要な発見として、**ユーザーの当初仮説「市場5番人気×RL1に妙味がある」は
実データでは否定された**（市場4-5人気×RL1: N=22, 勝率9.1%, ROI 38.6%）。加えて
残差モデル期では「RL1が市場4番人気以下」という状況自体がほぼ消滅している
（137レース中4頭のみ）。残差学習モデルは市場を土台にするため、AIの本命は
構造的に市場上位3人気に収束する。

代わりに**中位帯の乖離セルに一貫した好成績**が見つかった（全期間318レース）：

| セル | N | 勝率 | 単勝回収率 | 判定 |
|---|---|---|---|---|
| 市場2-3人気×RL4-5 | 123 | 23.6% | **144.9%** | BOOST |
| 市場6-9人気×RL4-5 | 168 | 10.1% | **145.7%** | BOOST |
| 市場6-9人気×RL2-3 | 88 | 9.1% | **130.3%** | BOOST |
| 市場2-3人気×RL6-9 | 92 | 6.5% | 37.7% | SUPPRESS |
| 市場4-5人気×RL10+ | 50 | 4.0% | 36.4% | SUPPRESS |

**「AIが本命に推す馬」ではなく「AIが中位に評価しているが市場はもっと低く見ている馬」
に妙味がある**という構造。従来の買い目ロジックは単勝候補をRL上位3頭に限定して
いたため、最も実績の良い「市場中位人気×RL4-5」の馬は**構造的に候補にすら
入らなかった**。

#### 実装内容
- `src/betting/rank_matrix_filter.py` 新規作成
  - `divergence_weekly.json` の最新 `rank_matrix` を読み（キャッシュ付き）、
    セル実績から `'boost'` / `'suppress'` / `None` を判定
  - 閾値: `MIN_N=50`（これ未満は判定しない）、`SUPPRESS_ROI=50`、`BOOST_ROI=120`
  - `_rank_band()` は `generate_stats.py` の帯定義と完全一致させること
    （ズレると集計と適用で違うセルを見ることになる。テストで検証済み）
  - 環境変数 `RANK_MATRIX_FILTER=0` で全体無効化（kill switch）
- `src/betting/bet_optimizer.py`:
  `build_optimal_bets(..., base_dir=None)` を追加（**省略時は完全に従来挙動**）。
  `_select_win()` は suppress馬を除外し、boost馬はRL4-5でも候補に含める。
  `_select_place()` は suppress除外のみ（複勝回収率は未測定のためboost拡張はしない）
- `src/betting/app_json.py`: 馬ごとのJSONに `mx_flag` を追加
- `index.html`: `recalcGumbelBets()`（直前オッズ取得後のクライアント側再計算）に
  サーバ側と同一のフィルタ規則を実装。馬テーブルに💎(boost)/▽(suppress)バッジ追加

#### ⚠ 既知の限界（重要）
- `rank_matrix` は**全期間累積**のため、旧・市場コピー型モデル期（〜2026-07-14）の
  データが混ざっている。残差モデル期のみのデータが蓄積するほど純化される
- N=50〜168はまだ小さく、数回の的中で回収率が動く段階。ただし「中位乖離帯の
  複数セルが揃って100%超」というパターンの一貫性は偶然にしては整っている
- 回収率は単勝ベースの実測のみ。複勝・馬連・三連複の回収率は未測定
- **これは意思決定層のフィルタであり、モデルの予測精度自体は変えていない**

#### テスト
`tests/test_rank_matrix_filter.py` 新規12テスト。North Starに従い
`divergence_weekly.json` は本番と同じフォーマットで実ファイルとして書き出して検証：
- boost/suppress/N不足/中間帯/順位不明/ファイル欠損/kill switchの各判定
- `_rank_band()` が `generate_stats.py` の帯定義と一致すること
- `build_optimal_bets` 統合: boost馬が単勝候補に入り選ばれること、suppress馬が
  高EVでも単勝・複勝から除外されること、**base_dir省略時は従来挙動が保たれること**
- `index.html` 側は `node --check` ＋ 同一入力でPython側と同じ選択結果になることを確認
- `python -m pytest tests/ -q` は383テスト通過（371+12、回帰なし）

#### 次回確認事項
次回の予想生成（金曜/土日ワークフロー）から、アプリの馬テーブルに💎/▽バッジが
表示され、単勝・複勝の買い目がフィルタを反映する。**💎が付いた馬が実際に
買い目に選ばれているか**、および数週間後のROI推移を確認すること。

---

### 2026-07-27②：人気×RL乖離マトリクス＋厳密単勝回収率を週次乖離分析に追加

#### 背景
ユーザーから「1番人気の馬のRLが5、5番人気の馬がRL1のような順位の食い違いを深く
考察・学習したい。早く回収率を上げたい」という要望を受けた。既存の乖離分析
（2026-07-13導入、確率比率ベース6バケット）を確認したところ、**勝率・複勝率のみで
回収率が無く、順位×順位の粒度も無い**ことが分かった。

さらに重要な指摘として「市場5番人気×RL1の馬は勝率では負けているが、馬券回収率的には
どうか？」という質問を受け、蓄積データ（累積318レース・3,983頭）から概算したところ、
**勝率で最悪だったAI強気乖離ゾーン（AI>>>市場: 勝率3.0%）が、推定平均オッズ41.2倍を
掛けると概算ROI 124%になる**という、勝率だけを見た従来の解釈（「AI強気乖離＝AIが
間違っている場所」）と逆の可能性が浮かび上がった。ただしこの概算は「バケット平均
オッズ×勝率」であり、実際に勝つ馬はバケット内の低オッズ側に偏るため上振れバイアスが
ある。厳密な回収率は`race_predictions`の各馬の実オッズ×実着順から計算する必要がある
（本環境ではkeiba.dbがLFSポインタのため計算できず、次回ワークフロー実行時に本番で
計算される形で実装した）。

#### 実装内容（`scripts/generate_stats.py` `calc_divergence_analysis()`）
1. **厳密単勝回収率 `_tansho_roi()`**: 全頭に単勝100円を賭けた場合の実回収率を
   「勝ち馬の実オッズ合計 ÷ 頭数」で厳密計算。既存の6バケット（bucket_stats）に
   `tansho_roi`キーを追加
2. **人気帯×RL帯マトリクス `rank_matrix`**: 人気帯（1/2-3/4-5/6-9/10+）×RL帯
   （同）の格子で、各セルのcount/win_rate/top3_rate/tansho_roiを算出。
   「市場1番人気×RL5」「市場5番人気×RL1」等のセルが直接数字で見える。
   popularity/rl_rankが99（不明）の馬はマトリクスから除外
3. `_save_divergence_weekly()`はdivergence辞書全体を保存する設計のため、
   `rank_matrix`は**コード変更なしで自動的にdivergence_weekly.jsonへ週次蓄積**される
4. `index.html`成績ページに「🎲 人気×AI順位 実績マトリクス」カードを追加。
   各セル上段=単勝回収率（100%以上=緑/80-100=橙/80未満=グレー）、下段=勝率(件数)。
   N<30のセルは薄字表示（サンプル不足の可視化）

#### この分析の位置づけ（ユーザーとの合意事項）
- これは**測定**であり、AIの予想精度自体を上げるものではない（ユーザーからの
  「それは根本的にAIの予想精度を上げていることにはならない」という指摘に同意済み）。
  根本的な精度向上は(a)base_margin成熟度ズレの修正再学習（07-24⑤）、(b)単勝専用
  学習シグナルのモデル、のモデル本体作業であり、これは次のColab作業として別途設計する
- 乖離ゾーンの劣勢（勝率ベース）の一部は「AIの土台が前夜の薄い人気」という構造問題
  （07-24⑤）の影響を受けている可能性があり、7/25導入の当日朝refreshの効果で今後
  数字が変わりうる。マトリクスの週次蓄積でrefresh導入後のデータを追跡できる
- セルごとのNが溜まった後、実績マイナスのセルの買い目を抑制する**買い目フィルタ**
  （モデル変更ではないため凍結方針と矛盾しない）を次段階として検討する

#### 次回結果取得時の確認事項
次回「日曜結果」ボタン押下（=sunday-results.yml実行）で、蓄積済み全レース分の
厳密な回収率マトリクスがstats.json・divergence_weekly.jsonに出力される。
**特に「市場4-5人気×RL1」「市場6-9人気×RL1」セルの単勝回収率が100%を超えているかが、
ユーザーの仮説（乖離ポケットに妙味がある）の最初の検証ポイント**。

#### テスト
`tests/test_divergence_analysis.py`に`TestRankMatrixAndRoi`を新規追加（4テスト）。
North Starに従い実際のsqlite3.Row（in-memory DB）で検証：
- bucket_statsにtansho_roiが含まれること
- **回収率が勝ち馬の実オッズから厳密計算されることを手計算で検算**
  （4レースの勝ち馬オッズ合計15.0 ÷ 24頭 = 62.5%が全セルの加重合計と一致）
- rank_matrixのセル構造・帯の妥当性
- 「市場5番人気×RL1で勝利」の馬が正しいセル（4-5×1）に入り、回収率に実オッズ
  10.0倍が計上されること
`python -m pytest tests/ -q`は371テスト通過（367+4、回帰なし）。
`index.html`は`node --check`構文検証＋実データ形状でのセル描画ロジック確認済み。

---

### 2026-07-27①：predict()系API呼び出しの網羅監査（2重sigmoidバグの横展開調査）

#### 背景
③で発見した「`raw_margin`という変数名の中身が実は既にsigmoid適用済みの確率で、
直後にsigmoidを再適用していた」バグについて、ユーザーから「以前の森・木・葉の
調査（07-23⑥）ではこのバグを見過ごしていた。この種のバグを調査段階で見つけるのは
難しいか、特定箇所を指定しないと見つけられないか」という質問を受けた。

このバグが07-23⑥で見つからなかった理由を分析した結果、(1) 出力が0.70〜0.73という
「もっともらしい」確率値になり明らかな異常に見えない、(2) sigmoidは単調関数なので
2重適用してもAUC（このプロジェクトの主要評価指標）では検出不可能、(3) 学習側評価
コードと推論側コードが同じ間違い方をしていたため、これまで有効だった「学習時と
推論時のパリティ比較」という調査手法では検出できない、という3点が原因と判明した。
実際に見つかったのはSHAP機能実装中に独立した経路（TreeSHAP寄与度合計）で同じ値を
再計算し、既存コードの値と食い違ったことがきっかけだった。

この経緯を踏まえ、「特定箇所を指定されなくても同種バグを見つけられるか」を検証する
ため、`.predict()`/`.predict_proba()`系API呼び出しを全て洗い出し、各呼び出しが
対応するモデルの目的関数（objective）と整合しているかを1件ずつ確認する監査を実施した。

#### 監査結果
`src/`・`scripts/`配下の`.predict(`/`.predict_proba(`呼び出し全26箇所を確認：

| ファイル | 箇所数 | 判定 |
|---|---|---|
| `engine.py`（残差学習分岐・`get_xgb_rating()`） | 5 | ③で修正済み |
| `engine.py`（ensemble分岐・pace_model・非残差分岐） | 4 | `predict_proba()`使用のため元々安全 |
| `rating_calibration.py`（B2_ndcg/pairwise用） | 2 | `predict_proba()`使用箇所は安全。`model.predict(dmat)`は`rank:ndcg`/`rank:pairwise`目的関数向けで元々正しい実装（下記で実証検証） |
| `dual_model.py`（同上） | 2 | 同上 |
| `compare_models.py`（同上） | 2 | 同上 |
| `calibrate_xgb.py` | 2 | `predict_proba()`および`IsotonicRegression.predict()`で安全 |
| `train_xgb.py`（残差評価・アンサンブル学習） | 6 | 残差評価2箇所は③で修正済み。他は`predict_proba()`で安全 |
| `train_pace_model.py`（多クラス分類器） | 2 | `XGBClassifier.predict()`が離散クラスラベルを正しく返す用途で使用、問題なし |
| `shap_explain.py` | 1 | SHAP計算自体（④で新規実装、正しい） |

**追加バグは見つからなかった。** ただし、`rating_calibration.py`/`dual_model.py`/
`compare_models.py`の3ファイルが依拠している「ランキング目的関数（`rank:ndcg`/
`rank:pairwise`）の`predict()`はsigmoid等の逆リンク変換を適用せず生スコアをそのまま
返す」という前提は、これまで一度も実証検証されておらず**単なる仮定**だった。今回
実際に両目的関数でBoosterを学習させ、`predict()`と`predict(output_margin=True)`が
完全一致することを確認した（xgboostのランキング目的関数はそもそも「確率」という
概念を持たないため、逆リンク変換が存在しない）。

また`1/(1+exp(-...))`のような手動sigmoid適用箇所も全て確認したが、③で修正した
箇所以外（ルールベースフォールバックの`total`変換、Platt scalingの意図的な
sigmoid適用）はいずれも正しい設計だった。

#### この監査手法の一般化可能性（ユーザーの質問への回答）
- **単なるコードリーディングでは見つけにくい**：出力が「もっともらしい値」になり、
  AUCのような主要評価指標では検出できないため
- **ただし「特定箇所の指定」が無くても発見可能**：今回のように「特定の危険な
  API呼び出しパターン（`.predict()`系）を全箇所列挙し、1件ずつ契約を確認する」
  という機械的・網羅的な監査手法は、07-23⑧・07-24①の学習/推論パリティ監査
  （`get_history_from_db()`と`_get_history_before()`のキーを1件ずつ突き合わせる）
  と同じ「網羅列挙型」の調査であり、事前に場所を指定されなくても実行できる
- **今回の発見自体は「独立検証」がきっかけ**：新機能（SHAP分解）が偶然にも
  同じ値を別経路で再計算する構造だったために発見できた。今後も「重要な導出値は
  独立した経路で再計算し一致を確認する」という手法自体を、特定のバグ調査目的
  でなくても意識的に適用する余地がある

#### 対応
`tests/test_residual_learning.py`に`TestRankingObjectivePredictHasNoHiddenTransform`
を新規追加（2テスト、`rank:ndcg`/`rank:pairwise`それぞれ）。今回実証確認した
「ランキング目的関数のpredict()に隠れた変換が無い」という前提を回帰テストとして
固定し、将来xgboostのバージョンアップでこの既定動作が変わった場合に検知できる
ようにした。

#### テスト
`python -m pytest tests/ -q`は367テスト通過（365+2、回帰なし）。

---

### 2026-07-26④：ability_marginをカテゴリ別SHAP寄与度に分解する説明可能性機能を追加

#### 背景
ユーザーから「AIが独自に予想を立てた後、オッズを見て“なぜこの馬は人気があるのに
AIは評価が低いのか”を要因分解して考える仕組みが欲しい。人間的思考でAIでは無理
というかナンセンスなのか？」という相談を受けた。②で残差学習モデルの出力を
`ability_margin`（市場非依存のAI評価）と`base_margin`（市場の評価）に分離済み
だったため、「AI vs 市場のギャップが+か-か」自体は既に計算上存在していた。
足りないのは「なぜそのギャップが出たか」を人間が読める理由に分解する部分で、
これはXGBoostのTreeSHAP（`pred_contribs=True`）を`ability_margin`に適用すれば
実現できると判断し、モデル自体は変更しない解釈専用レイヤーとして実装した。

#### 実装内容
- `src/features/shap_explain.py` 新規作成
  - `FEATURE_CATEGORY_MAP`: 本番の138特徴量列を15カテゴリ（騎手・厩舎・距離適性・
    コース適性・ペース適性・スピード能力・近走成績・クラス適性・血統・斤量馬体・
    枠順・馬場適性・ローテーション・AI自己補正・過去人気推移）に分類する辞書。
    テストで`data/xgb_feature_cols.json`の全列が網羅されていることを確認済み
  - `compute_ability_breakdown(booster, X_pred, feature_cols, base_margin)`:
    `Booster.predict(dmat, pred_contribs=True)`（TreeSHAP）で特徴量ごとの
    寄与度+バイアス項を取得し、バイアス項からbase_marginを差し引いた残りを
    「基準値」カテゴリとして扱うことで、カテゴリ別寄与度の合計が
    `ability_margin`（= raw_margin - base_margin）とほぼ一致するように分解する
- `src/features/engine.py`（`calc_all()` Pass 2、`_XGB_RESIDUAL`分岐）:
  `ability_margin`計算の直後に`compute_ability_breakdown()`を呼び、
  `ability_breakdown`として出力辞書に追加。SHAP計算失敗時は`_warn_shap_breakdown_failure()`
  で1回だけ警告した上で`None`にフォールバックし、予測自体（total/win_prob/
  ability_margin）は止めない設計（`_warn_xgb_inference_fallback`と同じ思想）
- `src/betting/app_json.py`（`_build_horses_list()`）: `ability_breakdown`を
  馬ごとのJSONに追加
- `index.html`: 馬テーブルのRL列に📊バッジを追加。`ability_breakdown`がある馬のみ、
  上位2つの＋要因・上位2つの－要因を`title`ツールチップで表示
  （例: 「AI評価の内訳: 騎手+0.35 / スピード能力+0.21 / 距離適性-0.30 / コース適性-0.12」）

#### なぜ「基準値」カテゴリが必要か
TreeSHAPの`pred_contribs=True`が返すバイアス項は、そのBoosterの学習データ全体に
対する平均的な予測値（グローバルな基準値）であり、推論時に個別instanceへ設定した
`base_margin`とは別物。実際に検証したところ、特徴量ごとの寄与度の単純合計は
`ability_margin`と一致せず、「バイアス項 - 推論時base_margin」を追加の
「基準値」カテゴリとして含めて初めて合計が一致することを確認した
（`sum(feature_contribs) + (bias_term - base_margin) == ability_margin`）。

#### パフォーマンス
本番モデル（138特徴量）で実測したところ、`pred_contribs=True`は1頭あたり約6.8ms
（`output_margin=True`単体の約5.9msとほぼ同等）で、週末ワークフローの規模
（約500頭）でも合計3.4秒程度の増加に収まることを確認済み。GitHub Actionsの
タイムアウトに影響する規模ではない。

#### 学習/推論パリティ
モデル自体・特徴量計算ロジックには一切触れていない。既存の推論経路
（`booster`・`X_pred`・`_bm`）から追加で1個（カテゴリ別分解）を計算するのみ。

#### テスト
`tests/test_shap_explain.py`新規作成（8テスト）。North Starに従い、実際に
`xgb.Booster`を使って検証：
- `FEATURE_CATEGORY_MAP`が`data/xgb_feature_cols.json`（本番の138特徴量）を
  完全に網羅していることを確認（未登録があれば失敗し早期に気づける設計）
- `compute_ability_breakdown()`の寄与度合計が`ability_margin`と一致すること、
  カテゴリへの正しい集約、contrib降順ソートを確認
- `calc_all()`が残差学習モデル時に`ability_breakdown`を含み、非残差モデル時は
  `None`になることを、実際に`save_model()`/`init_engine()`経由でロードした
  本物の残差モデル環境で確認
- `index.html`側は`node --check`でJS構文検証、`node -e`で実際のbreakdownデータを
  使ったバッジ文字列生成を直接確認
- `python -m pytest tests/ -q`は365テスト通過（359+6、回帰なし）

#### 今後
- カテゴリ分類（`FEATURE_CATEGORY_MAP`）は人間が設計した15分類であり、これが
  最適な粒度かはユーザーからのフィードバック次第で調整の余地がある
- 現状は表側のツールチップ表示のみ。乖離分析（`divergence_weekly.json`）と
  組み合わせて「AIが市場と食い違ったレースで、どのカテゴリが的中/外れに
  寄与したか」を週次で集計する拡張も将来検討できる（今回はスコープ外）
- `index.html`側のバッジ生成ロジックは自動テストの対象外（Node.js経由の
  手動クロスチェックのみ。以前から課題の「index.htmlのJSテストハーネス」が
  整備されればここも自動回帰の対象にできる）

---

### 2026-07-26③：残差学習モデルのraw_marginが2重sigmoidになっていたバグを修正

#### 背景
②のability_margin（AIが市場と違う評価をした理由を騎手・距離適性等のカテゴリで
説明するSHAP機能）を実装・検証している最中、`ability_margin + base_margin`から
再構成した確率が、独立に計算した期待値と一致しない現象に遭遇した。掘り下げた
結果、②自体とは別の、より根の深い既存バグを発見した。

#### 🔴 発見：`raw_margin`という変数名の中身が実は既に確率で、直後にsigmoidを再適用していた
`engine.py`の残差学習分岐（2026-07-12導入、`_XGB_RESIDUAL`分岐）:
```python
raw_margin = float(_XGB_FUKUSHO_MODEL.predict(_dmat)[0])   # ← 実際は確率が返る
prob = 1 / (1 + math.exp(-raw_margin))                      # ← さらにsigmoidを適用
```
XGBoostの`Booster.predict()`は`output_margin=True`を指定しない限り、
`binary:logistic`の逆リンク関数(sigmoid)を適用済みの**確率**を返す。`base_margin`を
DMatrixに設定していてもこの既定動作は変わらない。つまり`raw_margin`という変数名
にもかかわらず中身は既にsigmoid適用済みの確率であり、その後さらに`math.exp(-raw_margin)`
でsigmoidをもう一度掛けていた。同一パターンが`src/tools/train_xgb.py`の残差モデル
評価コード（val_prob計算、旧モデル比較）と、`engine.py`の未使用関数`get_xgb_rating()`
にも存在した。

実際の本番モデル（`data/xgb_fukusho_model.pkl`）で検証した数値:

| base_margin | 既存コードのprob（2重sigmoid） | 本来のprob |
|---|---|---|
| -1.5 | 0.6969 | 0.8328 |
| 0.0 | 0.7225 | 0.9571 |
| +1.5 | 0.7291 | 0.9901 |

base_marginを3ポイント動かしても既存コードの出力は0.70〜0.73にほぼ張り付いたまま
だった。本来は0.83〜0.99まで動くはずの値が大きく圧縮されていた。

#### 気づかれなかった理由
- sigmoidは単調増加関数のため2重に掛けても**順位（AUC）は変わらない**→AUC評価
  では検出不可能
- `_XGB_CALIBRATOR`（Isotonic回帰）が有効な場合、Isotonicは順位さえ合っていれば
  入力のスケールに関係なく正しい確率へ較正し直せるため、`cal_prob`（表示用複勝
  確率）は較正層で結果的にある程度救われていた可能性が高い
- ただし`rating`（`= raw_margin`、`bet_optimizer.py`のGumbelシミュレーション入力）と
  ②で追加した`ability_margin`は較正層を経由しないため、このバグの影響を直接受ける

#### 対応
- `src/features/engine.py`: 残差学習分岐と`get_xgb_rating()`の両方で
  `_XGB_FUKUSHO_MODEL.predict(_dmat, output_margin=True)`に修正
- `src/tools/train_xgb.py`: 残差モデルの評価コード（新モデル・旧モデル比較の
  両方）で同様に`output_margin=True`を追加
- `dual_model.py`/`rating_calibration.py`/`compare_models.py`の同様の
  `model.predict(dmat)`呼び出しも確認したが、これらは`rank:ndcg`/`rank:pairwise`
  （ランキング目的関数、B2_ndcg/pairwiseモデル）向けで、ランキング目的関数は
  `predict()`にsigmoid変換が無く元々正しい実装だったため対象外

#### 影響範囲
- **`cal_prob`/`win_prob`（表示用確率）**: 較正層（Isotonic）が入っていれば実害は
  限定的だったと推測されるが未検証。次回フォワードデータで較正曲線を確認すること
- **`rating`（Gumbelシミュレーション入力、📊EV買い目）**: 較正層を経由しないため
  直接影響。修正によりratingのスケールが本来の（より広い）log-odds幅に戻るため、
  `bet_optimizer.py`の`DEFAULT_GUMBEL_RATING_T = 2.5`は2026-07-06に**旧（非残差）
  モデルのtotal-5.0スケール**を基準にフォワードデータで最適化された値であり、
  今回のスケール修正後の残差モデルに対して再検証されていない。次回フォワード
  データ蓄積後、Gumbel買い目のlog-loss/ECEを確認し、必要ならT再校正を検討すること
  （**要フォローアップ**。今回は「今変えると二重補正リスク」の教訓に従い、
  Tの値自体は変更していない）
- `get_xgb_rating()`は本番のどこからも呼ばれていない未使用関数のため実害なし
- モデルの学習内容自体（重み・分割点）には影響なし。あくまで推論・評価コードの
  読み出し方のバグ

#### 学習/推論パリティ
モデル自体は変更していない。学習時の評価コード（`train_xgb.py`）と推論時の
コード（`engine.py`）の両方で同じ修正を適用したため、パリティは保たれる。

#### テスト
`tests/test_residual_learning.py`に`TestRawMarginTrueLogOdds`を新規追加（2テスト）。
North Starに従い、実際の`xgb.Booster`で検証：
- `test_cal_prob_matches_independent_output_margin_prediction`: engine.pyの内部
  実装を経由せず、テスト内で独立に`Booster.predict(dmat, output_margin=True)`を
  呼んで得た"真の"marginから計算した確率と、`calc_all()`が返す`cal_prob`が一致
  することを確認。②のテスト（`TestAbilityMarginExposure`）はengine.py内部の値
  同士を比較する自己整合性チェックのため今回のバグを検出できなかった。本テストは
  engine.pyの外側で独立に計算した期待値と比較するため、修正前のコードに対しては
  実際に失敗することを確認済み
- `test_cal_prob_spans_wide_range_across_popularity`: 1番人気と最下位人気で
  `cal_prob`が十分に離れることを確認（2重sigmoidバグ再発時の圧縮を検知する
  ガード）。修正前のコードに対しては実際に失敗することを確認済み

既存の`python -m pytest tests/ -q`は359テスト通過（357+2、回帰なし）。

#### 今後
- **要フォローアップ（優先度高）**: `DEFAULT_GUMBEL_RATING_T = 2.5`の再検証
  （上記「影響範囲」参照）
- ②で実装したability_marginの計算式自体（`raw_margin - base_margin`）は変更
  していない。今回の修正で`raw_margin`が正しく計算されるようになったことで、
  ability_marginも意図通りの値になる

---

### 2026-07-26②：直前オッズ取得時に勝率・複勝率をability_marginで再同期する仕組みを追加

#### 背景
ユーザーから「前日の薄い市場で情報取得した時点でA・B2頭の実力馬がいて、その時点でAに
買いが集中していたとする。その後Aの調子が悪化しBに人気が偏ってきたら、直前オッズ取得は
オッズだけ更新するので、AIの勝率は生成時点の高いまま・オッズだけ下がった見せかけのエッジが
出るのでは？」という指摘を受けた。実際にコードを確認したところ、`index.html`の
`updateOddsAndEV()`は従来オッズ（`h.odds`/`h.fukusho_odds`）を更新するのみで、
`tan_pct`（勝率）・`fuku_pct`（複勝率）は`to_app_json()`生成時点（前夜〜当日朝）の値の
まま据え置かれていた。指摘の通り、勝率が古いまま・オッズだけ新しいという「時点がずれた
EV」を表示し続ける構造的な問題があった。

合わせて「複勝確率だけでなく1着（単勝）の予測精度自体を上げたい」という要望も受け、
`train_xgb.py`を確認したところ`y_train = train_df['is_fukusho']`で本番モデルは
**複勝(3着内)の二値分類器として学習**されており、単勝確率(`tan_pct`)は
`softmax_probs(total, temperature=3.5)`というフィールド内相対化のみで導出され、
専用の単勝学習シグナルが一切存在しないことを確認した。この単勝精度向上（ランキング
モデル化）は大掛かりな変更でありフォワードデータ（残差モデル本番化は7/14からのみ、
現状69〜100レース強）ではまだ判断材料が足りないため、今回は着手を見送り、
まず即効性のある①（時点ずれ解消）のみ実装する方針でユーザーと合意した。

#### 実装内容：ability_margin（市場非依存のAI能力スコア）の公開とクライアント側再同期
残差学習モデルは`logit(p) = base_margin(人気由来) + f_AI(非市場特徴量)`という構造で、
`base_margin`は人気順位のみから決まる（Zipf近似）。この構造を利用し、
**`raw_margin - base_margin`（＝人気に依存しない「純粋なAI能力スコア」）を
`ability_margin`として推論結果に追加で保持**する設計とした。

- `src/features/engine.py`（`calc_all()` Pass 2、`_XGB_RESIDUAL`分岐）:
  `ability_margin = raw_margin - _bm`を計算し、出力辞書に`'ability_margin'`を追加
  （非残差モデル・ルールベースフォールバック時は`None`）
- `src/betting/app_json.py`（`_build_horses_list()`）: `ability_margin`を馬ごとのJSONに追加
- `index.html`: `_baseMarginFromPopularity()`（Zipf近似のbase_margin計算）・
  `_softmaxTotals()`（T=3.5固定softmax）・`_harvilleTop3()`（Harville公式、
  `calc_harville_probs()`の移植）の3つの純粋関数を新設。`updateOddsAndEV()`を、
  レース内の全馬が`ability_margin`を持つ場合（＝残差学習モデル）にのみ、
  直前オッズから再算出した新しい人気順位で`base_margin`を引き直し、
  `ability_margin + 新base_margin` → softmax → Harvilleという経路で
  `tan_pct`/`fuku_pct`/`ev`を再計算するよう変更。**1頭でも`ability_margin`が
  欠けていれば（非残差モデル運用時等）レース全体で再同期をスキップし、
  従来通り（勝率は生成時点のまま・オッズだけ更新）にフォールバック**する
  安全設計とした。

#### ⚠ 意図的な近似（本来のtotalとの差異、要理解）
`engine.py`の`total`（softmaxに渡される最終スコア）は、Pass 2のXGB生シグモイド確率
(`raw_prob*10`)だけでなく、その後に**レース内の相対順位に基づくブレンド
（`f_relative_score`、重み10%）・ペースボーナス・エラータグ補正係数(`et_factor`)**が
順に適用された値である。今回のクライアント側再同期は`ability_margin`（=XGB生
シグモイド確率相当、`cal_prob`と等価）のみを人気の変化に応じて引き直しており、
これら後段の補正（いずれも市場情報とは無関係な固定値）までは再現していない**近似**である。
後段補正は通常小さく（相対ブレンド10%・補正係数は概ね1.0付近）、かつオッズ変化とは
無関係な定数であるため、「時点のずれたEVを表示し続ける」という本来の問題（本セッションの
主眼）は解消されるが、直前オッズ取得後の`tan_pct`/`fuku_pct`は完全な backend 再計算とは
わずかに異なりうる。今後もし乖離が無視できないと分かった場合は、`f_relative_score`等の
後段補正も含めてJS側に移植することを検討する（ただし相対ブレンドはレース内全馬の
総入れ替えが必要でやや複雑）。

#### 本来の課題（07-24⑤）との関係
2026-07-24⑤で記録した「残差学習モデルのbase_marginが学習時(確定人気)と推論時
(発売直後の薄い人気)で情報の成熟度が違う」という課題は、**モデルの学習構造そのもの**に
起因する別の問題であり、今回の対応では解消していない（引き続き「残っている課題」表に
深刻度高として残置）。今回解決したのは、その一段下流にある「一度生成した予想の勝率が、
オッズだけ後から更新されると時点不整合を起こす」という表示・運用レイヤーの問題であり、
両者は関連するが別物である。

#### 学習/推論パリティ
今回はモデル自体・特徴量計算ロジックには一切触れていない。`ability_margin`は既存の
推論経路（`raw_margin`と`_bm`）から追加で1個計算するのみで、既存の`total`/`cal_prob`等の
計算過程・値は変更していない。

#### テスト
`tests/test_residual_learning.py`に`TestAbilityMarginExposure`を新規追加（2テスト）。
North Starに従い、実際に`xgb.Booster`を`.save_model()`で保存し`init_engine()`経由で
ロードした本物の残差モデル環境で検証：
- `test_ability_margin_reconstructs_cal_prob`: `ability_margin + base_margin(popularity)`
  から`cal_prob`（キャリブレーター未ロード時は生シグモイド確率と一致）が厳密に
  再構成できることを確認。**`total`ではなく`cal_prob`を検証対象にした**理由は
  上記「意図的な近似」の通り、`total`は後段補正で厳密再現できないため
- `test_ability_margin_none_without_residual_model`: 非残差モデル時に`ability_margin`が
  `None`になることを確認

また`index.html`側のJS実装は、同じ入力（`ability_margin`・人気順位）に対して
Python参照実装と完全に同じ`totals`/`win_probs`/`top3`を返すことを、一時スクリプトで
浮動小数点精度まで一致することを直接確認した（`node --check`によるJS構文検証も実施）。
既存の`python -m pytest tests/ -q`は357テスト通過（355+2、回帰なし）。

#### 今後
- 単勝(1着)専用の学習シグナルを持つモデル（例: `rank:ndcg`の残差学習対応版）は
  今回スコープ外。残差モデル本番化(7/14)からのフォワードデータがまだ少なく
  （kpi_weekly.json・divergence_weekly.jsonとも数週分のみ）、着手は次回以降の
  データ蓄積を待つ
- クライアント側再同期のJS関数群（`_baseMarginFromPopularity`/`_softmaxTotals`/
  `_harvilleTop3`）は自動テストの対象外（Node.js経由の手動クロスチェックのみ）。
  以前から課題として残っている「index.htmlのJSテストハーネス」（2026-07-25②で
  言及）が整備されればここも自動回帰の対象にできる

---

### 2026-07-26①：refresh_today()生成のlatest.jsonが実在しない翌日を表示するバグを修正

#### 背景
③で追加したrefreshモードを実際に本番実行した後、ユーザーがアプリのスクリーンショット
（日曜9:26時点、中京・新潟・札幌の3場・14頭立てのレース）を提示し「本日のレースです」
と述べたが、画面には「7月27日(月)」と表示されていた。JRA公式カレンダーを確認したところ
その週は7/25(土)・7/26(日)のみが開催日で、7/27(月)という開催日は実在しなかった。

#### 🔴 発見：`to_app_json()`の表示日付+1日ロジックが「当日実行」を考慮していなかった
`to_app_json()`の`display_dt`計算は
`jst_now + timedelta(days=1) if day_type in ('saturday', 'sunday') else jst_now`
となっており、「前夜に実行して翌日ぶんの予想を生成する」既存フロー
（`friday_predict.py`・`predict_next_day()`）専用の前提だった。③で追加した
`refresh_today()`は**当日の朝**に実行し`day_type`をその日の曜日から
`'saturday'`/`'sunday'`と推定するが、`to_app_json()`側はその違いを区別できず
`day_type`が`'sunday'`であれば無条件に+1日していたため、日曜朝に実行した
`refresh_today()`が生成した`latest.json`は実在しない月曜の日付を表示していた。
中身のレースデータ自体（当日の中京・新潟・札幌、オッズ・予想）は正しく、
表示ラベルのみが誤っていた。

#### 対応
`to_app_json()`に`same_day`引数（デフォルト`False`、後方互換）を追加。
`True`の場合は`jst_now`をそのまま表示日付に使い、+1日しない。
`refresh_today()`の`to_app_json()`呼び出しに`same_day=True`を追加。
`friday_predict.py`・`predict_next_day()`側の呼び出しは変更していない
（既存の「前夜生成→翌日表示」の挙動を維持）。

#### テスト
`tests/test_app_json_data_quality.py`に2テスト追加
（`same_day`省略時は従来通り+1日されること、`same_day=True`ではjst_now
そのものの日付になり実在しない翌日にならないことを確認）。既存の
`tests/test_weekend_refresh.py`の`refresh_today`テストが`to_app_json`呼び出しに
`same_day=True`が渡されていることを検証するよう更新（フェイク関数が新引数を
受け取れず落ちる状態だったため合わせて修正）。
`python -m pytest tests/ -q`は355テスト通過（353+2、回帰なし）。

---

### 2026-07-25④：find_r01_odds()の失敗内訳を診断ログに残すよう修正

#### 背景
③のrefreshモードをマージしユーザーから「今日のレース情報取得時、すでに馬券は
売り出されていたはずなのにオッズが0件だったのはなぜか」という指摘を受けた。
「発売直後で薄いから」という説明では説明がつかない鋭い指摘だったため、
実際のGitHub Actions実行ログで裏付けを取った。

#### 調査結果（実ログで確認）
2026-07-25 19:06 JST開始の週次実行では、新潟・中京・札幌の3会場すべてで
`find_r01_odds()`（オッズページaccessO.htmlのCNAME suffixを0〜255で総当たり
探索する関数）が「オッズR01: 未発見」となり、結果としてオッズ反映0頭/34Rに
なっていた。一方、全く同じ関数が1週間前（2026-07-18、福島・小倉・函館）は
3会場とも正常に成功し「オッズ反映: 481頭/35R」だった。同じコードが同じ
時間帯（週次実行）で明暗を分けたことになる。

同じ実行ログで、出走表ページ側の探索（`find_r01_shutuba`）も通常の高速経路
ではなく低速のフォールバックスキャンに落ちていた形跡があり、その時間帯の
JRA側のレスポンスが普段と異なっていた可能性を示唆しているが、
`find_r01_odds()`は失敗した個々の試行の詳細（パラメータエラーだったのか、
テーブルが無かったのか、通信例外だったのか）を`except Exception: continue`
で無条件に握りつぶしており、256回の試行がどう失敗したかの痕跡が一切
残らない実装だった。そのため「JRA側が重かった」という推測はできても、
確定的な原因特定はログの情報量不足でできなかった。

#### 対応
`find_r01_odds()`に、パラメータエラー/テーブルなし/例外の内訳カウンタと
直近の例外内容を追加。256件全て不一致で終わった場合のみ、内訳を1行の
警告ログとして出力する（成功時の挙動・戻り値は一切変更していない）。
次回同じ現象が起きた際に、原因の切り分け（JRA側の遅延か、タイムアウトか、
別の問題か）が実際のログから可能になる。

#### テスト
`tests/test_scraper.py`に3テスト追加（成功パスの挙動が変わらないこと、
全滅時にパラメータエラー内訳がログに残ること、通信例外がパラメータエラーと
区別してカウントされ例外内容もログに含まれることを確認）。
既存の`python -m pytest tests/ -q`は353テスト通過（350+3、回帰なし）。

#### 今後
今回はログ追加のみで、`find_r01_odds()`自体のリトライ・フォールバック機構は
未実装のまま（`find_r01_shutuba()`が2026-06-21に獲得したような多段フォール
バックは無い）。次に本当に原因が判明したら、その原因に応じた恒久対策
（タイムアウト延長、リトライ、代替スキャン等）を追加で検討する。

---

### 2026-07-25③：当日オッズによる予想更新モード refresh を追加、毎週土日朝に自動実行

#### 背景
②で追加したdata_qualityで「オッズ反映0頭」の障害自体は可視化できたが、ユーザーから
「発走前に間に合うならその場で直したい。あなたにその技術がないなら諦める」という
指摘と、「前日の薄いオッズが予想の土台のままでは、直前オッズ取得ボタンを押しても
良い予想にならないのでは」という根本的な懸念を受けた。

調査の結果、②で自分がGitHub Actions APIから直接`workflow_dispatch`で叩いた
`weekend.yml --mode saturday --force`は実は無意味だったと判明した。実行ログを
確認したところ、`weekend.py`はmode=saturdayの場合「本日の結果取得」→
「翌日の予想生成（`predict_next_day`、`jst_now + 1日`をターゲットにする設計）」
という順で動く。日曜(7/26)当日に実行すると「本日結果取得」は当然0件（レース未実施）、
「翌日予想」は月曜(7/27、非開催日)を対象にしてしまい0レースで`latest.json`は
一切更新されていなかった（今朝の日曜予想は手つかずのまま）。

また、アプリの「🔄 直前オッズ取得」ボタンはレースごとにブラウザから直接JRAの
オッズページを叩いて**画面表示のみ**をその場で更新するクライアント側処理で、
GitHub Actionsも`calc_all()`の再実行も伴わない。つまりRL順位・勝率（残差学習
モデルのbase_marginの土台であるpopularity由来）は前夜生成時点の薄いオッズの
ままで、直前オッズ取得ボタンをいくら押しても**表示オッズだけ動いてRL順位・
勝率は変わらない**（ユーザーが最初に指摘した「オッズとRL順位がほぼ同じ＝人気の
コピー」の直接の原因）。これは2026-07-24⑤で記録済みの構造的課題（学習時=確定
人気、推論時=前夜の薄い人気）の即効性のある症状の一つだった。

#### 対応：当日・発走前に予想を再生成する refresh モードを新設
`scripts/weekend.py`に`refresh_today(sess, hist_path, avg_bias, jst_now)`を追加。
既存の`predict_next_day()`と同じ実績のある処理列（`fetch_races_on_date`→
`fetch_odds_map`→`apply_odds_to_races`→`calc_all`→`select_quality_races`→
`to_app_json`）を、「翌日」ではなく**当日**（`jst_now`自身の日付）に対して実行し、
その時点の（前夜より票が積み上がった）オッズを土台にpopularity・base_margin・
RL順位・勝率を計算し直して`latest.json`を更新する。weekdayから`day_type`
（土曜/日曜）を自動判定。

**keiba.dbの実績記録（bets/bet_simulation/races）は書き換えない設計**とした。
`save_bets_db`は同一race_idへの再実行時、買い目の馬番が前回と違う場合は
新規行として追加されるだけで古い行を消さず、`log_bet_simulation`に至っては
重複排除が全く無い（`bet_simulation`テーブルへの毎回無条件INSERT）。
これらを refresh から呼ぶと同一レースの実績データが二重・三重に積み上がり
ROI集計を汚染するリスクがあった。`race_predictions`はUNIQUE制約による
安全な上書き（INSERT OR REPLACE相当）が既に効いているため、こちらだけは
refreshからも更新する。表示専用の`to_app_json`は元々`make_bets()`を内部で
呼び直して表示用の買い目を組み立てる設計のため、DBへの書き込みを経由せずに
最新の買い目を表示できる。

`main()`に`--mode refresh`を追加（既存のsaturday/sundayモードの処理は一切
変更していない）。`.github/workflows/weekend.yml`:
- workflow_dispatchの`mode`選択肢に`refresh`を追加（手動での即時実行用）
- `schedule: cron: '0 23 * * 5,6'`を追加（08:00 JST 毎週土曜・日曜に自動実行。
  多くのJRAレースの発走時刻(概ね10時前後)より前、かつ当日の投票がある程度
  積み上がった時間帯を狙った設定）
- 「Run weekend script」「Commit & Push」の両ステップで`github.event_name`が
  `schedule`かどうかを見て、schedule起動時はモードを`refresh`に固定するよう分岐

#### 学習/推論パリティ・North Starとの整合
今回はモデル自体・特徴量計算ロジックには一切触れておらず、既存の推論パイプライン
（`calc_all`等）を**いつ・どのオッズデータで実行するか**というスケジューリングの
変更のみ。「モデル変更は一旦停止」の合意とは別軸の対応であり抵触しない。
なお、本対応は前夜生成より当日の情報成熟度を上げる改善であって、2026-07-24⑤の
根本課題（学習時=確定人気 vs 推論時=馬券発売直後の人気、という情報成熟度のズレ
そのもの）を完全に解消するものではない（当日朝でも最終確定オッズよりは未成熟）。
根本解決には学習側base_marginの再定義かモデル再学習が必要で、これは引き続き
今後の課題として残る。

#### テスト
`tests/test_weekend_refresh.py`新規作成（5テスト）。モジュール内の関数を
monkeypatchし実ネットワークアクセス無しで検証：weekday!=土日ならスキップ・
レース0件ならlatest.jsonを書き換えない・`save_race_db`/`save_bets_db`/
`log_bet_simulation`が一切呼ばれないこと（呼ばれたら例外を送出するダミーで
検証）・day_typeが土日それぞれ正しく推定されること・`main()`のargparseに
`refresh`が追加されmain()内で`refresh_today`が呼ばれることをソースレベルで
確認。`.github/workflows/weekend.yml`は`python3 -c "import yaml; ..."`で
構文検証。既存の`python -m pytest tests/ -q`は350テスト通過（345+5、回帰なし）。

#### 今後
次回のColab再学習時、`race_predictions`にrefreshモードで上書きされた行が
複数回分蓄積される点に注意（同一レースについて前夜生成分→当日refresh分と
上書きされていくため、最終的に残るのは最後に実行されたrefreshの値。学習データ
抽出時は特に問題にならない設計だが、想定と異なる挙動に気づいた場合はここを
参照すること）。

---

### 2026-07-25②：オッズ取得率・レース取得失敗をlatest.jsonに可視化するdata_qualityセクションを追加

#### 背景
直前セッション（①）で新潟R11のparse失敗を修正した際、ユーザーから同時に「日曜レース
予想で直前オッズ取得後もオッズとRL順位がほぼ変わらない（人気のコピーに見える）」
「現時点（前日取得だけ）での推奨レース0」という報告も受けていた。GitHub Actions
の実行ログを直接確認したところ、当該実行では専用オッズページからのオッズ取得が
全馬0頭（`オッズ反映: 0頭/34R`）で完全に失敗しており、推奨レース0件はその直接の
結果だった。この種の障害（オッズ取得の全滅、個別レースのparse失敗）はこれまで
コンソールログにしか残らず、latest.json経由でアプリ側からもユーザー自身からも
一切気づく手段が無かった。①のバグ修正だけでは「次に別の原因で同じ現象が起きても
また同じやり取りを繰り返す」ため、再発時に気づけるようにする仕組みを追加した。

#### 対応
1. `src/scraper/jra_scraper.py`: `fetch_races_on_date()`の戻り値を
   `all_races`単体から`(all_races, failures)`のタプルに変更。
   `failures`は`[{'racecourse': str, 'race_num': int, 'reason': str}, ...]`形式で、
   **ページ自体が取得できた（＝そのレースは実在する）のにparseに失敗したもの
   のみ**を記録する。障害レースのスキップ、および該当venueがその日12レース
   未満で該当レース番号のページ自体が存在しないケース（suffix探索を尽くしても
   soupがNoneのまま）は、`fetch_results()`（結果取得側）の既存の挙動に合わせ、
   意図した挙動として含めない（多くのvenueは12レースに満たない日があり、
   これを全部「失敗」扱いにするとfailuresが常に大量のノイズで埋まってしまうため）
2. `src/betting/app_json.py`: `to_app_json()`に`odds_updated_count`・
   `parse_failures`パラメータを追加（両方省略可、後方互換）。
   `races_all`の全出走頭数に対する`odds_updated_count`の比率を
   `data_quality.odds_coverage`として、`parse_failures`をそのまま
   `data_quality.parse_failures`として出力JSONに追加
3. `scripts/weekend.py` / `scripts/friday_predict.py`: 両方とも
   `fetch_races_on_date()`の戻り値をタプルとして受け取り、`to_app_json()`
   呼び出しに`odds_updated_count=n_odds, parse_failures=parse_failures`を
   渡すよう更新
4. `index.html`: `data_quality.odds_coverage`が50%未満、または
   `data_quality.parse_failures`が空でない場合に、予想一覧の最上部
   （結果バナーの上）にオレンジの警告ボックスを表示するようにした
   （「⚠ 直前オッズ取得率0%（オッズ未反映のまま予想が生成された可能性）」
   「⚠ 取得失敗レース: 新潟R11(parse失敗)」等）

#### 学習/推論パリティ
今回は表示専用の変更で、予想の計算ロジック自体（`calc_all`等）には一切触れていない。

#### テスト
- `tests/test_scraper.py`に2テスト追加。実際のネットワーク呼び出しを避けるため
  `_try_fetch_shutuba`/`_parse_shutuba`/`get_kaisai_on_date`等をmonkeypatchし、
  (a)ページ自体が存在しない大半のレース番号はfailuresに記録されず、実際に
  parseに失敗したレースのみ記録されること、(b)障害レースのスキップは
  failuresに記録されないことを確認
- `tests/test_app_json_data_quality.py`新規作成（6テスト）。`calc_all`を
  空リストにモックしdata_quality計算自体を分離してテスト。オッズ反映率の
  計算（0%・部分・省略時None・0頭でのゼロ除算回避）、parse_failuresの
  受け渡し・デフォルト空リストを確認
- `node --check`でindex.htmlのJS構文を検証、実際の障害シナリオの数値
  （odds_coverage=0, parse_failures=[新潟R11]）でJS側の警告文生成を
  `node -e`で直接確認
- 既存の`python -m pytest tests/ -q`は345テスト通過（337+8、回帰なし）

#### 今回は対応していない改善案（別スコープ）
調査時に合わせて2つの改善案を検討したが、今回はスコープ外として見送った。
1. **オッズ取得が0件だった場合の自動リトライ**：GAS側の「直前オッズ取得」
   ボタンは`force`パラメータをworkflow_dispatchに転送しない設計のため、
   当日分の再生成を自分でトリガーする手段が無い（今回はGitHub Actions APIで
   直接`workflow_dispatch`を叩いて回避した）。恒常的な対応にはGAS
   （`gas/triggerWorkflow.gs`）の`force`転送、またはワークフロー側の
   オッズ0件検知時の自動再試行ロジックが必要
2. **index.htmlのJSテストハーネス**：`recalcGumbelBets()`のcal_prob/tan_pct
   混同バグ（2026-07-24④）のように、数ヶ月気づかれないJS計算バグが今後も
   起こりうる。既知の入出力ペアに対するNode.js経由の回帰テストを整備すると
   再発防止になるが、今回は着手していない

---

### 2026-07-25①：parse_header()が「メートル（芝・右）」の隣接形式以外で芝ダ判定できずレース全体のparse失敗になっていたバグを修正

#### 背景
ユーザーから「日曜レース予想で新潟11レースが未取得、昨日も同じ現象があった」という報告を受け、
週次ワークフロー（2026-07-25 19:06 JST開始、「土曜結果＋日曜予想」）の実行ログをGitHub
Actions APIで直接確認した。

#### 🔴 発見：`parse_header()`の芝ダ判定が「メートル（芝・右）」の隣接パターン1つしか試していなかった
実行ログに以下の警告と、そのレース自体のparse失敗が出力されていた。

```
⚠ surface判定失敗: 2026年7月26日（日曜） 2回新潟2日 発走時刻： 18時00分 3歳以上1勝クラス...
R11: parse失敗 (馬なし or 例外) suffix=8C
```

`src/scraper/parser.py`の`parse_header()`は距離・芝ダ・回りを1本の正規表現
`([\d,]+)\s*メートル\s*[（(]\s*([芝ダ])[^）)]*([右左])`でまとめて抽出しており、
「メートル」の直後に「（芝・右）」のような括弧書きが隣接していない見出し文
（例：「コース：1,000 メートル ダート 右回り」のような括弧なし表記）だと
即座に諦めて`surface='不明', distance=0`にフォールバックしていた。この状態の
まま後続処理に渡るとレース全体が「parse失敗」になっていた。

一方、結果ページ側のパース（`jra_scraper.py`）では同じ`parser.py`内の
`_detect_surface()`という、障害明示→「メートル（芝/ダ」隣接→「芝NNNN」/「ダNNNN」
→単独の「芝」/「ダート」記載、と4段階でフォールバックする堅牢な判定関数が
既に使われていたが、出馬表（予想対象レース）側の`parse_header()`はこれを
再利用しておらず、隣接パターン1本槍のままだった。

#### 対応
`parse_header()`の主パターンが失敗した場合、`_detect_surface()`（既存の堅牢な
多段判定）で芝ダを判定し、距離は`([\d,]+)\s*メートル`単独の正規表現、回りは
`([右左])`単独の正規表現で別々に拾うフォールバックを追加した。3つとも拾えない
場合のみ、従来通り`不明`/`0`にフォールバックし警告を出す。

#### テスト
`tests/test_scraper.py`に1テスト新規追加（実際のログに出た見出し文の形（メートルと
芝ダ表記が隣接しない括弧なしパターン）を再現し、distance=1000・surface='ダート'・
direction='右'が正しく拾えることを確認）。修正前のコードでは実際に
`distance=0, surface='不明'`になり失敗することを確認済み。全337テスト通過。

#### 今後
今回のフォールバックで「メートル・芝ダ・回りが同じ括弧内に隣接していない」パターンは
解消したが、実機の完全な生HTMLは未確認のため、今回のフォールバックでも拾えない
別パターンが存在する可能性はある。同じ現象（特定レースだけparse失敗）が再発する
場合は、そのレースの実機HTMLを確認して追加のフォールバックを検討すること。

---

### 2026-07-24⑤：残差学習モデルのbase_marginが学習時（確定人気）と推論時（発売直後の薄い人気）で情報の成熟度が違う問題を発見・記録

#### 背景
直前オッズ取得後にEV買い目・勝率表示が食い違って見える件のやり取りの中で、
「前日に取得した情報が予想の土台になるのは腑に落ちない。馬券売り出し直後、
まだ一票も買われてない馬もいるのでは」という指摘を受けた。市場ベースライン
（残差学習モデルのbase_margin）が実際いつの時点のオッズを使っているかをコードで
確認したところ、指摘の通りの構造的な問題が見つかった。

#### 🔴 発見：base_marginに使う人気順位が、学習時は確定人気・推論時は発売直後の薄いオッズ由来
`engine.py`の`calc_all()`は`popularity`を`win_odds`の安い順に並べて導出している：

```python
# popularity を win_odds 順位から導出（低オッズ=1番人気）
for _rank, _h in enumerate(
        sorted(_horses_in, key=lambda x: x.get('win_odds') or 999), 1):
    if not _h.get('popularity') or _h.get('popularity') == 99:
        _h['popularity'] = _rank
```

この`win_odds`は`jra_scraper.py`のコメントが明言する通り、金曜夜（土曜レース予想生成時）
時点の、馬券発売直後のまだ薄いオッズである：

```python
# 出馬表ページ(_parse_shutuba)にはオッズが載らない（特に前日=金曜）ため、
# 専用オッズページ(fetch_odds_map)で取得した値を各馬に反映する。
```

一方、この`popularity`は残差学習モデルの`base_margin`（予測の出発点、単なる1特徴量
ではなく最も重い役割）に使われている。学習データ側（`build_training_data.py`）は
`horse_history.popularity`をSELECTしているが、これは結果ページ由来の**レース確定時点の
最終人気**であり、CLAUDE.mdには既に2026-07-06時点で
「f_popularity: 現走人気（**予測時=朝オッズ由来、学習時=確定人気**）」と記載されていた
（当時は事実として記録されたのみで、残差学習のbase_marginとしての重み・影響への
言及は無かった）。

まとめると：
- **学習時**：確定人気（何万票もの投票を経た、情報的に成熟した市場コンセンサス）
- **推論時（本番）**：金曜夜・発売直後の薄いオッズ順位（初期のわずかな投票のみ）

モデルは「成熟した確定人気を土台にする」前提で学習されているのに、本番では
「発売直後の薄い人気」を土台として渡されている。これは本セッションで繰り返し
発見してきた学習/推論パリティ違反と同種の問題だが、今回は最も重要な入力
（base_margin）で発生している。

#### なぜ重要か（市場KPI・乖離分析の解釈への示唆）
市場ベースラインKPI（累積258レース、市場優位）や乖離分析（AI・市場が食い違うと
市場が勝つ回数がAIの2.3倍）は「AI vs 市場」の対決として解釈してきたが、AIが
土台にしている「市場の意見」自体が、比較対象の結果が確定する時点の本物の市場
（最終オッズ）よりそもそも古く未成熟な情報である可能性がある。だとすれば
これらの観測結果は、AIの手法自体の欠陥というより「金曜夜の薄い市場情報 vs
レース直前の成熟した市場情報」という不公平な比較を一部反映している可能性がある。

#### 対応
今回はコード変更・記録のみ。残差モデルの核心部分（base_margin）に関わる変更で
影響範囲が大きく、今朝合意した「モデル変更は一旦停止、フォワードデータ蓄積を
優先」との整合を優先し、今すぐの修正は見送った。「残っている課題」表に
深刻度**高**として新規追加し、次回モデル見直し時の最有力調査候補として記録した。

#### 今後の調査の方向性（着手する場合）
- 学習側のbase_marginを「確定人気」ではなく「学習データの各レースについて、
  同程度の情報成熟度の時点（例：発売開始から同じ経過時間、または前日夜時点の
  過去の直前オッズスナップショットがあればそれ）」に揃えて再学習し、AUC・
  市場KPIがどう変化するか比較する
- あるいは逆に、推論側でより成熟した時点のオッズ（土曜朝、日曜レースなら
  前日の情報を待つ等）を使えるよう運用フローを見直す
- どちらも大きな変更のため、着手する場合はユーザーと相談の上で行うこと

---

### 2026-07-24④：index.htmlのEV買い目再計算がcal_prob(複勝確率)を単勝勝率として誤用していたバグを修正＋表のEV列の見せ方を修正

#### 背景
ユーザーがアプリの実機スクリーンショットを提示し、「直前オッズ取得後のノシェリーのEVが、最初の勝率とは違う勝率で計算されているのでは」「勝率が平たいからか低人気の馬が軒並みEVありと出てしまう、おかしくないか」という2点を指摘した。

#### 🔴 発見①：`recalcGumbelBets()`（直前オッズ取得時のEV買い目再計算）が単勝・馬連でcal_prob（複勝確率）を単勝確率として使っていた
`index.html`の`recalcGumbelBets()`内、単勝候補の確率計算が

```js
const prob = (h.cal_prob || h.tan_pct / 100);
```

となっており、JSの`||`は左辺が真値なら常にそちらを採用するため、常に`cal_prob`（複勝＝3着内確率、約40〜50%）が使われ、`tan_pct`（単勝勝率、約10%前後）には実質到達しなかった。馬連の確率計算も同じパターンだった。

実機スクリーンショットの数値で検証したところ完全に一致した：
- 画面のEV: 2.89（ノシェリー、単勝5.8倍）
- cal_prob(0.4991) × 5.8倍 = 2.895 ≈ **2.89**
- 本来使うべきtan_pct(0.11) × 5.8倍 = 0.638（1.0未満で本来は候補にすら入らない）

馬連はさらに深刻で、Harville近似の式（単勝確率が前提）に複勝確率（約50%）を渡すと確率が1を超える不正な値になっていた（#2-#7のEV3.37も同じ式で再現）。

このバグの影響範囲は「直前オッズ取得」ボタンを押した後のクライアント側再計算のみ。朝の予想生成時にサーバー側（`bet_optimizer.py`、Gumbelシミュレーションで正しく単勝確率を算出）が出す`gumbel_bets`は影響を受けていなかった（実際、該当レースの朝データには単勝推奨自体が存在せず、馬連の組み合わせも画面表示と全く異なっていた＝直前オッズ取得のたびにこのバグ入りJS計算で完全に上書きされていたことを確認）。

#### 🔴 発見②：表の「EV」列（`pn × オッズ`の生値）が、既に「使えない」と検証済みの指標だった
表内の各馬ごとのEV列（`h.ev`、コメントは`EV表示（pn × win_odds）`）は、2026-07-05のセッションで既にバックテスト済みで「EV>=1.3でも実勝率8.5%≒baseline8%、選択シグナルとして機能しない。明らかなnon-value(EV<1.0)の除外用にのみ有効」と結論づけられていた指標（`ev_direct`）だった。

softmax温度T=3.5（過信防止のため意図的に高め設定）により、勝率がオッズの上昇ほど急激には下がらないため、`勝率×オッズ`という単純な掛け算ではオッズが大きい馬ほど機械的に数値が膨らみやすい。ユーザーが見たスクリーンショットでも、人気薄がほぼ全て緑（EV≧1.0）になっていた。これは2026-07-23⑦の乖離分析（AIが市場より強気になるほど実勝率が単調に下がる、AI>>>市場帯はわずか3.6%）とも整合する、既知の弱点の再確認だった。

#### 対応
1. `recalcGumbelBets()`の単勝・馬連の確率計算を`(h.tan_pct || 0) / 100`に統一（複勝欄・三連複欄は元々正しいフィールドを使用済みで対象外）
2. 表のEV列の緑ハイライト（`val-buy`／緑文字の`val-neutral`）を廃止し、1.0以上/未満とも同系統の目立たないグレー表示に統一。列見出しに「単体では買い目の根拠になりません」というtitleツールチップを追加

モデルの温度自体を今いじるのはリスクが高く（人気馬側の過信再燃・今朝合意した特徴量/モデル変更の一旦停止に反する）ため見送り、今回はUI表示側の是正のみに留めた。

#### テスト
`index.html`はPythonテスト対象外のため、以下で検証：
- `node --check`でJS構文検証
- 実機スクリーンショットの数値（tan_pct=11.0, cal_prob=0.4991, odds=5.8）を使い、修正後のロジックで実際に0.64（旧バグ値2.89とは別物、1.0未満で候補外になる）が出ることを`node -e`で直接確認
- 既存の`python -m pytest tests/ -q`は336テスト通過（Pythonコードは変更していないため影響なし、回帰確認のみ）

---

### 2026-07-24③：週次ワークフローの「Commit & Push」がgit rebase時にバイナリDBファイルでコンフリクトして失敗する既知バグを修正

#### 背景
「明日のレースの取得できませんでした。毎週おんなじことを繰り返してます」という報告を受け、
`friday-predict.yml`の実行履歴をGitHub Actions APIで直接調査した。

#### 調査結果：今回自体はバックエンドは成功していたが、過去に複数回の実失敗を確認
今回（2026-07-24 20:20 JST開始）のワークフロー実行ログを確認したところ、実際には
新潟・中京・札幌の34レースを正常に取得・保存・デプロイ済みだった（バックエンド自体は成功）。
ただし過去の実行履歴（2026-06-27〜2026-07-17）を遡ると、複数回の**実際の失敗**を確認した。
さらに実行時間が週を追うごとに伸びていることも判明:

| 週 | 実行時間 |
|---|---|
| 6/25〜6/26 | 1〜3分 |
| 7/3 | 6〜8分（一部失敗） |
| 7/10 | 11〜12分（一部失敗） |
| 7/17・7/24 | 15分 |

（血統・調教師所属・コース系スクレイピングの積み重ねによる増加と推測）

#### 🔴 発見：「Commit & Push」ステップがgit rebase中にバイナリDBファイルでコンフリクトして失敗していた
過去の失敗ログ（2026-07-03, 2026-07-10）を確認したところ、いずれも同じパターンだった:

```
error: cannot rebase: You have unstaged changes.
Applying autostash resulted in conflicts.
Encountered 1 file that should have been a pointer, but wasn't:
	data/keiba.db
```

`friday-predict.yml`等の最後のステップは「予測結果をローカルコミット→
`git pull --rebase --autostash origin main`→push」という手順だったが、
**予測処理自体が10〜15分かかる間にmainブランチに別の変更（コードPRのマージ等）が
pushされると**、rebase時に`data/keiba.db`のようなバイナリDBファイルでコンフリクトを
起こし、gitが自動マージできず失敗する。`concurrency: group: keiba-weekly`は
同グループ内のワークフロー同士の同時実行は防ぐが、**コードPRのマージ（このセッション中の
私自身のPRマージ作業を含む）はこのグループの外側で直接mainにpushされるため防げない**。
実行時間が伸びるほど、この「衝突の窓」が広がり続けていた。

#### 対応：git rebaseをやめ、「常にmain最新の上に上書きコミット」する方式に変更
これらのファイル（`latest.json`/`history.db`/`keiba.db`等）は毎回まるごと再生成される
ものであり、rebaseで履歴をつなぐ意味がそもそも無い。そこで4つのワークフロー
（`friday-predict.yml`/`weekend.yml`/`sunday-results.yml`/`monthly-retrain.yml`）の
「Commit & Push」系ステップを全て以下の方式に統一した:
1. 生成物を一時ディレクトリに退避（`friday-predict.yml`/`monthly-retrain.yml`は
   対象ファイルを明示指定、`weekend.yml`/`sunday-results.yml`は
   `git status --porcelain -- data/`で変更・新規ファイルを動的に検出）
2. `git fetch origin main && git reset --hard origin/main`で常に最新のmainを取得
3. 退避した生成物を復元 → `git add` → 変更が無ければ終了、あれば最新main上でコミット
4. push成功するまで最大5回リトライ（失敗のたびに1に戻って最新mainを取り直す）

これによりrebase/mergeが一切発生しなくなり、バイナリファイルのコンフリクトという
失敗モード自体を構造的に排除した。5回リトライしても失敗する場合のみ`exit 1`で
ジョブを失敗させる（無限に隠れて失敗し続けることを防ぐ）。

#### 見つけて即座に直した副次バグ（自分のfixのテスト中に発覚）
上記の設計を実際にローカルのgitリポジトリで再現テストしたところ、
`git add data/latest.json data/history.db data/keiba.db`のように**複数パスを
1回のgit addで指定すると、いずれか1つでも存在しないパスがあると全体が失敗し
何も追加されない**（exit 128、`fatal: pathspec ... did not match any files`）という
git自体の挙動を発見した。本番では3ファイルとも常に存在するため実害は無いはずだが、
将来的な既知課題（ファイルが誤って削除される等）への耐性を高めるため、
`friday-predict.yml`/`monthly-retrain.yml`の該当箇所を「1ファイルずつ存在確認して
`git add`」するループに修正した。

#### テスト
GitHub Actionsのワークフロー(YAML+bash)はpytestの対象外のため、以下の方法で検証した:
- 全4ファイルの`python3 -c "import yaml; yaml.safe_load(...)"`によるYAML構文検証
- 各`run:`ブロックを`bash -n`で構文チェック
- **North Starの「本番データ形状で検証する」に従い、実際に一時的なgit remote/clone環境を
  作り、「別のcloneが並行してmainにpushする」という本番と同じ競合状態を再現した上で、
  新しいpush方式が正しくmainの最新を取り込みつつ自分の生成物を失わずにpushできることを
  実際に確認**（`friday-predict.yml`の明示ファイルリスト方式・`weekend.yml`の
  `git status --porcelain`動的検出方式の両方を個別に検証、後者は新規ファイル作成の
  ケースも含む）
- 上記の検証中に前述の「複数パスgit add」バグを実際に検出・修正し、再検証で解消を確認
- 既存の`python -m pytest tests/ -q`は336テスト通過（Pythonコードは変更していないため
  影響なし、回帰確認のみ）

#### 今後の運用上の注意
このセッション中の私（Claude）自身のPRマージ作業は、週次ワークフローの実行中と
時間的に重なりうる。今回の修正でその衝突自体は解消されるが、実行時間が伸び続けている
（15分）こと自体は別課題として残る。次回以降、週次ワークフローの体感速度が
気になる場合は改めて調査すること。

---

### 2026-07-24②：馬体重を特徴量化（f_weight_trend_avg/f_weight_last_diff新設）＋血統×馬体重の相談

#### 背景
「馬体重はどう扱ってますか？」という質問を受け、調査した結果
`horse_history.body_weight`列は結果ページのスクレイピングでは取得しているが
**engine.pyのどのf_*関数からも参照されておらず、SQLで取得しても捨てられている
だけの完全な死んだデータ**だったことが判明した（充足率6.5%という既知の低さとは
別に、そもそも0%使われていなかった）。ユーザーから追加で「直前オッズ取得ボタンで
同じ場所から拾えないか」「体重増減×血統で成長曲線が分かるのでは」という提案を
受けた。

#### 調査結果
- 直前オッズ取得（`gas/getOdds.gs`）は`accessO.html`（単勝・複勝オッズ専用ページ）
  のみを叩いており、`fetch_odds_for_race`のパース処理はオッズの形をした
  セル（`X.X`/`X.X-Y.Y`）だけを拾う設計。同エンドポイントからの馬体重取得は
  現状のページ構成では見込み薄いと判断（実機HTML未確認のため断定はしない）
- 血統側は`SIRE_DB`に種牡馬ごとの`peak`（能力ピーク年齢）・`type`
  （early/standard/late＝早熟/普通/晩成）が既に定義されており、`f_blood()`が
  年齢とピークの差から成長度を計算し本番XGBに供給済み（`calc_features_for_xgb`
  内`feats['f_blood']`）。ただし実体重の増減とは組み合わさっていなかった
- ユーザーの意向：当日（レース直前）の馬体重取得は難易度が高いため見送り、
  **結果取得時に既に持っているbody_weight/body_weight_diffを予想特徴量に
  入れるべき**、という明確な指示を受けた

#### 対応：馬体重推移を独立特徴量として追加
血統の成長曲線（f_blood内のage/peak/type）と体重推移の相関は、手動の交互作用式を
組むよりXGBの木構造に任せる方が良いと判断（RL/CL分離の相談時と同じ理由：木モデルは
特徴量間の交互作用を自動学習できる）。よって`f_weight_trend_avg`
（直近5走のbody_weight_diff平均）・`f_weight_last_diff`（直近走の値）の
2特徴量のみを追加し、血統特徴量との組み合わせ方はモデル自身に学習させる設計とした。

学習側`_get_history_before()`（build_training_data.py）・推論側
`get_history_from_db()`（jra_scraper.py）の両方に`body_weight`/
`body_weight_diff`をSELECT・辞書に追加（学習/推論パリティを最初から確保、
本セッションで繰り返し発見したパリティ違反の教訓を踏まえて両方同時に実装）。
データが無い場合は`f_finish_time_avg`等の既存の時系列系特徴量と同じ規約で
`NaN`（0.0等への偽装なし、XGBが欠損方向として学習）とした。

#### 学習/推論パリティ
`calc_features_for_xgb()`は学習データ生成・推論の両方から呼ばれる共通関数で、
`_get_history_before()`/`get_history_from_db()`双方に同じタイミングで
body_weight/body_weight_diffを追加したため、コード変更のみでパリティが保たれる。

#### ⚠ 現状の限界（充足率6.5%）
`body_weight`の充足率は6.5%と低いまま（Stage3列マッピング崩れが原因と推定、
未修正の既知課題）。今回の変更はその低い充足率のデータでも「使える形にする」
配線を通しただけであり、抽出自体の精度は改善していない。データが乏しい間は
大半の馬でNaN（欠損）扱いになり、特徴量としての寄与は限定的と見込まれる。
抽出精度を上げるには実機の結果ページHTML（view-source）での列位置確認が必要
（2026-07-21④で「複数行にまたがるカード状レイアウトの可能性」を指摘済み、
未解決）。将来充足率が改善すれば、コードの変更なしに特徴量の実効性が
自動的に上がる設計。

#### テスト
`tests/test_scraper.py`に3テスト新規追加（North Starに従い、実際に
`save_history_db()`で一時DBへ書き込んでから`get_history_from_db()`で
読み出す形で検証）:
- `test_get_history_from_db_includes_body_weight`: 返り値にbody_weight/
  body_weight_diffが正しく含まれることの確認
- `test_calc_features_for_xgb_computes_weight_trend_from_history`: 過去2走の
  増減(+6, -2)からf_weight_trend_avg=2.0・f_weight_last_diff=6.0になることの
  統合確認
- `test_calc_features_for_xgb_weight_trend_nan_when_no_data`: データ欠損時に
  0.0等へ偽装せずNaNのまま渡ることの確認
全336テスト通過。

#### 今後の流れ
1. 実機の結果ページHTML確認による`body_weight`充足率改善（別スコープ、要実機HTML）
2. 次回のColab再学習で`f_weight_trend_avg`/`f_weight_last_diff`の重要度を確認
3. 当日馬体重（予想時点）の取得は今回見送り。将来検討する場合は出馬表側
   （`accessD.html`）の直前再取得を軸に、実機HTML確認から着手すること

---

### 2026-07-24①：推論時のget_history_from_db()がfinish_time/time_diff_secを返しておらず、f_time_diff_avg等が推論時は常にNaN/デフォルト値に落ちていたバグを修正

#### 背景
「木の次は、葉っぱです。細かいところまで調査して下さい」という依頼を受け、
直前のracecourse/corner_all欠落バグ（⑧）と同じ手口（`get_history_from_db()`が
返す辞書のキーと、`calc_features_for_xgb()`側が読むキー名の突き合わせ）で
`_get_history_before()`（学習側、build_training_data.py）と
`get_history_from_db()`（推論側、jra_scraper.py）が提供する全キーを一つずつ
比較する監査を行った。

#### 🔴 発見：finish_time/time_diff_secが推論時のhistoryに一度も入っておらず、track_conditionもキー名が違っていた
学習側`_get_history_before()`は過去走ごとに`finish_time`・`time_diff_sec`・
`track_condition`（`condition`とのエイリアスとして両方）を返すが、推論側
`get_history_from_db()`のSELECT文には`h.finish_time`/`h.time_diff_sec`が
含まれておらず、返す辞書にも`track_condition`キー（`condition`のみ存在）が
含まれていなかった。この結果、推論時は以下が軒並み機能不全になっていた:

- `calc_competitiveness()`（`r.get('time_diff_sec')`使用）→ `f_competitiveness`/
  `f_competitive_best`が常にデフォルト値(0.5, 0.5)
- `f_finish_time_avg`/`f_time_diff_avg`計算ブロック（`r.get('finish_time')`/
  `r.get('time_diff_sec')`使用）→ 常に`NaN`
- スピード指数計算（`calc_speed_figure(finish_time=r.get('finish_time'), ...)`）→
  `finish_time`が常に`None`のため`calc_speed_figure`が常に`None`を返し、
  `f_speed_fig_last/avg/max`とその相対特徴量`rl_f_speed_fig_*`（計9特徴量）が
  常に`NaN`
- `f_heavy_track_rate`（`r.get('track_condition', ...)`使用、`condition`への
  フォールバックが無い）→ 過去走が常に「稍重以上ではない」扱いになり常に
  デフォルト値0.33

本番モデルのfeature importanceを確認したところ、**`f_time_diff_avg`は
130特徴量中10位（1.756%）**という上位の重要度を持っており（学習時は機能して
いたことの裏付け）、他にも`f_speed_fig_last`(40位/0.699%)・`f_competitiveness`
(50位/0.633%)・`f_competitive_best`(60位/0.572%)・`f_finish_time_avg`
(87位/0.462%)・`f_heavy_track_rate`(94位/0.444%)・`f_speed_fig_avg`
(100位/0.434%)・`f_speed_fig_max`(126位/0.336%)と、直前のracecourse/corner_all
バグ（8特徴量）よりも多い**合計12以上の特徴量ブロック**が影響を受けていた。
特に`f_time_diff_avg`は今回発見した中で最も重要度の高い特徴量であり、
学習/推論パリティ違反としては本セッションで最大の影響範囲だった。

#### 対応
`src/scraper/jra_scraper.py`の`get_history_from_db()`の2つのSELECT文に
`h.finish_time, h.time_diff_sec`を追加し、結果辞書に`finish_time`/
`time_diff_sec`を追加。また`track_condition`を`condition`と同じ値の別名
として追加し、学習側`_get_history_before()`が既に採用している「両キー提供」
パターンに合わせた（個別の呼び出し箇所を`condition`にも対応させて回るより、
学習側と同じ辞書形にする方が今後の同種バグを防ぎやすいと判断）。
`horse_history`テーブルには両列とも既に存在し（`finish_time`は充足率95.2%、
`time_diff_sec`はfinish_time依存）、データの追加取得は不要でSELECT文の修正
のみで解消する。

#### テスト
`tests/test_scraper.py`に2テスト新規追加（North Starに従い、実際に
`save_history_db()`で一時DBへ書き込んでから`get_history_from_db()`で
読み出す形で検証）:
- `test_get_history_from_db_includes_finish_time_and_time_diff_sec`: 返り値に
  finish_time/time_diff_sec/track_conditionが正しく含まれることの確認
- `test_calc_features_for_xgb_uses_time_diff_from_inference_time_history`:
  その返り値をそのまま`calc_features_for_xgb()`に渡した場合に
  f_time_diff_avg/f_finish_time_avgが非NaNになることの統合確認
両テストとも修正前のコードに対しては実際に失敗する（NaNのまま）ことを確認済み。
全333テスト通過。

#### 見送った関連の発見（低優先度・別スコープ）
`calc_race_content_score(r)`（`r.get('agari_rank', finishers)`使用）も
`agari_rank`キーの欠如で常に最下位相当にフォールバックする同種の問題を抱えて
いるが、この関数はルールベース重み`_W`専用の`f_recent()`からのみ呼ばれており、
2026-07-21②で確認済みの通り本番XGB推論が正常動作している間は一切使われない
（XGB例外時のフォールバックのみで発動し、かつ発動時は警告ログが出る設計に
既になっている）。影響が限定的なため今回は対応を見送った。

#### ⚠ 本番モデルへの影響（次回Colab再学習で評価が必要）
racecourse/corner_allバグ（⑧）と同様、"欠落していた値を復元する"変更のため、
次回のColab再学習を待たずに、次回のワークフロー実行から直ちに本番モデルが
より正確な入力を受け取るようになる。学習データ自体は元々正しい値を持っていた
ため、モデルの学習内容自体は汚染されていない。ただし影響を受けた特徴量の中に
重要度10位の`f_time_diff_avg`が含まれるため、次回再学習でこの特徴量群の
重要度・分割点がどう変化するか確認することを推奨する。

---

### 2026-07-23⑧：推論時のget_history_from_db()がracecourse/corner_allを返しておらず、コース適性特徴量群が推論時は常にデフォルト値に落ちていたバグを修正

#### 背景
「森の後は、木を見て下さい。何か不具合はないですか？」という依頼を受け、
直前に追加したNorth Starルール（本番データ形状でテストする）を実際に適用する形で、
テストが薄い関数を中心に監査した。`get_history_from_db()`（推論時にcalc_all()が
使う馬の過去走を取得する関数、jra_scraper.py）にテストが一件も無いことに気づき、
精査したところ重大な学習/推論パリティ違反を発見した。

#### 🔴 発見：`calc_course_aptitude_features()`が使う`racecourse`/`corner_all`が推論時のhistoryに一度も入っていなかった
2026-06-25に導入されたコース適性特徴量群（f_same_course_rate/f_same_turn_rate/
f_straight_match/f_uphill_match/f_agari_at_similar/f_course_coverage、後に追加された
f_course_type_rate/f_uphill_severity_rateを含む計8特徴量）は、馬の過去走ごとに
`hrec.get('racecourse', '')`で競馬場名を取り出し、`get_course_profile(rc, sf)`で
その競馬場のコースプロファイルを引いてから今日のレースと比較する設計になっている。

ところが推論時（`calc_all()`）にこの`history`を構築する`get_history_from_db()`
（jra_scraper.py）のSELECT文には`racecourse`列も`corner_all`列も含まれておらず、
返される各過去走の辞書には`racecourse`キー自体が存在しなかった。そのため
`hrec.get('racecourse', '')`は常に空文字を返し、`get_course_profile('', surf)`は
存在しないキー（`'_芝'`等）を引いて常に`None`を返す。結果として
`if prof is None: continue`が**推論時は全ての過去走に対して毎回発動し、
この関数が計算する8特徴量全てが常にデフォルト値（未経験扱い）に落ちていた**。

一方、学習データ生成側（`build_training_data.py`の`_get_history_before()`）は
最初から`"racecourse": row['racecourse'] or ''`を正しく返しており、学習時は
実際のコース経験に基づいた値でモデルが学習されている。実際に本番モデルの
feature importanceを確認したところ、この8特徴量は0.37%〜0.68%の非ゼロな
重要度を持っており（学習時は機能していたことの裏付け）、**「学習時は本物の値で
学習し、推論時は常にプレースホルダー値を渡す」という典型的な学習/推論パリティ
違反**であることを確認した。f_post（直前セッション⑥）よりも影響範囲が広い
（1特徴量ではなく8特徴量のブロック全体）。

なお同様に`corner_all`（小回りコースでの3→4角の位置変動を見るサブ特徴量が使用）
も推論時には常に空文字だった。

#### 対応
`src/scraper/jra_scraper.py`の`get_history_from_db()`の2つのSELECT文
（通常検索・前方一致フォールバック検索）に`COALESCE(h.racecourse, '') as racecourse`・
`COALESCE(h.corner_all, '') as corner_all`を追加し、結果辞書にも`racecourse`/
`corner_all`キーを追加。`horse_history`テーブルには両列とも既に存在し
（`racecourse`は充足率100%、`corner_all`は94.5%、`docs/history_db_schema.md`で確認済み）、
データの追加取得は不要でSELECT文の修正のみで解消する。

#### テスト
`tests/test_scraper.py`に2テスト新規追加。North Star（本番データ形状でテスト）に
従い、手打ちdictではなく実際に`save_history_db()`で一時DBへ書き込んでから
`get_history_from_db()`で読み出す形で検証:
- `test_get_history_from_db_includes_racecourse_and_corner_all`: 返り値に
  racecourse/corner_allが正しく含まれることの確認
- `test_calc_course_aptitude_features_uses_inference_time_history`: その返り値を
  そのまま`calc_course_aptitude_features()`に渡した場合にf_course_coverageが
  非ゼロになることの統合確認
両テストとも修正前のコードに対しては実際に失敗する（f_course_coverage=0のまま）
ことを確認済み。全331テスト通過。

#### ⚠ 本番モデルへの影響（次回Colab再学習で評価が必要）
現行の`xgb_fukusho_model.pkl`はこの8特徴量が推論時常にデフォルト値になる状態を
前提に運用されてきた（学習データ自体は正しい値を持つため、モデルの学習内容は
汚染されていない）。今回の修正で次回のワークフロー実行から推論時にも実際の
コース経験値が入るようになるため、モデルが学習時に想定していた分布へようやく
一致する。次回のColab再学習でこの8特徴量の重要度・寄与がどう変化するか確認する
ことを推奨するが、コード修正のみで学習をしなくても、次回の予想生成から
直ちに（今の本番モデルのまま）より正確な値が使われるようになる点でf_post修正とは
性質が異なる（f_postは特徴量の"精度"を上げる変更、今回は"欠落してい値を復元する"
変更で、再学習を待たずに実際の予想品質改善が期待できる）。

---

### 2026-07-23⑦：全体総括（森を見る）＋「本番データ形状でテストする」ルールをNorth Starに格上げ

#### 背景
f_post修正（直前の⑥）の完了報告を受け、ユーザーから「葉っぱにとらわれれば木は見えず、
木にとらわれれば森は見えず。質問のたびに小さな修正ポイントが見つかる。一度全体の状況を
精査してほしい」という指摘を受けた。個別バグの修正（葉）を繰り返す一方、プロジェクト全体の
方向性（森）を俯瞰する機会がなかったことは事実であり、実データで裏付けた総括を行った。

#### 総括で確認した事実
1. **直近8日間（7/16〜7/23）で🔴発見バグが10件**。いずれもユーザーからの質問・実行結果
   報告がきっかけで発覚し、こちらから先に見つけたものはほぼ無かった
2. **市場KPIは依然として市場優位**（累積258レース: AI logloss 0.2589 vs 市場0.2351,
   delta+0.0238）。残差学習モデル本番投入後のクリーンなデータは7/18・7/19の2週69レースのみで
   「エッジがある」と言える段階に達していない
3. **乖離分析（7/18時点215レース）で、AIが市場より強気になるほど実際の勝率が単調に下がる**
   （AI<<市場10.7%→一致9.4%→AI>市場6.5%→AI>>市場3.9%→AI>>>市場3.6%）。AI・市場の
   本命が食い違ったレースで市場が勝った回数(37)はAIが勝った回数(16)の2.3倍。「AIが市場の
   見落としを見つける」という馬券戦略の根幹仮説は、現時点のデータではむしろ否定的
4. **特徴量は138個まで増加した一方、直近追加分（コース×距離系・f_post修正）は
   いずれも重要度1%未満**。根本問題（②③）と無関係な精緻化に労力を割き続け、複雑性の
   増加が①のバグ発生率を押し上げる悪循環になっている
5. bracket/win_odds/body_weightの充足率は2026-06-03から0〜6.5%のまま7週間以上未着手
   （「使われていないから低優先度」は循環論法：使われていないのは埋まっていないから）

#### 判明した①の根本原因（テスト方針の構造的欠陥）
直近の重大バグ（`sqlite3.Row.get()`未対応、スクレイパーのShift_JIS未指定、
GitHub/Drive間history.dbスキーマずれ等）はいずれも共通のパターンを持つ：
**テストが「本物に似ているが微妙に違う代用品」（手打ちdict、綺麗な自作HTML文字列、
作りたての単一DB）を使っていたため、テストは通り続けたまま本番だけ壊れていた**。
個別バグを直すたびにその場限りの修正で終わり、「テストの書き方そのものを変える」
レベルまで教訓が一般化されていなかったことが、①の高頻度発生の直接原因と判断した。

#### 対応：North Starに新ルールを追加
「🌟 North Star」に6番目のルールとして追加（既存6番「迷ったらDESIGN.md確認」は7番に繰り下げ）：
> テストは「本番と同じデータ型・同じフォーマット」で書く。DBの行を扱うコードは
> 手打ちdictではなく実際に`sqlite3.Row`として取り出したもので、スクレイパーは
> 綺麗な自作HTML文字列ではなく実機のHTML構造を模したfixtureで、モデルファイルを
> 読むコードは実際にpickle/xgboost UBJ等その形式で保存したファイルで検証する。

これにより、セッションを跨いでも劣化しない最重要ルールとして今後毎回参照される設計とした。

#### 今後の優先順位（森ベース、コード変更ではなく方針）
1. これ以上の特徴量追加は一旦停止（②③の根本問題と無関係なため）
2. 残差モデルのフォワードデータを最低N=300レース程度まで溜める（現状69レースでは
   結論を出せない。今できる分析はほぼ出尽くしており次に意味のある判断はデータ蓄積後）
3. 上記North Starルールを今後のテスト追加・レビュー時に実際に適用する
4. bracket/win_odds/body_weightは「本気で直す」か「正式に諦める」かを一度決める
   （7週間の塩漬けを継続しない）

#### 今回の変更範囲
CLAUDE.md（North Star追加＋本セクション）のみ。コード変更・テスト実行は無し
（分析と方針の合意のみのセッション）。

---

### 2026-07-23⑥：XGB特徴量f_postが枠順バイアスの実データ・距離帯別フォールバックを一切使っていなかったバグを修正

#### 背景
「各競馬場の特徴を網羅したが、実際のレースでの枠順の有利不利はどのような扱いをしてますか？」
という質問を受け、`f_post`（枠順バイアス特徴量）の実装を確認したところ、
XGBが実際に学習に使っている経路にバグを発見した。

#### 🔴 発見：`calc_features_for_xgb()`内のf_post計算が競馬場単体の固定値しか見ていなかった
枠順バイアスには本来3段階のフォールバックが設計されている
（優先度高い順: `_post_zone_bias`＝history.dbから`_build_and_save_stats()`が構築する
実データ統計 > `POST_BIAS_BY_ZONE`＝競馬場×距離帯の固定値 > `POST_BIAS`＝競馬場単体の
固定値）。この3段階ロジックは独立関数`f_post(h, race)`には正しく実装されていたが、
**この関数はルールベース重み`_W`専用で、本番のXGB推論からは呼ばれていない**。
XGBが実際に使う`calc_features_for_xgb()`内のインラインf_post計算は
`POST_BIAS.get(rc, 0)`（競馬場単体の固定値）のみを参照しており、
`_post_zone_bias`（実データ）にも`POST_BIAS_BY_ZONE`（距離帯別固定値）にも
一切アクセスしていなかった。結果として、本番XGBモデルは常に最も粗い
枠順シグナルのみで学習されており、距離帯（短距離/マイル/中距離/長距離）による
枠順有利不利の違いも、蓄積された実データによる補正も反映されていなかった。

なお`post_position`（独立関数側が使うキー）と`horse_num`（XGB側が使うキー）は
`parser.py`/`jra_scraper.py`確認の結果どちらも同じ馬番（umaban/num）を指しており、
ロジック統一に支障はないことを確認済み。

#### 対応
`src/features/engine.py`の`calc_features_for_xgb()`内のf_post計算を、
独立関数`f_post(h, race)`と同じ3段階フォールバック・符号規約
（`_post_zone_bias` > `POST_BIAS_BY_ZONE` > `POST_BIAS`、`POST_BIAS`は旧設定で
外枠正値のため符号反転して整合）に揃えた。この関数の実行時に既に定まっている
`rc`/`dist`/`n`をそのまま使う設計とし、別関数呼び出しによる重複デフォルト値は避けた。

#### 学習/推論パリティ
`calc_features_for_xgb()`は学習データ生成（`build_training_data.py`）と推論
（`calc_all()`）の両方から呼ばれる共通関数のため、コード変更のみでパリティは
自動的に保たれる。

#### ⚠ 本番モデルへの影響（次回Colab再学習で評価が必要）
現行の`xgb_fukusho_model.pkl`はf_postを**旧・競馬場単体固定値のみの分布**で
学習済み。今回の変更でf_postの値が距離帯・実データに応じてより精密になるため、
次回のColab再学習（build_training_data→train_xgb）でこの新しい分布に対して
再学習し、分割点を学習し直すことを推奨する。値は既存の妥当な範囲内（0-10）に
収まる保守的な変更のため、再学習前でも暴走リスクは低い。

#### テスト
`tests/test_features.py`に3テスト新規追加
（`_post_zone_bias`実データ不在時に距離帯別固定値へフォールバックすることの確認、
実データがあれば距離帯別固定値より優先されることの確認、未知の競馬場では
競馬場単体固定値＝中立5.0にフォールバックすることの確認）。全329テスト通過。

---

### 2026-07-23⑤：ペースモデル再学習の新旧Accuracy比較で同値が「悪化」と誤表示されるバグを修正

#### 背景
`KEIBA_XGB_retrain_v5.ipynb`セル3（展開予測モデル再学習）を実際にColabで実行した
ユーザーから出力を共有され、確認した際に発見。新モデルVal Accuracy=0.5436に対し、
「旧モデル Accuracy: 0.5436 (↓悪化)」と表示されていた。新旧の値が完全に同一
（0.5436=0.5436）にもかかわらず「悪化」と表示されるのは明らかにおかしい。

#### 🔴 発見：同値比較が常にelse節（悪化）に落ちるロジックバグ
`src/tools/train_pace_model.py`の比較ラベル生成が
`"↑改善" if acc > old_acc else "↓悪化"`という2値分岐になっており、
`acc == old_acc`（同値）の場合に`acc > old_acc`が`False`になるため、
常に`else`側の「↓悪化」に落ちてしまっていた。同値ケースを区別する分岐が
存在しなかった。

今回の同値自体は、このセッション中に発生した別トラブル（`database is locked`、
history.dbスキーマ不一致）のデバッグの過程でセル3を複数回実行したことに起因する
（同一データ・同一パラメータでの再実行なら同じAccuracyになりうる）偶然の可能性が
高く、データやモデル自体の異常ではない。ただし表示ロジック自体は同値ケースを
考慮しておらず、独立したバグとして修正が必要だった。

#### 対応
`_format_accuracy_delta(acc, old_acc)`ヘルパー関数を新設し、
`acc > old_acc` / `acc < old_acc` / 同値の3分岐に整理（同値は「→変化なし」）。
呼び出し箇所を差し替え。

#### テスト
`TestFormatAccuracyDelta`を新規追加（改善・悪化・同値の3ケース）。全326テスト通過。

---

### 2026-07-23④：GitHub/Drive間のhistory.dbスキーマずれによる再学習マージ失敗を修正

#### 背景
ユーザーが実際にColabで`KEIBA_XGB_retrain_v5.ipynb`のセル1bを実行したところ、
2回連続で異なるエラーが発生した。1回目は`database is locked`（ランタイム再起動で解消）、
2回目は`OperationalError: table horse_history has 34 columns but 36 values were supplied`。

#### 🔴 発見：GitHub側とDrive側でhistory.dbのスキーマが独立に進化し乖離していた
`data/history.db`は2026-07-14に初めてGit（LFS管理）にコミットされ、以降は週次の
GitHub Actionsワークフロー（weekend.yml/sunday-results.yml）が`save_history_db()`を
呼ぶたびに、そこに含まれる`ALTER TABLE`マイグレーション一式（sire/dam_sire/
trainer_affiliation等）が自動的に適用され続けていた。一方Drive側の`history.db`は
Colabのretrainノートブックが`save_history_db()`を呼ぶ経路を持たず、スキーマが
更新されないまま止まっていた（血統列追加の2026-07-17、調教師所属列追加の
2026-07-21より前の状態）。この結果、セル1bの`INSERT OR IGNORE ... SELECT *`が
列数不一致（Drive34列 vs GitHub36列）でエラー終了していた。

なお同時に判明した事実として、GitHub側の`history.db`は2026-07-14以降のみの
浅い履歴（5,342レース）であるのに対し、Drive側は運用開始以来蓄積された深い履歴
（11,416レース）を持っており、両者は「同じデータの新旧」ではなく
「起点の異なる別系統」であることも分かった。ただし`race_id`をキーにした
`INSERT OR IGNORE`である限り、件数差そのものはデータ破損の兆候ではなく
安全にマージできる。

#### 対応（`KEIBA_XGB_retrain_v5.ipynb`セル1b）
- マージ処理の前に`save_history_db([], base_dir=BASE_DIR)`を呼び、Drive側の
  `history.db`にALTER TABLEマイグレーションを安全に（空リストのため実データ追記
  なしで）適用し、スキーマをGitHub側と揃える一手間を追加
- 念のための多重防御として、マージのINSERT文を`SELECT *`から**両テーブルに
  共通する列名のみを明示指定する方式**に変更（`PRAGMA table_info`で両テーブルの
  列集合を取得し積集合を使用）。今後再びスキーマがずれても、共通列だけで
  安全にマージを継続でき、GitHub側にしかない列は警告表示のうえ除外される

#### 気づいたこと（今回はスコープ外）
GitHub ActionsとColab双方が独立に`history.db`を保持・更新している現状の構造は、
将来的に同じ`race_id`に対して異なる内容が入る余地を持つ（今回は発生していないが
検証もできていない）。次回時間のあるときに、両DBのrace_id重複部分の内容一致を
確認することが望ましい。

---

### 2026-07-23③：v5ノートブックのデータファイル取得漏れを修正

#### 背景
「XGB再学習でどこにコードを入れるか、GitHubから毎回ノートを取っているか」という
質問を受け、`KEIBA_XGB_retrain_v5.ipynb`の構造を確認した。

#### 発見：セル2の強制アップデートに新規データファイルが含まれていなかった
セル2「src/ 強制アップデート」は`src/features/engine.py`を含むコードファイル一式に
加え、`data/course_profiles.json`・`data/note_schema.json`もGitHubから取得する設計
だったが、2026-07-23②で新規作成した`data/course_distance_profiles.json`がこの
リストに入っていなかった。このままセル1〜10を実行すると、コード（engine.pyの
calc_course_distance_features()）は最新版に更新されるが、参照先のデータファイルが
Colab側に存在せず、`load_course_distance_profiles()`が常にNoneを返し、
f_dirt_turf_start/f_course_hill_diff/f_course_corner_tightの3特徴量が
**全レースでデフォルト値（実質無効）のまま学習されてしまう**ところだった。

#### 対応
セル2の`rel_data`リストに`'data/course_distance_profiles.json'`を追加。
あわせて末尾の起動確認（engine.pyの主要シンボル存在チェック）に
`calc_course_distance_features`のキーワードチェックも追加し、コード面での
反映漏れも即座に気づけるようにした。

#### 今後の運用
新しいdata/*.jsonファイルを追加した際は、`KEIBA_XGB_retrain_v5.ipynb`セル2の
`rel_data`リストへの追加を忘れないこと（同様の他ノートブックの強制アップデート
セルがあれば同様に確認）。

---

### 2026-07-23②：コース×距離特徴量を実装（ダート芝スタート・坂・コーナータイト度）

#### 背景
2026-07-23①でJRA全10場のコースデータ収集に成功した後、「①データ収集→②特徴量実装」の
順で進める合意のもと、実際の特徴量化に着手した。当初「まずダートの芝スタートフラグ
のみ実装し、坂・コーナー情報は見送る」という縮小案を提示したところ、ユーザーから
「中途半端な仕事はしない、矛盾した数値の検証をしてから全部やれ」と明確な指摘を受け、
仕切り直した。

#### 対応：データの再検証と全項目の実装
1. **web検索で拾った数値の矛盾を検証**：
   - 阪神「外回り芝1200m」という記述はJRA公式データ（1200mは内回りのみ）と矛盾 →
     再検索で否定。内回り専用、ゲート〜3コーナー243mと確定
   - 札幌芝1200mは「180m」「276m」と検索内で2つの数値が併存 → 再検索で「約400m」に
     ほぼ統一されることを確認
   - 京都ダート1200mの「900m」は一周1607.6mに対し長すぎ不自然 → 再検索で「約400-410m」
     が正しいと確認
2. **内外回りの発走距離リストの見落としを発見・修正**：
   阪神は内回り表・外回り表がそれぞれ独立した`発走距離`列を持つ構造で、当初
   `_first_value()`で最初のtableの値しか見ておらず、外回り専用の1600m/1800m/2400m/2600m
   が欠落していたと判明（中山・京都・新潟は1つのtableに(内)(外)タグ込みで全距離が
   入っており、この問題は起きていなかった）。全table行を直接読み直して解消
3. **`data/course_distance_profiles.json`を新規作成**（venue×surface×distanceの粒度）
   - `dirt_turf_start`: 競馬場ごとのダート芝スタート距離リスト（JRA公式プロース文、高信頼）
   - `loop_by_distance`: 中山・阪神・京都・新潟の芝について、距離→内回り/外回りの対応表
     （JRAの発走距離表記の(内)/(外)タグから作成。同じ距離は常にJRA公式データも参照）
   - `hill`: 競馬場×surface(×内外回り)ごとの主要な坂の高低差・方向・位置（JRA公式プロース文）
   - `corner_tightness`: 同、コーナーのタイト度（Tight/Normal/Wide/Very Wide、JRA公式の
     形容表現に基づく分類）
   - `_start_to_corner_m_reference`: web検索で得たスタート〜第1コーナー距離。信頼度が
     一様でないため特徴量化はせず参考データとして保持のみ
4. **`src/features/engine.py`に実装**
   - `load_course_distance_profiles()` / `calc_course_distance_features()`を新規追加
   - `_resolve_turf_loop()`: loop_by_distanceから距離→内外回りを解決するヘルパー
   - `calc_features_for_xgb()`に3特徴量を追加: `f_dirt_turf_start`（ダート芝スタート
     フラグ）/ `f_course_hill_diff`（坂の高低差）/ `f_course_corner_tight`（コーナー
     タイト度、Tight=1.0〜Very Wide=4.0の数値化）
   - 内外回りで坂・コーナー特性が同じ場（中山等）はloop接尾辞なしキーで登録し、
     ルックアップ時はloop付きキー→loop無しキーの順にフォールバックする設計
   - 未定義の組み合わせは安全なデフォルト（f_dirt_turf_start=0, f_course_hill_diff=0,
     f_course_corner_tight=2.0=Normal相当）に落ちる

#### 学習/推論パリティ
`calc_features_for_xgb()`は学習データ生成（`build_training_data.py`）と推論
（`calc_all()`）の両方から呼ばれる共通関数のため、コード変更のみでパリティは
自動的に保たれる。

#### ⚠ 本番モデルへの影響（次回Colab再学習で評価が必要）
今回追加した3特徴量は、既存の`xgb_fukusho_model.pkl`の学習時には存在しなかった
新規特徴量。**次回のColab再学習（build_training_data→train_xgb）で特徴量重要度・
AUCへの寄与を確認すること**。North Star「データなしの特徴量追加をしない」との
整合性については、dirt_turf_start/hill/corner_tightnessいずれもJRA公式サイトの
実データ（コース紹介プロース文・コースデータtable）を根拠としており、決め打ちでは
ない。ただし内外回りで値が同一の場合のフォールバック設計や、一部venueで
`hill.elevation_diff_m: null`（新潟内回り・ダート等、公式文に詳細記述がなく未確認）
のまま0.0にフォールバックする箇所があることは留意すること。

#### テスト
`tests/test_features.py`に11テスト新規追加（course_distance_profilesのロード、
dirt_turf_startの複数競馬場での判定、内外回りでcorner_tightnessが異なることの確認、
未定義競馬場でのデフォルトフォールバック確認、calc_features_for_xgb経由の統合確認）。
実装中に「中山は内外回りで坂の値が同じためloop接尾辞なしキーでしか登録していない」
ことに起因するlookup失敗を実際にテストで検出・修正した。全323テスト通過。

---

### 2026-07-23：コースデータ収集ツール、JRA全10場で基本データ取得に成功

#### 概要
2026-07-22⑥のh3見出しベース分類ロジック修正版（PR #87）をColabで再実行した結果、
**JRA全10競馬場で`course_basic.csv`の全項目（直線距離・高低差・一周距離・
ダート一周距離・幅員・回り方向）が取得できた**ことを確認した。画像
（コース平面図・立体図・芝/ダート高低断面図）も全10場で正しく検出・保存済み。

#### 確認できた値の一部
| 場 | 直線距離 | 高低差 | 芝一周 | ダート一周 |
|---|---|---|---|---|
| 中山 | 310m | 5.3m | 1667.1m | 1493m |
| 阪神 | 356.5m | 1.9m | 1689m | 1517.6m |
| 東京 | 525.9m | 2.7m | 2083.1m | 1899m |
| 京都 | 328.4m | 3.1m | 1782.8m | 1607.6m |
| （他6場も同様に全項目取得済み） | | | | |

途中2回、ユーザーが同じ空欄結果を再送してきた原因はColab側のモジュール/クローン
キャッシュ（ランタイム再起動やフォルダ名変更だけでは解消しないケースがあった）で
あり、コード側の問題ではなかった。最終的にGitHubの生ファイルを直接HTTPで取得して
`div.block_unit`文字列の有無を確認する診断コードで「本当に最新版が読み込まれて
いるか」を切り分けたところ、最新版に切り替わった時点で全場成功した。

#### 残る制約（当初の設計方針どおり据え置き）
ゴール前坂の位置・高低差のプロース文抽出（`elevation_features.csv`）は中山以外の
9場すべてで失敗している。各競馬場紹介文の言い回しが場ごとに大きく異なるため、
正規表現1本での汎化は困難と判断し、無理な自動化は追求しない。収集済みの断面図
画像・生テキスト（`{venue_eng}_raw.html`）を目視で読み取る後続作業に委ねる。
同様に`start_to_corner.csv`・`corner_features.csv`のコーナー形状分類も未着手。

#### 今後の流れ
1. 保存された10場ぶんの画像（コース図・断面図）を順次アップロードしてもらい、
   目視でelevation_features.csv / start_to_corner.csv / corner_features.csvの
   空欄を埋める
2. 十分なデータが揃った時点で、距離別データ構造（`course_profiles.json`相当）を
   設計し、`calc_course_aptitude_features()`に統合する特徴量を新設・XGB再学習で
   評価する（North Star「データなしの特徴量追加をしない」に従い、①が完了するまで
   特徴量化はしない）

---

### 2026-07-22⑥：コースデータ収集ツールを10場実行で検証・caption依存の分類ロジックを修正

#### 背景
2026-07-22⑤の修正版をユーザーがColabで実行し、10場中8場で基本データが取得できる
大きな前進があった。特にダートの芝スタート区間検出は6場（阪神1400m・中京1400m・
東京1600m・京都1400m・中山1200m・新潟1200m）で成功し、当初の相談内容に直接応える
結果が得られた。一方で阪神が完全に空欄、京都・札幌・函館・福島は直線距離のみ空欄、
小倉は逆に一周距離・幅員のみ空欄という部分的な欠落が残っていたため、実機HTML
（阪神・京都）を追加で見せてもらい原因を特定した。

#### 🔴 発見：table の caption 文字列に依存した分類ロジックが場によって全滅していた
`extract_course_tables()`は各`<table>`の`<caption>`テキストが「芝コース：コースデータ」
「芝コース：各コースデータ」のような文言を含むかどうかで芝/ダート・基本/各コースを
分類していたが、これは中山固有の書き方だった。実機確認の結果:
- **阪神**: captionは「内回り」「外回り」のみ（「芝コース」という文言を含まない）。
  ダートtableに至ってはcaption自体が存在しない。→ 分類条件に一つも一致せず、
  パース済みの行がまるごと握りつぶされていた（「コースデータtableが見つからない」
  警告の直接の原因）
- **京都**: 直線距離を含むtableにcaptionが無い（高低差を含む別tableにはcaptionあり）。
  → 直線距離側のtableだけ分類漏れし、高低差など他の値だけ取れるという中途半端な
  結果になっていた

#### 対応
`extract_course_tables()`を、table自身のcaptionではなく**親`div.block_unit`の
h3見出し**（「芝コース」「ダートコース」の完全一致）でグルーピングする方式に
全面変更。同じブロック内に複数table（阪神の内回り/外回り、中山の基本+各コース等）
があっても、h3見出しさえ一致すれば両方のtableの行をまとめて拾えるようになった。
h3見出しは中山・阪神・京都いずれの実機HTMLでも「芝コース」「ダートコース」と
完全一致することを確認済み（caption文字列より安定した分類キー）。

#### 副次的な発見：ダート芝スタート距離が既定候補リスト外だと行ごと消えていた
阪神の実機データで「ダートは1400メートル戦と2000メートル戦が芝スタート」という
一文を発見。1400mは既定の`DIRT_DISTANCES`候補リストに含まれるが**2000mは
含まれていなかった**ため、正しく検出できていても`distance_start.csv`の行として
出力されない欠落があった。`scrape_all_courses()`で、プロース文から実際に検出された
芝スタート距離を候補リストとの和集合に加えるよう修正し、候補リスト外の距離でも
行が失われないようにした。

#### 検証
実際にアップロードしてもらった阪神・京都の実機HTMLに対して修正後の関数を直接実行し、
阪神（直線356.5m・一周1689m・芝スタート1400m/2000m）・京都（直線328.4m・一周1782.8m・
芝スタート1400m）とも正しく抽出できることを確認済み。

#### 残る制約
ゴール前坂の位置・高低差のプロース文抽出は、中山以外の9場すべてで失敗している
（各場ごとに文章の構成が異なるため、正規表現1本での汎化は困難と判断）。これは
当初の設計方針どおり、収集済みの生テキスト・断面図画像を目視で読み取る後続作業に
委ねる。無理に自動化を追求しない。

#### テスト
阪神の実機構造を模したfixture（HANSHIN_LIKE_HTML）を追加し、caption文字列が
無い/異なる場合でもh3見出しで正しくグルーピングできることの回帰テスト、
および複数距離の芝スタート検出テストを追加（全26テスト）。全312テスト通過。

---

### 2026-07-22⑤：コースデータ収集ツールを実機HTMLで検証・全面書き直し

#### 背景
2026-07-22④で作成した`scrape_course_data.py`をユーザーがColabで実行したところ、
JRA全10場で画像4種すべて「未検出」、コース基本データも全項目空欄という結果になった。
実機の生HTML（中山競馬場、`{venue_eng}_raw.html`）をアップロードしてもらい、
実際のページ構造と突き合わせて原因を特定した。

#### 🔴 発見①：レスポンスのエンコーディング指定漏れで全ページが文字化けしていた
JRA公式サイトは`<meta charset="Shift_JIS">`でShift_JIS配信だが、`_get()`ヘルパーが
`r.encoding`を明示的に設定していなかった。`requests`のデフォルトのエンコーディング
判定に任せた結果、`r.text`が文字化け（mojibake）した状態でBeautifulSoupに渡り、
以降の日本語正規表現・見出し検索が一つも一致しなくなっていた。既存の
`src/scraper/jra_scraper.py`は全リクエストで`resp.encoding = 'shift_jis'`を
明示しており、このツールだけがその作法を踏襲していなかった。

#### 🔴 発見②：画像URLがルート相対だと誤認していた
JRA公式ページの`<img src="img/pic_course_3d.jpg">`はページURL相対パスだが、
旧コードは`f'{JRA_BASE}{src}'`（ルート相対想定）で結合しており、たとえ①を
修正しても不正なURLになっていた。`urllib.parse.urljoin(page_url, src)`に修正。

#### 実機で判明した本当のページ構造（中山競馬場で確認）
- 画像: `<div class="block_unit"><h3>{見出し}</h3>...<div class="img"><img src="..."></div>`
  の繰り返し。alt属性は空文字で使えないが、h3見出し（「コース立体図（右回り）」
  「芝コース高低断面図（右・内回り）」等）から確実に分類できる
- コースデータ: `<table><caption>芝コース：コースデータ</caption>...`という
  captionつきtableが複数（芝コース基本/芝コース各コースA・B・C/ダートコース）。
  th見出しでtd値を対応付けて読む設計に全面書き換え
- **「コース紹介」プロース文（`div.course_info`）に、回り・ゴール前坂の位置と
  高低差に加えて「ダートのレースは1200メートルのみが芝スタート」という一文が
  明記されていた**。これはユーザーが最初に例示した「ダートだがスタートから
  しばらく芝を走る」を、画像を目視することなくテキストから直接特定できる
  実例。中山の場合1200mが該当

#### 対応
- `src/tools/scrape_course_data.py`を全面書き直し
  - `_get()`にShift_JISエンコーディング指定を追加
  - `extract_course_images()`: div.block_unit + h3見出しから画像を分類・
    `urljoin`で正しいURLを解決。内回り/外回りで断面図が2枚ある場合は
    `_turf_elevation_2.png`のように連番で両方保存
  - `_parse_data_table()` / `extract_course_tables()`: captionつきtableを
    th見出し→td値のdictに変換し、芝基本/芝各コース/ダート基本に分類
  - `extract_course_info()`: コース紹介プロース文から回り・坂の位置と高低差
    （`残り180メートルから残り70メートル地点にかけて...上り坂の高低差は2.2メートル`
    のような言い回しを正規表現で抽出）・ダートの芝スタート距離を抽出
  - umasiru.com補完ロジックは削除（JRA公式単体で必要なデータが揃うことが
    実機検証で確認できたため、不要な複雑さを削減）
- 実際にアップロードしてもらった中山の生HTMLに対して全関数を直接実行し、
  期待通りの値（直線310m・高低差5.3m・一周1667.1m・右回り・ダート1200mが
  芝スタート等）が取れることを確認済み

#### 残る制約（他9場は未検証）
中山以外の9場のページ構造は今回まだ確認できていない。同じテンプレート
（`course_common.css`）を使っている可能性が高いが、内回り/外回りの有無や
文言は場によって異なりうるため、位置決め打ちではなくラベル・キーワード
探索を維持している。見つからない項目は例外を出さず空欄でログに警告する。

#### テスト
中山の実機HTML構造を模したfixtureで`tests/test_scrape_course_data.py`を
全面書き直し（24テスト）。全310テスト通過。

---

### 2026-07-22④：コース×距離特徴（ダート芝スタート区間等）のためのコースデータ収集ツールを新規作成

#### 背景
「各競馬場のコース・距離別の特徴（例: ダートだがスタートからしばらく芝を走る、
ゴール前がなだらかに上り坂等）をAIが学習し、より適性のある馬を選べないか」という
相談を受けた。既存の`data/course_profiles.json`（2026-06-25導入）を確認したところ、
venue×surfaceの20キーのみで**距離別の区別を持たない設計**だったため、同じ「東京_ダート」
でも距離によって発走後しばらく芝を走る区間の有無が異なる、という実際のJRAコース設計を
表現できないことが判明した。

#### 対応方針
DESIGN.mdの「決め打ちで実装しない」原則に従い、まずJRA公式サイトのコース図・
コースデータをこの環境から直接確認しようとしたが、jra.go.jp・Wikipedia・
umasiru.comいずれも本セッションのネットワークポリシーで403拒否され、
一般的な外部サイトへのアクセスが一切できないことを確認した（JRA固有の制限ではなく、
このセッションの許可リストが`anthropic.com`やパッケージレジストリ等に限定されている
ため）。

ユーザーと相談の上、①ネットワークアクセス可能なColab環境でこのタスク専用の
データ収集スクリプトを実行してもらう方針で合意。JRA公式サイトを優先ソース、
umasiru.com（https://umasiru.com/archives/category/racecourse）を補完ソースとする
方針で`src/tools/scrape_course_data.py`を新規作成した。

#### 実装内容
- `scrape_all_courses(output_dir)`: JRA全10競馬場について実行するメイン関数
  - JRA公式コースページ（`https://www.jra.go.jp/facilities/race/{venue_eng}/course/`）
    から画像（平面図/立体図/芝高低断面図/ダート高低断面図）とコース基本データ
    （直線距離・高低差・一周距離・幅員・回り方向）を取得
  - 取得できなかった項目はumasiru.comの該当競馬場記事で補完
  - 画像分類・数値抽出は両サイトの実際のHTML構造をこの環境から確認できないため、
    複数の候補正規表現・キーワードで探索し、見つからなければ**例外を出さず
    空欄のままログに警告**を出す設計（北星ルール「決め打ちで実装しない」に準拠）
  - 距離別スタート位置・スタート〜1コーナー距離・ゴール前坂の位置/高低差・
    コーナーのタイト度分類は、テキストとして明記されていない限り自動抽出できない
    （コース図・断面図を目視で読み取る必要がある）ため、対象距離ぶんの空行のみを
    CSVに用意し、画像を見た後の後続作業（別セッションでの目視読み取り）で埋める
    前提とした
  - 出力: `images/`（コース図・生HTML）、`csv/`（course_basic.csv /
    distance_start.csv / start_to_corner.csv / elevation_features.csv /
    corner_features.csv）、`scrape_log.txt`（競馬場ごとの取得結果ログ）
- **画像ファイルはGitHubリポジトリに含めない設計**。JRA/umasiru.comの著作物である
  画像をパブリックリポジトリに複製配布するのはリスクがあるため、Colab実行時は
  Google Drive等のリポジトリ外ディレクトリに保存する運用とする

#### 今後の流れ
1. ユーザーがColabで`scrape_all_courses()`を実行し、`images/`・`csv/`を生成
2. 生成された画像（コース図・断面図）をこのセッションにアップロードしてもらい、
   目視でstart_to_corner.csv / elevation_features.csv / corner_features.csvの
   空欄（スタート〜1コーナー距離、坂の位置・高低差、コーナー形状分類）を埋める
3. 十分なデータが揃った時点で`course_profiles.json`相当の距離別データ構造を設計し、
   `calc_course_aptitude_features()`に統合する特徴量を新設・XGB再学習で評価
   （North Star「データなしの特徴量追加をしない」に従い、①②が完了するまでは
   特徴量化しない）

#### テスト
`tests/test_scrape_course_data.py`新規作成。ネットワークアクセスを伴わない純粋
ロジック部分（正規表現抽出・画像分類・CSV書き出し）を18テストでカバー。
全304テスト通過。

---

### 2026-07-22③：エラータグ週次処理が2026-07-18から毎回無音で全滅していた重大バグを修正

#### 背景
「他に改修しておかなければならないことはないか」という確認を受け、CLAUDE.md内で
一度言及されたきり放置されていた項目を洗い出したところ、2026-07-18のセッションで
`⚠ エラータグ処理失敗（予想には影響なし）: 'sqlite3.Row' object has no attribute 'get'`
というエラーログが観測されていながら「今回のタイムアウト事故とは無関係、次回セッションで
調査」として記録されたのみで、その後一度も再調査されていなかったことが判明した。

#### 🔴 発見：`_build_race_result()`がsqlite3.Rowに`.get()`を呼び、週次エラータグ処理が起動直後に毎回例外で全滅していた
`src/features/error_tags.py`の`_build_race_result()`は、`history.db`から
`hist_conn.execute('SELECT * FROM race_history WHERE race_id = ?', ...).fetchone()`で
取得した`race_row`（`sqlite3.Row`、`conn.row_factory = sqlite3.Row`設定済み）を使い、
戻り値dictの各キーをほぼ全て`race_row['xxx']`というbracketアクセスで組み立てていたが、
**`race_class`列だけ`race_row.get('race_class', '')`という dict専用のメソッド呼び出しに
なっていた**。`sqlite3.Row`は`.get()`を持たないため、**この行は常に無条件で
`AttributeError`を送出**する。

`process_weekly_error_tags()`は対象日の全レースを1件でも処理する前に
`_build_race_result()`を呼ぶため、この例外は**その週の1レース目で即座に発生し、
関数全体がそこで中断**する。呼び出し元`scripts/sunday_results.py`側は
`try/except Exception`で囲って警告ログを出すだけで処理を継続するため、
ワークフロー自体は成功したように見えたまま、**エラータグ自動分類・週次補正
（`data/error_tags_weekly.json`への蓄積、および翌週予想への即時補正反映）が
毎週まるごと機能停止していた**。日付から見て2026-07-14の機能導入直後、
遅くとも2026-07-18確認時点から今回の修正まで、一度もこの処理が正常完走していない
可能性が高い。

#### 発見が遅れた理由（テストの穴）
`tests/test_error_tags.py`には`_build_race_result()`を実際の`sqlite3.Row`を使って
呼ぶテストが一つも存在せず、他の全テストは`race_result`をプレーンなdictとして
直接組み立てて`classify_race_tags()`に渡していた。dictは`.get()`を持つため、
このテストの書き方では今回のバグは決して再現しない。「本番のデータ型（sqlite3.Row）で
実際にテストする」ことの重要性を示す事例。

#### 対応
- `src/features/error_tags.py`: `'race_class': race_row.get('race_class', '')` を
  `'race_class': race_row['race_class'] or ''` に修正（bracketアクセス＋None時は
  空文字にフォールバック。race_classは94.7%充足、残り5.3%はNULLのため`or ''`で吸収）
- `tests/test_error_tags.py`: `TestBuildRaceResult`を新規追加。実際に`sqlite3.Row`を
  返す一時DB（race_history/horse_history）を構築し`_build_race_result()`を直接呼ぶ
  回帰テスト2件（race_class設定済み／NULLの両方）。修正前コードに対して実行すると
  実際に同じ`AttributeError`で失敗することを確認済み

#### 影響・今後
このバグはコードのみの問題で、データ側の欠損はない。修正後の次回日曜ワークフロー
実行から、エラータグの週次蓄積・条件別補正係数の自動更新が正常に再開する見込み。
過去に蓄積されなかった分のバックフィルはできないため、`data/error_tags_weekly.json`は
今回の修正以降のデータから改めて積み上がっていく。

#### テスト
新規2件（`TestBuildRaceResult`）追加。全286テスト通過。

---

### 2026-07-22②：直前オッズ急変アラートの結果蓄積を確認＋集計側の閾値をアプリのバッジ条件に整合

#### 背景
「直前オッズ取得ボタンで急激にオッズが下がった馬について、アラートで教えてくれる
システムだが、これに対する結果の蓄積はできているか」という質問を受けた。

#### 確認結果：蓄積の仕組み自体は機能している
`scripts/generate_stats.py`の`calc_odds_movement_analysis()`は
`race_predictions`（朝予想）と`odds_snapshots`（直前オッズ）を突き合わせて
朝→直前の変動率を算出し、日曜ワークフロー実行のたびに`data/divergence_weekly.json`へ
週次蓄積している。実データ（2日分・約4,600頭）で正常に動作していることを確認した。
ただし2日分はまだ傾向を語れる量ではなく、継続観察が必要な段階（急騰バケットの
勝率9.8% vs 横ばいバケット11.8%という初期値はサンプル不足でノイズの可能性が高い）。

#### 発見：アプリのバッジ判定条件と集計側の閾値がズレていた
`index.html`の`updateOddsAndEV()`が実際に「🔴急騰」「🟠上昇」バッジを出す条件は
**%下落かつオッズ差の絶対値**の複合条件（急騰: 30%以上下落 かつ 3.0倍以上の差、
上昇: 20%以上下落 かつ 2.0倍以上の差。1.0→0.9のような微小変動を除外するため）。
一方`calc_odds_movement_analysis()`の`_move_bucket(pct)`は%のみでバケット分類しており、
絶対差の条件が抜けていた。そのため「アプリ画面では急騰バッジが出ない微小な変動」が
集計上は急騰バケットに混入し、アラートの実績評価（バケット別勝率）がアプリの
実際の挙動とズレた数字になっていた。

#### 対応（`scripts/generate_stats.py`）
- `_is_hot(pct, abs_diff)` / `_is_warm(pct, abs_diff)` を新設。
  `index.html`の判定式と完全に同じ条件（pct<=-30 かつ abs_diff>=3.0 / pct<=-20 かつ abs_diff>=2.0）
- `_move_bucket(pct, abs_diff)` を再設計。%レンジは一致するがバッジ条件（絶対差）を
  満たさないレコードは「急騰相当(バッジ対象外)」「上昇相当(バッジ対象外)」という
  別バケットに分離し、バッジが実際に出たケースと混同しないようにした
- `ai_agrees_market`/`ai_disagrees_market`の急騰・上昇判定も同じ`_is_hot`/`_is_warm`基準に統一
- `big_risers`/`big_fallers`（変動幅ランキング表示用、バッジとは無関係の値）は対象外のまま維持

#### テスト
`test_movement_bucket_aligns_with_app_badge_thresholds`を新規追加
（バッジ条件を満たす急騰/上昇と、%のみ一致し満たさない急騰相当/上昇相当が
それぞれ正しく分離されることを確認）。全284テスト通過。

---

### 2026-07-22：JRA公式サイト取得情報の100%活用監査＋2件修正

#### 背景
「JRAから取得している情報は100%有意義に活用しているか、例えばコーナー通過順の
グルーピング表記`(1,*5)6,10(2,9)-(3,4)8=7`のような部分」という監査依頼を受け、
実機（sp.jra.jp）スクリーンショットと現行コードを突き合わせて棚卸しした。

#### 発見①：ペース判定ラベルも見出し不一致で長期未取得だった可能性
`_extract_weather_pace()`は「ペース」直後のH/M/S**1文字**のみを探索していたが、
実機の表記は「ペース判定：**ミドルペース**」という**単語表記**。かつこの関数は
ヘッダテーブル（`tables[0]`）のテキストしか見ておらず、ペース判定はヘッダとは
別セクション（タイム欄近辺）に存在するため、そもそも探索範囲にも入っていなかった。
`_extract_lap_times`の見出し不一致（2026-07-21③修正）と同種のバグ。

影響は限定的：`train_pace_model.py`の`_classify_pace()`は、このJRA公式ラベルが
無くても`first_3f`から独自にペース分類するフォールバックを持つ設計だったため、
致命的ではなかった（ただし`first_3f`自体も直前まで未計測だったため、2つの不具合が
重なって「ペース関連情報が実質機能していない」期間があった可能性が高い）。

**対応**：`_extract_weather_pace(header_text, full_text=None)`にfull_text引数を追加し
ページ全体を探索対象にした上で、単語表記（スロー/ミドル/ハイ）にも対応。
1文字表記（後方互換）・単語表記の両方をS/M/Hに正規化して返す。

#### 発見②：コーナー通過順の同着グルーピング表記は一切未収集だった
「全馬コーナー通過順位」セクションの`(1,*5)6,10(2,9)-(3,4)8=7`のような表記は、
ラップタイム抽出処理が「ここで打ち切る」ための目印として文字列検索されるだけで、
中身は一度も読まれていなかった。個別馬の通過順列（`corner_all`、例:`3-3-2-1`）は
別途正しく取得できているが、こちらは**同着・並走の詳細情報を持たない簡略版**。

**対応**：`_extract_corner_passage(soup)`を新規追加し、3コーナー・4コーナーの
グルーピング表記を生テキストのまま`race_history.corner_pass_3`/`corner_pass_4`に
保存する。構造の解釈（括弧=同着、-=差、==同タイム、等）はまだ行わず、**収集のみ**
（血統・調教師所属と同じ段階的導入方針。特徴量化はデータが貯まってから）。

#### 見送った項目（優先度低と判断）
馬主・生産牧場（勝馬の紹介欄）、競走中の出来事等（鞭使用の過怠金等の制裁情報）は、
予想精度への寄与が読みにくいため今回は着手しなかった。

#### テスト
`_extract_weather_pace`の単語表記対応テスト2件・後方互換テスト1件、
`_extract_corner_passage`の抽出テスト1件・セクション無しテスト1件、
`parse_result_soup`経由の統合テスト1件、`save_history_db`でのDB保存確認テスト1件を追加。
全283テスト通過。

---

### 2026-07-21⑥：調教師所属（栗東/美浦）のスクレイピング・DB蓄積を開始

#### 背景
「JRA-VANとJRA公式のデータの差を埋めたい。予想精度にどう繋げるか」という相談を受けた。
前回セッションで無料公式サイト（sp.jra.jp）の結果ページ実機画面から、調教師欄が
「西村真幸(栗東)」「秋本大介(美浦)」のように所属込みで表示されていることを確認済み
だったため、この差分（東西所属）を埋める形で着手した。血統(sire/dam_sire)と同じ
「まずスクレイピング+DB蓄積のみ行い、特徴量化はデータが溜まってから」という
段階的な導入方針を踏襲する。

#### 実装内容
- `src/scraper/jra_scraper.py`: `_split_trainer_affiliation(trainer_text)`を追加。
  正規表現`^(.+?)[\(（](栗東|美浦)[\)）]$`で調教師欄から名前と所属を分離する。
  所属表記が無い場合は`(名前, None)`を返す後方互換設計（列位置の想定がズレて
  違うテキストが入っても例外は出さずNoneになるだけ）
- `parse_result_soup()`: `trainer`（名前のみ、既存互換）と`trainer_affiliation`
  （'栗東'/'美浦'/None）の両方をfinishersに格納
- `src/utils/db.py`: `horse_history`に`trainer_affiliation`カラムを追加
  （ALTER TABLE migration、sire/dam_sireと同じ後方互換パターン）

#### 学習/推論パリティに関する設計判断（重要）
出馬表（予測対象の未来レース）ページで同じ「名前(栗東)」形式の所属表記が
取得できるかは未検証（実機の生HTMLを確認できていない）。そのため、出馬表
パーサー（`parser.py`）側の変更は今回**行っていない**。

代わりに、血統と同様に「調教師名は不変の属性」という性質を利用し、今後
history.dbに蓄積される`trainer_affiliation`から**「調教師名→所属」の辞書を
別途構築し、推論時はこの辞書から名前で引く**設計を予定している
（`_jockey_dict`/`_trainer_dict`と同じ発想）。これにより出馬表ページから
所属を再取得できなくても、一度でも結果ページで所属が判明した調教師については
推論時に参照可能になり、学習/推論パリティが自然に保たれる。

#### 今後の流れ（今回はスコープ外）
1. 数週間、結果取得のたびに`trainer_affiliation`が蓄積される（バックフィルなし）
2. 十分な件数が溜まったら「調教師名→所属」辞書を構築（`_build_horse_dicts()`的な仕組み）
3. 所属×開催競馬場の対応表（栗東=関西圏開催が地元、美浦=関東圏開催が地元）を用意し、
   「今回のレースが所属地の地元開催か」を表す特徴量を新設、次回のXGB再学習で評価する
4. **今回時点ではこの特徴量自体は未実装**。データなしで特徴量を追加しないという
   North Starルールに従う

#### テスト
`_split_trainer_affiliation`の単体テスト3件、`parse_result_soup`経由の統合テスト1件、
`save_history_db`でのDB保存確認テスト1件を追加。全276テスト通過。

---

### 2026-07-21⑤：月次自動再学習の重大バグ発見・修正＋土日パイプライン差異の是正

#### 背景
「今後はレース結果と予想の乖離やAIの穴、規則性を充実させたい。そのために結果の
品質を上げたい。土曜と日曜は同じ条件で結果取得できているか」という相談を受け、
weekend.py（土曜）とsunday_results.py（日曜）のコードを比較調査した。

#### 🔴 発見・修正①：monthly-retrain.ymlが2026-07-01から完全に機能停止していた
`monthly-retrain.yml`は毎月1日03:00 JSTにcronで自動発火し、`monthly_retrain.py`を
実行して結果を確認なしでmainへ直接pushする設計。GitHub Actionsの実行履歴を確認したところ、
**2026-07-01の初回実行が0秒で即座に失敗**していた。原因は
`from src.tools.calibrate_xgb import calibrate_xgb`という**存在しない関数名の
import**（実際の関数名は`run_xgb_calibration`）。importが即座に失敗するため、
本文の`train_xgb()`呼び出しにすら到達していなかった。

さらに調査の結果、たとえimportが直っていても**別の重大な設計不備**があった。
`train_xgb()`は`residual`引数のデフォルトが`False`で、`monthly_retrain.py`は
これを指定せずに呼んでいた。本番は2026-07-14以降、f_popularityを特徴量から
除外する残差学習モデルで稼働しているが、`residual=False`で学習すると
**f_popularityを含む市場コピー型の旧方式モデル**が生成される。AUC閾値0.75は
市場コピー型モデルでも余裕で通過してしまうため、このバグがもし放置されたまま
importエラーだけ直っていたら、**次回8/1の自動実行で残差学習モデルが
市場コピー型モデルに黙って置き換わっていた**（7/14に苦労して排除した
市場コピー依存が復活する）。加えて`residual=True`で学習した場合の保存先は
`xgb_fukusho_model_residual.pkl`等のサフィックス付きファイルであり、
本番が読み込むサフィックス無しファイルへの反映ステップも存在しなかった。

**対応**（`scripts/monthly_retrain.py`）:
- 存在しない関数を`import`していた行を削除
- `train_xgb(..., residual=True)`を明示指定
- 学習後、`xgb_fukusho_model_residual.pkl`/`xgb_feature_cols_residual.json`を
  本番ファイル（サフィックス無し）へ`shutil.copy2`で反映するステップを追加
- `xgb_calibrator.pkl`の自動更新は今回見送り。`run_xgb_calibration()`が
  `predict_proba()`前提の実装で残差学習モデル（`xgb.Booster`、`predict()`のみ）
  に非対応なため、誤ったキャリブレーションを自動生成するリスクを避けた。
  次回Colabでの手動キャリブレーション実行が必要（本番の予測順位自体には
  影響しない、複勝確率表示の較正がやや古いまま据え置かれるのみ）

#### 発見・修正②：日曜のshadow_betsは`was_recommended`が常に0だった
`weekend.py`（土曜）は`bets`テーブルから当日のrace_idを引いて
`record_all_shadow_bets(..., recommended_race_ids=_rec_ids)`と呼んでいたが、
`sunday_results.py`は同じ呼び出しを`recommended_race_ids`無しで行っており、
日曜に実際に推奨・購入したレースでも`shadow_bets.was_recommended`が常に0で
記録されていた。アプリの「直近レース一覧」の`rec`表示にのみ影響する軽微な
不整合だが、土日で同じ条件になるよう是正した。

#### 発見・修正③：`update_correction_table()`が死んだ計算だった
`sunday_results.py`が呼んでいた`update_correction_table()`は
`correction_table.json`を書き込むが、`engine.py`/`app_json.py`/`make_bets.py`の
どこからもこのファイルを読み込んでいないと判明した（エラータグ補正システムに
機能が引き継がれ、こちらの整理を忘れていたとみられる）。呼び出しを削除した
（`correction.py`内の関数定義自体は将来の再接続に備えて残置）。

#### 土日パイプラインの比較まとめ
コア部分（`fetch_results()`によるレース結果取得）は土日で完全に同一。
周辺処理（バイアス分析・翌日予想生成は土曜のみ、週次ROI集計表示は日曜のみ）は
設計上の意図的な非対称であり、問題ではない。

#### テスト
`scripts.monthly_retrain`のimport成功確認・`residual=True`指定確認・
本番ファイルへのコピー処理存在確認の3件、`scripts.sunday_results`の
import成功確認・`recommended_race_ids`受け渡し確認・`update_correction_table`
削除確認の3件を追加。全271テスト通過。

---

### 2026-07-21④：実機の結果ページ画面提供により2件の実バグを発見・修正

#### 背景
ユーザーからJRA-VAN（有料データサービス）とsp.jra.jp（無料公式サイト・スマホ版）
両方の結果ページのスクリーンショット提供を受けた。前者はスクレイパーの取得元とは別の
有料サービスだが、後者は実際にスクレイパーが読みに行く無料公式サイトと同系統のページで、
これまでネットワークアクセス不可のため実物を確認できなかった構造を初めて実データで検証できた。

#### 発見・修正①：`_extract_lap_times`の見出し不一致（first_3f/last_3f未計測の原因特定）
実機の「タイム」欄は「ハロンタイム」見出し（例: `9.5 - 11.1 - 11.6 - 12.2 - 12.4 - 12.8`）
と「上り」見出し（`4F 49.0 - 3F 37.4`）で構成されていたが、`_extract_lap_times()`は
「ラップタイム」という表記しか探索しておらず、**この見出し違いにより毎回`idx < 0`で
即座に`([], None, None)`を返し続けていた可能性が高い**。`docs/history_db_schema.md`に
「first_3f: 未計測（要再検証）」と記載されていた課題の直接的な原因と見られる。
「ラップタイム」（後方互換）と「ハロンタイム」の両方を探索するよう修正した。

#### 発見・修正②：`parse_dividends`が馬単(umatan)を解析しておらず決済が潜在的に壊れていた
`src/utils/db.py`の`bet_type in ('馬連','馬単')`決済ロジック（624行目付近）は
`divs.get('umatan', {}).get('payout', 0)`を参照する設計だったが、`parse_dividends()`は
単勝/複勝/馬連/ワイド/三連複しか解析しておらず、**`umatan`キーは一度も生成されていなかった**。
そのため馬単の的中判定（`is_hit`）自体は正しく動作するが、**払戻金`payout`は常に0円に
なる潜在バグ**があった（馬単を発券・記録していた場合、勝ってもROI集計上は0円としてしか
記録されない）。合わせて枠連(wakuren)・三連単(sanrentan)の解析も追加し、
「三連複」「三連単」の検索が漢数字表記のみだった箇所も実機で確認した数字表記
「3連複」「3連単」を後方互換で追加対応した。

#### 発見（未対応・実装は見送り）
以下は実機で存在を確認したが、今回は実装しなかった。理由を付記する。
- **調教師の東西所属**（「西村真幸(栗東)」のような表記）：実機で確認済み、特徴量化の
  価値はあるが、実際のスクレイパー取得元（www.jra.go.jp内部JRADB）が全く同じ表示形式か
  未検証。生HTMLでの構造確認が先
- **コーナー通過順の同着グルーピング表記**（`(1,*5)6,10(2,9)-(3,4)8=7`のような括弧表記）：
  現状の`corner_all`は数字のみ抽出しこの情報を破棄している。実機で存在確認したが、
  特徴量としての活用方法は未検討
- **毛色・減量記号**（★4kg減等）：実機で確認したが予測精度への寄与は不明瞭。優先度低
- **bracket/win_odds/body_weight埋まり率0〜6.5%の根本修正**：実機の結果ページは
  「馬名/人気→性齢/体重→騎手/調教師→タイム/着差」という**複数行にまたがるカード状レイアウト**
  で構成されており、`parse_result_soup`が前提としている「1行15列のフラットテーブル」という
  想定自体がズレている可能性が新たに浮上した。決め打ちでの列インデックス修正はDESIGN.mdの
  「やってはいけないこと」に該当するため、実際にスクレイパーが取得する生HTML（view-source）
  を確認してからでないと着手しない

#### テスト
`_extract_lap_times`の新表記対応テスト2件・旧表記後方互換テスト1件・見出しなしテスト1件、
`parse_dividends`の全券種網羅テスト1件・漢数字後方互換テスト1件を追加。
全265テスト通過。

---

### 2026-07-21③：騎手・調教師勝率のベイズ縮小導入（AUC向上施策①）

#### 背景
「AUCを上げるためにできることは？」という相談を受け、低リスクで実装可能な施策として
`jockey_stats_dict`/`trainer_stats_dict`のハードカットオフをベイズ縮小に置き換える案を
提示し、実施の合意を得た。2026-07-20②で`calc_features_for_xgb`の少走数レート系特徴量に
導入したベイズ縮小と同じ発想を、`_build_horse_dicts()`が作る騎手・調教師勝率にも適用する。

#### 発見した問題
`_build_horse_dicts()`の`new_jockey_dict`/`new_trainer_dict`は、`runs >= 10`の
ハードカットオフで9走以下の騎手・調教師を**丸ごと辞書から除外**していた。除外された
騎手・調教師は、lookup側（`calc_all()`）で一律デフォルト値（jockey:0.15 / trainer:0.12）に
フォールバックする。そのため「1走1着の若手騎手」も「一度も出走記録のない未知の騎手」も
区別されず同じ0.15になっていた。ハードカットオフは境界（9走→10走）で不連続に扱いが
変わる点でも、ベイズ縮小より情報の使い方が粗い。

#### 対応
`src/features/engine.py`に共通ヘルパー`_bayes_shrink(hits, n, prior, k)`を追加
（`_bayes_rate(hits_list, prior, k)`はこれの薄いラッパーに再実装、既存呼び出し元の
挙動は完全互換）。`_build_horse_dicts()`の辞書構築を次のように変更:
- `runs >= 10`の除外 → `runs > 0`（1走でもあれば辞書に入る）
- 生の勝率 → `_bayes_shrink(wins, runs, prior=0.15, k=10)`（騎手）
  / `_bayes_shrink(wins, runs, prior=0.12, k=10)`（調教師）
- prior はlookup側の既存デフォルト値（0.15/0.12）とあえて揃え、k=10は旧カットオフ
  「10走」に相当する事前分布の重みとした。n=0付近ではlookup側の従来デフォルトと
  ほぼ同じ値に、走数が増えるほど実測勝率に連続的に収束する

CSV由来の`_jockey_dict`/`_trainer_dict`（手打ち登録・優先）は対象外のまま据え置いた
（スコープを絞り、影響範囲を最小化するため）。

#### 学習/推論パリティ
`_build_horse_dicts()`は`init_engine()`から呼ばれ、学習データ生成
（`build_training_data.py`が`jockey_rate`/`trainer_rate`をDBから引く際も同じdictを参照）
と推論の両方で同じ辞書を使うため、コード変更のみでパリティは保たれる。

#### ⚠ 本番モデルへの影響（要フォローアップ）
2026-07-20②のベイズ縮小と同種の注意点。現行`xgb_fukusho_model.pkl`は
`f_jockey_rate`/`f_jockey`/`f_trainer`をこの変更前の分布（少走数は一律0.15/0.12）で
学習済み。今回の変更で少走数騎手・調教師の値が連続的に変わるため、
**次回のColab再学習で新しい分布に対して分割点を学習し直すことを推奨**。
値は保守的な範囲（priorと実測値の間）に収まるため、再学習前でも暴走リスクは低い。

#### テスト
`_bayes_shrink`の単体テスト2件、`_build_horse_dicts`の統合テスト2件
（少走数騎手が除外されずprior寄りの値になる／十分な走数があれば実測値に近づく）、
調教師priorの単体テスト1件を追加。全260テスト通過。

---

### 2026-07-21②：残課題の経緯調査による再評価＋XGB推論の無音フォールバックに警告追加

#### 背景
「残課題について、課題が発生した時期と現在までの経緯を調査して本当に取り組むべき
課題かどうか今一度調査してほしい」という依頼を受け、CLAUDE.md「残っている課題」表と
引き継ぎ書「未解決課題」表の各項目を、初出セッション〜現在までのコード変遷を
git履歴・実装を突き合わせて再調査した。

#### 発見：3件の「中」課題は実は現在の挙動にほぼ影響していない
1. **bracket/win_odds/body_weight 埋まり率0〜6.5%**（初出2026-06-03）：
   コード調査の結果、3列とも**どのf_*特徴量関数からも読まれていない**と判明。
   bracketはdb.py書き込みのみ、win_oddsは`engine.py`に
   「history.dbのwin_oddsは0%欠損のためpopularity（99.2%充足）を使う」との
   明示コメントがあり2026-07-06に既に代替済み、body_weightはSQL取得のみで
   特徴量化されていない。7週間以上「中」深刻度で残っていたが、実質無害化されていた
2. **重みの妥当性確認（rl/maturityが0.01）**（初出2026-05-25 Phase2導入時）：
   `calc_all()`を確認したところ、本番でXGBが正常動作している間、ルールベース重み
   `_W`（rl/maturityの住み処）は**一切使われていない**（use_xgb時は
   `total = raw_prob * 10`のみ）。`_W`が使われるのはXGB推論が例外を吐いた時の
   フォールバックのみ。2026-07-14の残差学習モデル本番化以降、本番予測は
   100%XGB由来のため、この課題の検証価値はほぼ無い
3. **B2_ndcg残差学習版の再学習**（初出2026-07-14頃）：
   `app_json.py`の本番呼び出し（386行目）は`make_bets_v2()`を`feat_df`無しで
   呼んでおり、dual_model（B2_ndcg）は本番で発動しない（CLAUDE.mdの
   「過去に試して撤回した判断」表に既述の凍結状態と一致）。使われていない
   モデルの改良のため優先度を下げた

上記3件はCLAUDE.md「残っている課題」表・引き継ぎ書「未解決課題」表とも
深刻度を「中」→「低」に格下げし、格下げ理由を明記した。
「過去データノートのセル7（pkl再生成）未実行」もCLAUDE.md内で他に一切
言及がなく、`_build_and_save_stats()`によるチューニング時自動再生成で
既に自然解消している可能性が高いため同様に格下げした。

#### 発見：調査中に見つかった未文書化の実リスク（重みの妥当性確認の調査中に発見）
`calc_all()`のPass 2、XGB推論を囲む`except Exception:`（旧2537行目）が
**無警告・無ログで静かにルールベーススコアへフォールバックする**構造のままだった。
これは2026-07-16のxgb_ensemble_model.pkl事故（TypeErrorが握りつぶされ、
XGB予測を一切使わずルールベーススコアのみで予想が生成されていたことに
気づけなかった件）を実際に引き起こしたのと**全く同じ箇所**。あの事故の原因
（紛れ込んだファイル）は削除・ガード済みだが、「例外を握りつぶして無警告で
フォールバックする」という構造自体は今回まで温存されていた。

#### 対応
`engine.py`に`_warn_xgb_inference_fallback(horse_name, err)`を追加。
`calc_all()`の`except Exception:`ブロックに組み込み、原因を問わず例外発生時は
必ず1回だけ警告ログを出す（`_XGB_INFERENCE_ERRORS_WARNED`セットで同一例外の
重複警告を抑制）。`_check_xgb_feature_coverage`と同様の設計で、
`init_engine()`実行時に警告抑制状態をリセットする。

**挙動は変更しない**（ルールベースへのフォールバック自体は従来どおり）。
今回追加したのは検知・可視化のみ。

#### テスト
`_warn_xgb_inference_fallback`の単体テスト3件を追加
（警告が出る／同一例外は2回目以降警告しない／異なる例外は再度警告する）。
全256テスト通過。

---

### 2026-07-21：残作業の棚卸し・value_gap削除・環境制約の明記

#### 背景
「引き継ぎ書とCLAUDE.mdにある残作業をやる」という依頼を受け、
`docs/KEIBA-AI_引き継ぎ書_追補_2026-06-28.md`の「未解決課題」表を1件ずつ精査した。

#### このコーディング環境固有の制約（重要・毎回確認すること）
本セッションの環境では以下が**不可能**と判明した。「残作業をやる」系の依頼を
受けた際は、着手前にこの制約を思い出すこと。
- `jra.go.jp`へのネットワークアクセス（プロキシポリシーで拒否）
- ~~`data/*.db` / 一部`.pkl`の実データ読み込み~~
  → **2026-07-27③で解決済み。下記の方法で取得可能**
- Colabでのモデル学習・Google Driveアクセス

> 🔑 **LFS管理ファイル（keiba.db/history.db等）の実データ取得方法（2026-07-27③発見）**
> `git checkout`ではLFSポインタ（133バイト）しか得られないが、
> `media.githubusercontent.com`からLFS実体を直接HTTP取得できる（git-lfs不要）:
> ```bash
> curl -sL -o /tmp/keiba.db \
>   "https://media.githubusercontent.com/media/hanagenuku/keiba_ai/main/data/keiba.db"
> ```
> これにより**ユーザーにColab実行を依頼せずとも、その場で本番実データの
> 分析・検証が可能**。「LFSだから読めない」と諦める前に必ずこの方法を試すこと。

一方、`.gitattributes`で`-filter -diff -merge`によりLFS除外指定されている
`data/horse_features.csv`（学習特徴量, 37MB）・`xgb_fukusho_model.pkl`・
`data/stats.json`・`data/kpi_weekly.json`・`data/divergence_weekly.json`等は
**このリポジトリに実データとして存在する**ため直接分析可能。

#### 完了した作業
1. **value_gap削除**（`src/betting/ev_filter.py`）
   `detect_value_horses()`内で常に0.0を代入するだけの死んだフィールド
   `entry['value_gap'] = 0.0`を削除。index.htmlのVal列表示は既にEV表示（📊EV買い目）に
   置き換わっており、削除の前提条件（2026-07-02セッションでの後述課題）は満たされていた。
   対応する`test_detect_value_horses_value_gap_always_zero`テストも削除
2. **pairwiseモデル削除検討のクローズ**：`src/tools/train_ranking_model.py`に残る
   `rank:pairwise`引数はコード調査の結果、**本番稼働中のB2_ndcgモデルも学習する
   共通トレーナーの選択肢の1つ**であり、2026-07-10のPhase A大掃除で削除済みの
   pairwiseモデル成果物（`xgb_ranking_pairwise.pkl`等）とは別物と確認。
   追加の削除作業は不要と判断しクローズ
3. **未解決課題表の棚卸し**：上記2件を表から除去し、残り5件（残差モデルROI検証・
   温度再校正・条件帯別AI優位分析・B2残差再学習・bracket/win_odds/body_weight埋まり率）
   それぞれに「このコーディング環境で今すぐ着手可能か」の注記を追加

#### 各未解決課題の現状（実データで確認できた範囲）
- **残差モデルのフォワードROI検証**：`data/kpi_weekly.json`（実データ）で追跡可能。
  2026-07-16のxgb_ensemble_model.pkl事故修正後のクリーンなデータは
  7/18（delta=+0.0283, 市場優位）・7/19（delta=-0.0016, ほぼ同等）の2日・69レースのみ。
  まだ結論を出せる量ではなく、継続観察が正しい状態
- **条件帯別のAI優位分析**：`data/divergence_weekly.json`（実データ）も同様に2週分のみで、
  バケット別勝率の差をノイズと区別できる段階に達していない。時期尚早と判断し見送り
- **温度再校正**：意図的に「今は変えない」が正しい状態（二重補正リスク）。着手不要
- **B2_ndcg残差再学習**：Colabでのモデル学習が必須。本環境では不可
- **bracket/win_odds/body_weight埋まり率0%**：`_extract_body_weight()`/`_extract_win_odds()`
  等は既にコード上は列位置ヒューリスティックとして実装されている（`tests/test_scraper.py`の
  `test_parse_result_soup_win_odds`で合成HTMLに対しては正しく動作することを確認済み）。
  ただし実際のJRA結果ページの列順と一致しているかはネットワークアクセス不可のため検証不能。
  **次回、Colabで`history.db`の実際の充足率を再確認するか、実際の結果ページHTMLを
  提供してもらう必要がある**（決め打ちでインデックスを変更するのはDESIGN.mdの
  「やってはいけないこと」に該当するため、実データ確認なしでは着手しない）

#### テスト
`test_betting.py`から死んだテスト1件を削除。全253テスト通過。

---

### 2026-07-20③：XGB推論の特徴量欠落を検知する軽量ガードを追加

#### 背景
外部記事シリーズ第6回（predict.py/推論パイプラインの有料部分推測）のレビューで、
「学習時と違う特徴量を渡す」「不足列を勝手に0埋めする」ことが検知不能なまま
予測が完成してしまう危険パターンとして紹介されていた。自プロジェクトの
`engine.py`の実際のXGB推論コードを確認したところ、同種のパターンが実在すると判明した。

#### 発見した問題
`calc_all()`のPass 2、および`get_xgb_rating()`では、いずれも

```python
xrow = {c: xfeats.get(c, 5.0) for c in _XGB_FEATURE_COLS}
```

という形で、`_XGB_FEATURE_COLS`（学習時に使った特徴量名リスト）にある列が
`xfeats`（`calc_features_for_xgb()`の出力）に存在しない場合、**警告もエラーも
出さずに一律5.0で穴埋め**していた。通常は学習/推論で同じ関数を呼ぶため発生しないが、
関数名の変更ミス・計算途中の例外握りつぶし・特徴量の追加/削除漏れ等があった場合、
この欠落は完全にサイレントに起こり、2026-07-16のxgb_ensemble_model.pkl事故
（TypeErrorがexceptで握りつぶされ気づかれなかった件）と同じ「見た目は正常に動くが
実際は劣化した予測をしている」状態を再現しうる箇所だった。

#### 対応
`engine.py`に`_check_xgb_feature_coverage(xfeats, feature_cols)`を追加。
欠落列があれば1回だけ（同じ欠落列の組み合わせ単位で）警告ログを出す
（`_XGB_MISSING_FEATS_WARNED`セットで重複警告を抑制、毎頭・毎レースでのログ洪水を防止）。
`calc_all()`のxrow構築直前と`get_xgb_rating()`の2箇所に組み込んだ。
`init_engine()`実行時（モデル/特徴量リストの再ロード時）に警告抑制状態をリセットする。

**挙動は変更しない**（欠落列は従来どおり5.0で穴埋めして推論を止めない）。
今回追加したのは検知・可視化のみ。本番の予測結果には一切影響しない。

#### テスト
`_check_xgb_feature_coverage`の単体テスト3件を追加
（欠落列がある場合に警告が出る／揃っていれば無警告／同じ欠落は2回目以降警告しない）。
全254テスト通過。

---

### 2026-07-20②：少走数レート系特徴量へのベイズ縮小導入

#### 背景
外部記事シリーズ第4回（feature_engineering.pyの有料部分推測）のレビューで、
「ベイズ補正勝率」という設計思想を評価した際、自プロジェクトの`engine.py`にも
同種の問題（過去走が1〜2走しかない馬の成績率が0.0/1.0に極端に振れる）が
複数箇所に存在すると判明したため、既存データのみで改善した。

#### 発見した問題
`calc_course_aptitude_features()`の`_rate()`ヘルパーおよび`calc_features_for_xgb()`内の
`f_dist_fukusho` / `f_course_fukusho` / `f_recent_fukusho` / `f_perf_highpace` /
`f_perf_slowpace`は、いずれも単純な `好走数 / 走数` の生レートだった。
そのため例えば「初挑戦コースで1走だけして1着」の馬は該当特徴量が1.0（＝完璧な適性）に
なり、逆に凡走1走のみなら0.0（＝確実に凡走）になっていた。母集団1〜2件のブレを
そのまま断定値としてXGBに渡していたことになる。
さらに`_rate()`は「未経験（該当走ゼロ）」の場合も一律0.0を返しており、
「経験済みで0%好走」と「未経験」が特徴量上で区別できなかった
（0.0という値だけを見ると木モデルには"確実に悪い"と誤読されうる）。

なお`f_jockey_rate`/`f_trainer_rate`の元になる騎手・調教師別勝率は既に
`runs >= 10`のハードカットオフでガードされており対象外（元々安全）。

#### 対応
`src/features/engine.py`に共通ヘルパー`_bayes_rate(hits_list, prior=0.33, k=3)`を追加。
Beta-Binomialの事後平均と同形の縮小推定
`(hits + prior*k) / (n + k)` を実装し、以下の計算に適用した:
- `calc_course_aptitude_features()`の`_rate()`
  （f_same_course_rate / f_same_turn_rate / f_straight_match / f_uphill_match /
  f_course_type_rate / f_uphill_severity_rate）
- `calc_features_for_xgb()`の f_dist_fukusho / f_course_fukusho / f_recent_fukusho /
  f_perf_highpace / f_perf_slowpace

`k=3`は「事前分布の重みを3走ぶんとみなす」設定。n=0では従来通りの中立値
（0.33 or 0.3）をそのまま返すため、"未経験"時の挙動は変わらない値に統一しつつ、
"経験済みだが少数"のケースでの極端値だけを緩和する。
`_default_course_features()`（コースプロファイル自体が未定義の場合の完全フォールバック）
および`f_beat_market_rate`・騎手/調教師勝率は対象外のまま据え置いた
（スコープを絞り、影響範囲を最小化するため）。

#### 学習/推論パリティ
`calc_features_for_xgb()`・`calc_course_aptitude_features()`はいずれも学習データ生成
（`build_training_data.py`）と推論（`engine.py calc_all()`）の両方から呼ばれる共通関数の
ため、コード変更のみでパリティは自動的に保たれる。

#### ⚠ 本番モデルへの影響（重要・要フォローアップ）
現行の`xgb_fukusho_model.pkl`はこれらの特徴量を**旧・生レート版の分布で学習済み**。
今回の変更は既存特徴量の値の意味を変える（0.0/1.0の極端値が中立値側に寄る）ため、
コードpush直後の次回ワークフロー実行から**推論時の入力分布が学習時と微妙にズレる**。
値は連続的にpriorへ寄るだけの保守的な変更であり暴走リスクは低いと判断して先行デプロイするが、
**次回のColab再学習（XGB retrain）でこの新しい特徴量分布に対して再学習し、
分割点を学習し直すことを推奨**。2026-07-17のf_blood()母父希釈バグ修正と同種の
「特徴量の意味を修正したが、真価は次回再学習で発揮される」ケースとして扱うこと。

#### テスト
既存5テスト（`test_course_aptitude_tokyo_specialist`等）の期待値をベイズ縮小後の値に更新、
新規5テスト追加（`_bayes_rate`単体の境界値・小サンプルほど縮小幅が大きいことの確認・
`calc_features_for_xgb`経由でのfukusho系特徴量の縮小確認）。全251テスト通過。

---

### 2026-07-20：外部記事レビューを踏まえたスクレイピング基盤の改善

#### 背景
競馬AI開発の外部記事（データ取得・スクレイピング設計がテーマ）をレビューし、
自プロジェクトとの比較で「参考にすべき点」として挙げた項目を実装した。

#### 実装内容
1. **429(Too Many Requests)のリトライ対応漏れを修正**（`scripts/_session.py`）
   `Retry`の`status_forcelist`に429を追加。元々500/502/503/504のみだった
2. **JRADBアクセスの共通ラッパー`_jradb_post()`を追加**（`src/scraper/jra_scraper.py`）
   `find_r01_shutuba` / `_try_fetch_shutuba` / `_try_fetch_result` / `fetch_horse_pedigree`
   の4箇所で重複していた「cname/CNAME両キー送信→shift_jis→パラメータエラー判定」を統一。
   headers省略やCNAMEキーのみ送信等の差異がある残り4箇所（`find_r01_result`,
   `find_r01_odds`, `fetch_odds_for_race`, `fetch_results`のStep1）はテストカバレッジが
   薄く、今回は安全側に倒してリファクタ対象から除外した
3. **DESIGN.mdの「やってはいけないこと」表に列位置決め打ちの教訓を追加**
   Stage3再スクレイプでbracket/win_odds/body_weightの列がズレたまま長期間気づかれなかった
   実例を根拠に追加
4. **`docs/history_db_schema.md`を新規作成**（history.dbのスキーマ契約書）
   race_history/horse_historyの全カラムの意味・充足率・既知の欠損を一覧化。
   `corner_3`/`field_size`/`corner_4`が常にNULLであることも明記（カラムの存在と
   データの充足は別問題という誤解を防ぐため）
5. **robots.txt/利用規約の確認を試みたが未達成**：この環境からはjra.go.jpへの
   ネットワークアクセスがブロックされているため確認できず。スキーマ契約書に
   「要ユーザー確認」として記録した

#### 回帰防止
`_jradb_post`統一で動作が変わりうる箇所に直接テストを追加
（`test_jradb_post_*`, `test_try_fetch_shutuba_*`, `test_try_fetch_result_*`）。
全246テスト通過。

---

### 2026-07-18：血統スクレイピング導入直後のCIタイムアウト事故 + 再発防止修正

#### 🔴 発生した事故
2026-07-17②で血統(父・母の父)スクレイピングを導入した直後の最初のweekend.yml実行
（2026-07-18 19:03 JST 実行、run_id 29640180616）で、**30分のジョブタイムアウトに到達し
強制キャンセルされた**。土曜結果34レース・日曜出走予定馬の取得処理自体は正常に進行していたが、
`Commit & Push`ステップに到達する前にキャンセルされたため、**その回で取得した土曜結果・
日曜予想の全データが保存されずに失われた**（GitHub Actionsランナーはジョブ終了時に破棄される）。

比較: 2026-07-11（血統機能導入前）の同ワークフローは11分51秒で完了。
2026-07-18は開始から29分39秒経過した時点でタイムアウトし、日曜出走予定3会場中
2会場（福島・小倉）の出馬表取得を終えた直後、3会場目（函館）の途中でキャンセルされた。

#### 根本原因
`_fill_pedigree()`のキャッシュ設計（history.dbに未記録の馬のみ新規取得）は、
**導入後2回目以降の実行では有効に機能する**が、**導入直後の初回実行では
history.dbの`sire`列が全馬で空のため、実質すべての馬が"新規"扱いになり、
土曜34レース約445頭＋日曜出走予定馬の全頭に対して`accessU.html`への
追加リクエスト（各0.3秒のsleep付き）が発生**し、想定を大幅に超える時間がかかった。

#### 対応（今回実施）
- `src/scraper/jra_scraper.py`: `_fill_pedigree()`に`budget`パラメータ（共有カウンタ）を追加。
  1回のワークフロー実行（`fetch_races_on_date`/`fetch_results`それぞれ）あたりの新規血統取得数を
  `PEDIGREE_FETCH_BUDGET_DEFAULT = 60`件に制限。上限に達した馬は静かにスキップし、
  次回の実行で改めて拾われる（数週間かけて段階的に埋まる設計に変更）
- `.github/workflows/{weekend,sunday-results,friday-predict}.yml`:
  `timeout-minutes: 30 → 40`（安全マージン）
- 回帰テスト2件追加（`test_fill_pedigree_respects_budget` / `_budget_shared_across_calls`）。
  修正前は`TypeError`で失敗、修正後は成功することを確認済み。全242テスト通過

#### ⚠ ユーザーへの影響・要対応事項
- **2026-07-18の土曜結果・日曜予想は保存されていない。ワークフローの再実行が必要**
  （本修正のマージ後に「土曜結果+日曜予想」ボタンを再度押すこと）
- 血統データの充足には、budgetの関係で当初想定よりやや時間がかかる
  （1日あたり最大60件×2パス=120件ペースで新規馬が埋まっていく）

#### 気づいたが今回は対応していない別件
上記ログ中に `⚠ エラータグ処理失敗（予想には影響なし）: 'sqlite3.Row' object has no attribute 'get'`
というエラーを確認。エラータグ自動分類システム（2026-07-14導入）側の既存バグとみられ、
今回のタイムアウト事故とは無関係。予想生成自体には影響しないため今回は未対応。
次回セッションでの調査候補。

---

### 2026-07-17②：血統(父・母の父)スクレイピング実装

#### 背景
前回セッションで発見した「母父(dam_sire)が一度もスクレイピングされておらず`f_blood()`が実質機能不全」
という問題（保留扱いだった）について、実際にJRA公式サイトの構造を調査したところ、
思っていたより低コストで実装できることが判明したため実装した。

#### 調査で判明した重要な事実
- JRA公式サイト（`www.jra.go.jp`）には出馬表(`accessD.html`)・オッズ(`accessO.html`)と
  **同じJRADB内部アクセス方式**で、競走馬の血統情報ページ`accessU.html`が存在する
- **出馬表・結果ページの馬名`<a>`タグの`href`に、その馬の`accessU.html?CNAME=...`への
  直リンクが最初から埋め込まれている**（実機で確認済み）。CNAME逆算のような複雑な処理は不要で、
  出走表を取得するついでにリンクをそのまま拾うだけで済む
- `accessU.html`は`<dt>父</dt><dd>ステルヴィオ</dd>` のような定義リスト(`<dl>`)構造。
  「父」「母の父」はクリーンな種牡馬名がそのまま入るが、「母」「母の母」は
  `"○○ 産駒"`という接尾辞付き表記になる（繁殖牝馬自体は現役馬でないための表示仕様）
- この環境（Claude Code on the web）は**ネットワークポリシーでjra.go.jpへの到達がブロックされている**
  （proxy側のegressポリシーによる拒否。JRA側のブロックではない）。実機検証はユーザーがColabで
  診断ノートを実行する形で行った

#### 実装内容
- `src/scraper/parser.py::parse_horse()`: 馬名リンクの`href`から`pedigree_cname`
  （`accessU.html`のCNAME）を抽出して保持
- `src/scraper/jra_scraper.py`:
  - `fetch_horse_pedigree(sess, cname)`: `accessU.html`から父・母の父を取得（"産駒"サフィックス除去）
  - `_fill_pedigree(sess, horses, hist_db_path)`: 血統を補完するメイン関数。
    **history.dbに既に記録済みの馬（sireが埋まっている馬）は再取得しない**キャッシュ設計。
    1頭の失敗が他馬・レース全体を止めないよう例外は個別に握りつぶす
  - `_parse_shutuba`結果ページパーサー（`parse_result_soup`）の両方で`pedigree_cname`を抽出
  - `fetch_races_on_date`（出馬表取得、予測直前）・`fetch_results`（結果取得、`hist_db_path`引数追加）
    の両方で`_fill_pedigree`を呼ぶよう統合
- `src/utils/db.py`: `horse_history`に`sire`/`dam_sire`カラムを追加（マイグレーション）、
  INSERT/UPDATE文にも反映
- `scripts/weekend.py`: `fetch_results`呼び出しに`hist_db_path`を渡すよう修正
- 回帰テスト7件を`tests/test_scraper.py`に追加（CNAME抽出・血統ページパース・キャッシュ挙動）。
  全240テスト通過

#### スコープ（ユーザー承認済み）
**今後の新規出走馬のみを対象とし、history.dbに蓄積済みの過去馬（数千頭規模）の
一括バックフィルは行わない。** 週次ワークフローの負荷増加を最小限に抑えつつ、
数週間かけて自然にデータが蓄積される設計。バックフィルする場合は別途判断が必要。

#### 今後の運用
- `f_blood()`は前回セッションの修正により、`dam_sire`が空なら父のみで評価・埋まっていれば
  自動的に父70%・母の父30%のブレンド評価に切り替わる前方互換設計のため、**追加のコード変更は不要**。
  データが溜まるにつれて自動的に血統評価の精度が上がっていく
- 数週間後、`horse_history.dam_sire`の充足率を確認し、十分溜まったらXGB再学習（Colab）で
  血統関連特徴量の重要度変化を確認するとよい
- 過去馬の一括バックフィルが必要と判断した場合は、`_fill_pedigree`を使って
  `history.db`の既存レコードから`pedigree_cname`を持たない馬を洗い出し、
  個別に`accessU.html`を叩くバッチ処理を別途作成する（Stage3再スクレイプと同様の位置づけ）

---

### 2026-07-17：追加特徴量の調査 + f_blood（血統）の母父希釈バグ修正

#### 背景
「特徴量を増やしすぎても市場に追いつけない、増やしすぎると過学習」というジレンマを踏まえ、
現行135特徴量とDESIGN.mdの方針を突き合わせて、追加すべき特徴量を調査した。

#### 調査結果サマリ
| 候補 | 状況 | 判定 |
|------|------|------|
| 騎手のコース・馬場別成績 | `(騎手, 競馬場, surface)`キーで既に実装済み | ✅対応済み |
| 調教師のコース・距離別成績 | 通算勝率のみ（条件分けなし） | 🟡未対応（将来候補） |
| 東西所属（美浦/栗東）・遠征適性 | スクレイピング自体が存在しない | 🟡未対応（将来候補、要検証） |
| 馬体重（絶対値・増減） | DBカラムはあるが実データ6.5%しか埋まっていない | 🔵データ品質問題が先（Stage3列マッピング修正が前提） |
| **血統（f_blood）** | **父はSIRE_DB(手打ち約58頭)、母父(dam_sire)は一度もスクレイピングされておらず常に空文字 → 実質機能不全だった** | 🔴**発見・修正済み（今回）** |

#### 🔴 発見：f_bloodの母父側が常にDEF_SIREにフォールバックし父側の実データを希釈していた
- `h.get('dam_sire', '')` は本番で常に空文字（`dam_sire`はどこにもスクレイピングされていない。
  `tune_weights.py`で`'dam_sire': ''`とハードコードされている箇所しか存在しない）
- そのため `dd = SIRE_DB.get('', DEF_SIRE)` は常に汎用平均値 `DEF_SIRE` になり、
  距離・馬場適性の計算で父の実データ（SIRE_DBに登録された約58頭のみ）を毎回30%薄めていた
- 例: ロードカナロア産駒（短距離特化・長距離苦手）でも、母父側の希釈により長距離適性が
  本来より高く評価される方向にバイアスがかかっていた

#### 対応（今回実施）
- `engine.py` の `f_blood()`: `dam_sire` が空の場合はブレンドせず父側のみで評価するよう修正
  （`dam_sire`が将来取得できるようになった場合は自動的に従来通りのブレンド計算に戻る、前方互換設計）
- 回帰テスト追加（`tests/test_features.py::TestFBlood`、修正前後で失敗/成功を確認済み）
- 全233テスト通過

#### 保留にした「母父の本格データ駆動化」（要判断・大きめの変更）
- `horse_history` テーブルには元々 `sire`/`dam_sire` カラム自体が存在せず、過去データに一切蓄積されていない
  （バックフィル不可）
- 母父を取得するには、現状スクレイピングしていない**馬個別プロフィールページへの新規アクセス**が必要
  （出馬表・結果ページには載っていない）→ 週末ワークフローのスクレイピング量・時間・失敗リスクが増える
- 本格的にデータ駆動の血統特徴量（`horse_dist_dict`等と同じ「DBから自動集計」方式）にするには
  ①スキーマ追加 ②新規スクレイピング先の実装 ③数週間〜数ヶ月のデータ蓄積 ④Colabでの再学習が必要
- → **今回は着手せず保留**。DESIGN.mdの「データなしの特徴量追加はしない」原則に従い、
  まず低リスクな希釈バグ修正のみ実施した。着手する場合はユーザーの明示的な判断を仰ぐこと

#### 次点の未対応候補（優先度順、次回検討）
1. 調教師のコース・距離帯別成績（`MIN_SAMPLES`ガード付きで薄いデータは信用しない設計に）
2. 東西所属・遠征フラグ（トレーナー名から(美浦)/(栗東)表記が実際に取得できるか要確認）
3. 馬体重系（Stage3列マッピング修正が前提。既存の「残っている課題」参照）

---

### 2026-07-16：残差モデルがアンサンブルモデル残骸で無効化される重大バグを修正 + リポジトリ整理

#### 🔴 発見した重大バグ（最優先で修正済み）
`data/xgb_ensemble_model.pkl`（127特徴量、通常学習のXGB+LightGBMアンサンブル）が
CLAUDE.mdに一切記載のないまま2026-07-14に単発コミットで紛れ込んでおり（v4ノートブックの
実験の消し忘れとみられる。現行のv4/v5ノートブックのどこにも`train_ensemble`の呼び出しはない）、
これが存在する限り `src/features/engine.py` の `init_engine()` が
**残差学習モデル（135特徴量、本番稼働中のはずのモデル）より優先してこのアンサンブルモデルをロード**していた。

さらに、`_XGB_RESIDUAL`フラグ（135特徴量モデル用）はTrueのまま変わらないため、推論時に
`_XGB_FUKUSHO_MODEL.predict(DMatrix)` が呼ばれるが、実際にロードされているのは
sklearn API の `XGBClassifier` のため **`TypeError: Not supported type for data.<class 'xgboost.core.DMatrix'>`** が発生し、
`calc_all()` の `except Exception:` で握りつぶされて **XGB予測を一切使わずルールベーススコアのみ**
（jockey/distance/pace/trainer中心、rl/maturityはほぼ無効）で予想が生成されていた可能性が高い。
ログにも出ないため気づきにくい状態だった。

再現テスト（`tests/test_residual_learning.py::TestEnsembleResidualConflict`）で実際に
TypeErrorを確認した上で修正:
- `data/xgb_ensemble_model.pkl` / `data/xgb_ensemble_cols.json` を削除
- `init_engine()` に safety guard を追加: `_XGB_RESIDUAL=True` の場合はアンサンブルロードを
  スキップする（将来また同様の実験ファイルが紛れ込んでも残差モデルが優先されるようにする）
- 回帰テストを追加（修正前は失敗、修正後は成功することを確認済み）

**⚠ 影響範囲**: このバグがいつから本番に影響していたか(2026-07-14のコミット以降、
次回ワークフロー実行までの間)は不明。7/19-20開始予定の残差モデルのフォワードROI検証は、
このバグ修正後のコードで初めて正しく実施されることになる。

#### リポジトリ整理（ユーザー依頼）
不要ファイル・デッドコードの調査を行い、慎重に確認した上で以下を削除:

- **デッドコード**: `src/features/correction.py` の `apply_correction()` / `classify_distance()`
  （呼び出し元なし。同等ロジックは`update_correction_table()`に統合済み）
- **未使用コード**: `src/betting/make_bets.py` の `_BET_SELECTOR` / `_BET_SELECTOR_LE`
  ロード処理（ロードされるが一切使われていなかった。前セッションで評価した
  「KEIBA_券種選択モデル.ipynb」の学習結果を読む処理だったが、そのモデル自体が
  実用に耐えないと判定済み）。`src/utils/model_registry.py` の `MODEL_FILES` からも該当エントリ削除
- **未参照ファイル**: `data/month_suffix_map.json`（現行コードから一切参照なし。
  唯一の参照元は完了済みの`KEIBA_Stage3_rescrape.ipynb`のみだった）
- **古いノートブック**（CLAUDE.mdの「Colabノートブック構成」表に記載の現行5本以外、
  明確な後継が存在する版・完了済み一回限り作業・前セッションで評価済みの実用に耐えないモデル）:
  `KEIBA_金曜_v6/v7.ipynb`, `KEIBA_土日_v6/v8.ipynb`, `KEIBA_日曜結果_v7.ipynb`,
  `KEIBA_チューニング_v2.ipynb`, `KEIBA_XGB_retrain.ipynb`（無印/v2/v3/v4）,
  `KEIBA_券種選択モデル.ipynb`, `KEIBA_Stage3_rescrape.ipynb`
  （いずれもgit履歴には残るため復元可能）

#### 気づいたが今回は対応していない不整合（要フォローアップ）
- `data/calibrator.pkl`（ルールベース用キャリブレーター）がCLAUDE.md記載にも関わらず
  実ファイルが存在しない（`os.path.exists`でガードされているためクラッシュはしない）
- `data/rating_temperature.json` の中身がCLAUDE.md記載と食い違う
  （記載: B2=T1.0・gumbel_rating=2.5キーあり / 実際: B2=T5.0のみ・gumbel_ratingキーなし）
  → bet_optimizer.pyがgumbel_rating欠損時にフォールバック定数（2.5）で動いているか要確認

---

### 2026-07-14②：残差学習モデル本番投入 + v5ノートブック

#### 概要
7/12に実装した残差学習モードをColabで実行し、**本番モデルを残差モデルに切替**。
f_popularityを除外し、AIが「市場からのズレ」だけを学習する構造に移行。

#### 残差モデルとは
- **旧モデル**: f_popularity（重要度24.6%）が支配 → 予測 ≈ 市場オッズのコピー → EV ≈ 1.0
- **残差モデル**: logit(市場確率) を固定ベースラインとして渡し、モデルは「市場からのズレ」だけを学習
  - `logit(p) = logit(p_market) + f_AI(非市場特徴量)`
  - f_popularity を特徴量から除外（135特徴量、旧136から-1）

#### Colab実行結果（v5ノートブック）
| 指標 | 通常モデル | 残差モデル | 差分 |
|------|-----------|-----------|------|
| AUC（同一split） | 0.8017 | **0.7974** | -0.0043 |
| 維持率 | — | **99.5%** | — |
| 特徴量数 | 136 | 135 | -1（f_popularity除外） |
| 学習データ | 150,739行 | 同左 | — |
| Val期間 | 5/24〜6/20 | 同左 | — |

#### 残差モデル重要度 Top10
```
f_cl_rank                    6.68%   ← クラス順位（最重要に浮上）
f_pos_avg_3                  3.64%   ← 直近3走平均着順
cl_f_dist_fukusho_rank       2.80%   ← 距離適性順位
f_member_level_avg           2.48%   ← メンバーレベル
f_pop_last                   2.40%   ← 前走人気（過去の市場評価、リークではない）
rl_f_member_level_avg_rank   2.33%   ← メンバーレベル相対
f_last2_pos3c                2.04%   ← 2走前複勝圏
f_agari_ability              2.02%   ← 末脚の強さ（Phase1距離適性から浮上）
f_time_diff_avg              1.72%   ← タイム差平均
f_pop_avg                    1.68%   ← 平均人気
```

#### 維持率99.5%の意味
- f_popularityは重要度24.6%だったが、除外してもAUCが0.5%しか落ちない
- = **AIの予測力の99.5%は市場コピーではなく独自情報に基づいている**
- = 馬券的エッジの可能性がある（予測が市場と独立 → EV計算に実質的な差が出る）

#### 本番反映状況
- `data/xgb_fukusho_model.pkl`: 残差モデル（xgboost Booster形式、UBJ）
- `data/xgb_feature_cols.json`: `"residual": true`, 135特徴量, val_auc 0.7974
- `data/xgb_calibrator.pkl`: 残差モデル用に再キャリブレーション済み
- `data/xgb_fukusho_model_residual.pkl`: 同一（本番と同じ）
- GitHub main に全てpush済み（2026-07-14 13:53 JST）
- engine.py の `_XGB_RESIDUAL` フラグが自動検出し、推論時にbase_marginを適用

#### v5ノートブック（KEIBA_XGB_retrain_v5.ipynb）
v4のセル6（残差学習実験）を拡張し、完全なワークフローを統合:
- セル7: 通常 vs 残差のレース単位AUC比較・特徴量重要度の対比
- セル8: 条件付き自動切替（残差AUC >= 通常AUC × 95%で発動）
  - バックアップ → ファイルコピー → キャリブレーション再実行
- セル9: 統合テスト（_XGB_RESIDUAL検出 + cal_prob合計チェック）
- セル10: pushメッセージにモード(normal/residual)を明記

#### 今後のアクション
| 優先 | 内容 | 前提 |
|------|------|------|
| **最高** | **フォワードROIで残差モデルのエッジ検証** | 次の週末（7/19-20）から自動蓄積 |
| 高 | 温度再校正（softmax T=3.5, gumbel T=2.5） | 残差モデルのフォワードデータ4週分 |
| 中 | 条件帯別のAI優位分析（どこでエッジが出るか） | divergence_weekly + 残差モデルのデータ |
| 中 | B2_ndcgモデルの残差学習版 | 単勝用B2も市場コピー排除すべきか検討 |

#### ⚠ 注意事項
- 旧モデル（通常版）は `*.bak_before_residual` でDriveにバックアップ済み
- 残差モデルの `.pkl` は xgboost Booster の UBJ形式（pickleではない）
  - `xgb.Booster()` + `.load_model()` でロード（`pickle.load()` は不可）
  - engine.py は `_XGB_RESIDUAL=True` 時に自動対応
- `calibrate_xgb.py` は残差モデル非対応（セル8で直接キャリブレーション実行で回避済み）

---

### 2026-07-14：エラータグ自動分類・週次補正システム

レース後に「AIがなぜ外したか」を12種のタグで自動分類し、翌週の予想に自動反映する仕組みを実装。

#### 2段階の活用
| | 処理 | 反映タイミング |
|--|--|--|
| **即時補正** | 条件別の補正係数を自動計算 → engine.py のスコアに乗算 | **翌週から自動** |
| **モデル学習** | タグ発生率を特徴量化（f_et_*）→ XGB再学習 | **月1再学習時** |

#### 12種のエラータグ
| タグ | 条件 |
|------|------|
| pace_miss | ペース予測と実際が不一致 |
| escape_win | 逃げ馬がAI低評価で勝利 |
| position_bias | 内/外枠が偏って好走 |
| style_miss | AI低評価の脚質が好走 |
| class_miss | 昇級馬がAI予想外に好走 |
| form_miss | 休み明け馬がAI予想外に好走 |
| dist_short_win | 距離短縮馬が好走 |
| dist_ext_win | 距離延長馬が好走 |
| heavy_upset | 重/不良で人気薄が好走 |
| mare_upset | 牝馬がAI予想外に好走 |
| young_upset | 3歳馬が古馬戦でAI予想外に好走 |
| jockey_switch_win | 乗り替わりで好走 |

#### 実装内容
- `src/features/error_tags.py` 新規作成
  - `classify_race_tags()`: 1レースのエラータグを分類
  - `accumulate_tags()`: 週次蓄積ファイルに追加 + 補正係数再計算
  - `get_correction_factor()`: 条件別補正係数を返す（馬個別ボーナス付き）
  - `calc_error_tag_features()`: XGB再学習用の特徴量生成
  - `process_weekly_error_tags()`: sunday_results.py から呼ばれる週次処理
- `src/features/engine.py`: calc_all の softmax 直前でエラータグ補正を適用
- `scripts/sunday_results.py`: エラータグ処理ステップ追加（失敗してもワークフロー不停止）
- `tests/test_error_tags.py`: 28テスト新規

#### 蓄積先
- `data/error_tags_weekly.json`（累積、同一race_idは重複防止）

#### 補正の仕組み
- venue × surface × 距離帯 × 馬場状態 の条件キーでタグ発生率を集計
- 条件内のタグ発生率が全体ベースラインの1.3倍以上 → 補正係数を引き上げ
- 馬個別マッチング: 該当パターンの馬（逃げ馬、短縮馬等）にさらにボーナス
- MIN_SAMPLES = 20件（データ不足の条件は補正しない）

---

### 2026-07-12：残差学習（base_margin）モード実装

Fableの提案に基づき、XGBの学習構造を変更するオプションを追加。
現行モデル（f_popularity含む119特徴量）はそのまま維持し、**並行で残差学習モデルを試せる**設計。

#### 概要
- **現行**: f_popularity がXGBの1特徴量 → モデルが市場をコピー（重要度24.6%）→ 予測≈市場 → EV出ない
- **残差学習**: logit(p_market) を固定ベースラインとして渡し、モデルは「市場からのズレ」だけを学習
  - `logit(p) = logit(p_market) + f_AI(非市場特徴量)`
  - 出力が正 = 市場が過小評価 = AIのエッジ

#### 実装内容
- `src/tools/train_xgb.py`:
  - `train_xgb(base_dir, residual=True)` で残差学習モード
  - `_popularity_to_base_margin()`: 人気順位 → Zipf分布 → logit 変換
  - f_popularity を特徴量から除外し、xgboost.train の base_margin に設定
  - 残差モデルは `xgb_fukusho_model_residual.pkl` / `xgb_feature_cols_residual.json` に保存
  - `xgb_feature_cols_residual.json` に `"residual": true` フラグ
- `src/features/engine.py`:
  - `_XGB_RESIDUAL` グローバルフラグ（init_engine で自動検出）
  - calc_all の Pass 2 で `_XGB_RESIDUAL=True` なら base_margin を DMatrix に設定して推論
- `tests/test_residual_learning.py`: 11テスト新規

#### Colabでの使い方
```python
# 1. 学習データ再生成（既存のまま）
from src.tools.build_training_data import build_training_data
build_training_data(BASE_DIR)

# 2. 残差学習モデルを学習
from src.tools.train_xgb import train_xgb
result = train_xgb(BASE_DIR, residual=True)
# → xgb_fukusho_model_residual.pkl / xgb_feature_cols_residual.json が生成

# 3. 現行モデルとAUC比較
print(f"残差: {result['auc']}")
print(f"現行: {result['old_model']}")  # 残差の旧モデルがなければ空

# 4. 本番に切り替える場合（残差モデルが優れていた場合のみ）
import shutil
shutil.copy('data/xgb_fukusho_model_residual.pkl', 'data/xgb_fukusho_model.pkl')
shutil.copy('data/xgb_feature_cols_residual.json', 'data/xgb_feature_cols.json')
# → init_engine が "residual": true を検出し、推論時に自動で base_margin を適用
```

#### 判定基準
- AUC が現行（0.8219）と同等以上 → 残差学習で市場コピーを排除しても精度維持 = エッジの源泉がAI側にある
- AUC が大幅低下 → AI独自の予測力が弱い = 市場コピーに依存していた（悪いニュースだが重要な事実）
- **feature_importance から f_popularity が消えること自体が成功の指標**（市場コピーの排除）

#### 安全性
- 現行モデル（xgb_fukusho_model.pkl）には**一切触れない**
- 残差モデルは別ファイル（_residual サフィックス）に保存
- 本番切替は手動コピーが必要（自動では切り替わらない）

---

### 2026-07-10 セッション②：直前オッズ変動時の買い目・推奨・急騰マーク対応

直前オッズ取得時に、オッズ変動を反映した3つの新機能を `index.html` に実装。

#### 1. 買い目変更（recalcGumbelBets RL化）
- クライアント側の `recalcGumbelBets()` を `bet_optimizer.py` と同じRL上位ベースロジックに更新
- 単勝: RL上位3頭からオッズ妙味(2〜30倍)×EV>=1.0の1点（旧: RL1固定）
- 複勝: RL上位5頭からRL順で最大2点、EV>=1.0足切り（旧: RL上位3頭からEV>=0.8）
- 馬連: RL上位5頭の組み合わせ、RL3含む優先、EV>=1.0、最大5点（旧: RL1×2の1点のみ）

#### 2. 推奨マーク更新（updateRecFlag）
- `updateRecFlag(race)` 新関数
- 推奨取消: RL1のオッズが1.5倍未満（ガチガチ＝妙味なし）or RL上位3頭全員EV<0.8
- 推奨追加: 元々非推奨でもRL上位3頭にEV>=1.2×オッズ2〜30倍の馬が出現
- レースヘッダーに「推奨取消」「NEW推奨」バッジ表示
- 理由テキスト付き（例: 「RL1が1.3倍（妙味なし）」）

#### 3. 人気急上昇馬マーク
- `updateOddsAndEV()` で朝オッズ→直前オッズの下落率を算出
  - `hot`: 30%以上下落 かつ 3倍以上変動（例: 15倍→8倍）→ 赤色「急騰」バッジ
  - `warm`: 20%以上下落 かつ 2倍以上変動 → オレンジ「上昇」バッジ
- 馬名の横にバッジ表示
- レース詳細上部にサマリー（「人気急上昇: #3 ナントカ(15.0→8.2)」）

#### ボーナス: オッズ変動サマリー
- 各レースの馬テーブル上部に、推奨変更理由＋急騰馬のサマリーを赤枠で表示

---

### 2026-07-10 セッション：EV買い目のRL上位ベース再設計 + 大掃除・KPI導入

#### Phase A: 大掃除（PR #46 マージ済み）
- pairwise モデル完全削除、value_gap 廃止、dual_model 凍結

#### Phase B: 市場ベースラインKPI（PR #47 マージ済み）
- `calc_model_kpi()` 追加（AI vs 市場 log-loss）、`tests/test_model_kpi.py` 10テスト

#### ワークフロー修正（PR #48 マージ済み）
- `data/xgb_ranking_pairwise.pkl` の LFS ポインタ不整合で全ワークフロー失敗 → ファイル削除で修正

#### 日付・買い目表示修正（PR #49 マージ済み）
- 金曜予想の日付が当日(金)になる問題 → saturday でも +1 日に修正
- 単勝/複勝の人気制限（暫定的な小手先対応、下記で本格修正）

#### EV買い目のRL上位ベース再設計（PR #50 マージ済み）
- `recalcGumbelBets` / `bet_optimizer.py` をRL上位ベースに書き換え
- 全137テスト通過

#### ポーリングキャッシュ修正（PR #51 マージ済み）
- raw.githubusercontent.com 優先でCDNキャッシュ問題を解消

---

### 2026-07-09 セッション：市場特徴量モデル再学習成功・本番反映（複勝AUC 0.68→0.82）

前セッション（07-06②）で追加した市場特徴量4個をColabで再学習し、**複勝AUCが市場と同等以上に到達**。
セル6で GitHub main にプッシュ済み。**次の週末ワークフローから本番稼働**。

#### 再学習結果（KEIBA_XGB_retrain_v3.ipynb / Colab）
| 指標 | 旧モデル | 新モデル | 判定 |
|------|--------:|--------:|------|
| 複勝AUC (val 06-06〜06-14) | 0.7941 | **0.8219** | 判定基準0.80突破 ✅ |
| Brier | 0.1805 | 0.1663 | 改善 |
| LogLoss | 0.5306 | 0.4965 | 改善 |
| 特徴量数 | 106 | **119** | 市場特徴量4個+その他 |

- **feature_importance トップ: `f_popularity` 24.60%（単独首位）**、`f_pop_last` 3.20%（3位）
  → 市場情報がモデルに強力に取り込まれた。「AIは市場を見ずに0.69の予測」状態を解消
- train_xgb は AUC改善時のみ自動採用する設計 → 新モデルが正式採用され `xgb_fukusho_model.pkl` 更新
- キャリブレーション（run_xgb_calibration）も自動実行済み。Test ECE 0.0367（Train 0.0011よりやや高い＝軽度の過学習兆候だが実用範囲）。cal_prob合計 平均2.664（理論3.0よりやや過小）

#### 決定的検証：同一期間で 新AI vs 市場 のAUC比較
```
複勝(3着内) AUC 同一期間比較（val 2026-06-06〜06-14, 約130レース）:
  新AI : 0.8219
  市場 : 0.8148   （市場スコア = -popularity）
```
- **AIが市場を +0.0071 上回った**。旧モデルの「複勝AUC 0.68 < 市場0.77（構造的敗北）」から逆転
- ⚠ **ただし差0.007は小さい**。N≈130だとAUCの標準誤差±0.02程度 → 統計的には「市場と同等〜わずかに上」が誠実な結論。「明確に超えた」と断言するにはフォワードでN=300超・DeLong検定が必要
- ⚠ f_popularity 重要度24.6%＝予測力の大半は市場のコピー。残り特徴量（AI残差）が0.007を上乗せ。この残差が本物かは要検証
- ⚠ **AUCで市場と並んでも馬券では控除率20-25%ぶん負ける**のが数理。勝つには「市場が間違える特定領域をAIが当てる」必要。0.007がその領域を指す可能性

#### 本番反映確認（origin/main）
- `data/xgb_feature_cols.json`: val_auc=0.8219, 特徴量119, trained_at 2026-07-09 12:50
- 市場特徴量4個すべて反映済み: f_popularity / f_pop_last / f_pop_avg / f_beat_market_rate
- コミット `b7e7aba model: retrain 119feat`（Colab セル6 が Contents API で直接push）

#### 次のアクション（データ蓄積後・急がない）
| 優先 | 内容 | 前提 |
|------|------|------|
| 高 | 新AI vs 市場のAUC継続追跡・DeLong検定 | フォワード N=300超（数週後） |
| 高 | 乖離レース分析（市場と違う本命で新AIが当たるか）| 新モデルのフォワードデータ |
| 中 | softmax T=3.5 / Gumbel T=2.5 の再フィット | 新モデルのフォワードデータ（下記⚠） |

#### ⚠ 温度の再校正が必要（重要・今はまだやらない）
- 現行の `softmax T=3.5`（engine.py）と `Gumbel rating T=2.5`（bet_optimizer.py `gumbel_rating`キー）は
  **旧モデル（AUC0.69）のフォワードデータでフィットしたもの**
- 新モデルは市場に寄って過信が減ったため、これらの温度は**強すぎる（フラット化しすぎ）可能性**
- 正しい手順: 新モデルで数週フォワードデータを取ってから再フィット。**今変えると二重補正リスク**
- また 07-06② の注記どおり、fukusho T=0.7（4-5月val）も新モデルでは要再校正

---

### 2026-07-06 セッション②

### 2026-07-06 セッション②：改善3点実装（P3リーク修正・P2温度校正・P1市場特徴量）

Gumbel検証の結論を受け **「現行方向性を維持したまま改善」** の3施策を実装。
方針: 市場をモデルに取り込み「モデル = 市場 + AI残差」構造にする（パイプライン不変）。

#### P3: shadow.py リーク修正（完了・即効）
- `record_all_shadow_bets` が calc_all を事後再実行するのを廃止
- race_predictions（朝の予想スナップショット）から RL1-3 を取得
- 朝予想がないレースは記録しない（リーク行を作らない）
- winner_pop はオッズ欠損時 None（従来は常に1になるバグ）
- `tests/test_shadow.py` 新規5テスト
- **⚠ 2026-07-06以前の shadow_bets 行はリーク済みデータ。集計から除外すること**

#### P2: Gumbel rating 温度校正（完了・即効）
- `make_bets_v2` 非feat_dfパス: rating（XGBマージン）を T=2.5 で割ってから
  `simulate_race` に渡す（`bet_optimizer.py`）
- 理由: T=1 のままだと P(勝利)=softmax(rating) が過信
  （フォワード実測: Gumbel RL1平均35% vs 実勝率16%）
- T=2.5 はフォワード96レースで log-loss 最適（RL1平均17.7% ≈ 実測15.6%、ECE 0.0095）
- `rating_temperature.json` に `gumbel_rating` キー追加（フォールバック定数 2.5）
- ⚠ 温度校正は「確率を正直にする」効果。**エッジは作らない**。
  買い目点数は _build_trio の最低点数保証と box モードのEV免除により大きくは減らない
- ⚠ 既存の fukusho T=0.7（4-5月val）はフォワードデータと矛盾（さらに過信を悪化させる方向）。
  dual_model パス使用時は要再校正

#### P1: 市場特徴量を XGB に追加（コード完了・**再学習待ち**）
- **発見: 従来モデルの106特徴量に市場情報（オッズ・人気）がゼロ**。
  AIは市場（AUC 0.83）を見ずに 0.69 の予測をしていた
- **データ検証: horse_history.popularity は 99.2% 充足**（win_odds は0%欠損のため人気を使う）
- 追加特徴量4個（`engine.py calc_features_for_xgb` 末尾）:
  | 特徴量 | 意味 |
  |--------|------|
  | f_popularity | 現走人気（予測時=朝オッズ由来、学習時=確定人気） |
  | f_pop_last | 前走人気 |
  | f_pop_avg | 直近5走の平均人気 |
  | f_beat_market_rate | 着順<人気だった率（市場の見立てを超えた率） |
- `calc_all`: popularity導出を Pass 1 の**前**に移動（xfeatsが参照するため）。
  確定人気が既に入っている馬は上書きしない
- `get_history_from_db`（予測側）と `build_training_data._get_history_before`（学習側）の
  両方に popularity を追加（学習/推論パリティ確保）
- 現行モデルは `xgb_feature_cols.json` の106列しか読まないため、
  **再学習まで新特徴量は無害に無視される（デプロイ安全）**
- `tests/test_market_features.py` 新規8テスト

#### 次のアクション: Colab再学習（ユーザー作業）
KEIBA_XGB_retrain_v3.ipynb（または チューニングノート）で:
```python
# 1. 学習データ再生成（新特徴量入りCSVを作る）
from src.tools.build_training_data import build_training_data
build_training_data(BASE_DIR)

# 2. XGB再学習
from src.tools.train_xgb import train_xgb   # 関数名はノート参照
# → 新しい xgb_fukusho_model.pkl / xgb_feature_cols.json が生成される

# 3. キャリブレーション再実行
from src.tools.calibrate_xgb import calibrate_xgb

# 4. 確認: AUCが市場（0.83）に近づいたか
#    xgb_feature_cols.json の val_auc をチェック。
#    0.80+ になっていれば市場情報の取り込み成功
```
**判定基準**: 再学習後 val_auc が 0.80 を超えなければ市場特徴量が効いていない
（feature_importance で f_popularity 系の寄与を確認する）。
成功したら次の週末からフォワードテストで「モデル vs 市場」のAUC差を追跡。

---

### 2026-07-06 セッション：Gumbel買い目の実力検証（重大な結論）

#### 検証の背景
ev_direct に識別力がないと判明したため、唯一の馬券根拠となる Gumbel シミュレーション
買い目（make_bets_v2 / 📊EV買い目）が本当に機能しているかを検証した。

#### 結論：**Gumbel買い目も市場に勝てていない（ユーザーの懸念どおり）**

#### 判明した事実

**① shadow_bets の ROI 135% はデータリーク（信用不可）**
- shadow_bets は結果取得時に calc_all を「再実行」して RL1 を決めている
  （shadow.py）。このとき馬の win_odds は**最終確定オッズ**
  → AIの特徴量に市場の最終判断が混入した事後予測。
- 証拠: shadow RL1 と朝予想 RL1 の一致率は **16%**（11/68）しかない。
- 朝予想スナップショット（race_predictions）ベースの真の RL1 単勝 ROI = **90.9%（損失）**。
- **⚠ 今後 shadow_bets / stats.json の ROI 数値を成績として扱わないこと。**
  （修正案: shadow.py が race_predictions から朝の RL1-3 を引くよう変更する）

**② 3つの買い目系統の区別（混同注意）**
| 系統 | 計算 | アプリ表示 |
|------|------|-----------|
| bets | make_bets()（ルールベース） | 通常の買い目欄 |
| gumbel_bets | make_bets_v2()（Gumbel×EV） | 📊EV買い目 |
| ev_direct | pn × odds | Val列 |

**③ Gumbel の数理的性質（重要）**
- Gumbel-Max トリックにより P(勝利) = softmax(rating, T=1) と**数学的に等価**
- → Gumbel の順位づけ = rating（XGBマージン）の順位づけ = モデルの識別力そのもの
- → シミュレーションを何回回しても**モデル以上の識別力は生まれない**
- 本番パス（app_json.py）は feat_df なしで呼ぶため rating = A_fukusho のマージン

**④ 識別力の直接比較（AUC, 98レース）**
| 予測対象 | AI | 市場(1/odds) |
|---------|----|----|
| 1着 | 0.693 | **0.833** |
| 3着内 | 0.676 | **0.766** |
市場が圧倒的に上。AIが市場と逆張りした部分はほぼ間違い。

**⑤ Gumbel買い目バックテスト（95レース・本番パス近似再現）**
rating を isotonic 逆変換で再構築し、本番と同じ
simulate_race(3000) → estimate_payouts → build_optimal_bets を実行:
| 券種 | 点数 | 的中率 | ROI |
|------|------|--------|-----|
| 単勝 | 55 | 1.8% | **32.5%** |
| 複勝 | 102 | 19.6% | 103.1%（推定配当） |
| 馬連 | 267 | 0.4% | 20.7%（推定配当） |
| 三連複 | 544 | 1.3% | 48.1%（推定配当） |
| **合計** | 968 | — | **45.5%（大損失）** |
- Gumbel RL1 確率: 平均25.9% vs 実勝率16.5% → 過信
- EV選択は「AIと市場の乖離が最大の馬」を選ぶ = AUCで劣る側の最大の間違いを選ぶ構造

**⑥ 唯一の非損失ポケット: AI×市場一致領域**
| RL1の市場人気 | N | 勝率 | 複勝率 | 単勝ROI |
|--------------|---|------|--------|---------|
| 1-2番人気（一致） | 47 | 23.4% | 66.0% | 66.0% |
| 3-4番人気 | 20 | 10.0% | 40.0% | 61.5% |
| 5番人気以下（乖離） | 30 | 10.0% | 30.0% | 149.7%※ |
※乖離帯の149.7%は3的中のみ（12-18倍が3本）による偶然の可能性大。N=100超まで判断保留。

#### 戦略的含意
- 公式データのみの現行モデル（AUC 0.69）では市場（0.83)に構造的に勝てない
- ev_direct も Gumbel も「モデル確率 × 市場オッズ」の構造上、モデルが市場に劣る限り
  どんな買い目最適化でも長期プラスにならない
- **方向性: 選択肢B（市場利用型）へ** — AI単体で勝負せず、
  (1) 買うレースを絞る（一致領域・得意領域のみ）
  (2) 外部情報（不利メモ・調教・Opus分析）で補強
  (3) データ蓄積を続け識別力改善は中期課題として継続

---

### 2026-07-05 セッション：精度分析・cal_prob修正・popularity導出・T=3.5校正

#### 精度分析結果（98レース 2026-06-27〜07-04）
| 指標 | 値 | 備考 |
|------|-----|------|
| RL1 実勝率 | 16.5% | 市場1番人気 33.3% の半分 |
| RL1 平均人気 | 3.9番人気 | AIと市場が常に違う本命を推す |
| ECE (旧T=2.0) | 0.0357 | win_prob 30%+が実際15%と乖離 |
| ECE (T=3.5) | **0.0169** | 52%改善 |
| ev_direct (EV>=1.3) 勝率 | 8.5% ≒ baseline | **識別力なし** |

#### 実施した修正
1. **softmax温度 T=2.0→T=3.5** (`src/features/engine.py`)
   - 理由: RL1予測33%→実際16%の乖離解消。スコアスプレッド7.5で42倍→8.5倍に圧縮
   - log-odds/T=3.0と実質同等。急いで切り替える必要なし
2. **popularity自動導出** (`src/features/engine.py` calc_all末尾)
   - win_odds昇順で popularity=1,2,3... を設定（低オッズ=1番人気）
   - save_race_predictionsがh.get('popularity', 99)で拾う → 正しく保存
3. **フィルタ閾値調整** (`src/betting/ev_filter.py`)
   - min_gap: 0.06→0.03（T変更でpn差が縮まるため）
   - min_win_prob: 0.12→0.10（RL1確率が18%前後になるため）
4. **DB自動修復** (`src/utils/db.py` init_db)
   - cal_prob>1.0を0.99にキャップ（旧market_correction残骸）
   - popularity=99をtansho_odds順位で補填
   - correction_enabled/factorをNULLクリア
   → **次回ワークフロー実行時に自動発動**

#### ev_direct について重要な理解（EV信頼性）
`ev_direct = pn(=win_prob) × tansho_odds` は**選択シグナルとして機能しない**。
- EV>=1.3でも実勝率8.5%≒baseline8%。閾値を上げても改善しない。
- 理由: softmax win_probは「フィールド内相対順位の確率化」であり、市場オッズが織り込む「絶対的勝率」とは別物。掛け算に識別力が生まれない。
- **役割**: 明らかなNon-value(EV<1.0)を除外する粗フィルタとしてのみ有効。
- **買い目の根拠**: Gumbel simulation EV（make_bets_v2）を使うこと。T変更の影響を受けない独立パス。

#### cal_prob と win_prob の役割分担（混同禁止）
| | cal_prob | win_prob |
|--|--|--|
| 入力 | XGB.predict_proba() | raw_prob×10（スコア化） |
| 処理 | IsotonicCalibrator | softmax(T=3.5) |
| 意味 | 個馬独立の複勝確率 | フィールド相対の勝率 |
| 制約 | sum≠1（12頭で2.1-3.3） | sum=1 |
二重校正ではなく異なる量を異なるツールで校正。

#### 中期アジェンダ（データ蓄積後）
| 優先度 | 内容 | 必要データ |
|--------|------|-----------|
| 低 | log-odds vs raw×10 AUC比較 | 500レース以上 |
| 低 | 15-20%帯×10-20倍ポケット確認 | N=15→50件以上で判断 |
| 低 | 中距離1800-2200m / 函館の改善測定 | 特徴量追加後100件以上 |

#### 残 popularity DB修復
- init_dbの自動修復コードは次回ワークフロー実行時に発動
- 6/27以前の古いレコード(pop=99)は次回実行まで未修正
- Colabで修復したい場合: `from src.utils.db import init_db; init_db(BASE_DIR)` を実行

### 最終更新: 2026-07-02 セッション③

---

### 2026-07-02 セッション③：総点検・タスク5完了・B2モデル有効化（PR #31 マージ済み）

#### タスク5: gumbel_bets をアプリ表示に接続（完了）
- `src/betting/app_json.py`:
  - `to_app_json()` に `base_dir=None` パラメータ追加（後方互換）
  - `make_bets_v2(n_sims=3000)` を try/except で各レースに呼び出し、`gumbel_bets` を race エントリに追加
  - `_format_gumbel_bets()` ヘルパー: 単勝/複勝/馬連は馬番・EV・推定配当を表示、三連複は点数・配当レンジ・合成オッズをまとめて1行表示
- `index.html`: 既存 `bets` 表示の直下に「📊 EV買い目」セクション追加（緑バッジで EV 表示）
- 次回ワークフロー実行後から latest.json に `gumbel_bets` が出力される

#### 総点検（6項目）結果
| 項目 | 結果 |
|------|------|
| ① データ整合性 | LFS環境のため Colab で要確認 |
| ② モデル一貫性 | B2ファイル未存在を検出→本セッションで解決 |
| ③ パイプライン通し | feat_df→dual_probs→optimal_bets→gumbel_bets→latest.json→アプリ 接続済み |
| ④ エッジケース | 5頭三連複/取消/新馬/空odds すべて安全 |
| ⑤ デッドコード | classify_chaos_grade 削除・staleコメント修正 |
| ⑥ 学習/推論パリティ | fillna(5.0)/add_relative_features 一致確認。データリーク再発なし |

#### デッドコード削除（完了）
- `src/betting/make_bets.py`: `classify_chaos_grade()` 削除（外部から未使用）
- `src/betting/app_json.py`: stale コメント修正
- `tests/test_betting.py`: 対応テスト2件削除
- 87テスト全通過

#### B2モデル学習・有効化（完了）
Colab（KEIBA_チューニング_v1.ipynb のセル1直後に追加）で実行：

```python
# B2 学習
from src.tools.train_ranking_model import train_ranking_model
train_ranking_model(BASE_DIR, objective='rank:ndcg', model_suffix='ndcg')

# 温度校正
from src.betting.rating_calibration import calibrate_all_models
calibrate_all_models(BASE_DIR, val_start='2026-04-01', val_end='2026-05-31')
```

**校正結果（実測）**:
| モデル | T | ECE |
|--------|---|-----|
| A fukusho | 0.7 | 0.0136 |
| B2 ranking_ndcg | **1.0** | **0.0043** |
| pairwise | 5.0 | 0.0064（不使用）|

⚠ B2 の最適温度は 0.7 ではなく **T=1.0** だった。`rating_temperature.json` に正しく保存済み。`dual_model.py` はこのファイルを参照するためコード修正不要。

**push 方法（Drive は git リポジトリではないため）**:
```python
from google.colab import userdata
import subprocess, shutil

PAT = userdata.get('GITHUB_PAT')
REPO = '/content/keiba_ai_push'
subprocess.run(f'git clone https://{PAT}@github.com/hanagenuku/keiba_ai.git {REPO}', shell=True)
for f in ['xgb_ranking_ndcg.pkl', 'xgb_ranking_feature_cols.json', 'rating_temperature.json']:
    shutil.copy(f'{BASE_DIR}/data/{f}', f'{REPO}/data/{f}')
cmds = [
    f'git -C {REPO} config user.email "bot@keiba_ai"',
    f'git -C {REPO} config user.name "keiba_ai bot"',
    f'git -C {REPO} add data/xgb_ranking_ndcg.pkl data/xgb_ranking_feature_cols.json data/rating_temperature.json',
    f'git -C {REPO} commit -m "Add B2 (rank:ndcg) model and temperature calibration"',
    f'git -C {REPO} push origin main',
]
for cmd in cmds:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout or r.stderr)
```

#### 未解決課題
| 課題 | 優先度 | 備考 |
|------|--------|------|
| value_gap 削除 | 低 | アプリの Val 列表示に使用中。EV表示に置き換えてから削除 |
| DB記録の bets = 推定オッズ | 低 | 設計上の制限。ROI集計時に注意（CLAUDE.md注記済み） |
| history.db 日付カバレッジ確認 | 中 | LFS環境のため Colab で確認 |

---

### 2026-07-02 セッション②：馬券エンジン（Gumbel確率×EV買い目生成）

タスク0-4 を実装。既存の `make_bets()` を置き換えない段階移行パス。

#### タスク0: パフォーマンス確認結果
- 16頭 × 2系統 × 20,000回 = **0.232秒/レース**、36R = **8.3秒**
- GitHub Actions タイムアウト問題なし。n_sims 削減不要。

#### タスク1: market_odds_map 状況
- `build_market_odds_from_races()` → `to_app_json()` は接続済み（表示レイヤー）
- `make_bets()` には**未接続**（make_bets_v2 で統合）
- 実オッズは単勝のみ。馬連・三連複は `estimate_payouts_from_win_odds()` で推定

> ⚠ **ROI集計上の注意（P2 既知制限）**: `scripts/friday_predict.py` と `scripts/weekend.py` の
> `make_bets(c)` 呼び出しは `market_odds_map` が構築される前に実行されるため、
> **keiba.db に保存される `bets`（旧方式）は常に推定オッズベース**。
> アプリ表示の `gumbel_bets` は `to_app_json` 経由で実オッズを正しく使う。
> 将来 ROI を集計する際は「DB記録のbets ≠ 実オッズ」に注意すること。

#### 実装内容
- `src/betting/bet_optimizer.py` 新規作成
  - `build_optimal_bets(probs, odds_map, horses, race)` — 券種横断EV買い目生成
  - `_select_win/place/quinella()` — 各券種の選択ロジック（点数上限付き）
  - `_build_trio()` — 三連複: 型に縛られないEVベース、**4〜20点保証、3頭1点禁止**
  - `_calc_synthetic_odds()` — 合成オッズ計算（警告のみ、切り捨てなし）
  - `determine_axis_structure()` — 複勝確率分布から軸構造判定（補助）
  - `make_bets_v2()` — Gumbel確率ベースの新買い目生成（段階移行用）
    - `feat_df` 渡せば dual_model（B2_ndcg単勝）、なければ horses['rating'] で単一シミュレーション
- `tests/test_bet_optimizer.py` 新規作成（17テスト全通過）
- `KEIBA_XGB_retrain_v3.ipynb` セル2 に `bet_optimizer.py` を追加

#### Colab での使い方
```python
from src.betting.bet_optimizer import make_bets_v2

# feat_df があれば dual_model が有効になる
bets, probs, odds_map, meta = make_bets_v2(
    horses, race, BASE_DIR,
    market_odds_map=market_odds_map,  # build_market_odds_from_races() の出力
    feat_df=feat_df,                  # horse_features.csv の1レース分（省略可）
    n_sims=20000,
)
print(f"三連複 {len(bets['trio'])} 点, 合成 {bets['summary']['syn_odds']:.1f} 倍")
print(f"投資 ¥{bets['summary']['total_amount']:,}  "
      f"配当 ¥{bets['summary']['payout_min']:,}〜{bets['summary']['payout_max']:,}")
```

#### 未実装（タスク5: アプリJSON反映）
- `bet_optimizer` 出力を `to_app_json` に繋ぐ作業は次回セッションで実施
- EV付き買い目 JSON 形式は仕様通り（`{"num":3,"odds":5.8,"prob":0.18,"ev":1.04}`）

---

### 2026-07-02 セッション：デュアルモデル実装（単勝 B2_ndcg / 他 A_fukusho）

3モデル比較（653レース）の結果に基づき、券種別にモデルを使い分けるデュアルモデルを実装。

#### 使い分け方針（暫定）
| 券種 | モデル | T（実測） | 根拠 |
|------|--------|-----------|------|
| 単勝 | B2_ndcg | **1.0** (ECE=0.0043) | 的中率 45.5% vs A 43.6% |
| 複勝・馬連・三連複 | A_fukusho | 0.7 (ECE=0.0136) | 複勝 80.6%, 馬連 23.3%, 三連複 21.6% |
| pairwise | 不使用 | 5.0 | 確率が均一すぎ（最下位） |

⚠ **暫定的な使い分け。単勝の差(45.5% vs 43.6%)は小さく誤差の可能性あり。
  1,000 レース超のデータ蓄積後に必ず再検証すること。
  ROI は推定配当ベースの理論値であり実際の収益とは異なる。**

#### 実装内容
- `src/betting/dual_model.py` 新規作成
  - `load_dual_models(base_dir)` — A + B2 モデル・特徴量・温度をキャッシュ付きロード
  - `merge_probs(probs_a, probs_b2)` — win を B2 で上書き、他は A を引き継ぐ
  - `build_dual_probs(feat_df, horse_nums, base_dir, n_sims)` — 2系統シミュレート→マージ
- `src/betting/make_bets.py`
  - `build_bets_from_simulation()` に `ratings_win=None` パラメータ追加
  - 渡した場合: B2 で 2 回目シミュレーション → win 確率を上書き（単勝デュアルモデル）
  - None のとき: 従来の単一モデル動作（後方互換）
- `tests/test_dual_model.py` 新規作成（6テスト全通過）

#### Colab での使い方（セル4c の後に追加）
```python
from src.betting.dual_model import build_dual_probs
from src.betting.ev_calculator import calc_ev_all_tickets, select_value_bets

# feat_df: 1レース分の horse_features.csv 行（place < 99 のみ）
# horse_nums: 馬番リスト
probs, meta = build_dual_probs(feat_df, horse_nums, BASE_DIR, n_sims=20000)
print(f"B2 available: {meta['b2_available']}, T_A={meta['T_A']}, T_B2={meta['T_B2']}")
```

または `build_bets_from_simulation` 経由:
```python
from src.betting.make_bets import build_bets_from_simulation
# ratings_win は dual_model._predict_b2_ratings() で取得
bets, probs, ev = build_bets_from_simulation(
    horses, odds_map, n_sims=20000, ratings_win=ratings_b2_scaled
)
```

---

### 2026-06-27 セッション：市場補正レイヤー導入

6/27（土）の32レース分析で AI RL1の3着内率15.6% vs 市場1番人気46.9%、
AIが市場と異なる本命を出した25Rで市場が12勝 vs AI 2勝という結果を受けて、
市場補正レイヤーを暫定導入。

#### 実装内容（branch: `claude/racing-data-pipeline-review-4easwb` → PR → main）
- `src/features/market_correction.py` 新規作成（`CORRECTION_FACTORS` / `apply_market_correction()`）
- `src/features/engine.py`: `calc_all()` の softmax 前に `apply_market_correction()` を統合
- `src/betting/app_json.py`: `cal_prob_raw`/`rl_rank_raw`/`correction_factor`/`correction_applied` を馬エントリに追加、`market_correction_enabled`/`honmei_changed_by_correction` をレースエントリに追加、トップレベル JSONに `market_correction_enabled` を追加
- `index.html`: 「🔧 市場補正 ON/OFF」バッジ常時表示、補正で順位変動した馬はRL欄に旧→新表示、本命変更時の注記
- `tests/test_market_correction.py` 新規作成（7テスト全通過）
- `CLAUDE.md`: 市場補正レイヤーセクション追加

---

### 最終更新: 2026-06-25

---

### 2026-06-25 セッション：コース適性・cal_prob修正・不利メモシステム（PRベース運用）

このセッションは全て **作業ブランチ → PR → CI green → squash merge** で main に反映済み。

#### ① コース適性特徴量6種（PR #10 マージ済み）
- `data/course_profiles.json` 新規（全10競馬場×芝/ダート=20コースの直線長・回り・坂を定義）
- `engine.py`: `load_course_profiles` / `get_course_profile` / `calc_course_aptitude_features` 追加
- `calc_features_for_xgb` に6特徴量統合（f_same_course_rate / f_same_turn_rate / f_straight_match / f_uphill_match / f_agari_at_similar / f_course_coverage）
- `init_engine` に `_BASE_DIR` 保持。course_profiles.json 不在/未定義コースはデフォルト0でフォールバック
- ※ AUC変化の確認は次回XGB再学習時（build_training_data は **xf 展開で自動取込・手動編集不要）

#### ② cal_prob保存バグ修正（PR #11 マージ済み・予想精度に直結）
- 原因: `calc_all` がキャリブレ済み複勝確率 cal_prob を計算後に出力辞書へ保持せず捨てていた
  → `race_predictions.cal_prob/fuku_prob` が常に0で保存 → correction.py の乖離学習が空回り
- 修正: `engine.py` out.append に `cal_prob` を追加（win_probはsoftmaxで上書きされるため別キー保持）
- 修正: `db.py save_race_predictions` の fuku_prob を非存在の fuku_pct ではなく Harville top3_prob(0-1) から保存
- これで「予測複勝確率 vs 実着順」の実値が蓄積され、RL順位×人気帯の系統的バイアスを週次補正できる

#### ③ 不利メモ入力システム（PR #12 マージ済み・スキーマ駆動）
- 目的: レース映像を見て出遅れ・不利・展開ロスを手動入力し特徴量化（JRDBのIDM記憶要素を簡易再現）
- `data/note_schema.json`（初期6項目）/ `race_notes` テーブル（JSON格納・UNIQUE(date,race_id,horse_num)で上書き）
- `db.py`: save_race_notes / get_latest_note_time / calc_handicap_from_notes / recalc_all_handicaps
- `engine.py`: calc_unlucky_features（直近補正値合計・前走・最大・カバレッジ。学習反映はデータ蓄積後）
- `gas/raceNotes.gs`（新規）/ `scripts/ingest_notes_log.py`（新規）/ index.html に📝動的入力UI
- weekend.yml / sunday-results.yml に取込ステップ追加（GAS_URL流用・未設定ならスキップ）
- **GAS設定 2026-06-25 完了**: raceNotes.gs 追加 + doGet に saveNote/getNotesLog/getNotes 追記 + 再デプロイ。
  権限はオッズ設定で承認済みのため流用（同一プロジェクト・同一スコープ）。
  `?action=getNotesLog` が `{"status":"ok","count":0,"rows":[]}` を返すことを確認済み。

#### 重要：週次予想は GitHub Actions で動く（Colab不要）
- スマホアプリの予想ボタン → GAS → GitHub Actions（friday-predict.yml / weekend.yml）が main をcheckoutして実行
- **コード変更は次回ボタン押下で自動反映**。data/*.json（course_profiles/note_schema）も**リポジトリに含まれるためActionsに自動で乗る** → Drive配置・強制アップデートセルは不要
- Drive配置/強制アップデートセルが要るのは **Colabでのチューニング・再学習時のみ**
- friday-predict 試験起動（2026-06-25木）: パイプライン正常・新コード読込OK。0レースは木曜=非開催日のため（想定通り）

#### 残: 不利メモの運用
- [ ] 週末、気になったレースの映像を見て📝で不利入力（週10〜20頭）→ race_notes に蓄積
- [ ] 学習反映はデータ2〜3ヶ月蓄積後

---

### 最終更新: 2026-06-23

---

### 2026-06-23 セッション：データパイプライン総点検＋修正（branch: `claude/racing-data-pipeline-review-4easwb`）

土日のスクレイピング→保存→乖離学習の全フローを点検し、以下の欠落・潜在バグを修正。

#### 修正（このブランチ）
**A. 土曜予想が race_predictions に保存されない問題（最重要）**
- 原因: `scripts/friday_predict.py` が `save_race_predictions()` を呼んでおらず、土曜の全レース予測がDBに残らない → 補正テーブル(correction_table.json)が日曜分のみで学習されていた
- 修正: friday_predict.py に全レース予測スナップショット保存ループを追加（weekend.py の日曜側と対称化）。これで土日フルのデータで乖離学習が回る

**B. ラップタイム未取得**
- `src/scraper/jra_scraper.py`: `_extract_lap_times()` を新設。結果ページの「ラップタイム」見出しから区間タイム（200m毎）を抽出し、`first_3f`/`last_3f` を算出
- `parse_result_soup()` の戻り値に `lap_times`(ハイフン連結) / `first_3f` / `last_3f` を追加
- `src/utils/db.py save_history_db()`: race_history へ lap_times / first_3f / last_3f を INSERT/UPDATE（従来 first_3f は None 固定だった）

**C. race_predictions に枠順(bracket)を蓄積**
- race_predictions スキーマに `bracket INTEGER` を追加（CREATE + ALTERマイグレーション）
- `update_prediction_results()` で結果ページの確定枠を COALESCE 充填（出馬表パースは枠未取得のため予測時は NULL）

**D. race_predictions 重複行バグ（乖離学習の二重カウント）**
- 原因: (race_id, horse_num) に一意制約が無く、INSERT OR REPLACE が実質ただのINSERT → 同一レース複数回保存で重複行
- 修正: init_db で重複行を DELETE 後 `idx_rp_uniq` UNIQUE INDEX を作成。以後は正しく上書き

**E. bets テーブル拡張列が init_db に無い潜在バグ**
- save_bets_db が書く racecourse/distance/surface/running_style/popularity/ai_score/ev_rank を init_db のマイグレーションに追加（新規DB・CIテストでのクラッシュを解消）。tests 17件 全passに復帰

#### 未対応（設計判断・外部依存が必要）
| 項目 | 理由 |
|------|------|
| body_weight/bracket/win_odds の埋まり率 | parse_result_soup は texts列の位置ヒューリスティック。実機の結果ページで埋まり率を要検証（来週末の実行ログで確認） |
| apply_correction() デッドコード | correction.py の関数は未使用。同等ロジックは engine.py にインライン実装済み（動作はする）。整理は任意 |

---

### 2026-06-23 セッション②：直前確定オッズの中央集約（branch: `claude/chokuzen-odds-logging` / PR）

「朝予想 vs 直前確定オッズ vs 結果」を後から突き合わせるため、直前オッズを中央DBに蓄積する仕組みを実装。

#### 仕組み
1. **GAS**: スマホの「直前オッズ取得」ボタン → `getOddsHandler` が `logOdds()` を呼び、
   Googleスプレッドシート(`keiba_odds_log` / 初回自動作成)へ `captured_at, race_id, horse_num, tansho, fukusho` を追記。
   新エンドポイント `getOddsLog`（`?action=getOddsLog&since=...`）でJSON取得。
   - 追加/変更ファイル: `gas/oddsLog.gs`(新規) / `gas/getOdds.gs`(logOdds呼び出し追加)
2. **DB**: `keiba.db` に `odds_snapshots` テーブル新設（`UNIQUE(race_id, horse_num, captured_at)` で重複取込防止）。
   `save_odds_snapshots()` / `get_latest_odds_snapshot_time()` を追加（`src/utils/db.py`）。
3. **取込**: `scripts/ingest_odds_log.py` が `GAS_URL?action=getOddsLog&since=<最新>` を叩き odds_snapshots へ保存。
   `weekend.yml` / `sunday-results.yml` にステップ追加（`env: GAS_URL=${{ secrets.GAS_URL }}`）。
   GAS_URL未設定なら安全にスキップ（no-op）。

#### ⚠️ 有効化に必要な手動作業（ユーザー）
- [x] `gas/oddsLog.gs` をGASプロジェクトに追加し、`doGet` に `if (action === 'getOddsLog') return getOddsLogHandler(e);` を追記（2026-06-23 完了）
- [x] `gas/getOdds.gs` の更新分（getOddsLoggedHandler ラッパー経由）も反映（2026-06-23 完了）
- [x] GASを再デプロイ（新バージョン）＋ SpreadsheetApp 権限承認（2026-06-23 完了）
- [x] GitHubリポジトリの Secrets に `GAS_URL`（GAS WebアプリURL）を登録（2026-06-23 完了）
- [ ] 来週末、直前ボタンを数回押す → 日曜結果ワークフローで odds_snapshots に入ることを確認

#### 後続タスク（データが溜まってから）
- 朝(race_predictions) × 直前(odds_snapshots) × 結果(history) を突き合わせる分析・補正
  （直前オッズでの value_gap 再計算 → 「朝は妙味でも直前で消える/出る」傾向の学習）

---

### 最終更新: 2026-06-23

---

### 2026-06-21 セッションで実施した修正・実装

#### バグ修正（全てmainにpush済み）

**① 東京結果が取得できない問題（最重要）**
- 原因: 東京R01が障害レース → `find_r01_result()` が障害をスキップして次を探す → 3連続パラメータエラー → break → None
- 修正: `find_r01_result()` / `find_r01_shutuba()` から障害スキップを削除。障害フィルタは下流（`parse_result_soup`）で行う
- ファイル: `src/scraper/jra_scraper.py`

**② ROI集計クラッシュ（sunday_results.py）**
- 原因: `SELECT b.*` + `r.racecourse` で racecourse 列が重複 → pandas groupby エラー
- 修正: 明示的な列指定に変更
- ファイル: `scripts/sunday_results.py`

**③ race_id キー不一致**
- 原因: `parse_result_soup()` が返す辞書に `'id'` と `'date'` キーがなかった
- 修正: `parse_result_soup()` に `'id'` / `'date'` を追加、`update_prediction_results()` / `save_race_predictions()` も両キー対応
- ファイル: `src/scraper/jra_scraper.py`, `src/utils/db.py`

**④ 予想上書き防止**
- 原因: 土曜夜に「土曜結果+日曜予想」を再実行すると latest.json が上書きされる
- 修正: `_already_generated()` で当日同タイプの生成済みチェック → `--force` で強制再生成
- ファイル: `scripts/weekend.py`

#### 新機能

**⑤ 結果取得ステータス表示**
- `generate_stats.py`: history.db から最終保存日・レース数・会場を取得し stats.json に `results_status` を追加
- `index.html`: 成績ページ冒頭に「📡 最終結果取得状況」カード（最終保存日・会場・実行時刻・成否）を表示
- **重要**: 0R取得の場合は赤字で「⚠️ 取得失敗（0R）」と表示

**⑥ AIの盲点パターン自動検出**
- `generate_stats.py`: `_calc_upset_patterns()` を追加
  - shadow_bets から「AI上位3頭外の馬が複勝内に来た（upset）」を自動集計
  - 波乱度/頭数/馬場/距離/会場/クラス/複合条件別に外れ率・全滅率・穴馬率を算出
  - データ5件以上の複合条件を盲点ランキングとして出力
- `index.html`: 成績ページに「🔍 AIの盲点パターン」カードを追加
- **注意**: 現在は表示のみ。予測へのフィードバックは未実装

**⑦ race_predictions テーブル・f_pred_gap 特徴量**
- 毎週の予想→結果照合で race_predictions に RL順位・実着順・乖離を蓄積
- `engine.py`: `calc_features_for_xgb()` に f_pred_gap_avg / f_pred_gap_worst / f_pred_gap_consistency を追加
- **制限**: 同じ馬の再出走時のみ有効。条件レベルの系統的バイアス修正には不十分

#### 現在のhistory.db状況（2026-06-21）

| 日付 | 阪神 | 函館 | 東京 |
|------|------|------|------|
| 土曜 6/20 | ✅ | ✅ | ✅ 35レース保存済み |
| 日曜 6/21 | ✅ | ✅ | ❌ 未保存（東京のみ欠損） |

東京日曜分（約11レース・約160頭）が欠損。コード修正済みなので来週の実行で取得可能。

#### ⚠️ 日曜結果ワークフローの実行タイミング（重要）
JRAのJRADBサービスは **20:30 JST頃に閉鎖**する。
- 21:27 JST の実行 → 全会場0件（閉鎖後）
- **正しい実行時間: 18:30〜20:00 JST（9:30〜11:00 UTC）**
- 土曜夜も同様（最終レース後17:30頃〜20:30頃の間に実行）

---

### 未解決の設計課題（Opusに相談予定）

**「なぜ外れるかを自動診断して自動修正するループ」の設計**

ユーザーの指摘：「0.5%勝率の馬が何度も馬券内に来る。AIはなぜ外すかを自動で見つけて自動修正すべき」

現状の問題：
- `f_pred_gap` は個馬補正に過ぎず、条件レベルの系統的バイアスを修正しない
- 盲点パターンは表示するだけで予測に反映されない
- モデル（XGBと重み）は月1回の手動再学習でしか更新されない

Opusに聞きたいこと：
1. 「なぜ外れたか」を自動診断する方法（SHAP値？誤差分解？）
2. 診断結果をもとに重みやモデルを週次で自動修正するループの設計
3. データが少ない段階でのノイズリスク対策
4. 根本的なアーキテクチャ変更が必要か

---

## ⚠️ 重要：設計指針書（必ず読むこと）

**`DESIGN.md`（このリポジトリのルート）を必ず参照すること。**
DESIGN.md の Phase 0〜3 はすべて実装完了（2026-05-25）。

### 完了済み（旧「次にやること」より）
- ✅ **スピード指数 XGB再学習**: 2026-06-18 21:51 に実行済み・本番反映済み。
  `data/xgb_feature_cols.json` の `trained_at` で確認可能。
  特徴量98個に `f_speed_fig_last/avg/max` および相対ランクを含む。
- ✅ **重み再チューニング**: optimal_weights.json は rl/maturity/rotation の新キーで再チューニング済み。

### 次にやること（優先順）
1. **重みの妥当性確認**（後述「重みの妥当性確認」セクション）← 任意・週末作業ではない
   - rl/maturity がほぼ無効化（0.01）されているのが意図通りか検証する
2. **Stage3 列マッピング修正**（bracket/win_odds/body_weight が 0〜6.5%）← 要調査
   - JRA結果ページの実際の列順を確認し、`parse_result_page()` の tx インデックスを修正
   - 修正後に Stage3 を再実行（再開ロジックあり・完了済み開催日は自動スキップ）
3. **週末の実運用**で動作確認・ROI計測

### 重みの妥当性確認（rl/maturity がほぼ無効化されている件）
現在 `optimal_weights.json` は jockey:0.29 / distance:0.26 / pace:0.20 / trainer:0.17 中心で、
実装した実力スコア rl/maturity が 0.01（ほぼ無効）になっている。チューナーが過去データで
「実力スコアを足しても的中率が上がらない」と判断した結果だが、以下で意図通りか確認する。

1. **チューニングノートのログを再確認**
   - `tune_weights.py` 実行時の Acc@1 / ECE を、rl/maturity を強制的に入れた版と比べる。
   - rl/maturity を 0.01 → 0.15 程度に手動で上げて、過去データでの Acc@1 が落ちないかを検証。
2. **特徴量の重複を疑う**
   - XGB 側に既に f_speed_fig 系（スピード指数＝実力）が入っているため、ルール側の f_rl が
     XGB と情報的に重複し、重み最適化で不要と判断された可能性。これは「無効化されて当然」で問題なし。
3. **判断基準**
   - rl を上げて Acc@1 が改善 → tune_weights の探索範囲/初期値の問題。再チューニング。
   - rl を上げても改善しない → 現状（0.01）が正しい。実力情報は XGB が担っているので
     ルール側 f_rl は冗長、という結論で確定。CLAUDE.md にその旨を記録して課題クローズ。

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
- コード変更は **作業ブランチ → Pull Request → CI確認 → main へマージ** の順で進める（main直push禁止）
- 自動データコミット（ワークフローのlatest.json/*.db等）は bot が main へ直接pushする（従来どおり）
- GitHubに**ないファイル**はユーザーに確認してから作業する

---

## git操作（PAT使用）
```bash
PAT="<ユーザーから取得>"
git remote set-url origin "https://${PAT}@github.com/hanagenuku/keiba_ai.git"
git push -u origin <branch-name>
git remote set-url origin "https://github.com/hanagenuku/keiba_ai.git"
```
