# keiba_ai プロジェクト 引き継ぎドキュメント

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
    'src/features/engine.py', 'src/features/speed_index.py',
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
| 課題 | 深刻度 | 備考 |
|------|--------|------|
| optimal_weights.json で rl/maturity がほぼ無効化（0.01） | 中 | 再チューニング自体は完了済み。実力スコアが効いていないのが意図通りか要検証（後述「重みの妥当性確認」） |
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

### 最終更新: 2026-06-28

---

### 2026-06-28 セッション：函館欠場バグ修正・Val/fuku_pct表示改善

#### PR #26: 函館レース欠場バグ修正（マージ済み）
- `src/scraper/jra_scraper.py`: suffix計算が外れたとき ±10 スキャンフォールバックを追加
  - 障害レースでR01から始まる開催など suffix がずれるケースに対応
  - `calc_suffix()` 計算失敗時、`delta=1..10` × `±` の順で候補 suffix を探索

#### PR #27: ev_filter/app_json フォールバック統一（マージ済み）
- `ev_filter.py detect_value_horses`: `fuku_prob` フォールバックを `cal_prob` → `pn` に変更
  - 旧: `top3_prob → cal_prob → pn`（`cal_prob` は softmax 後に上書きされる win_prob と異なるため不整合）
  - 新: `top3_prob → pn`（`app_json.py` の `fuku_pct` フォールバックと一致）
- `tests/test_betting.py`: `test_detect_value_horses_fallback_to_pn` にリネーム・期待値更新

#### PR #28: Val列をEV表示に変更・複勝率100%正規化（マージ待ち）
- **Val計算**: 複勝縁 `fuku_pct/100 - 0.8/fukusho` → EV = `tan_pct/100 × tansho_odds`
  - 直前オッズ取得後に「AI勝率予想 × 単勝オッズ」でEVを算出
  - 1.0未満 → 赤（マイナスEV）、1.0〜1.2 → 黄、1.2以上 → 緑（buy）
- **Val表示**: `+23%` 形式 → `1.35×` 形式
- **VALUE_GAP_THRESHOLD**: `0.10` → `1.0`（EV損益分岐点）
- **fuku_pct正規化**: `top3_prob×100`（合計≈300%）→ `top3_prob/3×100`（合計≈100%）
- **fuku強調閾値**: `38%` → `13%`（スケール変更対応）
- **buildValueChangeDetail**: 変動幅閾値 0.05→0.1、表示を×形式に統一

#### Val=-19%バグの根本原因（調査結果・記録）
- 函館7R #2ビーチイン（fuku_pct=88%, Val=-19%）は旧市場補正コードの残骸
- 市場補正ON時: `fuku_pct` は補正後（≈20%）だが表示は `fuku_pct_raw`（≈88%）→ 矛盾
- PR #25（市場補正削除）とPR #28（Val式変更）で完全解消

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
