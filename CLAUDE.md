# keiba_ai プロジェクト 引き継ぎドキュメント

> ⚠ **作業前に `docs/KEIBA-AI_引き継ぎ書_追補_2026-06-28.md` も必ず参照すること。**
> モデル状況・週次運用フロー・Colab手順・既知制限の詳細が記載されている。

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
| `KEIBA_過去データ一括取得_v4.ipynb` | 過去データ一括取得専用（GitHubには未push・Drive管理） |

> ⚠ `KEIBA_過去データ一括取得_v4.ipynb` はGitHubに含まれていない。Driveのみで管理。

## Google Drive パス
`/content/drive/MyDrive/keiba_ai/`

## データ・モデル構造
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

### 最終更新: 2026-07-09 セッション（市場特徴量モデル 本番反映）

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
