# keiba_ai システム構成（2026-11-09 時点）

> ルールと運用の注意は `CLAUDE.md`。経緯は `docs/history/`。このファイルは「**何がどこで動いているか**」だけを書く。
> 構成が変わったらこのファイルを**書き換える**（追記で積まない）。

---

## 1. 全体の流れ

```
 ┌─────────── 収集 ───────────┐   ┌──── 蓄積 ────┐   ┌──────── 予想 ────────┐   ┌──── 出力 ────┐
 JRA 公式（出馬表・オッズ・結果）──▶ data/history.db ──▶ src/features/engine.py ──▶ data/latest.json ──▶ index.html（GitHub Pages のアプリ）
 netkeiba（実オッズ・調査用）       data/keiba.db       calc_all()                  │                   │ 🔄直前オッズ（GAS経由）
 web 情報（yosounohone ほか）──▶ data/claude_web_findings.json ─────────────┐       │                   ▼
 スマホ入力（直前オッズ・不利メモ・パドック）──▶ GAS ──▶ スプレッドシート ──▶ ingest_*.py      │                 Google スプレッドシート
                                                                        ▼       ▼
                                                     scripts/build_product_article.py（src/product/）
                                                        └▶ 記事（有料／無料）→ Google ドキュメント（Drive）
                                                        └▶ prediction_archive/（公開スナップショット・上書き禁止）
 結果取得（夜）──▶ history.db 更新・PLレーティング更新・成績集計（data/stats.json）・公開記録と台帳に結果を追記
```

起動は Claude の Routine（朝 08:50・夜 18:40）が `workflow_dispatch` を撃つ形。cron は補助（`CLAUDE.md` §3）。

---

## 2. 予想の中身（`src/features/engine.py::calc_all`）

1. **特徴量**: `calc_features_for_xgb()` が1頭ずつ **134列** を作る（列名は `data/xgb_feature_cols_residual.json`）。
   主な部品: スピード指数（`speed_index.py`）・PLレーティング（`pl_rating.py`）・展開（`race_shape.py`・学習も推論もこの1本）・コース／距離（`data/course_*.json`）・騎手／調教師のベイズ縮小レート・近走（`f_recent`）・枠（`f_post`）など。
   🔴 市場オッズは特徴量に入れない（North Star #1）。
2. **モデル**: XGBoost の**残差学習**（`data/xgb_fukusho_model_residual.pkl`）。市場の人気から作る `base_margin` を土台にして、**市場からのずれだけ**を学習する。
   現行: `trained_at 2026-09-15`・train_end 2026-06-30・学習155,187行・val AUC 0.7846。木は61本（`best_iteration` は10だが、これは切片の通過で決まる値で信号の飽和ではない → `hitrate/INSTRUCTIONS.md` §1-2）。再学習は測って不採用（2026-10-19）。
   オッズが全滅したときは `base_margin` を中立化し、`ability_margin`（市場ゼロのAI評価）だけで順位を付ける。
3. **較正と確率**: `data/xgb_calibrator.pkl` で3着内確率に較正 → softmax で勝率。買い目側の Gumbel 温度は T=1.0（`data/rating_temperature.json`）。
4. **画面の3列**: `AI単`（`solo_rank`・市場ゼロ）／`適性`（`src/features/aptitude.py`）／`AI+市`（買い目と軸に使う）。
5. **レース信頼度** `conf`: `src/betting/app_json.py::race_confidence()` が軸の3着内確率から作る。`conf >= 66` が推奨レース（商品の有料対象）。
6. **出力**: `src/betting/app_json.py::to_app_json()` が `data/latest.json` を書く。アプリ（`index.html`）はこれを読むだけ。

後付けで予想を書き換える層は置かない方針。過去にあった市場補正レイヤー・`error_tags` 週次補正・買い目フィルタはすべて撤回済み（`CLAUDE.md` §4）。

---

## 3. 商品レイヤー（`src/product/`・本番の予想を**読むだけ**）

| ファイル | 役割 |
|---|---|
| `scope.py` | `article_kind()` が有料／無料／記事にしないを決める唯一の場所。`in_scope()` は台帳互換のため据え置き |
| `facts.py` | 記事に書いてよい事実の抽出（書けない事実は `biz/PHASE4_DESIGN.md`） |
| `marks.py` | 印（◎○▲…）と「市場の支持ほどAIは評価していない馬」の判定（`DANGER_MAX_POP=1` / `DANGER_MIN_SOLO=6`） |
| `commentary.py` | 見解の文章。`FORBIDDEN_PATTERNS` で陣営・調教・パドックに触れる文を落とす |
| `tickets.py` | 買い目。堅実コース／勝負コースの2コース。`CONF_STRONG = 66` |
| `article.py` | 記事の組み立て |
| `web_block.py` | Claude の web 見解の別枠表示（AI の数字は変えない）。全レース対象（2026-11-10〜） |
| `archive.py` | 公開スナップショット（`prediction_archive/`・上書き拒否） |
| `results.py` | 公開済みの予想に結果を追記 |
| `danger_log.py` | 「市場の支持ほど…」の警告と着順の台帳（`data/danger_favorites.json`） |
| `web_log.py` | web 見解と結果を情報源ごとに残す台帳（`data/web_findings_log.json`）。馬場・馬メモ・予想家の印（`pick`） |

入口は `scripts/build_product_article.py`（`--free` で無料版、`--publish` でスナップショット保存）。

---

## 4. GitHub Actions

| ワークフロー | 用途 | 起動 |
|---|---|---|
| `weekend.yml` | 予想の生成・更新（`--mode saturday / sunday / refresh`） | Routine から dispatch（朝）＋ cron 08:00 / 11:30 / 14:00 JST（追い更新） |
| `sunday-results.yml` | 結果取得・`history.db` 保存・オッズ／メモ取り込み・PL 更新・商品の結果追記 | Routine から dispatch（夜） |
| `collect-web-signals.yml` | web 由来シグナル（kayochin 指数・全レース全馬）の蓄積。非公開DBを artifact で持ち回る。採用0件なので本番には届かない。2026-10-05〜11-09 は失敗していた（11-10 修正） | cron 月曜朝 |
| `collect-odds.yml` | オッズ時系列の収集 | cron 金土・土日 |
| `test.yml` | テスト（PR の CI） | push / PR |
| `deploy-pages.yml` | アプリ（`index.html`）の公開 | main の `index.html`・`data/latest.json`・`data/stats.json` などが変わったとき |
| `friday-predict.yml` | 金曜夜に土曜分を作る | **使っていない** |
| `monthly-retrain.yml` | 月次再学習 | **cron 停止中**（dispatch のみ） |
| `backfill-dividends.yml` / `repair-popularity.yml` | 過去データの埋め戻し・修復 | 手動 |
| `fetch-netkeiba-odds.yml` / `eval-odds-anchor.yml` | D-2（実オッズアンカー）の検証用 | 手動（D-2 は不採用で閉じた） |
| `probe-*.yml`（6本） | 調査・診断用の一回もの | 手動。`probe-wide-odds.yml` だけ cron が残っている |

---

## 5. コードの地図

| 場所 | 中身 |
|---|---|
| `src/scraper/` | `jra_scraper.py`（出馬表・オッズ・結果。CNAME suffix 走査）・`parser.py`・`going.py`（当日の馬場）・`calendar.py`・`kayochin.py` / `yosounohone.py`（web 情報源） |
| `src/features/` | `engine.py`（特徴量と予想の本体・約3,300行）・`speed_index.py`・`pl_rating.py`・`race_shape.py`・`aptitude.py`・`claude_web.py`・`web_signals.py`・`shap_explain.py`・`horse_type.py`・`error_tags.py`（既定OFF） |
| `src/models/` | `predict.py`（softmax・較正）・`calibration.py`・`calibration_xgb.py` |
| `src/betting/` | `app_json.py`（latest.json）・`make_bets.py`・`axis_bets.py`（軸1頭ベース）・`bet_optimizer.py`・`shadow.py`（成績記録）・`ev_filter.py`・`rank_matrix_filter.py`（既定OFF・測定のみ）ほか |
| `src/product/` | §3 |
| `src/tools/` | 学習・較正・データ作成（`build_training_data.py`・`train_xgb.py`・`calibrate_xgb.py`・`train_race_shape.py`・`train_pace_model.py`・`tune_weights.py` ほか） |
| `src/utils/` | `db.py`（DB 読み書き・約1,800行）・`config.py`・`model_registry.py`（`save_version` / `rollback`）・`source_registry.py`（web 情報源の採用を縛る） |
| `scripts/` | ワークフローの入口（`weekend.py`・`sunday_results.py`・`build_product_article.py`・`generate_stats.py`・`ingest_*.py`・`record_*.py` ほか34本。`probe_*.py` は調査用） |
| `gas/` | Google Apps Script（直前オッズ取得・オッズ／不利メモのスプレッドシート記録・ワークフロー起動） |
| `tests/` | pytest（約100ファイル。fixture は実機 HTML を模す） |
| `index.html` | アプリ本体（約170KB・`latest.json` を読む） |
| ルートの `*.ipynb` | Colab 時代のノートブック。再学習はこの環境でも回るので**必須ではない** |

### `data/` の主なファイル

| ファイル | 中身 |
|---|---|
| `history.db`（LFS・約50MB） | 学習データ。`race_history` / `horse_history`（定義は `docs/history_db_schema.md`） |
| `keiba.db`（LFS） | 予想・ベット・結果（`bets`・`bet_simulation`・`results` ほか） |
| `latest.json` | その日の予想（アプリと記事の唯一の入力） |
| `stats.json` / `kpi_weekly.json` | 成績ページと市場比較 KPI |
| `race_results.json` | 結果 |
| `xgb_fukusho_model_residual.pkl` / `xgb_feature_cols_residual.json` / `xgb_calibrator.pkl` | 本番モデル |
| `pl_rating.pkl` / `speed_index_cache.pkl` / `race_shape_model.pkl` / `pace_model.pkl` | 特徴量の部品 |
| `course_*.json` | コースの物理データ・距離プロファイル |
| `claude_web_findings.json` / `web_findings_log.json` / `source_registry.json` | web 情報（当日分・台帳・情報源の採用状態） |
| `danger_favorites.json` | 「市場の支持ほど…」の台帳 |

---

## 6. 検証ディレクトリ（ルート直下・どれも本番からは読まれない）

すべて `CRITERIA*.md`（事前登録）→ 測定コード → `RESULTS*.md`（結果）の形。**結論はすべて出ている**。

| ディレクトリ | 何を測ったか | 結論 |
|---|---|---|
| `logic2026/` | 2026年の完全OOSで予想生成ロジック比較 | 市場フリーは不採用・本番を維持 |
| `retrain/` | 再学習（A）・較正のパリティ修正（B） | 両方不採用 |
| `danger/` | 「危険な人気馬」の3着内・1着 | 両方不採用 |
| `oikiri/` | yosounohone 追い切り★ | 不採用（回収率が控除率を超えない） |
| `websig/` | kayochinkeiba 能力指数・web 情報源の候補と信頼度基準 | 不採用／基準は登録済み・未測定 |
| `webinfo/` | web 見解の記録形式と検証 | 仕組み（測定ではない） |
| `spaia/` | SPAIA 順番待ちスコア | 不採用 |
| `betting/` | Betting Policy Gate 0・Step 1 | Step 1 で閉じた |
| `mispricing/` | 市場の過大評価（②-A・P3-01） | 不採用 |
| `noryoku/` | 能力軸の分解（Phase 2） | 診断 |
| `tenkai/` / `pace/` / `fpace/` / `lead/` | 展開・ペース・逃げの予測、コースの外挿 | 外挿は否定。展開は本番に接続済み |
| `zansa/` | 市場残差の発走地点ずれ | 採用も棄却もできない（計測器の限界） |
| `calib/` | 確率較正・ability 経路・EV18 | 較正は成功、市場超えは未達 |
| `mf2/` / `p4/` / `debut/` / `gate/` / `hitmiss/` / `race_select/` / `osusume/` / `dprime/` / `forecaster/` / `paddock/` | 市場フリー再構築・得意条件・新馬戦・ゲート・切り馬・推マーク・ワイド盤・ミスプライシング地図・パドック中間確認 | `RESULTS.md` 参照 |
| `bet_strategy_lab/` | 買い目戦略の事前登録 | 登録のみ |
| `outputs/` | 古いまとめ2本 | 記録のみ |

---

## 7. 複雑さの棚卸し（整理の候補・**未実施**）

このファイルを作ったときに見えた「整理すれば軽くなるが、触ると壊れうるもの」。動かすかどうかはユーザーが決める。

1. **検証ディレクトリ約30個がルート直下にある。** `experiments/` に移せば見通しは良くなるが、`CRITERIA` / `RESULTS` やコード、`docs/history/` の中のパス参照が切れる。移すなら参照の書き換えとセットで。
2. **使っていないワークフロー**: `friday-predict.yml`・`probe-*.yml` 6本（`probe-wide-odds.yml` は cron が生きている）・D-2 の2本。消すか `workflow_dispatch` のみにそろえる候補。
3. **既定OFFのまま残っているコード**: `rank_matrix_filter.py`（測定のみ）・`error_tags.py` の補正・`ev_filter.py`・ルールベース重み `_W`（XGB が例外を出したときのフォールバックでしか使われない）。
4. **Colab ノートブック6本**: 再学習はこの環境で回る。残すなら「Colab 時代の記録」と明記する。
5. **`src/features/engine.py`（約3,300行）と `src/utils/db.py`（約1,800行）** に責務が集中している。分割は効果が大きいが、学習/推論パリティ（North Star #3）を壊しやすいので、やるならテストで同値を固めてから。
