## プロジェクト概要
JRA競馬AI予想システム。Google Colab + Google Drive で運用。

## リポジトリ
- GitHub: `hanagenuku/keiba_ai`
- 本番ブランチ: `main`（Colabの強制アップデートセル・GAS・各ワークフローは `main` から取得）
- **コード変更の運用フロー（2026-06-23 変更 / 2026-08-24 追記）**:
  作業ブランチ → **Pull Request 作成 → CI(テスト)確認 → main へマージ**。
  `main` への直接pushは原則しない（コードレビュー・CIを必ず通すため）。
  ✅ **PR作成とマージの可否をユーザーに都度確認する必要はない**（2026-08-24にユーザーが明示）。
  テストが通っていれば、作業ブランチ→PR→マージまで一気に進めてよい。
  ⚠ ただし**何をマージしたかは必ず報告する**こと。特に本番の予想に影響する変更
  （モデル差し替え・買い目ロジック・表示する数字）は、変更点と実測の根拠を添えて伝える。
  ただし GitHub Actions（金曜/週末/日曜ワークフロー）が自動生成する
  **データコミット**（latest.json / *.db / stats等）は従来どおり bot が `main` へ直接pushする。

## Colabノートブック構成
| ファイル | 用途 |
|----------|------|
| `KEIBA_土日_v5_ROI.ipynb` | 土曜夜（土曜結果取得＋日曜予想）、日曜夜（日曜結果取得・照合）※GitHub Actionsが主体、Colabはチューニング用 |
| `KEIBA_金曜_v5_最新.ipynb` | 金曜夜（翌週レース確認・準備） |
| `KEIBA_チューニング_v1.ipynb` | 月1〜2回：重みチューニング＋キャリブレーション |
| `KEIBA_XGB_retrain_v5.ipynb` | XGB再学習＋残差学習モデル本番投入（セル1〜10を順に実行） |

> ✅ **再学習はColab専用ではない**（2026-08-28に実証）。この環境（コンテナ）で `build_training_data()` 15分 → `train_xgb(residual=True)` 15秒/シードで完走する。ただし生成される `horse_features.csv` は **128MB** でGitHubの非LFS上限100MBを超えるためコミットできない。
| `KEIBA_買い目フィルタ検証_v1.ipynb` | **買い目フィルタの大規模out-of-sample検証**（2025年末までで再学習→2026年で検証。約2,000レース規模。本番ファイルは上書きしない） |
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
  history.db      # 学習データ（race_history: 11,624件・2023-01-07〜 / horse_history: 159,671件、2026-08-05にDrive側の2023-2024データをマージ済み）
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
    'src/features/race_shape.py',
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
| `src/features/race_shape.py` | 展開（隊列・ペース）の特徴量。学習も推論も**この1本**を通す |
| `src/tools/train_race_shape.py` | 展開予想の2段モデルを学習（→ `data/race_shape_model.pkl`）|
| `src/scraper/going.py` | 当日の馬場状態を、その日すでに終わったレースの結果から取る |
| `src/utils/web_notes.py` | 週末に web から集めた記録の検証（`data/web_notes/`。予想には繋がない）|
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

