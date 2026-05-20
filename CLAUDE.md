# keiba_ai プロジェクト 引き継ぎドキュメント

## プロジェクト概要
JRA競馬AI予想システム。Google Colab + Google Drive で運用。

## リポジトリ
- GitHub: `hanagenuku/keiba_ai`
- 開発ブランチ: `claude/review-drive-document-ZehuT`
- 本番ブランチ: `main`（Colabの強制アップデートセルはmainから取得）

## Colabノートブック構成
| ファイル | 用途 |
|----------|------|
| `KEIBA_土日_v5_ROI.ipynb` | 土曜朝（出馬表取得・予想）、土曜夜・日曜夜（結果取得・照合） |
| `KEIBA_金曜_v5_最新.ipynb` | 金曜夜（翌週レース確認・準備） |
| `KEIBA_チューニング_v1.ipynb` | 月1〜2回：重みチューニング＋キャリブレーション |

## Google Drive パス
`/content/drive/MyDrive/keiba_ai/`

## データ・モデル構造
```
data/
  history.db      # 学習データ（horse_history: 35,134件 / race_history: 4,414件）
  keiba.db        # 予想・ベット結果（bets, bet_simulation, results）
  optimal_weights.json  # チューニング済み重み
  calibrator.pkl        # Isotonicキャリブレーター
  horse_dist_dict.pkl         # 馬×距離帯成績
  horse_course_dict.pkl       # 馬×コース成績
  horse_venue_dist_dict.pkl   # 馬×競馬場×距離帯成績
  post_zone_bias.pkl          # データ実績枠順バイアス

models/
  current.json    # 現行バージョン番号
  v1/             # バージョンスナップショット
    metadata.json
    calibrator.pkl
    optimal_weights.json
    xgb_fukusho_model.pkl（存在する場合）
    ...
```

## 最新の重み（2026-05-19チューニング結果）
```
distance : 0.2866
pace     : 0.2750
trainer  : 0.2420
post     : 0.0885
recent   : 0.0409
blood    : 0.0372
jockey   : 0.0100
bias     : 0.0100
weight   : 0.0100
```
Acc@1: 23.6%、ECE: 0.0005

## 強制アップデートセル（チューニングノートのセル1とセル2の間に挿入）
```python
import urllib.request, os, sys
BASE_URL = 'https://raw.githubusercontent.com/hanagenuku/keiba_ai/main'
files = [
    'src/tools/__init__.py',
    'src/tools/tune_weights.py',
    'src/tools/calibrate.py',
    'src/tools/analyze_divergence.py',
    'src/features/engine.py',
    'src/utils/config.py',
    'src/utils/db.py',
    'src/utils/model_registry.py',
    'src/scraper/parser.py',
    'src/scraper/jra_scraper.py',
    'src/models/__init__.py',
    'src/models/calibration.py',
    'src/models/predict.py',
    'src/betting/__init__.py',
    'src/betting/make_bets.py',
    'src/betting/ev_filter.py',
    'src/betting/app_json.py',
]
for rel in files:
    dest = f'{BASE_DIR}/{rel}'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(f'{BASE_URL}/{rel}', dest)
    print(f'✅ {rel}')
for key in list(sys.modules.keys()):
    if 'src' in key:
        del sys.modules[key]
print('🔄 完了')
```

## 主要ファイルと役割
| ファイル | 役割 |
|----------|------|
| `src/features/engine.py` | 特徴量計算エンジン。calc_all, f_pace, f_jockey等 |
| `src/models/predict.py` | softmax_probs, calibrate_and_renormalize, predict_race |
| `src/models/calibration.py` | PlattCalibrator, IsotonicCalibrator, load/save_calibrator |
| `src/betting/make_bets.py` | calc_ev, calc_kelly, make_bets, log_bet_simulation, init_betting |
| `src/betting/ev_filter.py` | ability_first_loose（EV×pnフィルタでレース厳選） |
| `src/betting/app_json.py` | to_app_json（アプリ用JSON生成） |
| `src/utils/model_registry.py` | save_version, load_version, rollback, list_versions |
| `src/scraper/jra_scraper.py` | JRAサイトスクレイピング。出馬表・結果取得 |
| `src/scraper/parser.py` | HTML解析。parse_horse（騎手・年齢・斤量・父名取得） |
| `src/tools/tune_weights.py` | 重みチューニング（scipy SLSQP、5回試行） |
| `src/tools/calibrate.py` | Isotonicキャリブレーション実行（クラスはmodels/calibration.pyに移動済み） |
| `src/tools/analyze_divergence.py` | AI確率vs市場オッズ乖離分析 |
| `src/utils/db.py` | DB操作。save_history_db（結果→history.db自動蓄積）含む |
| `src/utils/config.py` | 設定。POST_BIAS_BY_ZONE、KAISAI_CALENDAR、VENUE_ORDER等 |

## ノートブック側での新モジュールの使い方
```python
# 初期化（既存の init_engine 呼び出しの直後に追加）
from src.betting.make_bets import init_betting
init_betting(BASE_DIR, bankroll=100_000)

# 厳選レース取得
from src.betting.ev_filter import ability_first_loose
selected = ability_first_loose(races, bias_data, top_n=6)

# アプリ用JSON生成
from src.betting.app_json import to_app_json
result = to_app_json(selected, races_all, bias_data, jst_now)

# モデルバージョン保存（チューニング後）
from src.utils.model_registry import save_version
save_version(BASE_DIR, note='2026-05-20 pace修正後', metrics={'acc1': 0.236, 'ece': 0.0005})

# バージョン一覧・ロールバック
from src.utils.model_registry import list_versions, rollback
list_versions(BASE_DIR)
rollback(BASE_DIR, version=1)
```

## セッション履歴
### 2026-05-19：バグ修正・機能追加
1. **escape_count/front_count が常に0だった** → `_parse_shutuba`で集計するよう修正（paceの重みが0.037→0.275に改善）
2. **予想時に騎手・調教師・年齢・斤量が全馬定数だった** → `parser.py`で出馬表から取得、`calc_all`で辞書参照するよう修正
3. **save_history_db** → 毎週末の結果をhistory.dbに自動蓄積（学習データが週次で増加）
4. **SIRE_DB 16頭→58頭に拡充**（血統の重み0.01→0.037に改善）
5. **距離帯別PACE_STYLE_SCORE**（短距離/マイル/中距離/長距離で個別設定）
6. **VENUE_PACE_TENDENCY**（中山=先行有利+0.20、東京/新潟=差し有利）
7. **f_jockey 市場乖離補正**（有名騎手+低オッズ=既に織り込み済み→スコアを下げる）
8. **_infer_running_style改善**（履歴のrunning_styleを多数決で使用）
9. **枠順×距離バイアス・会場×距離成績** をinit_engine時に構築
10. **interval分析**（短間隔≤14日:-0.3、長期休養≥90日:-0.4、好間隔21-35日:+0.2）

### 2026-05-19〜20：モジュール分離リファクタリング
11. **src/models/calibration.py** → PlattCalibrator・IsotonicCalibratorをtools/calibrate.pyから分離
12. **src/models/predict.py** → softmax_probs・calibrate_and_renormalize をengine.pyから分離
13. **src/betting/make_bets.py** → calc_ev・calc_kelly・make_bets・log_bet_simulationをノートブックから分離
14. **src/betting/ev_filter.py** → ability_first_loose をノートブックから分離
15. **src/betting/app_json.py** → to_app_json をノートブックから分離（_build_horses_list等も整理）
16. **src/utils/model_registry.py** → save_version・load_version・rollback・list_versions を新規作成
17. **models/ ディレクトリ** → バージョン管理用スナップショット格納場所

## 残っている課題
| 課題 | 深刻度 | 備考 |
|------|--------|------|
| history.dbが8頭打ち切り（97%のレースが8頭） | 高 | 再スクレイピングで解消可能だが工数大 |
| 騎手DBが20件のみ（重み0.01のまま） | 中 | save_history_dbで蓄積すれば自然解消 |
| bet_simulationのai_probが旧データで0 | 低 | 新データ蓄積で自然解消 |
| analyze_divergenceのバケット分析が機能していない | 低 | 上記に依存 |
| ノートブック側のmake_bets等をsrc/bettingからimportするよう更新 | 中 | 現状はノートブック内にも旧定義が残存 |

## 毎週の運用フロー
1. **金曜夜**: KEIBA_金曜ノートブック実行（翌週レース確認）
2. **土曜朝**: 土日ノートブック実行（出馬表取得・予想生成）
3. **土曜夜**: 土日ノートブック実行（土曜結果取得・save_history_db・照合）
4. **日曜夜**: 土日ノートブック実行（日曜結果取得・save_history_db・照合）
5. **月1〜2回**: チューニングノートブック実行（重み再最適化・キャリブレーション更新）
6. **チューニング後**: save_version(BASE_DIR, ...) でモデルをバージョン保存

## 現在の作業状況（セッション引き継ぎ用）

### 最終更新: 2026-05-20

### 完了済み
- ROI計算バグ修正（`total * 100` → `sum(amount)`）→ analyze_divergence.py・main push済み
- 強制アップデートセルを土日ノートに追加（ユーザーが手動で実施済み）
- チューニングノートの強制アップデートセル：既存のまま機能している
- 日曜結果ノート（Drive）: 2箇所の修正を**ユーザーに指示済み・未確認**
  - セル3b → 強制アップデートセルに差し替え
  - `if place > 8: continue` を削除（fetch_sunday_results内）

### 次にやること（優先順）
1. **金曜ノート**に強制アップデートセルを追加（未対応）
2. **日曜結果ノート**の修正完了を確認
3. **ノートブック側の旧定義を整理**（make_bets等をsrc/bettingからimportするよう更新）
4. 将来: 実行時刻によるレース自動選別機能の実装検討

### 未解決・保留中
- 土日ノート: 8頭打ち切りバグなし（確認済み）
- history.dbの8頭打ち切り問題 → 今後の結果蓄積で自然改善（再スクレイピングは保留）
- 乖離分析バケットが1バケットに集中 → ai_prob=0の旧データが原因、新データ蓄積で自然解消

### セッション開始時の確認事項
- PATをユーザーから取得（毎セッション必要）
- 開発ブランチ: `claude/review-drive-document-ZehuT`
- mainへの反映は明示的に指示があった場合のみ

---

## git操作（PAT使用）
```bash
# PATはユーザーから毎セッション提供される（会話内で確認すること）
PAT="<ユーザーから取得>"
git remote set-url origin "https://${PAT}@github.com/hanagenuku/keiba_ai.git"
git push -u origin <branch-name>
git remote set-url origin "https://github.com/hanagenuku/keiba_ai.git"
```
