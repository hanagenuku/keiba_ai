# keiba_ai プロジェクト 引き継ぎドキュメント

## プロジェクト概要
JRA競馬AI予想システム。Google Colab + Google Drive で運用。

## リポジトリ
- GitHub: `hanagenuku/keiba_ai`
- 運用ブランチ: `main`（すべての変更はmainに直接push。Colabの強制アップデートセルもmainから取得）
- 旧開発ブランチ: `claude/review-drive-document-ZehuT`（現在は使用していない）

## Colabノートブック構成
| ファイル | 用途 |
|----------|------|
| `KEIBA_土日_v5_ROI.ipynb` | 土曜朝（出馬表取得・予想）、土曜夜・日曜夜（結果取得・照合） |
| `KEIBA_金曜_v5_最新.ipynb` | 金曜夜（翌週レース確認・準備） |
| `KEIBA_チューニング_v1.ipynb` | 月1〜2回：重みチューニング＋キャリブレーション |
| `KEIBA_過去データ一括取得_v4.ipynb` | 過去データ一括取得専用（GitHubには未push・Drive管理）。pw01skl10でJRA月別カレンダーをBFS収集→月範囲指定で一括スクレイピング。ラップタイム・通過順位も取得。8頭打ち切りバグを修正済み（v4_fixed）。 |

> ⚠ `KEIBA_過去データ一括取得_v4.ipynb` はGitHubに含まれていない。Driveのみで管理。

## Google Drive パス
`/content/drive/MyDrive/keiba_ai/`

## データ・モデル構造
```
data/
  history.db      # 学習データ（horse_history: 62,835件 / race_history: 4,646件）
  keiba.db        # 予想・ベット結果（bets, bet_simulation, results）
  optimal_weights.json  # チューニング済み重み
  calibrator.pkl        # Isotonicキャリブレーター
  horse_dist_dict.pkl         # 馬×距離帯成績
  horse_course_dict.pkl       # 馬×コース成績
  horse_venue_dist_dict.pkl   # 馬×競馬場×距離帯成績
  post_zone_bias.pkl          # データ実績枠順バイアス
  month_suffix_map.json       # JRA月別カレンダーBFS収集結果（483ヶ月分）※過去データノートで生成

models/
  current.json    # 現行バージョン番号
  v1/             # バージョンスナップショット
    metadata.json
    calibrator.pkl
    optimal_weights.json
    ...
```

## 最新の重み（2026-05-22チューニング結果 ※8頭補完後）
```
distance : 0.2667
pace     : 0.2666
trainer  : 0.2541
recent   : 0.1625
jockey   : 0.0100
blood    : 0.0100
post     : 0.0100
bias     : 0.0100
weight   : 0.0100
```
Acc@1: 19.6%、ECE: 0.0270、NLL: 2.3317
※Acc@1が下がったのは全頭データ（平均13.5頭）になったため。実質精度は改善。
※騎手DBが20件のみのためjockey重みが0.01に下がっている。蓄積で自然解消予定。

## 強制アップデートセル（チューニングノートのセル1とセル2の間に挿入）
```python
import urllib.request, os, sys
BASE_URL = 'https://raw.githubusercontent.com/hanagenuku/keiba_ai/main'
files = [
    'src/tools/__init__.py',
    'src/tools/tune_weights.py',
    'src/tools/calibrate.py',
    'src/tools/analyze_divergence.py',
    'src/tools/rescrape_history.py',
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
| history.db 8頭打ち切り（一部残存） | 中 | 202501〜202604の補完完了。残り8頭以下は実際の少頭数レースの可能性大 |
| 騎手DBが20件のみ（重み0.01のまま） | 中 | 土日ノートのsave_history_dbで週次蓄積→自然解消 |
| bet_simulationのai_probが旧データで0 | 低 | 新データ蓄積で自然解消 |
| analyze_divergenceのバケット分析が機能していない | 低 | 上記に依存 |
| 過去データノートのセル7（pkl再生成）未実行 | 中 | 補完後にセル7を実行してpkl/CSVを最新化すること |

## 毎週の運用フロー
1. **金曜夜**: KEIBA_金曜ノートブック実行（翌週レース確認）
2. **土曜朝**: 土日ノートブック実行（出馬表取得・予想生成）
3. **土曜夜**: 土日ノートブック実行（土曜結果取得・save_history_db・照合）
4. **日曜夜**: 土日ノートブック実行（日曜結果取得・save_history_db・照合）
5. **月1〜2回**: チューニングノートブック実行（重み再最適化・キャリブレーション更新）
6. **チューニング後**: save_version(BASE_DIR, ...) でモデルをバージョン保存

## 現在の作業状況（セッション引き継ぎ用）

### 最終更新: 2026-05-25

---

## ⚠️ 重要：設計指針書（必ず読むこと）

**`DESIGN.md`（このリポジトリのルート）に詳細な設計ロードマップがある。**
Googleドライブにも「keiba_ai 設計方針書（ロードマップ）」として保存済み。

### 設計の核心（要約）
- RL（レースレベル）＝絶対能力軸、CL（コースレベル）＝適性軸 の2本柱で評価する
- 市場オッズは特徴量に入れない（過去に議論済み・決定事項）
- 現在の最大問題：history.dbにrace_class・agari_rank・margin等がない→データ層の拡張が最優先

### 次にやること（優先順）
1. **Phase 0**：RL/CL分離表示の実装（engine.pyにcalc_rl_cl_ranks()を追加）
2. **Phase 1**：history.dbスキーマ拡張（race_class, agari_rank, margin, last_3f等を追加）
3. **過去データノートのセル7実行**（pkl・CSV再生成）← まだ未実行

### 2026-05-25 完了済み
- **スマホアプリ配信をGAS廃止→GitHub Pages直接配信に変更**
  - `data/latest.json` をGitHubにプッシュ、index.htmlが直接取得する方式
  - ノートブックの予想生成セルに `push_to_github(app_data, PAT)` を追加済み
  - 土日ノート・金曜ノート両方に追加が必要（金曜ノートは `app_data_sat` を使用）
- **Softmax temperature: 1.3 に調整**（ev_filter.pyのEV_THRESHOLD=1.05, WIN_PROB_MIN=0.06に変更）
- **DESIGN.md 新規作成**（RL/CLフレームワーク設計指針書）

---

### 完了済み（2026-05-22）
- **history.db 8頭打ち切り補完**：過去データ取得ノート（v4_fixed）で202501〜202604を再スクレイピング
  - horse_history: 34,086件 → 62,835件（約1.8倍）
  - 平均出走頭数: 8頭 → 13.5頭（全頭取得に改善）
- **過去データ取得ノートの修正**（`if place>8: continue`削除・save_race_history補完処理追加・スキップ条件修正）
  - 修正済みファイル（`KEIBA_過去データ一括取得_v4_fixed.ipynb`）をDriveに配置済み
- **重みチューニング・キャリブレーション再実行**（8頭補完後のデータで再最適化）
  - ECE: 0.0726 → 0.0270（大幅改善）
- **KAISAI_CALENDAR を2026年末まで更新**（5月〜12月の全会場・全開催日を追加）
  - 阪神 kai=03 の6/27-28誤記も修正
- **src/tools/rescrape_history.py** 新規作成（土日ノート末尾から使える補完ツール）

### 完了済み（2026-05-21以前）
- EV_THRESHOLD 1.05→1.10、WIN_PROB_MIN 0.06→0.08 に引き上げ
- ROI計算バグ修正
- 全ノート：強制アップデートセル・engine.py import化
- モジュール分離リファクタリング（betting/models/tools各種）

### 次にやること（優先順）
1. **過去データノートのセル7実行**（pkl・CSV再生成）← まだ未実行！チューニング前に必要だったが後回しになった。次回チューニング前に必ず実行すること
2. **週末の実運用**で動作確認
3. **騎手DBの充実**：土日ノートのsave_history_dbで週次蓄積→自然解消待ち

### 過去データ取得ノートの使い方
- Drive管理（GitHubには未push）：`KEIBA_過去データ一括取得_v4_fixed.ipynb`
- セル1→2→3→4（suffix_map読込）→6（月範囲指定）→7（pkl生成）の順で実行
- 月ごとに分割実行可能（途中停止→再開OK、スキップ機能あり）
- suffix_mapは483ヶ月分収集済み。再収集不要（新しい月が必要なら強制再収集セルを追加して実行）

### 未解決・保留中
- 乖離分析バケットが1バケットに集中 → ai_prob=0の旧データが原因、新データ蓄積で自然解消
- 調教タイム追加：保留（距離・馬場種別の正規化が複雑なため）
- オッズ変動の追跡：保留（複数回のノート手動実行が必要なため）
- 複数回実行時のbets重複問題：現状は「最後の予想で判断」の運用ルールで対応

### セッション開始時の確認事項
- PATをユーザーから取得（毎セッション必要）
- すべての変更は **main ブランチ**に直接push
- **ローカルgitリポジトリは使用しない**（家PC・会社PCともにローカルcloneは不要）
- GitHubに**ないファイル**（過去データ取得ノートなど）はユーザーに確認してから作業する

---

## git操作（PAT使用）
```bash
# PATはユーザーから毎セッション提供される（会話内で確認すること）
PAT="<ユーザーから取得>"
git remote set-url origin "https://${PAT}@github.com/hanagenuku/keiba_ai.git"
git push -u origin <branch-name>
git remote set-url origin "https://github.com/hanagenuku/keiba_ai.git"
```
