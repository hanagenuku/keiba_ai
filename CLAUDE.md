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

## データ構造
```
data/
  history.db      # 学習データ（horse_history: 35,134件 / race_history: 4,414件）
  keiba.db        # 予想・ベット結果（bets, bet_simulation, results）
  optimal_weights.json  # チューニング済み重み
  calibrator.pkl  # Isotonicキャリブレーター
  horse_dist_dict.pkl         # 馬×距離帯成績
  horse_course_dict.pkl       # 馬×コース成績
  horse_venue_dist_dict.pkl   # 馬×競馬場×距離帯成績（新規）
  post_zone_bias.pkl          # データ実績枠順バイアス（新規）
```

## 最新の重み（2026-05-19時点）
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
    'src/scraper/parser.py',
    'src/scraper/jra_scraper.py',
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
| `src/scraper/jra_scraper.py` | JRAサイトスクレイピング。出馬表・結果取得 |
| `src/scraper/parser.py` | HTML解析。parse_horse（騎手・年齢・斤量・父名取得） |
| `src/tools/tune_weights.py` | 重みチューニング（scipy SLSQP、5回試行） |
| `src/tools/calibrate.py` | Isotonicキャリブレーション |
| `src/tools/analyze_divergence.py` | AI確率vs市場オッズ乖離分析 |
| `src/utils/db.py` | DB操作。save_history_db（結果→history.db自動蓄積）含む |
| `src/utils/config.py` | 設定。POST_BIAS_BY_ZONE、KAISAI_CALENDAR等 |

## 直近セッション（2026-05-19）で実施した改善
### バグ修正
1. **escape_count/front_count が常に0だった** → `_parse_shutuba`で集計するよう修正（paceの重みが0.037→0.275に改善）
2. **予想時に騎手・調教師・年齢・斤量が全馬定数だった** → `parser.py`で出馬表から取得、`calc_all`で辞書参照するよう修正

### 機能追加
3. **save_history_db** → 毎週末の結果をhistory.dbに自動蓄積（学習データが週次で増加）
4. **SIRE_DB 16頭→58頭に拡充**（血統の重み0.01→0.037に改善）
5. **距離帯別PACE_STYLE_SCORE**（短距離/マイル/中距離/長距離で個別設定）
6. **VENUE_PACE_TENDENCY**（中山=先行有利+0.20、東京/新潟=差し有利）
7. **f_jockey 市場乖離補正**（有名騎手+低オッズ=既に織り込み済み→スコアを下げる）
8. **_infer_running_style改善**（履歴のrunning_styleを多数決で使用）
9. **枠順×距離バイアス・会場×距離成績** をinit_engine時に構築
10. **interval分析**（短間隔≤14日:-0.3、長期休養≥90日:-0.4、好間隔21-35日:+0.2）

## 残っている課題
| 課題 | 深刻度 | 備考 |
|------|--------|------|
| history.dbが8頭打ち切り（97%のレースが8頭） | 高 | 再スクレイピングで解消可能だが工数大 |
| 騎手DBが20件のみ（重み0.01のまま） | 中 | save_history_dbで蓄積すれば自然解消 |
| bet_simulationのai_probが旧データで0 | 低 | 新データ蓄積で自然解消 |
| analyze_divergenceのバケット分析が機能していない | 低 | 上記に依存 |

## 毎週の運用フロー
1. **金曜夜**: KEIBA_金曜ノートブック実行（翌週レース確認）
2. **土曜朝**: 土日ノートブック実行（出馬表取得・予想生成）
3. **土曜夜**: 土日ノートブック実行（土曜結果取得・save_history_db・照合）
4. **日曜夜**: 土日ノートブック実行（日曜結果取得・save_history_db・照合）
5. **月1〜2回**: チューニングノートブック実行（重み再最適化・キャリブレーション更新）

## git操作（PAT使用）
```bash
# PATはユーザーから毎セッション提供される（会話内で確認すること）
PAT="<ユーザーから取得>"
git remote set-url origin "https://${PAT}@github.com/hanagenuku/keiba_ai.git"
git push -u origin <branch-name>
git remote set-url origin "https://github.com/hanagenuku/keiba_ai.git"
```
