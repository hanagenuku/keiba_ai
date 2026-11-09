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
