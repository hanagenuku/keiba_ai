# 馬券戦略 v2 実装サマリ

**実装日**: 2026-06-08  
**ブランチ**: `claude/busy-hamilton-yeao8t`

---

## 実装内容

### タスク1: ev_filter.py — detect_value_horses() 追加

- `VALUE_GAP_THRESHOLD = 0.10` を定数として上部に定義
- `detect_value_horses(horses, market_odds_map)` 関数を追加
  - 市場複勝確率逆算式: `market_prob = 0.8 / fukusho_odds`（控除率20%考慮）
  - `market_odds_map` が空の場合は `value_gap = 0.0` を全馬に設定（エラー防止）
  - `fukusho_odds < 1.0` の場合は `market_prob = 0.0` にフォールバック
  - 結果を `value_gap` 降順でソートして返す

### タスク2: make_bets.py — 3関数追加 + make_bets() 改修

#### 追加関数

1. **`classify_chaos_grade(horses, chaos_score)`**
   - `chaos_score < 0.30` かつ rl_rank=1 の馬の人気 ≤ 2 → `'A'`（堅い）
   - `chaos_score > 0.55` または rl_rank=1 の馬の人気 ≥ 6 → `'C'`（大荒れ）
   - それ以外 → `'B'`（中荒れ）

2. **`select_sanrenpuku_bets(horses, chaos_grade, value_horses)`**
   - 波乱度A: rl_rank 1〜3 のボックス（全員4番人気以内の場合のみ）
   - 波乱度B: バリュー馬が1頭以上いる場合 → バリュー馬軸×rl_rank上位3頭流し
   - 波乱度C: スキップ（三連複は買わない）

3. **`select_bet_type(horses, chaos_grade, value_horses, market_odds_map)`**
   - 波乱度A + バリューあり → 単勝 + 三連複ボックス
   - 波乱度A + バリューなし → 複勝（守り）
   - 波乱度B + バリュー2頭以上 → ワイド + 三連複
   - 波乱度B + バリュー1頭 → 複勝 + ワイド
   - 波乱度B + バリューなし → 複勝
   - 波乱度C + バリューあり → 複勝少額
   - 波乱度C + バリューなし → スキップ（空リスト）

#### make_bets() 改修

- 引数に `market_odds_map=None` を追加（後方互換）
- `market_odds_map` が指定された場合は上記ルールベースで買い目を生成
- `None` の場合は従来の EV×スコアリングロジックにフォールバック

### タスク3: KEIBA_券種選択モデル.ipynb — 新規作成

- セル2の三連複オッズ推定を `chaos_score` に応じた3段階傾斜式に修正:
  - `chaos < 0.30`（断然系）: `max(250, (o1*o2*o3)^0.3 * 60)`
  - `0.30 ≤ chaos < 0.55`（中荒れ）: `max(400, (o1*o2*o3)^0.4 * 80)`
  - `chaos ≥ 0.55`（混戦）: `max(800, (o1*o2*o3)^0.5 * 120)`
- セル3の `groupby` に `observed=False` を追加（FutureWarning 解消）
- セル4に三連複 recall 合格基準チェックを追加（> 0.30）

### タスク4: app_json.py — 新フィールド追加

厳選レースの JSON に以下3フィールドを追加:

```json
{
  "chaos_grade": "B",
  "value_horses": [
    {
      "horse_num": 7,
      "horse_name": "馬名",
      "value_gap": 0.18,
      "cal_prob": 0.32,
      "market_prob": 0.14
    }
  ],
  "bet_reason": "波乱度B・バリュー馬2頭 → ワイド+三連複"
}
```

- `value_gap >= 0.10` の馬のみ `value_horses` に含める
- `value_horses` が空の場合は `[]` を出力

---

## テスト結果

| テスト | 結果 |
|--------|------|
| バリュー馬検出（#3: value_gap=0.16, #7: value_gap=-0.17） | ✅ PASS |
| 空マップフォールバック（value_gap=0.0） | ✅ PASS |
| 波乱度A分類（chaos=0.20, 人気1位） | ✅ PASS |
| 波乱度C分類（chaos=0.62, 人気8位） | ✅ PASS |
| 波乱度A → 単勝+三連複 | ✅ PASS |
| 波乱度C → 三連複なし | ✅ PASS |

---

## 残課題

- `bet_selector_model.pkl` は Colab で `KEIBA_券種選択モデル.ipynb` を実行して生成する必要がある
- 現在のデータ（history.db）は 99% が 8 頭以下・chaos が全部 0.5 と偏っているため、
  モデルの実用性向上には Stage3 補完後のデータで再学習が必要
- `make_bets()` の XGBoost 予測パス（`_BET_SELECTOR`）は `market_odds_map=None` 時のみ有効
