# history.db スキーマ契約書

> このドキュメントは `data/history.db` の `race_history` / `horse_history` テーブルの
> カラム定義・意味・既知の欠損状況をまとめた「スキーマ契約書」です。
>
> **背景**: history.db は2026-05以降、機能追加のたびに `ALTER TABLE` で有機的に育ってきました。
> カラムが存在することと、実際にデータが埋まっていることは別問題です（例: `bracket`列は
> 存在するが実データは0%）。この契約書は「カラムが存在する＝使えるデータ」という誤解を防ぐために、
> **新しいセッション・新しい特徴量を追加する前に必ず参照すること**を目的としています。
>
> スキーマを変更した場合（`ALTER TABLE`を追加した場合）は、必ずこのファイルも同時に更新してください。

---

## データソース

- 取得元: JRA公式サイト（`www.jra.go.jp`）の内部データベースシステム「JRADB」
  （`accessD.html`=出馬表/結果、`accessO.html`=オッズ、`accessS.html`=結果一覧、`accessU.html`=血統）
- netkeiba等の第三者サイトは使用していない
- **robots.txt / 利用規約の確認について**: この環境（Claude Code on the web）はネットワークポリシーで
  `jra.go.jp` への到達がブロックされているため、robots.txtおよび利用規約（`jra.go.jp/use/`）の内容を
  直接確認できていません。実際にJRAへアクセスできる環境（Colab等）で一度確認し、方針を確定させることを推奨します

---

## race_history テーブル

| カラム | 型 | 意味 | 充足率（2026-06-03時点、4,893件） |
|---|---|---|---|
| race_id | TEXT (PK) | レースID（`YYYYMMDD_場コード_レース番号`） | 100% |
| date | TEXT | 開催日（YYYY-MM-DD） | 100% |
| racecourse | TEXT | 競馬場名 | 100% |
| distance | INTEGER | 距離（m） | 100% |
| surface | TEXT | 芝/ダート/障害 | 100% |
| first_3f | REAL | 前半3F（ラップタイムから算出） | 2026-07-21時点で「未計測」の原因を特定・修正済み（実機確認により`_extract_lap_times`の見出し検索が「ラップタイム」表記のみで、実際のページの「ハロンタイム」表記と不一致だったと判明）。次回実行分から充足率改善見込み |
| race_name | TEXT | レース名 | 100% |
| race_class | TEXT | クラス（新馬/未勝利/1勝クラス/…/G1等） | 94.7% |
| track_condition | TEXT | 馬場状態（良/稍重/重/不良） | 94.6% |
| num_finishers | INTEGER | 出走頭数 | 95.0% |
| race_num | INTEGER | レース番号 | 2026-06-23以降追加、それ以前はNULL |
| lap_times | TEXT | ラップタイム（ハイフン区切り） | 2026-06-23以降追加 |
| last_3f | REAL | 後半3F | 2026-06-23以降追加 |
| weather | TEXT | 天候 | 2026-06-23以降追加 |
| pace_label | TEXT | ペース分類ラベル | train_pace_model.py用、事後生成 |

---

## horse_history テーブル

| カラム | 型 | 意味 | 充足率（2026-06-03時点、67,843件） |
|---|---|---|---|
| id | INTEGER (PK) | 内部連番 | — |
| race_id | TEXT | レースID | 100% |
| date | TEXT | 開催日 | 100% |
| racecourse | TEXT | 競馬場名 | 100% |
| horse_name | TEXT | 馬名 | 100% |
| horse_num | INTEGER | 馬番 | 100% |
| place | INTEGER | 着順 | 100% |
| running_style | TEXT | 脚質（逃げ/先行/差し/追込） | 推定値含む |
| agari3f | REAL | 上がり3F | 高 |
| jockey | TEXT | 騎手名 | 100% |
| trainer | TEXT | 調教師名 | 100% |
| corner_3 | INTEGER | （旧）3コーナー通過順位 | **常にNULL固定。`corner_all`に統合され未使用。新規特徴量では使わないこと** |
| distance | INTEGER | 距離 | 100% |
| surface | TEXT | 芝/ダート | 100% |
| popularity | INTEGER | 確定人気 | 高（未取得時は99固定） |
| tansho_payout | INTEGER | 単勝配当 | 1着馬のみ |
| fukusho_payout | INTEGER | 複勝配当 | 上位馬のみ |
| margin | REAL | 着差（秒換算） | 高 |
| agari_rank | INTEGER | 上がり3F順位（レース内） | 高 |
| class_grade | TEXT | クラス（race_historyから複写） | race_class依存 |
| field_size | INTEGER | （未使用） | **常にNULL。カラムはマイグレーションで追加されているがどこからも書き込まれていない** |
| corner_4 | INTEGER | （未使用） | **常にNULL。同上** |
| finish_time | REAL | 走破タイム（秒） | 95.2% |
| time_diff_sec | REAL | 1着とのタイム差（秒） | finish_time依存 |
| chakusa_text | TEXT | 着差表記（クビ/ハナ等の原文） | 高 |
| weight_load | REAL | 斤量 | 95.2% |
| sex | TEXT | 性別 | 90.8% |
| age | INTEGER | 年齢 | 90.8% |
| body_weight | INTEGER | 馬体重 | **6.5%。Stage3列マッピング崩れが原因と推定、未修正（残っている課題）** |
| body_weight_diff | INTEGER | 馬体重増減 | body_weightに連動、同様に低い |
| bracket | INTEGER | 枠番 | **0%。tx[1]が枠番でない可能性、列マッピング要確認（残っている課題）** |
| corner_all | TEXT | 全通過順位（ハイフン区切り） | 94.5% |
| win_odds | REAL | 単勝オッズ（結果ページ由来） | **0%。tx[11]が単勝オッズでない可能性、列マッピング要確認（残っている課題）** |
| sire | TEXT | 父（種牡馬名） | 2026-07-17〜。新規出走馬のみ順次蓄積、budget=60件/実行で段階的に埋まる |
| dam_sire | TEXT | 母の父 | 同上 |
| trainer_affiliation | TEXT | 調教師所属（栗東/美浦） | 2026-07-21〜。結果ページの調教師欄「名前(栗東)」表記から抽出。今後の結果取得分から順次蓄積（バックフィルなし）。**特徴量化は未実施**（下記「既知の注意点」参照） |

---

## 既知の注意点まとめ

1. **`corner_3` / `field_size` / `corner_4` は常にNULLです。** カラムがCREATE TABLE/ALTER TABLEに存在するからといって使えるデータだとは限りません。特徴量を追加する前に、実際に値が入っているかを`SELECT`で確認してください
2. **`bracket` / `win_odds` / `body_weight`（および連動する`body_weight_diff`）は充足率が著しく低い**（0〜6.5%）。これは`parse_result_soup`の列位置ヒューリスティック（`texts[N]`インデックス）がJRA結果ページの実際の列順とズレている可能性が高く、未修正の既知課題です（DESIGN.md「やってはいけないこと」参照）
3. **`sire`/`dam_sire`（血統）は2026-07-17に追加された新しいカラムで、過去に蓄積された馬には遡って入りません**（バックフィル未実施、スコープ外と判断済み）。今後出走する馬から順次蓄積されます
4. **`trainer_affiliation`（調教師所属）は2026-07-21に追加された新しいカラムで、同様にバックフィルなし**。データ蓄積後、`horse_dist_dict`等と同様の「調教師名→所属」辞書を`_build_horse_dicts()`相当の仕組みで構築し、推論時は出馬表ページの解析結果ではなくこの辞書から引く設計を想定している（出馬表ページで所属表記が同じ形式で取得できるか未検証のため、学習/推論パリティを辞書経由で担保する）。特徴量（例:「今回のレース場が所属地の地元開催か」）は蓄積データが揃うまで未実装
5. スキーマを変更する場合は、`src/utils/db.py`の`migrations`リストに`ALTER TABLE`を追加するだけでなく、**このファイルの表も同時に更新すること**
