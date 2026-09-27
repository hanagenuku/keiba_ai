# `data/claude_web_findings.json` — Claude の web 見解の受け渡し仕様

各会場の **10R・11R** について、Claude が開催日の朝に web から集めた情報を
アプリに渡すための1ファイル。読み手は `src/features/claude_web.py`。

## この層が何をしないか（最初に読む）

このプロジェクトでは「予想の出力を後から書き換える層」が **5件すべて撤回**されている
（市場補正レイヤー / `error_tags` 週次補正 / `rank_matrix_filter` / AI xx%バッジ /
ROI予測150%）。そのためこの層は **AI側の数字を1つも変えない**:

- `score` / `total` / `win_prob` / `cal_prob` / `rl_rank` / `fit_rank` /
  `bets` / `gumbel_bets` / `rec` / `conf` を**読まないし書かない**
- 画面では AI の列とは**別枠**（🌐 Claude の web 見解）に出る
- だから後から A/B が組める（AI単体 vs AI＋web を同じレースで実配当で比べられる）

`tests/test_claude_web_findings.py::test_notes_do_not_change_probabilities_or_bets`
がこれをコードで固定している。**このテストを緩めて通さないこと。**

## 期待値（先に書いておく）

- 2026-09-26（N=1）: 別AIが web 情報でアプリの印を組み替えた案は序列の精度が
  ρ +0.381 で、**市場人気順 +0.548 より下**だった
- 2026-09-27（23レース）: web の能力指数とレース内順位の相関は
  **指数~市場 +0.825 / AI~市場 +0.627 / AI~指数 +0.630**
  ＝ **web指数のほうが市場に近い**。独立な情報ではない
- websig（134,874頭・2026-09-25）: AUC は +0.002〜0.004 動き LogLoss は3窓とも改善したが、
  **複勝回収率は 79.7% → 79.1% で悪化**

→ **上乗せを期待しない。** 作ってあるのは「AI単体 vs AI＋web」を同じ条件で
測れる形にするため。採否は実配当で決める。

## 入れてよい情報 / 入れない情報

| | |
|---|---|
| ✅ 入れる | **馬場状態・トラックバイアス**（`f_track_cond` は推論時 0.0=良 固定で、`going.py` は当日1R目が終わるまで効かない） |
| ✅ 入れる | **調教タイム・陣営コメント**（2026-08-16 に「丸ごと欠けている唯一の次元」と確定） |
| ❌ 入れない | web の印の平均・人気分析・朝イチオッズ分析 ＝ **市場オッズの言い換え**。市場は `base_margin` から既に入っている |

対象を 10R/11R に限る理由は2つ。① web の記事が存在するのは特別・重賞だけで、
未勝利・新馬には情報が無い ② 2026-08-18 実測で **OPだけAIの上乗せがマイナス
（−0.0039）** ＝ 情報がある母集団が「磨いても取れない」と測られた母集団と一致する。
**広げないこと。**

## 取得してよい情報源（2026-09-28 時点の実測）

壁は4層あり、`Full` ネットワークポリシーで消えるのは①だけ。

| 層 | 理由 | 対象 |
|---|---|---|
| 1. robots で明示拒否 | サイトが `Disallow: /` | `sponichi.co.jp` / `p.keibabook.co.jp` / `s.keibabook.co.jp` / `umanity.jp` |
| 2. robots は許可だがサーバーが 403 | 出口IP/UAレピュテーション | `tospo-keiba.jp` / `www.jra.go.jp` |
| 3. robots は許可だがサイトが reset | site 側の接続切断 | `keibalab.jp` |
| 4. Claude Code の取得ポリシー | ハーネス側 | `nikkansports.com` |
| 5. **使える** | — | `www.bloodline-trackbias.work` / `kayochinkeiba.com` / `www.keibanomiryoku.com` / `muryou-keiba-ai.jp` / `netkeiba`(API/db) |

🔑 **検索（WebSearch）の層は 1〜4 に縛られない。** 1〜4 のサイトの内容も抜粋で入る。
「本文の全文取得ができない」だけで、情報が取れないのではない。

⚠ **UA を偽装して 403 を回避しない。**
⚠ `umanity.jp` は `ClaudeBot`/`Claude-User`/`Claude-SearchBot`/`anthropic-ai` を名指し拒否。
  `tospo` / `nikkansports` / `umanity` は **`GPTBot`/`ChatGPT-User` も拒否**しており、
  robots の非対称は「Claude だけ不利」ではない。
⚠ `netkeiba` は規約が「入手データは私的利用に限定」。**このファイルに netkeiba 由来の
  オッズ・タイム等を書き込まない**（公開リポジトリに入る）。馬場状態や自分の判断の
  根拠として読むのは可。
⚠ `JBIS` は利用規約第8条2項が「加工・編集することなく…利用するものとします」と
  書いており、**ユーザーの明示的な判断が無いうちは使わない**。

## スキーマ（`schema_version: 1`）

```json
{
  "schema_version": 1,
  "date": "20261003",
  "collected_at": "2026-10-03T07:15:00+09:00",
  "races": {
    "20261003_06_11": {
      "venue": "中山",
      "race_num": 11,
      "going": {"surface": "芝", "state": "重", "expected": "やや重〜重"},
      "order": [9, 13, 14, 15],
      "marks": {"9": "◎", "13": "○", "14": "▲"},
      "horses": {
        "13": {"note": "栗東坂路 52.4-38.2-24.9-12.6。陣営「キャリアで一番」・馬場不問",
               "direction": "up",
               "sources": ["https://www.keibanomiryoku.com/..."]},
        "15": {"note": "陣営は乾いた馬場を好むと明言。今日は重", "direction": "down"}
      },
      "summary": "重馬場で調教評価と道悪適性がはっきりしている2頭を上に見る",
      "sources": ["https://www.bloodline-trackbias.work/...", "https://..."]
    }
  }
}
```

### 検証ルール（`build_claude_view` が実際に弾くもの）

| 条件 | 挙動 |
|---|---|
| `race_num` が 10/11 以外 | `None`（絶対に出ない） |
| `date` が対象日と不一致 | `None` ＋ 警告。**前の開催の印を今日の画面に出さない** |
| レース entry に `sources` が無い | `None` ＋ 警告。裏づけの無い見解は通さない |
| `schema_version` が 1 以外 | ファイルごと無視 ＋ 警告 |
| そのレースに居ない馬番 | その項目だけ落とし、理由を `dropped` に残して画面に出す |
| `order` に重複 | 同じく落として `dropped` に記録 |
| 未定義の印（`◎○▲△☆✕消` 以外） | 落として `dropped` に記録 |
| `order` / `marks` / `horses` が全部空 | `None` |

`coverage`（序列が付いた馬の割合）が 100% 未満なら画面に明記される。
**欠けたまま「序列」と名乗らせない。**

## 運用手順（開催日の朝）

1. `data/latest.json` を `origin/main` から読む
   （⚠ `races` は**会場名キーの dict**、レース番号のキーは `r`）
2. 各会場の 10R/11R について、上表の「使える」情報源＋WebSearch で
   **馬場・トラックバイアス・調教・陣営コメント**を集める
3. `data/claude_web_findings.json` を書く（`date` は開催日・`collected_at` は実時刻）
4. `data/` 配下のデータコミットとして `main` に push する
   （North Star #2 の例外。**コードは触らない**）
5. 08:00 / 11:30 / 14:00 JST の refresh が `latest.json` を作り直すと画面に出る

⚠ **前日夜の生成（19:06 JST）には間に合わない。** `kayochinkeiba` の公開は
前日 19:17〜19:39、`bloodline-trackbias` は 19:00/20:00 固定。拾えるのは当日朝の
refresh 以降。だから朝に書く。

## 後から測るとき

`claude_view.order` と実着順、`rl_rank`（AI+市）と実着順を同じレースで並べれば
**AI単体 / AI＋web / 市場**の3者をレース内 Spearman と実配当で比較できる。
⚠ North Star #7: 数レースの結果で採否を決めない。事前登録カードを書いてから測る。
