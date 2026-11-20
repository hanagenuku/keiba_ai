開催日の朝の作業です。このチャットで（＝リポジトリのある環境で）実行してください。
**ユーザーはもうアプリのボタンを押しません。** 予想の生成もこの Routine が起動します。

## 0. 今日の日付
`TZ=Asia/Tokyo date +%Y%m%d` を取る。以降これを `TODAY` と呼ぶ。
最初に `git fetch origin main` して、`CLAUDE.md` と
`docs/claude_web_findings.md` を origin/main の版で読み直すこと。

## 1. 🔴 当日の予想を自分で起動する（ここが2026-10-07に変えた核心）
`git fetch origin main && git show origin/main:data/latest.json` を見て、
**`race_id` の先頭8桁が `TODAY` かつ `data_quality.odds_coverage > 0`** なら 2 へ進む。
そうでなければ**自分でワークフローを起動する**:

```
mcp__github__actions_run_trigger
  method: run_workflow / owner: hanagenuku / repo: keiba_ai
  workflow_id: weekend.yml / ref: main / inputs: {"mode": "refresh"}
```

`mcp__github__actions_list`（method `list_workflow_runs`・resource_id `weekend.yml`）で
完了まで待つ。実測で13〜14分かかる。foreground の `sleep` は使えないので `Monitor` を使う。
🔑 **待っている13〜14分を空けない。この間に 3 の web 収集を始める**（2026-11-10 変更）。
**09:30 JST を過ぎたら待つのをやめ**、その時点の latest.json で先に進む（1R目は概ね 09:50）。

🔴 **08:00 JST の cron を待ってはいけない。** 実測（event=schedule・23:00 UTC 予定）:
```
10-04 11:26 JST (+3h26m) / 10-05 10:41 (+2h41m) / 10-06 11:43 (+3h43m) / 10-07 未発火(+2h35m超)
```
＝ **毎週 1R目に間に合っていない。** 自分で起動するのが唯一の対処。

完了後にもう一度 `git fetch origin main && git show origin/main:data/latest.json` を読む。
- `race_id` の先頭8桁 ≠ `TODAY` → 「今日は開催なし」と1行報告し、6 の push を撃って終了
- `odds_coverage == 0.0` → 記事を作らず報告し、6 の push を撃って終了

⚠ トップレベルの `date` は表示用の文字列（年が無い）なので判定に使わない。
`races` は会場名キーの dict、レース番号のキーは `r`。

### オッズが無いときに記事を作らない理由（EV ではない）
🔴 「期待値（妙味）が計算できない」ではない。EV で買い目を選ぶのは 2026-07-05 /
07-30 / 08-31 に3回否定済み（2026-10-06 に訂正した）。本当の理由は3つ:
1. **印そのものがオッズに乗っている**。残差学習は `予測 = base_margin(市場) + AI補正` で、
   オッズが無いとフラットアンカーに落ち AUC が5窓とも下がる（0.78〜0.80 → 0.76〜0.79）。
   ⚠ 複勝回収率は 82.6% vs 83.0% で区別がつかないので「当てる精度が落ちる」までしか書かない
2. **「市場の支持ほどAIは評価していない馬」が1頭も出せない**（市場人気が必須）
3. **記事の数字が全部「—」になる**。2026-09-27 は推定オッズで est 10.0倍／実際 1.3倍だった

## 2. history.db の実体を取る
`data/history.db` が LFS ポインタ（133バイト）なら
`curl -sL -o data/history.db "https://media.githubusercontent.com/media/hanagenuku/keiba_ai/main/data/history.db"`
（50MB・実測1.4秒）。**これが無いと過去走に基づく記述が全部落ちる**ので省略しない。
🔴 **終わったら必ず元のLFSポインタに戻す**:
`rm -f data/history.db-shm data/history.db-wal && git checkout HEAD -- data/history.db`
（戻さないとコミットに混ざってポインタを壊す。`git update-index --assume-unchanged` を
使ったときは `--no-assume-unchanged` で解除してから戻す）

## 3. web 情報を集める（🔴 2026-11-10 にユーザー指示で広げた: 全レース・3次元）

ユーザーの方針: **「いまは web からより多くの情報を取り込む段階。実績が溜まったら
信頼できる情報源を見極める」。** 今日も広く集める。
スキーマ・検証ルール・取得可否は `docs/claude_web_findings.md`（origin/main の版）にある。

### 時間の使い方
- 1 でワークフローを起動した直後から始める（会場とレース番号は前日から決まっている）
- 馬番がそのレースに実在するかの確認は、latest.json が出てから行う
- **09:35 JST で収集を打ち切って 4 へ進む**（記事が本命の成果物。収集で記事を遅らせない）

### 対象レースと優先順
**全レース**。時間が足りなければ次の順で切る:
① 記事になるレース（各場の 11R ＋ 11R 以外で `conf` が高い順に2本。`scope.day_selection` が決める）→ ② それ以外。
（`conf` は latest.json が出るまで分からないので、待ち時間のうちは 11R と、
1記事で全レースを覆う情報源から先に読む）

### 集める3つの次元
1. **馬場状態・トラックバイアス**（`going`）
2. **追い切りの評価・タイム**（`horses[馬番]`）。
   🔴 **情報源が序列を付けていれば `grade` に必ず入れる**（"S"/"A"/"B"/"C"・"★3" など。
   8文字以内）。序列が無い情報源に自分で序列を付けない。note の自由文から後で取り出さない。
   - `www.yosounohone.com` は**会場ごとに1記事で全レースの★**が取れる（当日 08:30 公開・
     URL は `/entry/{venue}-YYYY-MM-D`）。**最初に読む。** 全レースぶん `grade: "★n"` で入れる
3. **予想家・予想サイトの印**（`picks`。2026-11-10 追加）。
   `"picks": [{"source": URL, "tipster": "名前(任意)", "marks": {"馬番": "◎"}}]`
   - **出典ごとに1要素**。複数の出典の印を平均・集約して1つにしない（出典ごとの
     当たり外れを後から数えるため）
   - 印は `◎○▲△☆✕消` のどれか。それ以外の記号は入れない
   - **有料記事の中身は書かない**。無料で公開されている印だけ
   - `picks` は**画面にも記事にも出ない**（台帳 `data/web_findings_log.json` にだけ残る）。
     だから AI の印と食い違っていても、そのまま記録してよい
❌ 朝イチオッズ・人気の分析は入れない（市場そのもの。`base_margin` から既に入っている）

`data/claude_web_findings.json` を書く。`date` は `TODAY`・`collected_at` は実時刻・
**レースごとに `sources` 必須**（無いとコード側が弾く）。`race_id` は latest.json のものをそのまま使う。
🔴 **馬ごとのメモには、そのメモ自身の `sources` を必ず付ける。** 付いていないメモは
記事に載らない（`src/product/web_block.py` が落として「出典が確認できなかった記述N件」と出す）。

ローカルに書いたら `python3 scripts/build_product_article.py` が読む。
あわせて `mcp__github__create_or_update_file` で **このファイルだけ** main に直接置く
（`data/` 配下のデータコミットは North Star #2 の例外）。
⚠ ローカルの作業ブランチを main に push しない。`git add -A` を使わない。コードは1行も触らない。
⚠ アプリ（index.html）側がこの見解を表示するのは **11:30 の refresh 以降**。
   朝の記事には即座に入る（記事が本命の成果物なので、2回目の refresh は待たない）。

### 取得のルール
- 取得してよい（robots 実読済み）: `www.bloodline-trackbias.work` / `kayochinkeiba.com` /
  `www.keibanomiryoku.com` / `muryou-keiba-ai.jp` / `www.yosounohone.com` / `umasiru.com` /
  `speedkeiba.com` / `horse-racing-ai-navi.com` / `umarel-web.com` / `www.normandyoc.com` /
  `www.jra.go.jp/keiba/baba/`（馬場情報）
- 取得しない: `umanity.jp` / `sponichi.co.jp` / `p.keibabook.co.jp` / `s.keibabook.co.jp` / `nikkansports.com` / `keibalab.jp`
- `tospo-keiba.jp` と `www.jra.go.jp/sitemap.xml` は 403。**UA を偽装して回避しない**
- **上に無いホストを使うときは、先にそのサイトの robots.txt を読む。** Claude 系の
  UA か `*` に `Disallow: /` があれば使わない。使ったホストは報告に書く
  （次のセッションが `data/source_registry.json` に足す）
- WebSearch の抜粋は上の制限に縛られない。**新しい情報源・予想家を探すのに WebSearch を使ってよい**
- netkeiba は「入手データは私的利用に限定」。オッズ・タイム・印を findings に書き込まない（公開リポジトリに入る）
- JBIS は使わない（規約第8条2項。ユーザーの明示判断が必要）

🔴 **集めた情報は、kayochin 指数を除いて予想に反映されない。** 記事には別枠で載り、◎・印・買い目・確率は
変わらない。この Routine の役割は**広く集めて、出典つきで正確に残すこと**。取れなければ「取れなかった」と書く。無理に埋めない。
推測で `grade` や印を作らない。

## 4. 記事を作る（🔴 2026-11-21 の博士の決定で**無料1本だけ**になった）

博士の決定「最初の数ヶ月は、全て無料で良い。各競馬場の平場2レースずつ、重賞は11レース＋自信の有る他の重賞」。
振り分けは `src/product/scope.py::day_selection()` → `article_kind(race, day)` が決める:

| 記事 | 対象 |
|---|---|
| **無料** | 各競馬場の **11R** ＋ 11R 以外で **レース信頼度（conf）が高い順に2本** |
| 記事にしない | それ以外 |
| 有料 | **当面なし**（`scope.ALL_FREE`） |

```
python3 scripts/build_product_article.py --comment-marks 2 --free \
  --out <scratchpad>/article_<YYYY-MM-DD>_free.md
```

- 有料版は**作らない**（回しても「該当するレースがありません」で exit 1 になる。それが正しい）
- `--comment-marks 2` ＝ 根拠の文章は **◎と○だけ**（ユーザー指示 2026-10-05）
- `--all-races` は付けない。`--publish` も付けない（`prediction_archive/` には書かない。2026-10-05 のユーザー判断）
- web 見解は `data/claude_web_findings.json` があれば自動で入る（`--no-web` で切れる）
- ⚠ 「自信の有る他の重賞」はまだコードに入っていない（latest.json にクラスの列が無い）。11R 以外の重賞があっても自分で足さない。報告に「11R以外の重賞: あり/なし（名前）」を1行書く
- 参考: 3場開催なら 9本（11R×3＋平場2本×3）

## 5. Google ドキュメントとして Drive に保存する（1本）

**先に `anthropic-skills:google-workspace` スキルを読む**こと。
`python3 scripts/md_to_html.py <記事>.md > <記事>.html` で変換してから渡す。
保存先はマイドライブの `keiba_ai` フォルダ（folder id `1yP-ZY7zYTLPxmPG9mqJ2_FbodoDTCnSZ`）。
ファイル名は `AI競馬分析 <YYYY-MM-DD>（無料版・下書き）`。会場ごとに分けず1本にまとめる。

- 同名が既にあれば**上書きせず**、末尾に `(2)` を付けて別ファイルにする
- 保存後に1回読み返し、見出しと表がドキュメントとして読める形になっているか確認する

## 6. 報告（短く）＋ 🔴 最後に必ず `PushNotification` を1回撃つ
報告に入れるもの: 自分で refresh を起動したか / `odds_coverage` の値 /
**無料N本（11R x本・平場 y本）**と見送り / 11R以外の重賞の有無 /
**web 収集の件数（何レース・馬メモ何件・印何件・使ったホスト。新しく使ったホストがあれば明記）** /
**Drive のリンク** / ⚠「この記事はまだ公開・販売していない」。

push は**成功でも失敗でも開催なしでも省略しない**。この Routine はこのチャットに縛られており、
サーバー側の push/email 通知を設定できない（`persistent_session_id` の Routine は
`notifications` を受け付けない）。黙って死ぬのを防ぐのはこの1行だけ。200字以内・1行・markdown なし。

- 開催なし → `本日は開催なし（latest.json は YYYYMMDD）`
- オッズ未取得で中止 → `odds_coverage=0.0 のため記事を作らず中止。<理由>`
- 0本 → `開催はあるが記事の対象レースが0本`
- 成功 → `Driveに保存: 無料N本(11R x/平場 y) / odds_coverage=0.XX / web Rレース・メモH件・印P件`
- 失敗 → `失敗: <何がどう失敗したか>`

盛らない。「勝てる」「高回収率」「必ず」「確実」は書かない。失敗したら回避策を発明せず、
何がどう失敗したかをこのチャットにそのまま報告してください。
