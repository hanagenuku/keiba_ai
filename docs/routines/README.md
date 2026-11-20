# Routine の指示文（正本）

Routine の指示文は、その Routine が動いているチャットからしか変更できない
（別のセッションからの更新は拒否される・2026-11-10 に確認）。
そのため、変えたい内容はここに置き、Routine のチャットで
「`docs/routines/<ファイル>` の内容でプロンプトを更新して」と頼む。

| ファイル | Routine | 最後に Routine へ反映した日 |
|---|---|---|
| `morning_prompt.md` | 開催日の朝 08:50（`trig_01ErX4dpJUxkPKdT61w6hefi`） | **2026-11-20**（全部無料・1本版。`scope.day_selection`） |

⚠ **正本は `morning_prompt.md` の1本だけにする。** 2026-11-21 版は
`morning_2026-11-21.md` という別ファイルの下書きとして入った（#389）が、
反映と同時に `morning_prompt.md` へ移して下書きは消した。日付つきの
下書きを残すと、次のセッションがどちらを読めばよいか分からなくなる
（過去の版は git の履歴にある）。

✅ **2026-11-10 に、このチャット（`session_01E2Nf93oj888cNMW2xDjNyR`）から
`update_trigger` でプロンプトを更新できることを実測した。** Routine の
`persistent_session_id` がこのセッションなので、ここからなら通る。
更新したら**この表の日付も直す**（ファイルと Routine がずれたまま残らないように）。
次回発火は 2026-11-21(土) 08:50 JST。
