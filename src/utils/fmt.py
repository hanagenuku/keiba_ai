"""表示用の整形ヘルパー（値が欠けていても落ちないこと）。"""


def format_odds(value, unknown='オッズ不明'):
    """単勝オッズをログ・画面表示用の文字列にする。

    🔴 `win_odds` は「キーはあるが値が None」という状態を取りうる:
      - 出馬表ページにオッズ列が無い（前日夜の生成時はこれが普通）
      - `_sanitize_odds_book` / `_sanitize_win_odds` が壊れた盤を丸ごと無効化した

    `h.get('win_odds', 0)` の既定値は**キーが存在しない時にしか効かない**ため、
    素朴に `f'{h.get("win_odds", 0):.1f}倍'` と書くと None で TypeError になる。
    2026-09-12 の土曜夜実行（weekend.py --mode saturday）はこれで落ち、
    日曜の予想が丸ごと生成されなかった。
    2026-08-07④の `agari_rank=None` と同じ「欠損の表し方が .get() で救えない」型。

    オッズ不明のときは 0.0倍 と偽らず、そう書く（作り話をしない）。
    単勝オッズは元返しの 1.0倍 が下限なので、1.0 未満も「不明」として扱う。
    """
    if value is None:
        return unknown
    try:
        v = float(value)
    except (TypeError, ValueError):
        return unknown
    if v < 1.0:
        return unknown
    return f'{v:.1f}倍'
