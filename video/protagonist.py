import random

TSUKKOMI = {
    "ロケット": [
        "おいおいおい…",
        "それ馬か？",
        "速すぎて目がついていかん",
        "え、なんで？？",
    ],
    "一気": [
        "どこから来たんや",
        "さっきまで最後尾やったやろ",
        "外から何かが飛んできた",
        "え、見えてなかったんやけど",
    ],
    "ワープ": [
        "それ反則やろ",
        "物理法則は？",
        "ワープ禁止ちゃうかったっけ",
        "瞬間移動できるんか…",
    ],
    "未来改変": [
        "AI壊れたんか？",
        "予測が完全に外れとる",
        "未来が変わったぞこれ",
        "俺の予想は何やったんや",
    ],
    "まくり": [
        "3コーナーから来た",
        "じわじわ来てる…怖い",
        "まくりか、まくりやな",
    ],
    "崩壊": [
        "逃げ馬が沈んだ",
        "やっぱり無理やったか",
        "ペース速すぎたんや",
    ],
    "出遅れ": [
        "スタートから終わってる",
        "これは厳しい",
        "ゲート出るの遅すぎ",
    ],
    "ハナ": [
        "一気に先頭取りにいった",
        "逃げる気満々やな",
        "ハイペースになるかも",
    ],
    "battery_warning": [
        "あ、電池やばい",
        "まずい…タイムマシンが",
        "バッテリー警告出てる",
    ],
    "battery_critical": [
        "帰れなくなる",
        "もうちょっとだけ見たい",
        "電力が…",
        "ゴールの直前で…！",
    ],
    "battery_blackout": [
        "（暗転）",
        "…帰ってきた",
        "…結果は見えなかった",
    ],
}


class Protagonist:
    def __init__(self):
        self.current_line: str = ""
        self.display_frames: int = 0
        self.DISPLAY_DURATION: int = 90

    def react(self, event_type: str):
        lines = TSUKKOMI.get(event_type, [])
        if lines:
            self.current_line = random.choice(lines)
            self.display_frames = self.DISPLAY_DURATION

    def react_battery(self, battery_phase: str):
        key = f"battery_{battery_phase}"
        lines = TSUKKOMI.get(key, [])
        if lines and self.display_frames <= 0:
            self.current_line = random.choice(lines)
            self.display_frames = self.DISPLAY_DURATION

    def tick(self):
        if self.display_frames > 0:
            self.display_frames -= 1
        else:
            self.current_line = ""

    def get_display(self) -> str:
        return self.current_line
