import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy import VideoClip, concatenate_videoclips
except ImportError:  # moviepy < 2.0
    from moviepy.editor import VideoClip, concatenate_videoclips

# 内部描画解像度（ドット絵） -> 4倍拡大して出力
W, H = 320, 180
SCALE = 4
OUT_W, OUT_H = W * SCALE, H * SCALE
FPS = 30

BG_COLOR = (0, 0, 0)
GREEN = (0, 255, 64)
RED = (255, 32, 32)
GRASS_COLOR = (34, 120, 50)
GRASS_DARK = (28, 100, 42)
WHITE = (255, 255, 255)
SKY_COLOR = (120, 190, 230)
STAND_COLOR = (150, 150, 160)
STAND_DARK = (110, 110, 120)
RAIL_COLOR = (235, 235, 235)
DIALOG_BG = (250, 250, 250)
DIALOG_BORDER = (20, 20, 20)
TEXT_COLOR = (20, 20, 20)
BODY_COLOR = (120, 80, 50)

HORSE_COLORS = [
    (230, 60, 60),
    (60, 130, 230),
    (240, 200, 40),
    (90, 220, 120),
    (200, 100, 230),
    (240, 140, 40),
    (80, 220, 220),
    (220, 220, 220),
]

LEG_COLOR = (40, 30, 20)

# 横向き（側面視点）の馬+ジョッキー ドット絵。ギャロップの2フレーム。
# '.'=透明 'B'=馬体 'J'=ジョッキー(馬番カラー) 'D'=脚
HORSE_SIDE_FRAMES = [
    [
        "....JJ......",
        "...JJJJ.....",
        "...BBBBBB...",
        "..BBBBBBBB..",
        ".BBBBBBBBBB.",
        "D.D....D.D..",
        "D.D....D.D..",
        "............",
    ],
    [
        "....JJ......",
        "...JJJJ.....",
        "...BBBBBB...",
        "..BBBBBBBB..",
        ".BBBBBBBBBB.",
        "...D....D...",
        "...D....D...",
        "............",
    ],
]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

_font_cache = {}


def get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def horse_color(number):
    return HORSE_COLORS[(number - 1) % len(HORSE_COLORS)]


def draw_horse_side(draw, x, y, jockey_color, frame=0):
    """側面視点の馬+ジョッキーを (x, y) を左上として描画する。"""
    bitmap = HORSE_SIDE_FRAMES[frame % len(HORSE_SIDE_FRAMES)]
    for row, bits in enumerate(bitmap):
        for col, bit in enumerate(bits):
            if bit == ".":
                continue
            color = {"B": BODY_COLOR, "J": jockey_color, "D": LEG_COLOR}[bit]
            px, py = x + col, y + row
            if 0 <= px < W and 0 <= py < H:
                draw.point((px, py), fill=color)


def generate_commentary(horses_by_number, initial_order, positions, segments):
    """各区間の実況テキストを生成する。"""
    texts = []
    prev_order = initial_order
    for i, seg in enumerate(segments):
        order = positions[seg]
        leader = horses_by_number[order[0]]["name"]
        chaser = horses_by_number[order[1]]["name"] if len(order) > 1 else ""
        prev_leader = horses_by_number[prev_order[0]]["name"]

        if order[0] != prev_order[0]:
            text = f"{leader}が{prev_leader}をかわした！{chaser}食いさがる"
        else:
            text = f"{leader}が先頭を守る！{chaser}食いさがる"

        if seg == segments[-1]:
            text += f" ゴール！{leader}が一着！"

        texts.append(text)
        prev_order = order
    return texts


def draw_dialog_box(draw, leader_color, text, progress):
    """画面下部に実況テロップのダイアログボックスを描画する。"""
    box_y0 = H - 32
    draw.rectangle([0, box_y0, W, H], fill=DIALOG_BG)
    draw.rectangle([0, box_y0, W, H], outline=DIALOG_BORDER, width=2)

    # 左側のポートレート枠
    draw.rectangle([4, box_y0 + 4, 28, H - 4], fill=leader_color)
    draw.rectangle([4, box_y0 + 4, 28, H - 4], outline=DIALOG_BORDER, width=1)

    font = get_font(9)
    lines = textwrap.wrap(text, width=20)
    chars_shown = int(sum(len(line) for line in lines) * progress)

    remaining = chars_shown
    ty = box_y0 + 5
    for line in lines:
        if remaining <= 0:
            break
        show = line[: min(len(line), remaining)]
        remaining -= len(show)
        draw.text((34, ty), show, fill=TEXT_COLOR, font=font)
        ty += 11


def to_frame(img):
    """320x180 の PIL Image を 1280x720 にニアレストネイバーで拡大しndarray化"""
    img = img.resize((OUT_W, OUT_H), Image.NEAREST)
    return np.array(img)


# ---------------------------------------------------------------------------
# シーン1: ブート画面（3秒）
# ---------------------------------------------------------------------------

def make_scene1(race_name, duration=3.0):
    lines = [
        "FUTURE VISION SYSTEM v2.3",
        "BOOTING...",
        f"TARGET COORDINATES: {race_name}",
        "TEMPORAL JUMP INITIATED",
    ]
    total_chars = sum(len(line) for line in lines)
    font = get_font(10)

    def make_frame(t):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        progress = min(t / duration, 1.0)
        chars_to_show = int(total_chars * progress)

        remaining = chars_to_show
        y = 30
        for line in lines:
            if remaining <= 0:
                break
            show = line[: min(len(line), remaining)]
            remaining -= len(show)
            draw.text((10, y), show, fill=GREEN, font=font)
            y += 18

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# シーン2: レース本体（40秒）
# ---------------------------------------------------------------------------

def make_scene2(data, positions, duration=40.0):
    horses = data["horses"]
    n = len(horses)
    horses_by_number = {h["number"]: h for h in horses}
    numbers_sorted = sorted(h["number"] for h in horses)
    lane_of = {number: i for i, number in enumerate(numbers_sorted)}

    initial_order = [h["number"] for h in sorted(horses, key=lambda h: h["score"], reverse=True)]
    segments = ["400m", "300m", "200m", "100m"]
    seg_duration = duration / len(segments)
    commentary = generate_commentary(horses_by_number, initial_order, positions, segments)

    font_label = get_font(12)

    track_y0, track_y1 = 38, H - 32
    lane_height = (track_y1 - track_y0) / max(n, 1)

    BASE_X = 60
    SPREAD = 170

    def smoothstep(p):
        p = max(0.0, min(1.0, p))
        return p * p * (3 - 2 * p)

    def make_frame(t):
        img = Image.new("RGB", (W, H), SKY_COLOR)
        draw = ImageDraw.Draw(img)

        # スタンド（簡易）
        draw.rectangle([0, 14, W, 28], fill=STAND_COLOR)
        scroll_far = int(t * 20) % 16
        for sx in range(-16, W + 16, 16):
            draw.rectangle([sx - scroll_far, 16, sx - scroll_far + 8, 26], fill=STAND_DARK)

        # 馬場（芝）と横スクロールするストライプ
        draw.rectangle([0, track_y0, W, track_y1], fill=GRASS_COLOR)
        scroll = int(t * 60) % 20
        for sx in range(-20, W + 20, 20):
            draw.rectangle(
                [sx - scroll, track_y0, sx - scroll + 10, track_y1],
                fill=GRASS_DARK,
            )

        # 内側ラチ（白線）
        draw.rectangle([0, track_y0 - 3, W, track_y0], fill=RAIL_COLOR)

        seg_index = min(int(t // seg_duration), len(segments) - 1)
        seg = segments[seg_index]
        seg_progress = (t % seg_duration) / seg_duration
        ease = smoothstep(seg_progress)

        prev_order = initial_order if seg_index == 0 else positions[segments[seg_index - 1]]
        cur_order = positions[seg]

        gallop_frame = int(t * 8) % 2

        for number in numbers_sorted:
            prev_rank = prev_order.index(number)
            cur_rank = cur_order.index(number)
            rank = prev_rank + (cur_rank - prev_rank) * ease

            offset = (n - 1 - rank) * (SPREAD / max(n - 1, 1))
            x = int(BASE_X + offset)
            lane = lane_of[number]
            y = int(track_y0 + lane * lane_height + (lane_height - 8) / 2)
            y += -1 if gallop_frame == 0 else 0

            draw_horse_side(draw, x, y, horse_color(number), frame=gallop_frame)
            draw.text((x + 1, y - 9), str(number), fill=WHITE, font=get_font(8))

        # 残り距離表示
        draw.text((6, 4), f"残り{seg}", fill=WHITE, font=font_label)

        # 実況テロップ
        text = commentary[seg_index]
        box_progress = min(seg_progress / 0.6, 1.0)
        leader_color = horse_color(cur_order[0])
        draw_dialog_box(draw, leader_color, text, box_progress)

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# シーン3: SIGNAL LOST（7秒）
# ---------------------------------------------------------------------------

def make_scene3(duration=7.0):
    font_big = get_font(20)
    font_small = get_font(10)

    def make_frame(t):
        noise = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        draw = ImageDraw.Draw(img)

        if int(t * 4) % 2 == 0:
            text = "SIGNAL LOST"
            bbox = draw.textbbox((0, 0), text, font=font_big)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((W - tw) / 2, (H - th) / 2 - 10), text, fill=RED, font=font_big)

        sub_text = "STAMINA CRITICAL - OBSERVATION TERMINATED"
        bbox = draw.textbbox((0, 0), sub_text, font=font_small)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) / 2, H - 30), sub_text, fill=RED, font=font_small)

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# シーン4: 最終予想（10秒）
# ---------------------------------------------------------------------------

def make_scene4(data, duration=10.0):
    horses = data["horses"]
    top3 = sorted(horses, key=lambda h: h["score"], reverse=True)[:3]
    bets = data.get("recommended_bets", [])
    font_title = get_font(16)
    font_body = get_font(11)

    def make_frame(t):
        img = Image.new("RGB", (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)

        draw.text((20, 10), "AI ANALYSIS RESULT", fill=GREEN, font=font_title)

        y = 36
        for i, h in enumerate(top3):
            line = f"{i + 1}. {h['number']} {h['name']} (score {h['score']})"
            draw.text((20, y), line, fill=horse_color(h["number"]), font=font_body)
            y += 16

        y += 8
        draw.text((20, y), "RECOMMENDED BETS", fill=GREEN, font=font_body)
        y += 14
        for bet in bets:
            draw.text((20, y), f"- {bet}", fill=WHITE, font=font_body)
            y += 14

        if t > duration - 3:
            text = "GOOD LUCK"
            bbox = draw.textbbox((0, 0), text, font=font_title)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((W - tw) / 2, H - 30), text, fill=GREEN, font=font_title)

        return to_frame(img)

    return VideoClip(make_frame, duration=duration)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def render_video(data, positions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    clip1 = make_scene1(data["race_name"])
    clip2 = make_scene2(data, positions)
    clip3 = make_scene3()
    clip4 = make_scene4(data)

    final = concatenate_videoclips([clip1, clip2, clip3, clip4])

    filename = f"race_{data['race_date']}_{data['race_name']}.mp4"
    output_path = os.path.join(output_dir, filename)
    final.write_videofile(output_path, fps=FPS, codec="libx264", audio=False)
    return output_path
