import os

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

# 8x8 ドット絵の馬シルエット
HORSE_BITMAP = [
    "00111000",
    "01111100",
    "11111110",
    "11111111",
    "01111110",
    "00100100",
    "00100100",
    "01100110",
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


def draw_horse_icon(draw, x, y, color):
    """8x8 ドット絵の馬を (x, y) を左上として描画する。"""
    for row, bits in enumerate(HORSE_BITMAP):
        for col, bit in enumerate(bits):
            if bit == "1":
                px, py = x + col, y + row
                if 0 <= px < W and 0 <= py < H:
                    draw.point((px, py), fill=color)


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
    full_text = "\n".join(lines)
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
    segments = ["400m", "300m", "200m", "100m"]
    seg_duration = duration / len(segments)
    font_small = get_font(10)
    font_label = get_font(14)

    track_y0, track_y1 = 10, 150
    lane_height = (track_y1 - track_y0) / max(n, 1)

    def make_frame(t):
        img = Image.new("RGB", (W, H), GRASS_COLOR)
        draw = ImageDraw.Draw(img)

        # 横スクロールする芝のストライプ
        scroll = int(t * 40) % 20
        for sx in range(-20, W + 20, 20):
            draw.rectangle(
                [sx - scroll, track_y0, sx - scroll + 10, track_y1],
                fill=GRASS_DARK,
            )

        seg_index = min(int(t // seg_duration), len(segments) - 1)
        seg = segments[seg_index]
        order = positions[seg]  # 先頭(1着想定)から最後尾の馬番リスト

        # 残り距離表示
        distance_label = seg
        draw.text((W - 70, 8), distance_label, fill=WHITE, font=font_label)

        # 馬を描画（順位が高いほど右側=先頭側に近づく）
        track_width = W - 40
        for rank, number in enumerate(order):
            x = 10 + int(track_width * (1 - rank / max(n - 1, 1)) * 0.85)
            y = int(track_y0 + rank * lane_height) + 2
            draw_horse_icon(draw, x, y, horse_color(number))
            draw.text((x - 2, y - 10), str(number), fill=WHITE, font=font_small)

        # 右側に現在順位を表示
        draw.rectangle([W - 60, 0, W, H], fill=(0, 0, 0))
        draw.text((W - 58, 4), "RANK", fill=WHITE, font=font_small)
        for i, number in enumerate(order):
            name = horses_by_number[number]["name"]
            label = f"{i + 1}.{number} {name[:4]}"
            draw.text((W - 58, 16 + i * 12), label, fill=horse_color(number), font=font_small)

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
