import json
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy import VideoFileClip, ImageClip, ImageSequenceClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, ImageClip, ImageSequenceClip, concatenate_videoclips

W, H = 1280, 720
FPS = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GREEN = (0, 255, 65)
RED   = (255, 50, 50)
YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

HORSE_COLORS = [
    (220, 45,  45),
    (50,  110, 215),
    (225, 190, 25),
    (60,  195, 80),
    (180, 65,  215),
    (245, 245, 245),
    (55,  200, 210),
    (230, 100, 175),
    (225, 118, 22),
    (140, 88,  48),
]

def horse_color(number):
    return HORSE_COLORS[(number - 1) % len(HORSE_COLORS)]


def _font(size=16):
    candidates = [
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_glitch(frame, intensity=0.08):
    noise_mask = np.random.random(frame.shape[:2]) < intensity
    frame = frame.copy()
    frame[noise_mask] = np.random.randint(0, 256, (noise_mask.sum(), 3), dtype=np.uint8)
    for _ in range(8):
        y = np.random.randint(0, frame.shape[0])
        shift = np.random.randint(-30, 30)
        frame[y] = np.roll(frame[y], shift, axis=0)
    return frame


# ---------------------------------------------------------------------------
# Phase 1: cinematic.mov
# ---------------------------------------------------------------------------

def _make_cinematic_placeholder(config, duration=8.0):
    """cinematic.mov がない場合のAI映像風プレースホルダー"""
    race_name = config.get("race_name", "RACE")
    fn = _font(32)
    fn_sm = _font(18)
    total_frames = int(duration * FPS)
    frames = []
    for i in range(total_frames):
        t = i / FPS
        img = Image.new("RGB", (W, H), BLACK)
        draw = ImageDraw.Draw(img)

        # スキャンライン背景
        for y in range(0, H, 4):
            alpha = int(15 + 10 * np.sin(y * 0.05 + t * 2))
            draw.line([0, y, W, y], fill=(0, alpha, 0))

        # 流れる縦ライン
        for x_offset in range(0, W, 80):
            x = (x_offset + int(t * 60)) % W
            draw.line([x, 0, x, H], fill=(0, 40, 0))

        # 中央テキスト
        blink = int(t * 3) % 3 != 0
        if blink:
            draw.text((W // 2 - 220, H // 2 - 80), "AI TEMPORAL SCAN", font=_font(52), fill=GREEN)
        draw.text((W // 2 - 160, H // 2),      f"TARGET: {race_name}", font=fn, fill=(0, 200, 50))
        draw.text((W // 2 - 200, H // 2 + 60), "FUTURE EVENT DETECTED", font=fn_sm, fill=(0, 150, 40))
        draw.text((W // 2 - 140, H // 2 + 90), f"T-{int((duration - t) * 10) / 10:.1f}s TO LOCK", font=fn_sm, fill=RED)

        # コーナーフレーム
        sz = 30
        for cx, cy in [(0, 0), (W, 0), (0, H), (W, H)]:
            sx = 1 if cx == 0 else -1
            sy = 1 if cy == 0 else -1
            draw.line([cx, cy, cx + sx * sz, cy], fill=GREEN, width=2)
            draw.line([cx, cy, cx, cy + sy * sz], fill=GREEN, width=2)

        frames.append(np.array(img))
    return ImageSequenceClip(frames, fps=FPS)


def _make_evidence_placeholder(config):
    """evidence.png がない場合の証拠写真風プレースホルダー"""
    race_name = config.get("race_name", "RACE")
    race_date = config.get("race_date", "2026-06-28")
    img = Image.new("RGB", (W, H), (10, 10, 10))
    draw = ImageDraw.Draw(img)

    # グリッド
    for x in range(0, W, 40):
        draw.line([x, 0, x, H], fill=(30, 30, 30))
    for y in range(0, H, 40):
        draw.line([0, y, W, y], fill=(30, 30, 30))

    # 楕円コース
    draw.ellipse([200, 100, 1080, 620], outline=(80, 80, 80), width=3)
    draw.ellipse([320, 200, 960, 520],  outline=(50, 50, 50), width=2)

    # ラベル
    draw.text((W // 2 - 200, H // 2 - 60), f"[EVIDENCE FRAME]", font=_font(48), fill=(160, 160, 160))
    draw.text((W // 2 - 150, H // 2 + 20), race_name, font=_font(36), fill=(120, 120, 120))
    draw.text((W // 2 - 100, H // 2 + 70), race_date, font=_font(24), fill=(90, 90, 90))

    os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
    img.save(os.path.join(BASE_DIR, "assets", "evidence_placeholder.png"))
    return img


def make_phase1(config):
    cine_path = os.path.join(BASE_DIR, "assets", "cinematic.mov")
    clip = VideoFileClip(cine_path)

    # リサイズして黒帯付き1280×720に
    clip = clip.resized(height=720)
    if clip.w != W:
        try:
            clip = clip.on_color(size=(W, H), color=(0, 0, 0))
        except Exception:
            clip = clip.resized((W, H))

    # オーバーレイ: 全フレームに半透明テキストを合成
    fn_small = _font(20)

    def overlay(get_frame, t):
        frame = get_frame(t).copy()
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        lines = ["FUTURE VISION SYSTEM v2.3", "TEMPORAL FOOTAGE – UNVERIFIED"]
        x, y = 18, 16
        pad = 4
        # 黒背景
        bw = 330; bh = 52
        overlay_img = Image.new("RGBA", (bw, bh), (0, 0, 0, 160))
        img.paste(Image.fromarray(np.array(overlay_img)[:, :, :3]),
                  (x - pad, y - pad),
                  Image.fromarray(np.array(overlay_img)[:, :, 3]))
        draw.text((x, y),      lines[0], font=fn_small, fill=WHITE)
        draw.text((x, y + 26), lines[1], font=fn_small, fill=(200, 200, 200))
        return np.array(img.convert("RGB"))

    clip = clip.image_transform(overlay)

    # 末尾1.5秒 ホワイトアウト
    dur = clip.duration

    def white_fade(get_frame, t):
        frame = get_frame(t)
        fade_start = dur - 1.5
        if t < fade_start:
            return frame
        alpha = (t - fade_start) / 1.5
        return (frame * (1 - alpha) + 255 * alpha).astype(np.uint8)

    clip = clip.image_transform(white_fade)
    return clip


# ---------------------------------------------------------------------------
# Phase 2: evidence.png (~4s)
# ---------------------------------------------------------------------------

def make_phase2(config, ev_path=None):
    if ev_path is None:
        ev_path = os.path.join(BASE_DIR, "assets", "evidence.png")
    src = Image.open(ev_path).convert("RGB").resize((W, H), Image.LANCZOS)
    src_arr = np.array(src)

    race_name = config.get("race_name", "RACE")
    race_date = config.get("race_date", "2026.06.28")
    ts = race_date.replace("-", ".")

    fn_hdr = _font(22)
    fn_ts  = _font(28)

    total_dur = 4.0  # seconds
    total_frames = int(total_dur * FPS)
    frames = []

    for i in range(total_frames):
        t = i / FPS

        if t < 0.8:
            # フェードイン
            alpha = t / 0.8
            frame = (src_arr * alpha).astype(np.uint8)
        elif t < 3.2:
            frame = src_arr.copy()
        elif t < 3.6:
            # グリッチ
            intensity = 0.05 + 0.15 * ((t - 3.2) / 0.4)
            frame = add_glitch(src_arr, intensity)
        else:
            # ブラックアウト + SIGNAL DISRUPTED
            alpha = 1.0 - (t - 3.6) / 0.4
            frame = (src_arr * max(0, alpha)).astype(np.uint8)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        if t >= 0.8 and t < 3.6:
            # 左上テキスト
            draw.rectangle([10, 10, 400, 70], fill=(0, 0, 0))
            draw.text((16, 14), "OBSERVATION RECORD", font=fn_hdr, fill=GREEN)
            draw.text((16, 40), f"CAPTURED FRAMES – {race_name}", font=fn_hdr, fill=GREEN)

            # 右下点滅タイムスタンプ
            if int(t * 2) % 2 == 0:
                ts_text = f"██ {ts} 15:40 JST ██"
                draw.rectangle([W - 420, H - 50, W - 10, H - 10], fill=(0, 0, 0))
                draw.text((W - 416, H - 46), ts_text, font=fn_ts, fill=RED)

        if t >= 3.6:
            # SIGNAL DISRUPTED
            fn_big = _font(56)
            draw.text((W // 2 - 240, H // 2 - 30), "SIGNAL DISRUPTED", font=fn_big, fill=RED)

        frames.append(np.array(img.convert("RGB")))

    return ImageSequenceClip(frames, fps=FPS)


# ---------------------------------------------------------------------------
# Phase 3: radar screen (~25s)
# ---------------------------------------------------------------------------

PHASES = [
    (0,  2000, "GATE OPEN",  "ゲートが開きました"),
    (5,  1600, "1600m",      "先頭集団が形成された"),
    (10, 1200, "1200m",      "中団から動きが出てきた"),
    (15, 800,  "800m",       "残り800m！"),
    (20, 400,  "400m",       "映像信号に切替を試みます..."),
]
PHASE_DUR = 5
RADAR_DUR = 25


def _phase_at(t):
    idx = min(int(t / PHASE_DUR), len(PHASES) - 1)
    return PHASES[idx]


def _sorted_horses(horses, t):
    rng = np.random.default_rng(int(t * 10))
    phase_idx = min(int(t / PHASE_DUR), len(PHASES) - 1)
    return sorted(
        horses,
        key=lambda h: h["score"] + rng.uniform(0, 20 - phase_idx * 2),
        reverse=True,
    )


def render_radar_frame(t, config, fn_sm, fn_md, fn_lg, fn_xl):
    img = Image.new("RGB", (W, H), BLACK)
    draw = ImageDraw.Draw(img)

    horses = config["horses"]
    race_name = config.get("race_name", "RACE")
    distance = config.get("distance", 2000)
    race_date = config.get("race_date", "2026.06.28").replace("-", ".")

    phase_t, dist_remain, phase_label, commentary = _phase_at(t)
    ranked = _sorted_horses(horses, t)

    blink = int(t * 2) % 2 == 0

    # ── ヘッダー ──
    dot = "●" if blink else "○"
    hdr = f"FUTURE VISION SYSTEM v2.3  |  TEMPORAL OBSERVATION  |  {dot} SCANNING"
    draw.rectangle([0, 0, W, 36], fill=(0, 30, 0))
    draw.text((14, 8), hdr, font=fn_sm, fill=GREEN)
    draw.line([0, 36, W, 36], fill=GREEN, width=1)

    # ── タイトル ──
    draw.text((14, 50), f"{race_name}  G1  {distance}m", font=fn_lg, fill=GREEN)
    draw.line([0, 90, W, 90], fill=(0, 80, 0), width=1)

    # ── ミニマップ（左200×180） ──
    MAP_X, MAP_Y, MAP_W, MAP_H = 14, 100, 200, 160
    draw.rectangle([MAP_X, MAP_Y, MAP_X + MAP_W, MAP_Y + MAP_H], outline=GREEN, width=1)
    cx, cy = MAP_X + MAP_W // 2, MAP_Y + MAP_H // 2
    rx, ry = 80, 60
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=GREEN, width=2)
    draw.ellipse([cx - rx + 18, cy - ry + 14, cx + rx - 18, cy + ry - 14],
                 outline=(0, 120, 0), width=1)

    progress = 1.0 - dist_remain / distance
    angle_rad = progress * 2 * np.pi - np.pi / 2
    for rank_i, h in enumerate(ranked):
        offset = 0.06 * rank_i
        a = angle_rad - offset
        dx = int(rx * np.cos(a))
        dy = int(ry * np.sin(a))
        dot_x, dot_y = cx + dx, cy + dy
        color = YELLOW if rank_i == 0 else horse_color(h["number"])
        r = 5 if rank_i == 0 else 3
        draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r], fill=color)

    draw.text((MAP_X + 4, MAP_Y + MAP_H + 4), f"← {dist_remain}m  {phase_label}", font=fn_sm, fill=GREEN)

    # ── 右カラム ──
    RX = MAP_X + MAP_W + 24
    ry_pos = 100

    # 残り距離 + プログレスバー
    draw.text((RX, ry_pos), f"残り距離: {dist_remain}m  ▶  {phase_label}", font=fn_md, fill=GREEN)
    ry_pos += 30
    bar_w = W - RX - 20
    filled = int(bar_w * progress)
    draw.rectangle([RX, ry_pos, RX + filled, ry_pos + 14], fill=GREEN)
    draw.rectangle([RX + filled, ry_pos, RX + bar_w, ry_pos + 14], fill=(0, 60, 0))
    draw.rectangle([RX, ry_pos, RX + bar_w, ry_pos + 14], outline=GREEN, width=1)
    ry_pos += 26

    draw.line([RX, ry_pos, W - 10, ry_pos], fill=(0, 80, 0), width=1)
    ry_pos += 8

    # 馬リスト
    name_bar_w = 220
    for rank_i, h in enumerate(ranked):
        rank_label = f"{rank_i + 1}位"
        num_label  = f"[{h['number']:2d}]"
        score_bar  = int(name_bar_w * h["score"] / 100)
        name       = h["name"]

        color = horse_color(h["number"])
        if rank_i == 0:
            draw.rectangle([RX - 2, ry_pos - 2, W - 10, ry_pos + 22], fill=(0, 40, 0))

        draw.text((RX,       ry_pos), rank_label, font=fn_md, fill=GREEN)
        draw.text((RX + 52,  ry_pos), num_label,  font=fn_md, fill=color)
        draw.rectangle([RX + 100, ry_pos + 4, RX + 100 + score_bar, ry_pos + 16], fill=color)
        draw.rectangle([RX + 100 + score_bar, ry_pos + 4,
                        RX + 100 + name_bar_w, ry_pos + 16], fill=(0, 40, 0))
        draw.text((RX + 100 + name_bar_w + 8, ry_pos), name, font=fn_md, fill=WHITE)
        ry_pos += 28

    # ── ログエリア ──
    LOG_Y = H - 120
    draw.line([0, LOG_Y, W, LOG_Y], fill=(0, 80, 0), width=1)
    draw.text((14, LOG_Y + 8), "▶ OBSERVATION LOG", font=fn_sm, fill=(0, 180, 0))
    draw.text((14, LOG_Y + 30), commentary, font=fn_md, fill=GREEN)

    # ── ステータス行 ──
    STATUS_Y = H - 36
    draw.line([0, STATUS_Y, W, STATUS_Y], fill=GREEN, width=1)
    sig = "▓" * 7
    status = f"SIGNAL: {sig} 99%    TEMPORAL COORDS: {race_date}"
    draw.text((14, STATUS_Y + 6), status, font=fn_sm, fill=GREEN)

    # ── フェーズ4末尾: SIGNAL LOST ──
    if t >= 22.0:
        alpha = (t - 22.0) / 3.0  # 0→1 over 3s
        noise_intensity = 0.05 + 0.25 * alpha
        arr = np.array(img)
        # 赤ノイズ
        mask = np.random.random(arr.shape[:2]) < noise_intensity
        red_pixels = np.zeros((mask.sum(), 3), dtype=np.uint8)
        red_pixels[:, 0] = np.random.randint(150, 255, mask.sum())
        arr[mask] = red_pixels
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        if blink:
            draw.text((W // 2 - 160, H // 2 - 50), "SIGNAL LOST", font=fn_xl, fill=RED)
            draw.text((W // 2 - 290, H // 2 + 30),
                      "STAMINA CRITICAL — OBSERVATION TERMINATED AT 50m",
                      font=fn_md, fill=RED)

        if t >= 24.0:
            fade = (t - 24.0) / 1.0
            arr2 = np.array(img)
            arr2 = (arr2 * (1 - fade)).astype(np.uint8)
            img = Image.fromarray(arr2)

    return img


def make_phase3(config):
    fn_sm = _font(16)
    fn_md = _font(20)
    fn_lg = _font(30)
    fn_xl = _font(64)

    total_frames = RADAR_DUR * FPS
    frames = []
    for i in range(total_frames):
        t = i / FPS
        frame = render_radar_frame(t, config, fn_sm, fn_md, fn_lg, fn_xl)
        frames.append(np.array(frame.convert("RGB")))
    return ImageSequenceClip(frames, fps=FPS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(json_path):
    with open(json_path, encoding="utf-8") as f:
        config = json.load(f)

    clips = []

    cine_path = os.path.join(BASE_DIR, "assets", "cinematic.mov")
    if os.path.exists(cine_path):
        print("フェーズ1: cinematic.mov を読み込み中...")
        clips.append(make_phase1(config))
    else:
        print("フェーズ1: cinematic.mov なし → AI映像をプログラム生成")
        clips.append(_make_cinematic_placeholder(config))

    ev_path = os.path.join(BASE_DIR, "assets", "evidence.png")
    if not os.path.exists(ev_path):
        print("フェーズ2: evidence.png なし → 証拠写真をプログラム生成")
        _make_evidence_placeholder(config)
        ev_path = os.path.join(BASE_DIR, "assets", "evidence_placeholder.png")
    print("フェーズ2: 証拠写真を処理中...")
    clips.append(make_phase2(config, ev_path))

    print("フェーズ3: レーダー画面を生成中...")
    clips.append(make_phase3(config))

    final = concatenate_videoclips(clips, method="compose")
    out_path = os.path.join(
        BASE_DIR, "output",
        f"final_{config['race_date']}_{config['race_name']}.mp4"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final.write_videofile(out_path, fps=FPS, codec="libx264", audio=False)
    print(f"✅ 出力完了: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_input.json")
