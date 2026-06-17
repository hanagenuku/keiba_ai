import json
import sys
import os

from horse import Horse
from race_engine import RaceEngine
from commentary import generate_commentary
from renderer import render_frame, render_title_card, render_result_card, DramaState
from composer import export_mp4
from battery import BatteryState
from protagonist import Protagonist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def horse_to_dict(h: Horse) -> dict:
    return {
        "number":       h.number,
        "name":         h.name,
        "style":        h.style,
        "power":        h.power,
        "jockey_color": h.jockey_color,
        "screen_x":     h.screen_x,
        "rank":         h.rank,
        "comment":      h.comment,
    }


def main(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    horses = [
        Horse(
            number=h["number"],
            name=h["name"],
            style=h["style"],
            power=h["power"],
            jockey_color=h.get("jockey_color", "#888888"),
            expected_rank=h.get("expected_rank", 0),
            late_speed=h.get("late_speed", 50),
            pace_fit=h.get("pace_fit", "M"),
            drama_hint=h.get("drama_hint"),
            comment=h.get("comment", ""),
        )
        for h in data["horses"]
    ]

    engine      = RaceEngine(horses)
    battery     = BatteryState()
    protagonist = Protagonist()
    drama       = DramaState()
    prev_battery_phase = "normal"

    frames = []
    commentary = ["スタート！", "各馬一斉にスタート！"]
    bg_scroll  = 0.0

    print("🎬 タイトルカード生成中...")
    frames += render_title_card(data, len(horses))

    print("🏇 レースシミュレーション開始...")
    for frame_idx in range(engine.TOTAL_FRAMES):
        engine.step()
        bg_scroll += 3.5
        dist = engine.dist_remaining()

        battery.update(dist)

        # ドラマ発動チェック → DramaState 更新 + ツッコミ
        newly = engine.drama.get_newly_fired_events(dist)
        for evt in newly:
            protagonist.react(evt.event_type)
            if evt.event_type in ("ロケット", "一気", "ワープ"):
                drama.event     = {"ロケット":"rocket","一気":"charge","ワープ":"warp"}[evt.event_type]
                drama.intensity = 1.0
                drama.flash     = 0.35
            elif evt.event_type == "まくり":
                drama.event     = "makuri"
                drama.intensity = 0.7

        # ドラマを徐々にフェードアウト
        drama.flash     = max(0.0, drama.flash     - 0.04)
        drama.intensity = max(0.0, drama.intensity - 0.015)
        if drama.intensity <= 0:
            drama.event = ""

        # 電池フェーズ変化 → ツッコミ
        if battery.phase != prev_battery_phase:
            protagonist.react_battery(battery.phase)
            prev_battery_phase = battery.phase

        # 実況更新
        if engine.commentary_queue:
            entry = engine.commentary_queue.pop(0)
            drama_active = engine.drama.get_active_events(dist)
            commentary = generate_commentary(
                entry["dist"], entry["horses"], drama_active)

        h_dicts = [horse_to_dict(h) for h in horses]

        frame = render_frame(
            horses=h_dicts,
            bg_scroll=bg_scroll,
            commentary=commentary,
            dist_remaining=dist,
            tick=frame_idx,
            race_info=data,
            tsukkomi=protagonist.get_display(),
            battery_phase=battery.phase,
            battery_intensity=battery.noise_intensity,
            drama=drama,
        )

        frames.append(frame)
        protagonist.tick()

        if battery.is_finished():
            print(f"⚡ 電池切れ！残り{dist:.0f}mで強制終了")
            break

        if frame_idx % 100 == 0:
            pct = frame_idx / engine.TOTAL_FRAMES * 100
            print(f"  {pct:.0f}% (残り{dist:.0f}m)")

    print("🎬 結果カード生成中...")
    h_dicts = [horse_to_dict(h) for h in horses]
    frames += render_result_card(h_dicts, data)

    out_name = f"race_{data['race_date']}_{data['race_name']}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    print("🎬 動画エンコード中...")
    export_mp4(frames, out_path, fps=30)


if __name__ == "__main__":
    json_input = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "test_input.json")
    main(json_input)
