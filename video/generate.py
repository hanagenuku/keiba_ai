import json
import sys

from simulator import simulate_race
from renderer import render_video


def main(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    positions = simulate_race(data["horses"])
    output_path = render_video(data, positions, "output/")
    print(f"OK: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1])
