import random

SEGMENTS = ["400m", "300m", "200m", "100m"]


def simulate_race(horses, jitter=20):
    """各区間ごとの順位（馬番リスト、先頭から最後尾）を返す。

    scoreが高い馬ほど先頭に近くなるが、各区間でscore±jitterのランダム要素を
    加えて再ソートすることで順位変動を演出する。
    """
    positions = {}
    for seg in SEGMENTS:
        ranked = sorted(
            horses,
            key=lambda h: h["score"] + random.uniform(0, jitter),
            reverse=True,
        )
        positions[seg] = [h["number"] for h in ranked]
    return positions
