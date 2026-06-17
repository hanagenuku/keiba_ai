import numpy as np

try:
    from moviepy import ImageSequenceClip
except ImportError:
    from moviepy.editor import ImageSequenceClip


def export_mp4(frames: list, output_path: str, fps: int = 30):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio=False,
        logger=None,
    )
    print(f"✅ 出力完了: {output_path}")
