from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tracker import TrackResult


@lru_cache(maxsize=8)
def _font(size: int):
    candidates = [
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _tag(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int,
    background: tuple[int, int, int],
) -> None:
    font = _font(font_size)
    bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0]) + 18
    height = max(1, bbox[3] - bbox[1]) + 12
    height_frame, width_frame = frame.shape[:2]
    x = int(np.clip(x, 0, max(0, width_frame - width)))
    y = int(np.clip(y, 0, max(0, height_frame - height)))
    width, height = min(width, width_frame - x), min(height, height_frame - y)
    if width < 1 or height < 1:
        return
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), background, -1)
    patch = Image.fromarray(
        cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2RGB)
    )
    ImageDraw.Draw(patch).text(
        (9 - bbox[0], 5 - bbox[1]), text, font=font, fill=(255, 255, 255)
    )
    frame[y : y + height, x : x + width] = cv2.cvtColor(
        np.array(patch), cv2.COLOR_RGB2BGR
    )


def draw_tracking(
    frame: np.ndarray,
    result: TrackResult | None,
    label: str,
    fps: float | None = None,
    show_points: bool = True,
    show_fps: bool = True,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    font_size = max(14, min(25, round(width / 55)))
    if result is not None and result.polygon is not None:
        poly = np.rint(result.polygon).astype(np.int32)
        cv2.polylines(image, [poly], True, (15, 15, 15), 5, cv2.LINE_AA)
        cv2.polylines(image, [poly], True, (105, 238, 102), 2, cv2.LINE_AA)
        for x, y in poly:
            cv2.circle(image, (int(x), int(y)), 4, (35, 225, 240), -1, cv2.LINE_AA)
        if show_points:
            for x, y in result.points:
                cv2.circle(
                    image,
                    (int(x), int(y)),
                    2,
                    (0, 184, 255) if 'SIFT' in result.method else (103, 234, 107),
                    -1,
                    cv2.LINE_AA,
                )
        top = int(np.min(result.polygon[:, 1]))
        left = int(np.min(result.polygon[:, 0]))
        _tag(
            image,
            label or 'Объект',
            left,
            top - (font_size + 23),
            font_size,
            (47, 105, 47),
        )
    if show_fps:
        method = result.method if result else '—'
        points = result.inliers if result else 0
        text = (
            f'{method}  |  points: {points}  |  processing: {fps:.1f} FPS'
            if fps
            else f'{method}  |  points: {points}  |  processing: -- FPS'
        )
        _tag(
            image,
            text,
            10,
            height - (font_size + 26),
            max(12, font_size - 3),
            (42, 35, 25),
        )
    return image
