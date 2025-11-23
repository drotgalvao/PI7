from typing import Tuple, Union
import re
import cv2
import numpy as np


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    if not hex_color:
        raise ValueError("hex_color vazio")
    hex_color = hex_color.strip()
    names = {
        "white": "FFFFFF",
        "black": "000000",
        "brown": "b58863",
        "beige": "f0d9b5",
    }
    if hex_color.lower() in names:
        hex_color = names[hex_color.lower()]
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", hex_color):
        raise ValueError(f"Formato de cor inválido: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def generate_chessboard_image(
    square_size: int = 80,
    board_size: int = 8,
    light_color: Union[str, Tuple[int, int, int]] = "#f0d9b5",
    dark_color: Union[str, Tuple[int, int, int]] = "#b58863",
    labels: bool = False,
) -> np.ndarray:
    if isinstance(light_color, str):
        light = _hex_to_bgr(light_color)
    else:
        light = tuple(int(x) for x in light_color)
    if isinstance(dark_color, str):
        dark = _hex_to_bgr(dark_color)
    else:
        dark = tuple(int(x) for x in dark_color)

    border = int(square_size * 0.6) if labels else 0
    img_size = board_size * square_size + 2 * border
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    origin = border
    for r in range(board_size):
        for c in range(board_size):
            top = origin + r * square_size
            left = origin + c * square_size
            bottom = top + square_size
            right = left + square_size
            color = light if (r + c) % 2 == 0 else dark
            cv2.rectangle(img, (left, top), (right, bottom), color, thickness=-1)

    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, square_size / 120.0)
        thickness = max(1, int(square_size / 40))
        text_color = (0, 0, 0)
        files = [chr(ord("a") + i) for i in range(board_size)]
        ranks = [str(board_size - i) for i in range(board_size)]

        for i, f in enumerate(files):
            cx = origin + i * square_size + square_size // 2
            tx = cx
            ty = origin + board_size * square_size + int(border * 0.75)
            (w, h), _ = cv2.getTextSize(f, font, scale, thickness)
            cv2.putText(
                img,
                f,
                (tx - w // 2, ty + h // 2),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        for i, rlabel in enumerate(ranks):
            cy = origin + i * square_size + square_size // 2
            tx = int(border * 0.25)
            ty = cy + int((scale * 10) / 2)
            (w, h), _ = cv2.getTextSize(rlabel, font, scale, thickness)
            cv2.putText(
                img,
                rlabel,
                (tx, ty + h // 2),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    return img


__all__ = ["generate_chessboard_image"]
