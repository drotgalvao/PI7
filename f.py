import json
import os
from typing import Dict, List

import cv2
import numpy as np

from chessimg2pos import predict_fen
from generate_chessboard import generate_chessboard_image


CODE_TO_TEMPLATE = {
    "wK": "white_king",
    "wQ": "white_queen",
    "wR": "white_rook",
    "wB": "white_bishop",
    "wN": "white_knight",
    "wP": "white_pawn",
    "bK": "black_king",
    "bQ": "black_queen",
    "bR": "black_rook",
    "bB": "black_bishop",
    "bN": "black_knight",
    "bP": "black_pawn",
}


def fen_to_piece_list(fen: str) -> List[Dict[str, str]]:
    cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
    fen_map = {
        "p": "bP",
        "r": "bR",
        "n": "bN",
        "b": "bB",
        "q": "bQ",
        "k": "bK",
        "P": "wP",
        "R": "wR",
        "N": "wN",
        "B": "wB",
        "Q": "wQ",
        "K": "wK",
    }

    pieces: List[Dict[str, str]] = []
    ranks = fen.split("/")
    rank_num = 8
    for rank in ranks:
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                square = cols[file_idx] + str(rank_num)
                code = fen_map.get(char)
                if code:
                    pieces.append({"square": square, "piece": code})
                file_idx += 1
        rank_num -= 1
    return pieces


def _overlay_alpha(background: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    h, w = sprite.shape[:2]
    if sprite.shape[2] == 4:
        alpha = sprite[:, :, 3:4] / 255.0
        fg_rgb = sprite[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((h, w, 1), dtype=np.float32)
        fg_rgb = sprite.astype(np.float32)

    roi = background[y : y + h, x : x + w].astype(np.float32)
    blended = alpha * fg_rgb + (1.0 - alpha) * roi
    background[y : y + h, x : x + w] = blended.astype(np.uint8)


def _load_template(
    piece_code: str, cache: Dict[str, np.ndarray], crops_dir: str
) -> np.ndarray:
    if piece_code in cache:
        return cache[piece_code]

    tpl_name = CODE_TO_TEMPLATE.get(piece_code)
    if tpl_name is None:
        raise ValueError(f"Código de peça desconhecido: {piece_code}")

    path = os.path.join(crops_dir, f"{tpl_name}.png")
    sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if sprite is None:
        raise FileNotFoundError(f"Sprite não encontrado: {path}")

    cache[piece_code] = sprite
    return sprite


def render_board(
    pieces: List[Dict[str, str]],
    output_png: str,
    square_size: int = 112,
    labels: bool = True,
    crops_dir: str = "crops",
) -> None:
    board_img = generate_chessboard_image(
        square_size=square_size, board_size=8, labels=labels
    )
    border = int(square_size * 0.6) if labels else 0
    cache: Dict[str, np.ndarray] = {}

    for entry in pieces:
        square = entry["square"].lower()
        piece_code = entry["piece"]
        if len(square) != 2:
            continue
        file_idx = ord(square[0]) - ord("a")
        rank_idx = 8 - int(square[1])
        if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
            continue

        sprite = _load_template(piece_code, cache, crops_dir)
        max_dim = int(square_size * 0.82)
        h_s, w_s = sprite.shape[:2]
        scale = max_dim / max(h_s, w_s)
        new_w = max(1, int(w_s * scale))
        new_h = max(1, int(h_s * scale))
        sprite_resized = cv2.resize(
            sprite, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        x0 = border + file_idx * square_size + (square_size - new_w) // 2
        y0 = border + rank_idx * square_size + (square_size - new_h) // 2
        _overlay_alpha(board_img, sprite_resized, x0, y0)

    cv2.imwrite(output_png, board_img)


def main() -> None:
    fen = predict_fen("tabuleiro.png")
    pieces = fen_to_piece_list(fen)
    output = {"pieces": pieces}

    json_path = "fen_position.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    render_board(pieces, "fen_reconstructed.png")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nJSON salvo em {json_path}")
    print("PNG salvo em fen_reconstructed.png")


if __name__ == "__main__":
    main()
