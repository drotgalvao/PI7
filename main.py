import argparse
from utils.image_loader import load_image
from utils.detection import detect_pieces, filter_by_rank
from utils.nms import non_max_suppression
from utils.crop_saver import save_crop, interactive_crop


def main():
    parser = argparse.ArgumentParser(
        description="Detecta e recorta peças em uma imagem de tabuleiro."
    )
    parser.add_argument(
        "image", help="Caminho para a imagem do tabuleiro (ex: board.png)"
    )
    parser.add_argument(
        "--out", "-o", default="crops", help="Diretório de saída para recortes"
    )
    parser.add_argument(
        "--gray-thresh",
        type=int,
        default=100,
        help="Limiar de cinza para detectar peças escuras (0-255)",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0005,
        help="Área mínima relativa para contornos",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.05,
        help="Área máxima relativa para contornos",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Seleciona automaticamente o melhor candidato (maior área) e salva sem interface interativa",
    )
    args = parser.parse_args()

    img = load_image(args.image)
    h, w = img.shape[:2]

    boxes, mask = detect_pieces(
        img,
        polarity="dark",
        gray_thresh=args.gray_thresh,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )

    boxes_white, mask_white = detect_pieces(
        img,
        polarity="light",
        gray_thresh=140,  # Reduzido de 155 para 140
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
    )

    if not boxes:
        print("Nenhuma peça preta detectada. Tentando com diferentes thresholds...")
        for t in (120, 140, 160):
            boxes_t, mask_t = detect_pieces(
                img,
                polarity="dark",
                gray_thresh=t,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
            )
            if boxes_t:
                print(f"Encontrado(s) {len(boxes_t)} candidato(s) com gray_thresh={t}")
                boxes = boxes_t
                mask = mask_t
                break

    if not boxes_white or len(boxes_white) < 10:
        print(
            f"Poucas peças brancas detectadas ({len(boxes_white)}). Tentando com diferentes thresholds..."
        )
        for t in (120, 135, 150):
            bw, mw = detect_pieces(
                img,
                polarity="light",
                gray_thresh=t,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
            )
            if bw and len(bw) > len(boxes_white):
                print(
                    f"Encontrado(s) {len(bw)} candidato(s) brancos com gray_thresh={t}"
                )
                boxes_white = bw
                mask_white = mw
                if len(bw) >= 16:
                    break

    before_n_black = len(boxes)
    boxes = non_max_suppression(boxes, iou_thresh=0.35)
    after_n_black = len(boxes)

    before_n_white = len(boxes_white)
    boxes_white = non_max_suppression(boxes_white, iou_thresh=0.35)
    after_n_white = len(boxes_white)

    print(
        f"Black: {before_n_black} -> {after_n_black} after NMS; White: {before_n_white} -> {after_n_white} after NMS"
    )

    pieces_by_rank = {}
    for box in boxes:
        from utils.detection import get_rank_for_y

        x, y, ww, hh, area = box
        rank = get_rank_for_y(y + hh / 2, h)
        if rank not in pieces_by_rank:
            pieces_by_rank[rank] = {"black": [], "white": []}
        pieces_by_rank[rank]["black"].append(box)

    for box in boxes_white:
        from utils.detection import get_rank_for_y

        x, y, ww, hh, area = box
        rank = get_rank_for_y(y + hh / 2, h)
        if rank not in pieces_by_rank:
            pieces_by_rank[rank] = {"black": [], "white": []}
        pieces_by_rank[rank]["white"].append(box)

    if args.auto:
        if not boxes and not boxes_white:
            print("Nenhum candidato para salvar automaticamente.")
        else:
            saved_count = 0
            saved_types = {"black": set(), "white": set()}


            piece_names = {
                0: {
                    "black": [
                        "rook",
                        "knight",
                        "bishop",
                        "queen",
                        "king",
                        "bishop",
                        "knight",
                        "rook",
                    ]
                },
                1: {"black": ["pawn"] * 8},
                6: {"white": ["pawn"] * 8},
                7: {
                    "white": [
                        "rook",
                        "bishop",
                        "king",
                        "knight",
                        "queen",
                        "bishop",
                        "knight",
                        "rook",
                    ]
                },
            }

            for rank in sorted(pieces_by_rank.keys()):
                for color in ["black", "white"]:
                    pieces = pieces_by_rank[rank][color]
                    if not pieces:
                        continue

                    pieces = sorted(pieces, key=lambda b: b[0])

                    for idx, box in enumerate(pieces):
                        if rank in piece_names and color in piece_names[rank]:
                            base_names = piece_names[rank][color]
                            if idx < len(base_names):
                                piece_name = base_names[idx]
                            else:
                                piece_name = "piece"
                        else:
                            piece_name = "piece"

                        if piece_name in saved_types[color]:
                            continue

                        filename = f"{color}_{piece_name}.png"
                        polarity = "dark" if color == "black" else "light"

                        ok, out_path = save_crop(
                            img,
                            box,
                            args.out,
                            name=filename,
                            padding=10,
                            polarity=polarity,
                        )
                        if ok:
                            saved_count += 1
                            saved_types[color].add(piece_name)
                            print(f"Salvo: {out_path}")
                        else:
                            print(f"Falha ao salvar: {out_path}")

            print(f"\nTotal de peças únicas salvas: {saved_count}")
        return

    saved = interactive_crop(img, boxes, mask, args.out)
    if saved:
        print("Recortes salvos:", saved)
    else:
        print("Nenhum recorte salvo.")


if __name__ == "__main__":
    main()
