"""
Script para extrair a rainha branca manualmente.
A rainha branca está em uma casa branca, tornando difícil a detecção automática.
"""

import cv2
import numpy as np
from utils.crop_saver import create_transparent_crop


def extract_white_queen(img_path):
    """
    Extrai a rainha branca baseado em coordenadas conhecidas.
    Posição: rank 7 (última fileira), coluna D (4ª posição da esquerda)
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Calcular tamanho de cada casa
    square_size = w / 8.0

    # Rainha branca está em D1 (rank 7, file D = coluna 3 começando do 0)
    # Rank 7 = última fileira (y de 7/8 * h até h)
    # File D = 4ª coluna (x de 3/8 * w até 4/8 * w)

    file_d = 3  # Coluna D (0-indexed)
    rank_7 = 7  # Última fileira

    x_start = int(file_d * square_size)
    x_end = int((file_d + 1) * square_size)
    y_start = int(rank_7 * square_size)
    y_end = int((rank_7 + 1) * square_size)

    print(f"Extraindo rainha branca da região:")
    print(f"  X: {x_start} - {x_end}")
    print(f"  Y: {y_start} - {y_end}")
    print(f"  Tamanho: {x_end - x_start} x {y_end - y_start}")

    # Extrair região da casa
    square_region = img[y_start:y_end, x_start:x_end]

    # Agora precisamos encontrar a peça dentro desta casa branca
    # Usar edge detection para encontrar o contorno da peça
    gray = cv2.cvtColor(square_region, cv2.COLOR_BGR2GRAY)

    # Equalização de histograma para melhorar contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Edge detection
    edges = cv2.Canny(gray_enhanced, 30, 100)

    # Dilatar para conectar bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("Nenhum contorno encontrado! Usando região completa da casa.")
        crop_raw = square_region
    else:
        # Pegar o maior contorno (provavelmente a peça)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(largest_contour)

        # Adicionar margem de 5 pixels
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        ww = min(square_region.shape[1] - x, ww + 2 * margin)
        hh = min(square_region.shape[0] - y, hh + 2 * margin)

        print(f"  Peça encontrada em: x={x}, y={y}, w={ww}, h={hh}")
        crop_raw = square_region[y : y + hh, x : x + ww]

    # Criar crop transparente
    crop = create_transparent_crop(crop_raw, polarity="light")

    # Converter para BGRA se necessário
    if crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

    h_crop, w_crop = crop.shape[:2]

    # Criar máscara de fundo usando flood fill a partir das bordas
    # A peça tem bordas pretas, então o flood fill parará nelas
    gray_crop = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Criar máscara para flood fill
    mask = np.zeros((h_crop + 2, w_crop + 2), np.uint8)
    background_mask = np.zeros((h_crop, w_crop), np.uint8)

    # Fazer flood fill a partir de todos os pontos da borda
    # Isso marca todo o fundo branco externo, mas para nas bordas pretas da peça
    seed_points = []

    # Borda superior e inferior
    for x in range(0, w_crop, 5):
        seed_points.append((x, 0))
        seed_points.append((x, h_crop - 1))

    # Borda esquerda e direita
    for y in range(0, h_crop, 5):
        seed_points.append((0, y))
        seed_points.append((w_crop - 1, y))

    # Aplicar flood fill de cada ponto da borda
    for seed in seed_points:
        if background_mask[seed[1], seed[0]] == 0:  # Se ainda não foi marcado
            temp_mask = mask.copy()
            cv2.floodFill(
                gray_crop,
                temp_mask,
                seed,
                255,
                loDiff=30,
                upDiff=30,
                flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
            )
            # Adicionar à máscara de fundo
            background_mask = cv2.bitwise_or(background_mask, temp_mask[1:-1, 1:-1])

    # Inverter a máscara (queremos manter a peça, não o fundo)
    foreground_mask = cv2.bitwise_not(background_mask)

    # Erosão leve para remover pixels brancos residuais nas bordas
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.erode(foreground_mask, kernel_erode, iterations=1)

    # Aplicar a máscara no canal alpha
    crop[:, :, 3] = cv2.bitwise_and(crop[:, :, 3], foreground_mask)

    # Remover bordas completamente transparentes
    alpha = crop[:, :, 3]
    coords = cv2.findNonZero(alpha)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = crop[y : y + h, x : x + w]

    # Salvar
    output_path = "crops/white_queen.png"
    cv2.imwrite(output_path, crop)
    print(f"\n✓ Rainha branca salva em: {output_path}")

    return crop


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "tabuleiro.png"
    extract_white_queen(img_path)
