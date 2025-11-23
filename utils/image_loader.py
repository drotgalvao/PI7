import cv2


def load_image(path, max_width=1200):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img
