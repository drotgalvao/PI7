import os
import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_crop(img, box, out_dir, name="black_pawn.png", padding=8, polarity="dark"):
    x, y, ww, hh, area = box
    h, w = img.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + ww + padding)
    y1 = min(h, y + hh + padding)
    crop = img[y0:y1, x0:x1]
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, name)

    rgba = create_transparent_crop(crop, polarity=polarity)
    if rgba is not None:
        ok = cv2.imwrite(out_path, rgba)
        return ok, out_path
    else:
        ok = cv2.imwrite(out_path, crop)
        return ok, out_path


def create_transparent_crop(crop, polarity="dark"):
    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if polarity == "dark":
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if polarity == "light":
        green_eroded = cv2.erode(green_mask, kernel_small, iterations=2)
        th_nogreen = cv2.bitwise_and(th, cv2.bitwise_not(green_eroded))
    else:
        th_nogreen = cv2.bitwise_and(th, cv2.bitwise_not(green_mask))

    if cv2.countNonZero(th_nogreen) < 50:
        th_nogreen = th.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(th_nogreen, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = int(np.argmax(areas))
    mask = np.zeros_like(mask_clean)
    cv2.drawContours(mask, contours, max_idx, 255, thickness=-1)

    mask_no_green = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))
    if cv2.countNonZero(mask_no_green) > 0:
        mask = mask_no_green

    if polarity == "dark":
        try:
            edges = cv2.Canny(blur, 30, 100)
            edk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_d = cv2.dilate(edges, edk, iterations=1)

            gc_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[green_mask > 0] = cv2.GC_BGD
            gc_mask[mask > 0] = cv2.GC_PR_FGD
            gc_mask[edges_d > 0] = cv2.GC_PR_FGD
            fg_seed = cv2.erode(mask, kernel_small, iterations=3)
            gc_mask[fg_seed > 0] = cv2.GC_FGD

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                crop, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
            )
            grabcut_fgd = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype("uint8")
            if cv2.countNonZero(grabcut_fgd) > 50:
                mask = grabcut_fgd
                mask[green_mask > 0] = 0
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel_close, iterations=2
                )

                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 30, 255])
                white_mask = cv2.inRange(hsv_crop, lower_white, upper_white)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))

                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, kernel_small, iterations=1
                )
        except Exception:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    elif polarity == "light":
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        mask[green_mask > 0] = 0

        try:
            edges = cv2.Canny(blur, 50, 150)
            edk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_d = cv2.dilate(edges, edk, iterations=1)

            gc_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[green_mask > 0] = cv2.GC_BGD
            gc_mask[mask > 0] = cv2.GC_PR_FGD
            gc_mask[edges_d > 0] = cv2.GC_PR_FGD
            fg_seed = cv2.erode(mask, kernel_small, iterations=2)
            gc_mask[fg_seed > 0] = cv2.GC_FGD

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                crop, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
            )
            grabcut_fgd = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype("uint8")
            if cv2.countNonZero(grabcut_fgd) > 50:
                mask = grabcut_fgd
                mask[green_mask > 0] = 0
        except Exception:
            pass

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, dilate_k, iterations=1)

    alpha = cv2.GaussianBlur(mask_dilated, (5, 5), 0)
    alpha[green_mask > 0] = 0

    if polarity == "dark":
        is_white = np.all(crop > 170, axis=2)
        alpha[is_white] = 0

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.erode(alpha, kernel_erode, iterations=1)

    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(crop)
    rgba = cv2.merge([b, g, r, alpha])
    return rgba


def interactive_crop(img, boxes, mask, out_dir, padding=8):
    h, w = img.shape[:2]
    ensure_dir(out_dir)

    debug = img.copy()
    for i, (x, y, ww, hh, area) in enumerate(boxes):
        cv2.rectangle(debug, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
    cv2.imshow("detected (mask)", mask)
    cv2.imshow("candidates", debug)
    idx = 0
    saved = []
    cur_padding = padding

    while True:
        if idx < 0:
            idx = 0
        if idx >= len(boxes):
            print("Fim das candidatas.")
            break
        x, y, ww, hh, area = boxes[idx]
        x0 = max(0, x - cur_padding)
        y0 = max(0, y - cur_padding)
        x1 = min(w, x + ww + cur_padding)
        y1 = min(h, y + hh + cur_padding)
        crop = img[y0:y1, x0:x1]

        vis = img.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"idx={idx+1}/{len(boxes)} pad={cur_padding} area={area:.0f}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("selection", vis)
        cv2.imshow("crop", crop)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("n"), 83):  # próximo
            idx += 1
        elif key in (ord("p"), 81):  # anterior
            idx -= 1
        elif key == ord("+") or key == ord("="):
            cur_padding += 4
        elif key == ord("-"):
            cur_padding = max(0, cur_padding - 4)
        elif key == ord("s") or key == ord("S"):
            out_path = os.path.join(out_dir, f"black_pawn_{idx+1}.png")
            ok = cv2.imwrite(out_path, crop)
            if ok:
                print("Salvo:", out_path)
                saved.append(out_path)
            else:
                print("Falha ao salvar:", out_path)
            idx += 1
        elif key in (ord("q"), 27):
            break
        else:
            idx += 1

    cv2.destroyAllWindows()
    return saved
