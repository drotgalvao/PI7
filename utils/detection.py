import cv2
import numpy as np


def detect_pieces(
    img, polarity="dark", gray_thresh=100, min_area_ratio=0.0005, max_area_ratio=0.05
):
    if polarity == "dark":
        return detect_dark_contours(img, gray_thresh, min_area_ratio, max_area_ratio)
    else:
        return detect_light_contours(img, gray_thresh, min_area_ratio, max_area_ratio)


def detect_dark_contours(
    img, gray_thresh=100, min_area_ratio=0.0005, max_area_ratio=0.05
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = min_area_ratio * (w * h)
    max_area = max_area_ratio * (w * h)

    boxes = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area or area > max_area:
            continue
        boxes.append((x, y, ww, hh, area))

    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    return boxes, closed


def detect_light_contours(
    img, gray_thresh=155, min_area_ratio=0.0005, max_area_ratio=0.05
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = min_area_ratio * (w * h)
    max_area = max_area_ratio * (w * h)

    boxes = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < min_area or area > max_area:
            continue
        boxes.append((x, y, ww, hh, area))

    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    return boxes, closed


def get_rank_for_y(y, h):
    rank_height = h / 8.0
    return int(y // rank_height)


def filter_by_rank(boxes, img_height, target_rank):
    filtered = []
    for b in boxes:
        x, y, ww, hh, area = b
        rank = get_rank_for_y(y + hh / 2, img_height)
        if rank == target_rank:
            filtered.append(b)
    return filtered
