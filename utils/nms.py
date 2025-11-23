import numpy as np


def non_max_suppression(boxes, iou_thresh=0.3):
    if not boxes:
        return []

    arr = np.array(
        [[b[0], b[1], b[0] + b[2], b[1] + b[3], b[4]] for b in boxes], dtype=float
    )
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(areas)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        rem_areas = areas[idxs[1:]]
        iou = inter / (areas[i] + rem_areas - inter + 1e-9)

        inds = np.where(iou <= iou_thresh)[0]
        idxs = idxs[inds + 1]

    kept_boxes = [boxes[i] for i in keep]
    return kept_boxes
