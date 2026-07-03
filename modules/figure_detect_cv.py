# modules/figure_detect_cv.py
import os, cv2, numpy as np
from typing import List, Tuple, Optional
from modules import config as cfg

BBox = Tuple[int, int, int, int]  # (x,y,w,h)

def _load_u8_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _clip(x, lo, hi): return max(lo, min(int(round(x)), hi))

def _nms_xyxy(boxes, scores, iou_thr=0.5):
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32); s = np.array(scores, dtype=np.float32)
    x1,y1,x2,y2 = b.T
    areas = (x2-x1+1)*(y2-y1+1)
    order = s.argsort()[::-1]; keep=[]
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1); h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

def _boxes_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    if aa <= 0 or bb <= 0: return 0.0
    return inter / float(aa + bb - inter + 1e-6)

def _xyxy_to_xywh(b):
    x1,y1,x2,y2 = b
    return (int(round(x1)), int(round(y1)), int(round(x2-x1)), int(round(y2-y1)))

def detect_figures_boxes(
    image_path: str,
    exclude_boxes: Optional[List[BBox]] = None,
    iou_exclude: float = 0.3,
    save_debug_dir: Optional[str] = None
) -> List[BBox]:
    """
    Return a list of non-table 'figure' boxes in ORIGINAL coords (x,y,w,h).
    Uses multiscale edge+dilate CV. Optionally excludes regions overlapping
    provided boxes (e.g., tables) by IoU.
    """
    bgr0 = _load_u8_bgr(image_path)
    H0, W0 = bgr0.shape[:2]

    scales        = getattr(cfg, "FIG_MS_SCALES", (1.25, 1.75, 2.25))
    canny         = getattr(cfg, "FIG_CANNY", (60, 160))
    dilate_inner  = int(getattr(cfg, "FIG_DILATE_INNER", 12))
    dilate_merge  = int(getattr(cfg, "FIG_DILATE_MERGE", 24))
    min_area_frac = float(getattr(cfg, "FIG_MIN_AREA_FRAC", 3e-5))
    pad_frac      = float(getattr(cfg, "FIG_PAD_FRAC", 0.03))

    all_xyxy, all_scores = [], []

    for scale in scales:
        if scale == 1.0:
            bgr = bgr0
        else:
            bgr = cv2.resize(bgr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny[0], canny[1])

        k1 = max(1, int(round(dilate_inner * scale)))
        k2 = max(1, int(round(dilate_merge * scale)))
        ker1 = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
        ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))

        thick = cv2.dilate(edges, ker1, 1)
        mask  = cv2.dilate(thick, ker2, 1)

        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        min_area_pix = int(min_area_frac * W0 * H0)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            area_orig = area / (scale*scale)
            if area_orig < min_area_pix:
                continue
            ar = w / max(1.0, h)
            if ar > 100 or ar < 0.01:
                continue
            x1 = x/scale; y1 = y/scale; x2 = (x+w)/scale; y2 = (y+h)/scale
            all_xyxy.append([x1, y1, x2, y2])
            all_scores.append(float(scale) * float(area_orig))

    keep = _nms_xyxy(all_xyxy, all_scores, iou_thr=0.5)
    xyxy = [all_xyxy[i] for i in keep]

    # pad + clip
    out_xyxy = []
    for (x1,y1,x2,y2) in xyxy:
        w = x2-x1; h = y2-y1
        px = pad_frac * w; py = pad_frac * h
        x1 = _clip(x1 - px, 0, W0-1); y1 = _clip(y1 - py, 0, H0-1)
        x2 = _clip(x2 + px, 0, W0-1); y2 = _clip(y2 + py, 0, H0-1)
        if x2 > x1 and y2 > y1:
            out_xyxy.append([x1,y1,x2,y2])

    # exclusion vs provided boxes (e.g., tables)
    if exclude_boxes:
        excl_xyxy = []
        for (x,y,w,h) in exclude_boxes:
            excl_xyxy.append([x, y, x+w, y+h])
        filtered = []
        for b in out_xyxy:
            if all(_boxes_iou_xyxy(b, e) <= iou_exclude for e in excl_xyxy):
                filtered.append(b)
        out_xyxy = filtered

    # save debug if requested
    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        vis = bgr0.copy()
        for (x1,y1,x2,y2) in out_xyxy:
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 3)
        cv2.imwrite(os.path.join(save_debug_dir, "figures_detected.png"), vis)

    # convert to (x,y,w,h)
    boxes_xywh = [_xyxy_to_xywh(b) for b in out_xyxy]
    return boxes_xywh

# # (old demo main kept for ad-hoc testing; not used by pipeline)
# if __name__ == "__main__":
#     IMG = r"C:\path\to\image.tif"
#     OUT = os.path.join(os.path.dirname(IMG), "cv_figures_ms")
#     os.makedirs(OUT, exist_ok=True)
#     boxes = detect_figures_boxes(IMG, save_debug_dir=OUT)
#     print("Detected", len(boxes), "figure boxes")
