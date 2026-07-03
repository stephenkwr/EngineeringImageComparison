import cv2, numpy as np
from typing import List, Tuple, Optional, Dict

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

# ---------- helpers ----------
def _binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return 255 - bin_img  # invert: lines become white

def _vh_masks(bin_inv: np.ndarray, v_ratio=30, h_ratio=30):
    H, W = bin_inv.shape[:2]
    klen_v = max(10, H // v_ratio)
    klen_h = max(10, W // h_ratio)
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen_v))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (klen_h, 1))
    vmask = cv2.dilate(cv2.erode(bin_inv, vker, 1), vker, 1)
    hmask = cv2.dilate(cv2.erode(bin_inv, hker, 1), hker, 1)
    return vmask, hmask

def _line_positions(mask: np.ndarray, axis=0, min_len_ratio=0.6, min_gap=4) -> List[int]:
    H, W = mask.shape
    profile = mask.sum(axis=0 if axis == 0 else 1)
    thr = 0.50 * 255 * (H if axis == 0 else W)
    hits = np.where(profile >= thr)[0]
    lines: List[int] = []
    if hits.size:
        start = prev = hits[0]
        for i in hits[1:]:
            if i == prev + 1:
                prev = i
            else:
                lines.append((start + prev) // 2)
                start = prev = i
        lines.append((start + prev) // 2)

    filtered: List[int] = []
    for x in lines:
        col = mask[:, x] if axis == 0 else mask[x, :]
        coverage = (col > 0).sum() / (H if axis == 0 else W)
        if coverage >= min_len_ratio and (not filtered or abs(x - filtered[-1]) >= min_gap):
            filtered.append(int(x))
    return filtered

def _build_cells(img_shape, vmask, hmask):
    H, W = img_shape[:2]
    xs = _line_positions(vmask, axis=0, min_len_ratio=0.65, min_gap=3)
    ys = _line_positions(hmask, axis=1, min_len_ratio=0.65, min_gap=3)
    if not xs: xs = [0, W - 1]
    if not ys: ys = [0, H - 1]
    if xs[0] > 2: xs = [0] + xs
    if xs[-1] < W - 3: xs = xs + [W - 1]
    if ys[0] > 2: ys = [0] + ys
    if ys[-1] < H - 3: ys = ys + [H - 1]
    xs, ys = sorted(set(xs)), sorted(set(ys))

    cells: List[List[Optional[BBox]]] = []
    for r in range(len(ys) - 1):
        y0, y1 = ys[r], ys[r + 1]
        row: List[Optional[BBox]] = []
        for c in range(len(xs) - 1):
            x0, x1 = xs[c], xs[c + 1]
            pad = 2
            x0i, y0i = max(0, x0 + pad), max(0, y0 + pad)
            x1i, y1i = min(W - 1, x1 - pad), min(H - 1, y1 - pad)
            w, h = x1i - x0i, y1i - y0i
            row.append((x0i, y0i, w, h) if (w > 8 and h > 8) else None)
        cells.append(row)
    return xs, ys, cells

def _crop_by_frac(img: np.ndarray, frac: Tuple[float, float, float, float]):
    # frac = (t, r, b, l) in [0..1] of image height/width (kept compatible with your config)
    t, r, b, l = frac
    H, W = img.shape[:2]
    x0 = int(round(l * W)); y0 = int(round(t * H))
    x1 = int(round(W * (1 - r))); y1 = int(round(H * (1 - b)))
    x0 = max(0, min(x0, W - 1)); y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W)); y1 = max(y0 + 1, min(y1, H))
    return img[y0:y1, x0:x1], (x0, y0)

# ---------- public API: single table (legacy) ----------
def detect_table_and_cells(img_path: str,
                           v_ratio: int = 30,
                           h_ratio: int = 30,
                           min_table_area_ratio: float = 0.10,
                           pre_scale: float = 1.0,
                           search_crop_frac=(0.0, 0.0, 0.0, 0.0)):
    """
    Returns (ABSOLUTE coords of the ORIGINAL image):
      table_bbox: (x,y,w,h)
      xs_abs, ys_abs: grid line positions
      cells_abs: 2D list of (x,y,w,h) or None
      debug: masks at detection scale (for inspection)
    """
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise FileNotFoundError(img_path)

    # optional margin crop (t,r,b,l)
    if any(search_crop_frac):
        imgC, (x_off, y_off) = _crop_by_frac(img0, search_crop_frac)
    else:
        imgC = img0
        x_off = y_off = 0

    # upsample for detection ONLY
    scale = max(0.1, float(pre_scale))
    imgS = cv2.resize(imgC, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else imgC

    bin_inv = _binarize(imgS)
    vmask_full, hmask_full = _vh_masks(bin_inv, v_ratio, h_ratio)
    grid = cv2.bitwise_or(vmask_full, hmask_full)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Hs, Ws = grid.shape[:2]
    table_bboxS: Optional[BBox] = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) / float(Ws * Hs) >= min_table_area_ratio:
            pad = int(3 * scale)
            x = max(0, x - pad); y = max(0, y - pad)
            w = min(Ws - x, w + 2 * pad); h = min(Hs - y, h + 2 * pad)
            table_bboxS = (x, y, w, h)
    if table_bboxS is None:
        table_bboxS = (0, 0, Ws, Hs)

    xS, yS, wS, hS = table_bboxS
    roiS = imgS[yS:yS + hS, xS:xS + wS]

    bin_inv_roi = _binarize(roiS)
    vmask_roi, hmask_roi = _vh_masks(bin_inv_roi, v_ratio, h_ratio)
    xsS, ysS, cellsS = _build_cells(roiS.shape, vmask_roi, hmask_roi)

    inv = 1.0 / scale
    def us(v): return int(round(v * inv))
    table_bbox = (x_off + us(xS), y_off + us(yS), us(wS), us(hS))
    xs_abs = [x_off + us(xS + x) for x in xsS]
    ys_abs = [y_off + us(yS + y) for y in ysS]

    cells_abs: List[List[Optional[BBox]]] = []
    for row in cellsS:
        out = []
        for rect in row:
            if rect is None:
                out.append(None)
            else:
                x, y, w, h = rect
                out.append((x_off + us(xS + x), y_off + us(yS + y), us(w), us(h)))
        cells_abs.append(out)

    debug = {"vmask_full": vmask_full, "hmask_full": hmask_full,
             "grid_full": grid, "vmask_roi": vmask_roi, "hmask_roi": hmask_roi}
    return table_bbox, xs_abs, ys_abs, cells_abs, debug

# ---------- NEW: detect ALL tables ----------
def detect_all_tables(img_path: str,
                      v_ratio: int = 30,
                      h_ratio: int = 30,
                      min_table_area_ratio: float = 0.03,
                      pre_scale: float = 1.0,
                      search_crop_frac=(0.0, 0.0, 0.0, 0.0)) -> List[Dict]:
    """
    Return a list of tables on the page.
    Each item: { 'bbox': BBox, 'xs': List[int], 'ys': List[int], 'cells': List[List[BBox]] }
    All coordinates in ORIGINAL image space.
    """
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise FileNotFoundError(img_path)

    # optional margin crop (t,r,b,l)
    if any(search_crop_frac):
        imgC, (x_off, y_off) = _crop_by_frac(img0, search_crop_frac)
    else:
        imgC = img0
        x_off = y_off = 0

    # detect at upscale
    scale = max(0.1, float(pre_scale))
    imgS = cv2.resize(imgC, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else imgC

    bin_inv = _binarize(imgS)
    vmask_full, hmask_full = _vh_masks(bin_inv, v_ratio, h_ratio)
    grid = cv2.bitwise_or(vmask_full, hmask_full)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Hs, Ws = grid.shape[:2]
    tables: List[Dict] = []
    inv = 1.0 / scale

    if not cnts:
        return tables

    # collect all sufficiently large contours as tables
    for c in cnts:
        xS, yS, wS, hS = cv2.boundingRect(c)
        if (wS * hS) / float(Ws * Hs) < min_table_area_ratio:
            continue
        pad = int(3 * scale)
        xS = max(0, xS - pad); yS = max(0, yS - pad)
        wS = min(Ws - xS, wS + 2 * pad); hS = min(Hs - yS, hS + 2 * pad)

        roiS = imgS[yS:yS + hS, xS:xS + wS]
        bin_inv_roi = _binarize(roiS)
        vmask_roi, hmask_roi = _vh_masks(bin_inv_roi, v_ratio, h_ratio)
        xsS, ysS, cellsS = _build_cells(roiS.shape, vmask_roi, hmask_roi)

        def us(v): return int(round(v * inv))
        bbox_abs = (x_off + us(xS), y_off + us(yS), us(wS), us(hS))
        cells_abs: List[List[Optional[BBox]]] = []
        for row in cellsS:
            out = []
            for rect in row:
                if rect is None:
                    out.append(None)
                else:
                    x, y, w, h = rect
                    out.append((x_off + us(xS + x), y_off + us(yS + y), us(w), us(h)))
            cells_abs.append(out)

        tables.append({"bbox": bbox_abs, "xs": [], "ys": [], "cells": cells_abs})

    # sort reading order: top-to-bottom, left-to-right
    tables.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    return tables

# ---------- overlays ----------
def draw_overlay(img_path: str,
                 xs_abs, ys_abs, cells_abs,
                 table_bbox=None,
                 out_path: str = "table_cells_overlay.png",
                 grid_color=(255, 255, 0),
                 cell_color=(0, 0, 255),
                 bbox_color=(0, 255, 0),
                 grid_thickness=4,
                 cell_thickness=3,
                 bbox_thickness=5,
                 alpha=0.65,
                 annotate=False):
    img = cv2.imread(str(img_path))
    if img is None: raise FileNotFoundError(img_path)
    canvas = img.copy()
    H, W = img.shape[:2]
    for x in xs_abs:
        cv2.line(canvas, (x, 0), (x, H - 1), grid_color, grid_thickness, cv2.LINE_AA)
    for y in ys_abs:
        cv2.line(canvas, (0, y), (W - 1, y), grid_color, grid_thickness, cv2.LINE_AA)
    for r, row in enumerate(cells_abs):
        for c, rect in enumerate(row):
            if not rect: continue
            x, y, w, h = rect
            cv2.rectangle(canvas, (x, y), (x + w, y + h), cell_color, cell_thickness, cv2.LINE_AA)
            if annotate:
                cv2.putText(canvas, f"{r},{c}", (x + 3, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{r},{c}", (x + 3, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if table_bbox:
        x, y, w, h = table_bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bbox_color, bbox_thickness, cv2.LINE_AA)
    out = cv2.addWeighted(img, 1.0, canvas, alpha, 0)
    cv2.imwrite(out_path, out)

def draw_overlay_multi(img_path: str, tables: List[Dict], out_path: str, annotate: bool = True) -> str:
    img = cv2.imread(str(img_path))
    if img is None: raise FileNotFoundError(img_path)
    vis = img.copy()
    for i, t in enumerate(tables, 1):
        x, y, w, h = t["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3, cv2.LINE_AA)
        if annotate:
            cv2.putText(vis, f"T{i}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        for row in t["cells"]:
            for rc in row:
                if not rc: continue
                cx, cy, cw, ch = rc
                cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (0, 165, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    return out_path
