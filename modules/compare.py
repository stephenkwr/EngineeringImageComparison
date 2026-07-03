# modules/compare.py
import os, re
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np

from modules.detect import detect_table_and_cells, detect_all_tables
from modules.align import align_new_to_old, ecc_refine_affine
from modules.figure_detect_cv import detect_figures_boxes
from modules import config as cfg

BBox = Tuple[int, int, int, int]

# ------------------------------ helpers --------------------------------------
def _roi(img: np.ndarray, rect: BBox) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]

def _inner(rect: BBox, inset: int, W: int, H: int) -> Optional[BBox]:
    x, y, w, h = rect
    x0 = max(0, x + inset); y0 = max(0, y + inset)
    x1 = min(W, x + w - inset); y1 = min(H, y + h - inset)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)

def _bin_pair(a_bgr: np.ndarray, b_bgr: np.ndarray):
    ga = cv2.GaussianBlur(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    ta, _ = cv2.threshold(ga, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tb, _ = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = int((ta + tb) // 2)
    _, ba = cv2.threshold(ga, t, 255, cv2.THRESH_BINARY_INV)
    _, bb = cv2.threshold(gb, t, 255, cv2.THRESH_BINARY_INV)
    return ba, bb, ga, gb

def _erode1(img_bin: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.erode(img_bin, k)

def _overlap_slices(h: int, w: int, dx: int, dy: int):
    x0 = max(0, dx); y0 = max(0, dy)
    x1 = min(w, w + dx); y1 = min(h, h + dy)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None, None, 0
    aslice = (slice(y0, y1), slice(x0, x1))
    bslice = (slice(y0 - dy, y1 - dy), slice(x0 - dx, x1 - dx))
    return aslice, bslice, (y1 - y0) * (x1 - x0)

def _xor_ratio(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    m = cv2.bitwise_xor(a_bin, b_bin)
    return float((m > 0).mean())

def _best_shift(a_bin: np.ndarray, b_bin: np.ndarray, max_shift: int = 2):
    h, w = a_bin.shape
    best = 1.0
    best_dx = 0; best_dy = 0
    best_as = None; best_bs = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            aslice, bslice, area = _overlap_slices(h, w, dx, dy)
            if area <= 1:
                continue
            r = _xor_ratio(a_bin[aslice], b_bin[bslice])
            if r < best:
                best = r; best_dx = dx; best_dy = dy
                best_as = aslice; best_bs = bslice
    return best, best_dx, best_dy, best_as, best_bs

def _max_blob_ratio(mask: np.ndarray) -> float:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nb <= 1:
        return 0.0
    max_area = stats[1:, cv2.CC_STAT_AREA].max()
    return max_area / float(mask.shape[0] * mask.shape[1])

def _edge_touch_ratio(mask: np.ndarray, border: int = 2) -> float:
    h, w = mask.shape
    edge = np.zeros_like(mask, dtype=np.uint8)
    edge[:border, :] = 1; edge[-border:, :] = 1
    edge[:, :border] = 1; edge[:, -border:] = 1
    edge_hits = np.logical_and(mask > 0, edge > 0).sum()
    total = (mask > 0).sum()
    if total == 0: return 0.0
    return edge_hits / float(total)

# ------------------------------ (optional) OCR --------------------------------
_OCR_AVAILABLE = False
try:
    if bool(getattr(cfg, "OCR_ENABLE", False)):
        import pytesseract
        from pytesseract import Output
        _OCR_AVAILABLE = True
        if getattr(cfg, "TESSERACT_CMD", None):
            pytesseract.pytesseract.tesseract_cmd = str(cfg.TESSERACT_CMD)
except Exception:
    _OCR_AVAILABLE = False

_OCR_ENABLED   = bool(getattr(cfg, "OCR_ENABLE", False)) and _OCR_AVAILABLE
_OCR_CONF_MIN  = float(getattr(cfg, "OCR_CONF_MIN", 0.88))
_OCR_NUM_TOL   = float(getattr(cfg, "OCR_NUM_TOL", 0.0))
_OCR_WHITELIST = getattr(cfg, "OCR_WHITELIST",
                         "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-+/%()[]:")

def _prep_for_ocr(img_bgr: np.ndarray, target_h: int = 64) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h < 3 or w < 3:
        _, b_small = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b_small
    if h != target_h:
        new_w = max(1, int(round(w * target_h / float(h))))
        g = cv2.resize(g, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b

def _norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-").replace("，", ",").replace("．", ".")
    return s

def _ocr_text(img_bgr: np.ndarray, psm: int = 7):
    if not _OCR_ENABLED:
        return "", 0.0
    h, w = img_bgr.shape[:2]
    if h < 8 or w < 8:
        return "", 0.0
    b = _prep_for_ocr(img_bgr)
    wl = _OCR_WHITELIST.replace('"', '').replace("'", "")
    config = f'--psm {psm} -c tessedit_char_whitelist={wl}'
    try:
        from pytesseract import Output
        data = pytesseract.image_to_data(b, output_type=Output.DICT, config=config)
    except Exception:
        return "", 0.0
    words, confs = [], []
    for t, c in zip(data.get("text", []), data.get("conf", [])):
        try: c = float(c)
        except Exception: c = -1
        if t and c >= 0:
            words.append(t); confs.append(c/100.0)
    text = _norm_text(" ".join(words))
    conf = (sum(confs)/len(confs)) if confs else 0.0
    return text, conf

def _numbers_equal(a: str, b: str, tol: float = 0.0) -> bool:
    try:
        av = float(re.findall(r"-?\d+(?:\.\d+)?", a)[0])
        bv = float(re.findall(r"-?\d+(?:\.\d+)?", b)[0])
        return abs(av - bv) <= tol
    except Exception:
        return False

def _cells_equal_by_ocr(a_roi: np.ndarray, b_roi: np.ndarray,
                        conf_min: float = _OCR_CONF_MIN, num_tol: float = _OCR_NUM_TOL):
    if not _OCR_ENABLED:
        return None
    ta, ca = _ocr_text(a_roi)
    tb, cb = _ocr_text(b_roi)
    if ca < conf_min or cb < conf_min:
        return None
    if ta.lower() == tb.lower():
        return True
    if _numbers_equal(ta, tb, tol=num_tol):
        return True
    return False

# ---------------- drift-robust decision (HYBRID, tables) ----------------------
def _decide_change(a_roi: np.ndarray, b_roi: np.ndarray, abs_t: int, ratio_t: float) -> bool:
    ab, bb, ga_full, gb_full = _bin_pair(a_roi, b_roi)
    ab = _erode1(ab); bb = _erode1(bb)

    max_shift = int(getattr(cfg, "MICRO_SHIFT_MAX", 2))
    best_ratio, dx, dy, aslice, bslice = _best_shift(ab, bb, max_shift=max_shift)
    if aslice is None:
        ocr_same = _cells_equal_by_ocr(a_roi, b_roi)
        return (ocr_same is False)

    ab_sub = ab[aslice]; bb_sub = bb[bslice]
    diff_mask = cv2.bitwise_xor(ab_sub, bb_sub)

    ga_sub = ga_full[aslice]; gb_sub = gb_full[bslice]
    if abs_t > 0:
        gd = cv2.absdiff(ga_sub, gb_sub)
        _, m_abs = cv2.threshold(gd, abs_t, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.bitwise_and(diff_mask, m_abs)

    b_ignore = int(getattr(cfg, "BORDER_IGNORE_PX", 0))
    if b_ignore > 0 and min(diff_mask.shape) > 2 * b_ignore:
        diff_mask[:b_ignore, :] = 0; diff_mask[-b_ignore:, :] = 0
        diff_mask[:, :b_ignore] = 0; diff_mask[:, -b_ignore:] = 0

    area    = int(diff_mask.size)
    diff_px = int((diff_mask > 0).sum())
    ratio   = diff_px / float(area) if area > 0 else 0.0

    if ratio > ratio_t and _max_blob_ratio(diff_mask) < 0.003:
        ratio = 0.0
    if ratio > ratio_t and _edge_touch_ratio(diff_mask, border=2) > 0.65:
        ratio = 0.0

    min_px   = int(getattr(cfg, "CHANGE_MIN_PIX", 0))
    min_blob = int(getattr(cfg, "CHANGE_MIN_BLOB", 0))

    largest_blob_ok = True
    if min_blob > 0:
        nb, _, stats, _ = cv2.connectedComponentsWithStats(
            (diff_mask > 0).astype(np.uint8), connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if nb > 1 else 0
        largest_blob_ok = (largest >= min_blob)

    changed_px = (ratio > ratio_t) and (diff_px >= min_px) and largest_blob_ok

    ocr_same = _cells_equal_by_ocr(a_roi, b_roi)
    if changed_px:
        if ocr_same is True:
            return False
        return True
    else:
        if ocr_same is False:
            return True
        return False

def _fill_on_new_poly(new_orig_bgr: np.ndarray, rect_old: BBox, H_inv,
                      color=(0, 0, 255), alpha: float = 0.35) -> None:
    if H_inv is None:
        return
    x, y, w, h = rect_old
    pts = np.float32([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]).reshape(-1, 1, 2)
    pts2 = cv2.perspectiveTransform(pts, H_inv).astype(np.int32)
    overlay = new_orig_bgr.copy()
    cv2.fillPoly(overlay, [pts2.reshape(-1, 2)], color)
    cv2.addWeighted(overlay, alpha, new_orig_bgr, 1 - alpha, 0, dst=new_orig_bgr)

def _warp_mask_to_new(mask_old: np.ndarray, H_inv, new_shape_wh: Tuple[int,int]) -> np.ndarray:
    """Warp a binary mask (old image space) onto NEW image space using H_inv."""
    if H_inv is None:
        # no homography — assume sizes match and use direct paste into same coords
        H_old, W_old = mask_old.shape[:2]
        Wn, Hn = new_shape_wh
        if (W_old, H_old) != (Wn, Hn):
            mask_old = cv2.resize(mask_old, (Wn, Hn), interpolation=cv2.INTER_NEAREST)
        return mask_old
    Wn, Hn = new_shape_wh
    return cv2.warpPerspective(mask_old, H_inv, (Wn, Hn), flags=cv2.INTER_NEAREST)

def _paint_mask_red(canvas_bgr: np.ndarray, mask_new_space: np.ndarray) -> None:
    """Directly set all mask>0 pixels to red on the canvas (no alpha)."""
    m = (mask_new_space > 0)
    canvas_bgr[m] = (0, 0, 255)

# ------------------------------ single & multi (tables) -----------------------
def _union_bbox(boxes: List[BBox]) -> BBox:
    xs = [x for (x, y, w, h) in boxes] + [x + w for (x, y, w, h) in boxes]
    ys = [y for (x, y, w, h) in boxes] + [y + h for (x, y, w, h) in boxes]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1 - x0, y1 - y0)

def compare_multi(old_path: str, new_path: str, out_path: str = "diff_on_new_multi.png",
                  pre_scale: float = getattr(cfg, "PRE_SCALE", 2.0),
                  search_crop_frac = getattr(cfg, "SEARCH_CROP_FRAC", (0.03, 0.02, 0.03, 0.02)),
                  inset: int = getattr(cfg, "INSET", 12),
                  debug_dir: Optional[str] = None):
    """
    (Tables only) — retained for compatibility with your previous main.py
    """
    old_bgr = cv2.imread(old_path, cv2.IMREAD_COLOR)
    new_bgr = cv2.imread(new_path, cv2.IMREAD_COLOR)
    if old_bgr is None or new_bgr is None:
        raise FileNotFoundError("Could not read one or both images.")

    tables_old: List[Dict] = detect_all_tables(
        old_path, pre_scale=pre_scale, search_crop_frac=search_crop_frac,
        min_table_area_ratio=0.02
    )
    if not tables_old:
        raise RuntimeError("No tables detected on OLD.")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # align once globally
    new_aligned, H, H_inv, ok = align_new_to_old(old_bgr, new_bgr)
    # refine using union of table boxes
    union = _union_bbox([t["bbox"] for t in tables_old])
    new_aligned, _, _ = ecc_refine_affine(old_bgr, new_aligned, union)

    if (not ok) or (H is None) or (H_inv is None):
        new_aligned = cv2.resize(new_bgr, (old_bgr.shape[1], old_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        H_inv = None
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "new_aligned_full.png"), new_aligned)

    canvas = new_bgr.copy()
    total_cells = total_flags = 0
    Hh, Ww = old_bgr.shape[:2]

    # global sanity over union
    og = cv2.cvtColor(_roi(old_bgr, union), cv2.COLOR_BGR2GRAY)
    ng = cv2.cvtColor(_roi(new_aligned, union), cv2.COLOR_BGR2GRAY)
    diff_ratio = (cv2.absdiff(og, ng) > 8).mean()
    hard_mode = diff_ratio < 0.02

    for t in tables_old:
        cells_old = t["cells"]
        for row in cells_old:
            for rect in row:
                if not rect: continue
                inner = _inner(rect, inset, Ww, Hh)
                if inner is None: continue
                a = _roi(old_bgr, inner)
                b = _roi(new_aligned, inner)
                if a.size == 0 or b.size == 0: continue

                abs_t   = cfg.ABSDIFF_THRESH if hard_mode else max(cfg.ABSDIFF_THRESH, 10)
                ratio_t = cfg.CHANGED_RATIO   if hard_mode else max(cfg.CHANGED_RATIO, 0.02)

                total_cells += 1
                if _decide_change(a, b, abs_t, ratio_t):
                    _fill_on_new_poly(canvas, rect, H_inv, color=(0,0,255), alpha=0.35)
                    total_flags += 1

    cv2.imwrite(out_path, canvas)
    return out_path, total_cells, total_flags, len(tables_old)

# ------------------------------ NEW: full pipeline ----------------------------
def compare_tables_and_figures(old_path: str, new_path: str, out_path: str = "diff_all_on_new.png",
                               pre_scale: float = getattr(cfg, "PRE_SCALE", 2.0),
                               search_crop_frac = getattr(cfg, "SEARCH_CROP_FRAC", (0.03, 0.02, 0.03, 0.02)),
                               inset: int = getattr(cfg, "INSET", 12),
                               debug_dir: Optional[str] = None):
    """
    Pipeline as requested:
    1) Align NEW to OLD (global homography, then ECC on union of tables if any).
    2) canvas = copy of ORIGINAL NEW (we paint red here).
    3) Parse base components: tables (detect_all_tables) and figures (detect_figures_boxes).
    4) Classify tables vs figures (figures exclude table regions by IoU).
    5) Tables: run table cell comparison; highlight changed cells on canvas (red).
    6) Figures: pixel-to-pixel comparison inside each figure bbox, build per-pixel mask of diffs,
       warp that mask to NEW space, set those pixels to pure red.
    7) Save canvas and return counts.
    """
    old_bgr = cv2.imread(old_path, cv2.IMREAD_COLOR)
    new_bgr = cv2.imread(new_path, cv2.IMREAD_COLOR)
    if old_bgr is None or new_bgr is None:
        raise FileNotFoundError("Could not read one or both images.")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # --- detect tables on OLD (for alignment refine + table stage)
    tables_old: List[Dict] = detect_all_tables(
        old_path, pre_scale=pre_scale, search_crop_frac=search_crop_frac,
        min_table_area_ratio=float(getattr(cfg, "TABLE_MIN_AREA_RATIO", 0.02))
    )

    # --- align NEW->OLD
    new_aligned, H, H_inv, ok = align_new_to_old(old_bgr, new_bgr)

    # refine on union of tables if we have any; else refine on whole image bbox
    if tables_old:
        union = _union_bbox([t["bbox"] for t in tables_old])
    else:
        Hh, Ww = old_bgr.shape[:2]; union = (0, 0, Ww, Hh)
    new_aligned, _, _ = ecc_refine_affine(old_bgr, new_aligned, union)

    if (not ok) or (H is None) or (H_inv is None):
        new_aligned = cv2.resize(new_bgr, (old_bgr.shape[1], old_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        H_inv = None

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "aligned_full.png"), new_aligned)

    # --- 2) output canvas
    canvas = new_bgr.copy()

    # --- 3&4) parse components
    table_boxes = [t["bbox"] for t in tables_old] if tables_old else []
    figure_boxes = detect_figures_boxes(old_path, exclude_boxes=table_boxes,
                                        save_debug_dir=(os.path.join(debug_dir, "figures") if debug_dir else None))

    # --- 5) TABLES: compare like before
    total_cells = total_cell_flags = 0
    if tables_old:
        # decide hard/easy mode on union region
        og = cv2.cvtColor(_roi(old_bgr, union), cv2.COLOR_BGR2GRAY)
        ng = cv2.cvtColor(_roi(new_aligned, union), cv2.COLOR_BGR2GRAY)
        diff_ratio = (cv2.absdiff(og, ng) > 8).mean()
        hard_mode = diff_ratio < 0.02

        Hh, Ww = old_bgr.shape[:2]
        for t in tables_old:
            for row in t["cells"]:
                for rect in row:
                    if not rect: continue
                    inner = _inner(rect, inset, Ww, Hh)
                    if inner is None: continue
                    a = _roi(old_bgr, inner)
                    b = _roi(new_aligned, inner)
                    if a.size == 0 or b.size == 0: continue

                    abs_t   = cfg.ABSDIFF_THRESH if hard_mode else max(cfg.ABSDIFF_THRESH, 10)
                    ratio_t = cfg.CHANGED_RATIO   if hard_mode else max(cfg.CHANGED_RATIO, 0.02)

                    total_cells += 1
                    if _decide_change(a, b, abs_t, ratio_t):
                        # highlight cell region in red (semi-transparent fill)
                        _fill_on_new_poly(canvas, rect, H_inv, color=(0,0,255), alpha=0.35)
                        total_cell_flags += 1

    # --- 6) FIGURES: pixel-to-pixel, every differing pixel -> red
    # --- 6) FIGURES: drift-robust pixel comparison (micro-shift + edge/bin gates)
    total_figs = len(figure_boxes)
    total_fig_diff_px = 0

    # thresholds (tune in config)
    fig_abs_t   = int(getattr(cfg, "FIG_ABSDIFF_THRESH", 0) or getattr(cfg, "ABSDIFF_THRESH", 0))
    micro_shift = int(getattr(cfg, "MICRO_SHIFT_MAX", 2))
    min_blob    = int(getattr(cfg, "CHANGE_MIN_BLOB", 30))

    def _prep_gray_bin(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        # Otsu + invert: lines -> 1s
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thin cleanup
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        b = cv2.erode(b, k, 1)
        return g, b

    for fbox in figure_boxes:
        x, y, w, h = fbox
        if w <= 1 or h <= 1: 
            continue
        a_roi = _roi(old_bgr, fbox)
        b_roi = _roi(new_aligned, fbox)
        if a_roi.shape != b_roi.shape or a_roi.size == 0:
            continue

        ga, ba = _prep_gray_bin(a_roi)
        gb, bb = _prep_gray_bin(b_roi)

        # find best small integer shift on binarized maps (very fast)
        best_r, dx, dy, aslice, bslice = _best_shift(ba, bb, max_shift=micro_shift)

        if aslice is None:
            # fallback: strict absdiff with a higher threshold to avoid AA noise
            gd = cv2.absdiff(ga, gb)
            thr = max(fig_abs_t, 12)  # 8–16 is typical for drawings
            _, mask_local = cv2.threshold(gd, thr, 255, cv2.THRESH_BINARY)
            # small component suppression
            nb, _, stats, _ = cv2.connectedComponentsWithStats(mask_local, connectivity=8)
            if nb > 1 and min_blob > 0:
                keep = np.zeros_like(mask_local)
                for i in range(1, nb):
                    if stats[i, cv2.CC_STAT_AREA] >= min_blob:
                        x0 = stats[i, cv2.CC_STAT_LEFT]
                        y0 = stats[i, cv2.CC_STAT_TOP]
                        ww = stats[i, cv2.CC_STAT_WIDTH]
                        hh = stats[i, cv2.CC_STAT_HEIGHT]
                        keep[y0:y0+hh, x0:x0+ww] = np.maximum(
                            keep[y0:y0+hh, x0:x0+ww], mask_local[y0:y0+hh, x0:x0+ww])
                mask_local = keep
            # place into full OLD-space mask in this box
            mask_full = np.zeros(old_bgr.shape[:2], dtype=np.uint8)
            mask_full[y:y+h, x:x+w] = mask_local
        else:
            # aligned overlap only
            ba_sub = ba[aslice]; bb_sub = bb[bslice]
            # XOR on binarized edges
            diff = cv2.bitwise_xor(ba_sub, bb_sub)

            # gate by absdiff on grayscale to kill halo/AA
            gd = cv2.absdiff(ga[aslice], gb[bslice])
            thr = max(fig_abs_t, 12)
            _, gate = cv2.threshold(gd, thr, 255, cv2.THRESH_BINARY)
            diff = cv2.bitwise_and(diff, gate)

            # remove tiny specks
            if min_blob > 0:
                nb, labels, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
                if nb > 1:
                    keep = np.zeros_like(diff)
                    for i in range(1, nb):
                        if stats[i, cv2.CC_STAT_AREA] >= min_blob:
                            keep[labels == i] = 255
                    diff = keep

            # place into full OLD-space mask with the found shift
            mask_full = np.zeros(old_bgr.shape[:2], dtype=np.uint8)
            ys, ye = aslice[0].start, aslice[0].stop
            xs, xe = aslice[1].start, aslice[1].stop
            mask_full[y+ys:y+ye, x+xs:x+xe] = diff

        # warp OLD-space mask to NEW and paint red
        mask_new = _warp_mask_to_new(mask_full, H_inv, (new_bgr.shape[1], new_bgr.shape[0]))
        total_fig_diff_px += int((mask_new > 0).sum())
        _paint_mask_red(canvas, mask_new)
    # --- 7) write result
    cv2.imwrite(out_path, canvas)
    return {
        "out_path": out_path,
        "tables_detected": len(table_boxes),
        "table_cells_compared": total_cells,
        "table_cells_flagged": total_cell_flags,
        "figures_detected": total_figs,
        "figure_diff_pixels": total_fig_diff_px
    }
