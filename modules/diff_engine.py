# modules/diff_engine.py
"""
Drift-robust raster diff for engineering drawings (isometric / schematic / line art).

Why this exists
---------------
The old pipeline segmented first (tables + dilated "figure" boxes) and then XOR'd
binarized line art. That fails on real drawings because:
  * one global homography can't cancel *locally varying* micro-shifts, and
  * XOR of thin lines explodes on sub-pixel misregistration (1px shift -> a
    "double line" ghost along the whole stroke), and
  * dilation merges nearby symbols into one blob, so old/new segment differently.

This module removes hard segmentation from the critical path and instead does:

  1. load (raster)                      -> grayscale + BGR
  2. binarize ink (adaptive/Sauvola)    -> robust to scan/background differences
  3. coarse global align (ECC)          -> cancels small global translation/rotation
  4. LOCAL elastic align                -> tiled phase-correlation -> dense
     (the key step)                        displacement field -> remap NEW onto OLD.
                                           Residual misalignment becomes < ~1-2 px
                                           *everywhere*, including isometric regions.
  5. tolerant bidirectional diff         -> distance-transform test. A stroke that
     (distance transform)                   merely shifted still has a neighbor within
                                           tolerance -> NOT flagged. Genuinely
                                           added / removed / moved ink has no neighbor
                                           -> flagged. Directions are separated:
                                           ADDED (in new, not old) vs REMOVED (in old,
                                           not new).
  6. clean specks + cluster into boxes
  7. colored overlay (green = added, red = removed) + boxes for easy spotting.

CPU-only, deterministic, fully offline. No OCR / no segmentation needed to highlight.
Text/dimension *value* reporting (OCR) is layered on top separately, later.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# --------------------------------------------------------------------------- #
# 1. loading
# --------------------------------------------------------------------------- #
def load_gray_bgr(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read an image as (grayscale, BGR). Falls back to PIL for exotic TIFFs."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, bgr


# --------------------------------------------------------------------------- #
# 2. binarize ink  (ink -> 255, background -> 0)
# --------------------------------------------------------------------------- #
def ink_mask(gray: np.ndarray, block: int = 41, C: int = 10) -> np.ndarray:
    """Adaptive threshold so uneven background / scan tone doesn't flip strokes."""
    if block % 2 == 0:
        block += 1
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    b = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, C,
    )
    return b


# --------------------------------------------------------------------------- #
# 3. coarse global alignment (ECC on a downscaled copy, then scale the warp)
# --------------------------------------------------------------------------- #
def coarse_align(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    mode: int = cv2.MOTION_EUCLIDEAN,
    max_dim: int = 1000,
    iters: int = 200,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, bool]:
    """
    Estimate a global warp mapping NEW -> OLD frame. Returns (warp_2x3, ok).
    Runs ECC on a downscaled copy for speed/robustness, then rescales the warp.
    """
    H, W = old_gray.shape[:2]
    scale = min(1.0, max_dim / float(max(H, W)))
    o = cv2.resize(old_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else old_gray
    n = cv2.resize(new_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else new_gray

    o = o.astype(np.float32) / 255.0
    n = n.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(o, n, warp, mode, criteria, None, 5)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), False

    # rescale translation back to full res (rotation/scale part is scale-invariant)
    if scale < 1:
        warp[0, 2] /= scale
        warp[1, 2] /= scale
    return warp, True


def warp_affine_to_old(img: np.ndarray, warp_2x3: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = shape_hw
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    # Replicate edges instead of white-filling, so rotation corners don't create
    # fake "removed" ink at the borders.
    return cv2.warpAffine(img, warp_2x3, (W, H), flags=flags,
                          borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# 4. LOCAL elastic alignment  (the key step)
# --------------------------------------------------------------------------- #
def _hann(win: int) -> np.ndarray:
    w = np.hanning(win).astype(np.float32)
    return np.outer(w, w)


def estimate_displacement_field(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    step: int = 96,
    win: int = 160,
    max_disp: float = 25.0,
    min_ink_frac: float = 0.01,
    min_response: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tiled phase-correlation -> dense (dx, dy) field, at full image resolution.

    For each control point we phase-correlate an OLD window against the NEW window.
    The shift aligns NEW to OLD. Low-ink / low-confidence / outlier tiles are
    discarded and interpolated from trustworthy neighbours, then the field is
    smoothed (it should vary slowly across the page).
    Returns float32 map_x, map_y for cv2.remap (sampling coords into NEW).
    """
    H, W = old_gray.shape[:2]
    half = win // 2
    # Reflect-pad by half a window so control points can be centred over the
    # true image borders too (otherwise the outer margin gets an extrapolated,
    # inaccurate shift -> false slivers on long lines near the edges).
    of = cv2.copyMakeBorder(old_gray, half, half, half, half, cv2.BORDER_REFLECT).astype(np.float32)
    nf = cv2.copyMakeBorder(new_gray, half, half, half, half, cv2.BORDER_REFLECT).astype(np.float32)
    ink = (of < 128).astype(np.float32)  # dark = ink (pre-binarize proxy)

    # Regularly spaced control-point centres (padded coords) whose span maps
    # exactly onto original [0,W-1]x[0,H-1], so the coarse field resizes cleanly.
    n_x = max(2, int(np.ceil(W / step)) + 1)
    n_y = max(2, int(np.ceil(H / step)) + 1)
    xs = np.linspace(half, half + W - 1, n_x).round().astype(int)
    ys = np.linspace(half, half + H - 1, n_y).round().astype(int)
    gh, gw = len(ys), len(xs)

    DX = np.full((gh, gw), np.nan, np.float32)
    DY = np.full((gh, gw), np.nan, np.float32)
    hann = _hann(win)

    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            ow = of[y0:y1, x0:x1]
            nw = nf[y0:y1, x0:x1]
            if ow.shape != (win, win):
                continue
            if ink[y0:y1, x0:x1].mean() < min_ink_frac:
                continue  # blank tile -> no reliable shift
            try:
                (sx, sy), resp = cv2.phaseCorrelate(ow * hann, nw * hann)
            except cv2.error:
                continue
            if resp < min_response:
                continue
            if abs(sx) > max_disp or abs(sy) > max_disp:
                continue
            DX[iy, ix] = sx
            DY[iy, ix] = sy

    DX, DY = _fill_and_smooth(DX, DY)

    # upsample the coarse grid field to full resolution
    DXf = cv2.resize(DX, (W, H), interpolation=cv2.INTER_LINEAR)
    DYf = cv2.resize(DY, (W, H), interpolation=cv2.INTER_LINEAR)

    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    map_x = grid_x + DXf
    map_y = grid_y + DYf
    return map_x, map_y


def _fill_and_smooth(DX: np.ndarray, DY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fill invalid (nan) grid cells from valid neighbours, then smooth."""
    def fill(a: np.ndarray) -> np.ndarray:
        mask = np.isnan(a)
        if mask.all():
            return np.zeros_like(a)
        if not mask.any():
            return a
        # simple, dependency-free fill: seed invalid cells with the global mean,
        # then relax by iterative neighbour averaging so the field stays smooth.
        out = a.copy()
        out[mask] = np.nanmean(a)
        for _ in range(50):
            blur = cv2.blur(out, (3, 3))
            out[mask] = blur[mask]
        return out

    DXf = fill(DX)
    DYf = fill(DY)
    k = (3, 3)
    DXf = cv2.GaussianBlur(DXf, k, 0)
    DYf = cv2.GaussianBlur(DYf, k, 0)
    return DXf, DYf


def local_align(old_gray: np.ndarray, new_gray: np.ndarray, **kw) -> np.ndarray:
    """Remap NEW onto OLD using the estimated dense displacement field."""
    map_x, map_y = estimate_displacement_field(old_gray, new_gray, **kw)
    return cv2.remap(new_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# 5. tolerant bidirectional diff (distance transform)
# --------------------------------------------------------------------------- #
def tolerant_diff(
    old_ink: np.ndarray,
    new_ink: np.ndarray,
    tol_px: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ADDED   = new ink whose nearest OLD ink is farther than tol_px.
    REMOVED = old ink whose nearest NEW ink is farther than tol_px.
    A stroke that only shifted <= tol_px has a neighbour -> not flagged.
    """
    old_bin = (old_ink > 0).astype(np.uint8)
    new_bin = (new_ink > 0).astype(np.uint8)

    # distance from every pixel to the nearest OLD ink pixel
    dt_old = cv2.distanceTransform(1 - old_bin, cv2.DIST_L2, 3)
    dt_new = cv2.distanceTransform(1 - new_bin, cv2.DIST_L2, 3)

    added = ((new_bin > 0) & (dt_old > tol_px)).astype(np.uint8) * 255
    removed = ((old_bin > 0) & (dt_new > tol_px)).astype(np.uint8) * 255
    return added, removed


# --------------------------------------------------------------------------- #
# 6. clean specks + cluster into boxes
# --------------------------------------------------------------------------- #
def clean_mask(mask: np.ndarray, min_area: int = 12, close: int = 3) -> np.ndarray:
    """Drop tiny specks and close small gaps so real changes read as solid marks."""
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    nb, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def cluster_boxes(mask: np.ndarray, merge_dist: int = 25, min_area: int = 12) -> List[BBox]:
    """Group nearby changed pixels into boxes to draw callouts around them."""
    if not mask.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_dist, merge_dist))
    grouped = cv2.dilate(mask, k)
    nb, _, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    boxes: List[BBox] = []
    for i in range(1, nb):
        x, y, w, h, area = stats[i]
        # area here is of the dilated blob; keep if it contains real change
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


# --------------------------------------------------------------------------- #
# 7. overlay rendering
# --------------------------------------------------------------------------- #
def render_overlay(
    base_bgr: np.ndarray,
    added: np.ndarray,
    removed: np.ndarray,
    alpha: float = 0.9,
    draw_boxes: bool = True,
) -> np.ndarray:
    """green = added (in NEW, not OLD), red = removed (in OLD, not NEW)."""
    out = base_bgr.copy()
    out[removed > 0] = (0, 0, 255)     # red
    out[added > 0] = (0, 180, 0)       # green
    if draw_boxes:
        for (x, y, w, h) in cluster_boxes(cv2.bitwise_or(added, removed)):
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 140, 255), 2)  # orange callout
    return out


# --------------------------------------------------------------------------- #
# top-level orchestration
# --------------------------------------------------------------------------- #
def compare(
    old_path: str,
    new_path: str,
    out_path: str = "diff_engine_out.png",
    tol_px: float = 3.0,
    min_area: int = 12,
    debug: bool = False,
) -> Dict:
    """Full raster compare. Returns counts + writes a colored overlay to out_path."""
    old_gray, old_bgr = load_gray_bgr(old_path)
    new_gray, new_bgr = load_gray_bgr(new_path)

    # make NEW the same canvas size as OLD (same-layout assumption)
    if new_gray.shape != old_gray.shape:
        new_gray = cv2.resize(new_gray, (old_gray.shape[1], old_gray.shape[0]))
        new_bgr = cv2.resize(new_bgr, (old_bgr.shape[1], old_bgr.shape[0]))

    # 3. coarse global align
    warp, ok = coarse_align(old_gray, new_gray)
    new_coarse = warp_affine_to_old(new_gray, warp, old_gray.shape[:2]) if ok else new_gray

    # 4. local elastic align (residual field)
    new_aligned = local_align(old_gray, new_coarse)

    # 2. binarize both
    old_ink = ink_mask(old_gray)
    new_ink = ink_mask(new_aligned)

    # 5. tolerant diff
    added, removed = tolerant_diff(old_ink, new_ink, tol_px=tol_px)

    # 6. clean
    added = clean_mask(added, min_area=min_area)
    removed = clean_mask(removed, min_area=min_area)

    # 7. overlay (render on OLD reference frame)
    overlay = render_overlay(old_bgr, added, removed)
    cv2.imwrite(out_path, overlay)

    result = {
        "out_path": out_path,
        "added_px": int((added > 0).sum()),
        "removed_px": int((removed > 0).sum()),
        "change_boxes": len(cluster_boxes(cv2.bitwise_or(added, removed))),
        "coarse_align_ok": bool(ok),
    }
    if debug:
        base = out_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base}_old_ink.png", old_ink)
        cv2.imwrite(f"{base}_new_ink.png", new_ink)
        cv2.imwrite(f"{base}_added.png", added)
        cv2.imwrite(f"{base}_removed.png", removed)
        cv2.imwrite(f"{base}_new_aligned.png", new_aligned)
    return result
