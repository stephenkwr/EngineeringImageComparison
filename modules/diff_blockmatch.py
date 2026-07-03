# modules/diff_blockmatch.py
"""
Translation-tolerant drawing diff for engineering / isometric drawings.

Why this engine (vs modules/diff_engine.py)
--------------------------------------------
Measurement on real Suzuki Y17 assembly revisions showed that individual parts,
balloons and text blocks translate INDEPENDENTLY by 40-380 px between revisions
(NCC 0.93-0.99 -> identical content, just relocated). A single smooth
displacement field cannot warp identical parts by +200 and -200 px in different
directions, so diff_engine.py drowned the output in edge-doubling.

This engine treats the diff as "does matching content exist NEARBY in the other
drawing?" in three tiers:

  Tier 1  Block matching (coarse-to-fine).
          Tile each image; for every inked block find its best-matching offset in
          the other image within a large search radius (covers big, per-part moves).
          -> a per-block offset field that may vary sharply between neighbouring parts.

  Tier 2  Per-block tolerant explain (distance/dilation test).
          Align each block by its own offset; ink that has a counterpart within
          `tol` px is "explained" (moved, not changed). Overlapping blocks vote,
          so a pixel explained by ANY covering offset is cleared.
          -> bulk of moved parts goes black.

  Tier 3  Component verification.
          Each surviving flagged blob is template-matched in the other drawing
          within a radius. Found -> it merely moved -> suppress. Not found ->
          genuine ADD / REMOVE / CHANGE -> keep.
          -> kills independently-moved balloons / leaders / screws; keeps real
             edits (added text, swapped logo, changed scale digit).

Output: ADDED (green, in NEW not OLD) + REMOVED (red, in OLD not NEW), speck-cleaned,
clustered into orange callout boxes. CPU-only, offline, deterministic.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

BBox = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# loading / binarize
# --------------------------------------------------------------------------- #
def load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        from PIL import Image
        g = np.array(Image.open(path).convert("L"))
    return g


def ink_mask(gray: np.ndarray, block: int = 51, C: int = 10) -> np.ndarray:
    """Ink -> 255. Adaptive so uneven scan tone doesn't flip strokes."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block | 1, C,
    )


# --------------------------------------------------------------------------- #
# fast "are these the same drawing?" pre-check
# --------------------------------------------------------------------------- #
def similarity(old_path: str, new_path: str) -> float:
    """
    Fast, size-agnostic score in [0,1]: ~1.0 = same drawing, near 0 = unrelated.
    Loads both at 1/8 resolution (near-instant), aligns by a coarse global shift,
    and measures bidirectional dilated ink overlap. Used to warn the user before
    running the (slow) full compare on images that aren't two revisions of the
    same drawing.
    """
    o = cv2.imread(old_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    n = cv2.imread(new_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if o is None:
        g = load_gray(old_path); o = cv2.resize(g, (max(1, g.shape[1] // 8), max(1, g.shape[0] // 8)))
    if n is None:
        g = load_gray(new_path); n = cv2.resize(g, (max(1, g.shape[1] // 8), max(1, g.shape[0] // 8)))
    n = cv2.resize(n, (o.shape[1], o.shape[0]))

    _, oi = cv2.threshold(o, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, ni = cv2.threshold(n, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    o_tot, n_tot = int(oi.sum()), int(ni.sum())
    if o_tot == 0 or n_tot == 0:
        return 0.0

    try:
        (dx, dy), _ = cv2.phaseCorrelate(o.astype(np.float32), n.astype(np.float32))
    except cv2.error:
        dx = dy = 0.0
    idx, idy = int(round(dx)), int(round(dy))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ~2px @1/8 ≈ 16px full
    od = cv2.dilate(oi, k)
    best = 0.0
    for (sx, sy) in [(0, 0), (idx, idy), (-idx, -idy)]:
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        ns = cv2.warpAffine(ni, M, (oi.shape[1], oi.shape[0]))
        nd = cv2.dilate(ns, k)
        rec_n = cv2.bitwise_and(ns, od).sum() / float(ns.sum() + 1)
        rec_o = cv2.bitwise_and(oi, nd).sum() / float(o_tot)
        best = max(best, min(rec_n, rec_o))
    return float(best)


# --------------------------------------------------------------------------- #
# Tier 1: coarse-to-fine block offset
# --------------------------------------------------------------------------- #
def _best_offset(
    src_gray: np.ndarray, dst_gray: np.ndarray,
    src_low: np.ndarray, dst_low: np.ndarray,
    cy: int, cx: int, block: int, search: int, D: int,
    bright: bool = False, edge_thr: int = 22,
) -> Optional[Tuple[int, int, float]]:
    """
    Offset (dx, dy) s.t. dst[p + off] ~ src[p] for the block centred at (cx, cy)
    in SRC. Coarse search at 1/D resolution, then a small full-res refine.
    Returns (dx, dy, score) in FULL-res px, or None if the block is too empty.

    `bright=False` (classic): ink is dark, blank-block gate = few dark pixels.
    `bright=True` (general): fed a `struct` (ΔE) map where ink is BRIGHT, so the
    blank gate flips to "few high-energy pixels".
    """
    H, W = src_gray.shape
    Hl, Wl = src_low.shape
    b = block
    bl = max(8, b // D)
    Rl = max(4, search // D)

    ty0, tx0 = (cy - b // 2), (cx - b // 2)
    tyl, txl = ty0 // D, tx0 // D
    t = src_low[tyl:tyl + bl, txl:txl + bl]
    if t.shape != (bl, bl):
        return None
    empty = (t > edge_thr).mean() < 0.01 if bright else (t < 128).mean() < 0.01
    if empty:
        return None  # essentially blank

    sy0, sx0 = max(0, tyl - Rl), max(0, txl - Rl)
    sy1, sx1 = min(Hl, tyl + bl + Rl), min(Wl, txl + bl + Rl)
    reg = dst_low[sy0:sy1, sx0:sx1]
    if reg.shape[0] < bl or reg.shape[1] < bl:
        return None
    r = cv2.matchTemplate(reg, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(r)
    dx = (sx0 + loc[0] - txl) * D
    dy = (sy0 + loc[1] - tyl) * D

    # full-res refine within +-D around the coarse pick
    tf = src_gray[ty0:ty0 + b, tx0:tx0 + b]
    if tf.shape == (b, b):
        oy0, ox0 = max(0, ty0 + dy - D), max(0, tx0 + dx - D)
        regf = dst_gray[oy0:oy0 + b + 2 * D, ox0:ox0 + b + 2 * D]
        if regf.shape[0] >= b and regf.shape[1] >= b:
            rf = cv2.matchTemplate(regf, tf, cv2.TM_CCOEFF_NORMED)
            _, s2, _, l2 = cv2.minMaxLoc(rf)
            dx = (ox0 + l2[0]) - tx0
            dy = (oy0 + l2[1]) - ty0
            score = float(s2)
    return int(dx), int(dy), float(score)


def _changed_mask(
    src_gray: np.ndarray, dst_gray: np.ndarray,
    src_ink: np.ndarray, dst_ink: np.ndarray,
    D: int, block: int, step: int, search: int,
    tol: int, min_score: float,
) -> np.ndarray:
    """
    SRC ink that has NO counterpart in DST after per-block alignment.
    (Tiers 1 + 2.) Returns a uint8 mask in SRC coordinates.
    """
    H, W = src_gray.shape
    src_low = cv2.resize(src_gray, (W // D, H // D), interpolation=cv2.INTER_AREA)
    dst_low = cv2.resize(dst_gray, (W // D, H // D), interpolation=cv2.INTER_AREA)

    explained = np.zeros((H, W), np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    b = block

    for cy in range(b // 2, H - b // 2, step):
        for cx in range(b // 2, W - b // 2, step):
            by0, bx0 = cy - b // 2, cx - b // 2
            src_blk = src_ink[by0:by0 + b, bx0:bx0 + b]
            if src_blk.max() == 0:
                continue  # no ink to explain here
            off = _best_offset(src_gray, dst_gray, src_low, dst_low,
                               cy, cx, b, search, D)
            if off is None or off[2] < min_score:
                continue
            dx, dy, _ = off
            # aligned dst ink under this block
            dy0, dx0 = by0 + dy, bx0 + dx
            dst_blk = dst_ink[max(0, dy0):dy0 + b, max(0, dx0):dx0 + b]
            if dst_blk.shape[0] < 4 or dst_blk.shape[1] < 4:
                continue
            # pad/crop dst_blk back to (b, b) aligned to src_blk origin
            canvas = np.zeros((b, b), np.uint8)
            yy = 0 if dy0 >= 0 else -dy0
            xx = 0 if dx0 >= 0 else -dx0
            hh = min(b - yy, dst_blk.shape[0])
            ww = min(b - xx, dst_blk.shape[1])
            canvas[yy:yy + hh, xx:xx + ww] = dst_blk[:hh, :ww]
            dst_d = cv2.dilate(canvas, k)
            exp = cv2.bitwise_and(src_blk, dst_d)
            explained[by0:by0 + b, bx0:bx0 + b] = cv2.bitwise_or(
                explained[by0:by0 + b, bx0:bx0 + b], exp)

    return cv2.bitwise_and(src_ink, cv2.bitwise_not(explained))


# --------------------------------------------------------------------------- #
# Tier 3: component verification
# --------------------------------------------------------------------------- #
def _verify_components(
    mask: np.ndarray, self_gray: np.ndarray, ref_gray: np.ndarray,
    search: int, ncc_thr: float, min_area: int, pad: int,
    max_dim: int = 700, max_components: int = 6000,
) -> np.ndarray:
    """
    Keep only flagged blobs whose local appearance is NOT found elsewhere in
    ref_gray within `search` px. Suppresses content that merely moved.

    Runtime is bounded: if there are more than `max_components` flagged blobs
    (a sign the images are largely different) verification is skipped and the
    mask is kept as-is; blobs larger than `max_dim` px are kept without the
    (expensive) template search rather than driving a giant matchTemplate.
    """
    H, W = self_gray.shape
    nb, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if nb - 1 > max_components:
        return mask  # pathological: too many changes to verify economically
    keep = np.zeros_like(mask)
    for i in range(1, nb):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) > max_dim:
            keep[labels == i] = 255   # too big to verify cheaply -> keep flagged
            continue
        # template = the blob's neighbourhood (with context) from SELF image
        tx0, ty0 = max(0, x - pad), max(0, y - pad)
        tx1, ty1 = min(W, x + w + pad), min(H, y + h + pad)
        tmpl = self_gray[ty0:ty1, tx0:tx1]
        th, tw = tmpl.shape
        if th < 6 or tw < 6:
            keep[labels == i] = 255
            continue
        sy0, sx0 = max(0, ty0 - search), max(0, tx0 - search)
        sy1, sx1 = min(H, ty1 + search), min(W, tx1 + search)
        region = ref_gray[sy0:sy1, sx0:sx1]
        if region.shape[0] < th or region.shape[1] < tw:
            keep[labels == i] = 255
            continue
        r = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
        if float(r.max()) < ncc_thr:
            keep[labels == i] = 255   # not found elsewhere -> genuine change
    return keep


# --------------------------------------------------------------------------- #
# cleanup / render
# --------------------------------------------------------------------------- #
def _suppress_leaders(mask: np.ndarray, min_diag: int = 80, max_fill: float = 0.06) -> np.ndarray:
    """
    Drop long, sparse, near-linear residuals (balloon leader lines in exploded
    views). A leader fills only a few percent of its bounding box; text/logos/
    dimension values fill much more, so they survive.
    """
    nb, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = mask.copy()
    for i in range(1, nb):
        w, h, area = (stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                      stats[i, cv2.CC_STAT_AREA])
        diag = (w * w + h * h) ** 0.5
        fill = area / float(max(1, w * h))
        if diag >= min_diag and fill <= max_fill:
            out[labels == i] = 0
    return out


def _clean(mask: np.ndarray, min_area: int, close: int = 3) -> np.ndarray:
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    nb, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def cluster_boxes(mask: np.ndarray, merge_dist: int = 40) -> List[BBox]:
    if not mask.any():
        return []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_dist, merge_dist))
    grouped = cv2.dilate(mask, k)
    nb, _, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    return [(int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]))
            for i in range(1, nb)]


def render(base_gray: np.ndarray, added: np.ndarray, removed: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    out[removed > 0] = (0, 0, 255)     # red   = in OLD, gone in NEW
    out[added > 0] = (0, 180, 0)       # green = new in NEW
    for (x, y, w, h) in cluster_boxes(cv2.bitwise_or(added, removed)):
        cv2.rectangle(out, (x - 8, y - 8), (x + w + 8, y + h + 8), (0, 140, 255), 3)
    return out


# =========================================================================== #
# GENERAL MODE  (color / any-type)  — additive; the classic path above is
# untouched.  Separates the MATCH representation (`struct`, a polarity/palette-
# invariant ΔE-from-local-paper + multi-channel gradient map) from the DIFF
# representation (`fg`, a hysteresis foreground), then reuses the SAME block-
# matching core.  A color-change layer runs AFTER structural alignment.
# =========================================================================== #
def load_bgr(path: str) -> np.ndarray:
    """Read as BGR (PIL fallback for exotic TIFFs)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return bgr


def _page_background_lab(bgr: np.ndarray, ring_frac: float = 0.03):
    """Robust page-background colour (Lab) + polarity flag, from the border ring."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, W = lab.shape[:2]
    m = max(2, int(min(H, W) * ring_frac))
    ring = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                           lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    paper_L = float(bg[0]) * (100.0 / 255.0)
    return bg, {"dark_bg": paper_L < 50.0, "paper_L": paper_L}


def _reconstruct(seed: np.ndarray, mask: np.ndarray, iters: int = 64) -> np.ndarray:
    """Morphological geodesic reconstruction (hysteresis): grow seed within mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prev = cv2.bitwise_and(seed, mask)
    for _ in range(iters):
        cur = cv2.bitwise_and(cv2.dilate(prev, k), mask)
        if np.array_equal(cur, prev):
            break
        prev = cur
    return prev


def foreground_signal(bgr: np.ndarray, K: int = 41, t_hi: float = 12.0, t_lo: float = 6.0,
                      grad_w: float = 0.12, struct_div: float = 30.0):
    """
    Colour/polarity-invariant representations:
      struct : uint8 ΔE-from-LOCAL-paper (Lab) combined with a multi-channel
               (Di Zenzo) gradient; ink is BRIGHT; palette- and polarity-invariant;
               NCC-stable -> fed to the block matcher.
      fg     : hysteresis-thresholded binary foreground (ink=255) of ANY hue,
               including light strokes luminance thresholding misses -> fed to the diff.
    Reduces to standard behaviour on dark-on-light line art.
    """
    L8, a8, b8 = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    Lp = cv2.medianBlur(L8, K | 1); ap = cv2.medianBlur(a8, K | 1); bp = cv2.medianBlur(b8, K | 1)
    dL = (L8.astype(np.float32) - Lp.astype(np.float32)) * (100.0 / 255.0)
    da = a8.astype(np.float32) - ap.astype(np.float32)
    db = b8.astype(np.float32) - bp.astype(np.float32)
    dE = cv2.sqrt(dL * dL + da * da + db * db)
    g = np.zeros_like(dE)
    for ch in (L8, a8, b8):
        gx = cv2.Scharr(ch, cv2.CV_32F, 1, 0); gy = cv2.Scharr(ch, cv2.CV_32F, 0, 1)
        g = cv2.max(g, cv2.magnitude(gx, gy))
    g *= (100.0 / (float(g.max()) + 1e-6))
    dE = cv2.max(dE, grad_w * g)
    seed = (dE >= t_hi).astype(np.uint8)
    grow = (dE >= t_lo).astype(np.uint8)
    fg = _reconstruct(seed, grow) * 255
    struct = np.clip(dE * (255.0 / struct_div), 0, 255).astype(np.uint8)
    return struct, fg.astype(np.uint8)


def _changed_mask_general(struct_src, struct_dst, fg_src, fg_dst,
                          D, block, step, search, tol, min_score,
                          lab_src=None, lab_dst=None):
    """
    Like _changed_mask but matches on `struct` (bright ink) and explains on `fg`.
    If lab_src/lab_dst given, also accumulates, in SRC frame, the co-structure mask
    and the per-block-aligned DST colour (for the colour-change layer).
    Returns (changed_mask, coink_or_None, aligned_lab_or_None).
    """
    H, W = struct_src.shape
    s_low = cv2.resize(struct_src, (W // D, H // D), interpolation=cv2.INTER_AREA)
    d_low = cv2.resize(struct_dst, (W // D, H // D), interpolation=cv2.INTER_AREA)
    explained = np.zeros((H, W), np.uint8)
    want_color = lab_src is not None and lab_dst is not None
    coink = np.zeros((H, W), np.uint8) if want_color else None
    aligned = np.zeros((H, W, 3), np.float32) if want_color else None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    b = block

    for cy in range(b // 2, H - b // 2, step):
        for cx in range(b // 2, W - b // 2, step):
            by0, bx0 = cy - b // 2, cx - b // 2
            src_blk = fg_src[by0:by0 + b, bx0:bx0 + b]
            if src_blk.max() == 0:
                continue
            off = _best_offset(struct_src, struct_dst, s_low, d_low,
                               cy, cx, b, search, D, bright=True)
            if off is None or off[2] < min_score:
                continue
            dx, dy, _ = off
            dy0, dx0 = by0 + dy, bx0 + dx
            dst_blk = fg_dst[max(0, dy0):dy0 + b, max(0, dx0):dx0 + b]
            if dst_blk.shape[0] < 4 or dst_blk.shape[1] < 4:
                continue
            canvas = np.zeros((b, b), np.uint8)
            yy = 0 if dy0 >= 0 else -dy0
            xx = 0 if dx0 >= 0 else -dx0
            hh = min(b - yy, dst_blk.shape[0]); ww = min(b - xx, dst_blk.shape[1])
            canvas[yy:yy + hh, xx:xx + ww] = dst_blk[:hh, :ww]
            dst_d = cv2.dilate(canvas, k)
            exp = cv2.bitwise_and(src_blk, dst_d)
            sub = explained[by0:by0 + b, bx0:bx0 + b]
            sub[:] = cv2.bitwise_or(sub, exp)
            if want_color:
                # colour is compared only on STRICT ink-on-ink overlap so both
                # samples are real strokes (robust to 1px residual misalignment).
                exp_exact = cv2.bitwise_and(src_blk, canvas)
                em = exp_exact > 0
                if em.any():
                    lab_blk = lab_dst[max(0, dy0):dy0 + b, max(0, dx0):dx0 + b]
                    labc = np.zeros((b, b, 3), np.float32)
                    labc[yy:yy + hh, xx:xx + ww] = lab_blk[:hh, :ww]
                    aligned[by0:by0 + b, bx0:bx0 + b][em] = labc[em]
                    coink[by0:by0 + b, bx0:bx0 + b][em] = 255

    changed = cv2.bitwise_and(fg_src, cv2.bitwise_not(explained))
    return changed, coink, aligned


def _global_recolor_followers(ao, bo, an, bn, co, quant, min_bin_frac=0.02):
    """
    Detect a decorative GLOBAL palette swap and return (followers_mask, n_remapped):
      followers_mask : pixels that follow the dominant old->new (a,b) colour mapping
      n_remapped     : how many significant old-colour classes map to a DIFFERENT class
    A single recoloured element (n_remapped < 2) is a genuine change, not a theme swap.
    Deterministic joint histogram; no clustering.
    """
    q = max(1, 256 // quant)
    ko = (((ao + 128).astype(np.int32)) // q) * quant + (((bo + 128).astype(np.int32)) // q)
    kn = (((an + 128).astype(np.int32)) // q) * quant + (((bn + 128).astype(np.int32)) // q)
    nb = quant * quant
    ko = np.clip(ko, 0, nb - 1); kn = np.clip(kn, 0, nb - 1)
    idx = ko[co] * nb + kn[co]
    joint = np.bincount(idx, minlength=nb * nb).reshape(nb, nb)
    old_counts = joint.sum(axis=1)
    total = float(old_counts.sum()) + 1e-6
    dominant_new = joint.argmax(axis=1)
    n_remapped = int(sum(1 for ob in range(nb)
                         if old_counts[ob] >= min_bin_frac * total and dominant_new[ob] != ob))
    follows = np.zeros(ao.shape, bool)
    follows[co] = (kn[co] == dominant_new[ko[co]])
    return follows, n_remapped


def detect_color_changes(lab_src, lab_aligned, coink, dHue=18.0, dChroma=12.0,
                         min_chroma=14.0, min_area=60, quant=16, ignore_global=True):
    """
    Recolour detection over co-structure pixels (post-alignment).
    Deliberately CONSERVATIVE: flags only where BOTH sides are genuinely COLOURED
    (chroma above `min_chroma`) and the hue/chroma differs — i.e. a coloured element
    was recoloured. Neutral black/grey/near-white strokes are never flagged (that
    keeps line-weight/anti-alias/scan noise out). A decorative GLOBAL palette swap
    (>=2 colour classes remapped consistently) is suppressed (requirement a) while a
    LOCAL recolour survives (requirement b).
    """
    co = coink > 0   # strict ink-on-ink overlap from the matcher
    if not co.any():
        return np.zeros(coink.shape, np.uint8)
    ao = lab_src[..., 1] - 128; bo = lab_src[..., 2] - 128
    an = lab_aligned[..., 1] - 128; bn = lab_aligned[..., 2] - 128
    dab = np.sqrt((ao - an) ** 2 + (bo - bn) ** 2)
    Co = np.sqrt(ao * ao + bo * bo); Cn = np.sqrt(an * an + bn * bn)
    ho = np.degrees(np.arctan2(bo, ao)); hn = np.degrees(np.arctan2(bn, an))
    dh = np.abs((ho - hn + 180) % 360 - 180)
    both_colored = np.minimum(Co, Cn) > min_chroma
    changed = co & both_colored & ((dab > dChroma) | (dh > dHue))
    if ignore_global:
        follows, n_remapped = _global_recolor_followers(ao, bo, an, bn, co, quant)
        if n_remapped >= 2:                       # decorative palette swap -> suppress followers
            changed &= ~follows
    return _clean((changed.astype(np.uint8)) * 255, min_area)


def _render_general(struct_ref, added, removed, color=None):
    """Polarity-aware base (dark ink on white regardless of input polarity) +
    green=added, red=removed, magenta=colour-change, with callout boxes."""
    base = cv2.cvtColor(255 - struct_ref, cv2.COLOR_GRAY2BGR)
    base[removed > 0] = (0, 0, 255)
    base[added > 0] = (0, 180, 0)
    if color is not None:
        base[color > 0] = (255, 0, 255)
    for (x, y, w, h) in cluster_boxes(cv2.bitwise_or(added, removed)):
        cv2.rectangle(base, (x - 8, y - 8), (x + w + 8, y + h + 8), (0, 140, 255), 3)
    if color is not None:
        for (x, y, w, h) in cluster_boxes(color):
            cv2.rectangle(base, (x - 8, y - 8), (x + w + 8, y + h + 8), (255, 0, 255), 3)
    return base


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def compare(
    old_path: str, new_path: str, out_path: str = "diff_blockmatch.png",
    D: int = 4, block: int = 256, step: int = 128, search: int = 640,
    tol: int = 3, min_score: float = 0.45,
    verify_ncc: float = 0.6, verify_pad: int = 16,
    min_area: int = 60, suppress_leaders: bool = True,
    ignore_border_px: int = 70,
    mode: str = "classic", detect_color: bool = False,
    debug: bool = False,
) -> Dict:
    """
    mode="classic" (default): the original, validated dark-on-light line-art diff.
                              Byte-for-byte the prior behaviour — never regressed.
    mode="general":           color/polarity-invariant matching for ANY drawing type
                              (colored, blueprint, light strokes), plus optional
                              color-change detection (magenta). Reduces to the classic
                              result on plain B&W line art.
    """
    if mode == "general":
        return _compare_general(
            old_path, new_path, out_path, D, block, step, search, tol, min_score,
            verify_ncc, verify_pad, min_area, suppress_leaders, ignore_border_px,
            detect_color, debug)

    old = load_gray(old_path)
    new = load_gray(new_path)
    if new.shape != old.shape:
        new = cv2.resize(new, (old.shape[1], old.shape[0]))

    old_ink = ink_mask(old)
    new_ink = ink_mask(new)

    # Tiers 1+2
    added = _changed_mask(new, old, new_ink, old_ink, D, block, step, search, tol, min_score)
    removed = _changed_mask(old, new, old_ink, new_ink, D, block, step, search, tol, min_score)

    # Tier 3
    added = _verify_components(added, new, old, search, verify_ncc, min_area, verify_pad)
    removed = _verify_components(removed, old, new, search, verify_ncc, min_area, verify_pad)

    if suppress_leaders:
        added = _suppress_leaders(added)
        removed = _suppress_leaders(removed)

    if ignore_border_px > 0:
        b = ignore_border_px
        for m in (added, removed):
            m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0

    added = _clean(added, min_area)
    removed = _clean(removed, min_area)

    overlay = render(old, added, removed)  # draw on OLD reference frame
    cv2.imwrite(out_path, overlay)

    res = {
        "out_path": out_path,
        "added_px": int((added > 0).sum()),
        "removed_px": int((removed > 0).sum()),
        "change_regions": len(cluster_boxes(cv2.bitwise_or(added, removed))),
    }
    if debug:
        base = out_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base}_added.png", added)
        cv2.imwrite(f"{base}_removed.png", removed)
    return res


def _compare_general(old_path, new_path, out_path, D, block, step, search, tol,
                     min_score, verify_ncc, verify_pad, min_area, suppress_leaders,
                     ignore_border_px, detect_color, debug) -> Dict:
    bgr_o = load_bgr(old_path)
    bgr_n = load_bgr(new_path)
    if bgr_n.shape[:2] != bgr_o.shape[:2]:
        bgr_n = cv2.resize(bgr_n, (bgr_o.shape[1], bgr_o.shape[0]))

    struct_o, fg_o = foreground_signal(bgr_o)
    struct_n, fg_n = foreground_signal(bgr_n)
    lab_o = cv2.cvtColor(bgr_o, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_n = cv2.cvtColor(bgr_n, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Structural diff (color-invariant): match on struct, explain on fg
    added, _, _ = _changed_mask_general(struct_n, struct_o, fg_n, fg_o,
                                        D, block, step, search, tol, min_score)
    removed, coink, aligned = _changed_mask_general(struct_o, struct_n, fg_o, fg_n,
                                                    D, block, step, search, tol, min_score,
                                                    lab_src=lab_o, lab_dst=lab_n)

    added = _verify_components(added, struct_n, struct_o, search, verify_ncc, min_area, verify_pad)
    removed = _verify_components(removed, struct_o, struct_n, search, verify_ncc, min_area, verify_pad)

    if suppress_leaders:
        added = _suppress_leaders(added)
        removed = _suppress_leaders(removed)

    def _clip_border(m):
        if ignore_border_px > 0 and m is not None:
            b = ignore_border_px
            m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0

    _clip_border(added); _clip_border(removed)
    added = _clean(added, min_area)
    removed = _clean(removed, min_area)

    color = None
    if detect_color and coink is not None:
        color = detect_color_changes(lab_o, aligned, coink, min_area=min_area)
        _clip_border(color)
        # don't double-report where structure already flagged
        color = cv2.bitwise_and(color, cv2.bitwise_not(cv2.bitwise_or(added, removed)))

    overlay = _render_general(struct_o, added, removed, color)
    cv2.imwrite(out_path, overlay)

    res = {
        "out_path": out_path,
        "added_px": int((added > 0).sum()),
        "removed_px": int((removed > 0).sum()),
        "color_px": int((color > 0).sum()) if color is not None else 0,
        "change_regions": len(cluster_boxes(cv2.bitwise_or(added, removed))),
        "color_regions": len(cluster_boxes(color)) if color is not None else 0,
    }
    if debug:
        base = out_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base}_added.png", added)
        cv2.imwrite(f"{base}_removed.png", removed)
        cv2.imwrite(f"{base}_struct_old.png", struct_o)
        cv2.imwrite(f"{base}_fg_old.png", fg_o)
        if color is not None:
            cv2.imwrite(f"{base}_color.png", color)
    return res
