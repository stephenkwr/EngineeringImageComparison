# modules/align.py
import cv2, numpy as np
from typing import Tuple, Optional

def align_new_to_old(old_bgr, new_bgr):
    g0 = cv2.cvtColor(old_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape[:2]
    orb = cv2.ORB_create(5000)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 20 or len(k1) < 20:
        return cv2.resize(new_bgr, (w, h)), None, None, False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d0, d1, k=2)
    good = [m for m, n in matches if m.distance < 0.75*n.distance]
    if len(good) < 30:
        return cv2.resize(new_bgr, (w, h)), None, None, False
    src = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # old
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # new
    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(new_bgr, (w, h)), None, None, False
    warped = cv2.warpPerspective(new_bgr, H, (w, h), flags=cv2.INTER_NEAREST)
    return warped, H, np.linalg.inv(H), True

def ecc_refine_affine(old_bgr, new_aligned_bgr, table_bbox):
    """
    Refine NEW->OLD alignment using ECC on the table bbox (fast & sub-pixel).
    Returns (refined_bgr, warp_2x3 or None, improved: bool).
    """
    x, y, w, h = table_bbox
    old_g = cv2.cvtColor(old_bgr, cv2.COLOR_BGR2GRAY)
    new_g = cv2.cvtColor(new_aligned_bgr, cv2.COLOR_BGR2GRAY)

    # ROIs must be same size
    roi_old = old_g[y:y+h, x:x+w]
    roi_new = new_g[y:y+h, x:x+w]
    if roi_old.size == 0 or roi_new.size == 0:
        return new_aligned_bgr, None, False

    # ECC expects float32 in [0,1]
    A = roi_old.astype(np.float32) / 255.0
    B = roi_new.astype(np.float32) / 255.0

    warp_mode = cv2.MOTION_AFFINE
    warp = np.eye(2, 3, dtype=np.float32)  # start from identity
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)

    try:
        cc, warp = cv2.findTransformECC(A, B, warp, warp_mode, criteria, None, 5)
    except cv2.error:
        return new_aligned_bgr, None, False

    refined = cv2.warpAffine(new_aligned_bgr, warp,
                             (old_bgr.shape[1], old_bgr.shape[0]),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    return refined, warp, True
