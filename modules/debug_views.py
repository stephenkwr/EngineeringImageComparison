# modules/debug_views.py
import cv2
from modules.detect import detect_table_and_cells, draw_overlay

def dump_detection_debug(img_path: str, out_prefix: str,
                         pre_scale: float, search_crop_frac):
    tb, xs, ys, cells, dbg = detect_table_and_cells(
        img_path, pre_scale=pre_scale, search_crop_frac=search_crop_frac
    )
    draw_overlay(img_path, xs, ys, cells, tb, out_path=f"{out_prefix}_overlay.png", annotate=True)
    for k in ["vmask_full","hmask_full","grid_full","vmask_roi","hmask_roi"]:
        cv2.imwrite(f"{out_prefix}_{k}.png", dbg[k])
    return f"{out_prefix}_overlay.png"
