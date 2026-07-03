#!/usr/bin/env python3
"""
Drawing comparison entry point.

Engine: modules/diff_blockmatch.py  (translation-tolerant block matching).
Handles engineering / isometric drawings where parts, balloons and text blocks
move INDEPENDENTLY by tens-to-hundreds of px between revisions. Such moves are
treated as "unchanged, relocated"; only genuine add/remove/content changes are
flagged (green = added in NEW, red = removed from OLD).

Usage:
    python3 compare_drawings.py OLD.png NEW.png [-o out.png] [--debug]
    python3 compare_drawings.py            # Tk file-picker fallback

The bundled .venv interpreter path is stale (built on another machine), so run
with a working Python 3.13 that can see the venv packages:

    PYTHONPATH=".venv/Lib/site-packages" python3 compare_drawings.py old.png new.png
"""
import argparse
import os
import sys

# make the bundled .venv packages importable regardless of interpreter
_HERE = os.path.dirname(os.path.abspath(__file__))
_SP = os.path.join(_HERE, ".venv", "Lib", "site-packages")
if os.path.isdir(_SP) and _SP not in sys.path:
    sys.path.insert(0, _SP)
sys.path.insert(0, _HERE)

import cv2  # noqa: E402
from modules import diff_blockmatch as bm  # noqa: E402


def _write_preview(path: str, max_w: int = 2000) -> str:
    im = cv2.imread(path)
    if im is None:
        return ""
    h, w = im.shape[:2]
    if w <= max_w:
        return path
    s = max_w / w
    small = cv2.resize(im, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)
    prev = path.rsplit(".", 1)[0] + "_preview.png"
    cv2.imwrite(prev, small)
    return prev


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two engineering drawings.")
    ap.add_argument("old", nargs="?", help="OLD image (.tif/.png/.jpg)")
    ap.add_argument("new", nargs="?", help="NEW image (.tif/.png/.jpg)")
    ap.add_argument("-o", "--out", default="diff_result.png", help="output overlay path")
    ap.add_argument("--search", type=int, default=640,
                    help="max px a part may move and still count as 'moved, not changed'")
    ap.add_argument("--tol", type=int, default=3, help="residual-shift tolerance (px)")
    ap.add_argument("--min-area", type=int, default=60, help="drop change blobs smaller than this")
    ap.add_argument("--keep-leaders", action="store_true",
                    help="do NOT suppress balloon leader lines")
    ap.add_argument("--mode", choices=["classic", "general"], default="classic",
                    help="classic = B&W line art (default); general = any/colored drawing")
    ap.add_argument("--color", action="store_true",
                    help="(general mode) also flag color changes in magenta")
    ap.add_argument("--debug", action="store_true", help="also write added/removed masks")
    args = ap.parse_args()

    old_path, new_path = args.old, args.new
    if not old_path or not new_path:
        from modules.GUI import GUI_For_User
        old_path, new_path = GUI_For_User()
    if not old_path or not new_path:
        print("Need both OLD and NEW images.")
        sys.exit(1)

    res = bm.compare(
        old_path, new_path, out_path=args.out,
        search=args.search, tol=args.tol, min_area=args.min_area,
        suppress_leaders=not args.keep_leaders, mode=args.mode,
        detect_color=args.color, debug=args.debug,
    )

    print("Added   (green, in NEW not OLD): ", res["added_px"], "px")
    print("Removed (red,   in OLD not NEW): ", res["removed_px"], "px")
    print("Change regions:", res["change_regions"])
    if res.get("color_regions"):
        print("Color changes (magenta):", res["color_px"], "px in", res["color_regions"], "regions")
    print("Full overlay:", res["out_path"])
    prev = _write_preview(res["out_path"])
    if prev and prev != res["out_path"]:
        print("Preview     :", prev)

    try:
        os.startfile(os.path.abspath(prev or res["out_path"]))  # type: ignore[attr-defined]
    except Exception:
        pass


if __name__ == "__main__":
    main()
