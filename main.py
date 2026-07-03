# main.py
from modules.GUI import GUI_For_User
from modules.compare import compare_tables_and_figures
from modules import config as cfg
from datetime import datetime
import time, os, platform, subprocess

def open_with_default_viewer(path: str) -> None:
    """Open a file with the OS default app (Win/macOS/Linux)."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"[warn] Output not found: {path}")
        return
    try:
        if platform.system() == "Windows":
            os.startfile(path)                 # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"[warn] Could not open '{path}': {e}")

if __name__ == "__main__":
    start = time.perf_counter()
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")

    old_img_path, new_img_path = GUI_For_User()

    result = compare_tables_and_figures(
        old_img_path, new_img_path,
        out_path="diff_on_new.png",
        pre_scale=cfg.PRE_SCALE,
        search_crop_frac=cfg.SEARCH_CROP_FRAC,
        inset=cfg.INSET
        # debug_dir="debug_all"
    )

    print(
        f"Tables: {result['tables_detected']} | "
        f"Cells Compared: {result['table_cells_compared']} | "
        f"Cells Flagged: {result['table_cells_flagged']} | "
        f"Figures: {result['figures_detected']} | "
        f"Figure diff px: {result['figure_diff_pixels']} -> {result['out_path']}"
    )
    print(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Elapsed: {time.perf_counter()-start:.2f}s")

    # Auto-open the output image
    open_with_default_viewer(result["out_path"])

    input("Press ENTER to exit…")
