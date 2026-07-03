# modules/config.py
PRE_SCALE = 3.0
SEARCH_CROP_FRAC = (0.03, 0.02, 0.03, 0.02)
INSET = 16

# Pixel diff thresholds (tables)
ABSDIFF_THRESH = 4
CHANGED_RATIO  = 0.00

# ----- OCR options (optional; the code auto-disables OCR if not installed) -----
OCR_ENABLE   = False
OCR_CONF_MIN = 0.88
OCR_NUM_TOL  = 0.0
OCR_WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-+/%()[]:"

# If Windows can't find Tesseract, set the full path here (or leave None if PATH is set)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BORDER_IGNORE_PX = 1      # ignore this many pixels around each cell ROI
CHANGE_MIN_PIX   = 20      # absolute changed pixels required to flag
CHANGE_MIN_BLOB  = 20      # largest connected XOR blob must be at least this big
MICRO_SHIFT_MAX  = 5       # try ±N-pixel integer shifts to cancel tiny jitter

# --- Parsing / component detection knobs ---
TABLE_MIN_AREA_RATIO = 0.02    # fraction of (W*H) at detection scale to keep as a table
TABLE_GRID_MIN_FRAC  = 0.012   # (kept for compatibility; not used in current detector)

# Figure (non-table) detector — multiscale CV (no new files)
FIG_MS_SCALES        = (1.25, 1.75, 2.25)   # multiscale upsample factors
FIG_CANNY            = (60, 160)            # Canny thresholds
FIG_DILATE_INNER     = 12                   # inner dilate (scaled per level)
FIG_DILATE_MERGE     = 24                   # merge dilate (scaled per level)
FIG_MIN_AREA_FRAC    = 3e-5                 # in ORIGINAL image fraction
FIG_PAD_FRAC         = 0.03                 # padding applied to final figure boxes

# Figure comparison thresholds (pixel-to-pixel)
# None -> fall back to ABSDIFF_THRESH; ratio isn't used for figures (we mark every differing pixel)
FIG_ABSDIFF_THRESH   = 12
FIG_CHANGED_RATIO    = None  # unused for figures; kept for symmetry
