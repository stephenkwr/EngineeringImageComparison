# Engineering Image Comparison

Engineering Image Comparison is a Python/OpenCV tool for comparing two revisions
of an engineering drawing and generating a visual diff image. It is designed for
mechanical, isometric, schematic, and drawing-sheet workflows where parts,
annotations, balloons, tables, and figures may shift between revisions.

Instead of doing a simple pixel-by-pixel subtraction, the project tries to
distinguish between content that merely moved and content that genuinely changed.
The output highlights additions, removals, table cell changes, figure changes,
and optional color changes in a review-friendly overlay.

## What It Does

- Lets a user choose an old drawing and a new drawing through a Tkinter GUI.
- Provides a command-line entry point for comparing files directly.
- Aligns the new drawing against the old drawing before comparison.
- Detects table regions and compares their cells with drift-tolerant logic.
- Detects non-table figure regions using multiscale OpenCV edge detection.
- Uses translation-tolerant block matching to ignore parts, text blocks, balloons,
  and leaders that moved but did not actually change.
- Produces a color-coded output image:
  - Green: added content in the new drawing.
  - Red: removed content from the old drawing.
  - Orange boxes: grouped change regions.
  - Magenta: optional color-only changes in general/color mode.
- Opens or previews the generated diff image after comparison.

## Main Use Cases

- Reviewing revision changes between two engineering drawing exports.
- Checking whether drawing notes, dimensions, labels, or table cells changed.
- Comparing isometric assembly drawings where components are repositioned between
  revisions.
- Finding added or removed features in line-art drawings without manually
  inspecting the entire sheet.
- Comparing drawing sheets that contain both tabular revision information and
  figure/part diagrams.
- Producing a visual review artifact that can be shared with teammates or used
  during design-check workflows.
- Running quick local checks before releasing or submitting updated drawing files.

## Why It Is More Than A Pixel Diff

Engineering drawings often change layout between revisions even when the actual
content is unchanged. A simple XOR or absolute pixel difference can produce large
false positives when:

- the whole drawing is slightly shifted,
- individual parts move independently,
- text blocks or balloons are repositioned,
- thin lines are misregistered by a few pixels,
- scan/export noise changes the anti-aliasing,
- tables and drawing figures require different comparison strategies.

This project handles those cases with multiple comparison paths:

- `modules/diff_blockmatch.py` performs translation-tolerant block matching. It
  searches nearby regions for matching content so moved-but-unchanged elements
  can be suppressed.
- `modules/compare.py` combines table detection, figure detection, alignment, and
  per-region comparison into a fuller drawing-sheet pipeline.
- `modules/align.py` uses ORB feature matching, homography estimation, and ECC
  refinement to align the new drawing to the old drawing.
- `modules/detect.py` detects table grids and cells from line structure.
- `modules/figure_detect_cv.py` finds non-table figure regions with multiscale
  edge detection and connected components.
- `modules/diff_engine.py` contains an alternate elastic-alignment raster diff
  engine for drift-robust drawing comparisons.

## Interfaces

### GUI

The main GUI is in `gui_app.py`. It provides:

- old/new drawing file pickers,
- output file selection,
- drawing type options,
- numeric tuning controls for movement tolerance and minimum change size,
- a similarity pre-check that warns when two images appear unrelated,
- a preview of the generated diff,
- a button to open the full-size result.

The batch file `Compare Drawings.bat` is intended as a simple Windows launcher.

### Command Line

`compare_drawings.py` can compare two drawings from the command line:

```powershell
python compare_drawings.py old.png new.png -o diff_result.png
```

Useful options include:

- `--search`: maximum movement, in pixels, that can still count as unchanged.
- `--tol`: residual shift tolerance.
- `--min-area`: minimum change blob size.
- `--keep-leaders`: keeps balloon leader-line differences instead of suppressing
  them.
- `--mode classic`: optimized for black-and-white line art.
- `--mode general`: supports colored or non-classic drawings.
- `--color`: flags color-only changes in magenta when using general mode.
- `--debug`: writes intermediate masks for inspection.

## Supported Inputs

The tool is built around raster drawing files, including:

- PNG,
- JPG/JPEG,
- TIFF/TIF.

It is intended for exported drawing images rather than native CAD files.

## Output

The generated output is an image overlay that makes differences easy to inspect.
Depending on the entry point and mode, the output may include:

- a full-resolution diff image,
- a preview image for large outputs,
- added/removed masks when debug mode is enabled,
- visual boxes around grouped change regions.

## Dependencies

The source uses:

- Python,
- OpenCV,
- NumPy,
- Pillow,
- Tkinter,
- optional Tesseract OCR support through `pytesseract`.

OCR-related code exists for table/text comparison, but OCR is disabled by default
in `modules/config.py`.

## Project Structure

```text
compare_drawings.py       Command-line comparison entry point
gui_app.py                Full Tkinter GUI application
main.py                   Earlier GUI-driven table/figure comparison entry point
modules/align.py          Drawing alignment helpers
modules/compare.py        Combined table and figure comparison pipeline
modules/config.py         Thresholds and tuning parameters
modules/detect.py         Table and cell detection
modules/diff_blockmatch.py Translation-tolerant drawing diff engine
modules/diff_engine.py    Alternate elastic-alignment raster diff engine
modules/figure_detect_cv.py Figure region detection
modules/GUI.py            Simple file-picker GUI helper
```

## Notes

The repository intentionally ignores generated build outputs, Python caches, test
images, and local virtual environments. Large drawing samples should stay outside
Git history unless they are handled through a storage strategy such as Git LFS.

