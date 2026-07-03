@echo off
REM Double-click launcher for the Drawing Comparison GUI.
REM Uses Python 3.13 (matches the bundled .venv wheels) via the py launcher.
cd /d "%~dp0"
py -3.13 "%~dp0gui_app.py"
if errorlevel 1 (
  echo.
  echo The app exited with an error. Make sure Python 3.13 is installed
  echo ^(the "py -3.13" launcher must work^). Press any key to close.
  pause >nul
)
