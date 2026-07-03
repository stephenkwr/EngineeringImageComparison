@echo off
REM One-click build of the single-file executable: dist\CompareDrawings.exe
REM Requires: Python 3.13 (py -3.13) + the bundled .venv packages (incl. PyInstaller).
cd /d "%~dp0"
set PYTHONPATH=%~dp0.venv\Lib\site-packages
echo Building CompareDrawings.exe  (this takes 1-2 minutes)...
py -3.13 -m PyInstaller --clean --noconfirm compare_gui.spec
if errorlevel 1 (
  echo.
  echo BUILD FAILED. Make sure "py -3.13" works and .venv\Lib\site-packages exists.
) else (
  echo.
  echo DONE -^> dist\CompareDrawings.exe   ^(share this single file^)
)
echo Press any key to close.
pause >nul
