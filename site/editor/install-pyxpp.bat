@echo off
echo Installing pyauto-xpp for the editor at https://zetaone.github.io/pyauto-xpp/editor/
echo.
powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://zetaone.github.io/pyauto-xpp/editor/install.ps1 | iex"
echo.
pause
