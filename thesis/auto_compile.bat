@echo off
echo Starting auto-compilation monitor...
echo This script will automatically compile your thesis and clean up temporary files.
powershell -ExecutionPolicy Bypass -File auto_compile.ps1
pause
