@echo off
chcp 65001 >nul
echo Starting Studio0808 VoxCPM (English Version)...
python Studio0808_VoxCPM.py --lang en
if %errorlevel% neq 0 (
    pause
)
