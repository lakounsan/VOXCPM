@echo off
chcp 65001 >nul
echo Starting Studio0808 VoxCPM (Khmer Version / ភាសាខ្មែរ)...
python Studio0808_VoxCPM.py --lang km
if %errorlevel% neq 0 (
    pause
)
