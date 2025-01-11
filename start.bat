@echo off
echo Starting InfinityCalc...

if not exist venv (
    echo Error: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python src/main.py
pause 