@echo off
echo Starting Advanced Calculator...

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found! Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment and run the calculator
call venv\Scripts\activate.bat
python src/main.py

pause 