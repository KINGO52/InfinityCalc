@echo off
echo Setting up InfinityCalc...

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Checking Python version...
python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.8 or higher is required
    pause
    exit /b 1
)

if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install numpy matplotlib sympy scipy customtkinter

echo Setup completed successfully!
echo Run start.bat to launch InfinityCalc
pause 