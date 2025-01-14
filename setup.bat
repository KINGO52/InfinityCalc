@echo off
echo Setting up InfinityCalc...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; version=sys.version_info; exit(0 if version.major>=3 and version.minor>=8 else 1)" >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.8 or higher is required!
    pause
    exit /b 1
)

:: Remove existing virtual environment if it exists
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment!
    pause
    exit /b 1
)

:: Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

:: Install each dependency separately to better handle errors
echo Installing numpy...
pip install numpy>=1.24.0
echo Installing matplotlib...
pip install matplotlib>=3.7.0
echo Installing sympy...
pip install sympy>=1.12
echo Installing scipy...
pip install scipy>=1.11.0
echo Installing customtkinter...
pip install customtkinter>=5.2.0
echo Installing mpmath...
pip install mpmath>=1.3.0

:: Check if installation was successful
python -c "import numpy, matplotlib, sympy, scipy, customtkinter, mpmath" >nul 2>&1
if errorlevel 1 (
    echo Error: Failed to install one or more dependencies!
    pause
    exit /b 1
)

echo Setup completed successfully!
echo You can now run the calculator using start.bat
pause 