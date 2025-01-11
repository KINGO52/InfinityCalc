@echo off
echo Setting up Advanced Calculator...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Python 3.8 or higher is required!
    pause
    exit /b 1
)

:: Remove existing venv if it exists
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

:: Install each dependency separately to handle potential issues
echo Installing NumPy...
pip install numpy==1.21.0
echo Installing Matplotlib...
pip install matplotlib==3.4.0
echo Installing SymPy...
pip install sympy==1.8
echo Installing SciPy...
pip install scipy==1.7.0
echo Installing CustomTkinter...
pip install customtkinter==5.1.2

:: Check if installation was successful
python -c "import numpy, matplotlib, sympy, scipy, customtkinter" >nul 2>&1
if errorlevel 1 (
    echo Error: Some dependencies failed to install correctly.
    echo Please try running setup.bat again or install dependencies manually.
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo You can now run the calculator using run.bat
echo.
pause 