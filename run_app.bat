@echo off
setlocal enabledelayedexpansion

set VENV_DIR=venv
set PYTHON=%VENV_DIR%\Scripts\python.exe
set PIP=%VENV_DIR%\Scripts\pip.exe
set BRANCH=main

echo ==========================================
echo  DataSleuth Launcher - by Cyber Wizard 🔥
echo ==========================================

:: Step 0: Git pull if updates are available
if exist ".git" (
    echo [*] Checking for updates via Git...
    git remote update >nul 2>&1

    for /f "tokens=*" %%i in ('git rev-parse HEAD') do set LOCAL=%%i
    for /f "tokens=*" %%i in ('git rev-parse origin/%BRANCH%') do set REMOTE=%%i

    if not "!LOCAL!"=="!REMOTE!" (
        echo [*] Updates found. Pulling latest changes...
        git pull origin %BRANCH%
        if errorlevel 1 (
            echo [!] Git pull failed.
            pause
            exit /b 1
        )
    ) else (
        echo [*] Already up to date.
    )
) else (
    echo [!] Git repository not found. Skipping Git update check.
)

:: Step 1: Check and create virtual environment
if not exist %PYTHON% (
    echo [*] Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [!] Failed to create venv. Is Python installed and added to PATH?
        pause
        exit /b 1
    )
)

:: Step 2: Install or upgrade pip using python -m pip
echo [*] Upgrading pip...
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
    echo [!] pip upgrade failed.
    pause
    exit /b 1
)

:: Step 3: Install dependencies
echo [*] Installing requirements...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [!] Failed to install dependencies.
    pause
    exit /b 1
)

:: Step 4: Launch the Streamlit app
echo [*] Launching Streamlit app...
%PYTHON% -m streamlit run app.py

endlocal
pause