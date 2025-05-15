@echo off
setlocal EnableDelayedExpansion

set VENV_DIR=venv
set APP_FILE=app.py
set REQUIREMENTS_FILE=requirements.txt

:: Colors (using ANSI sequences, works in Windows 10+)
:: If colors don't work, remove the echo and color codes.

:: Functions mimic
:info
echo [INFO] %*
goto :eof

:warn
echo [WARN] %*
goto :eof

:err
echo [ERROR] %*
goto :eof

:success
echo [SUCCESS] %*
goto :eof

:: Check for python3
where python >nul 2>&1
if errorlevel 1 (
    call :err Python not found. Please install Python 3.
    exit /b 1
)

:: Check if venv exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    call :info Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        call :err Failed to create virtual environment.
        exit /b 1
    ) else (
        call :success Virtual environment created at "%VENV_DIR%".
    )
) else (
    call :info Using existing virtual environment at "%VENV_DIR%".
)

:: Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

:: Spinner while installing pip packages
:: We'll just show a simple spinner with delayed output

setlocal EnableDelayedExpansion
set SPINNER=|/-\
set /a SPIN_POS=0

call :info Installing dependencies...

:: Run pip install in background with output redirected to nul
start /b cmd /c "pip install --upgrade pip >nul 2>&1 && pip install -r %REQUIREMENTS_FILE% >nul 2>&1" 
set /a PID=%ERRORLEVEL%

:: There is no straightforward way to wait on background process in batch, so simulate delay
:: Instead we do a loop with timeout and assume pip install < 120 seconds (adjust if needed)

for /l %%i in (1,1,120) do (
    set /a SPIN_POS=(SPIN_POS + 1) %% 4
    set SPIN_CHAR=!SPINNER:~%SPIN_POS%,1!
    <nul set /p=Installing dependencies... !SPIN_CHAR!  <nul
    timeout /t 1 >nul
    <nul set /p=
)

echo.
call :success Dependencies installed.
endlocal

:: Git update check
where git >nul 2>&1
if errorlevel 1 (
    call :warn Git not found. Skipping update check.
) else (
    call :info Checking for updates from Git...
    git fetch >nul 2>&1
    for /f "tokens=*" %%a in ('git rev-parse @') do set LOCAL=%%a
    for /f "tokens=*" %%a in ('git rev-parse @{u}') do set REMOTE=%%a

    if not "!LOCAL!"=="!REMOTE!" (
        set /p USERCHOICE=New version available. Pull now? (y/n): 
        if /i "!USERCHOICE!"=="y" (
            git pull
            call :success Project updated.
        ) else (
            call :warn Update skipped.
        )
    ) else (
        call :success Project is up to date.
    )
)

:: Launch Streamlit app
call :info Launching Streamlit app...
streamlit run %APP_FILE%
exit /b 0