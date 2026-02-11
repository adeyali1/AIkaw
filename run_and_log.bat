@echo off
python -u app.py > output.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo App crashed with error level %ERRORLEVEL% >> crash.log
)
