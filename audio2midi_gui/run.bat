@echo off
if "%1"=="debug" (
    set DEBUG=1
)
python app/main.py
