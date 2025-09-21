@echo off
echo Setting up virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

if "%1"=="debug" (
    set DEBUG=1
)

echo Starting GUI...
python app/main.py
