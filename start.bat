@echo off
echo 🏥 AI Risk Prediction Engine - Quick Start
echo ==========================================

if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run: python setup_env.py
    pause
    exit /b 1
)

echo ✅ Activating virtual environment...
call venv\Scripts\activate.bat

echo 🚀 Starting Streamlit dashboard...
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run app.py