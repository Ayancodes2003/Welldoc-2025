#!/bin/bash

echo "🏥 AI Risk Prediction Engine - Quick Start"
echo "=========================================="

if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python setup_env.py"
    exit 1
fi

echo "✅ Activating virtual environment..."
source venv/bin/activate

echo "🚀 Starting Streamlit dashboard..."
echo "Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo

streamlit run app.py