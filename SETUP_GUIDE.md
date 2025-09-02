# 🏥 AI Risk Prediction Engine - Setup Guide

This guide will help you get the AI Risk Prediction Engine running in a virtual environment.

## 📁 Project Structure

```
Welldoc-2025/
├── 📄 app.py                    # Main Streamlit application
├── 📋 requirements.txt          # Python dependencies
├── 🐳 Dockerfile               # Container configuration
├── 📖 README.md                # Project documentation
├── ⚙️ setup_env.py             # Virtual environment setup
├── 🚀 start.bat / start.sh     # Quick start scripts
├── 🧪 test_system.py           # System tests
├── 📋 demo_script.py           # Demo walkthrough
├── 🎯 run.py                   # Interactive startup menu
├── 📊 SETUP_GUIDE.md           # This file
├── 
├── src/                        # Source code
│   ├── models/
│   │   ├── 📊 data_loader.py   # Dynamic data pipeline
│   │   └── 🤖 baseline_models.py # ML models
│   ├── explain/
│   │   └── 🔍 explainer.py     # SHAP explanations
│   └── ui/
│       ├── 🖥️ components.py    # Dashboard views
│       └── 🛠️ utils.py         # UI utilities
├──
├── data/                       # Your dataset goes here
├── artifacts/                  # Saved models
└── venv/                      # Virtual environment (created)
```

## 🚀 Quick Setup (Recommended)

### Step 1: Automated Setup
```bash
python setup_env.py
```

This will:
- ✅ Create virtual environment (`venv/`)
- ✅ Install all dependencies
- ✅ Test the installation
- ✅ Provide next steps

### Step 2: Start the Dashboard

**Windows:**
```batch
start.bat
```
OR
```batch
venv\Scripts\activate
streamlit run app.py
```

**Mac/Linux:**
```bash
bash start.sh
```
OR
```bash
source venv/bin/activate
streamlit run app.py
```

### Step 3: Access Dashboard
Open your browser to: **http://localhost:8501**

## 🛠️ Manual Setup (Alternative)

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```batch
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Installation
```bash
python test_system.py
```

### 5. Run Dashboard
```bash
streamlit run app.py
```

## 📊 Using Your Own Data

### Supported Formats:
- **MIMIC-IV**: `patients.csv`, `admissions.csv`, `chartevents.csv`, `labevents.csv`
- **eICU**: `patient.csv`, `vitalPeriodic.csv`, `lab.csv`
- **Custom**: Any CSV files with patient data

### Setup:
1. Place your data files in the `data/` directory
2. Turn off "Demo Mode" in the dashboard sidebar
3. The system will auto-detect your data format
4. Dashboard adapts automatically

## 🎯 Dashboard Features

### 👥 Cohort Overview
- Risk distribution across patients
- Sortable patient table
- Risk category breakdown
- Quick patient selection

### 👤 Patient Detail
- Individual risk assessment
- Top 5 risk factors explained
- Patient timeline visualization
- Clinical recommendations

### ⚙️ Admin Dashboard  
- Model performance metrics
- Global feature importance
- Data quality overview
- Performance plots

## 🧪 Testing & Demo

### Run System Tests
```bash
python test_system.py
```

### Interactive Demo Guide
```bash
python demo_script.py
```

### Easy Startup Menu
```bash
python run.py
```

## 🐳 Docker Option

### Build Container
```bash
docker build -t risk-prediction-dashboard .
```

### Run Container
```bash
docker run -p 8501:8501 risk-prediction-dashboard
```

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# Remove existing venv
rm -rf venv  # Mac/Linux
rmdir /s venv  # Windows

# Recreate
python setup_env.py
```

### Dependency Issues
```bash
# Activate venv first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Upgrade pip
python -m pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Test individual components
python -c "from src.models.data_loader import DynamicDataLoader; print('✅ Data loader OK')"
python -c "from src.models.baseline_models import RiskPredictor; print('✅ Models OK')"
```

### Port Already in Use
```bash
# Find and kill process using port 8501
netstat -ano | findstr :8501  # Windows
lsof -ti:8501 | xargs kill -9  # Mac/Linux

# Or use different port
streamlit run app.py --server.port 8502
```

## 📋 Development Commands

```bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Run tests
python test_system.py

# Start dashboard
streamlit run app.py

# Demo walkthrough
python demo_script.py

# Interactive menu
python run.py

# Test individual components
python src/models/data_loader.py
python src/models/baseline_models.py
python src/explain/explainer.py
```

## ✅ Success Indicators

You should see:
- ✅ Virtual environment created
- ✅ All dependencies installed 
- ✅ System tests passing
- ✅ Dashboard opens at http://localhost:8501
- ✅ Demo mode works with synthetic data
- ✅ All three dashboard views functional

## 🆘 Getting Help

1. **Check system tests**: `python test_system.py`
2. **Review error messages** in terminal
3. **Try demo mode first** (toggle in sidebar)
4. **Check Python version**: Python 3.9+ required
5. **Verify virtual environment** is activated
6. **Review setup steps** in this guide

## 🎉 Ready to Use!

Once setup is complete:
1. Dashboard runs at **http://localhost:8501**
2. **Demo mode** provides immediate functionality
3. **Add your data** to `data/` directory when ready
4. **Turn off demo mode** to use real predictions
5. **Explore all three views**: Cohort, Patient Detail, Admin

---

**Project Status**: ✅ Complete and Ready for Use  
**Demo Mode**: ✅ Fully Functional  
**Real Data Support**: ✅ MIMIC-IV, eICU, Custom formats  
**Documentation**: ✅ Complete with examples