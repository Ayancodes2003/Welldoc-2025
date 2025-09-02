# AI Risk Prediction Engine - Streamlit Dashboard Prototype

A comprehensive dashboard for predicting 90-day deterioration risk in chronic care patients, built with Streamlit and designed to work with various healthcare datasets including MIMIC-IV, eICU, or custom formats.

## 🏥 Features

- **Dynamic Data Pipeline**: Automatically adapts to different dataset structures (MIMIC-IV, eICU, custom formats)
- **Baseline Models**: Logistic Regression and Random Forest with automatic feature engineering
- **SHAP Explanations**: Global and patient-level risk factor explanations in plain English
- **Interactive Dashboard**: 
  - Cohort overview with risk distribution and sortable patient table
  - Patient detail view with risk timeline and personalized explanations
  - Admin dashboard with model performance metrics
- **Demo Mode**: Works out-of-the-box with synthetic data for demonstration
- **Clinical Recommendations**: AI-generated next steps based on risk factors

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd Welldoc-2025
```

2. **Run automated setup**
```bash
python setup_env.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Test the installation
- Provide next steps

3. **Activate virtual environment and run**
```bash
# Windows
venv\Scripts\activate
streamlit run app.py

# Mac/Linux
source venv/bin/activate
streamlit run app.py
```

### Option 2: Manual Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Welldoc-2025
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the dashboard**
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Option 3: Docker

1. **Build the container**
```bash
docker build -t risk-prediction-dashboard .
```

2. **Run the container**
```bash
docker run -p 8501:8501 risk-prediction-dashboard
```

## 📊 Using Your Own Data

The system is designed to automatically detect and adapt to different dataset formats:

### MIMIC-IV Format
Place your MIMIC-IV files in the `data/` directory:
```
data/
├── patients.csv
├── admissions.csv
├── chartevents.csv
└── labevents.csv
```

### eICU Format
```
data/
├── patient.csv
├── vitalPeriodic.csv
└── lab.csv
```

### Custom Format
Any CSV files with patient data. The system will automatically:
- Detect patient ID columns
- Map common medical features
- Generate appropriate risk models

### Loading Your Data
1. Turn off "Demo Mode" in the sidebar
2. Place your data files in the `data/` directory
3. The system will automatically detect the format and load your data

## 🏗️ Architecture

```
src/
├── models/
│   ├── data_loader.py      # Dynamic data loading and format detection
│   └── baseline_models.py  # ML models with automatic feature engineering
├── explain/
│   └── explainer.py        # SHAP-based explanations
└── ui/
    ├── components.py       # Dashboard views
    └── utils.py           # UI helper functions
app.py                      # Main Streamlit application
```

## 📈 Dashboard Views

### 1. Cohort Overview
- Risk score distribution across all patients
- Sortable patient table with demographics
- Risk category breakdown (High/Medium/Low)
- Quick patient selection for detailed analysis

### 2. Patient Detail
- Individual risk assessment with gauge visualization
- Top 5 risk factors with plain-English explanations
- Patient timeline with vital signs trends
- Personalized clinical recommendations

### 3. Admin Dashboard
- Model performance metrics (AUC-ROC, AUC-PR, Accuracy)
- Global feature importance analysis
- Data quality overview
- Performance visualization plots

## 🔧 Configuration

The system automatically configures itself based on your data, but you can customize:

### Feature Engineering
- Vital signs aggregation (mean, std, trends)
- Lab value analysis (recent values, abnormal counts)
- Admission patterns (frequency, length of stay)
- Medication complexity

### Risk Thresholds
- Low risk: < 30%
- Medium risk: 30-70%
- High risk: > 70%

## 🧪 Demo Mode

Demo mode provides a complete working demonstration with:
- 50 synthetic patients with realistic medical profiles
- Age-correlated risk scores
- Mock explanations and recommendations
- All dashboard functionality working out-of-the-box

## 📋 Requirements

- Python 3.9+
- Streamlit 1.28+
- scikit-learn 1.3+
- SHAP 0.42+
- pandas, numpy, plotly

See `requirements.txt` for complete list.

## 🏥 Model Details

### Baseline Models
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Better performance with feature interactions
- Automatic feature scaling and encoding
- Cross-validation for robust evaluation

### Feature Engineering
- **Demographics**: Age categories, gender encoding
- **Vitals**: Statistical aggregations, variability measures, trends
- **Labs**: Recent values, abnormal counts, change detection
- **Admissions**: Frequency, length patterns, recent activity
- **Medications**: Counts, high-risk drug classes

### Explainability
- SHAP values for individual predictions
- Feature importance rankings
- Plain-English explanations
- Clinical recommendations based on risk factors

## 🔐 Security & Privacy

- No PHI stored permanently
- Local processing (no data sent to external services)
- Docker isolation for deployment
- Configurable data retention policies

## 📊 Performance

Typical performance on demo data:
- Model training: < 30 seconds
- Risk prediction: < 1 second per patient
- Dashboard rendering: < 3 seconds
- Memory usage: < 2GB

## 🛠️ Development

### Testing the Pipeline
```bash
# Test data loader
python src/models/data_loader.py

# Test baseline models
python src/models/baseline_models.py

# Test explainer
python src/explain/explainer.py
```

### Adding New Data Sources
1. Create new adapter class inheriting from `DatasetAdapter`
2. Implement `detect_format()`, `load_and_standardize()`, and `get_feature_config()`
3. Add to `DynamicDataLoader.adapters` list

### Customizing Explanations
Edit `explanation_templates` in `RiskExplainer` to customize how features are described in plain English.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Future Enhancements

- [ ] Time-to-event modeling (survival analysis)
- [ ] Real-time data streaming
- [ ] Advanced transformer models
- [ ] Multi-outcome predictions
- [ ] Integration with EHR systems
- [ ] Automated model retraining
- [ ] Advanced visualization options

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Review the demo mode for expected functionality
- Check data format requirements above

---

**Note**: This is a prototype designed for demonstration and development purposes. For production use in clinical settings, ensure appropriate validation, regulatory compliance, and integration with clinical workflows.