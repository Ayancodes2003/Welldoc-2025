"""
AI Risk Prediction Engine - Enhanced Streamlit Dashboard

A comprehensive, clinician-friendly dashboard for predicting 90-day deterioration risk
in chronic care patients with dynamic dataset support and advanced UI features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import zipfile
import tempfile
from pathlib import Path
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.data_loader import DynamicDataLoader
from src.models.baseline_models import RiskPredictor
from src.explain.explainer import RiskExplainer
from src.ui.components import CohortView, PatientDetailView, AdminView
from src.ui.utils import (
    initialize_session_state, create_risk_gauge, create_sparkline,
    format_risk_score, get_risk_category_color
)
from src.ui.feedback import initialize_feedback_system
from src.models.survival_models import SurvivalModelManager
from src.models.evaluation_advanced import ModelCalibration, DecisionCurveAnalysis

def handle_dataset_upload():
    """Handle dataset upload and processing"""
    
    with st.sidebar:
        st.header("📁 Dataset Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'zip'],
            help="Upload a CSV file or ZIP containing multiple CSV files (MIMIC-IV, eICU, etc.)"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing uploaded dataset..."):
                try:
                    # Create temp directory for uploaded data
                    temp_dir = Path("temp_upload")
                    temp_dir.mkdir(exist_ok=True)
                    
                    if uploaded_file.name.endswith('.zip'):
                        # Handle ZIP file
                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        data_path = str(temp_dir)
                    else:
                        # Handle single CSV file
                        csv_path = temp_dir / uploaded_file.name
                        with open(csv_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        data_path = str(temp_dir)
                    
                    # Process with DynamicDataLoader
                    loader = DynamicDataLoader()
                    datasets = loader.load_dataset(data_path)
                    feature_config = loader.get_feature_config()
                    
                    # Save processed data to main data directory
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    
                    for name, df in datasets.items():
                        df.to_csv(data_dir / f"{name}.csv", index=False)
                    
                    # Update session state
                    st.session_state.datasets = datasets
                    st.session_state.feature_config = feature_config
                    st.session_state.predictions = None  # Reset predictions
                    st.session_state.models_trained = False
                    st.session_state.uploaded_dataset = True
                    
                    # Clean up temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    st.success(f"✅ Dataset uploaded successfully!")
                    st.info(f"Detected format: {loader.current_adapter.__class__.__name__}")
                    st.info(f"Tables: {list(datasets.keys())}")
                    
                    return datasets, feature_config
                    
                except Exception as e:
                    st.error(f"Error processing dataset: {str(e)}")
                    return None, None
        
        return None, None

def load_data_with_fallback():
    """Load data with fallback to pre-loaded dataset"""
    
    # Check if we have an uploaded dataset
    if (hasattr(st.session_state, 'uploaded_dataset') and 
        st.session_state.uploaded_dataset and 
        st.session_state.datasets is not None):
        return st.session_state.datasets, st.session_state.feature_config
    
    # Try to load from data directory (diabetes dataset)
    try:
        loader = DynamicDataLoader()
        if Path("data/patients.csv").exists() and Path("data/outcomes.csv").exists():
            datasets = loader.load_dataset("data/")
            feature_config = loader.get_feature_config()
            st.session_state.datasets = datasets
            st.session_state.feature_config = feature_config
            return datasets, feature_config
        else:
            # Fallback to demo data
            datasets = loader.create_demo_dataset()
            feature_config = loader.get_feature_config()
            st.session_state.datasets = datasets
            st.session_state.feature_config = feature_config
            return datasets, feature_config
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def create_sidebar_filters(datasets):
    """Create sidebar filters for data exploration"""
    
    with st.sidebar:
        st.header("🔍 Filters")
        
        filters = {}
        
        # Age filter if available
        if 'patients' in datasets and 'age' in datasets['patients'].columns:
            age_col = datasets['patients']['age']
            if age_col.dtype in ['int64', 'float64']:
                age_range = st.slider(
                    "Age Range",
                    min_value=int(age_col.min()),
                    max_value=int(age_col.max()),
                    value=(int(age_col.min()), int(age_col.max()))
                )
                filters['age_range'] = age_range
        
        # Risk level filter
        risk_filter = st.multiselect(
            "Risk Levels",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )
        filters['risk_levels'] = risk_filter
        
        return filters

def main():
    """Main Streamlit application with enhanced UI"""
    st.set_page_config(
        page_title="AI Risk Prediction Engine",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize feedback system
    initialize_feedback_system()
    
    # Enhanced CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Risk Prediction Engine</h1>', unsafe_allow_html=True)
    st.markdown("**Enhanced dashboard for 90-day deterioration risk prediction with dynamic dataset support**")
    
    # Handle dataset upload
    uploaded_data, uploaded_config = handle_dataset_upload()
    
    # Load data with fallback
    datasets, feature_config = load_data_with_fallback()
    
    if datasets is None:
        st.error("Could not load any dataset. Please upload a dataset or check data directory.")
        return
    
    # Create sidebar filters
    filters = create_sidebar_filters(datasets)
    
    # Initialize models
    if not hasattr(st.session_state, 'models_trained') or not st.session_state.models_trained:
        with st.spinner("Training models on current dataset..."):
            trained_models = {}
            
            # Train core models
            for model_type in ["Logistic Regression", "Random Forest"]:
                try:
                    predictor = RiskPredictor(model_type=model_type)
                    metrics = predictor.train(datasets, feature_config)
                    trained_models[model_type] = {
                        'predictor': predictor,
                        'metrics': metrics
                    }
                except Exception as e:
                    st.warning(f"Failed to train {model_type}: {str(e)}")
            
            # Train Cox PH survival model
            try:
                survival_manager = SurvivalModelManager(feature_config)
                
                # We need to create a feature engineer for survival training
                from src.models.baseline_models import FeatureEngineer
                feature_engineer = FeatureEngineer(feature_config)
                
                survival_results = survival_manager.train_survival_models(datasets, feature_engineer)
                
                if 'cox' in survival_results and 'error' not in survival_results['cox']:
                    trained_models['Cox PH'] = {
                        'survival_manager': survival_manager,
                        'metrics': survival_results['cox']
                    }
                    st.success(f"✅ Cox PH model trained successfully! C-index: {survival_results['cox'].get('c_index', 'N/A')}")
                else:
                    st.warning("Cox PH model training failed, using mock survival data")
                    
                # Store survival results for dashboard
                st.session_state.survival_results = survival_results
                    
            except Exception as e:
                st.warning(f"Failed to train Cox PH model: {str(e)}")
            
            st.session_state.trained_models = trained_models
            st.session_state.models_trained = True
    
    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["👥 Cohort View", "👤 Patient Detail", "📊 Admin Panel"])
    
    with tab1:
        cohort_view = CohortView(datasets, st.session_state.trained_models, None)
        cohort_view.render_enhanced(filters)
    
    with tab2:
        detail_view = PatientDetailView(datasets, st.session_state.trained_models, None)
        detail_view.render_enhanced()
    
    with tab3:
        admin_view = AdminView(datasets, st.session_state.trained_models, None)
        admin_view.render_enhanced()

if __name__ == "__main__":
    main()