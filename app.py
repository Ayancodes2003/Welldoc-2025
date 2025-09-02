"""
AI Risk Prediction Engine - Streamlit Dashboard Prototype

A comprehensive dashboard for predicting 90-day deterioration risk in chronic care patients.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.baseline_models import RiskPredictor
from src.explain.explainer import RiskExplainer
from src.ui.components import CohortView, PatientDetailView, AdminView
from src.ui.utils import load_data, initialize_session_state, initialize_models

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Risk Prediction Engine",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { color: #d62728; }
    .risk-medium { color: #ff7f0e; }
    .risk-low { color: #2ca02c; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Risk Prediction Engine</h1>', unsafe_allow_html=True)
    st.markdown("**Predicting 90-day deterioration risk in chronic care patients**")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        view = st.radio(
            "Select View:",
            ["Cohort Overview", "Patient Detail", "Admin Dashboard"],
            help="Choose the dashboard view"
        )
        
        st.divider()
        
        # Demo mode toggle
        demo_mode = st.checkbox("Demo Mode", value=True, help="Use synthetic data and mock predictions")
        
        # Model selection
        if not demo_mode:
            model_type = st.selectbox(
                "Model Type:",
                ["Logistic Regression", "Random Forest", "Transformer"],
                help="Select the prediction model"
            )
        else:
            model_type = "Mock Model"
            st.info("🎭 Demo mode active - using synthetic data")
    
    # Initialize components
    try:
        # Load data and model
        data = load_data(demo_mode=demo_mode)
        
        # Initialize models with caching
        predictor, explainer = initialize_models(
            data, 
            model_type=model_type, 
            demo_mode=demo_mode,
            _feature_config=st.session_state.feature_config
        )
        
        # Route to appropriate view
        if view == "Cohort Overview":
            cohort_view = CohortView(data, predictor, explainer)
            cohort_view.render()
        elif view == "Patient Detail":
            detail_view = PatientDetailView(data, predictor, explainer)
            detail_view.render()
        elif view == "Admin Dashboard":
            admin_view = AdminView(data, predictor, explainer)
            admin_view.render()
            
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        st.info("Please ensure all dependencies are installed and data is available.")

if __name__ == "__main__":
    main()