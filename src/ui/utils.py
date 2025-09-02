"""
UI Utility Functions

Helper functions for the Streamlit dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import os

# Add src directory to path if not already there
current_dir = Path(__file__).parent.parent.parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from models.data_loader import DynamicDataLoader
from models.baseline_models import RiskPredictor
from explain.explainer import RiskExplainer

@st.cache_resource
def load_data(demo_mode: bool = True, data_path: str = None) -> Dict[str, pd.DataFrame]:
    """Load and cache dataset"""
    
    loader = DynamicDataLoader()
    
    if demo_mode:
        # Create demo dataset
        datasets = loader.create_demo_dataset()
        st.session_state.feature_config = loader.get_feature_config()
    else:
        # Load real dataset
        if not data_path:
            data_path = "data"
        
        datasets = loader.load_dataset(data_path)
        st.session_state.feature_config = loader.get_feature_config()
    
    return datasets

@st.cache_resource
def initialize_models(datasets: Dict[str, pd.DataFrame], model_type: str = "Random Forest", 
                     demo_mode: bool = True, _feature_config: Dict = None) -> tuple:
    """Initialize and cache models"""
    
    # Initialize predictor
    predictor = RiskPredictor(model_type=model_type, demo_mode=demo_mode)
    
    if not demo_mode:
        # Train model on real data
        metrics = predictor.train(datasets, _feature_config)
        st.session_state.training_metrics = metrics
    
    # Initialize explainer
    explainer = RiskExplainer(predictor)
    if not demo_mode:
        explainer.initialize_explainer(datasets)
    
    return predictor, explainer

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'datasets' not in st.session_state:
        st.session_state.datasets = None
    
    if 'feature_config' not in st.session_state:
        st.session_state.feature_config = {}
    
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = {}

def create_risk_gauge(risk_score: float, title: str = "Risk Score") -> go.Figure:
    """Create a gauge chart for risk score"""
    
    # Determine color based on risk level
    if risk_score < 0.3:
        color = "green"
    elif risk_score < 0.7:
        color = "orange" 
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_sparkline(values: List[float], title: str = "") -> go.Figure:
    """Create a small sparkline chart"""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color='blue', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        height=100,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig

def create_feature_importance_chart(importance_data: List[Dict]) -> go.Figure:
    """Create horizontal bar chart for feature importance"""
    
    features = [item['description'] for item in importance_data]
    importances = [item['percentage'] for item in importance_data]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig.update_layout(
        title="Most Important Risk Factors",
        xaxis_title="Importance (%)",
        yaxis_title="Features",
        height=400
    )
    
    return fig

def create_patient_timeline(patient_data: pd.DataFrame, risk_predictions: pd.DataFrame) -> go.Figure:
    """Create patient timeline with risk trajectory"""
    
    fig = go.Figure()
    
    # Add risk score line if available
    if not risk_predictions.empty and 'date' in risk_predictions.columns:
        fig.add_trace(go.Scatter(
            x=risk_predictions['date'],
            y=risk_predictions['risk_score'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='red')
        ))
    
    # Add vital signs if available
    if not patient_data.empty:
        for vital in ['heart_rate', 'systolic_bp', 'temperature']:
            if vital in patient_data.columns:
                # Normalize to 0-1 scale for visualization
                values = patient_data[vital].dropna()
                if len(values) > 0:
                    normalized = (values - values.min()) / (values.max() - values.min())
                    
                    fig.add_trace(go.Scatter(
                        x=patient_data.index[:len(normalized)],
                        y=normalized,
                        mode='lines',
                        name=vital.replace('_', ' ').title(),
                        opacity=0.7
                    ))
    
    fig.update_layout(
        title="Patient Risk and Vital Signs Timeline",
        xaxis_title="Time",
        yaxis_title="Normalized Values",
        height=400
    )
    
    return fig

def create_cohort_risk_distribution(predictions: pd.DataFrame) -> go.Figure:
    """Create histogram of risk score distribution"""
    
    fig = px.histogram(
        predictions, 
        x='risk_score',
        nbins=20,
        title="Risk Score Distribution Across Cohort",
        labels={'risk_score': 'Risk Score', 'count': 'Number of Patients'}
    )
    
    # Add risk threshold lines
    fig.add_vline(x=0.3, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    return fig

def format_risk_score(score: float) -> str:
    """Format risk score as percentage with color"""
    
    percentage = score * 100
    
    if score < 0.3:
        color = "green"
    elif score < 0.7:
        color = "orange"
    else:
        color = "red"
    
    return f"<span style='color: {color}; font-weight: bold;'>{percentage:.1f}%</span>"

def get_risk_category_color(category: str) -> str:
    """Get color for risk category"""
    
    colors = {
        'Low': 'green',
        'Medium': 'orange', 
        'High': 'red'
    }
    
    return colors.get(category, 'gray')

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Create confusion matrix heatmap"""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    return fig

def create_roc_curve_plot(y_true: np.ndarray, y_scores: np.ndarray) -> go.Figure:
    """Create ROC curve plot"""
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue')
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400
    )
    
    return fig

def display_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       help_text: Optional[str] = None):
    """Display a metric card"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
    
    return col1, col2

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    
    return numerator / denominator if denominator != 0 else default

def get_patient_summary(patient_id: str, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Get summary information for a patient"""
    
    summary = {'patient_id': patient_id}
    
    # Get demographics
    if 'patients' in datasets:
        patient_demo = datasets['patients'][datasets['patients']['patient_id'] == patient_id]
        if not patient_demo.empty:
            summary.update({
                'age': patient_demo['age'].iloc[0] if 'age' in patient_demo.columns else 'Unknown',
                'gender': patient_demo['gender'].iloc[0] if 'gender' in patient_demo.columns else 'Unknown'
            })
    
    # Get recent vitals
    if 'vitals' in datasets:
        patient_vitals = datasets['vitals'][datasets['vitals']['patient_id'] == patient_id]
        if not patient_vitals.empty:
            # Get most recent measurements
            recent_vitals = patient_vitals.tail(1)
            vital_cols = ['heart_rate', 'systolic_bp', 'temperature', 'oxygen_saturation']
            
            for col in vital_cols:
                if col in recent_vitals.columns:
                    summary[f'recent_{col}'] = recent_vitals[col].iloc[0]
    
    # Get admission count
    if 'admissions' in datasets:
        patient_admissions = datasets['admissions'][datasets['admissions']['patient_id'] == patient_id]
        summary['admission_count'] = len(patient_admissions)
    
    return summary

def filter_patients_by_risk(predictions: pd.DataFrame, risk_filter: str) -> pd.DataFrame:
    """Filter patients by risk category"""
    
    if risk_filter == "All":
        return predictions
    elif risk_filter == "High Risk (>70%)":
        return predictions[predictions['risk_score'] > 0.7]
    elif risk_filter == "Medium Risk (30-70%)":
        return predictions[(predictions['risk_score'] >= 0.3) & (predictions['risk_score'] <= 0.7)]
    elif risk_filter == "Low Risk (<30%)":
        return predictions[predictions['risk_score'] < 0.3]
    else:
        return predictions