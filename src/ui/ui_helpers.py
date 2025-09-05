"""
Enhanced UI Helper Functions

Additional UI components for the enhanced dashboard including calibration plots,
DCA curves, survival curves, and advanced visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def create_calibration_plot(y_true: np.ndarray = None, y_prob: np.ndarray = None, 
                          model_name: str = "Model") -> go.Figure:
    """Create calibration plot with reliability diagram"""
    
    if y_true is not None and y_prob is not None:
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_prob)
    else:
        # Create example calibration data
        mean_predicted_value = np.linspace(0.1, 0.9, 9)
        # Slightly miscalibrated example
        fraction_of_positives = mean_predicted_value * 0.9 + 0.05
        brier_score = 0.142
    
    # Create figure
    fig = go.Figure()
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash'),
        showlegend=True
    ))
    
    # Add model calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name=f'{model_name}',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Calibration Plot - {model_name}<br><sub>Brier Score: {brier_score:.3f}</sub>',
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=600,
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_dca_plot(y_true: np.ndarray = None, y_prob: np.ndarray = None,
                   model_name: str = "Model") -> go.Figure:
    """Create Decision Curve Analysis plot"""
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    if y_true is not None and y_prob is not None:
        n = len(y_true)
        prevalence = np.mean(y_true)
        
        # Initialize net benefit arrays
        net_benefit_model = np.zeros(len(thresholds))
        net_benefit_all = np.zeros(len(thresholds))
        net_benefit_none = np.zeros(len(thresholds))
        
        for i, threshold in enumerate(thresholds):
            # Model strategy: treat if predicted risk >= threshold
            predicted_positive = (y_prob >= threshold)
            
            # True positives and false positives
            tp = np.sum((predicted_positive == 1) & (y_true == 1))
            fp = np.sum((predicted_positive == 1) & (y_true == 0))
            
            # Net benefit for model
            net_benefit_model[i] = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            
            # Net benefit for "treat all" strategy
            net_benefit_all[i] = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            
            # Net benefit for "treat none" strategy
            net_benefit_none[i] = 0
    else:
        # Create example DCA data
        net_benefit_model = 0.15 * np.exp(-5 * (thresholds - 0.3)**2) - 0.02
        net_benefit_all = 0.25 - 2.5 * thresholds
        net_benefit_none = np.zeros(len(thresholds))
    
    # Create figure
    fig = go.Figure()
    
    # Add model curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=net_benefit_model,
        mode='lines',
        name=model_name,
        line=dict(color='blue', width=3)
    ))
    
    # Add "treat all" curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=net_benefit_all,
        mode='lines',
        name='Treat All',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add "treat none" curve
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=net_benefit_none,
        mode='lines',
        name='Treat None',
        line=dict(color='gray', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Decision Curve Analysis - {model_name}',
        xaxis_title='Risk Threshold',
        yaxis_title='Net Benefit',
        xaxis=dict(range=[0, 1]),
        template='plotly_white',
        width=700,
        height=500
    )
    
    return fig

def create_survival_curve(survival_data: Dict[str, Any], 
                         patient_id: str = None) -> go.Figure:
    """Create survival curve plot"""
    
    fig = go.Figure()
    
    if patient_id:
        # Individual patient survival curve
        if 'survival_functions' in survival_data and patient_id in survival_data['survival_functions']:
            timeline = survival_data.get('timeline', np.arange(0, 91))
            survival_prob = survival_data['survival_functions'][patient_id]
            
            fig.add_trace(go.Scatter(
                x=timeline,
                y=survival_prob,
                mode='lines',
                name=f'Patient {patient_id}',
                line=dict(color='blue', width=3)
            ))
    else:
        # Kaplan-Meier curves for risk groups
        if 'high_risk' in survival_data:
            high_risk_data = survival_data['high_risk']
            fig.add_trace(go.Scatter(
                x=high_risk_data['timeline'],
                y=high_risk_data['survival_function'],
                mode='lines',
                name=f"High Risk (n={high_risk_data['n_patients']})",
                line=dict(color='red', width=3)
            ))
        
        if 'low_risk' in survival_data:
            low_risk_data = survival_data['low_risk']
            fig.add_trace(go.Scatter(
                x=low_risk_data['timeline'],
                y=low_risk_data['survival_function'],
                mode='lines',
                name=f"Low Risk (n={low_risk_data['n_patients']})",
                line=dict(color='green', width=3)
            ))
    
    # Update layout
    fig.update_layout(
        title='Survival Curve Analysis',
        xaxis_title='Days',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        width=700,
        height=500
    )
    
    return fig

def create_model_comparison_table(trained_models: Dict[str, Any]) -> pd.DataFrame:
    """Create model comparison table"""
    
    comparison_data = []
    
    for model_name, model_data in trained_models.items():
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            
            if model_name == 'Cox PH':
                # Survival model metrics
                comparison_data.append({
                    'Model': model_name,
                    'C-Index': metrics.get('c_index', 'N/A'),
                    'AUROC': 'N/A',
                    'AUPRC': 'N/A',
                    'Log Likelihood': metrics.get('log_likelihood', 'N/A'),
                    'Events': metrics.get('n_events', 'N/A')
                })
            else:
                # Classification model metrics
                comparison_data.append({
                    'Model': model_name,
                    'C-Index': 'N/A',
                    'AUROC': f"{metrics.get('auc_roc', 0):.3f}",
                    'AUPRC': f"{metrics.get('auc_pr', 0):.3f}",
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Features': metrics.get('n_features', 'N/A')
                })
    
    return pd.DataFrame(comparison_data)

def create_dataset_summary_panel(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Create dataset summary information"""
    
    summary = {
        'total_patients': 0,
        'total_records': 0,
        'table_count': len(datasets),
        'feature_count': 0
    }
    
    # Calculate totals
    for table_name, df in datasets.items():
        summary['total_records'] += len(df)
        summary['feature_count'] += len(df.columns)
        
        # Count unique patients
        if 'patient_id' in df.columns:
            unique_patients = df['patient_id'].nunique()
            summary['total_patients'] = max(summary['total_patients'], unique_patients)
    
    # Basic statistics
    if 'patients' in datasets:
        patients_df = datasets['patients']
        summary['total_patients'] = len(patients_df)
        
        # Age distribution if available
        if 'age' in patients_df.columns:
            age_col = patients_df['age']
            if age_col.dtype in ['int64', 'float64']:
                summary['age_mean'] = age_col.mean()
                summary['age_range'] = (age_col.min(), age_col.max())
    
    # Outcome statistics
    if 'outcomes' in datasets:
        outcomes_df = datasets['outcomes']
        outcome_cols = [col for col in outcomes_df.columns 
                       if 'event' in col.lower() or 'deterioration' in col.lower()]
        if outcome_cols:
            outcome_col = outcome_cols[0]
            event_rate = outcomes_df[outcome_col].mean()
            summary['event_rate'] = event_rate
            summary['total_events'] = int(outcomes_df[outcome_col].sum())
    
    # Data completeness
    if 'patients' in datasets:
        completeness = (1 - datasets['patients'].isnull().sum() / len(datasets['patients'])).mean()
        summary['data_completeness'] = completeness
    
    return summary

def create_shap_explanation_chart(shap_values: np.ndarray, 
                                feature_names: List[str],
                                patient_id: str = None) -> go.Figure:
    """Create SHAP explanation chart"""
    
    if len(shap_values.shape) > 1:
        # For multiple patients, show average
        mean_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        # Single patient
        mean_shap = np.abs(shap_values)
    
    # Get top features
    top_indices = np.argsort(mean_shap)[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_shap[top_indices]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation='h',
        marker=dict(
            color=top_values,
            colorscale='RdYlBu_r',
            showscale=True
        )
    ))
    
    title = f"Top Risk Factors"
    if patient_id:
        title += f" - Patient {patient_id}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_risk_trajectory_chart(patient_data: pd.DataFrame, 
                               risk_history: pd.DataFrame = None) -> go.Figure:
    """Create patient risk trajectory over time"""
    
    fig = go.Figure()
    
    # If we have risk history, plot it
    if risk_history is not None and not risk_history.empty:
        fig.add_trace(go.Scatter(
            x=risk_history.get('date', range(len(risk_history))),
            y=risk_history.get('risk_score', []),
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
    
    # Add risk threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold")
    
    fig.update_layout(
        title="Patient Risk Trajectory",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=400
    )
    
    return fig

def create_clinical_action_checklist(risk_drivers: List[str], 
                                   risk_score: float) -> List[str]:
    """Generate clinical action checklist based on risk factors"""
    
    actions = []
    
    # High-level risk actions
    if risk_score > 0.7:
        actions.extend([
            "🚨 Consider immediate clinical assessment",
            "📞 Schedule follow-up within 24-48 hours",
            "💊 Review current medication regimen"
        ])
    elif risk_score > 0.3:
        actions.extend([
            "⚠️ Enhanced monitoring recommended",
            "📅 Schedule follow-up within 1-2 weeks"
        ])
    
    # Factor-specific actions
    for driver in risk_drivers[:3]:  # Top 3 drivers
        driver_lower = driver.lower()
        
        if 'age' in driver_lower:
            actions.append("👴 Consider age-appropriate care protocols")
        elif 'bp' in driver_lower or 'blood pressure' in driver_lower:
            actions.append("🩺 Monitor blood pressure closely")
        elif 'heart' in driver_lower:
            actions.append("❤️ Cardiac monitoring recommended")
        elif 'glucose' in driver_lower or 'diabetes' in driver_lower:
            actions.append("🍯 Review glucose management")
        elif 'medication' in driver_lower or 'drug' in driver_lower:
            actions.append("💊 Medication reconciliation needed")
        elif 'admission' in driver_lower:
            actions.append("🏥 Review admission patterns")
    
    return actions[:6]  # Limit to 6 actions