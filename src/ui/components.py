"""
Streamlit UI Components

Main dashboard views for the risk prediction engine.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .utils import (
    create_risk_gauge, create_sparkline, create_feature_importance_chart,
    create_patient_timeline, create_cohort_risk_distribution, format_risk_score,
    get_risk_category_color, create_confusion_matrix_plot, create_roc_curve_plot,
    display_metric_card, get_patient_summary, filter_patients_by_risk
)

class CohortView:
    """Cohort overview dashboard showing all patients and their risk scores"""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame], predictor, explainer):
        self.datasets = datasets
        self.predictor = predictor
        self.explainer = explainer
    
    def render(self):
        """Render the cohort view dashboard"""
        
        st.header("👥 Cohort Risk Overview")
        st.markdown("Overview of risk predictions across all patients in the cohort.")
        
        # Generate predictions if not cached
        if st.session_state.predictions is None:
            with st.spinner("Generating risk predictions..."):
                predictions = self.predictor.predict(self.datasets)
                st.session_state.predictions = predictions
        else:
            predictions = st.session_state.predictions
        
        # Summary metrics
        self._render_summary_metrics(predictions)
        
        # Risk distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_risk_distribution(predictions)
        
        with col2:
            self._render_risk_summary_stats(predictions)
        
        # Patient table with filters
        st.divider()
        self._render_patient_table(predictions)
    
    def _render_summary_metrics(self, predictions: pd.DataFrame):
        """Render summary metrics cards"""
        
        total_patients = len(predictions)
        high_risk_count = len(predictions[predictions['risk_score'] > 0.7])
        medium_risk_count = len(predictions[(predictions['risk_score'] >= 0.3) & (predictions['risk_score'] <= 0.7)])
        low_risk_count = len(predictions[predictions['risk_score'] < 0.3])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Patients",
                total_patients,
                help="Total number of patients in the cohort"
            )
        
        with col2:
            st.metric(
                "High Risk",
                high_risk_count,
                f"{high_risk_count/total_patients*100:.1f}%",
                help="Patients with >70% deterioration risk"
            )
        
        with col3:
            st.metric(
                "Medium Risk", 
                medium_risk_count,
                f"{medium_risk_count/total_patients*100:.1f}%",
                help="Patients with 30-70% deterioration risk"
            )
        
        with col4:
            st.metric(
                "Low Risk",
                low_risk_count,
                f"{low_risk_count/total_patients*100:.1f}%", 
                help="Patients with <30% deterioration risk"
            )
    
    def _render_risk_distribution(self, predictions: pd.DataFrame):
        """Render risk score distribution chart"""
        
        st.subheader("Risk Score Distribution")
        
        fig = create_cohort_risk_distribution(predictions)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_summary_stats(self, predictions: pd.DataFrame):
        """Render risk summary statistics"""
        
        st.subheader("Risk Statistics")
        
        mean_risk = predictions['risk_score'].mean()
        median_risk = predictions['risk_score'].median()
        std_risk = predictions['risk_score'].std()
        
        st.metric("Average Risk", f"{mean_risk:.1%}")
        st.metric("Median Risk", f"{median_risk:.1%}")
        st.metric("Risk Variability", f"{std_risk:.1%}")
        
        # Risk category pie chart
        risk_counts = predictions['risk_category'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Categories",
            color_discrete_map={
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_patient_table(self, predictions: pd.DataFrame):
        """Render searchable and sortable patient table"""
        
        st.subheader("Patient Risk Table")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level:",
                ["All", "High Risk (>70%)", "Medium Risk (30-70%)", "Low Risk (<30%)"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Risk Score (High to Low)", "Risk Score (Low to High)", "Patient ID"]
            )
        
        with col3:
            search_term = st.text_input("Search Patient ID:")
        
        # Apply filters
        filtered_predictions = filter_patients_by_risk(predictions, risk_filter)
        
        if search_term:
            filtered_predictions = filtered_predictions[
                filtered_predictions['patient_id'].str.contains(search_term, case=False)
            ]
        
        # Apply sorting
        if sort_by == "Risk Score (High to Low)":
            filtered_predictions = filtered_predictions.sort_values('risk_score', ascending=False)
        elif sort_by == "Risk Score (Low to High)":
            filtered_predictions = filtered_predictions.sort_values('risk_score', ascending=True)
        else:
            filtered_predictions = filtered_predictions.sort_values('patient_id')
        
        # Enhance table with additional info
        enhanced_table = self._enhance_patient_table(filtered_predictions)
        
        # Display table with selection
        selected_indices = st.dataframe(
            enhanced_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Handle patient selection
        if selected_indices['selection']['rows']:
            selected_idx = selected_indices['selection']['rows'][0]
            selected_patient = enhanced_table.iloc[selected_idx]['Patient ID']
            
            if st.button("View Patient Detail", type="primary"):
                st.session_state.selected_patient = selected_patient
                st.rerun()
    
    def _enhance_patient_table(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Enhance prediction table with additional patient information"""
        
        enhanced = predictions.copy()
        enhanced['Patient ID'] = enhanced['patient_id']
        enhanced['Risk Score'] = enhanced['risk_score'].apply(lambda x: f"{x:.1%}")
        enhanced['Risk Category'] = enhanced['risk_category']
        
        # Add demographic info if available
        if 'patients' in self.datasets:
            demo_info = []
            for _, row in enhanced.iterrows():
                patient_summary = get_patient_summary(row['patient_id'], self.datasets)
                demo_info.append({
                    'Age': patient_summary.get('age', 'Unknown'),
                    'Gender': patient_summary.get('gender', 'Unknown'),
                    'Admissions': patient_summary.get('admission_count', 0)
                })
            
            demo_df = pd.DataFrame(demo_info)
            enhanced = pd.concat([enhanced, demo_df], axis=1)
        
        # Add trend sparklines (mock for now)
        enhanced['Risk Trend'] = '📈' if np.random.random() > 0.5 else '📉'
        
        # Add top risk driver
        enhanced['Top Risk Factor'] = enhanced.apply(
            lambda row: self._get_top_risk_factor(row['patient_id']), axis=1
        )
        
        # Select and order columns
        display_columns = ['Patient ID', 'Risk Score', 'Risk Category', 'Risk Trend']
        
        if 'Age' in enhanced.columns:
            display_columns.extend(['Age', 'Gender', 'Admissions'])
        
        display_columns.append('Top Risk Factor')
        
        return enhanced[display_columns]
    
    def _get_top_risk_factor(self, patient_id: str) -> str:
        """Get the top risk factor for a patient (simplified)"""
        
        # This would normally use the explainer to get the top factor
        # For now, return a mock factor
        factors = ['Age', 'Blood Pressure', 'Heart Rate', 'Previous Admissions', 'Medications']
        return np.random.choice(factors)

class PatientDetailView:
    """Detailed patient view with risk explanation and recommendations"""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame], predictor, explainer):
        self.datasets = datasets
        self.predictor = predictor
        self.explainer = explainer
    
    def render(self):
        """Render the patient detail view"""
        
        st.header("👤 Patient Risk Detail")
        
        # Patient selection
        patient_id = self._render_patient_selector()
        
        if not patient_id:
            st.info("Please select a patient to view detailed risk analysis.")
            return
        
        # Generate prediction and explanation for selected patient
        with st.spinner("Analyzing patient risk factors..."):
            patient_data = self._get_patient_data(patient_id)
            
            if patient_data.empty:
                st.error(f"No data found for patient {patient_id}")
                return
            
            # Get prediction
            predictions = self.predictor.predict({
                key: df[df['patient_id'] == patient_id] 
                for key, df in self.datasets.items()
            })
            
            if predictions.empty:
                st.error("Could not generate prediction for this patient")
                return
            
            risk_score = predictions['risk_score'].iloc[0]
            risk_category = predictions['risk_category'].iloc[0]
            
            # Get explanation
            explanation = self.explainer.explain_patient(patient_id, self.datasets)
        
        # Render patient profile
        self._render_patient_profile(patient_id, risk_score, risk_category)
        
        # Render risk explanation
        self._render_risk_explanation(explanation)
        
        # Render patient timeline
        self._render_patient_timeline(patient_id)
        
        # Render recommendations
        self._render_recommendations(explanation)
    
    def _render_patient_selector(self) -> Optional[str]:
        """Render patient selection interface"""
        
        # Get list of available patients
        patient_ids = []
        for df in self.datasets.values():
            if 'patient_id' in df.columns:
                patient_ids.extend(df['patient_id'].unique())
        
        patient_ids = sorted(list(set(patient_ids)))
        
        # Check if patient was selected from cohort view
        if st.session_state.selected_patient:
            default_idx = patient_ids.index(st.session_state.selected_patient) if st.session_state.selected_patient in patient_ids else 0
        else:
            default_idx = 0
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_patient = st.selectbox(
                "Select Patient:",
                patient_ids,
                index=default_idx,
                help="Choose a patient to view detailed risk analysis"
            )
        
        with col2:
            if st.button("Clear Selection"):
                st.session_state.selected_patient = None
                st.rerun()
        
        return selected_patient
    
    def _get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Get all data for a specific patient"""
        
        patient_data = pd.DataFrame()
        
        for table_name, df in self.datasets.items():
            if 'patient_id' in df.columns:
                table_data = df[df['patient_id'] == patient_id]
                if not table_data.empty:
                    # Add table source column
                    table_data = table_data.copy()
                    table_data['data_source'] = table_name
                    
                    if patient_data.empty:
                        patient_data = table_data
                    else:
                        # Merge or concatenate based on structure
                        try:
                            patient_data = pd.concat([patient_data, table_data], ignore_index=True)
                        except:
                            pass  # Skip if merge fails
        
        return patient_data
    
    def _render_patient_profile(self, patient_id: str, risk_score: float, risk_category: str):
        """Render patient profile section"""
        
        st.subheader(f"Patient Profile: {patient_id}")
        
        # Get patient summary
        patient_summary = get_patient_summary(patient_id, self.datasets)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            st.write(f"Age: {patient_summary.get('age', 'Unknown')}")
            st.write(f"Gender: {patient_summary.get('gender', 'Unknown')}")
            st.write(f"Admissions: {patient_summary.get('admission_count', 0)}")
        
        with col2:
            st.markdown("**Recent Vitals**")
            for key, value in patient_summary.items():
                if key.startswith('recent_'):
                    vital_name = key.replace('recent_', '').replace('_', ' ').title()
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        st.write(f"{vital_name}: {value:.1f}")
        
        with col3:
            st.markdown("**Risk Assessment**")
            
            # Risk gauge
            fig = create_risk_gauge(risk_score, "90-Day Risk")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_explanation(self, explanation: Dict[str, Any]):
        """Render risk factor explanation"""
        
        st.subheader("🔍 Risk Factor Analysis")
        
        if 'explanations' not in explanation or not explanation['explanations']:
            st.warning("No risk factor explanations available")
            return
        
        st.markdown(f"**Risk Score: {explanation['risk_score']:.1%}** ({explanation['risk_category']} Risk)")
        
        # Display top risk factors
        st.markdown("**Top 5 Risk Factors:**")
        
        for i, exp in enumerate(explanation['explanations'][:5], 1):
            
            # Create impact badge
            impact_color = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }.get(exp.get('impact', 'Medium'), '🟡')
            
            direction_icon = '⬆️' if exp['direction'] == 'increases' else '⬇️'
            
            st.markdown(f"""
            **{i}. {exp['description']}**  
            {impact_color} {exp['impact']} Impact | {direction_icon} {exp['direction'].title()} Risk  
            """)
    
    def _render_patient_timeline(self, patient_id: str):
        """Render patient timeline with vitals and risk"""
        
        st.subheader("📊 Patient Timeline")
        
        # Get patient vitals over time
        if 'vitals' in self.datasets:
            patient_vitals = self.datasets['vitals'][
                self.datasets['vitals']['patient_id'] == patient_id
            ]
            
            if not patient_vitals.empty:
                # Create timeline plot
                fig = create_patient_timeline(patient_vitals, pd.DataFrame())
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No vital signs data available for timeline")
        else:
            st.info("No vital signs data available")
    
    def _render_recommendations(self, explanation: Dict[str, Any]):
        """Render clinical recommendations"""
        
        st.subheader("📋 Clinical Recommendations")
        
        if 'explanations' in explanation:
            recommendations = self.explainer.generate_recommendations(explanation['explanations'])
            
            for i, recommendation in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {recommendation}")
        else:
            # Default recommendations
            st.markdown("**1.** Continue monitoring patient's condition")
            st.markdown("**2.** Follow standard care protocols")
            st.markdown("**3.** Contact healthcare team if condition changes")

class AdminView:
    """Admin dashboard with model performance metrics"""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame], predictor, explainer):
        self.datasets = datasets
        self.predictor = predictor
        self.explainer = explainer
    
    def render(self):
        """Render the admin dashboard"""
        
        st.header("⚙️ Model Performance Dashboard")
        st.markdown("Model performance metrics and global feature importance analysis.")
        
        # Model information
        self._render_model_info()
        
        # Performance metrics
        if hasattr(self.predictor, 'training_metrics') and self.predictor.training_metrics:
            self._render_performance_metrics()
        else:
            st.info("Model performance metrics not available in demo mode")
        
        # Global explanations
        self._render_global_explanations()
        
        # Data quality metrics
        self._render_data_quality()
    
    def _render_model_info(self):
        """Render model information section"""
        
        st.subheader("Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", self.predictor.model_type)
        
        with col2:
            demo_status = "Demo Mode" if self.predictor.demo_mode else "Production Mode"
            st.metric("Status", demo_status)
        
        with col3:
            training_status = "Trained" if self.predictor.is_trained else "Not Trained"
            st.metric("Training Status", training_status)
    
    def _render_performance_metrics(self):
        """Render model performance metrics"""
        
        st.subheader("Model Performance")
        
        metrics = getattr(self.predictor, 'training_metrics', {})
        
        if not metrics:
            st.warning("No training metrics available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auc_roc = metrics.get('auc_roc', 0)
            st.metric("AUC-ROC", f"{auc_roc:.3f}")
        
        with col2:
            auc_pr = metrics.get('auc_pr', 0)
            st.metric("AUC-PR", f"{auc_pr:.3f}")
        
        with col3:
            accuracy = metrics.get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.1%}")
        
        with col4:
            n_features = metrics.get('n_features', 0)
            st.metric("Features", f"{n_features}")
        
        # Model performance visualizations (mock data for demo)
        if st.button("Show Performance Plots"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Mock ROC curve
                y_true = np.random.choice([0, 1], 100)
                y_scores = np.random.random(100)
                fig_roc = create_roc_curve_plot(y_true, y_scores)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                # Mock confusion matrix  
                y_pred = (y_scores > 0.5).astype(int)
                fig_cm = create_confusion_matrix_plot(y_true, y_pred)
                st.plotly_chart(fig_cm, use_container_width=True)
    
    def _render_global_explanations(self):
        """Render global feature importance"""
        
        st.subheader("Global Feature Importance")
        
        # Get global explanations
        global_exp = self.explainer.get_global_explanations(self.datasets)
        
        if global_exp['feature_importance']:
            # Summary text
            if global_exp['summary']:
                st.markdown(f"**Summary:** {global_exp['summary']}")
            
            # Feature importance chart
            fig = create_feature_importance_chart(global_exp['feature_importance'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("**Detailed Feature Importance:**")
            importance_df = pd.DataFrame(global_exp['feature_importance'])
            st.dataframe(importance_df, use_container_width=True)
        
        else:
            st.info("Global feature importance not available")
    
    def _render_data_quality(self):
        """Render data quality metrics"""
        
        st.subheader("Data Quality Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Summary:**")
            
            for table_name, df in self.datasets.items():
                st.write(f"- {table_name.title()}: {len(df)} records")
        
        with col2:
            st.markdown("**Data Completeness:**")
            
            # Calculate missing data percentages
            for table_name, df in self.datasets.items():
                if not df.empty:
                    missing_pct = (df.isnull().sum() / len(df) * 100).mean()
                    st.write(f"- {table_name.title()}: {missing_pct:.1f}% missing")
        
        # Feature configuration
        if st.session_state.feature_config:
            with st.expander("Feature Configuration"):
                st.json(st.session_state.feature_config)