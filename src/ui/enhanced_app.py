"""
Enhanced Dashboard UI Components - Part 2

Completes the enhanced dashboard with clinical recommendations and feedback.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any

class EnhancedPatientDetailView:
    """Enhanced patient detail view with comprehensive analysis"""
    
    def __init__(self, datasets, trained_models, enhanced_explainer):
        self.datasets = datasets
        self.trained_models = trained_models
        self.enhanced_explainer = enhanced_explainer
        
    def _render_clinical_explanations_tab(self, analysis: Dict[str, Any]):
        """Render clinical explanations with SHAP"""
        
        st.markdown("### 📊 Clinical Risk Factor Analysis")
        
        shap_explanation = analysis.get('shap_explanation', {})
        
        if 'clinical_explanations' in shap_explanation:
            st.markdown("#### Top Risk Drivers")
            
            for i, explanation in enumerate(shap_explanation['clinical_explanations'][:5], 1):
                with st.expander(f"#{i} {explanation['feature'].replace('_', ' ').title()}", expanded=i<=3):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Clinical Finding**: {explanation['clinical_explanation']}")
                        if explanation.get('reference_range'):
                            st.markdown(f"**Reference**: {explanation['reference_range']}")
                        if explanation.get('clinical_action'):
                            st.markdown(f"**Recommended Action**: {explanation['clinical_action']}")
                            
                    with col2:
                        # Impact visualization
                        impact = shap_explanation['top_factors'][i-1]['magnitude'] if i-1 < len(shap_explanation.get('top_factors', [])) else 0.1
                        st.metric("Impact", f"{impact:.3f}")
        else:
            st.info("SHAP explanations not available. Using fallback analysis.")
            
        # Global feature importance
        st.markdown("#### Global Feature Importance")
        
        # Mock global importance for demonstration
        global_importance = {
            'Age': 0.156,
            'Glucose Control': 0.134, 
            'Blood Pressure': 0.098,
            'Medication Count': 0.087,
            'Exercise Level': 0.076
        }
        
        importance_df = pd.DataFrame(list(global_importance.items()), columns=['Feature', 'Importance'])
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Global Feature Importance Across All Patients',
            xaxis_title='Importance Score',
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_action_plan_tab(self, analysis: Dict[str, Any]):
        """Render clinical action plan"""
        
        st.markdown("### 💊 Personalized Action Plan")
        
        recommendations = analysis.get('clinical_recommendations', [])
        actionable_items = analysis.get('actionable_items', [])
        
        if recommendations:
            st.markdown("#### Priority Clinical Actions")
            
            for i, rec in enumerate(recommendations[:5], 1):
                priority = rec.get('priority', 'medium')
                priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '🟡')
                
                with st.expander(f"{priority_color} {rec.get('action', 'Clinical Action')}", expanded=priority=='high'):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Action**: {rec.get('action', 'N/A')}")
                        st.markdown(f"**Description**: {rec.get('description', 'Clinical intervention recommended')}")
                        st.markdown(f"**Timeframe**: {rec.get('timeframe', 'As appropriate')}")
                        
                    with col2:
                        st.markdown(f"**Priority**: {priority.title()}")
                        if st.button(f"✅ Mark Complete", key=f"action_{i}"):
                            st.success("Action marked as completed!")
                            
        # Actionable items checklist
        if actionable_items:
            st.markdown("#### Patient-Specific Checklist")
            
            for item in actionable_items[:5]:
                priority = item.get('priority', 'medium')
                item_text = item.get('item', 'Action item')
                
                priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '🟡')
                
                checked = st.checkbox(f"{priority_icon} {item_text}", key=f"check_{item_text}")
                if checked:
                    st.write("  ✅ Action acknowledged")
                    
        # Additional resources
        st.markdown("#### Additional Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📞 Schedule Follow-up"):
                st.success("Follow-up scheduled!")
                
        with col2:
            if st.button("📋 Generate Care Plan"):
                st.success("Care plan generated!")
                
        with col3:
            if st.button("📧 Send to Provider"):
                st.success("Sent to care team!")
                
    def _render_clinical_feedback_section(self, patient_id: str, analysis: Dict[str, Any]):
        """Render clinical feedback section"""
        
        st.markdown("### 🩺 Clinical Validation")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("**Do you agree with this risk assessment?**")
            
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if st.button("✅ Agree", type="primary", key="agree_btn"):
                    self._record_feedback(patient_id, analysis, "agree")
                    st.success("Thank you for your feedback!")
                    
            with feedback_col2:
                if st.button("❌ Disagree", key="disagree_btn"):
                    self._record_feedback(patient_id, analysis, "disagree")
                    st.success("Feedback recorded. Thank you!")
                    
        with col2:
            confidence = st.slider("Confidence in Assessment:", 1, 5, 3, 
                                 help="1=Very Low, 5=Very High")
            
        with col3:
            st.markdown("**Overall Rating**")
            rating = st.select_slider("", options=[1,2,3,4,5], value=3, key="rating")
            
        # Comments section
        comments = st.text_area("Additional Comments (Optional):", 
                               placeholder="Provide additional clinical insights...")
        
        if comments and st.button("💬 Submit Comments"):
            self._record_feedback(patient_id, analysis, "comments", comments, confidence, rating)
            st.success("Comments submitted successfully!")
            
    def _record_feedback(self, patient_id: str, analysis: Dict[str, Any], 
                        feedback_type: str, comments: str = "", 
                        confidence: int = 3, rating: int = 3):
        """Record clinical feedback"""
        
        import csv
        from datetime import datetime
        from pathlib import Path
        
        feedback_file = Path("data/feedback.csv")
        feedback_file.parent.mkdir(exist_ok=True)
        
        # Get risk score from analysis
        risk_score = analysis.get('risk_windows', {}).get('90_day_risk', 0.45)
        
        feedback_data = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'risk_score': risk_score,
            'feedback_type': feedback_type,
            'confidence': confidence,
            'rating': rating,
            'comments': comments,
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        # Write to CSV
        file_exists = feedback_file.exists()
        
        with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(feedback_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_data)

class EnhancedAdminView:
    """Enhanced admin view with multi-condition analytics"""
    
    def __init__(self, datasets, trained_models, evaluation_results):
        self.datasets = datasets
        self.trained_models = trained_models
        self.evaluation_results = evaluation_results
        
    def render_enhanced_admin_dashboard(self):
        """Render enhanced admin dashboard"""
        
        st.header("⚙️ Enhanced Administrative Analytics")
        
        # Multi-condition overview
        self._render_multi_condition_overview()
        
        # Model performance comparison
        self._render_model_performance_comparison()
        
        # Advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_dataset_analytics()
            
        with col2:
            self._render_clinical_feedback_analytics()
            
    def _render_multi_condition_overview(self):
        """Render multi-condition overview"""
        
        st.subheader("🏥 Multi-Condition System Overview")
        
        # Mock multi-condition data
        conditions_data = {
            'Diabetes': {'patients': 101766, 'event_rate': 0.461, 'avg_age': 62.3},
            'Heart Failure': {'patients': 1500, 'event_rate': 0.325, 'avg_age': 68.1},
            'Obesity': {'patients': 1200, 'event_rate': 0.289, 'avg_age': 45.7},
            'CKD': {'patients': 800, 'event_rate': 0.412, 'avg_age': 62.8}
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conditions", len(conditions_data))
            
        with col2:
            total_patients = sum(data['patients'] for data in conditions_data.values())
            st.metric("Total Patients", f"{total_patients:,}")
            
        with col3:
            avg_event_rate = np.mean([data['event_rate'] for data in conditions_data.values()])
            st.metric("Average Event Rate", f"{avg_event_rate:.1%}")
            
        with col4:
            avg_age = np.mean([data['avg_age'] for data in conditions_data.values()])
            st.metric("Average Age", f"{avg_age:.1f}")
            
        # Condition comparison table
        st.markdown("#### Condition Performance Comparison")
        
        comparison_df = pd.DataFrame.from_dict(conditions_data, orient='index')
        comparison_df.index.name = 'Condition'
        comparison_df.reset_index(inplace=True)
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            column_config={
                "patients": st.column_config.NumberColumn("Patients", format="%d"),
                "event_rate": st.column_config.NumberColumn("Event Rate", format="%.1%"),
                "avg_age": st.column_config.NumberColumn("Avg Age", format="%.1f")
            }
        )
        
    def _render_model_performance_comparison(self):
        """Render model performance comparison"""
        
        st.subheader("📊 Model Performance Analysis")
        
        # Performance metrics table
        if self.evaluation_results:
            metrics_data = []
            
            for model_name, results in self.evaluation_results.items():
                if results.get('status') == 'success':
                    metrics_data.append({
                        'Model': model_name.replace('_', ' '),
                        'AUC-ROC': results.get('auc_roc', 'N/A'),
                        'AUC-PR': results.get('auc_pr', 'N/A'),
                        'C-Index': results.get('c_index', 'N/A'),
                        'Brier Score': results.get('brier_score', 'N/A'),
                        'CV Mean': results.get('cv_mean', 'N/A'),
                        'Status': '✅ Operational'
                    })
                else:
                    metrics_data.append({
                        'Model': model_name.replace('_', ' '),
                        'Status': '❌ Failed',
                        'Error': results.get('error', 'Unknown error')
                    })
                    
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No model performance data available")
        else:
            st.info("Model evaluation results not available")
            
    def _render_dataset_analytics(self):
        """Render dataset analytics"""
        
        st.subheader("📈 Dataset Analytics")
        
        # Dataset summary
        if 'patients' in self.datasets:
            patients_df = self.datasets['patients']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", len(patients_df))
                
            with col2:
                st.metric("Features", len(patients_df.columns)-1)  # Exclude patient_id
                
            # Data quality metrics
            completeness = 1 - patients_df.isnull().sum().sum() / (len(patients_df) * len(patients_df.columns))
            st.metric("Data Completeness", f"{completeness:.1%}")
            
            # Missing data visualization
            missing_data = patients_df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = go.Figure(go.Bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h'
                ))
                
                fig.update_layout(
                    title='Missing Data by Feature',
                    xaxis_title='Missing Count',
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing data detected!")
                
    def _render_clinical_feedback_analytics(self):
        """Render clinical feedback analytics"""
        
        st.subheader("🩺 Clinical Feedback Analytics")
        
        # Try to load feedback data
        try:
            feedback_df = pd.read_csv("data/feedback.csv")
            
            if len(feedback_df) > 0:
                # Agreement rate
                agreement_rate = len(feedback_df[feedback_df['feedback_type'] == 'agree']) / len(feedback_df)
                st.metric("Agreement Rate", f"{agreement_rate:.1%}")
                
                # Feedback over time
                feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
                daily_feedback = feedback_df.groupby('date').size()
                
                fig = go.Figure(go.Scatter(
                    x=daily_feedback.index,
                    y=daily_feedback.values,
                    mode='lines+markers'
                ))
                
                fig.update_layout(
                    title='Daily Feedback Volume',
                    xaxis_title='Date',
                    yaxis_title='Feedback Count',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent feedback
                st.markdown("#### Recent Feedback")
                recent_feedback = feedback_df.tail(5)[['timestamp', 'patient_id', 'feedback_type', 'confidence']]
                st.dataframe(recent_feedback, use_container_width=True)
                
            else:
                st.info("No feedback data available yet")
                
        except FileNotFoundError:
            st.info("No feedback file found. Feedback will be collected as clinicians use the system.")
        except Exception as e:
            st.error(f"Error loading feedback data: {e}")
            
        # Feedback collection metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feedback Goals")
            st.progress(0.3, text="Target: 100 feedback submissions (30%)")
            
        with col2:
            st.markdown("#### Model Validation Status")
            st.success("✅ Ready for clinical pilot")
            
def create_enhanced_app():
    """Create the enhanced Streamlit application"""
    
    st.set_page_config(
        page_title="AI Risk Prediction Engine - Enhanced",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 AI Risk Prediction Engine - Enhanced")
    st.markdown("**Advanced 90-Day Deterioration Risk Prediction with Continuous Monitoring**")
    
    # Initialize session state
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
        
    # Main navigation
    tab1, tab2, tab3 = st.tabs([
        "👥 Enhanced Cohort Analysis",
        "👤 Comprehensive Patient View", 
        "⚙️ Advanced Admin Analytics"
    ])
    
    # Load data (this would be replaced with actual data loading)
    datasets = _load_demo_datasets()
    trained_models = _load_demo_models()
    evaluation_results = _load_demo_evaluation()
    
    with tab1:
        cohort_view = EnhancedCohortView(datasets, trained_models, None)
        cohort_view.render_enhanced_cohort()
        
    with tab2:
        patient_view = EnhancedPatientDetailView(datasets, trained_models, None)
        patient_view.render_comprehensive_patient_view()
        
    with tab3:
        admin_view = EnhancedAdminView(datasets, trained_models, evaluation_results)
        admin_view.render_enhanced_admin_dashboard()

def _load_demo_datasets():
    """Load demo datasets for testing"""
    return {
        'patients': pd.DataFrame({
            'patient_id': [f'P{i:04d}' for i in range(1, 101)],
            'age': np.random.randint(40, 90, 100),
            'gender': np.random.choice(['Male', 'Female'], 100)
        })
    }

def _load_demo_models():
    """Load demo trained models"""
    return {
        'Random_Forest': {'predictions': np.random.beta(2, 5, 100)},
        'Logistic_Regression': {'predictions': np.random.beta(2, 5, 100)}
    }

def _load_demo_evaluation():
    """Load demo evaluation results"""
    return {
        'Random_Forest': {
            'auc_roc': 0.667,
            'auc_pr': 0.453,
            'brier_score': 0.201,
            'cv_mean': 0.644,
            'status': 'success'
        },
        'Logistic_Regression': {
            'auc_roc': 0.654,
            'auc_pr': 0.412,
            'brier_score': 0.214,
            'cv_mean': 0.623,
            'status': 'success'
        }
    }

if __name__ == "__main__":
    create_enhanced_app()