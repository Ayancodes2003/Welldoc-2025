"""
Feedback Widget and Management System

Implements 'Agree/Disagree' feedback widget for predictions
and saves responses to feedback.csv for trust and model improvement.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages prediction feedback collection and storage"""
    
    def __init__(self, feedback_file: str = "feedback.csv"):
        self.feedback_file = Path(feedback_file)
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> pd.DataFrame:
        """Load existing feedback data"""
        
        if self.feedback_file.exists():
            try:
                df = pd.read_csv(self.feedback_file)
                logger.info(f"Loaded {len(df)} feedback records")
                return df
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                return self._create_empty_feedback_df()
        else:
            logger.info("No existing feedback file found, creating new one")
            return self._create_empty_feedback_df()
    
    def _create_empty_feedback_df(self) -> pd.DataFrame:
        """Create empty feedback DataFrame with proper schema"""
        
        return pd.DataFrame(columns=[
            'timestamp', 'patient_id', 'prediction_id', 'model_name',
            'risk_score', 'risk_category', 'feedback', 'user_id',
            'session_id', 'comments', 'confidence_level'
        ])
    
    def save_feedback(self, patient_id: str, prediction_data: Dict[str, Any], 
                     feedback: str, user_id: str = "anonymous", 
                     comments: str = "", confidence_level: Optional[int] = None) -> bool:
        """
        Save feedback for a prediction
        
        Args:
            patient_id: Patient identifier
            prediction_data: Dictionary with prediction information
            feedback: 'agree' or 'disagree'
            user_id: User providing feedback
            comments: Optional comments
            confidence_level: Confidence level (1-5 scale)
            
        Returns:
            Success status
        """
        
        try:
            # Create feedback record
            feedback_record = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id,
                'prediction_id': prediction_data.get('prediction_id', f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'model_name': prediction_data.get('model_name', 'Unknown'),
                'risk_score': prediction_data.get('risk_score', np.nan),
                'risk_category': prediction_data.get('risk_category', 'Unknown'),
                'feedback': feedback.lower(),
                'user_id': user_id,
                'session_id': st.session_state.get('session_id', 'unknown'),
                'comments': comments,
                'confidence_level': confidence_level
            }
            
            # Add to feedback data
            new_row = pd.DataFrame([feedback_record])
            self.feedback_data = pd.concat([self.feedback_data, new_row], ignore_index=True)
            
            # Save to file
            self.feedback_data.to_csv(self.feedback_file, index=False)
            
            logger.info(f"Feedback saved for patient {patient_id}: {feedback}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get aggregated feedback summary statistics"""
        
        if len(self.feedback_data) == 0:
            return {
                'total_feedback': 0,
                'agreement_rate': 0.0,
                'feedback_by_risk': {},
                'feedback_by_model': {},
                'recent_feedback': []
            }
        
        try:
            # Overall statistics
            total_feedback = len(self.feedback_data)
            agree_count = len(self.feedback_data[self.feedback_data['feedback'] == 'agree'])
            agreement_rate = agree_count / total_feedback if total_feedback > 0 else 0.0
            
            # Feedback by risk category
            feedback_by_risk = {}
            if 'risk_category' in self.feedback_data.columns:
                for risk_cat in self.feedback_data['risk_category'].unique():
                    if pd.notna(risk_cat):
                        risk_feedback = self.feedback_data[self.feedback_data['risk_category'] == risk_cat]
                        risk_agree = len(risk_feedback[risk_feedback['feedback'] == 'agree'])
                        risk_total = len(risk_feedback)
                        feedback_by_risk[risk_cat] = {
                            'total': risk_total,
                            'agree': risk_agree,
                            'disagree': risk_total - risk_agree,
                            'agreement_rate': risk_agree / risk_total if risk_total > 0 else 0.0
                        }
            
            # Feedback by model
            feedback_by_model = {}
            if 'model_name' in self.feedback_data.columns:
                for model in self.feedback_data['model_name'].unique():
                    if pd.notna(model):
                        model_feedback = self.feedback_data[self.feedback_data['model_name'] == model]
                        model_agree = len(model_feedback[model_feedback['feedback'] == 'agree'])
                        model_total = len(model_feedback)
                        feedback_by_model[model] = {
                            'total': model_total,
                            'agree': model_agree,
                            'disagree': model_total - model_agree,
                            'agreement_rate': model_agree / model_total if model_total > 0 else 0.0
                        }
            
            # Recent feedback (last 10)
            recent_feedback = self.feedback_data.sort_values('timestamp', ascending=False).head(10).to_dict('records')
            
            return {
                'total_feedback': total_feedback,
                'agreement_rate': agreement_rate,
                'agree_count': agree_count,
                'disagree_count': total_feedback - agree_count,
                'feedback_by_risk': feedback_by_risk,
                'feedback_by_model': feedback_by_model,
                'recent_feedback': recent_feedback
            }
            
        except Exception as e:
            logger.error(f"Error generating feedback summary: {e}")
            return {
                'total_feedback': 0,
                'agreement_rate': 0.0,
                'error': str(e)
            }

class FeedbackWidget:
    """Streamlit widget for collecting prediction feedback"""
    
    def __init__(self, feedback_manager: FeedbackManager):
        self.feedback_manager = feedback_manager
    
    def render_simple_feedback(self, patient_id: str, prediction_data: Dict[str, Any], 
                             key_suffix: str = "") -> Optional[str]:
        """
        Render simple agree/disagree feedback widget
        
        Args:
            patient_id: Patient ID
            prediction_data: Prediction information
            key_suffix: Unique suffix for widget keys
            
        Returns:
            Feedback response or None
        """
        
        st.markdown("### 🤝 Clinician Feedback")
        st.markdown("*Do you agree with this risk assessment?*")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        feedback_given = False
        feedback_response = None
        
        with col1:
            if st.button("✅ Agree", key=f"agree_{patient_id}_{key_suffix}", 
                        help="I agree with this risk assessment"):
                feedback_response = "agree"
                feedback_given = True
        
        with col2:
            if st.button("❌ Disagree", key=f"disagree_{patient_id}_{key_suffix}",
                        help="I disagree with this risk assessment"):
                feedback_response = "disagree"
                feedback_given = True
        
        if feedback_given:
            # Show success message and save feedback
            success = self.feedback_manager.save_feedback(
                patient_id=patient_id,
                prediction_data=prediction_data,
                feedback=feedback_response,
                user_id=st.session_state.get('user_id', 'anonymous')
            )
            
            if success:
                st.success(f"Thank you for your feedback! ({feedback_response.title()})")
                st.rerun()  # Refresh to show updated state
            else:
                st.error("Error saving feedback. Please try again.")
        
        return feedback_response
    
    def render_detailed_feedback(self, patient_id: str, prediction_data: Dict[str, Any],
                               key_suffix: str = "") -> Dict[str, Any]:
        """
        Render detailed feedback form with comments and confidence
        """
        
        st.markdown("### 📝 Detailed Feedback")
        
        with st.form(f"feedback_form_{patient_id}_{key_suffix}"):
            # Feedback selection
            feedback = st.radio(
                "Do you agree with this risk assessment?",
                options=["agree", "disagree"],
                format_func=lambda x: "✅ Agree" if x == "agree" else "❌ Disagree",
                key=f"feedback_radio_{patient_id}_{key_suffix}"
            )
            
            # Confidence level
            confidence = st.slider(
                "How confident are you in your assessment?",
                min_value=1, max_value=5, value=3,
                help="1 = Not confident, 5 = Very confident",
                key=f"confidence_{patient_id}_{key_suffix}"
            )
            
            # Comments
            comments = st.text_area(
                "Additional comments (optional):",
                placeholder="Please provide any additional context or reasoning...",
                key=f"comments_{patient_id}_{key_suffix}"
            )
            
            # Submit button
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                success = self.feedback_manager.save_feedback(
                    patient_id=patient_id,
                    prediction_data=prediction_data,
                    feedback=feedback,
                    comments=comments,
                    confidence_level=confidence,
                    user_id=st.session_state.get('user_id', 'anonymous')
                )
                
                if success:
                    st.success("Thank you for your detailed feedback!")
                    st.rerun()
                else:
                    st.error("Error saving feedback. Please try again.")
                
                return {
                    'feedback': feedback,
                    'confidence': confidence,
                    'comments': comments,
                    'success': success
                }
        
        return {}
    
    def render_feedback_summary_widget(self) -> None:
        """Render feedback summary for admin dashboard"""
        
        st.markdown("### 📊 Feedback Summary")
        
        summary = self.feedback_manager.get_feedback_summary()
        
        if summary['total_feedback'] == 0:
            st.info("No feedback collected yet.")
            return
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Feedback",
                value=summary['total_feedback']
            )
        
        with col2:
            st.metric(
                label="Agreement Rate",
                value=f"{summary['agreement_rate']:.1%}"
            )
        
        with col3:
            st.metric(
                label="Agree / Disagree",
                value=f"{summary['agree_count']} / {summary['disagree_count']}"
            )
        
        # Feedback by risk category
        if summary['feedback_by_risk']:
            st.markdown("#### Feedback by Risk Category")
            risk_data = []
            for risk_cat, data in summary['feedback_by_risk'].items():
                risk_data.append({
                    'Risk Category': risk_cat,
                    'Total': data['total'],
                    'Agree': data['agree'],
                    'Disagree': data['disagree'],
                    'Agreement Rate': f"{data['agreement_rate']:.1%}"
                })
            
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
        
        # Recent feedback
        if summary['recent_feedback']:
            st.markdown("#### Recent Feedback")
            recent_df = pd.DataFrame(summary['recent_feedback'])
            
            # Select relevant columns
            display_cols = ['timestamp', 'patient_id', 'risk_category', 'feedback', 'comments']
            available_cols = [col for col in display_cols if col in recent_df.columns]
            
            if available_cols:
                st.dataframe(
                    recent_df[available_cols].head(5),
                    use_container_width=True
                )

# Initialize feedback system in session state
def initialize_feedback_system():
    """Initialize feedback system in Streamlit session state"""
    
    if 'feedback_manager' not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()
    
    if 'feedback_widget' not in st.session_state:
        st.session_state.feedback_widget = FeedbackWidget(st.session_state.feedback_manager)
    
    # Generate session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

if __name__ == "__main__":
    # Test feedback system
    feedback_manager = FeedbackManager("test_feedback.csv")
    
    # Test saving feedback
    test_prediction = {
        'model_name': 'Random Forest',
        'risk_score': 0.75,
        'risk_category': 'High'
    }
    
    success = feedback_manager.save_feedback(
        patient_id="P001",
        prediction_data=test_prediction,
        feedback="agree",
        comments="Matches clinical assessment"
    )
    
    print(f"Feedback saved: {success}")
    
    # Test summary
    summary = feedback_manager.get_feedback_summary()
    print(f"Summary: {summary}")