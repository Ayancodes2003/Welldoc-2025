"""
Risk Explainability Module

Provides SHAP-based explanations for risk predictions in plain English.
Handles both global and patient-specific explanations.
"""
import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskExplainer:
    """Generate explanations for risk predictions"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.explanation_templates = self._load_explanation_templates()
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load templates for converting features to plain English"""
        
        return {
            # Age-related
            'age': "Patient age of {value} years",
            'age_over_65': "Patient is over 65 years old" if True else "Patient is under 65 years old",
            'age_over_75': "Patient is over 75 years old" if True else "Patient is under 75 years old", 
            'age_over_85': "Patient is over 85 years old" if True else "Patient is under 85 years old",
            
            # Vital signs
            'heart_rate_mean': "Average heart rate of {value:.0f} bpm",
            'heart_rate_std': "Heart rate variability (std: {value:.1f})",
            'systolic_bp_mean': "Average systolic blood pressure of {value:.0f} mmHg",
            'systolic_bp_std': "Blood pressure variability (std: {value:.1f})",
            'temperature_mean': "Average temperature of {value:.1f}°F",
            'oxygen_saturation_mean': "Average oxygen saturation of {value:.1f}%",
            'respiratory_rate_mean': "Average respiratory rate of {value:.0f} breaths/min",
            
            # Recent trends
            'heart_rate_recent_trend': "Recent heart rate trend ({direction})",
            'systolic_bp_recent_trend': "Recent blood pressure trend ({direction})",
            
            # Lab values
            'glucose_mean': "Average glucose level of {value:.0f} mg/dL",
            'glucose_last': "Most recent glucose level of {value:.0f} mg/dL",
            'glucose_high': "Number of high glucose readings: {value:.0f}",
            'creatinine_mean': "Average creatinine level of {value:.1f} mg/dL",
            'creatinine_last': "Most recent creatinine level of {value:.1f} mg/dL",
            'hemoglobin_mean': "Average hemoglobin level of {value:.1f} g/dL",
            'sodium_mean': "Average sodium level of {value:.0f} mEq/L",
            'potassium_mean': "Average potassium level of {value:.1f} mEq/L",
            
            # Admissions
            'admission_count': "Total number of admissions: {value:.0f}",
            'admissions_last_year': "Admissions in the last year: {value:.0f}",
            'los_mean': "Average length of stay: {value:.1f} days",
            'los_max': "Longest stay: {value:.0f} days",
            'long_stay': "Number of long stays (>7 days): {value:.0f}",
            
            # Medications
            'total_medications': "Total medications: {value:.0f}",
            'active_medications': "Currently active medications: {value:.0f}",
            'insulin_count': "Insulin medications: {value:.0f}",
            'diuretic_count': "Diuretic medications: {value:.0f}",
            'anticoagulant_count': "Blood thinner medications: {value:.0f}",
            
            # Default template
            'default': "{feature}: {value}"
        }
    
    def initialize_explainer(self, datasets: Dict[str, pd.DataFrame], sample_size: int = 100):
        """Initialize SHAP explainer with sample data"""
        
        if not self.predictor.is_trained:
            logger.warning("Model not trained, using mock explanations")
            return
        
        logger.info("Initializing SHAP explainer...")
        
        try:
            # Create features for background dataset
            features_df = self.predictor.feature_engineer.create_features(datasets)
            X_background = features_df[self.predictor.feature_engineer.feature_names].fillna(0)
            
            # Scale if needed
            if 'scaler' in self.predictor.feature_engineer.scalers:
                X_background = self.predictor.feature_engineer.scalers['scaler'].transform(X_background)
            
            # Sample for background
            if len(X_background) > sample_size:
                background_idx = np.random.choice(len(X_background), sample_size, replace=False)
                X_background = X_background.iloc[background_idx] if isinstance(X_background, pd.DataFrame) else X_background[background_idx]
            
            # Initialize appropriate SHAP explainer
            if hasattr(self.predictor.model, 'predict_proba'):
                if 'Random Forest' in self.predictor.model_type or hasattr(self.predictor.model, 'estimators_'):
                    self.explainer = shap.TreeExplainer(self.predictor.model)
                else:
                    self.explainer = shap.Explainer(self.predictor.model.predict_proba, X_background)
            else:
                self.explainer = shap.Explainer(self.predictor.model.predict, X_background)
            
            self.feature_names = self.predictor.feature_engineer.feature_names
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            self.explainer = None
    
    def explain_patient(self, patient_id: str, datasets: Dict[str, pd.DataFrame], 
                       top_n: int = 5) -> Dict[str, Any]:
        """
        Generate explanation for a specific patient
        
        Args:
            patient_id: Patient identifier
            datasets: Dictionary of DataFrames
            top_n: Number of top features to explain
            
        Returns:
            Dictionary with explanations and values
        """
        
        # Get patient features
        features_df = self.predictor.feature_engineer.create_features(datasets)
        patient_features = features_df[features_df['patient_id'] == patient_id]
        
        if patient_features.empty:
            return {'error': f'Patient {patient_id} not found'}
        
        # Get prediction
        prediction = self.predictor.predict({
            key: df[df['patient_id'] == patient_id] for key, df in datasets.items()
        })
        
        risk_score = prediction['risk_score'].iloc[0] if not prediction.empty else 0.5
        
        explanation = {
            'patient_id': patient_id,
            'risk_score': risk_score,
            'risk_category': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
            'explanations': []
        }
        
        if self.explainer is None or not self.predictor.is_trained:
            # Generate mock explanations
            explanation['explanations'] = self._generate_mock_explanations(patient_features, top_n)
            explanation['shap_available'] = False
        else:
            # Generate SHAP explanations
            try:
                explanation['explanations'] = self._generate_shap_explanations(patient_features, top_n)
                explanation['shap_available'] = True
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}, using mock explanations")
                explanation['explanations'] = self._generate_mock_explanations(patient_features, top_n)
                explanation['shap_available'] = False
        
        return explanation
    
    def _generate_shap_explanations(self, patient_features: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Generate SHAP-based explanations"""
        
        X_patient = patient_features[self.feature_names].fillna(0)
        
        # Scale if needed
        if 'scaler' in self.predictor.feature_engineer.scalers:
            X_patient = self.predictor.feature_engineer.scalers['scaler'].transform(X_patient)
        
        # Get SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(X_patient)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
        else:
            shap_values = self.explainer(X_patient).values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]  # For binary classification
        
        # Get feature contributions
        feature_contributions = []
        for i, feature in enumerate(self.feature_names):
            contribution = shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]
            value = X_patient.iloc[0, i] if isinstance(X_patient, pd.DataFrame) else X_patient[0][i]
            
            feature_contributions.append({
                'feature': feature,
                'contribution': contribution,
                'value': value,
                'direction': 'increases' if contribution > 0 else 'decreases'
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Convert to plain English
        explanations = []
        for contrib in feature_contributions[:top_n]:
            explanation_text = self._feature_to_text(contrib['feature'], contrib['value'])
            
            explanations.append({
                'feature': contrib['feature'],
                'description': explanation_text,
                'contribution': contrib['contribution'],
                'direction': contrib['direction'],
                'impact': 'High' if abs(contrib['contribution']) > 0.1 else 'Medium' if abs(contrib['contribution']) > 0.05 else 'Low'
            })
        
        return explanations
    
    def _generate_mock_explanations(self, patient_features: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Generate mock explanations when SHAP is not available"""
        
        # Get feature importances from model if available
        if hasattr(self.predictor, 'feature_importances') and self.predictor.feature_importances:
            importances = self.predictor.feature_importances
        else:
            # Use random importances
            importances = {feat: np.random.random() for feat in patient_features.columns if feat != 'patient_id'}
        
        # Create mock contributions
        explanations = []
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:top_n]:
            if feature in patient_features.columns:
                value = patient_features[feature].iloc[0]
                
                # Mock contribution based on importance and value
                contribution = importance * np.random.normal(0, 0.1)
                
                explanation_text = self._feature_to_text(feature, value)
                
                explanations.append({
                    'feature': feature,
                    'description': explanation_text,
                    'contribution': contribution,
                    'direction': 'increases' if contribution > 0 else 'decreases',
                    'impact': 'High' if abs(contribution) > 0.1 else 'Medium' if abs(contribution) > 0.05 else 'Low'
                })
        
        return explanations
    
    def _feature_to_text(self, feature: str, value: float) -> str:
        """Convert feature name and value to plain English description"""
        
        # Handle trend features
        if 'trend' in feature:
            direction = 'increasing' if value > 0 else 'decreasing' if value < 0 else 'stable'
            base_feature = feature.replace('_recent_trend', '').replace('_trend', '')
            
            if base_feature in self.explanation_templates:
                template = self.explanation_templates[base_feature]
                return template.format(direction=direction)
        
        # Handle boolean features
        if feature.endswith(('_over_65', '_over_75', '_over_85')) and value in [0, 1]:
            template = self.explanation_templates.get(feature, self.explanation_templates['default'])
            if isinstance(template, str) and '{value}' not in template:
                return template
        
        # Handle regular features
        if feature in self.explanation_templates:
            template = self.explanation_templates[feature]
            try:
                return template.format(value=value)
            except (KeyError, ValueError):
                return template
        
        # Default formatting
        feature_display = feature.replace('_', ' ').title()
        
        if isinstance(value, float):
            if value.is_integer():
                return f"{feature_display}: {int(value)}"
            else:
                return f"{feature_display}: {value:.2f}"
        else:
            return f"{feature_display}: {value}"
    
    def get_global_explanations(self, datasets: Dict[str, pd.DataFrame], top_n: int = 10) -> Dict[str, Any]:
        """Generate global feature importance explanations"""
        
        global_explanations = {
            'feature_importance': [],
            'summary': ''
        }
        
        if hasattr(self.predictor, 'feature_importances') and self.predictor.feature_importances:
            # Sort features by importance
            sorted_features = sorted(
                self.predictor.feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            total_importance = sum(self.predictor.feature_importances.values())
            
            for feature, importance in sorted_features[:top_n]:
                percentage = (importance / total_importance * 100) if total_importance > 0 else 0
                
                global_explanations['feature_importance'].append({
                    'feature': feature,
                    'importance': importance,
                    'percentage': percentage,
                    'description': self._get_feature_description(feature)
                })
            
            # Generate summary
            top_3_features = [item['description'] for item in global_explanations['feature_importance'][:3]]
            global_explanations['summary'] = (
                f"The most important factors for predicting deterioration risk are: "
                f"{', '.join(top_3_features[:2])}, and {top_3_features[2] if len(top_3_features) > 2 else 'other clinical indicators'}."
            )
        
        return global_explanations
    
    def _get_feature_description(self, feature: str) -> str:
        """Get a human-readable description of a feature"""
        
        descriptions = {
            'age': 'Patient age',
            'age_over_65': 'Age over 65',
            'age_over_75': 'Age over 75',
            'age_over_85': 'Age over 85',
            'heart_rate_mean': 'Average heart rate',
            'heart_rate_std': 'Heart rate variability',
            'systolic_bp_mean': 'Average blood pressure',
            'systolic_bp_std': 'Blood pressure variability',
            'glucose_mean': 'Average blood glucose',
            'creatinine_mean': 'Kidney function (creatinine)',
            'admission_count': 'Number of previous admissions',
            'admissions_last_year': 'Recent hospitalization frequency',
            'total_medications': 'Number of medications',
            'los_mean': 'Average length of stay'
        }
        
        return descriptions.get(feature, feature.replace('_', ' ').title())
    
    def generate_recommendations(self, explanations: List[Dict]) -> List[str]:
        """Generate clinical recommendations based on risk factors"""
        
        recommendations = []
        
        for explanation in explanations[:3]:  # Top 3 factors
            feature = explanation['feature']
            direction = explanation['direction']
            
            if 'age' in feature and direction == 'increases':
                recommendations.append("Consider more frequent monitoring due to advanced age")
            
            elif 'heart_rate' in feature:
                if direction == 'increases':
                    recommendations.append("Monitor cardiac status and consider cardiology consultation")
                else:
                    recommendations.append("Evaluate for bradycardia and medication effects")
            
            elif 'bp' in feature or 'pressure' in feature:
                if direction == 'increases':
                    recommendations.append("Review blood pressure management and medication compliance")
                else:
                    recommendations.append("Monitor for hypotension and adjust medications as needed")
            
            elif 'glucose' in feature and direction == 'increases':
                recommendations.append("Review diabetes management and consider endocrinology consultation")
            
            elif 'creatinine' in feature and direction == 'increases':
                recommendations.append("Monitor kidney function and adjust medications for renal impairment")
            
            elif 'admission' in feature and direction == 'increases':
                recommendations.append("Consider discharge planning and care coordination to reduce readmissions")
            
            elif 'medication' in feature and direction == 'increases':
                recommendations.append("Review medication list for interactions and simplification opportunities")
        
        # Add default recommendations if none generated
        if not recommendations:
            recommendations = [
                "Continue current monitoring and treatment plan",
                "Follow up as scheduled with primary care provider",
                "Contact healthcare team if symptoms worsen"
            ]
        
        return recommendations[:5]  # Limit to 5 recommendations

if __name__ == "__main__":
    # Test the explainer
    from data_loader import DynamicDataLoader
    from baseline_models import RiskPredictor
    
    # Load demo data
    loader = DynamicDataLoader()
    datasets = loader.create_demo_dataset()
    feature_config = loader.get_feature_config()
    
    # Train model
    predictor = RiskPredictor(model_type="Random Forest")
    metrics = predictor.train(datasets, feature_config)
    
    # Initialize explainer
    explainer = RiskExplainer(predictor)
    explainer.initialize_explainer(datasets)
    
    # Test patient explanation
    patient_id = datasets['patients']['patient_id'].iloc[0]
    explanation = explainer.explain_patient(patient_id, datasets)
    
    print(f"Explanation for patient {patient_id}:")
    print(f"Risk score: {explanation['risk_score']:.3f}")
    print("Top risk factors:")
    for exp in explanation['explanations']:
        print(f"- {exp['description']} ({exp['direction']} risk)")
    
    # Test global explanations
    global_exp = explainer.get_global_explanations(datasets)
    print(f"\nGlobal model summary: {global_exp['summary']}")