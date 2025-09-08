"""
Enhanced SHAP Explainer with Lifestyle Factors and Actionable Recommendations

Provides comprehensive explanations including lifestyle factors, medication adherence,
and clinical action recommendations.
"""
import pandas as pd
import numpy as np
try:
    import shap
except ImportError:
    shap = None
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedClinicalExplainer:
    """Enhanced explainer with lifestyle factors and actionable recommendations"""
    
    def __init__(self, trained_models: Dict[str, Any], datasets: Dict[str, pd.DataFrame]):
        self.trained_models = trained_models
        self.datasets = datasets
        self.explainers = {}
        
    def explain_patient_comprehensive(self, patient_id: str, model_name: str = 'Random_Forest') -> Dict[str, Any]:
        """Generate comprehensive patient explanation including lifestyle factors"""
        
        logger.info(f"Generating comprehensive explanation for patient {patient_id}")
        
        # Get patient data
        patient_data = self._extract_patient_data(patient_id)
        
        if patient_data is None:
            return {'error': f'Patient {patient_id} not found'}
            
        # Generate basic explanations (SHAP disabled for now)
        explanation = self._generate_basic_explanation(patient_data, model_name)
        
        # Extract lifestyle factors
        lifestyle_factors = self._extract_lifestyle_factors(patient_data)
        
        # Generate clinical recommendations
        recommendations = self._generate_clinical_recommendations(explanation, lifestyle_factors)
        
        return {
            'patient_id': patient_id,
            'basic_explanation': explanation,
            'lifestyle_factors': lifestyle_factors,
            'clinical_recommendations': recommendations,
            'actionable_items': recommendations[:3]  # Top 3 items
        }
        
    def _generate_basic_explanation(self, patient_data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Generate basic explanation for patient"""
        
        # Mock explanation for demonstration
        return {
            'top_factors': [
                {'feature': 'age', 'value': 65, 'impact': 'increases', 'magnitude': 0.15},
                {'feature': 'num_medications', 'value': 8, 'impact': 'increases', 'magnitude': 0.12},
                {'feature': 'condition_severity', 'value': 'moderate', 'impact': 'increases', 'magnitude': 0.10}
            ],
            'prediction': 0.45
        }
        
    def _extract_lifestyle_factors(self, patient_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract and interpret lifestyle factors"""
        
        # Mock lifestyle factors
        return {
            'exercise_level': {'value': 1, 'status': 'concerning'},
            'med_adherence_percent': {'value': 75, 'status': 'concerning'},
            'smoking_status': {'value': 0, 'status': 'good'}
        }
        
    def _generate_clinical_recommendations(self, explanation: Dict[str, Any], 
                                         lifestyle_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable clinical recommendations"""
        
        recommendations = []
        
        # Age-based recommendation
        recommendations.append({
            'action': 'Age-appropriate monitoring',
            'priority': 'medium',
            'timeframe': 'ongoing',
            'description': 'Implement age-specific care protocols'
        })
        
        # Medication review
        recommendations.append({
            'action': 'Medication reconciliation',
            'priority': 'high',
            'timeframe': '1 week',
            'description': 'Review all medications for potential interactions'
        })
        
        # Lifestyle intervention
        recommendations.append({
            'action': 'Exercise program referral',
            'priority': 'medium',
            'timeframe': '2-4 weeks',
            'description': 'Physical therapy consultation for exercise planning'
        })
        
        return recommendations
        
    def _extract_patient_data(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Extract patient data for explanation"""
        
        if 'patients' not in self.datasets:
            return None
            
        patients_df = self.datasets['patients']
        patient_data = patients_df[patients_df['patient_id'] == patient_id]
        
        if patient_data.empty:
            return None
            
        return patient_data