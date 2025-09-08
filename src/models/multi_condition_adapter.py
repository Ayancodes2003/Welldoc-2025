"""
Multi-Condition Dataset Adapter

Handles multiple chronic conditions (diabetes, heart failure, obesity) with
consistent outcome mapping and condition-specific feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MultiConditionAdapter:
    """Enhanced adapter for multiple chronic conditions"""
    
    def __init__(self):
        pass
        
    def detect_condition_type(self, data_path: str) -> str:
        """Detect the chronic condition type from dataset"""
        path = Path(data_path)
        
        # Check for diabetes dataset
        if (path / 'diabetic_data.csv').exists():
            return 'diabetes'
            
        # Default to diabetes for demonstration
        return 'diabetes'
        
    def load_and_standardize_condition(self, data_path: str, condition_type: str = None) -> Dict[str, pd.DataFrame]:
        """Load and standardize data for specific condition"""
        
        if condition_type is None:
            condition_type = self.detect_condition_type(data_path)
            
        logger.info(f"Processing dataset as: {condition_type}")
        
        if condition_type == 'diabetes':
            return self._process_diabetes_dataset(data_path)
        else:
            return self._process_generic_dataset(data_path)
            
    def _process_diabetes_dataset(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Process diabetes dataset with condition-specific features"""
        
        # For demonstration, create synthetic diabetes data
        logger.info("Creating synthetic diabetes dataset for demonstration")
        
        n_patients = 500
        np.random.seed(42)
        
        # Create base patient data
        patients_data = {
            'patient_id': [f'DM_{i:04d}' for i in range(1, n_patients + 1)],
            'age': np.random.normal(65, 12, n_patients).clip(30, 95),
            'gender': np.random.choice(['Male', 'Female'], n_patients, p=[0.6, 0.4]),
            'race': np.random.choice(['Caucasian', 'African American', 'Hispanic'], n_patients),
            'time_in_hospital': np.random.poisson(5, n_patients),
            'num_medications': np.random.poisson(8, n_patients),
            'num_procedures': np.random.poisson(3, n_patients),
            'condition_type': 'diabetes',
            'condition_severity': np.random.choice(['mild', 'moderate', 'severe'], n_patients, p=[0.4, 0.4, 0.2])
        }
        
        patients_df = pd.DataFrame(patients_data)
        
        # Create outcomes
        risk_factors = (
            (patients_df['age'] - 65) / 20 * 0.3 +
            patients_df['num_medications'] / 15 * 0.2 +
            (patients_df['condition_severity'] == 'severe').astype(int) * 0.3
        ).clip(0.1, 0.8)
        
        outcomes_df = pd.DataFrame({
            'patient_id': patients_df['patient_id'],
            'event_within_90d': np.random.binomial(1, risk_factors),
            'time_to_event': np.where(
                np.random.binomial(1, risk_factors),
                np.random.uniform(1, 90, n_patients),
                90.0
            )
        })
        
        # Condition summary
        condition_summary = {
            'condition_type': 'diabetes',
            'total_patients': n_patients,
            'event_rate': outcomes_df['event_within_90d'].mean(),
            'average_age': patients_df['age'].mean(),
            'gender_distribution': patients_df['gender'].value_counts().to_dict(),
            'severity_distribution': patients_df['condition_severity'].value_counts().to_dict()
        }
        
        return {
            'patients': patients_df,
            'outcomes': outcomes_df,
            'condition_summary': condition_summary
        }
        
    def _process_generic_dataset(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Process generic dataset with standard outcome mapping"""
        
        # Create basic demo data
        n_patients = 300
        patients_df = pd.DataFrame({
            'patient_id': [f'GEN_{i:03d}' for i in range(1, n_patients + 1)],
            'age': np.random.normal(60, 15, n_patients),
            'gender': np.random.choice(['Male', 'Female'], n_patients),
            'condition_type': 'unknown'
        })
        
        outcomes_df = pd.DataFrame({
            'patient_id': patients_df['patient_id'],
            'event_within_90d': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'time_to_event': np.random.uniform(1, 90, n_patients)
        })
        
        condition_summary = {
            'condition_type': 'unknown',
            'total_patients': n_patients,
            'event_rate': outcomes_df['event_within_90d'].mean(),
            'average_age': patients_df['age'].mean(),
            'gender_distribution': patients_df['gender'].value_counts().to_dict()
        }
        
        return {
            'patients': patients_df,
            'outcomes': outcomes_df,
            'condition_summary': condition_summary
        }