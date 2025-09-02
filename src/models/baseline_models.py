"""
Baseline Risk Prediction Models

Dynamic models that adapt to available features automatically.
Supports Logistic Regression, Random Forest, and mock predictions.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from typing import Dict, List, Optional, Tuple, Any
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Dynamically engineers features from available data"""
    
    def __init__(self, feature_config: Dict[str, Any]):
        self.feature_config = feature_config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def create_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features from available datasets
        
        Args:
            datasets: Dictionary of DataFrames (patients, vitals, labs, etc.)
            
        Returns:
            Feature matrix DataFrame
        """
        
        logger.info("Creating features from available data...")
        
        # Start with patient demographics
        if 'patients' in datasets:
            features_df = self._create_demographic_features(datasets['patients'])
        else:
            # Extract unique patients from other tables
            patient_ids = set()
            for df in datasets.values():
                if 'patient_id' in df.columns:
                    patient_ids.update(df['patient_id'].unique())
            
            features_df = pd.DataFrame({'patient_id': list(patient_ids)})
        
        # Add vital signs features
        if 'vitals' in datasets:
            vital_features = self._create_vital_features(datasets['vitals'])
            features_df = features_df.merge(vital_features, on='patient_id', how='left')
        
        # Add lab features
        if 'labs' in datasets:
            lab_features = self._create_lab_features(datasets['labs'])
            features_df = features_df.merge(lab_features, on='patient_id', how='left')
        
        # Add admission features
        if 'admissions' in datasets:
            admission_features = self._create_admission_features(datasets['admissions'])
            features_df = features_df.merge(admission_features, on='patient_id', how='left')
        
        # Add medication features
        if 'medications' in datasets:
            med_features = self._create_medication_features(datasets['medications'])
            features_df = features_df.merge(med_features, on='patient_id', how='left')
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns if col != 'patient_id']
        
        logger.info(f"Created {len(self.feature_names)} features for {len(features_df)} patients")
        
        return features_df
    
    def _create_demographic_features(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from patient demographics"""
        
        features = patients_df[['patient_id']].copy()
        
        # Age features
        if 'age' in patients_df.columns:
            features['age'] = patients_df['age']
            features['age_over_65'] = (patients_df['age'] >= 65).astype(int)
            features['age_over_75'] = (patients_df['age'] >= 75).astype(int)
            features['age_over_85'] = (patients_df['age'] >= 85).astype(int)
        
        # Gender
        if 'gender' in patients_df.columns:
            if 'gender' not in self.encoders:
                self.encoders['gender'] = LabelEncoder()
                features['gender_encoded'] = self.encoders['gender'].fit_transform(patients_df['gender'].fillna('Unknown'))
            else:
                features['gender_encoded'] = self.encoders['gender'].transform(patients_df['gender'].fillna('Unknown'))
        
        # Ethnicity if available
        if 'ethnicity' in patients_df.columns:
            if 'ethnicity' not in self.encoders:
                self.encoders['ethnicity'] = LabelEncoder()
                features['ethnicity_encoded'] = self.encoders['ethnicity'].fit_transform(patients_df['ethnicity'].fillna('Unknown'))
            else:
                features['ethnicity_encoded'] = self.encoders['ethnicity'].transform(patients_df['ethnicity'].fillna('Unknown'))
        
        return features
    
    def _create_vital_features(self, vitals_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from vitals data"""
        
        # Get vital sign columns
        vital_cols = self.feature_config.get('vital_features', [])
        available_vitals = [col for col in vital_cols if col in vitals_df.columns]
        
        if not available_vitals:
            # Auto-detect vital columns
            numeric_cols = vitals_df.select_dtypes(include=[np.number]).columns
            available_vitals = [col for col in numeric_cols if col not in ['patient_id']]
        
        features = []
        
        for patient_id in vitals_df['patient_id'].unique():
            patient_vitals = vitals_df[vitals_df['patient_id'] == patient_id]
            
            patient_features = {'patient_id': patient_id}
            
            for vital in available_vitals:
                if vital in patient_vitals.columns:
                    values = patient_vitals[vital].dropna()
                    
                    if len(values) > 0:
                        # Aggregated statistics
                        patient_features[f'{vital}_mean'] = values.mean()
                        patient_features[f'{vital}_std'] = values.std() if len(values) > 1 else 0
                        patient_features[f'{vital}_min'] = values.min()
                        patient_features[f'{vital}_max'] = values.max()
                        patient_features[f'{vital}_range'] = values.max() - values.min()
                        
                        # Recent values (last 7 days if date info available)
                        if len(values) > 7:
                            recent_values = values.tail(7)
                            patient_features[f'{vital}_recent_mean'] = recent_values.mean()
                            patient_features[f'{vital}_recent_trend'] = recent_values.iloc[-1] - recent_values.iloc[0]
                        
                        # Count of measurements
                        patient_features[f'{vital}_count'] = len(values)
                    else:
                        # Fill with defaults if no data
                        for suffix in ['_mean', '_std', '_min', '_max', '_range', '_recent_mean', '_recent_trend', '_count']:
                            patient_features[f'{vital}{suffix}'] = np.nan
            
            features.append(patient_features)
        
        return pd.DataFrame(features)
    
    def _create_lab_features(self, labs_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from lab data"""
        
        # Get lab columns
        lab_cols = self.feature_config.get('lab_features', [])
        available_labs = [col for col in lab_cols if col in labs_df.columns]
        
        if not available_labs:
            # Auto-detect lab columns
            numeric_cols = labs_df.select_dtypes(include=[np.number]).columns
            available_labs = [col for col in numeric_cols if col not in ['patient_id']]
        
        features = []
        
        for patient_id in labs_df['patient_id'].unique():
            patient_labs = labs_df[labs_df['patient_id'] == patient_id]
            
            patient_features = {'patient_id': patient_id}
            
            for lab in available_labs:
                if lab in patient_labs.columns:
                    values = patient_labs[lab].dropna()
                    
                    if len(values) > 0:
                        # Aggregated statistics
                        patient_features[f'{lab}_mean'] = values.mean()
                        patient_features[f'{lab}_last'] = values.iloc[-1] if len(values) > 0 else np.nan
                        patient_features[f'{lab}_min'] = values.min()
                        patient_features[f'{lab}_max'] = values.max()
                        
                        # Abnormal values (simplified thresholds)
                        if lab.lower() in ['glucose']:
                            patient_features[f'{lab}_high'] = (values > 140).sum()
                        elif lab.lower() in ['creatinine']:
                            patient_features[f'{lab}_high'] = (values > 1.5).sum()
                        
                        patient_features[f'{lab}_count'] = len(values)
                    else:
                        for suffix in ['_mean', '_last', '_min', '_max', '_high', '_count']:
                            patient_features[f'{lab}{suffix}'] = np.nan
            
            features.append(patient_features)
        
        return pd.DataFrame(features)
    
    def _create_admission_features(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from admission data"""
        
        features = []
        
        for patient_id in admissions_df['patient_id'].unique():
            patient_admissions = admissions_df[admissions_df['patient_id'] == patient_id]
            
            patient_features = {
                'patient_id': patient_id,
                'admission_count': len(patient_admissions)
            }
            
            # Length of stay features
            if 'length_of_stay' in patient_admissions.columns:
                los_values = patient_admissions['length_of_stay'].dropna()
                if len(los_values) > 0:
                    patient_features['los_mean'] = los_values.mean()
                    patient_features['los_max'] = los_values.max()
                    patient_features['long_stay'] = (los_values > 7).sum()
            
            # Recent admissions
            if 'admission_date' in patient_admissions.columns:
                admission_dates = pd.to_datetime(patient_admissions['admission_date'])
                recent_date = admission_dates.max() - timedelta(days=365)
                recent_admissions = admission_dates[admission_dates >= recent_date]
                patient_features['admissions_last_year'] = len(recent_admissions)
            
            features.append(patient_features)
        
        return pd.DataFrame(features)
    
    def _create_medication_features(self, medications_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from medication data"""
        
        features = []
        
        for patient_id in medications_df['patient_id'].unique():
            patient_meds = medications_df[medications_df['patient_id'] == patient_id]
            
            patient_features = {
                'patient_id': patient_id,
                'total_medications': len(patient_meds)
            }
            
            # Active medications
            if 'active' in patient_meds.columns:
                active_meds = patient_meds[patient_meds['active'] == True]
                patient_features['active_medications'] = len(active_meds)
            
            # High-risk medication classes (simplified)
            if 'medication' in patient_meds.columns:
                med_names = patient_meds['medication'].str.lower()
                
                # Count specific medication types
                patient_features['insulin_count'] = med_names.str.contains('insulin', na=False).sum()
                patient_features['diuretic_count'] = med_names.str.contains('furosemide|lasix', na=False).sum()
                patient_features['anticoagulant_count'] = med_names.str.contains('warfarin|heparin', na=False).sum()
            
            features.append(patient_features)
        
        return pd.DataFrame(features)
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix"""
        
        # Separate numeric and categorical columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if col != 'patient_id':
                features_df[col] = features_df[col].fillna(features_df[col].median())
        
        return features_df

class RiskPredictor:
    """Main risk prediction model with dynamic adaptation"""
    
    def __init__(self, model_type: str = "Random Forest", demo_mode: bool = False):
        self.model_type = model_type
        self.demo_mode = demo_mode
        self.model = None
        self.feature_engineer = None
        self.is_trained = False
        self.feature_importances = None
        self.training_metrics = {}
    
    def train(self, datasets: Dict[str, pd.DataFrame], feature_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the risk prediction model
        
        Args:
            datasets: Dictionary of DataFrames
            feature_config: Feature configuration from data loader
            
        Returns:
            Training metrics
        """
        
        logger.info(f"Training {self.model_type} model...")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # Create features
        features_df = self.feature_engineer.create_features(datasets)
        
        # Get outcomes
        if 'outcomes' in datasets:
            outcomes_df = datasets['outcomes']
            outcome_col = feature_config.get('outcome_column', 'deterioration_90d')
        else:
            # Create mock outcomes for demo
            outcomes_df = pd.DataFrame({
                'patient_id': features_df['patient_id'],
                outcome_col: np.random.choice([0, 1], len(features_df), p=[0.7, 0.3])
            })
        
        # Merge features with outcomes
        train_df = features_df.merge(outcomes_df[['patient_id', outcome_col]], on='patient_id', how='inner')
        
        # Prepare training data
        X = train_df[self.feature_engineer.feature_names]
        y = train_df[outcome_col]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Scale features if using Logistic Regression
        if 'Logistic' in self.model_type:
            if 'scaler' not in self.feature_engineer.scalers:
                self.feature_engineer.scalers['scaler'] = StandardScaler()
                X = self.feature_engineer.scalers['scaler'].fit_transform(X)
            else:
                X = self.feature_engineer.scalers['scaler'].transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        if self.model_type == "Logistic Regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.training_metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'accuracy': (y_pred == y_test).mean(),
            'n_features': len(self.feature_engineer.feature_names),
            'n_samples': len(train_df)
        }
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(self.feature_engineer.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importances = dict(zip(self.feature_engineer.feature_names, np.abs(self.model.coef_[0])))
        
        self.is_trained = True
        
        logger.info(f"Model trained successfully. AUC-ROC: {self.training_metrics['auc_roc']:.3f}")
        
        return self.training_metrics
    
    def predict(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate risk predictions for patients
        
        Args:
            datasets: Dictionary of DataFrames
            
        Returns:
            DataFrame with patient_id and risk_score
        """
        
        if self.demo_mode:
            return self._generate_mock_predictions(datasets)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features_df = self.feature_engineer.create_features(datasets)
        
        # Prepare features
        X = features_df[self.feature_engineer.feature_names].fillna(0)
        
        # Scale if needed
        if 'scaler' in self.feature_engineer.scalers:
            X = self.feature_engineer.scalers['scaler'].transform(X)
        
        # Predict
        risk_scores = self.model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'patient_id': features_df['patient_id'],
            'risk_score': risk_scores,
            'risk_category': pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        })
        
        return results
    
    def _generate_mock_predictions(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate realistic mock predictions for demo mode"""
        
        # Get patient IDs from any available dataset
        patient_ids = []
        for df in datasets.values():
            if 'patient_id' in df.columns:
                patient_ids.extend(df['patient_id'].unique())
        
        patient_ids = list(set(patient_ids))
        
        # Generate realistic risk scores with some correlation to age if available
        if 'patients' in datasets and 'age' in datasets['patients'].columns:
            patients_df = datasets['patients']
            results = []
            
            for patient_id in patient_ids:
                patient_data = patients_df[patients_df['patient_id'] == patient_id]
                
                if not patient_data.empty:
                    age = patient_data['age'].iloc[0]
                    
                    # Age-based base risk
                    if age > 80:
                        base_risk = 0.4
                    elif age > 70:
                        base_risk = 0.25
                    elif age > 60:
                        base_risk = 0.15
                    else:
                        base_risk = 0.08
                    
                    # Add random variation
                    risk_score = base_risk + np.random.normal(0, 0.15)
                    risk_score = np.clip(risk_score, 0.01, 0.95)
                    
                else:
                    risk_score = np.random.beta(2, 5)  # Skewed toward lower risk
                
                results.append({
                    'patient_id': patient_id,
                    'risk_score': risk_score
                })
        else:
            # Generate random but realistic risk scores
            results = []
            for patient_id in patient_ids:
                risk_score = np.random.beta(2, 5)  # Skewed toward lower risk
                results.append({
                    'patient_id': patient_id,
                    'risk_score': risk_score
                })
        
        results_df = pd.DataFrame(results)
        results_df['risk_category'] = pd.cut(
            results_df['risk_score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        return results_df
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top feature importances"""
        
        if not self.feature_importances:
            return {}
        
        # Sort by importance
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, path: str):
        """Save trained model and feature engineer"""
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'feature_importances': self.feature_importances
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and feature engineer"""
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.model_type = model_data.get('model_type', 'Unknown')
        self.training_metrics = model_data.get('training_metrics', {})
        self.feature_importances = model_data.get('feature_importances', {})
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test the predictor
    from data_loader import DynamicDataLoader
    
    # Load demo data
    loader = DynamicDataLoader()
    datasets = loader.create_demo_dataset()
    feature_config = loader.get_feature_config()
    
    # Train model
    predictor = RiskPredictor(model_type="Random Forest")
    metrics = predictor.train(datasets, feature_config)
    
    print("Training metrics:", metrics)
    
    # Generate predictions
    predictions = predictor.predict(datasets)
    print(f"Generated predictions for {len(predictions)} patients")
    print(predictions.head())