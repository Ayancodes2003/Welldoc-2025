"""
Enhanced Model Training Pipeline

Integrates continuous monitoring, multi-condition support, and 90-day risk prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
# from lifelines import CoxPHFitter  # Temporarily disabled
import logging
from typing import Dict, List, Optional, Tuple, Any
from .continuous_monitoring import ContinuousMonitoringProcessor
from .multi_condition_adapter import MultiConditionAdapter

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced model trainer with continuous monitoring and multi-condition support"""
    
    def __init__(self, monitoring_windows: List[int] = [30, 60, 90, 180]):
        self.monitoring_windows = monitoring_windows
        self.continuous_processor = ContinuousMonitoringProcessor(monitoring_windows)
        self.multi_condition_adapter = MultiConditionAdapter()
        self.trained_models = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def train_enhanced_models(self, datasets: Dict[str, pd.DataFrame], 
                            condition_type: str = None) -> Dict[str, Any]:
        """Train enhanced models with continuous monitoring features"""
        
        logger.info("Training enhanced models with continuous monitoring...")
        
        # Generate continuous monitoring features
        feature_matrix = self.continuous_processor.generate_continuous_features(datasets)
        
        # Prepare target variable
        if 'outcomes' in datasets:
            targets = datasets['outcomes']['event_within_90d']
            time_to_event = datasets['outcomes']['time_to_event']
        else:
            # Create synthetic outcomes for demonstration
            targets = np.random.choice([0, 1], len(feature_matrix), p=[0.7, 0.3])
            time_to_event = np.random.uniform(1, 90, len(feature_matrix))
        
        # Align features and targets
        feature_matrix = feature_matrix.set_index('patient_id')
        
        # Prepare features for modeling
        X = self._prepare_features(feature_matrix)
        y = targets[:len(X)]
        
        # Train multiple models
        self.trained_models = {}
        
        # Enhanced Logistic Regression with 90-day focus
        lr_model = self._train_logistic_regression(X, y)
        self.trained_models['Logistic_Regression'] = lr_model
        
        # Enhanced Random Forest
        rf_model = self._train_random_forest(X, y)
        self.trained_models['Random_Forest'] = rf_model
        
        # Cox PH for survival analysis with 90-day horizon (temporarily disabled)
        # if len(time_to_event) >= len(X):
        #     cox_model = self._train_cox_ph(X, y, time_to_event[:len(X)])
        #     self.trained_models['Cox_PH'] = cox_model
        
        # Evaluate all models
        self.evaluation_results = self._evaluate_models(X, y)
        
        # Generate feature importance analysis
        self._analyze_feature_importance(X)
        
        return {
            'trained_models': self.trained_models,
            'evaluation_results': self.evaluation_results,
            'feature_importance': self.feature_importance,
            'feature_metadata': self.continuous_processor.get_feature_metadata()
        }
        
    def _prepare_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling with proper scaling and encoding"""
        
        # Select numeric features only
        numeric_features = feature_matrix.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(numeric_features),
            columns=numeric_features.columns,
            index=numeric_features.index
        )
        
        logger.info(f"Prepared {len(scaled_features.columns)} features for modeling")
        return scaled_features
        
    def _train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train enhanced logistic regression with 90-day prediction focus"""
        
        logger.info("Training Logistic Regression model...")
        
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Train model
        model.fit(X, y)
        
        # Generate predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Calculate metrics
        auc_roc = roc_auc_score(y, y_pred_proba)
        auc_pr = average_precision_score(y, y_pred_proba)
        brier_score = brier_score_loss(y, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        return {
            'model': model,
            'predictions': y_pred_proba,
            'predictions_binary': y_pred,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'brier_score': brier_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_coefficients': dict(zip(X.columns, model.coef_[0]))
        }
        
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train enhanced random forest model"""
        
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            oob_score=True
        )
        
        # Train model
        model.fit(X, y)
        
        # Generate predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Calculate metrics
        auc_roc = roc_auc_score(y, y_pred_proba)
        auc_pr = average_precision_score(y, y_pred_proba)
        brier_score = brier_score_loss(y, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        return {
            'model': model,
            'predictions': y_pred_proba,
            'predictions_binary': y_pred,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'brier_score': brier_score,
            'oob_score': model.oob_score_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
    def _train_cox_ph(self, X: pd.DataFrame, y: pd.Series, time_to_event: pd.Series) -> Dict[str, Any]:
        """Train Cox Proportional Hazards model for 90-day survival analysis (temporarily disabled)"""
        
        logger.info("Cox PH model training temporarily disabled...")
        
        # Return fallback model
        return {
            'model': None,
            'c_index': 0.5,
            'error': 'Cox PH temporarily disabled',
            'fallback': True
        }
            
    def _evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate all trained models"""
        
        evaluation = {}
        
        for model_name, model_data in self.trained_models.items():
            if model_name == 'Cox_PH' and model_data.get('fallback'):
                evaluation[model_name] = {
                    'status': 'failed',
                    'error': model_data.get('error')
                }
                continue
                
            if model_name in ['Logistic_Regression', 'Random_Forest']:
                evaluation[model_name] = {
                    'auc_roc': model_data['auc_roc'],
                    'auc_pr': model_data['auc_pr'],
                    'brier_score': model_data['brier_score'],
                    'cv_mean': model_data['cv_mean'],
                    'cv_std': model_data['cv_std'],
                    'status': 'success'
                }
            elif model_name == 'Cox_PH':
                evaluation[model_name] = {
                    'c_index': model_data['c_index'],
                    'status': 'success'
                }
                
        return evaluation
        
    def _analyze_feature_importance(self, X: pd.DataFrame):
        """Analyze feature importance across models"""
        
        # Random Forest feature importance
        if 'Random_Forest' in self.trained_models:
            rf_importance = self.trained_models['Random_Forest']['feature_importance']
            self.feature_importance['random_forest'] = dict(
                sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        
        # Logistic Regression coefficients
        if 'Logistic_Regression' in self.trained_models:
            lr_coef = self.trained_models['Logistic_Regression']['feature_coefficients']
            # Sort by absolute value of coefficients
            lr_sorted = sorted(lr_coef.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            self.feature_importance['logistic_regression'] = dict(lr_sorted)
        
        # Cox PH hazard ratios
        if 'Cox_PH' in self.trained_models and not self.trained_models['Cox_PH'].get('fallback'):
            hazard_ratios = self.trained_models['Cox_PH']['hazard_ratios']
            # Sort by deviation from 1.0 (neutral hazard ratio)
            hr_sorted = sorted(hazard_ratios.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)[:10]
            self.feature_importance['cox_ph'] = dict(hr_sorted)