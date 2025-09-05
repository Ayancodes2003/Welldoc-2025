"""
Comprehensive Diabetes Dataset Training Pipeline

Implements:
1. Dataset detection & preprocessing
2. Model training (LR, RF, CoxPH) with 5-fold CV
3. Full evaluation metrics (AUROC, AUPRC, C-Index, Brier, Calibration, DCA)
4. SHAP explanations
5. Dashboard integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pickle
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for explanations
import shap

# Import our modules
from src.models.data_loader import DynamicDataLoader
from src.models.baseline_models import FeatureEngineer, RiskPredictor
from src.models.survival_models import SurvivalModelManager
from src.models.evaluation_advanced import ModelCalibration, DecisionCurveAnalysis

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiabetesPipeline:
    """Complete diabetes dataset training and evaluation pipeline"""
    
    def __init__(self, data_path: str = "data/", results_path: str = "results/"):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = DynamicDataLoader()
        self.datasets = None
        self.feature_config = None
        self.feature_engineer = None
        
        # Models and results
        self.models = {}
        self.metrics = {}
        self.explanations = {}
        self.survival_results = {}
        
    def run_complete_pipeline(self):
        """Run the complete diabetes pipeline"""
        
        logger.info("=== Starting Diabetes Dataset Pipeline ===")
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Train models with 5-fold CV
        self.train_models_with_cv()
        
        # Step 4: Generate survival analysis
        self.run_survival_analysis()
        
        # Step 5: Generate explanations
        self.generate_explanations()
        
        # Step 6: Evaluate models (calibration, DCA)
        self.evaluate_models()
        
        # Step 7: Save results
        self.save_results()
        
        logger.info("=== Diabetes Pipeline Completed Successfully ===")
        
    def load_and_preprocess_data(self):
        """Load diabetes dataset and verify preprocessing"""
        
        logger.info("Loading diabetes dataset...")
        
        # Load datasets (should trigger DiabetesAdapter)
        self.datasets = self.loader.load_dataset(str(self.data_path))
        self.feature_config = self.loader.get_feature_config()
        
        logger.info(f"Loaded datasets: {list(self.datasets.keys())}")
        logger.info(f"Patients: {len(self.datasets['patients'])}")
        logger.info(f"Outcomes: {len(self.datasets['outcomes'])}")
        
        # Verify outcome distribution
        outcome_dist = self.datasets['outcomes']['event_within_90d'].value_counts()
        logger.info(f"Outcome distribution: {outcome_dist.to_dict()}")
        
    def engineer_features(self):
        """Engineer features for the diabetes dataset"""
        
        logger.info("Engineering features...")
        
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.features_df = self.feature_engineer.create_features(self.datasets)
        
        # Merge with outcomes
        self.ml_data = self.features_df.merge(
            self.datasets['outcomes'], on='patient_id', how='inner'
        )
        
        logger.info(f"Feature matrix shape: {self.features_df.shape}")
        logger.info(f"Generated features: {len(self.feature_engineer.feature_names)}")
        
        # Prepare X and y for ML
        self.X = self.ml_data[self.feature_engineer.feature_names].fillna(0)
        self.y = self.ml_data['event_within_90d']
        
        logger.info(f"ML dataset: X={self.X.shape}, y={self.y.shape}")
        
    def train_models_with_cv(self):
        """Train Logistic Regression, Random Forest, and prepare for CoxPH with 5-fold CV"""
        
        logger.info("Training models with 5-fold cross-validation...")
        
        # Setup 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Standardize features for Logistic Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Train each model
        for model_name, model in models_config.items():
            logger.info(f"Training {model_name}...")
            
            # Use scaled features for Logistic Regression
            X_train = X_scaled if model_name == 'Logistic Regression' else self.X
            
            # Cross-validation scoring
            scoring = ['roc_auc', 'average_precision', 'accuracy']
            cv_results = cross_validate(
                model, X_train, self.y, cv=cv, 
                scoring=scoring, return_train_score=True
            )
            
            # Train final model on full dataset
            model.fit(X_train, self.y)
            
            # Generate predictions for further analysis
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(X_train)
            
            # Calculate additional metrics
            brier_score = brier_score_loss(self.y, y_pred_proba)
            conf_matrix = confusion_matrix(self.y, y_pred)
            
            # Store model and metrics
            self.models[model_name] = {
                'model': model,
                'scaler': scaler if model_name == 'Logistic Regression' else None,
                'predictions': y_pred_proba,
                'predictions_binary': y_pred
            }
            
            self.metrics[model_name] = {
                'cv_auroc_mean': cv_results['test_roc_auc'].mean(),
                'cv_auroc_std': cv_results['test_roc_auc'].std(),
                'cv_auprc_mean': cv_results['test_average_precision'].mean(),
                'cv_auprc_std': cv_results['test_average_precision'].std(),
                'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
                'cv_accuracy_std': cv_results['test_accuracy'].std(),
                'brier_score': brier_score,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': self._get_feature_importance(model, model_name)
            }
            
            logger.info(f"{model_name} - AUROC: {self.metrics[model_name]['cv_auroc_mean']:.3f} ± {self.metrics[model_name]['cv_auroc_std']:.3f}")
            logger.info(f"{model_name} - AUPRC: {self.metrics[model_name]['cv_auprc_mean']:.3f} ± {self.metrics[model_name]['cv_auprc_std']:.3f}")
    
    def _get_feature_importance(self, model, model_name: str) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        
        feature_names = self.feature_engineer.feature_names
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression
            importance = np.abs(model.coef_[0])
        else:
            importance = np.ones(len(feature_names))
        
        # Sort by importance
        feature_importance = dict(zip(feature_names, importance))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def run_survival_analysis(self):
        """Run survival analysis using CoxPH"""
        
        logger.info("Running survival analysis...")
        
        try:
            # Initialize survival model manager
            survival_manager = SurvivalModelManager(self.feature_config)
            
            # Train survival models
            survival_results = survival_manager.train_survival_models(
                self.datasets, self.feature_engineer
            )
            
            self.survival_results = survival_results
            
            # Add C-index to metrics for Cox model
            if 'cox' in survival_results:
                cox_metrics = survival_results['cox']
                if 'c_index' in cox_metrics:
                    self.metrics['Cox PH'] = {
                        'c_index': cox_metrics['c_index'],
                        'log_likelihood': cox_metrics.get('log_likelihood', 0),
                        'n_events': cox_metrics.get('n_events', 0),
                        'n_censored': cox_metrics.get('n_censored', 0)
                    }
                    
                    logger.info(f"Cox PH - C-index: {cox_metrics['c_index']:.3f}")
            
        except Exception as e:
            logger.error(f"Survival analysis failed: {e}")
            self.survival_results = {'error': str(e)}
    
    def generate_explanations(self):
        """Generate SHAP explanations for trained models"""
        
        logger.info("Generating SHAP explanations...")
        
        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']
                
                # Sample data for SHAP (use subset for efficiency)
                n_samples = min(500, len(self.X))  # Increase sample size but keep manageable
                X_sample = self.X.sample(n_samples, random_state=42)
                
                if model_name == 'Logistic Regression':
                    # Use scaled features for Logistic Regression
                    scaler = model_data['scaler']
                    X_sample_scaled = scaler.transform(X_sample)
                    explainer = shap.Explainer(model, X_sample_scaled)
                    shap_values = explainer(X_sample_scaled)
                else:
                    # Random Forest
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # For binary classification, use positive class shap values
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Positive class
                
                # Store global feature importance from SHAP
                if hasattr(shap_values, 'values'):
                    mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
                else:
                    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                
                feature_names = self.feature_engineer.feature_names
                shap_importance = dict(zip(feature_names, mean_shap_values))
                sorted_shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
                
                self.explanations[model_name] = {
                    'global_importance': sorted_shap_importance,
                    'top_5_features': list(sorted_shap_importance.keys())[:5],
                    'shap_values_sample': shap_values,
                    'sample_data': X_sample
                }
                
                logger.info(f"{model_name} - Top 5 features: {self.explanations[model_name]['top_5_features']}")
                
            except Exception as e:
                logger.error(f"SHAP explanation failed for {model_name}: {e}")
                self.explanations[model_name] = {'error': str(e)}
    
    def evaluate_models(self):
        """Evaluate models with calibration and DCA analysis"""
        
        logger.info("Evaluating models with calibration and DCA...")
        
        try:
            # Initialize analyzers
            calibration_analyzer = ModelCalibration()
            dca_analyzer = DecisionCurveAnalysis()
            
            for model_name, model_data in self.models.items():
                y_pred_proba = model_data['predictions']
                
                # Calibration analysis
                try:
                    prob_true, prob_pred = calibration_curve(self.y, y_pred_proba, n_bins=10)
                    calibration_results = calibration_analyzer.calculate_calibration_metrics(
                        self.y, y_pred_proba
                    )
                    
                    self.metrics[model_name]['calibration'] = {
                        'brier_score': brier_score_loss(self.y, y_pred_proba),
                        'calibration_curve': {
                            'prob_true': prob_true.tolist(),
                            'prob_pred': prob_pred.tolist()
                        },
                        'calibration_results': calibration_results
                    }
                    
                except Exception as e:
                    logger.error(f"Calibration analysis failed for {model_name}: {e}")
                    self.metrics[model_name]['calibration'] = {'error': str(e)}
                
                # Decision Curve Analysis
                try:
                    threshold_range = np.arange(0.01, 0.99, 0.01)
                    dca_results = dca_analyzer.calculate_net_benefit(
                        self.y, y_pred_proba, threshold_range
                    )
                    
                    self.metrics[model_name]['dca'] = {
                        'thresholds': threshold_range.tolist(),
                        'net_benefit': dca_results['net_benefit_model'].tolist(),
                        'net_benefit_all': dca_results['net_benefit_all'].tolist(),
                        'net_benefit_none': dca_results['net_benefit_none'].tolist()
                    }
                    
                except Exception as e:
                    logger.error(f"DCA analysis failed for {model_name}: {e}")
                    self.metrics[model_name]['dca'] = {'error': str(e)}
                    
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
    
    def save_results(self):
        """Save all results to files"""
        
        logger.info("Saving results...")
        
        # Save metrics as JSON
        metrics_file = self.results_path / 'diabetes_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save explanations (excluding SHAP objects)
        explanations_summary = {}
        for model_name, exp_data in self.explanations.items():
            explanations_summary[model_name] = {
                'global_importance': exp_data.get('global_importance', {}),
                'top_5_features': exp_data.get('top_5_features', []),
                'error': exp_data.get('error', None)
            }
        
        explanations_file = self.results_path / 'diabetes_explanations.json'
        with open(explanations_file, 'w') as f:
            json.dump(explanations_summary, f, indent=2)
        
        # Save survival results
        survival_file = self.results_path / 'diabetes_survival.json'
        with open(survival_file, 'w') as f:
            json.dump(self.survival_results, f, indent=2, default=str)
        
        # Save models
        models_file = self.results_path / 'diabetes_models.pkl'
        models_to_save = {
            name: {
                'model': data['model'],
                'scaler': data.get('scaler')
            }
            for name, data in self.models.items()
        }
        
        with open(models_file, 'wb') as f:
            pickle.dump(models_to_save, f)
        
        # Create summary report
        self.create_summary_report()
        
        logger.info(f"Results saved to {self.results_path}")
    
    def create_summary_report(self):
        """Create a summary report of the pipeline results"""
        
        report_lines = [
            "# Diabetes Dataset Pipeline Results",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Summary",
            f"- Total patients: {len(self.datasets['patients'])}",
            f"- Features generated: {len(self.feature_engineer.feature_names)}",
            f"- Event rate: {self.y.mean():.1%}",
            "",
            "## Model Performance",
        ]
        
        for model_name, metrics in self.metrics.items():
            report_lines.extend([
                f"### {model_name}",
                f"- AUROC: {metrics.get('cv_auroc_mean', 'N/A'):.3f} ± {metrics.get('cv_auroc_std', 0):.3f}",
                f"- AUPRC: {metrics.get('cv_auprc_mean', 'N/A'):.3f} ± {metrics.get('cv_auprc_std', 0):.3f}",
                f"- Brier Score: {metrics.get('brier_score', 'N/A'):.3f}",
            ])
            
            if model_name == 'Cox PH':
                report_lines.append(f"- C-Index: {metrics.get('c_index', 'N/A'):.3f}")
            
            report_lines.append("")
        
        # Add top features
        report_lines.extend([
            "## Top Features (SHAP)",
            ""
        ])
        
        for model_name, exp_data in self.explanations.items():
            if 'top_5_features' in exp_data:
                report_lines.extend([
                    f"### {model_name}",
                    *[f"- {feat}" for feat in exp_data['top_5_features']],
                    ""
                ])
        
        # Save report
        report_file = self.results_path / 'diabetes_pipeline_report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to {report_file}")

def main():
    """Run the complete diabetes pipeline"""
    
    pipeline = DiabetesPipeline()
    pipeline.run_complete_pipeline()
    
    print("\n=== Pipeline Summary ===")
    print(f"Results saved to: {pipeline.results_path}")
    print("\nFiles generated:")
    for file_path in pipeline.results_path.glob('*'):
        print(f"  - {file_path.name}")
    
    # Display key metrics
    print("\n=== Key Metrics ===")
    for model_name, metrics in pipeline.metrics.items():
        print(f"{model_name}:")
        if 'cv_auroc_mean' in metrics:
            print(f"  AUROC: {metrics['cv_auroc_mean']:.3f} ± {metrics['cv_auroc_std']:.3f}")
            print(f"  AUPRC: {metrics['cv_auprc_mean']:.3f} ± {metrics['cv_auprc_std']:.3f}")
        if 'c_index' in metrics:
            print(f"  C-Index: {metrics['c_index']:.3f}")

if __name__ == "__main__":
    main()