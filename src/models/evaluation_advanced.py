"""
Model Calibration and Decision Curve Analysis

Implements calibration plots, Brier score, and Decision Curve Analysis
for model evaluation and clinical utility assessment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ModelCalibration:
    """Model calibration analysis and visualization"""
    
    def __init__(self):
        self.calibration_data = {}
        
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate calibration metrics for a model
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics and data
        """
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        max_calibration_error = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Calculate reliability metrics
        bin_counts = []
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include the last boundary
                bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            bin_counts.append(np.sum(bin_mask))
        
        return {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'max_calibration_error': max_calibration_error,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'bin_counts': bin_counts,
            'bin_boundaries': bin_boundaries,
            'n_bins': n_bins
        }
    
    def create_calibration_plot(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              model_name: str = "Model", n_bins: int = 10) -> go.Figure:
        """Create interactive calibration plot"""
        
        # Calculate calibration data
        cal_data = self.calculate_calibration_metrics(y_true, y_prob, n_bins)
        
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
            x=cal_data['mean_predicted_value'],
            y=cal_data['fraction_of_positives'],
            mode='lines+markers',
            name=f'{model_name}',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            showlegend=True
        ))
        
        # Add confidence intervals (simplified)
        # Calculate bin sizes for error estimation
        bin_sizes = np.array(cal_data['bin_counts'])
        bin_sizes = np.where(bin_sizes == 0, 1, bin_sizes)  # Avoid division by zero
        
        # Estimate standard errors
        p = cal_data['fraction_of_positives']
        se = np.sqrt(p * (1 - p) / bin_sizes)
        upper_ci = np.clip(p + 1.96 * se, 0, 1)
        lower_ci = np.clip(p - 1.96 * se, 0, 1)
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([cal_data['mean_predicted_value'], 
                             cal_data['mean_predicted_value'][::-1]]),
            y=np.concatenate([upper_ci, lower_ci[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Calibration Plot - {model_name}<br><sub>Brier Score: {cal_data["brier_score"]:.3f}, Calibration Error: {cal_data["calibration_error"]:.3f}</sub>',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 model_name: str = "Model", n_bins: int = 10) -> go.Figure:
        """Create reliability diagram with histogram"""
        
        cal_data = self.calculate_calibration_metrics(y_true, y_prob, n_bins)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Reliability Diagram', 'Prediction Distribution'),
            vertical_spacing=0.1
        )
        
        # Reliability plot (top)
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=cal_data['mean_predicted_value'],
                y=cal_data['fraction_of_positives'],
                mode='lines+markers',
                name=model_name,
                line=dict(color='blue', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Histogram (bottom)
        fig.add_trace(
            go.Histogram(
                x=y_prob,
                nbinsx=n_bins,
                name='Prediction Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Mean Predicted Probability", row=1, col=1)
        fig.update_yaxes(title_text="Fraction of Positives", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_layout(
            title=f'Model Reliability Analysis - {model_name}',
            height=700,
            template='plotly_white'
        )
        
        return fig

class DecisionCurveAnalysis:
    """Decision Curve Analysis for clinical utility assessment"""
    
    def __init__(self):
        self.dca_data = {}
    
    def calculate_net_benefit(self, y_true: np.ndarray, y_prob: np.ndarray,
                            thresholds: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Calculate net benefit for Decision Curve Analysis
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            thresholds: Risk thresholds to evaluate
            
        Returns:
            Dictionary with net benefit data
        """
        
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)
        
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
        
        return {
            'thresholds': thresholds,
            'net_benefit_model': net_benefit_model,
            'net_benefit_all': net_benefit_all,
            'net_benefit_none': net_benefit_none
        }
    
    def create_dca_plot(self, y_true: np.ndarray, y_prob: np.ndarray,
                       model_name: str = "Model", thresholds: np.ndarray = None) -> go.Figure:
        """Create Decision Curve Analysis plot"""
        
        # Calculate net benefit
        dca_data = self.calculate_net_benefit(y_true, y_prob, thresholds)
        
        # Create figure
        fig = go.Figure()
        
        # Add model curve
        fig.add_trace(go.Scatter(
            x=dca_data['thresholds'],
            y=dca_data['net_benefit_model'],
            mode='lines',
            name=model_name,
            line=dict(color='blue', width=3)
        ))
        
        # Add "treat all" curve
        fig.add_trace(go.Scatter(
            x=dca_data['thresholds'],
            y=dca_data['net_benefit_all'],
            mode='lines',
            name='Treat All',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add "treat none" curve
        fig.add_trace(go.Scatter(
            x=dca_data['thresholds'],
            y=dca_data['net_benefit_none'],
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
    
    def calculate_clinical_impact(self, y_true: np.ndarray, y_prob: np.ndarray,
                                threshold: float = 0.3) -> Dict[str, Any]:
        """Calculate clinical impact metrics at a specific threshold"""
        
        predicted_positive = (y_prob >= threshold)
        
        # Calculate confusion matrix elements
        tp = np.sum((predicted_positive == 1) & (y_true == 1))
        fp = np.sum((predicted_positive == 1) & (y_true == 0))
        tn = np.sum((predicted_positive == 0) & (y_true == 0))
        fn = np.sum((predicted_positive == 0) & (y_true == 1))
        
        total = len(y_true)
        
        # Calculate rates
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Calculate clinical impact
        intervention_rate = (tp + fp) / total
        detection_rate = tp / total
        
        return {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'intervention_rate': intervention_rate,
            'detection_rate': detection_rate,
            'number_needed_to_intervene': 1 / intervention_rate if intervention_rate > 0 else np.inf,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

class AdvancedEvaluationSuite:
    """Combined calibration and DCA evaluation suite"""
    
    def __init__(self):
        self.calibration = ModelCalibration()
        self.dca = DecisionCurveAnalysis()
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_prob: np.ndarray,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation including calibration and DCA
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name of the model for display
            
        Returns:
            Complete evaluation results
        """
        
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Calibration analysis
            calibration_metrics = self.calibration.calculate_calibration_metrics(y_true, y_prob)
            calibration_plot = self.calibration.create_calibration_plot(y_true, y_prob, model_name)
            reliability_plot = self.calibration.create_reliability_diagram(y_true, y_prob, model_name)
            
            # Decision Curve Analysis
            dca_data = self.dca.calculate_net_benefit(y_true, y_prob)
            dca_plot = self.dca.create_dca_plot(y_true, y_prob, model_name)
            
            # Clinical impact at different thresholds
            clinical_impacts = {}
            for threshold in [0.2, 0.3, 0.5, 0.7]:
                clinical_impacts[f'threshold_{threshold}'] = self.dca.calculate_clinical_impact(
                    y_true, y_prob, threshold
                )
            
            # Store results
            results = {
                'model_name': model_name,
                'calibration_metrics': calibration_metrics,
                'dca_data': dca_data,
                'clinical_impacts': clinical_impacts,
                'plots': {
                    'calibration': calibration_plot,
                    'reliability': reliability_plot,
                    'dca': dca_plot
                }
            }
            
            self.evaluation_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            # Return basic results on failure
            results = {
                'model_name': model_name,
                'error': str(e),
                'calibration_metrics': {'brier_score': np.nan},
                'dca_data': {},
                'clinical_impacts': {},
                'plots': {}
            }
        
        return results
    
    def compare_models(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> go.Figure:
        """
        Compare multiple models on the same DCA plot
        
        Args:
            models_data: Dictionary with model_name: (y_true, y_prob) pairs
            
        Returns:
            Combined DCA plot
        """
        
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
            dca_data = self.dca.calculate_net_benefit(y_true, y_prob)
            
            fig.add_trace(go.Scatter(
                x=dca_data['thresholds'],
                y=dca_data['net_benefit_model'],
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=3)
            ))
        
        # Add reference lines
        if models_data:
            first_model = list(models_data.values())[0]
            y_true_ref = first_model[0]
            dca_ref = self.dca.calculate_net_benefit(y_true_ref, y_true_ref)  # Use for reference
            
            fig.add_trace(go.Scatter(
                x=dca_ref['thresholds'],
                y=dca_ref['net_benefit_all'],
                mode='lines',
                name='Treat All',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=dca_ref['thresholds'],
                y=dca_ref['net_benefit_none'],
                mode='lines',
                name='Treat None',
                line=dict(color='black', width=2)
            ))
        
        fig.update_layout(
            title='Model Comparison - Decision Curve Analysis',
            xaxis_title='Risk Threshold',
            yaxis_title='Net Benefit',
            template='plotly_white',
            width=800,
            height=600
        )
        
        return fig
    
    def generate_evaluation_summary(self, model_name: str) -> Dict[str, Any]:
        """Generate summary statistics for model evaluation"""
        
        if model_name not in self.evaluation_results:
            return {}
        
        results = self.evaluation_results[model_name]
        cal_metrics = results['calibration_metrics']
        
        # Find optimal threshold based on DCA
        dca_data = results['dca_data']
        optimal_idx = np.argmax(dca_data['net_benefit_model'])
        optimal_threshold = dca_data['thresholds'][optimal_idx]
        max_net_benefit = dca_data['net_benefit_model'][optimal_idx]
        
        summary = {
            'model_name': model_name,
            'calibration': {
                'brier_score': cal_metrics['brier_score'],
                'calibration_error': cal_metrics['calibration_error'],
                'max_calibration_error': cal_metrics['max_calibration_error']
            },
            'clinical_utility': {
                'optimal_threshold': optimal_threshold,
                'max_net_benefit': max_net_benefit,
                'threshold_range_positive_benefit': len(dca_data['net_benefit_model'][dca_data['net_benefit_model'] > 0])
            }
        }
        
        return summary

if __name__ == "__main__":
    # Test evaluation suite
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate correlated probabilities
    noise = np.random.normal(0, 0.2, n_samples)
    y_prob = np.clip(y_true * 0.7 + 0.2 + noise, 0, 1)
    
    # Test evaluation
    evaluator = AdvancedEvaluationSuite()
    results = evaluator.evaluate_model(y_true, y_prob, "Test Model")
    
    print("Evaluation completed:")
    print(f"Brier Score: {results['calibration_metrics']['brier_score']:.3f}")
    print(f"Calibration Error: {results['calibration_metrics']['calibration_error']:.3f}")