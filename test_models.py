from src.models.data_loader import DynamicDataLoader
from src.models.baseline_models import FeatureEngineer, RiskPredictor

print("=== MODEL TRAINING STATUS ===")
print()

# Load datasets
loader = DynamicDataLoader()
datasets = loader.load_dataset('data/')
feature_config = loader.get_feature_config()

print(f"✅ Dataset Loading: COMPLETE")
print(f"   - Dataset format: {loader.current_adapter.__class__.__name__}")
print(f"   - Tables: {list(datasets.keys())}")
print()

# Test model training
print("🤖 Testing Model Training...")
try:
    predictor = RiskPredictor(model_type='Random Forest')
    metrics = predictor.train(datasets, feature_config)
    
    print(f"✅ Random Forest Training: COMPLETE")
    print(f"   - AUROC: {metrics['auc_roc']:.3f}")
    print(f"   - AUPRC: {metrics['auc_pr']:.3f}")
    print(f"   - Features: {metrics['n_features']}")
    print()
    
    # Test predictions
    predictions = predictor.predict(datasets)
    print(f"✅ Prediction Generation: COMPLETE")
    print(f"   - Predictions for {len(predictions)} patients")
    print(f"   - Risk distribution: {predictions['risk_category'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"❌ Model training error: {e}")