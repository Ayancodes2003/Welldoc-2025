import pandas as pd

print("=== DIABETES DATASET PIPELINE STATUS ===")
print()

# Check processed data
try:
    patients = pd.read_csv('data/patients.csv')
    outcomes = pd.read_csv('data/outcomes.csv')
    
    print(f"✅ Data Preprocessing: COMPLETE")
    print(f"   - Patients: {len(patients):,} records with {patients.shape[1]} features")
    print(f"   - Outcomes: {len(outcomes):,} records")
    print(f"   - Event rate: {outcomes['event_within_90d'].mean():.1%}")
    print()
    
    print("Sample features in patients data:")
    print("  ", list(patients.columns[:10]))
    print()
    
    print("Key diabetes features found:")
    diabetes_features = [col for col in patients.columns if any(term in col.lower() for term in 
                        ['insulin', 'glucose', 'a1c', 'diab', 'age', 'race', 'medication'])]
    print("  ", diabetes_features[:8])
    
except Exception as e:
    print(f"❌ Data loading error: {e}")